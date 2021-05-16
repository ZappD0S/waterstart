from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.distributions as dist
import torch.jit as jit
import torch.nn as nn

from .array_mapping import DictBasedArrayMapper, MarketDataArrayMapper
from .array_mapping.base_mapper import BaseArrayMapper
from .price import LatestMarketData
from .symbols import TradedSymbolInfo


@dataclass
class TradesState:
    trades_sizes: torch.Tensor
    trades_prices: torch.Tensor

    @cached_property
    def pos_sizes(self) -> torch.Tensor:
        return self.trades_sizes.sum(dim=0)


@dataclass
class ModelState:
    trades_state: TradesState
    balance: torch.Tensor
    hidden_state: torch.Tensor


@dataclass
class MarketState:
    prev_market_data_arr: torch.Tensor
    latest_market_data: LatestMarketData


@dataclass
class RawMarketState:
    market_data: torch.Tensor
    sym_prices: torch.Tensor
    margin_rate: torch.Tensor


T = TypeVar("T", MarketState, RawMarketState)
U = TypeVar("U", MarketState, RawMarketState)


@dataclass
class ModelInput(Generic[T]):
    market_state: T
    trades_state: TradesState
    hidden_state: torch.Tensor
    balance: torch.Tensor


@dataclass
class MaskedTensor:
    arr: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if not self.arr.ndim == self.mask.ndim == 1:
            raise ValueError()

        if self.arr.numel() != self.mask.numel():
            raise ValueError()

        if self.mask.dtype != torch.bool:
            raise ValueError()


@dataclass
class ModelInference:
    pos_sizes: torch.Tensor


# TODO: we need a better name..
@dataclass
class ModelInferenceWithMap(ModelInference):
    pos_sizes_map: Mapping[TradedSymbolInfo, float]
    market_data_arr: torch.Tensor
    hidden_state: torch.Tensor


@dataclass
class RawModelOutput:
    z_samples: torch.Tensor
    z_logprobs: torch.Tensor
    exec_samples: torch.Tensor
    exec_logprobs: torch.Tensor
    fractions: torch.Tensor


@dataclass
class ModelInferenceWithRawOutput(ModelInference):
    raw_model_output: RawModelOutput


@dataclass
class MinStepMax:
    min: torch.Tensor
    step: torch.Tensor
    max: torch.Tensor

    def __post_init__(self):
        if not self.min.shape == self.step.shape == self.max.shape:
            raise ValueError()


V = TypeVar("V")


class Model:
    def __init__(
        self,
        # TODO: maybe group the modules in a dataclass?
        cnn: nn.Module,
        gated_trans: nn.Module,
        iafs: Sequence[nn.Module],
        emitter: nn.Module,
        market_data_arr_mapper: MarketDataArrayMapper,
        traded_sym_arr_mapper: DictBasedArrayMapper[TradedSymbolInfo],
    ) -> None:
        self._cnn = cnn
        self._gated_trans = gated_trans
        self._iafs = iafs
        self._emitter = emitter
        # TODO: these need to be taken from the modules
        self.win_len: int = ...
        self.max_trades: int = ...
        self.hidden_dim: int = ...

        self._market_data_arr_mapper = market_data_arr_mapper
        self._traded_sym_arr_mapper = traded_sym_arr_mapper
        self._min_step_max = self._compute_min_step_max_arr()

    def _compute_min_step_max_arr(self) -> MinStepMax:
        min_step_max_map = {
            key: (
                key.symbol.minVolume / 100 * key.symbol.lotSize / 100,
                key.symbol.stepVolume / 100 * key.symbol.lotSize / 100,
                key.symbol.maxVolume / 100 * key.symbol.lotSize / 100,
            )
            for key in self._traded_sym_arr_mapper.keys
        }

        rec_arr = np.fromiter(
            self._traded_sym_arr_mapper.iterate_index_to_value(min_step_max_map),
            dtype=[
                ("inds", int),
                (
                    "vals",
                    [
                        ("min", float),
                        ("step", float),
                        ("max", float),
                    ],
                ),
            ],
        )

        inds: np.ndarray = rec_arr["inds"]
        min_step_max: np.ndarray = rec_arr["vals"]
        min_step_max = min_step_max[inds]

        return MinStepMax(
            min=min_step_max["min"], step=min_step_max["step"], max=min_step_max["max"]
        )

    def _partial_map_to_masked_tensor(
        self, mapping: Mapping[TradedSymbolInfo, float]
    ) -> MaskedTensor:
        rec_arr: np.ndarray = np.fromiter(
            self._traded_sym_arr_mapper.iterate_index_to_value_partial(mapping),
            dtype=[("inds", int), ("vals", float)],
        )

        arr: torch.Tensor = torch.zeros(rec_arr.size, dtype=float)  # type: ignore
        mask: torch.Tensor = torch.zeros(rec_arr.size, dtype=bool)  # type: ignore
        inds = torch.from_numpy(rec_arr["inds"])  # type: ignore
        vals = torch.from_numpy(rec_arr["vals"])  # type: ignore
        mask[inds] = True
        arr[inds] = vals

        return MaskedTensor(arr, mask)

    def _masked_tensor_to_partial_map(
        self, masked_arr: MaskedTensor
    ) -> Mapping[TradedSymbolInfo, float]:
        arr, mask = masked_arr.arr, masked_arr.mask
        inds_list: list[int] = torch.arange(arr.numel())[mask].tolist()  # type: ignore
        new_pos_sizes_list: list[float] = arr[mask].tolist()  # type: ignore

        return self._traded_sym_arr_mapper.build_from_index_to_value_map_partial(
            dict(zip(inds_list, new_pos_sizes_list))
        )

    def _obj_to_tensor(self, mapper: BaseArrayMapper[V], obj: V) -> torch.Tensor:
        rec_arr: np.ndarray = np.fromiter(
            mapper.iterate_index_to_value(obj), dtype=[("inds", int), ("vals", float)]
        )

        arr: torch.Tensor = torch.full(rec_arr.shape, np.nan, dtype=float)  # type: ignore
        inds = torch.from_numpy(rec_arr["inds"])  # type: ignore
        vals = torch.from_numpy(rec_arr["vals"])  # type: ignore
        arr[inds] = vals

        if arr.isnan().any():
            raise ValueError()

        return arr

    def _tensor_to_obj(self, mapper: BaseArrayMapper[V], arr: torch.Tensor) -> V:
        l: list[float] = arr.tolist()  # type: ignore
        return mapper.build_from_index_to_value_map(dict(enumerate(l)))

    def evaluate(self, model_input: ModelInput[MarketState]) -> ModelInferenceWithMap:
        market_state = model_input.market_state

        # TODO: move to device?

        latest_market_data_arr = self._obj_to_tensor(
            self._market_data_arr_mapper, market_state.latest_market_data.market_feat
        )

        market_data_arr = model_input.market_state.prev_market_data_arr
        market_data_arr[:, :-1] = market_data_arr[:, 1:].clone()
        market_data_arr[:, -1] = latest_market_data_arr

        sym_prices = self._obj_to_tensor(
            self._traded_sym_arr_mapper, market_state.latest_market_data.sym_prices_map
        )
        margin_rate = self._obj_to_tensor(
            self._traded_sym_arr_mapper, market_state.latest_market_data.margin_rate_map
        )

        raw_market_state = RawMarketState(
            market_data=market_data_arr.unsqueeze(0),
            sym_prices=sym_prices.unsqueeze(1),
            margin_rate=margin_rate.unsqueeze(1),
        )

        trades_state = model_input.trades_state

        raw_model_input = ModelInput(
            raw_market_state,
            TradesState(
                trades_state.trades_sizes.unsqueeze(2),
                trades_state.trades_prices.unsqueeze(2),
            ),
            model_input.hidden_state.unsqueeze(0),
            model_input.balance.unsqueeze(0),
        )

        with torch.no_grad():
            model_infer = self.evaluate_raw(raw_model_input)

        exec_mask = model_infer.raw_model_output.exec_samples == 1

        exec_mask = exec_mask.squeeze(1)

        new_pos_sizes = model_infer.pos_sizes.squeeze(1)
        hidden_state = model_infer.raw_model_output.z_samples.squeeze(0)

        new_pos_sizes_map = self._masked_tensor_to_partial_map(
            MaskedTensor(new_pos_sizes, exec_mask)
        )

        return ModelInferenceWithMap(
            pos_sizes=new_pos_sizes,
            market_data_arr=market_data_arr,
            pos_sizes_map=new_pos_sizes_map,
            hidden_state=hidden_state,
        )

    @jit.script  # type: ignore
    def _compute_new_pos_sizes(
        self,
        fractions: torch.Tensor,
        exec_mask: torch.Tensor,
        dep_cur_pos_sizes: torch.Tensor,
        margin_rate: torch.Tensor,
        unused: torch.Tensor,
    ) -> torch.Tensor:
        new_pos_sizes = torch.empty_like(dep_cur_pos_sizes)  # type: ignore
        remaining = unused

        for i in range(len(new_pos_sizes)):
            available = dep_cur_pos_sizes[i].abs() + remaining
            dep_cur_new_pos_size: torch.Tensor = torch.where(
                exec_mask[i], fractions[i] * available, dep_cur_pos_sizes[i]
            )

            new_pos_size = dep_cur_new_pos_size * margin_rate[i]

            abs_new_pos_size = new_pos_size.abs()
            new_pos_sign = new_pos_size.sign()

            abs_new_pos_size = (
                abs_new_pos_size - abs_new_pos_size % self._min_step_max.step[i]
            )
            abs_new_pos_size = abs_new_pos_size.new_zeros([]).where(
                abs_new_pos_size < self._min_step_max.min[i], abs_new_pos_size
            )
            abs_new_pos_size = self._min_step_max.max[i].where(
                abs_new_pos_size > self._min_step_max.max[i], self._min_step_max.max[i]
            )

            new_pos_sizes[i] = new_pos_sign * abs_new_pos_size
            remaining = available - new_pos_sizes[i].abs()

        return new_pos_sizes

    def evaluate_raw(
        self, model_input: ModelInput[RawMarketState]
    ) -> ModelInferenceWithRawOutput:
        market_state = model_input.market_state

        # TODO: input checks

        scaled_market_data = market_state.market_data.clone()
        for src_ind, dst_inds in self._market_data_arr_mapper.scaling_inds:
            scaled_market_data[:, dst_inds] /= scaled_market_data[:, src_ind, -1]  # type: ignore

        trades_state = model_input.trades_state
        rel_prices = trades_state.trades_prices / market_state.sym_prices

        dep_cur_trade_sizes = trades_state.trades_sizes / market_state.margin_rate
        dep_cur_pos_sizes = dep_cur_trade_sizes.sum(dim=0)

        balance = model_input.balance
        unused = balance - dep_cur_pos_sizes.abs().sum(dim=0)
        assert torch.all(unused >= 0)

        # NOTE: the leverage cancels out so we obtain the margin
        rel_margins = torch.cat([dep_cur_trade_sizes.flatten(0, 1), unused]) / balance
        trades_data = torch.cat([rel_margins, rel_prices.flatten(0, 1)])

        model_output = self._evaluate(
            trades_data, scaled_market_data, model_input.hidden_state
        )
        exec_mask = model_output.z_samples == 1

        new_pos_sizes = self._compute_new_pos_sizes(
            model_output.fractions,
            exec_mask,
            dep_cur_pos_sizes,
            market_state.margin_rate,
            unused,
        )

        return ModelInferenceWithRawOutput(new_pos_sizes, model_output)

    # TODO: find better name
    def _evaluate(
        self,
        trades_data: torch.Tensor,
        scaled_market_data: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> RawModelOutput:

        out: torch.Tensor = self._cnn(scaled_market_data, trades_data.movedim(0, -1))
        out = out.movedim(1, 0).unflatten(1, trades_data[0].shape)  # type: ignore

        z_loc: torch.Tensor
        z_scale: torch.Tensor
        z_loc, z_scale = self._gated_trans(out, hidden_state)

        z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self._iafs)
        z_sample: torch.Tensor = z_dist.rsample()  # type: ignore
        z_logprobs: torch.Tensor = z_dist.log_prob(z_sample)  # type: ignore

        exec_logits: torch.Tensor
        fractions: torch.Tensor
        exec_logits, fractions = self._emitter(z_sample)

        exec_logits = exec_logits.movedim(-1, 0)
        fractions = fractions.movedim(-1, 0)

        exec_dist = dist.Bernoulli(logits=exec_logits)
        exec_samples: torch.Tensor = exec_dist.sample()  # type: ignore
        exec_logprobs: torch.Tensor = exec_dist.log_prob(exec_samples)  # type: ignore

        return RawModelOutput(
            z_samples=z_sample,
            z_logprobs=z_logprobs,
            exec_samples=exec_samples,
            exec_logprobs=exec_logprobs,
            fractions=fractions,
        )

    def update_trades(
        self,
        trades_state: TradesState,
        new_pos_sizes: Mapping[TradedSymbolInfo, float],
        open_prices: Mapping[TradedSymbolInfo, float],
    ) -> TradesState:
        ...

    def update_trades_raw(
        self,
        trades_state: TradesState,
        new_pos_sizes: torch.Tensor,
        open_prices: torch.Tensor,
    ) -> TradesState:

        # new_pos_sizes = np.where(success_mask, self._new_pos_sizes, pos_sizes)
        pos_sizes = trades_state.pos_sizes

        trades_sizes = trades_state.trades_sizes
        trades_prices = trades_state.trades_prices

        new_trade_sizes = new_pos_sizes - pos_sizes

        add_trades_sizes = trades_sizes.clone()
        add_trades_sizes[:-1] = trades_sizes[1:]
        add_trades_sizes[-1] = new_trade_sizes

        add_trades_prices = trades_prices.clone()
        add_trades_prices[:-1] = trades_prices[1:]
        add_trades_prices[-1] = open_prices

        close_trade_sizes = new_trade_sizes
        cum_trades_sizes = trades_sizes.cumsum(dim=0)
        left_diffs = close_trade_sizes + cum_trades_sizes

        right_diffs = torch.empty_like(left_diffs)
        right_diffs[1:] = left_diffs[:-1]
        right_diffs[0] = close_trade_sizes

        close_trade_mask = new_pos_sizes * left_diffs <= 0
        reduce_trade_mask = left_diffs * right_diffs < 0
        open_pos_mask = pos_sizes * new_pos_sizes < 0

        remove_trades_sizes = trades_sizes.clone()

        remove_trades_sizes[close_trade_mask] = 0.0
        remove_trades_sizes[reduce_trade_mask] = left_diffs[reduce_trade_mask]
        remove_trades_sizes[-1, open_pos_mask] = new_pos_sizes[open_pos_mask]

        remove_trades_prices = trades_prices.clone()
        remove_trades_prices[close_trade_mask] = 0.0
        remove_trades_prices[-1, open_pos_mask] = open_prices[open_pos_mask]

        latest_trade_sizes = trades_sizes[-1]
        add_mask = (latest_trade_sizes * new_trade_sizes > 0) | (
            latest_trade_sizes == 0
        )
        new_trades_sizes = torch.where(add_mask, add_trades_sizes, remove_trades_sizes)
        new_trades_prices = torch.where(
            add_mask, add_trades_prices, remove_trades_prices
        )

        return TradesState(
            trades_sizes=new_trades_sizes, trades_prices=new_trades_prices
        )

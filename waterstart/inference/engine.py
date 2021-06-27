from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.distributions as dist
import torch.jit as jit
import torch.nn as nn

from ..array_mapping import DictBasedArrayMapper, MarketDataArrayMapper
from ..array_mapping.base_mapper import BaseArrayMapper
from ..inference.model import CNN, Emitter, GatedTransition
from ..symbols import TradedSymbolInfo
from . import (
    MarketState,
    ModelInferenceWithMap,
    ModelInferenceWithRawOutput,
    ModelInput,
    RawMarketState,
    RawModelOutput,
    TradesState,
)


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
class MinStepMax:
    min: torch.Tensor
    step: torch.Tensor
    max: torch.Tensor

    def __post_init__(self):
        if not self.min.shape == self.step.shape == self.max.shape:
            raise ValueError()


T = TypeVar("T")


class InferenceEngine:
    def __init__(
        self,
        # TODO: maybe group the modules in a dataclass?
        cnn: CNN,
        gated_trans: GatedTransition,
        iafs: Sequence[nn.Module],
        emitter: Emitter,
        market_data_arr_mapper: MarketDataArrayMapper,
        traded_sym_arr_mapper: DictBasedArrayMapper[TradedSymbolInfo],
    ) -> None:
        if market_data_arr_mapper.n_fields != cnn.market_features:
            raise ValueError()

        if traded_sym_arr_mapper.n_fields != cnn.n_sym:
            raise ValueError()

        self._cnn = cnn
        self._gated_trans = gated_trans
        self._iafs = iafs
        self._emitter = emitter
        self.n_sym = cnn.n_sym
        self.window_size = cnn.window_size
        self.max_trades = cnn.max_trades
        self.hidden_dim = gated_trans.z_dim

        self._market_data_arr_mapper = market_data_arr_mapper
        self._traded_sym_arr_mapper = traded_sym_arr_mapper
        self._min_step_max = self._compute_min_step_max_arr()

    # TODO: put the following conversion methods in a separate file
    def _compute_min_step_max_arr(self) -> MinStepMax:
        min_step_max_map = {
            key: (
                key.symbol.minVolume / 100 * key.symbol.lotSize / 100,
                key.symbol.stepVolume / 100 * key.symbol.lotSize / 100,
                key.symbol.maxVolume / 100 * key.symbol.lotSize / 100,
            )
            for key in self._traded_sym_arr_mapper.keys
        }

        rec_arr = np.fromiter(  # type: ignore
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

        inds: npt.NDArray[np.int64] = rec_arr["inds"]
        vals: npt.NDArray[Any] = rec_arr["vals"]
        min_step_max = np.empty_like(vals)  # type: ignore
        min_step_max[inds] = vals

        return MinStepMax(
            min=torch.from_numpy(min_step_max["min"]),  # type: ignore
            step=torch.from_numpy(min_step_max["step"]),  # type: ignore
            max=torch.from_numpy(min_step_max["max"]),  # type: ignore
        )

    def _partial_map_to_masked_tensor(
        self, mapping: Mapping[TradedSymbolInfo, float]
    ) -> MaskedTensor:
        rec_arr: npt.NDArray[Any] = np.fromiter(  # type: ignore
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

    def _obj_to_tensor(self, mapper: BaseArrayMapper[T], obj: T) -> torch.Tensor:
        rec_arr: npt.NDArray[Any] = np.fromiter(  # type: ignore
            mapper.iterate_index_to_value(obj), dtype=[("inds", int), ("vals", float)]
        )

        arr: torch.Tensor = torch.full(  # type: ignore
            rec_arr.shape, np.nan, dtype=float
        )
        inds = torch.from_numpy(rec_arr["inds"])  # type: ignore
        vals = torch.from_numpy(rec_arr["vals"])  # type: ignore
        arr[inds] = vals

        if arr.isnan().any():
            raise ValueError()

        return arr

    def _tensor_to_obj(self, mapper: BaseArrayMapper[T], arr: torch.Tensor) -> T:
        l: list[float] = arr.tolist()  # type: ignore
        return mapper.build_from_index_to_value_map(dict(enumerate(l)))

    def evaluate(self, model_input: ModelInput[MarketState]) -> ModelInferenceWithMap:
        market_state = model_input.market_state
        latest_market_data = market_state.latest_market_data
        # TODO: move to device?

        latest_market_data_arr = self._obj_to_tensor(
            self._market_data_arr_mapper, latest_market_data
        )

        market_data_arr = market_state.prev_market_data_arr
        market_data_arr[:, :-1] = market_data_arr[:, 1:].clone()
        market_data_arr[:, -1] = latest_market_data_arr

        symbols_data_map = latest_market_data.symbols_data_map
        sym_prices = self._obj_to_tensor(
            self._traded_sym_arr_mapper,
            {
                sym: symbols_data_map[sym].price_trendbar.close
                for sym in self._traded_sym_arr_mapper.keys
            },
        )
        margin_rate = self._obj_to_tensor(
            self._traded_sym_arr_mapper,
            {
                sym: symbols_data_map[sym].dep_to_base_trendbar.close
                for sym in self._traded_sym_arr_mapper.keys
            },
        )

        raw_market_state = RawMarketState(
            market_data=market_data_arr.unsqueeze(0),
            sym_prices=sym_prices.unsqueeze(-1),
            margin_rate=margin_rate.unsqueeze(-1),
        )

        trades_state = model_input.trades_state

        raw_model_input = ModelInput(
            raw_market_state,
            TradesState(
                trades_state.trades_sizes.unsqueeze(-1),
                trades_state.trades_prices.unsqueeze(-1),
            ),
            model_input.hidden_state.unsqueeze(0),
            model_input.balance.unsqueeze(-1),
        )

        with torch.no_grad():
            model_infer = self.evaluate_raw(raw_model_input)

        exec_mask = model_infer.raw_model_output.exec_samples == 1
        exec_mask = exec_mask.squeeze(-1)

        new_pos_sizes = model_infer.pos_sizes.squeeze(-1)
        hidden_state = model_infer.raw_model_output.z_sample.squeeze(0)

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
        unused_margin: torch.Tensor,
    ) -> torch.Tensor:
        new_pos_sizes = torch.empty_like(dep_cur_pos_sizes)  # type: ignore

        for i in range(self.n_sym):
            available_margin = dep_cur_pos_sizes[i].abs() + unused_margin
            dep_cur_new_pos_size = torch.where(
                exec_mask[i], fractions[i] * available_margin, dep_cur_pos_sizes[i]
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
                abs_new_pos_size > self._min_step_max.max[i], abs_new_pos_size
            )

            new_pos_sizes[i] = new_pos_sign * abs_new_pos_size
            unused_margin = available_margin - abs_new_pos_size

        return new_pos_sizes

    def evaluate_raw(
        self, model_input: ModelInput[RawMarketState]
    ) -> ModelInferenceWithRawOutput:
        market_state = model_input.market_state

        # TODO: input checks

        scaled_market_data = market_state.market_data.clone()
        for src_ind, dst_inds in self._market_data_arr_mapper.scaling_inds:
            scaled_market_data[:, dst_inds] /= scaled_market_data[  # type: ignore
                :, src_ind, None, -1, None
            ]

        trades_state = model_input.trades_state
        rel_prices = trades_state.trades_prices / market_state.sym_prices

        # TODO: make dep_cur_trade_sizes this a cached property?
        dep_cur_trade_sizes = trades_state.trades_sizes / market_state.margin_rate
        dep_cur_pos_sizes = dep_cur_trade_sizes.sum(dim=0)

        balance = model_input.balance
        unused = balance - dep_cur_pos_sizes.abs().sum(dim=0)
        assert torch.all(unused >= 0)

        # NOTE: the leverage cancels out so we obtain the margin
        rel_margins = torch.cat((dep_cur_trade_sizes.flatten(0, 1), unused)) / balance
        trades_data = torch.cat((rel_margins, rel_prices.flatten(0, 1)))

        model_output = self._evaluate(
            trades_data, scaled_market_data, model_input.hidden_state
        )
        exec_mask = model_output.z_sample == 1

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

        z_loc: torch.Tensor
        z_scale: torch.Tensor
        z_loc, z_scale = self._gated_trans(out, hidden_state)

        z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self._iafs)
        z_sample: torch.Tensor = z_dist.rsample()  # type: ignore
        z_logprob: torch.Tensor = z_dist.log_prob(z_sample)  # type: ignore

        exec_logits: torch.Tensor
        fractions: torch.Tensor
        exec_logits, fractions = self._emitter(z_sample)

        exec_logits = exec_logits.movedim(-1, 0)
        fractions = fractions.movedim(-1, 0)

        exec_dist = dist.Bernoulli(logits=exec_logits)
        exec_samples: torch.Tensor = exec_dist.sample()  # type: ignore
        exec_logprobs: torch.Tensor = exec_dist.log_prob(exec_samples)  # type: ignore

        return RawModelOutput(
            z_sample=z_sample,
            z_logprob=z_logprob,
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
        # TODO
        ...

    @staticmethod
    def _add_case_update_trades(
        trades_state: TradesState,
        new_trade_size: torch.Tensor,
        open_prices: torch.Tensor,
    ) -> TradesState:
        trades_sizes = trades_state.trades_sizes
        reached_max_trades_mask = trades_sizes[0] != 0

        new_trades_sizes = trades_sizes.clone()
        new_trades_sizes[:-1] = trades_sizes[1:]
        new_trades_sizes[-1] = new_trade_size
        new_trades_sizes = trades_sizes.where(reached_max_trades_mask, new_trades_sizes)

        trades_prices = trades_state.trades_prices
        new_trades_prices = trades_prices.clone()
        new_trades_prices[:-1] = trades_prices[1:]
        new_trades_prices[-1] = open_prices
        new_trades_prices = trades_prices.where(
            reached_max_trades_mask, new_trades_prices
        )

        return TradesState(new_trades_sizes, new_trades_prices)

    @staticmethod
    def _remove_case_update_trades(
        trades_state: TradesState,
        new_pos_size: torch.Tensor,
        close_trade_size: torch.Tensor,
        open_prices: torch.Tensor,
    ) -> TradesState:
        trades_sizes = trades_state.trades_sizes

        cum_trades_sizes = trades_sizes.cumsum(dim=0)
        left_diffs = close_trade_size + cum_trades_sizes

        right_diffs = torch.empty_like(left_diffs)
        right_diffs[1:] = left_diffs[:-1]
        right_diffs[0] = close_trade_size

        close_trade_mask = new_pos_size * left_diffs <= 0
        reduce_trade_mask = left_diffs * right_diffs < 0
        open_pos_mask = trades_state.pos_size * new_pos_size < 0

        new_trades_sizes = trades_sizes.clone()

        new_trades_sizes[close_trade_mask] = 0.0
        new_trades_sizes[reduce_trade_mask] = left_diffs[reduce_trade_mask]
        new_trades_sizes[-1, open_pos_mask] = new_pos_size[open_pos_mask]

        trades_prices = trades_state.trades_prices
        new_trades_prices = trades_prices.clone()
        new_trades_prices[close_trade_mask] = 0.0
        new_trades_prices[-1, open_pos_mask] = open_prices[open_pos_mask]

        return TradesState(new_trades_sizes, new_trades_prices)

    def update_trades_raw(
        self,
        trades_state: TradesState,
        new_pos_size: torch.Tensor,
        open_prices: torch.Tensor,
    ) -> TradesState:
        pos_size = trades_state.pos_size
        new_trade_size = close_trade_size = new_pos_size - pos_size

        add_case_trade_state = self._add_case_update_trades(
            trades_state, new_trade_size, open_prices
        )
        remove_case_trade_state = self._remove_case_update_trades(
            trades_state, new_pos_size, close_trade_size, open_prices
        )

        no_open_trade_mask = trades_state.trades_sizes[-1] == 0
        add_mask = no_open_trade_mask | (pos_size.abs() < new_pos_size.abs())
        new_trades_sizes = torch.where(
            add_mask,
            add_case_trade_state.trades_sizes,
            remove_case_trade_state.trades_sizes,
        )
        new_trades_prices = torch.where(
            add_mask,
            add_case_trade_state.trades_prices,
            remove_case_trade_state.trades_prices,
        )

        return TradesState(new_trades_sizes, new_trades_prices)

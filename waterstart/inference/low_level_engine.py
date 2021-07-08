from __future__ import annotations

from collections import Sequence
from dataclasses import dataclass
from typing import Collection, Mapping
from waterstart.symbols import TradedSymbolInfo
from waterstart.array_mapping.dict_based_mapper import DictBasedArrayMapper
from waterstart.array_mapping.market_data_mapper import MarketDataArrayMapper

import torch
import torch.distributions as dist
import torch.jit as jit
import torch.nn as nn

from ..inference.model import CNN, Emitter, GatedTransition
from . import (
    ModelInferenceWithRawOutput,
    ModelInput,
    RawMarketState,
    RawModelOutput,
    TradesState,
)
from .utils import map_to_tensors


class NetworkModules(nn.Module):
    def __init__(
        self,
        cnn: CNN,
        gated_trans: GatedTransition,
        iafs: Sequence[nn.Module],
        emitter: Emitter,
        market_data_arr_mapper: MarketDataArrayMapper,
        traded_sym_arr_mapper: DictBasedArrayMapper[int],
        traded_symbols: Collection[TradedSymbolInfo],
    ):
        super().__init__()

        if (n_traded_sym := cnn.n_traded_sym) != emitter.n_traded_sym:
            raise ValueError()

        if (market_features := cnn.market_features) != market_data_arr_mapper.n_fields:
            raise ValueError()

        id_to_traded_sym = {sym.id: sym for sym in traded_symbols}

        if len(traded_symbols) < len(id_to_traded_sym):
            raise ValueError()

        if id_to_traded_sym.keys() != traded_sym_arr_mapper.keys:
            raise ValueError()

        self._cnn = cnn
        self._gated_trans = gated_trans
        self._iafs = iafs
        self._emitter = emitter
        self._market_data_arr_mapper = market_data_arr_mapper
        self._traded_sym_arr_mapper = traded_sym_arr_mapper
        # self._traded_symbols = traded_symbols
        self._id_to_traded_sym = id_to_traded_sym

        self._n_traded_sym = n_traded_sym
        self._market_features = market_features
        self._max_trades = cnn.max_trades
        self._window_size = cnn.window_size

    @property
    def n_traded_sym(self) -> int:
        return self._n_traded_sym

    @property
    def market_features(self) -> int:
        return self._market_features

    @property
    def max_trades(self) -> int:
        return self._max_trades

    @property
    def cnn(self) -> CNN:
        return self._cnn

    @property
    def gated_trans(self) -> GatedTransition:
        return self._gated_trans

    @property
    def iafs(self) -> Sequence[nn.Module]:
        return self._iafs

    @property
    def emitter(self) -> Emitter:
        return self._emitter

    @property
    def market_data_arr_mapper(self) -> MarketDataArrayMapper:
        return self._market_data_arr_mapper

    @property
    def traded_sym_arr_mapper(self) -> DictBasedArrayMapper[int]:
        return self._traded_sym_arr_mapper

    @property
    def id_to_traded_sym(self) -> Mapping[int, TradedSymbolInfo]:
        return self._id_to_traded_sym


@dataclass
class MinStepMax:
    min: torch.Tensor
    step: torch.Tensor
    max: torch.Tensor

    def __post_init__(self):
        if not self.min.ndim == self.step.ndim == self.max.ndim == 1:
            raise ValueError()

        if not (size := self.min.numel()) == self.step.numel() == self.max.numel():
            raise ValueError()

        self._size = size

    @property
    def size(self) -> int:
        return self._size


# TODO: maybe rename to ModelEvaluator?
class LowLevelInferenceEngine:
    def __init__(self, net_modules: NetworkModules) -> None:
        self._net_modules = net_modules
        self._min_step_max = self._compute_min_step_max_arr(net_modules)
        self._scaling_idx = net_modules.market_data_arr_mapper.scaling_idxs

    @property
    def net_modules(self) -> NetworkModules:
        return self._net_modules

    @staticmethod
    def _compute_min_step_max_arr(net_modules: NetworkModules) -> MinStepMax:
        min_step_max_map = {
            sym_id: (
                (sym := syminfo.symbol).minVolume / 100 * sym.lotSize / 100,
                sym.stepVolume / 100 * sym.lotSize / 100,
                sym.maxVolume / 100 * sym.lotSize / 100,
            )
            for sym_id, syminfo in net_modules.id_to_traded_sym.items()
        }

        min, step, max = map_to_tensors(
            net_modules.traded_sym_arr_mapper, min_step_max_map
        )

        return MinStepMax(min=min, step=step, max=max)

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

        for i in range(self._net_modules.n_traded_sym):
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

    def evaluate(
        self, model_input: ModelInput[RawMarketState]
    ) -> ModelInferenceWithRawOutput:
        market_state = model_input.market_state

        # TODO: input checks

        scaled_market_data = market_state.market_data.clone()
        for src_ind, dst_inds in self._scaling_idx:
            scaled_market_data[:, dst_inds] /= scaled_market_data[  # type: ignore
                :, src_ind, None, -1, None
            ]

        trades_state = model_input.trades_state
        rel_prices = trades_state.trades_prices / market_state.midpoint_prices

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

        new_pos_sizes = self._compute_new_pos_sizes(
            model_output.fractions,
            model_output.exec_mask,
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
        net_modules = self._net_modules

        cnn_output: torch.Tensor = net_modules.cnn(
            scaled_market_data, trades_data.movedim(0, -1)
        )

        z_loc: torch.Tensor
        z_scale: torch.Tensor
        z_loc, z_scale = net_modules.gated_trans(cnn_output, hidden_state)

        z_dist = dist.TransformedDistribution(
            dist.Normal(z_loc, z_scale), net_modules.iafs
        )
        z_sample: torch.Tensor = z_dist.rsample()  # type: ignore
        z_logprob: torch.Tensor = z_dist.log_prob(z_sample)  # type: ignore

        exec_logits: torch.Tensor
        fractions: torch.Tensor
        exec_logits, fractions = net_modules.emitter(z_sample)

        exec_logits = exec_logits.movedim(-1, 0)
        fractions = fractions.movedim(-1, 0)

        exec_dist = dist.Bernoulli(logits=exec_logits)
        exec_samples: torch.Tensor = exec_dist.sample()  # type: ignore
        exec_logprobs: torch.Tensor = exec_dist.log_prob(exec_samples)  # type: ignore

        return RawModelOutput(
            cnn_output=cnn_output,
            z_sample=z_sample,
            z_logprob=z_logprob,
            exec_samples=exec_samples,
            exec_logprobs=exec_logprobs,
            fractions=fractions,
        )

    @staticmethod
    def open_trades(
        trades_state: TradesState,
        new_pos_size: torch.Tensor,
        open_prices: torch.Tensor,
    ) -> TradesState:
        pos_size = trades_state.pos_size
        trades_sizes = trades_state.trades_sizes

        reached_max_trades_mask = trades_sizes[0] != 0

        # NOTE: this mask matches the case pos_size != 0 and new_pos_size == 0
        # but is eliminated by the following condition
        same_sign_mask = pos_size * new_pos_size >= 0
        greater_new_pos_mask = pos_size.abs() < new_pos_size.abs()
        open_pos_mask = ~reached_max_trades_mask & same_sign_mask & greater_new_pos_mask

        new_trades_sizes = trades_sizes.clone()
        new_trades_sizes[:-1] = trades_sizes[1:]
        new_trades_sizes[-1] = new_pos_size
        new_trades_sizes = new_trades_sizes.where(open_pos_mask, trades_sizes)

        trades_prices = trades_state.trades_prices
        new_trades_prices = trades_prices.clone()
        new_trades_prices[:-1] = trades_prices[1:]
        new_trades_prices[-1] = open_prices
        new_trades_prices = new_trades_prices.where(open_pos_mask, trades_prices)

        return TradesState(new_trades_sizes, new_trades_prices)

    @staticmethod
    def close_or_reduce_trades(
        trades_state: TradesState,
        new_pos_size: torch.Tensor,
    ) -> TradesState:
        trades_sizes = trades_state.trades_sizes

        cum_trades_sizes = trades_sizes.cumsum(dim=0)
        close_trade_size = new_pos_size - trades_state.pos_size
        left_diffs = close_trade_size + cum_trades_sizes

        right_diffs = torch.empty_like(left_diffs)
        right_diffs[1:] = left_diffs[:-1]
        right_diffs[0] = close_trade_size

        close_trade_mask = new_pos_size * left_diffs <= 0
        reduce_trade_mask = left_diffs * right_diffs < 0

        new_trades_sizes = trades_sizes.clone()

        new_trades_sizes[close_trade_mask] = 0.0
        new_trades_sizes[reduce_trade_mask] = left_diffs[reduce_trade_mask]

        trades_prices = trades_state.trades_prices
        new_trades_prices = trades_prices.clone()
        new_trades_prices[close_trade_mask] = 0.0

        return TradesState(new_trades_sizes, new_trades_prices)

    # def update_trades(
    #     self,
    #     trades_state: TradesState,
    #     new_pos_size: torch.Tensor,
    #     open_prices: torch.Tensor,
    # ) -> TradesState:
    #     pos_size = trades_state.pos_size
    #     new_trade_size = close_trade_size = new_pos_size - pos_size

    #     add_case_trade_state = self._add_case_update_trades(
    #         trades_state, new_trade_size, open_prices
    #     )
    #     remove_case_trade_state = self._remove_case_update_trades(
    #         trades_state, new_pos_size, close_trade_size, open_prices
    #     )

    #     no_open_trade_mask = trades_state.trades_sizes[-1] == 0
    #     add_mask = no_open_trade_mask | (pos_size.abs() < new_pos_size.abs())
    #     new_trades_sizes = torch.where(
    #         add_mask,
    #         add_case_trade_state.trades_sizes,
    #         remove_case_trade_state.trades_sizes,
    #     )
    #     new_trades_prices = torch.where(
    #         add_mask,
    #         add_case_trade_state.trades_prices,
    #         remove_case_trade_state.trades_prices,
    #     )

    #     return TradesState(new_trades_sizes, new_trades_prices)

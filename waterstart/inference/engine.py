from __future__ import annotations

from collections import Mapping

import torch

from . import (
    MarketState,
    ModelInferenceWithMap,
    ModelInput,
    RawMarketState,
    TradesState,
)
from .low_level_engine import LowLevelInferenceEngine, NetworkModules
from .utils import (
    MaskedTensor,
    map_to_tensors,
    masked_tensor_to_partial_map,
    obj_to_tensor,
    partial_map_to_masked_tensors,
)


class InferenceEngine:
    def __init__(self, net_modules: NetworkModules) -> None:
        self._low_level_engine = LowLevelInferenceEngine(net_modules)
        self._market_data_arr_mapper = net_modules.market_data_arr_mapper
        self._traded_sym_arr_mapper = net_modules.traded_sym_arr_mapper

    def evaluate(self, model_input: ModelInput[MarketState]) -> ModelInferenceWithMap:
        market_state = model_input.market_state
        latest_market_data = market_state.latest_market_data
        # TODO: move to device?

        latest_market_data_arr = obj_to_tensor(
            self._market_data_arr_mapper, latest_market_data
        )

        market_data_arr = market_state.prev_market_data_arr
        market_data_arr[:, :-1] = market_data_arr[:, 1:].clone()
        market_data_arr[:, -1] = latest_market_data_arr

        symbols_data_map = latest_market_data.symbols_data_map

        market_state_map = {
            sym: (
                (sym_data := symbols_data_map[sym]).price_trendbar.close,
                sym_data.spread_trendbar.close,
                sym_data.base_to_dep_trendbar.close,
                sym_data.quote_to_dep_trendbar.close,
            )
            for sym in self._traded_sym_arr_mapper.keys
        }

        midpoint_prices, spreads, margin_rate, quote_to_dep_rate = map_to_tensors(
            self._traded_sym_arr_mapper, market_state_map
        )

        raw_market_state = RawMarketState(
            market_data=market_data_arr.unsqueeze(0),
            midpoint_prices=midpoint_prices.unsqueeze(-1),
            spreads=spreads.unsqueeze(-1),
            margin_rate=margin_rate.unsqueeze(-1),
            quote_to_dep_rate=quote_to_dep_rate.unsqueeze(-1),
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

        with torch.inference_mode():
            model_infer = self._low_level_engine.evaluate(raw_model_input)

        raw_model_output = model_infer.raw_model_output
        exec_mask = raw_model_output.exec_mask.squeeze(-1)

        new_pos_sizes = model_infer.pos_sizes.squeeze(-1)
        hidden_state = raw_model_output.z_sample.squeeze(0)

        new_pos_sizes_map = masked_tensor_to_partial_map(
            self._traded_sym_arr_mapper, MaskedTensor(new_pos_sizes, exec_mask)
        )

        return ModelInferenceWithMap(
            pos_sizes=new_pos_sizes,
            market_data_arr=market_data_arr,
            pos_sizes_map=new_pos_sizes_map,
            hidden_state=hidden_state,
        )

    def update_trades(
        self,
        trades_state: TradesState,
        new_pos_sizes_and_prices_map: Mapping[int, tuple[float, float]],
    ) -> TradesState:

        size_masked_tens, price_masked_tens = partial_map_to_masked_tensors(
            self._traded_sym_arr_mapper, new_pos_sizes_and_prices_map
        )

        if not torch.all(
            (opened_mask := size_masked_tens.mask) == price_masked_tens.mask
        ):
            raise ValueError()

        new_pos_sizes = size_masked_tens.arr
        new_pos_prices = price_masked_tens.arr

        new_pos_sizes = new_pos_sizes.where(opened_mask, trades_state.pos_size)
        new_pos_prices = new_pos_prices.where(opened_mask, new_pos_prices.new_zeros([]))

        low_level_engine = self._low_level_engine
        trades_state = low_level_engine.close_or_reduce_trades(
            trades_state, new_pos_sizes
        )
        trades_state = low_level_engine.open_trades(
            trades_state, new_pos_sizes, new_pos_prices
        )

        return trades_state

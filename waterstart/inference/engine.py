from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch

from ..array_mapping.utils import (
    MaskedArrays,
    map_to_arrays,
    masked_array_to_partial_map,
    obj_to_array,
    partial_map_to_masked_arrays,
)
from ..price import MarketData
from . import AccountState, ModelInput, MarketState
from .low_level_engine import LowLevelInferenceEngine, NetworkModules


class InferenceEngine:
    def __init__(self, net_modules: NetworkModules) -> None:
        self._low_level_engine = LowLevelInferenceEngine(net_modules)
        self._market_data_arr_mapper = net_modules.market_data_arr_mapper
        self._traded_sym_arr_mapper = net_modules.traded_sym_arr_mapper

        self._n_traded_sym = net_modules.n_traded_sym
        self._window_size = net_modules.window_size
        self._max_trades = net_modules.max_trades
        self._market_features = net_modules.raw_market_data_size
        hidden_state_size = net_modules.hidden_state_size

        self._account_state = AccountState(
            torch.zeros((self._max_trades, self._n_traded_sym, 1), dtype=torch.float32),
            torch.zeros((self._max_trades, self._n_traded_sym, 1), dtype=torch.float32),
            torch.zeros((1,)),
        )

        self._hidden_state = torch.zeros((hidden_state_size, 1))
        self._market_state_arr: Optional[torch.Tensor] = None
        self._market_data_list: list[MarketData[float]] = []

        self._raw_market_state: Optional[MarketState] = None

        self._sym_left_to_update: set[int] = set()
        self._trades_sizes_and_prices_map: dict[int, tuple[float, float]] = {}
        self._market_state_arr_full: bool = False
        self._balance_initialized = False

    @property
    def net_modules(self) -> NetworkModules:
        return self._low_level_engine.net_modules

    def _build_raw_market_state(
        self, market_data: MarketData[float], market_state_arr: torch.Tensor
    ) -> MarketState:
        symbols_data_map = market_data.symbols_data_map
        market_state_map = {
            sym: (
                (sym_data := symbols_data_map[sym]).price_trendbar.close,
                sym_data.spread_trendbar.close,
                sym_data.base_to_dep_trendbar.close,
                sym_data.quote_to_dep_trendbar.close,
            )
            for sym in self._traded_sym_arr_mapper.keys
        }

        midpoint_prices, spreads, margin_rate, quote_to_dep_rate = map(
            torch.from_numpy,  # type: ignore
            map_to_arrays(self._traded_sym_arr_mapper, market_state_map, np.float32),
        )

        return MarketState(
            market_data=market_state_arr.unsqueeze(0),
            midpoint_prices=midpoint_prices.unsqueeze(-1),
            spreads=spreads.unsqueeze(-1),
            margin_rate=margin_rate.unsqueeze(-1),
            quote_to_dep_rate=quote_to_dep_rate.unsqueeze(-1),
        )

    def evaluate(self) -> Mapping[int, float]:
        # TODO: suppose the balance it actually 0. We should throw an exception, right?
        if (
            not self._market_state_arr_full
            or self._sym_left_to_update
            or not self._balance_initialized
        ):
            raise RuntimeError()

        if (raw_market_state := self._raw_market_state) is None:
            raise RuntimeError()

        account_state = self._account_state
        raw_model_input = ModelInput(
            raw_market_state, account_state, self._hidden_state
        )

        with torch.inference_mode():
            model_infer = self._low_level_engine.evaluate(raw_model_input)

        raw_model_output = model_infer.raw_model_output
        exec_mask = raw_model_output.exec_mask & account_state.available_trades_mask

        new_pos_sizes = model_infer.pos_sizes
        new_pos_sizes_map = masked_array_to_partial_map(
            self._traded_sym_arr_mapper,
            MaskedArrays(
                new_pos_sizes.squeeze(-1).numpy(), exec_mask.squeeze(-1).numpy()
            ),
        )

        self._sym_left_to_update = set(new_pos_sizes_map)

        return new_pos_sizes_map

    def _shift_and_maybe_update_market_state(
        self, market_state_arr: torch.Tensor, market_data: Optional[MarketData[float]]
    ) -> None:
        if self._market_state_arr_full:
            market_state_arr[:, :-1] = market_state_arr[:, 1:].clone()
            self._market_state_arr_full = False
            self._raw_market_state = None

        if market_data is None:
            if not self._market_state_arr_full:
                raise RuntimeError()

            return

        market_state_arr[:, -1] = torch.from_numpy(  # type: ignore
            obj_to_array(self._market_data_arr_mapper, market_data)
        )
        self._raw_market_state = self._build_raw_market_state(
            market_data, market_state_arr
        )

    def _init_market_state(self, market_data: MarketData[float]) -> None:
        market_data_list = self._market_data_list
        market_data_list.append(market_data)

        if len(market_data_list) < self._window_size - 1:
            return

        market_state_arr = self._market_state_arr = torch.empty(
            (self._market_features, self._window_size), dtype=torch.float32
        )

        for i, market_data in enumerate(market_data_list):
            market_state_arr[:, i] = torch.from_numpy(  # type: ignore
                obj_to_array(self._market_data_arr_mapper, market_data)
            )

    def update_market_state(self, market_data: MarketData[float]) -> None:
        if (market_state_arr := self._market_state_arr) is None:
            self._init_market_state(market_data)
            return

        self._shift_and_maybe_update_market_state(market_state_arr, market_data)

    def update_balance(self, new_balance: float) -> None:
        self._balance_initialized = True
        self._account_state.balance[...] = new_balance  # type: ignore

    def shift_market_state_arr(self) -> None:
        if (market_state_arr := self._market_state_arr) is None:
            raise RuntimeError()

        self._shift_and_maybe_update_market_state(market_state_arr, None)

    def update_symbol_state(self, sym_id: int, size: float, price: float) -> bool:

        # TODO: this is only correct once the state is initialized
        # the first time around we should do this check!
        if not (sym_left_to_update := self._sym_left_to_update):
            raise RuntimeError()

        sym_left_to_update.remove(sym_id)
        # TODO: it would be good to check if the trade is closing or opening
        # so that we can check whether the balance should be present or not

        trades_sizes_and_prices_map = self._trades_sizes_and_prices_map
        trades_sizes_and_prices_map[sym_id] = (size, price)

        # if balance is not None:
        #     self.update_balance(balance)

        if done := not sym_left_to_update:
            self._update_trades(trades_sizes_and_prices_map)
            trades_sizes_and_prices_map.clear()

        return done

    def skip_symbol_update(self, sym_id: int) -> bool:
        if not (sym_left_to_update := self._sym_left_to_update):
            raise RuntimeError()

        sym_left_to_update.remove(sym_id)

        if done := not sym_left_to_update:
            trades_sizes_and_prices_map = self._trades_sizes_and_prices_map
            self._update_trades(trades_sizes_and_prices_map)
            trades_sizes_and_prices_map.clear()

        return done

    def _update_trades(
        self, trades_sizes_and_prices_map: Mapping[int, tuple[float, float]]
    ) -> None:
        account_state = self._account_state

        masked_arrs = partial_map_to_masked_arrays(
            self._traded_sym_arr_mapper, trades_sizes_and_prices_map, np.float32
        )
        opened_mask = torch.from_numpy(masked_arrs.mask)  # type: ignore
        new_trades_sizes, new_pos_prices = map(
            torch.from_numpy, masked_arrs  # type: ignore
        )

        new_pos_sizes = account_state.pos_size + new_trades_sizes

        low_level_engine = self._low_level_engine
        account_state, _ = low_level_engine.close_or_reduce_trades(
            account_state, new_pos_sizes
        )
        # TODO: find decent name
        account_state, opened_mask2 = low_level_engine.open_trades(
            account_state, new_pos_sizes, new_pos_prices
        )

        if not torch.all(opened_mask == opened_mask2):
            raise RuntimeError()

        self._account_state = account_state

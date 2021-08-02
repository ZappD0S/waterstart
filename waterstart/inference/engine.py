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
from . import AccountState, ModelInput, RawMarketState
from .low_level_engine import LowLevelInferenceEngine, NetworkModules


class InferenceEngine:
    def __init__(self, net_modules: NetworkModules) -> None:
        self._low_level_engine = LowLevelInferenceEngine(net_modules)
        self._market_data_arr_mapper = net_modules.market_data_arr_mapper
        self._traded_sym_arr_mapper = net_modules.traded_sym_arr_mapper

        self._n_traded_sym = net_modules.n_traded_sym
        self._window_size = net_modules.window_size
        self._max_trades = net_modules.max_trades
        self._market_features = net_modules.market_features
        hidden_state_size = net_modules.hidden_state_size

        self._account_state = AccountState(
            torch.zeros((self._max_trades, self._n_traded_sym, 1), dtype=torch.float32),
            torch.zeros((self._max_trades, self._n_traded_sym, 1), dtype=torch.float32),
            torch.zeros((1,)),
        )

        self._hidden_state = torch.zeros((hidden_state_size, 1))
        self._market_state_arr: Optional[torch.Tensor] = None
        self._market_data_list: list[MarketData[float]] = []

        self._market_state_arr_full: bool = False
        self._pending_account_state_update: bool = False
        self._pending_balance_update: bool = True

    @property
    def net_modules(self) -> NetworkModules:
        return self._low_level_engine.net_modules

    def evaluate(self, market_data: MarketData[float]) -> Mapping[int, float]:
        # TODO: move to device?
        if (market_state_arr := self._market_state_arr) is None:
            raise RuntimeError()

        if (
            self._market_state_arr_full
            or self._pending_account_state_update
            or self._pending_balance_update
        ):
            raise RuntimeError()

        latest_market_data_arr = torch.from_numpy(  # type: ignore
            obj_to_array(self._market_data_arr_mapper, market_data)
        )

        market_state_arr[:, -1] = latest_market_data_arr
        self._market_state_arr_full = True

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

        raw_market_state = RawMarketState(
            market_data=market_state_arr.unsqueeze(0),
            midpoint_prices=midpoint_prices.unsqueeze(-1),
            spreads=spreads.unsqueeze(-1),
            margin_rate=margin_rate.unsqueeze(-1),
            quote_to_dep_rate=quote_to_dep_rate.unsqueeze(-1),
        )

        account_state = self._account_state
        raw_model_input = ModelInput(
            raw_market_state, account_state, self._hidden_state
        )

        with torch.inference_mode():
            model_infer = self._low_level_engine.evaluate(raw_model_input)

        raw_model_output = model_infer.raw_model_output
        exec_mask = raw_model_output.exec_mask & account_state.availabe_trades_mask

        new_pos_sizes = model_infer.pos_sizes
        new_pos_sizes_map = masked_array_to_partial_map(
            self._traded_sym_arr_mapper,
            MaskedArrays(
                new_pos_sizes.squeeze(-1).numpy(), exec_mask.squeeze(-1).numpy()
            ),
        )

        self._pending_balance_update = self._pending_account_state_update = False
        return new_pos_sizes_map

    def _shift_and_maybe_update_market_state(
        self, market_state_arr: torch.Tensor, market_data: Optional[MarketData[float]]
    ):
        if self._market_state_arr_full != market_data is None:
            raise RuntimeError()

        self._market_state_arr_full = False

        market_state_arr[:, :-1] = market_state_arr[:, 1:].clone()

        if market_data is None:
            return

        market_state_arr[:, -2] = torch.from_numpy(  # type: ignore
            obj_to_array(self._market_data_arr_mapper, market_data)
        )

    def update_market_state(self, market_data: MarketData[float]) -> None:
        if (market_state_arr := self._market_state_arr) is not None:
            self._shift_and_maybe_update_market_state(market_state_arr, market_data)
            return

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

    def update_balance(
        self, new_balance: float, deposit_or_withdraw: bool = False
    ) -> None:
        if not deposit_or_withdraw:
            self._pending_balance_update = False

        self._account_state.balance[...] = new_balance  # type: ignore

    def shift_market_state(self) -> None:
        if (market_state_arr := self._market_state_arr) is None:
            raise RuntimeError()

        self._shift_and_maybe_update_market_state(market_state_arr, None)

    def update_state(
        self,
        new_sizes_and_prices_map: Mapping[int, tuple[float, float]],
        new_balance: float,
    ) -> None:
        if (market_state_arr := self._market_state_arr) is None:
            raise RuntimeError()

        self._shift_and_maybe_update_market_state(market_state_arr, None)
        self.update_balance(new_balance)
        self.update_trades(new_sizes_and_prices_map)

    def update_trades(
        self,
        new_sizes_and_prices_map: Mapping[int, tuple[float, float]],
        # as_trade_sizes: bool = False,
    ) -> AccountState:
        account_state = self._account_state

        masked_arrs = partial_map_to_masked_arrays(
            self._traded_sym_arr_mapper, new_sizes_and_prices_map, np.float32
        )
        opened_mask = torch.from_numpy(masked_arrs.mask)  # type: ignore
        new_sizes, new_pos_prices = map(torch.from_numpy, masked_arrs)  # type: ignore

        # if as_trade_sizes:
        #     new_trades_sizes = new_sizes.where(opened_mask, new_sizes.new_zeros())
        #     new_pos_sizes = account_state.pos_size + new_trades_sizes
        # else:
        new_pos_sizes = new_sizes.where(opened_mask, account_state.pos_size)

        new_pos_prices = new_pos_prices.where(opened_mask, new_pos_prices.new_zeros([]))

        low_level_engine = self._low_level_engine
        account_state = low_level_engine.close_or_reduce_trades(
            account_state, new_pos_sizes
        )
        # TODO: find decent name
        account_state, opened_mask2 = low_level_engine.open_trades(
            account_state, new_pos_sizes, new_pos_prices
        )

        if not torch.all(opened_mask == opened_mask2):
            raise RuntimeError()

        self._pending_account_state_update = False
        return account_state

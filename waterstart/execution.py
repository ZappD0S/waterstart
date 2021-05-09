from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np

from .array_mapping import DictBasedArrayMapper, MarketDataArrayMapper
from .model import Model
from .price import MarketData
from .symbols import TradedSymbolInfo

# TODO: We need a class that takes an instance of the one below that actually
# executes the orders. Eventually this class will implement Observer and will
# produce a Summary of the current state of the account and the trades that were
# opened or closed


class Executor:
    def __init__(
        self,
        model: Model,
        # TODO: eventually the we'll get the two objects below
        # from the model as well...
        market_data_arr_mapper: MarketDataArrayMapper,
        traded_sym_arr_mapper: DictBasedArrayMapper[TradedSymbolInfo],
    ) -> None:
        self._model = model
        self._market_data_arr_mapper = market_data_arr_mapper
        self._traded_sym_arr_mapper = traded_sym_arr_mapper

        self._win_len = model.win_len
        self._max_trades = model.max_trades
        self.n_sym = traded_sym_arr_mapper.n_fields
        self.n_feat = market_data_arr_mapper.n_fields

        self._market_data: np.ndarray = np.zeros(  # type: ignore
            (self.n_feat, self._win_len), dtype=float
        )
        self._trade_sizes: np.ndarray = np.zeros(  # type: ignore
            (self.n_sym, self._max_trades), dtype=float
        )
        self._trade_prices: np.ndarray = np.zeros(  # type: ignore
            (self.n_sym, self._max_trades), dtype=float
        )
        self._hidden_state: np.ndarray = np.zeros(  # type: ignore
            (model.hidden_dim,), dtype=float
        )

        self._min_step_max = self._compute_min_step_max_arr()
        # shape: (self.n_sym,)
        self._new_pos_sizes: Optional[np.ndarray] = None

    @property
    def validated(self) -> bool:
        return self._new_pos_sizes is None

    @staticmethod
    def _build_array(it: Iterable[tuple[int, float]]) -> np.ndarray:
        rec_dtype = np.dtype([("inds", int), ("vals", float)])
        rec_arr: np.ndarray = np.fromiter(it, dtype=rec_dtype)

        arr = np.full_like(rec_arr.size, np.nan, dtype=float)  # type: ignore
        inds = rec_arr["inds"]
        vals = rec_arr["vals"]
        arr[inds] = vals

        if np.isnan(arr).any():
            raise ValueError()

        return arr

    def _compute_min_step_max_arr(self) -> np.ndarray:
        min_step_max_map = {
            key: (
                key.symbol.minVolume / 100 * key.symbol.lotSize / 100,
                key.symbol.stepVolume / 100 * key.symbol.lotSize / 100,
                key.symbol.maxVolume / 100 * key.symbol.lotSize / 100,
            )
            for key in self._traded_sym_arr_mapper.keys
        }

        dtype = np.dtype(
            [
                ("inds", int),
                (
                    "vals",
                    [
                        ("min", float),
                        ("step", float),
                        ("max", float),
                    ],
                ),
            ]
        )
        rec_arr = np.fromiter(
            self._traded_sym_arr_mapper.iterate_index_to_value(min_step_max_map),
            dtype=dtype,
        )
        inds = rec_arr["inds"]
        min_step_max = rec_arr["vals"]
        return min_step_max[inds]

    def _round_pos_sizes(self, raw_pos_sizes: np.ndarray) -> np.ndarray:
        abs_pos_sizes = np.abs(raw_pos_sizes)
        signs = np.sign(raw_pos_sizes)

        abs_pos_sizes -= abs_pos_sizes % self._min_step_max["step"]
        abs_pos_sizes[abs_pos_sizes < self._min_step_max["min"]] = 0.0
        abs_pos_sizes[abs_pos_sizes > self._min_step_max["max"]] = self._min_step_max[
            "max"
        ]

        return signs * abs_pos_sizes

    def execute(
        self,
        # TODO: aggregate al the arguments in a single class?
        market_data: MarketData[float],
        sym_price_map: Mapping[TradedSymbolInfo, float],
        margin_rate_map: Mapping[TradedSymbolInfo, float],
        balance: float,
    ) -> Mapping[TradedSymbolInfo, float]:
        if self._new_pos_sizes is not None:
            raise RuntimeError()

        latest_market_data = self._build_array(
            self._market_data_arr_mapper.iterate_index_to_value(market_data)
        )

        self._market_data[:, :-1] = self._market_data[:, 1:].copy()
        self._market_data[:, -1] = latest_market_data

        sym_prices = self._build_array(
            self._traded_sym_arr_mapper.iterate_index_to_value(sym_price_map)
        )

        margin_rate = self._build_array(
            self._traded_sym_arr_mapper.iterate_index_to_value(margin_rate_map)
        )

        scaled_market_data = self._market_data.copy()
        for src_ind, dst_inds in self._market_data_arr_mapper.scaling_inds:
            scaled_market_data[dst_inds] /= scaled_market_data[src_ind, -1]

        rel_prices = self._trade_prices / sym_prices[:, np.newaxis]

        dep_cur_trade_sizes: np.ndarray = self._trade_sizes / margin_rate[:, np.newaxis]
        abs_pos_sizes = np.abs(dep_cur_trade_sizes).sum(axis=1)
        unused = balance - abs_pos_sizes.sum()
        assert unused >= 0
        rel_margins = np.append(dep_cur_trade_sizes.ravel(), unused) / balance

        trades_data: np.ndarray = np.concatenate([rel_margins, rel_prices])

        # TODO: should we do that within model?
        # TODO: add batch dim
        exec_mask, fractions, self._hidden_state = self._model(
            trades_data, scaled_market_data, self._hidden_state
        )
        # TODO: remove batch dim

        new_pos_sizes: np.ndarray = np.empty_like(abs_pos_sizes)  # type: ignore
        remaining = unused
        # TODO: numba
        for i in range(self._max_trades):
            available = abs_pos_sizes[i] + remaining
            new_pos_sizes[i] = fractions[i] * available
            remaining = available - abs(new_pos_sizes[i])

        new_pos_sizes = self._round_pos_sizes(new_pos_sizes)
        self._new_pos_sizes = new_pos_sizes

        exec_mask_list: list[bool] = exec_mask.tolist()
        new_pos_sizes_list: list[float] = new_pos_sizes.tolist()

        new_pos_sizes_map = self._traded_sym_arr_mapper.build_from_index_to_value_map(
            dict(enumerate(zip(exec_mask_list, new_pos_sizes_list)))
        )

        return {
            sym: pos_size for sym, (exec, pos_size) in new_pos_sizes_map.items() if exec
        }

    def validate(self, opened_pos_prices_map: Mapping[TradedSymbolInfo, float]) -> None:
        if self._new_pos_sizes is None:
            raise RuntimeError()

        open_prices_map = {
            sym: (
                opened_pos_prices_map[sym]
                if (found := sym in opened_pos_prices_map)
                else 0.0,
                found,
            )
            for sym in self._traded_sym_arr_mapper.keys
        }

        rec_arr = np.fromiter(
            self._traded_sym_arr_mapper.iterate_index_to_value(open_prices_map),
            dtype=[("price", float), ("mask", bool)],
        )

        success_mask = rec_arr["mask"]
        open_prices = rec_arr["price"]

        pos_sizes = self._trade_sizes.sum(axis=1)  # type: ignore
        new_pos_sizes = np.where(success_mask, self._new_pos_sizes, pos_sizes)

        new_trade_sizes = new_pos_sizes - pos_sizes

        add_trades_sizes = self._trade_sizes.copy()
        add_trades_sizes[:, :-1] = self._trade_sizes[:, 1:]
        add_trades_sizes[:, -1] = new_trade_sizes

        add_trades_prices = self._trade_prices.copy()
        add_trades_prices[:, :-1] = self._trade_prices[:, 1:]
        add_trades_prices[:, -1] = open_prices

        close_trade_sizes = new_trade_sizes
        cum_trades_sizes = self._trade_sizes.cumsum(axis=-1)  # type: ignore
        left_diffs = close_trade_sizes[:, np.newaxis] + cum_trades_sizes

        right_diffs = np.empty_like(left_diffs)  # type: ignore
        right_diffs[:, 1:] = left_diffs[:, :-1]
        right_diffs[:, 0] = close_trade_sizes

        close_trade_mask = new_pos_sizes * left_diffs <= 0
        reduce_trade_mask = left_diffs * right_diffs < 0
        open_pos_mask = pos_sizes * new_pos_sizes < 0

        remove_trades_sizes = self._trade_sizes.copy()

        remove_trades_sizes[close_trade_mask] = 0.0
        remove_trades_sizes[reduce_trade_mask] = left_diffs[reduce_trade_mask]
        remove_trades_sizes[open_pos_mask, -1] = new_pos_sizes[open_pos_mask]

        remove_trades_prices = self._trade_prices.copy()
        remove_trades_prices[close_trade_mask] = 0.0
        remove_trades_prices[open_pos_mask, -1] = open_prices[open_pos_mask]

        latest_trade_sizes = self._trade_sizes[:, -1]
        add_mask = (latest_trade_sizes * new_trade_sizes > 0) | (
            latest_trade_sizes == 0
        )
        self._trade_sizes = np.where(add_mask, add_trades_sizes, remove_trades_sizes)
        self._trade_prices = np.where(add_mask, add_trades_prices, remove_trades_prices)

        self._new_pos_sizes = None

from collections.abc import Mapping, Sequence

import numpy as np

from ..symbols import SymbolInfo, TradedSymbolInfo
from ..utils import is_contiguous
from . import AggregationData, BidAskTicks, SymbolData, Tick, TrendBar


class PriceAggregator:
    def __init__(self) -> None:
        self._data_type = np.dtype([("time", "f4"), ("price", "f4")])

    @staticmethod
    def _rescale(arr: np.ndarray, min: float, max: float) -> np.ndarray:
        return (arr - min) / (max - min)

    def _build_interp_data(self, data: Sequence[Tick]) -> tuple[np.ndarray, np.ndarray]:
        if not data:
            raise ValueError()

        data_arr = np.array(data, dtype=self._data_type)  # type: ignore

        time = data_arr["time"]
        price = data_arr["price"]

        if not np.all(time[:-1] <= time[1:]):
            raise ValueError()

        return time, price

    @staticmethod
    def _compute_hlc_array(data: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                data.max(axis=-1),  # type: ignore
                data.min(axis=-1),  # type: ignore
                data[..., -1],
            ],
            axis=-1,
        )

    def _update_trendbar(self, tb: TrendBar[float], hlc: np.ndarray) -> TrendBar[float]:
        if not (hlc.ndim == 1 and hlc.size == 3):
            raise ValueError()

        new_tb: TrendBar[float] = TrendBar(
            high=max(tb.high, hlc[0]),
            low=min(tb.low, hlc[1]),
            close=hlc[2],
        )
        return new_tb

    def aggregate(self, aggreg_data: AggregationData) -> AggregationData:
        tick_data_map = aggreg_data.tick_data_map
        sym_tb_data_map = aggreg_data.tb_data_map

        symbols_set = tick_data_map.keys()
        traded_symbols_set = sym_tb_data_map.keys()
        if not (traded_symbols_set and traded_symbols_set <= symbols_set):
            raise ValueError()

        n_symbols = len(symbols_set)
        n_traded_symbols = len(traded_symbols_set)
        sym_to_index: dict[SymbolInfo, int] = {
            sym: ind for ind, sym in enumerate(traded_symbols_set)
        }
        sym_to_index.update(
            (sym, ind + n_traded_symbols)
            for ind, sym in enumerate(symbols_set - traded_symbols_set)
        )
        assert is_contiguous(sorted(sym_to_index.values()))

        conv_chains_inds: tuple[list[int], list[int], list[int]] = ([], [], [])
        conv_chain_sym_inds: list[int] = []
        longest_chain_len = 0

        for traded_sym in sym_tb_data_map:
            traded_sym_ind = sym_to_index[traded_sym]

            chains = traded_sym.conv_chains
            for i, chain in enumerate((chains.base_asset, chains.quote_asset)):
                chain_len = len(chain)
                longest_chain_len = max(longest_chain_len, chain_len)

                conv_chain_sym_inds.extend(sym_to_index[sym] for sym in chain)

                conv_chains_inds[0].extend(range(chain_len))
                conv_chains_inds[1].extend((traded_sym_ind,) * chain_len)
                conv_chains_inds[2].extend((i,) * chain_len)

        # TODO: find better name
        sym_to_data_points: dict[
            SymbolInfo,
            tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        ] = {}
        dt = np.inf
        start, end = -np.inf, np.inf

        for sym, tick_data in tick_data_map.items():
            bid_times, bid_prices = self._build_interp_data(tick_data.bid)
            ask_times, ask_prices = self._build_interp_data(tick_data.ask)
            sym_to_data_points[sym] = ((bid_times, bid_prices), (ask_times, ask_prices))

            start = max(start, bid_times[0], ask_times[0])
            end = min(end, bid_times[-1], ask_times[-1])
            dt = min(dt, np.diff(bid_prices).min(), np.diff(ask_times).min())

        assert dt != np.inf
        assert longest_chain_len != 0
        steps = round(2 / dt)
        x = np.linspace(0, 1, steps, endpoint=True)  # type: ignore
        interp_arr = np.full((n_symbols, 2, x.size), np.nan)  # type: ignore

        new_tick_data_map: Mapping[SymbolInfo, BidAskTicks] = {}

        for traded_sym, bid_ask_ticks in tick_data_map.items():
            traded_sym_ind = sym_to_index[traded_sym]
            data_points = sym_to_data_points[traded_sym]

            interp_bid_ask_prices = (np.empty_like(x), np.empty_like(x))  # type: ignore

            for i, (times, prices) in enumerate(data_points):
                times = self._rescale(times, start, end)
                interp_bid_ask_prices[i][...] = np.interp(x, times, prices)

            interp_bid_prices, interp_ask_prices = interp_bid_ask_prices
            avg_prices = (interp_bid_prices + interp_ask_prices) / 2
            spreads = interp_ask_prices - interp_bid_prices
            assert np.all(spreads >= 0)
            interp_arr[traded_sym_ind, 0] = avg_prices
            interp_arr[traded_sym_ind, 1] = spreads

            left_tick_data: tuple[list[Tick], list[Tick]] = ([], [])

            for i, ticks in enumerate(bid_ask_ticks):
                times = data_points[i][0]
                last_used_index: int = times.searchsorted(end)  # type: ignore
                left_tick_data[i].extend(ticks[last_used_index:])

            new_tick_data_map[traded_sym] = BidAskTicks(*left_tick_data)

        assert not np.isnan(interp_arr).any()

        conv_chains_arr = np.ones((longest_chain_len, n_traded_symbols, 2, x.size))  # type: ignore

        conv_chains_arr[conv_chains_inds] = interp_arr[conv_chain_sym_inds, 0]
        conv_chains_arr: np.ndarray = conv_chains_arr.prod(axis=0)  # type: ignore

        price_spread_hlc = self._compute_hlc_array(interp_arr[:n_traded_symbols])
        conv_chains_hlc = self._compute_hlc_array(conv_chains_arr)

        new_sym_tb_data: Mapping[TradedSymbolInfo, SymbolData[float]] = {}

        for traded_sym, sym_tb_data in sym_tb_data_map.items():
            traded_sym_ind = sym_to_index[traded_sym]

            cur_price_tb = sym_tb_data.price_trendbar
            cur_spread_tb = sym_tb_data.spread_trendbar
            price_hlc, spread_hlc = price_spread_hlc[traded_sym_ind]

            new_price_tb = self._update_trendbar(cur_price_tb, price_hlc)
            new_spread_tb = self._update_trendbar(cur_spread_tb, spread_hlc)

            dep_to_base_tb = sym_tb_data.dep_to_base_trendbar
            dep_to_quote_tb = sym_tb_data.dep_to_quote_trendbar
            dep_to_base_hlc, dep_to_quote_hlc = conv_chains_hlc[traded_sym_ind]

            new_dep_to_base_tb = self._update_trendbar(dep_to_base_tb, dep_to_base_hlc)
            new_dep_to_quote_tb = self._update_trendbar(
                dep_to_quote_tb, dep_to_quote_hlc
            )

            new_sym_data: SymbolData[float] = SymbolData(
                price_trendbar=new_price_tb,
                spread_trendbar=new_spread_tb,
                dep_to_base_trendbar=new_dep_to_base_tb,
                dep_to_quote_trendbar=new_dep_to_quote_tb,
            )

            new_sym_tb_data[traded_sym] = new_sym_data

        return AggregationData(new_tick_data_map, new_sym_tb_data)

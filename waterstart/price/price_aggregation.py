from collections.abc import Mapping, Sequence

import numpy as np

from ..symbols import SymbolInfo, TradedSymbolInfo
from . import AggregationData, SymbolData, Tick, BidAskTicks, TrendBar


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
        new_tb: TrendBar[float] = TrendBar(
            high=max(tb.high, hlc[0]),
            low=min(tb.low, hlc[1]),
            close=hlc[2],
        )
        return new_tb

    def aggregate(self, aggreg_data: AggregationData) -> AggregationData:

        sym_to_index: dict[SymbolInfo, int] = {}
        # TODO: find better name
        sym_to_data_points: dict[
            SymbolInfo,
            tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        ] = {}
        dt = np.inf
        count = 0
        start, end = -np.inf, np.inf

        def update(sym: SymbolInfo) -> int:
            if sym in sym_to_index:
                return sym_to_index[sym]

            nonlocal count, dt, start, end

            index = count
            count += 1
            sym_to_index[sym] = index

            tick_data = tick_data_map[sym]
            bid_times, bid_prices = self._build_interp_data(tick_data.bid)
            ask_times, ask_prices = self._build_interp_data(tick_data.ask)
            sym_to_data_points[sym] = ((bid_times, bid_prices), (ask_times, ask_prices))

            start = max(start, bid_times[0], ask_times[0])
            end = min(end, bid_times[-1], ask_times[-1])
            dt = min(dt, np.diff(bid_prices).min(), np.diff(bid_prices).min())

            return index

        traded_sym_inds: list[int] = []
        conv_chains_inds: tuple[list[int], list[int], list[int]] = ([], [], [])
        conv_chain_sym_inds: list[int] = []
        max_chain_len = 0

        tick_data_map = aggreg_data.tick_data_map
        sym_tb_data_map = aggreg_data.tb_data_map

        traded_symbols = sym_tb_data_map.keys()

        for sym in traded_symbols:
            sym_index = update(sym)
            traded_sym_inds.append(sym_index)

            chains = sym.conv_chains
            for i, chain in enumerate([chains.base_asset, chains.quote_asset]):
                chain_len = len(chain)
                max_chain_len = max(max_chain_len, chain_len)

                inds = (update(sym) for sym in chain)
                conv_chain_sym_inds.extend(inds)

                conv_chains_inds[0].extend(range(chain_len))
                conv_chains_inds[1].extend([sym_index] * chain_len)
                conv_chains_inds[2].extend([i] * chain_len)

        assert dt != np.inf
        assert max_chain_len != 0
        steps = round(2 / dt)
        x = np.linspace(0, 1, steps, endpoint=True)  # type: ignore
        n_symbols = len(tick_data_map)
        n_traded_symbols = len(traded_symbols)
        interp_arr = np.full((n_symbols, 2, x.size), np.nan)  # type: ignore

        new_tick_data_map: Mapping[SymbolInfo, BidAskTicks] = {}

        for sym, data_points in tick_data_map.items():
            sym_index = sym_to_index[sym]
            data_points = sym_to_data_points[sym]

            interp_bid_ask_prices: tuple[np.ndarray, np.ndarray] = (np.empty_like(x), np.empty_like(x))  # type: ignore

            for i, (times, prices) in enumerate(data_points):
                times = self._rescale(times, start, end)
                interp_bid_ask_prices[i][...] = np.interp(x, times, prices)

            interp_bid_prices, interp_ask_prices = interp_bid_ask_prices
            avg_prices = (interp_bid_prices + interp_ask_prices) / 2
            spreads = interp_ask_prices - interp_bid_prices
            assert np.all(spreads >= 0)
            interp_arr[sym_index, 0] = avg_prices
            interp_arr[sym_index, 1] = spreads

            left_tick_data: tuple[list[Tick], list[Tick]] = ([], [])
            tick_data = tick_data_map[sym]

            for i, (times, _) in enumerate(data_points):
                last_used_index: int = times.searchsorted(end)  # type: ignore
                left_tick_data[i].extend(tick_data[i][last_used_index:])

            new_tick_data_map[sym] = BidAskTicks(*left_tick_data)

        assert not np.isnan(interp_arr).any()

        conv_chains_arr = np.ones((max_chain_len, n_traded_symbols, 2, x.size))  # type: ignore

        conv_chains_arr[conv_chains_inds] = interp_arr[conv_chain_sym_inds, 0]
        conv_chains_arr: np.ndarray = conv_chains_arr.prod(axis=0)  # type: ignore

        price_spread_hlc = self._compute_hlc_array(interp_arr[traded_sym_inds])
        conv_chains_hlc = self._compute_hlc_array(conv_chains_arr)

        new_sym_tb_data: Mapping[TradedSymbolInfo, SymbolData[float]] = {}

        for sym, sym_tb_data in sym_tb_data_map.items():
            sym_index = sym_to_index[sym]

            cur_price_tb = sym_tb_data.price_trendbar
            cur_spread_tb = sym_tb_data.spread_trendbar
            price_hlc, spread_hlc = price_spread_hlc[sym_index]

            new_price_tb = self._update_trendbar(cur_price_tb, price_hlc)
            new_spread_tb = self._update_trendbar(cur_spread_tb, spread_hlc)

            dep_to_base_tb = sym_tb_data.dep_to_base_trendbar
            dep_to_quote_tb = sym_tb_data.dep_to_quote_trendbar
            dep_to_base_hlc, dep_to_quote_hlc = conv_chains_hlc[sym_index]

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

            new_sym_tb_data[sym] = new_sym_data

        return AggregationData(new_tick_data_map, new_sym_tb_data)

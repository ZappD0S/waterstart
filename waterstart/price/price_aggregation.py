from collections.abc import Mapping, Sequence

import numpy as np
from scipy.interpolate import interp1d  # type: ignore

from ..symbols import SymbolInfo, TradedSymbolInfo
from . import PriceSnapshot, SymbolData, TrendBar, AggregationData


class PriceAggregator:
    def __init__(self) -> None:
        self._data_type = np.dtype([("bid", "f4"), ("ask", "f4"), ("time", "f4")])

    def _build_interp_data(
        self, data: Sequence[PriceSnapshot]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not data:
            raise ValueError()

        data_arr = np.array(data, dtype=self._data_type)  # type: ignore

        price = (data_arr["bid"] + data_arr["ask"]) / 2
        spread = data_arr["ask"] - data_arr["bid"]
        time = data_arr["time"]

        if not np.all(time[:-1] <= time[1:]):
            raise ValueError()

        xp = time
        yp = np.stack([price, spread])
        return xp, yp

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

    def aggregate(self, aggreg_data: AggregationData) -> AggregationData:

        sym_to_index: dict[SymbolInfo, int] = {}
        sym_to_data_points: dict[SymbolInfo, tuple[np.ndarray, np.ndarray]] = {}
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

            xp, yp = self._build_interp_data(raw_data_map[sym])
            sym_to_data_points[sym] = (xp, yp)

            start = max(start, xp[0])
            end = min(end, xp[-1])
            dt = min(dt, np.diff(xp).min())

            return index

        traded_sym_inds: list[int] = []
        conv_chains_inds: tuple[list[int], list[int], list[int]] = ([], [], [])
        conv_chain_sym_inds: list[int] = []
        max_chain_len = 0

        raw_data_map = aggreg_data.raw_data_map
        sym_tb_data_map = aggreg_data.tb_data_map

        traded_symbols = sym_tb_data_map.keys()

        for sym in traded_symbols:
            index = update(sym)
            traded_sym_inds.append(index)

            chains = sym.conv_chains
            for i, chain in enumerate([chains.base_asset, chains.quote_asset]):
                chain_len = len(chain)
                max_chain_len = max(max_chain_len, chain_len)

                inds = (update(sym) for sym in chain)
                conv_chain_sym_inds.extend(inds)

                conv_chains_inds[0].extend(range(chain_len))
                conv_chains_inds[1].extend([index] * chain_len)
                conv_chains_inds[2].extend([i] * chain_len)

        assert dt != np.inf
        assert max_chain_len != 0
        steps = round(2 / dt)
        x = np.linspace(0, 1, steps, endpoint=True)  # type: ignore
        n_symbols = len(raw_data_map)
        n_traded_symbols = len(traded_symbols)
        interp_arr = np.full((n_symbols, 2, x.size), np.nan)  # type: ignore

        new_raw_data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]] = {}

        for sym, sym_raw_data in raw_data_map.items():
            xp, yp = sym_to_data_points[sym]

            last_used_index: int = xp.searchsorted(end)  # type: ignore
            new_raw_data_map[sym] = sym_raw_data[last_used_index:]

            index = sym_to_index[sym]
            xp = (xp - start) / (end - start)
            interp = interp1d(xp, yp, copy=False, assume_sorted=True)
            interp_arr[index] = interp(x)  # type: ignore

        assert not np.isnan(interp_arr).any()

        conv_chains_arr = np.ones((max_chain_len, n_traded_symbols, 2, x.size))  # type: ignore

        conv_chains_arr[conv_chains_inds] = interp_arr[conv_chain_sym_inds, 0]
        conv_chains_arr: np.ndarray = conv_chains_arr.prod(axis=0)  # type: ignore

        price_spread_hlc = self._compute_hlc_array(interp_arr[traded_sym_inds])
        conv_chains_hlc = self._compute_hlc_array(conv_chains_arr)

        new_sym_tb_data: Mapping[TradedSymbolInfo, SymbolData[float]] = {}

        def update_trendbar(tb: TrendBar[float], hlc: np.ndarray) -> TrendBar[float]:
            new_tb: TrendBar[float] = TrendBar(
                high=max(tb.high, hlc[0]),
                low=min(cur_price_tb.low, hlc[1]),
                close=hlc[2],
            )
            return new_tb

        for sym, sym_tb_data in sym_tb_data_map.items():
            index = sym_to_index[sym]

            cur_price_tb = sym_tb_data.price_trendbar
            cur_spread_tb = sym_tb_data.spread_trendbar
            price_hlc, spread_hlc = price_spread_hlc[index]

            new_price_tb = update_trendbar(cur_price_tb, price_hlc)
            new_spread_tb = update_trendbar(cur_spread_tb, spread_hlc)

            dep_to_base_tb = sym_tb_data.dep_to_base_trendbar
            dep_to_quote_tb = sym_tb_data.dep_to_quote_trendbar
            dep_to_base_hlc, dep_to_quote_hlc = conv_chains_hlc[index]

            new_dep_to_base_tb = update_trendbar(dep_to_base_tb, dep_to_base_hlc)
            new_dep_to_quote_tb = update_trendbar(dep_to_quote_tb, dep_to_quote_hlc)

            new_sym_data: SymbolData[float] = SymbolData(
                price_trendbar=new_price_tb,
                spread_trendbar=new_spread_tb,
                dep_to_base_trendbar=new_dep_to_base_tb,
                dep_to_quote_trendbar=new_dep_to_quote_tb,
            )

            new_sym_tb_data[sym] = new_sym_data

        return AggregationData(new_raw_data_map, new_sym_tb_data)

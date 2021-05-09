from collections.abc import Mapping, Sequence
from typing import TypeVar, cast

import numpy as np
from scipy.interpolate import interp1d  # type: ignore

from ..symbols import SymbolInfo, TradedSymbolInfo
from . import MarketData, PriceSnapshot, BasePriceAggregator

T = TypeVar("T")


class PriceAggregator(BasePriceAggregator[MarketData[float]]):
    def __init__(self) -> None:
        self._data_type = np.dtype([("bid", "f4"), ("ask", "f4"), ("time", "f4")])

    def _build_interp_data(
        self, data: Sequence[PriceSnapshot]
    ) -> tuple[np.ndarray, np.ndarray]:
        data_arr = np.array(data, dtype=self._data_type)  # type: ignore

        price = (data_arr["bid"] + data_arr["ask"]) / 2
        spread = data_arr["ask"] - data_arr["bid"]
        time = data_arr["time"]

        if not np.all(time[:-1] <= time[1:]):
            raise ValueError()

        xp = (time - time[0]) / (time[-1] - time[0])
        yp = np.stack([price, spread])
        return xp, yp

    @staticmethod
    def _compute_hlc_array(data: np.ndarray) -> np.ndarray:
        return np.stack([data.max(axis=-1), data.min(axis=-1), data[..., -1]], axis=-1)  # type: ignore

    # TODO: this method is too long
    def aggregate(
        self,
        data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]],
        traded_symbols: Sequence[TradedSymbolInfo],
        time_of_day: float,
        delta_to_last: float,
    ) -> MarketData[float]:
        if not 0.0 <= time_of_day < 1.0:
            raise ValueError()

        if delta_to_last <= 0.0:
            raise ValueError()

        sym_to_index: dict[SymbolInfo, int] = {}
        index_to_interp: dict[int, interp1d] = {}
        dt = np.inf
        count = 0

        def update(sym: SymbolInfo) -> int:
            if sym in sym_to_index:
                return sym_to_index[sym]

            nonlocal count, dt

            index = count
            count += 1
            sym_to_index[sym] = index

            xp, yp = self._build_interp_data(data_map[sym])

            interp = interp1d(xp, yp, copy=False, assume_sorted=True)
            index_to_interp[index] = interp

            dt = min(dt, np.diff(xp).min())

            return index

        traded_sym_inds: list[int] = []
        conv_chains_inds: tuple[list[int], list[int], list[int]] = ([], [], [])
        conv_chain_sym_inds: list[int] = []
        max_chain_len = 0

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
        n_symbols = len(data_map)
        n_traded_symbols = len(traded_symbols)
        interp_arr = np.full((n_symbols, 2, x.size), np.nan)  # type: ignore

        for index, interp in index_to_interp.items():
            interp_arr[index] = interp(x)  # type: ignore

        assert not np.isnan(interp_arr).any()

        conv_chains_arr = np.ones((max_chain_len, n_traded_symbols, 2, x.size))  # type: ignore

        conv_chains_arr[conv_chains_inds] = interp_arr[conv_chain_sym_inds, 0]
        conv_chains_arr = cast(np.ndarray, conv_chains_arr.prod(axis=0))  # type: ignore

        price_spread_hlc = self._compute_hlc_array(interp_arr[traded_sym_inds])
        conv_chains_hlc = self._compute_hlc_array(conv_chains_arr)

        # sym_data_map: Mapping[TradedSymbolInfo, SymbolData] = {}

        # for sym in traded_symbols:
        #     index = sym_to_index[sym]
        #     price_hlc, spread_hlc = hlc_arr[index]
        #     base_asset_inds, quote_asset_inds = conv_chain_inds_map[sym]

        #     hlcs = [
        #         price_hlc,
        #         spread_hlc,
        #         interp_arr[base_asset_inds, 0].prod(axis=0),
        #         interp_arr[quote_asset_inds, 0].prod(axis=0),
        #     ]

        #     sym_tbs = (TrendBar(*hlc) for hlc in hlcs)
        #     sym_data_map[sym] = SymbolData(*sym_tbs)

        # return MarketData(
        #     sym_data_map, time_of_day=time_of_day, delta_to_last=delta_to_last
        # )

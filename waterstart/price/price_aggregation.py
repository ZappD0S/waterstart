from collections import Collection, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ..symbols import SymbolInfo, TradedSymbolInfo
from ..utils import is_contiguous
from . import AggregationData, SymbolData, Tick, TrendBar

T = TypeVar("T", bound=np.floating)


class PriceAggregator:
    def __init__(
        self,
        traded_symbols: Collection[TradedSymbolInfo],
        symbols: Collection[SymbolInfo],
    ) -> None:
        if not traded_symbols:
            raise ValueError()

        id_to_traded_sym = {sym.id: sym for sym in traded_symbols}
        id_to_sym = {sym.id: sym for sym in symbols}
        n_traded_symbols = len(traded_symbols)

        if n_traded_symbols < len(id_to_traded_sym):
            raise ValueError()

        n_symbols = len(symbols)

        if n_symbols < len(id_to_sym):
            raise ValueError()

        if not (id_to_traded_sym.keys() <= id_to_sym.keys()):
            raise ValueError()

        self._traded_symbols = traded_symbols
        self._symbols = symbols

        sym_id_to_idx = {sym_id: idx for idx, sym_id in enumerate(id_to_traded_sym)}
        sym_id_to_idx.update(
            (sym_id, idx + n_traded_symbols)
            for idx, sym_id in enumerate(id_to_sym.keys() - id_to_traded_sym.keys())
        )
        assert is_contiguous(sorted(sym_id_to_idx.values()))

        assets_set = {
            asset_id
            for sym in traded_symbols
            for asset_id in (
                sym.conv_chains.base_asset.asset_id,
                sym.conv_chains.quote_asset.asset_id,
            )
        }

        asset_id_to_idx = {asset_id: idx for idx, asset_id in enumerate(assets_set)}

        conv_chains_idxs: tuple[list[int], list[int]] = ([], [])
        conv_chain_sym_idxs: list[int] = []
        reciprocal_mask_idxs: tuple[list[int], list[int]] = ([], [])

        dep_to_base_quote_idxs: tuple[list[int], list[int]] = ([], [])
        dep_to_base_quote_asset_idxs: list[int] = []

        longest_chain_len = 0

        for traded_sym_id, traded_sym in id_to_traded_sym.items():
            traded_sym_idx = sym_id_to_idx[traded_sym_id]

            chains = traded_sym.conv_chains
            for i, chain in enumerate((chains.base_asset, chains.quote_asset)):
                chain_len = len(chain)
                longest_chain_len = max(longest_chain_len, chain_len)
                asset_idx = asset_id_to_idx[chain.asset_id]

                conv_chain_sym_idxs.extend(sym_id_to_idx[sym.id] for sym in chain)
                conv_chains_idxs[0].extend(range(chain_len))
                conv_chains_idxs[1].extend((asset_idx,) * chain_len)

                reciprocal_idxs = [i for i, sym in enumerate(chain) if sym.reciprocal]
                reciprocal_mask_idxs[0].extend(reciprocal_idxs)
                reciprocal_mask_idxs[1].extend((asset_idx,) * len(reciprocal_idxs))

                dep_to_base_quote_asset_idxs.append(asset_idx)
                dep_to_base_quote_idxs[0].append(traded_sym_idx)
                dep_to_base_quote_idxs[1].append(i)

        self._n_symbols = n_symbols
        self._n_assets = len(assets_set)
        self._n_traded_symbols = n_traded_symbols
        self._sym_id_to_idx = sym_id_to_idx

        # TODO: turn the lists of indices into arrays
        self._conv_chains_idxs = conv_chains_idxs
        self._conv_chain_sym_idxs = conv_chain_sym_idxs

        assert longest_chain_len != 0
        self._longest_chain_len = longest_chain_len

        self._reciprocal_mask: npt.NDArray[bool] = np.zeros(  # type: ignore
            (longest_chain_len, self._n_assets), dtype=bool
        )
        self._reciprocal_mask[reciprocal_mask_idxs] = True

        self._dep_to_base_quote_idxs = dep_to_base_quote_idxs
        self._dep_to_base_quote_asset_idxs = dep_to_base_quote_idxs

    @staticmethod
    def _rescale(
        arr: npt.NDArray[np.float32], min: float, max: float
    ) -> npt.NDArray[np.float32]:
        return (arr - min) / (max - min)

    def _build_interp_data(
        self, data: Sequence[Tick]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        if not data:
            raise ValueError()

        data_arr = np.array(  # type: ignore
            data, dtype=[("time", "f4"), ("price", "f4")]
        )

        time: npt.NDArray[np.float32] = data_arr["time"]
        price: npt.NDArray[np.float32] = data_arr["price"]

        if not np.all(time[:-1] <= time[1:]):  # type: ignore
            raise ValueError()

        return time, price

    @staticmethod
    def _compute_hlc_array(data: npt.NDArray[T]) -> npt.NDArray[T]:
        return np.stack(  # type: ignore
            [
                data.max(axis=-1),  # type: ignore
                data.min(axis=-1),  # type: ignore
                data[..., -1],
            ],
            axis=-1,
        )

    def _update_trendbar(
        self, tb: TrendBar[float], hlc: npt.NDArray[np.float32]
    ) -> TrendBar[float]:
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

        if tick_data_map.keys() != self._symbols:
            raise ValueError()

        if sym_tb_data_map.keys() != self._traded_symbols:
            raise ValueError()

        # TODO: find better name
        sym_to_data_points: dict[
            int,
            tuple[
                tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
                tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
            ],
        ] = {}
        dt = np.inf
        start, end = -np.inf, np.inf

        for sym_id, tick_data in tick_data_map.items():
            bid_times, bid_prices = self._build_interp_data(tick_data.bid)
            ask_times, ask_prices = self._build_interp_data(tick_data.ask)
            sym_to_data_points[sym_id] = (
                (bid_times, bid_prices),
                (ask_times, ask_prices),
            )

            start = max(start, bid_times[0], ask_times[0])
            end = min(end, bid_times[-1], ask_times[-1])
            dt = min(
                dt,
                np.diff(bid_prices).min(),  # type: ignore
                np.diff(ask_times).min(),  # type: ignore
            )

        assert dt != np.inf
        steps = round(2 / dt)
        x: npt.NDArray[np.float32]
        x = np.linspace(0, 1, steps, endpoint=True, dtype=np.float32)  # type: ignore
        interp_arr: npt.NDArray[np.float32]
        interp_arr = np.full((self._n_symbols, 2, x.size), np.nan)  # type: ignore

        sym_id_to_idx = self._sym_id_to_idx

        for sym_id, bid_ask_ticks in tick_data_map.items():
            traded_sym_idx = sym_id_to_idx[sym_id]
            data_points = sym_to_data_points[sym_id]

            interp_bid_ask_prices = (np.empty_like(x), np.empty_like(x))  # type: ignore

            for i, (times, prices) in enumerate(data_points):
                times = self._rescale(times, start, end)
                interp_prices: npt.NDArray[np.float32]
                interp_prices = np.interp(x, times, prices)  # type: ignore
                interp_bid_ask_prices[i][...] = interp_prices

            interp_bid_prices, interp_ask_prices = interp_bid_ask_prices
            avg_prices = (interp_bid_prices + interp_ask_prices) / 2
            spreads = interp_ask_prices - interp_bid_prices
            assert np.all(spreads >= 0)  # type: ignore
            interp_arr[traded_sym_idx, 0] = avg_prices
            interp_arr[traded_sym_idx, 1] = spreads

            for i, ticks in enumerate(bid_ask_ticks):
                times = data_points[i][0]
                last_used_idx: int = times.searchsorted(end)  # type: ignore
                del ticks[:last_used_idx]

        assert not np.isnan(interp_arr).any()  # type: ignore

        conv_chains_arr = np.ones(  # type: ignore
            (self._longest_chain_len, self._n_assets, x.size)
        )
        conv_chains_arr[self._conv_chains_idxs] = interp_arr[
            self._conv_chain_sym_idxs, 0
        ]
        conv_chains_arr[self._reciprocal_mask] **= -1
        conv_chains_arr = conv_chains_arr.prod(axis=0)  # type: ignore

        n_traded_symbols = self._n_traded_symbols
        dep_to_base_quote_arr: npt.NDArray[np.float32] = np.full(  # type: ignore
            (n_traded_symbols, 2, x.size), np.nan, dtype=np.float32
        )
        dep_to_base_quote_arr[self._dep_to_base_quote_idxs] = conv_chains_arr[
            self._dep_to_base_quote_asset_idxs
        ]
        assert not np.isnan(dep_to_base_quote_arr).any()  # type: ignore

        traded_interp_arr: npt.NDArray[np.float32] = interp_arr[:n_traded_symbols]
        price_spread_hlc = self._compute_hlc_array(traded_interp_arr)
        dep_to_base_quote_hlc = self._compute_hlc_array(dep_to_base_quote_arr)

        new_sym_tb_data: dict[int, SymbolData[float]] = {}

        # TODO: maybe update sym_tb_data_map instead of creating a new one?
        for traded_sym_id, sym_tb_data in sym_tb_data_map.items():
            traded_sym_idx = sym_id_to_idx[traded_sym_id]

            price_tb = sym_tb_data.price_trendbar
            spread_tb = sym_tb_data.spread_trendbar
            price_hlc, spread_hlc = price_spread_hlc[traded_sym_idx]

            new_price_tb = self._update_trendbar(price_tb, price_hlc)
            new_spread_tb = self._update_trendbar(spread_tb, spread_hlc)

            dep_to_base_tb = sym_tb_data.dep_to_base_trendbar
            dep_to_quote_tb = sym_tb_data.dep_to_quote_trendbar
            dep_to_base_hlc, dep_to_quote_hlc = dep_to_base_quote_hlc[traded_sym_idx]

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

            new_sym_tb_data[traded_sym_id] = new_sym_data

        return AggregationData(tick_data_map, new_sym_tb_data)

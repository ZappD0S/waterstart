from collections.abc import Sequence, Set

import numpy as np

from ..symbols import SymbolInfo, TradedSymbolInfo
from ..utils import is_contiguous
from . import AggregationData, BidAskTicks, SymbolData, Tick, TrendBar


class PriceAggregator:
    def __init__(
        self, traded_symbols_set: Set[TradedSymbolInfo], symbols_set: Set[SymbolInfo]
    ) -> None:
        if not (traded_symbols_set and traded_symbols_set <= symbols_set):
            raise ValueError()

        self._traded_symbols_set = traded_symbols_set
        self._symbols_set = symbols_set

        n_traded_symbols = len(traded_symbols_set)

        sym_to_index: dict[SymbolInfo, int] = {
            sym: ind for ind, sym in enumerate(traded_symbols_set)
        }
        sym_to_index.update(
            (sym, ind + n_traded_symbols)
            for ind, sym in enumerate(symbols_set - traded_symbols_set)
        )
        assert is_contiguous(sorted(sym_to_index.values()))

        assets_set = {
            asset_id
            for sym in traded_symbols_set
            for asset_id in (
                sym.conv_chains.base_asset.asset_id,
                sym.conv_chains.quote_asset.asset_id,
            )
        }

        asset_to_index = {asset_id: ind for ind, asset_id in enumerate(assets_set)}

        conv_chains_inds: tuple[list[int], list[int]] = ([], [])
        conv_chain_sym_inds: list[int] = []
        reciprocal_mask_inds: tuple[list[int], list[int]] = ([], [])

        dep_to_base_quote_inds: tuple[list[int], list[int]] = ([], [])
        dep_to_base_quote_asset_inds: list[int] = []

        longest_chain_len = 0

        for traded_sym in traded_symbols_set:
            traded_sym_ind = sym_to_index[traded_sym]

            chains = traded_sym.conv_chains
            for i, chain in enumerate((chains.base_asset, chains.quote_asset)):
                chain_len = len(chain)
                longest_chain_len = max(longest_chain_len, chain_len)
                asset_ind = asset_to_index[chain.asset_id]

                conv_chain_sym_inds.extend(sym_to_index[sym] for sym in chain)
                conv_chains_inds[0].extend(range(chain_len))
                conv_chains_inds[1].extend((asset_ind,) * chain_len)

                reciprocal_inds = [i for i, sym in enumerate(chain) if sym.reciprocal]
                reciprocal_mask_inds[0].extend(reciprocal_inds)
                reciprocal_mask_inds[1].extend((asset_ind,) * len(reciprocal_inds))

                dep_to_base_quote_asset_inds.append(asset_ind)
                dep_to_base_quote_inds[0].append(traded_sym_ind)
                dep_to_base_quote_inds[1].append(i)

        self._n_symbols = len(symbols_set)
        self._n_assets = len(assets_set)
        self._n_traded_symbols = n_traded_symbols
        self._sym_to_index = sym_to_index

        self._conv_chains_inds = conv_chains_inds
        self._conv_chain_sym_inds = conv_chain_sym_inds

        assert longest_chain_len != 0
        self._longest_chain_len = longest_chain_len

        self._reciprocal_mask = np.zeros(  # type: ignore
            (longest_chain_len, self._n_assets), dtype=bool
        )
        self._reciprocal_mask[reciprocal_mask_inds] = True

        self._dep_to_base_quote_inds = dep_to_base_quote_inds
        self._dep_to_base_quote_asset_inds = dep_to_base_quote_inds

    @staticmethod
    def _rescale(arr: np.ndarray, min: float, max: float) -> np.ndarray:
        return (arr - min) / (max - min)

    def _build_interp_data(self, data: Sequence[Tick]) -> tuple[np.ndarray, np.ndarray]:
        if not data:
            raise ValueError()

        data_arr = np.array(  # type: ignore
            data, dtype=[("time", "f4"), ("price", "f4")]
        )

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

        if tick_data_map.keys() != self._symbols_set:
            raise ValueError()

        if sym_tb_data_map.keys() != self._traded_symbols_set:
            raise ValueError()

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
        steps = round(2 / dt)
        x = np.linspace(0, 1, steps, endpoint=True)  # type: ignore
        interp_arr = np.full((self._n_symbols, 2, x.size), np.nan)  # type: ignore

        sym_to_index = self._sym_to_index
        new_tick_data_map: dict[SymbolInfo, BidAskTicks] = {}

        for sym, bid_ask_ticks in tick_data_map.items():
            traded_sym_ind = sym_to_index[sym]
            data_points = sym_to_data_points[sym]

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

            new_tick_data_map[sym] = BidAskTicks(*left_tick_data)

        assert not np.isnan(interp_arr).any()

        conv_chains_arr = np.ones(  # type: ignore
            (self._longest_chain_len, self._n_assets, x.size)
        )
        conv_chains_arr[self._conv_chains_inds] = interp_arr[
            self._conv_chain_sym_inds, 0
        ]
        conv_chains_arr[self._reciprocal_mask] **= -1
        conv_chains_arr = conv_chains_arr.prod(axis=0)  # type: ignore

        n_traded_symbols = self._n_traded_symbols
        dep_to_base_quote_arr = np.full(  # type: ignore
            (n_traded_symbols, 2, x.size), np.nan
        )
        dep_to_base_quote_arr[self._dep_to_base_quote_inds] = conv_chains_arr[
            self._dep_to_base_quote_asset_inds
        ]
        assert not np.isnan(dep_to_base_quote_arr).any()

        price_spread_hlc = self._compute_hlc_array(interp_arr[:n_traded_symbols])
        dep_to_base_quote_hlc = self._compute_hlc_array(dep_to_base_quote_arr)

        new_sym_tb_data: dict[TradedSymbolInfo, SymbolData[float]] = {}

        for traded_sym, sym_tb_data in sym_tb_data_map.items():
            traded_sym_ind = sym_to_index[traded_sym]

            cur_price_tb = sym_tb_data.price_trendbar
            cur_spread_tb = sym_tb_data.spread_trendbar
            price_hlc, spread_hlc = price_spread_hlc[traded_sym_ind]

            new_price_tb = self._update_trendbar(cur_price_tb, price_hlc)
            new_spread_tb = self._update_trendbar(cur_spread_tb, spread_hlc)

            dep_to_base_tb = sym_tb_data.dep_to_base_trendbar
            dep_to_quote_tb = sym_tb_data.dep_to_quote_trendbar
            dep_to_base_hlc, dep_to_quote_hlc = dep_to_base_quote_hlc[traded_sym_ind]

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

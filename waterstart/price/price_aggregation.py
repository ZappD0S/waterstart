from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np
from scipy.interpolate import interp1d

from ..symbols import SymbolInfo, SymbolInfoWithConvChains
from .price_tracker import PriceSnapshot

T = TypeVar("T", bound=Enum)


class BasePriceAggregator(Generic[T], ABC):
    @abstractmethod
    def aggregate(
        self,
        data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]],
        traded_symbols: Sequence[SymbolInfoWithConvChains],
        time_of_day: float,
        time_delta: float,
    ) -> Mapping[SymbolInfoWithConvChains, Mapping[T, float]]:
        ...


class TrendBarData(Enum):
    SYM_HIGH = auto()
    SYM_LOW = auto()
    SYM_CLOSE = auto()

    SPREAD_HIGH = auto()
    SPREAD_LOW = auto()
    SPREAD_CLOSE = auto()

    BASE_DEP_HIGH = auto()
    BASE_DEP_LOW = auto()
    BASE_DEP_CLOSE = auto()

    QUOTE_DEP_HIGH = auto()
    QUOTE_DEP_LOW = auto()
    QUOTE_DEP_CLOSE = auto()

    TIME_OF_DAY = auto()
    TIME_DELTA = auto()


class PriceAggregator(BasePriceAggregator[TrendBarData]):
    def __init__(self) -> None:
        self._data_type = np.dtype([("bid", "f4"), ("ask", "f4"), ("time", "f4")])

    def _build_interp_data(
        self, data: Sequence[PriceSnapshot]
    ) -> tuple[np.ndarray, np.ndarray]:
        data_arr = np.array(data, dtype=self._data_type)

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
        return np.stack([data.max(axis=-1), data.min(axis=-1), data[..., -1]], axis=-1)

    def _build_hlc_map(
        self,
        hlc_data: np.ndarray,
        hlc_fields_list: Sequence[tuple[TrendBarData, TrendBarData, TrendBarData]],
    ) -> Iterator[tuple[TrendBarData, float]]:
        if not 1 <= hlc_data.ndim <= 2:
            raise ValueError()

        if hlc_data.shape[-1] != 3:
            raise ValueError()

        # hlc_data = hlc_data[tuple(np.newaxis for _ in range(2 - hlc_data.ndim))]
        hlc_data = hlc_data[np.newaxis if hlc_data.ndim == 1 else slice(None)]

        if hlc_data.shape[0] != len(hlc_fields_list):
            raise ValueError()

        for hlc_fields, hlc_vals in zip(hlc_fields_list, hlc_data.tolist()):
            yield from zip(hlc_fields, hlc_vals)

    def aggregate(
        self,
        data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]],
        traded_symbols: Sequence[SymbolInfoWithConvChains],
        time_of_day: float,
        time_delta: float,
    ) -> Mapping[SymbolInfoWithConvChains, Mapping[TrendBarData, float]]:
        if not 0.0 <= time_of_day < 1.0:
            raise ValueError()

        if time_delta <= 0.0:
            raise ValueError()

        sym_to_index: dict[SymbolInfo, int] = {}
        index_to_interp: dict[int, interp1d] = {}
        conv_chain_inds_map: dict[SymbolInfo, tuple[list[int], list[int]]] = {}
        dt = np.inf
        count = 0

        def update(sym: SymbolInfo) -> int:
            if sym in sym_to_index:
                return sym_to_index[sym]

            try:
                sym_data = data_map[sym]
            except KeyError:
                raise ValueError()

            nonlocal count, dt

            index = count
            count += 1
            sym_to_index[sym] = index

            xp, yp = self._build_interp_data(sym_data)

            interp = interp1d(xp, yp, copy=False, assume_sorted=True)
            index_to_interp[index] = interp

            dt = min(dt, np.diff(xp).min())

            return index

        for sym in traded_symbols:
            _ = update(sym)

            chains = sym.conv_chains
            base_asset_inds = [update(sym) for sym in chains.base_asset]
            quote_asset_inds = [update(sym) for sym in chains.quote_asset]

            conv_chain_inds_map[sym] = (base_asset_inds, quote_asset_inds)

        assert dt != np.inf
        steps = round(2 / dt)
        x = np.linspace(0, 1, steps, endpoint=True)
        interpolated = np.full((len(index_to_interp), 2, x.size), np.nan)

        for index, interp in index_to_interp.items():
            interpolated[index] = interp(x)

        assert not np.isnan(interpolated).any()

        hlc_data = self._compute_hlc_array(interpolated)
        hlc_fields: list[tuple[TrendBarData, TrendBarData, TrendBarData]]

        res: Mapping[SymbolInfoWithConvChains, Mapping[TrendBarData, float]] = {}

        for sym in traded_symbols:
            sym_tb_data: dict[TrendBarData, float] = {
                TrendBarData.TIME_OF_DAY: time_of_day,
                TrendBarData.TIME_DELTA: time_delta,
            }

            hlc_fields = [
                (TrendBarData.SYM_HIGH, TrendBarData.SYM_LOW, TrendBarData.SYM_CLOSE),
                (
                    TrendBarData.SPREAD_HIGH,
                    TrendBarData.SPREAD_LOW,
                    TrendBarData.SPREAD_CLOSE,
                ),
            ]
            index = sym_to_index[sym]
            sym_tb_data.update(self._build_hlc_map(hlc_data[index], hlc_fields))

            base_asset_inds, quote_asset_inds = conv_chain_inds_map[sym]

            base_to_dep_data = interpolated[base_asset_inds, 0].prod(axis=0)
            base_to_dep_hlc_data = self._compute_hlc_array(base_to_dep_data)
            hlc_fields = [
                (
                    TrendBarData.BASE_DEP_HIGH,
                    TrendBarData.BASE_DEP_LOW,
                    TrendBarData.BASE_DEP_CLOSE,
                )
            ]
            sym_tb_data.update(self._build_hlc_map(base_to_dep_hlc_data, hlc_fields))

            quote_to_dep_data = interpolated[quote_asset_inds, 0].prod(axis=0)
            quote_to_dep_hlc_data = self._compute_hlc_array(quote_to_dep_data)
            hlc_fields = [
                (
                    TrendBarData.QUOTE_DEP_HIGH,
                    TrendBarData.QUOTE_DEP_LOW,
                    TrendBarData.QUOTE_DEP_CLOSE,
                )
            ]
            sym_tb_data.update(self._build_hlc_map(quote_to_dep_hlc_data, hlc_fields))

            # res[TrendBarData.TIME_OF_DAY] = time_of_day
            # res[TrendBarData.TIME_DELTA] = time_delta

            res[sym] = sym_tb_data

        return res

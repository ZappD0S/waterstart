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
        # NOTE: the two arrays are price and spread
        data_map: Mapping[int, Sequence[PriceSnapshot]],
        time_of_day: float,
        time_delta: float,
    ) -> Iterable[tuple[int, Mapping[T, float]]]:
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
    def __init__(self, symbol_map: Mapping[int, SymbolInfoWithConvChains]) -> None:
        self._symbol_map = symbol_map
        self._id_to_index = {sym_id: index for index, sym_id in enumerate(symbol_map)}
        self._data_type = np.dtype([("bid", "f4"), ("ask", "f4"), ("time", "f4")])

        def _build_index_list(sym_infos: Sequence[SymbolInfo]):
            return [self._id_to_index[sym_info.id] for sym_info in sym_infos]

        self._conv_chains_index_map = {
            sym_id: (
                _build_index_list(sym_info.conv_chains.base_asset),
                _build_index_list(sym_info.conv_chains.quote_asset),
            )
            for sym_id, sym_info in self._symbol_map.items()
        }

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
        data_map: Mapping[int, Sequence[PriceSnapshot]],
        time_of_day: float,
        time_delta: float,
    ) -> Iterator[tuple[int, Mapping[TrendBarData, float]]]:
        if not data_map.keys() == self._symbol_map.keys():
            raise ValueError()

        if not 0 <= time_of_day < 1:
            raise ValueError()

        if time_delta <= 0:
            raise ValueError()

        index_to_interp: dict[int, interp1d] = {}
        dt = np.inf

        for sym_id, data in data_map.items():
            xp, yp = self._build_interp_data(data)
            interp = interp1d(xp, yp, copy=False, assume_sorted=True)
            index = self._id_to_index[sym_id]
            index_to_interp[index] = interp

            dt = min(dt, np.diff(xp).min())

        assert dt != np.inf
        steps = round(2 / dt)
        x = np.linspace(0, 1, steps, endpoint=True)
        interpolated = np.full((len(index_to_interp), 2, x.size), np.nan)

        for index, interp in index_to_interp.items():
            interpolated[index] = interp(x)

        assert not np.isnan(interpolated).any()

        hlc_data = self._compute_hlc_array(interpolated)
        hlc_fields: list[tuple[TrendBarData, TrendBarData, TrendBarData]]

        for sym_id, (
            base_asset_inds,
            quote_asset_inds,
        ) in self._conv_chains_index_map.items():
            res: dict[TrendBarData, float] = {
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
            index = self._id_to_index[sym_id]
            res.update(self._build_hlc_map(hlc_data[index], hlc_fields))

            base_to_dep_data = interpolated[base_asset_inds, 0].prod(axis=0)
            base_to_dep_hlc_data = self._compute_hlc_array(base_to_dep_data)
            hlc_fields = [
                (
                    TrendBarData.BASE_DEP_HIGH,
                    TrendBarData.BASE_DEP_LOW,
                    TrendBarData.BASE_DEP_CLOSE,
                )
            ]
            res.update(self._build_hlc_map(base_to_dep_hlc_data, hlc_fields))

            quote_to_dep_data = interpolated[quote_asset_inds, 0].prod(axis=0)
            quote_to_dep_hlc_data = self._compute_hlc_array(quote_to_dep_data)
            hlc_fields = [
                (
                    TrendBarData.QUOTE_DEP_HIGH,
                    TrendBarData.QUOTE_DEP_LOW,
                    TrendBarData.QUOTE_DEP_CLOSE,
                )
            ]
            res.update(self._build_hlc_map(quote_to_dep_hlc_data, hlc_fields))

            # res[TrendBarData.TIME_OF_DAY] = time_of_day
            # res[TrendBarData.TIME_DELTA] = time_delta

            yield sym_id, res

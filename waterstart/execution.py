import asyncio
from abc import ABC
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import ClassVar, Generic, Type, TypeVar

import numpy as np

from waterstart.client import OpenApiClient
from waterstart.symbols import SymbolInfo


@dataclass(frozen=True)
class Trendbar:
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class PriceSnapshot:
    sym_trendbar: Trendbar
    base_deposit_rate: float
    quote_deposit_rate: float
    time_of_day: float

    def __post_init__(self):
        if not 0 <= self.time_of_day < 1:
            raise ValueError()


@dataclass(frozen=True)
class BaseIndices(ABC):
    _check_inds: ClassVar[bool] = False

    def get_inds(self) -> Iterable[int]:
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, int):
                yield val
            elif isinstance(val, BaseIndices):
                yield from val.get_inds()
            else:
                raise ValueError()

    def __post_init__(self):
        if not self._check_inds:
            return

        sorted_inds = sorted(self.get_inds())

        if not sorted_inds:
            raise ValueError()

        if sorted_inds[0] != 0:
            raise ValueError()

        if not all(b - a == 1 for a, b in zip(sorted_inds[:-1], sorted_inds[1:])):
            print(sorted_inds)
            raise ValueError()


@dataclass(frozen=True)
class TrendbarIndices(BaseIndices):
    high_ind: int
    low_ind: int
    close_ind: int


@dataclass(frozen=True)
class VectorIndices(BaseIndices):
    _check_inds: ClassVar[bool] = True

    sym_trendbar_inds: TrendbarIndices
    quote_to_dep_ind: int
    base_to_dep_ind: int
    time_of_day_ind: int
    delta_ind: int


T = TypeVar("T", bound=IntEnum)


class Executor(Generic[T]):
    def __init__(
        self,
        client: OpenApiClient,
        enum_type: Type[T],
        sym_to_ind_map: Mapping[SymbolInfo, int],
        win_len: int,
        max_trades: int,
    ) -> None:
        self.client = client
        # TODO check consectuive
        self._sym_to_ind_map = sym_to_ind_map
        self._ind_to_sym_map: Mapping[int, SymbolInfo] = {
            ind: sym for sym, ind in sym_to_ind_map.items()
        }
        self.n_sym = len(sym_to_ind_map)
        self.n_feat = len(enum_type)

        # TODO: add batch dim before passing to the model
        self._market_data = np.zeros(
            (self.n_feat, self.n_sym, win_len), dtype=np.float32
        )
        self._pos_data = np.zeros((self.n_sym, max_trades), dtype=np.float32)

    def get_inds(
        self, prices_map: Mapping[SymbolInfo, Iterable[tuple[T, float]]]
    ) -> Iterator[tuple[int, int, float]]:
        for sym, it in prices_map.items():
            sym_ind = self._sym_to_ind_map[sym]
            for spec, val in it:
                yield sym_ind, spec.value, val

    async def execute(self, prices_map: Mapping[SymbolInfo, Iterable[tuple[T, float]]]):
        rec_arr = np.fromiter(
            self.get_inds(prices_map),
            [
                ("sym_id", np.int64),
                ("feat_id", np.int64),
                ("value", np.float32),
            ],
        )

        sym_inds = rec_arr["sym_id"]
        feat_inds = rec_arr["feat_id"]
        vals = rec_arr["value"]

        self._market_data = np.roll(self._market_data, shift=-1, axis=2)
        latest_market_data = self._market_data[..., -1]
        latest_market_data[...] = np.nan
        latest_market_data[feat_inds, sym_inds] = vals

        if np.isnan(latest_market_data).any():
            raise ValueError()

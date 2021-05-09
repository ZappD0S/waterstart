from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

from ..symbols import SymbolInfo, TradedSymbolInfo

T = TypeVar("T")


class PriceSnapshot(NamedTuple):
    bid: float
    ask: float
    time: float


@dataclass
class TrendBar(Generic[T]):
    high: T
    low: T
    close: T


@dataclass
class SymbolData(Generic[T]):
    price_trendbar: TrendBar[T]
    spread_trendbar: TrendBar[T]
    dep_to_base_trendbar: TrendBar[T]
    dep_to_quote_trendbar: TrendBar[T]


@dataclass
class MarketData(Generic[T]):
    symbols_data_map: Mapping[TradedSymbolInfo, SymbolData[T]]
    time_of_day: T
    delta_to_last: T


class BasePriceAggregator(Generic[T], ABC):
    @abstractmethod
    def aggregate(
        self,
        data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]],
        traded_symbols: Sequence[TradedSymbolInfo],
        time_of_day: float,
        delta_to_last: float,
    ) -> T:
        ...

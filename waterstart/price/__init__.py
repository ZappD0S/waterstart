from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, NamedTuple, TypeVar

from ..symbols import SymbolInfo, TradedSymbolInfo

T = TypeVar("T")


@dataclass
class LatestMarketData:
    market_feat: MarketFeatures[float]
    sym_prices_map: Mapping[TradedSymbolInfo, float]
    margin_rate_map: Mapping[TradedSymbolInfo, float]


class PriceSnapshot(NamedTuple):
    bid: float
    ask: float
    time: float


@dataclass
class TrendBar(Generic[T]):
    high: T
    low: T
    close: T

    @staticmethod
    def build_default() -> TrendBar[float]:
        tb: TrendBar[float] = TrendBar(high=float("-inf"), low=float("inf"), close=0)
        return tb


@dataclass
class SymbolData(Generic[T]):
    price_trendbar: TrendBar[T]
    spread_trendbar: TrendBar[T]
    dep_to_base_trendbar: TrendBar[T]
    dep_to_quote_trendbar: TrendBar[T]

    @staticmethod
    def build_default() -> SymbolData[float]:
        return SymbolData(
            TrendBar.build_default(),
            TrendBar.build_default(),
            TrendBar.build_default(),
            TrendBar.build_default(),
        )


@dataclass
class MarketFeatures(Generic[T]):
    symbols_data_map: Mapping[TradedSymbolInfo, SymbolData[T]]
    time_of_day: T
    delta_to_last: T


@dataclass
class AggregationData:
    raw_data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]]
    tb_data_map: Mapping[TradedSymbolInfo, SymbolData[float]]


# class BasePriceAggregator(ABC):
#     @abstractmethod
#     def aggregate(
#         self,
#         data_map: Mapping[SymbolInfo, Sequence[PriceSnapshot]],
#         current_symbols_data: Mapping[TradedSymbolInfo, SymbolData[float]],
#     ) -> Mapping[TradedSymbolInfo, SymbolData[float]]:
#         ...

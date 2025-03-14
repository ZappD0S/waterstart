from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T", covariant=True)


@dataclass
class TrendBar(Generic[T]):
    high: T
    low: T
    close: T

    @staticmethod
    def build_default() -> TrendBar[float]:
        return TrendBar(high=float("-inf"), low=float("inf"), close=0.0)


@dataclass
class SymbolData(Generic[T]):
    price_trendbar: TrendBar[T]
    spread_trendbar: TrendBar[T]
    base_to_dep_trendbar: TrendBar[T]
    quote_to_dep_trendbar: TrendBar[T]

    @staticmethod
    def build_default() -> SymbolData[float]:
        return SymbolData(
            TrendBar.build_default(),
            TrendBar.build_default(),
            TrendBar.build_default(),
            TrendBar.build_default(),
        )


@dataclass
class MarketData(Generic[T]):
    symbols_data_map: Mapping[int, SymbolData[T]]
    time_of_day: T
    delta_to_last: T


class Tick(NamedTuple):
    time: float
    price: float


class TickType(IntEnum):
    BID = 0
    ASK = 1


@dataclass
class TickData:
    sym_id: int
    type: TickType
    tick: Tick


class BidAskTicks(NamedTuple):
    bid: list[Tick]
    ask: list[Tick]

    def __add__(self, other: BidAskTicks) -> BidAskTicks:  # type: ignore
        return BidAskTicks(self.bid + other.bid, self.ask + other.ask)


@dataclass
class AggregationData:
    bid_ask_ticks_map: Mapping[int, BidAskTicks]
    tb_data_map: Mapping[int, SymbolData[float]]

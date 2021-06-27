from __future__ import annotations

from abc import ABC, abstractmethod
from collections import AsyncGenerator, Iterator, Mapping, MutableSequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Collection, Final, Generic, Iterable, NamedTuple, TypeVar

from ..openapi import ProtoOATrader
from ..schedule import ExecutionSchedule
from ..symbols import SymbolInfo, TradedSymbolInfo

T = TypeVar("T")


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
class MarketData(Generic[T]):
    symbols_data_map: dict[TradedSymbolInfo, SymbolData[T]]
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
    sym: SymbolInfo
    type: TickType
    tick: Tick


class BidAskTicks(NamedTuple):
    bid: MutableSequence[Tick]
    ask: MutableSequence[Tick]


@dataclass
class AggregationData:
    tick_data_map: Mapping[SymbolInfo, BidAskTicks]
    tb_data_map: Mapping[TradedSymbolInfo, SymbolData[float]]


class BaseTickDataGenerator(ABC):
    PRICE_CONV_FACTOR: Final[int] = 10 ** 5

    def __init__(
        self,
        trader: ProtoOATrader,
        exec_schedule: ExecutionSchedule,
    ):
        # TODO: this will not be necessary anymore..
        self.trader = trader
        # TODO: make these properties?
        self.exec_schedule = exec_schedule
        traded_symbols = exec_schedule.traded_symbols
        id_to_symbol = {sym.id: sym for sym in self._get_all_symbols(traded_symbols)}
        self.symbols: Collection[SymbolInfo] = id_to_symbol.values()
        self.traded_symbols = traded_symbols

    @staticmethod
    def _get_all_symbols(
        traded_symbols: Iterable[TradedSymbolInfo],
    ) -> Iterator[SymbolInfo]:
        for traded_sym in traded_symbols:
            yield traded_sym

            chains = (
                traded_sym.conv_chains.base_asset,
                traded_sym.conv_chains.quote_asset,
            )

            for chain in chains:
                for sym in chain:
                    yield sym

    @abstractmethod
    def generate_ticks(self) -> AsyncGenerator[TickData, None]:
        ...

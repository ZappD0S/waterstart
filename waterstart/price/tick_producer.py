from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Collection, Iterable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncContextManager, Final

from ..client.trader import TraderClient
from ..openapi import (
    ProtoOASpotEvent,
    ProtoOASubscribeSpotsReq,
    ProtoOASubscribeSpotsRes,
    ProtoOAUnsubscribeSpotsReq,
    ProtoOAUnsubscribeSpotsRes,
)
from ..symbols import SymbolInfo, TradedSymbolInfo
from ..utils import ComposableAsyncIterable
from . import Tick, TickData, TickType


class BaseTicksProducer(ABC):
    PRICE_CONV_FACTOR: Final[int] = 10 ** 5

    @abstractmethod
    def generate_ticks_up_to(self, end: float) -> AsyncIterator[TickData]:
        ...


class BaseTicksProducerFactory(ABC):
    def __init__(self, traded_symbols: Collection[TradedSymbolInfo]):
        self._traded_symbols = traded_symbols
        self._id_to_sym = {sym.id: sym for sym in self._get_all_symbols(traded_symbols)}
        self._symbols = self._id_to_sym.values()

    @property
    def traded_symbols(self) -> Collection[TradedSymbolInfo]:
        return self._traded_symbols

    @property
    def symbols(self) -> Collection[SymbolInfo]:
        return self._symbols

    @staticmethod
    def _get_all_symbols(
        traded_symbols: Iterable[TradedSymbolInfo],
    ) -> Iterator[SymbolInfo]:
        for traded_sym in traded_symbols:
            yield traded_sym

            chains = traded_sym.conv_chains
            for chain in (chains.base_asset, chains.quote_asset):
                for sym in chain:
                    yield sym

    @abstractmethod
    def get_ticks_gen(self, start: float) -> AsyncContextManager[BaseTicksProducer]:
        ...


class LiveTicksProducerFactory(BaseTicksProducerFactory):
    def __init__(
        self, traded_symbols: Collection[TradedSymbolInfo], client: TraderClient
    ):
        super().__init__(traded_symbols)
        self._client = client
        self._id_to_sym = {sym.id: sym for sym in self._symbols}

    @asynccontextmanager
    async def get_ticks_gen(self, start: float) -> AsyncIterator[BaseTicksProducer]:
        _ = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOASubscribeSpotsReq(
                ctidTraderAccountId=trader_id, symbolId=self._id_to_sym
            ),
            ProtoOASubscribeSpotsRes,
        )

        try:
            # TODO: ideally we would like to have maxsize=1
            # so that we are sure that the events returned are up to date
            # but we need to check if this way we lose too many of them
            async with self._client.register_type_for_trader(ProtoOASpotEvent) as gen:
                yield LiveTicksProducer(start, gen)  # type: ignore
        finally:
            _ = await self._client.send_request_from_trader(
                lambda trader_id: ProtoOAUnsubscribeSpotsReq(
                    ctidTraderAccountId=trader_id, symbolId=self._id_to_sym
                ),
                ProtoOAUnsubscribeSpotsRes,
            )


class LiveTicksProducer(BaseTicksProducer):
    def __init__(self, start: float, gen: AsyncIterator[ProtoOASpotEvent]) -> None:
        self._gen = gen
        self._start = start

    async def generate_ticks_up_to(self, end: float) -> AsyncIterator[TickData]:
        PRICE_CONV_FACTOR = self.PRICE_CONV_FACTOR

        async def timeout_it():
            yield await timeout

        now = time.time()
        await asyncio.sleep(self._start - now)
        sentinel = object()
        timeout = asyncio.sleep(end - now, result=sentinel)

        async with AsyncExitStack() as stack:
            event_gen = await stack.enter_async_context(
                ComposableAsyncIterable.from_it(self._gen)
            )

            timeout_gen = await stack.enter_async_context(
                ComposableAsyncIterable.from_it(timeout_it())
            )

            async for maybe_event in event_gen | timeout_gen:
                if maybe_event is sentinel:
                    break

                t = time.time()
                event = maybe_event
                assert isinstance(event, ProtoOASpotEvent)
                sym_id = event.symbolId

                bid_tick = Tick(event.bid / PRICE_CONV_FACTOR, t)
                ask_tick = Tick(event.ask / PRICE_CONV_FACTOR, t)

                yield TickData(sym_id, TickType.BID, bid_tick)
                yield TickData(sym_id, TickType.ASK, ask_tick)

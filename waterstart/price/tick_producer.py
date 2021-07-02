from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import AsyncIterator, Awaitable, Collection, Iterable, Iterator
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Final

from ..client.app import AppClient
from ..openapi import (
    ProtoOASpotEvent,
    ProtoOASubscribeSpotsReq,
    ProtoOASubscribeSpotsRes,
    ProtoOATrader,
    ProtoOAUnsubscribeSpotsReq,
    ProtoOAUnsubscribeSpotsRes,
)
from ..symbols import SymbolInfo, TradedSymbolInfo
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
    def get_ticks_gen_starting_from(
        self, start: float
    ) -> AsyncContextManager[BaseTicksProducer]:
        ...


class LiveTicksProducerFactory(BaseTicksProducerFactory):
    def __init__(
        self,
        traded_symbols: Collection[TradedSymbolInfo],
        trader: ProtoOATrader,
        client: AppClient,
    ):
        super().__init__(traded_symbols)
        self._trader = trader
        self._client = client
        self._id_to_sym = {sym.id: sym for sym in self._symbols}

    @asynccontextmanager
    async def get_ticks_generator_starting_from(
        self, start: float
    ) -> AsyncIterator[BaseTicksProducer]:
        spot_sub_req = ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=self._trader.ctidTraderAccountId,
            symbolId=self._id_to_sym,
        )

        _ = await self._client.send_request(spot_sub_req, ProtoOASubscribeSpotsRes)

        try:
            async with self._client.register_type(ProtoOASpotEvent) as gen:
                yield LiveTicksProducer(start, gen)
        finally:
            spot_unsub_req = ProtoOAUnsubscribeSpotsReq(
                ctidTraderAccountId=self._trader.ctidTraderAccountId,
                symbolId=self._id_to_sym,
            )
            _ = await self._client.send_request(
                spot_unsub_req, ProtoOAUnsubscribeSpotsRes
            )


class LiveTicksProducer(BaseTicksProducer):
    def __init__(self, start: float, gen: AsyncIterator[ProtoOASpotEvent]) -> None:
        self._gen = gen
        self._start = start

    async def generate_ticks_up_to(self, end: float) -> AsyncIterator[TickData]:
        gen = self._gen
        PRICE_CONV_FACTOR = self.PRICE_CONV_FACTOR

        async def next_event() -> ProtoOASpotEvent:
            async for event in gen:
                return event

            raise RuntimeError()

        now = time.time()
        await asyncio.sleep(self._start - now)
        timeout: Awaitable[Any] = asyncio.sleep(end - now)
        timeout_task = asyncio.create_task(timeout)

        while True:
            event_task = asyncio.create_task(next_event())
            done: asyncio.Task[Any]
            [done], _ = await asyncio.wait(
                (timeout_task, event_task), return_when=asyncio.FIRST_COMPLETED
            )

            if done == timeout_task:
                break

            t = time.time()
            event = await event_task
            sym_id = event.symbolId

            yield TickData(sym_id, TickType.BID, Tick(event.bid / PRICE_CONV_FACTOR, t))
            yield TickData(sym_id, TickType.ASK, Tick(event.ask / PRICE_CONV_FACTOR, t))

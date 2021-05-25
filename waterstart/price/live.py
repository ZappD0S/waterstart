import time
from collections import AsyncGenerator
from contextlib import asynccontextmanager

from ..client import OpenApiClient
from ..openapi import (
    ProtoOASpotEvent,
    ProtoOASubscribeSpotsReq,
    ProtoOASubscribeSpotsRes,
    ProtoOATrader,
    ProtoOAUnsubscribeSpotsReq,
    ProtoOAUnsubscribeSpotsRes,
)
from ..price import TickData
from ..schedule import ExecutionSchedule
from . import BaseTickDataGenerator, Tick, TickType


class LiveTickDataGenerator(BaseTickDataGenerator):
    def __init__(
        self,
        trader: ProtoOATrader,
        client: OpenApiClient,
        exec_schedule: ExecutionSchedule,
    ):
        super().__init__(trader, exec_schedule)
        self._client = client
        self._id_to_sym = {sym.id: sym for sym in self.symbols}

    @asynccontextmanager
    async def _spot_event_subscription(self):
        spot_sub_req = ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=self.trader.ctidTraderAccountId,
            symbolId=self._id_to_sym,
        )
        _ = await self._client.send_and_wait_response(
            spot_sub_req, ProtoOASubscribeSpotsRes
        )

        try:
            yield
        finally:
            spot_unsub_req = ProtoOAUnsubscribeSpotsReq(
                ctidTraderAccountId=self.trader.ctidTraderAccountId,
                symbolId=self._id_to_sym,
            )
            _ = await self._client.send_and_wait_response(
                spot_unsub_req, ProtoOAUnsubscribeSpotsRes
            )

    async def _generate_ticks(self) -> AsyncGenerator[TickData, None]:
        async with self._spot_event_subscription():
            async with self._client.register_types(ProtoOASpotEvent) as gen:
                async for event in gen:
                    t = time.time()
                    sym = self._id_to_sym[event.symbolId]

                    yield TickData(
                        sym, TickType.BID, Tick(event.bid / self.PRICE_CONV_FACTOR, t)
                    )
                    yield TickData(
                        sym, TickType.ASK, Tick(event.ask / self.PRICE_CONV_FACTOR, t)
                    )

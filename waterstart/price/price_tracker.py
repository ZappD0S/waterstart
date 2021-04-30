import asyncio
import datetime
import time
from collections.abc import AsyncIterator, Mapping, MutableSequence
from contextlib import asynccontextmanager
from enum import Enum
from typing import Final, Generic, MutableMapping, NamedTuple, Sequence, TypeVar, cast

from ..client import OpenApiClient
from ..observable import Observable
from ..openapi import (
    ProtoOASpotEvent,
    ProtoOASubscribeSpotsReq,
    ProtoOASubscribeSpotsRes,
    ProtoOATrader,
    ProtoOAUnsubscribeSpotsReq,
    ProtoOAUnsubscribeSpotsRes,
)
from ..schedule import ExecutionSchedule
from ..symbols import SymbolInfo, SymbolInfoWithConvChains
from .price_aggregation import BasePriceAggregator


class PriceSnapshot(NamedTuple):
    bid: float
    ask: float
    time: float


T = TypeVar("T", bound=Enum)


class PriceTracker(
    Generic[T], Observable[Mapping[SymbolInfoWithConvChains, Mapping[T, float]]]
):
    PRICE_CONV_FACTOR: Final[int] = 10 ** 5
    ONE_DAY: Final[datetime.timedelta] = datetime.timedelta(days=1)

    def __init__(
        self,
        client: OpenApiClient,
        trader: ProtoOATrader,
        exec_schedule: ExecutionSchedule,
        price_aggregator: BasePriceAggregator[T],
    ) -> None:
        # TODO: make maxsize bigger than 1 for safety?
        super().__init__()
        self._client = client
        self._trader = trader
        self._traded_symbols = list(exec_schedule.traded_symbols)
        self._id_to_sym_info = {sym.id: sym for sym in self._traded_symbols}
        self._exec_schedule = exec_schedule
        self._price_aggregator = price_aggregator
        # TODO: maybe make a method for this
        self._data_map: MutableMapping[int, MutableSequence[PriceSnapshot]] = {
            sym.id: []
            for traded_sym in exec_schedule.traded_symbols
            for sym in [cast(SymbolInfo, traded_sym)]
            + [
                sym
                for chain in (
                    traded_sym.conv_chains.base_asset,
                    traded_sym.conv_chains.quote_asset,
                )
                for sym in chain
            ]
        }
        self._data_lock = asyncio.Lock()

    @classmethod
    def _compute_time_of_day(cls, dt: datetime.datetime) -> float:
        time_of_day = (
            datetime.datetime.combine(datetime.date.min, dt.time())
            - datetime.datetime.min
        )
        return time_of_day / cls.ONE_DAY

    async def _get_async_iterator(
        self,
    ) -> AsyncIterator[Mapping[SymbolInfoWithConvChains, Mapping[T, float]]]:
        now = datetime.datetime.now()
        next_trading_time = self._exec_schedule.next_valid_time(now)
        last_trading_time = self._exec_schedule.last_valid_time(now)

        while True:
            await asyncio.sleep((next_trading_time - now).total_seconds())

            time_of_day = self._compute_time_of_day(next_trading_time)
            delta_to_last = (next_trading_time - last_trading_time) / self.ONE_DAY

            data_map: dict[SymbolInfo, Sequence[PriceSnapshot]] = {}
            skip: bool = False

            for sym_id in self._data_map:
                async with self._data_lock:
                    sym_data = self._data_map[sym_id]
                    if not sym_data:
                        skip = True
                        break

                    self._data_map[sym_id] = [sym_data[-1]]

                sym = self._id_to_sym_info[sym_id]
                data_map[sym] = sym_data

            if skip:
                continue

            yield self._price_aggregator.aggregate(
                data_map, self._traded_symbols, time_of_day, delta_to_last
            )

            last_trading_time = next_trading_time
            now = datetime.datetime.now()
            next_trading_time = self._exec_schedule.next_valid_time(now)

    @asynccontextmanager
    async def _spot_event_subscription(self):
        spot_sub_req = ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=self._trader.ctidTraderAccountId,
            symbolId=self._id_to_sym_info,
        )
        _ = await self._client.send_and_wait_response(
            spot_sub_req, ProtoOASubscribeSpotsRes
        )

        # ProtoOASubscribeLiveTrendbarReq(
        #     ctidTraderAccountId=self._trader.ctidTraderAccountId,
        #     period=M30,
        #     symbolId=...
        # )

        try:
            yield
        finally:
            spot_unsub_req = ProtoOAUnsubscribeSpotsReq(
                ctidTraderAccountId=self._trader.ctidTraderAccountId,
                symbolId=[sym.id for sym in self._exec_schedule.traded_symbols],
            )
            _ = await self._client.send_and_wait_response(
                spot_unsub_req, ProtoOAUnsubscribeSpotsRes
            )

    async def _track_prices(self) -> None:
        async with self._spot_event_subscription():
            async with self._client.register_types(ProtoOASpotEvent) as gen:
                async for event in gen:
                    t = time.time()
                    bid = event.bid / self.PRICE_CONV_FACTOR
                    ask = event.ask / self.PRICE_CONV_FACTOR
                    snapshot = PriceSnapshot(bid=bid, ask=ask, time=t)

                    async with self._data_lock:
                        sym_data = self._data_map[event.symbolId]

                    sym_data.append(snapshot)

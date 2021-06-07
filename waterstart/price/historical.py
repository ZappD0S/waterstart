import asyncio
import datetime
import time
from collections import AsyncGenerator
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import IntEnum
from math import log2
from typing import Final, Iterable, Iterator, Mapping, Sequence, Union

from ..client.app import AppClient
from ..openapi import (
    ASK,
    BID,
    ProtoOAErrorRes,
    ProtoOAGetTickDataReq,
    ProtoOAGetTickDataRes,
    ProtoOAQuoteType,
    ProtoOATickData,
    ProtoOATrader,
)
from ..price import TickData
from ..schedule import ExecutionSchedule
from ..symbols import SymbolInfo
from . import BaseTickDataGenerator, Tick, TickType


class SizeType(IntEnum):
    TooLong = -1
    TooShort = 1


class Direction(IntEnum):
    Left = -1
    Right = 1


@dataclass
class State:
    step: int
    ref_pos: int
    latest_done: int
    size_type: SizeType
    narrowing: bool = False
    same_side_count: int = 0


# TODO: create aliases the long types


class HistoricalTickDataGenerator(BaseTickDataGenerator):
    REQUESTS_PER_INTERVAL: Final[int] = 5
    INTERVAL: Final[float] = 1.0  # in seconds
    MAX_SAME_SIDE_SEARCHES: Final[int] = 4
    TICK_TYPE_MAP: Final[Mapping[TickType, ProtoOAQuoteType.V]] = {
        TickType.BID: BID,
        TickType.ASK: ASK,
    }

    def __init__(
        self,
        trader: ProtoOATrader,
        client: AppClient,
        exec_schedule: ExecutionSchedule,
        start_time: datetime.datetime,
        n_intervals: int,
    ):
        super().__init__(trader, exec_schedule)
        self._client = client
        # self._trader = trader
        self._first_trading_time, self._last_trading_time = self._compute_time_range(
            start_time, n_intervals
        )
        self._last_req_time = 0.0
        self._req_count = 0

    def _compute_time_range(
        self, start_time: datetime.datetime, n_intervals: int
    ) -> tuple[datetime.datetime, datetime.datetime]:
        last_trading_time = self.exec_schedule.last_valid_time(start_time)
        first_trading_time = last_trading_time
        for _ in range(n_intervals - 1):
            first_trading_time -= self.exec_schedule.trading_interval
            first_trading_time = self.exec_schedule.last_valid_time(first_trading_time)

        return first_trading_time, last_trading_time

    @classmethod
    def _convert_to_tick_data(
        cls, ticks: Sequence[ProtoOATickData], sym: SymbolInfo, tick_type: TickType
    ) -> Iterator[TickData]:
        def iterate_tick_data() -> Iterator[TickData]:
            price = 0
            timestamp = 0
            for tick in ticks:
                price += tick.tick
                timestamp += tick.timestamp
                yield TickData(
                    sym,
                    tick_type,
                    Tick(timestamp / 1000, price / cls.PRICE_CONV_FACTOR),
                )

        return reversed(list(iterate_tick_data()))

    async def _download_chunk(
        self,
        gen: AsyncIterator[Union[ProtoOAGetTickDataRes, ProtoOAErrorRes]],
        sym: SymbolInfo,
        tick_type: TickType,
        chunk_start: int,
        chunk_end: int,
    ) -> ProtoOAGetTickDataRes:
        if self._req_count == self.REQUESTS_PER_INTERVAL:
            now = time.time()
            await asyncio.sleep(self._last_req_time + self.INTERVAL - now)
            self._req_count = 0
            self._last_req_time = now

        res = await self._client.send_request_using_gen(
            ProtoOAGetTickDataReq(
                ctidTraderAccountId=self.trader.ctidTraderAccountId,
                symbolId=sym.id,
                type=self.TICK_TYPE_MAP[tick_type],
                fromTimestamp=chunk_start,
                toTimestamp=chunk_end,
            ),
            gen,
        )
        self._req_count += 1

        return res

    async def _download_initial(
        self,
        queue: asyncio.Queue[Iterable[TickData]],
        gen: AsyncIterator[Union[ProtoOAGetTickDataRes, ProtoOAErrorRes]],
        start: int,
        end: int,
        resolution: int,
    ) -> dict[SymbolInfo, State]:
        sym_to_state: dict[SymbolInfo, State] = {}
        has_more = False

        for sym in self.symbols:
            latest_donwloaded = 0
            has_more = False

            for tick_type in TickType:
                chunk = await self._download_chunk(gen, sym, tick_type, start, end)

                await queue.put(
                    self._convert_to_tick_data(chunk.tickData, sym, tick_type)
                )

                # NOTE: we take the first element because the list is inverted..
                latest_donwloaded = max(latest_donwloaded, chunk.tickData[0].timestamp)
                has_more |= chunk.hasMore

            sym_to_state[sym] = State(
                resolution,
                resolution,
                latest_donwloaded if has_more else end,
                SizeType.TooLong if has_more else SizeType.TooShort,
            )

        return sym_to_state

    async def _download_tick_data(
        self,
        queue: asyncio.Queue[Iterable[TickData]],
        gen: AsyncIterator[Union[ProtoOAGetTickDataRes, ProtoOAErrorRes]],
    ) -> None:
        start_ms = int(self._first_trading_time.timestamp() * 1000)
        end_ms = int(self._last_trading_time.timestamp() * 1000)
        period_ms = end_ms - start_ms
        resolution = int(log2(period_ms))

        sym_to_state = await self._download_initial(
            queue, gen, start_ms, end_ms, resolution
        )

        while True:
            done = True

            for sym, state in sym_to_state.items():
                latest_done = state.latest_done

                if latest_done >= end_ms:
                    continue

                same_side_count = state.same_side_count
                assert not state.narrowing or same_side_count == 0

                reached_max_same_side = same_side_count == self.MAX_SAME_SIDE_SEARCHES

                size_type = state.size_type
                factor = 2 ** (-1 if state.narrowing else size_type)

                # TODO: check bounds
                step = int(state.step * factor)
                search_pos = state.ref_pos + (
                    0 if reached_max_same_side else size_type * step
                )

                chunk_size = int(period_ms * search_pos / resolution)
                chunk_start = latest_done
                chunk_end = latest_done + chunk_size

                if chunk_end < end_ms:
                    done = False
                else:
                    chunk_end = end_ms

                latest_chunk_timestamp = 0
                has_more = False

                for tick_type in TickType:
                    res = await self._download_chunk(
                        gen, sym, tick_type, chunk_start, chunk_end
                    )
                    await queue.put(
                        self._convert_to_tick_data(res.tickData, sym, tick_type)
                    )
                    latest_chunk_timestamp = max(
                        latest_chunk_timestamp, res.tickData[0].timestamp
                    )
                    has_more |= res.hasMore

                new_size_type = SizeType.TooLong if has_more else SizeType.TooShort
                # TODO: is this always right?
                state.step = step

                if state.narrowing:
                    if new_size_type == size_type:
                        state.ref_pos = search_pos
                    else:
                        if reached_max_same_side:
                            state.narrowing = False
                            state.same_side_count = 0
                            state.size_type = new_size_type
                        else:
                            state.same_side_count += 1
                else:
                    if new_size_type == size_type:
                        state.ref_pos = search_pos
                    else:
                        state.narrowing = True

            if done:
                break

    async def generate_ticks(self) -> AsyncGenerator[TickData, None]:
        async with self._client.register_type(ProtoOAGetTickDataRes) as gen:
            queue: asyncio.Queue[Iterable[TickData]] = asyncio.Queue()
            download_task = asyncio.create_task(self._download_tick_data(queue, gen))

            try:
                while not (download_task.done() and queue.empty()):
                    for tick in await queue.get():
                        yield tick
            finally:
                download_task.cancel()

                try:
                    await download_task
                except asyncio.CancelledError:
                    pass

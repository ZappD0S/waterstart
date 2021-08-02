import asyncio
import time
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence, Set
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import ClassVar, Final, Optional, Union

from ..client.trader import TraderClient
from ..openapi import (
    ASK,
    BID,
    ProtoOAErrorRes,
    ProtoOAGetTickDataReq,
    ProtoOAGetTickDataRes,
    ProtoOAQuoteType,
    ProtoOATickData,
)
from ..price import TickData
from ..symbols import TradedSymbolInfo
from . import Tick, TickType
from .tick_producer import BaseTicksProducer, BaseTicksProducerFactory
from ..utils import ComposableAsyncIterator


class SizeType(IntEnum):
    TooLong = -1
    TooShort = 1


class State:
    MAX_SAME_SIDE_SEARCHES: ClassVar[int] = 4

    def __init__(self, start_ms: int) -> None:
        self._latest_done: int = start_ms
        self._ref_chunk: int = 0
        self._step: int = 1
        self._size_type: SizeType = SizeType.TooShort
        self._narrowing: bool = False
        self._same_side_count: int = 0
        self._probe_chunk: Optional[int] = None

    @property
    def latest_done(self) -> int:
        return self._latest_done

    def next_chunk_end(self) -> int:
        if (probe_chunk := self._probe_chunk) is not None:
            return self._latest_done + probe_chunk

        narrowing = self._narrowing
        size_type = self._size_type

        exp = -1 if narrowing else 1

        # TODO: if one of the two ifs below is true,
        # we have to skip the update too, right?
        if self._step == 1 and exp == -1:
            exp = 0

        # abs_step = self._step = int(self._step * 2 ** exp)
        abs_step = int(self._step * 2 ** exp)
        # step = size_type * abs_step

        probe_chunk = self._ref_chunk + size_type * abs_step

        # TODO: loop while probe_chunk < 1 or (self._step == 1 and exp == -1)

        if probe_chunk < 1:
            probe_chunk = 1
            # what about step?

        # if self._ref_chunk == 1 and step < 0:
        #     step = 0

        self._probe_chunk = probe_chunk  # = self._ref_chunk + step
        return self._latest_done + probe_chunk

    def update(self, has_more: Optional[bool], latest_downloaded: int) -> None:
        if (probe_chunk := self._probe_chunk) is None:
            raise RuntimeError()

        if has_more is None:
            self._latest_done = latest_downloaded
            return

        self._probe_chunk = None
        self._ref_chunk = probe_chunk

        if has_more:
            new_size_type = SizeType.TooLong
            self._latest_done = latest_downloaded
        else:
            new_size_type = SizeType.TooShort
            self._latest_done += probe_chunk

        if new_size_type == self._size_type:
            if self._narrowing:
                if self._same_side_count == self.MAX_SAME_SIDE_SEARCHES:
                    self._narrowing = False
                    self._same_side_count = 0
                else:
                    self._same_side_count += 1
        else:
            self._size_type = new_size_type
            if not self._narrowing:
                self._narrowing = True

        assert self._narrowing or self._same_side_count == 0


class HistoricalTicksProducer(BaseTicksProducer):
    MAX_REQUESTS_PER_SECOND: Final[int] = 5
    TICK_TYPE_MAP: Final[Mapping[TickType, ProtoOAQuoteType.V]] = {
        TickType.BID: BID,
        TickType.ASK: ASK,
    }

    def __init__(
        self,
        client: TraderClient,
        gen: AsyncIterator[Union[ProtoOAGetTickDataRes, ProtoOAErrorRes]],
        sym_ids: Set[int],
        start: float,
    ):
        super().__init__()
        self._client = client
        self._gen = gen
        self._last_req_time = 0.0
        self._req_count = 0
        self._start = start
        start_ms = int(start * 1000)
        self._sym_id_to_state = {sym_id: State(start_ms) for sym_id in sym_ids}

    @classmethod
    def _convert_to_tick_data(
        cls, ticks: Sequence[ProtoOATickData], sym_id: int, tick_type: TickType
    ) -> Iterator[TickData]:
        PRICE_CONV_FACTOR = cls.PRICE_CONV_FACTOR

        def iterate_tick_data() -> Iterator[TickData]:
            price = 0
            timestamp = 0
            for tick in ticks:
                price += tick.tick
                timestamp += tick.timestamp
                yield TickData(
                    sym_id,
                    tick_type,
                    Tick(timestamp / 1000, price / PRICE_CONV_FACTOR),
                )

        return reversed(list(iterate_tick_data()))

    async def _download_chunk(
        self,
        gen: AsyncIterator[Union[ProtoOAGetTickDataRes, ProtoOAErrorRes]],
        sym_id: int,
        tick_type: TickType,
        chunk_start: int,
        chunk_end: int,
    ) -> ProtoOAGetTickDataRes:
        now = time.time()

        if self._req_count == self.MAX_REQUESTS_PER_SECOND:
            await asyncio.sleep(self._last_req_time + 1.0 - now)
            self._req_count = 0

        self._last_req_time = now
        self._req_count += 1

        return await self._client.send_request_from_trader(
            lambda trader_id: ProtoOAGetTickDataReq(
                ctidTraderAccountId=trader_id,
                symbolId=sym_id,
                type=self.TICK_TYPE_MAP[tick_type],
                fromTimestamp=chunk_start,
                toTimestamp=chunk_end,
            ),
            ProtoOAGetTickDataRes,
            gen,
        )

    async def generate_ticks_up_to(self, end: float) -> AsyncIterator[TickData]:
        now = time.time()
        await asyncio.sleep(self._start - now)

        end_ms = int(end * 1000)
        tick_types = tuple(TickType)

        while True:
            done = True

            for sym_id, state in self._sym_id_to_state.items():
                latest_done = state.latest_done

                if latest_done >= end_ms:
                    continue

                chunk_start = latest_done
                chunk_end = state.next_chunk_end()

                if chunk_end < end_ms:
                    done = False
                    skip_update = False
                else:
                    chunk_end = end_ms
                    skip_update = True

                latest_downloaded = 0
                has_more = False

                for tick_type in tick_types:
                    res = await self._download_chunk(
                        self._gen, sym_id, tick_type, chunk_start, chunk_end
                    )
                    ticks = res.tickData

                    if ticks:
                        for tick_data in self._convert_to_tick_data(
                            ticks, sym_id, tick_type
                        ):
                            yield tick_data
                    else:
                        skip_update = True

                    latest_downloaded = max(
                        latest_downloaded, res.tickData[0].timestamp
                    )
                    has_more |= res.hasMore

                state.update(None if skip_update else has_more, latest_downloaded)

            if done:
                break


class HistoricalTicksProducerFactory(BaseTicksProducerFactory):
    def __init__(
        self, client: TraderClient, traded_symbols: Collection[TradedSymbolInfo]
    ):
        super().__init__(traded_symbols)
        self._client = client

    @asynccontextmanager
    async def get_ticks_gen(self, start: float) -> AsyncIterator[BaseTicksProducer]:
        res_gen_cm = self._client.register_type(ProtoOAGetTickDataRes)
        err_gen_cm = self._client.register_type(ProtoOAErrorRes)

        async with res_gen_cm as res_gen, err_gen_cm as err_gen:
            gen = ComposableAsyncIterator.from_it(res_gen) | err_gen
            yield HistoricalTicksProducer(
                self._client, gen, self._id_to_sym.keys(), start
            )

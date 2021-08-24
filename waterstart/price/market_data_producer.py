import asyncio
import datetime
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from typing import Awaitable, Final

from ..client.trader import TraderClient
from ..datetime_utils import delta_to_midnight
from ..schedule import ExecutionSchedule
from . import AggregationData, BidAskTicks, MarketData, SymbolData
from .historical import HistoricalTicksProducerFactory
from .price_aggregation import PriceAggregator
from .tick_producer import (
    BaseTicksProducer,
    BaseTicksProducerFactory,
    LiveTicksProducerFactory,
)


class BaseMarketDataProducer(ABC):
    MAX_TICKS_LEN: int = 1000
    ONE_DAY: Final[datetime.timedelta] = datetime.timedelta(days=1)

    # TODO: maybe take the pool from the outside?
    def __init__(self, tick_producer_factory: BaseTicksProducerFactory):
        self._tick_producer_factory = tick_producer_factory
        self._symbols = tick_producer_factory.symbols
        self._traded_symbols = tick_producer_factory.traded_symbols
        self._aggregator = PriceAggregator(self._traded_symbols, self._symbols)

    @abstractmethod
    def _iterate_trading_times(self) -> Iterator[datetime.datetime]:
        ...

    async def generate_market_data(self) -> AsyncIterator[MarketData[float]]:
        trading_times_it = self._iterate_trading_times()
        last_trading_time = next(trading_times_it)

        async with self._tick_producer_factory.get_ticks_gen(
            last_trading_time.timestamp()
        ) as tick_producer:
            async for market_data in self._generate_market_data(
                trading_times_it, tick_producer, last_trading_time
            ):
                yield market_data

    async def _generate_market_data(
        self,
        trading_times: Iterable[datetime.datetime],
        tick_producer: BaseTicksProducer,
        last_trading_time: datetime.datetime,
    ) -> AsyncIterator[MarketData[float]]:
        symbols = self._symbols
        aggregator = self._aggregator
        MAX_TICKS_LEN = self.MAX_TICKS_LEN
        ONE_DAY = self.ONE_DAY

        default_tb_data_map = {
            sym.id: SymbolData.build_default() for sym in self._traded_symbols
        }

        async def get_default_aggreg_data(
            bid_ask_ticks_map: Mapping[int, BidAskTicks]
        ) -> AggregationData:
            return AggregationData(bid_ask_ticks_map, default_tb_data_map)

        loop = asyncio.get_running_loop()

        for next_trading_time in trading_times:
            bid_ask_ticks_map = {sym.id: BidAskTicks([], []) for sym in symbols}
            aggreg_task: Awaitable[AggregationData]
            aggreg_task = get_default_aggreg_data(bid_ask_ticks_map)

            async for tick_data in tick_producer.generate_ticks_up_to(
                next_trading_time.timestamp()
            ):
                ask_bid_ticks = bid_ask_ticks_map[tick_data.sym_id]
                ticks = ask_bid_ticks[tick_data.type]
                ticks.append(tick_data.tick)

                if len(ticks) < MAX_TICKS_LEN:
                    continue

                try:
                    aggreg_data = await asyncio.wait_for(asyncio.shield(aggreg_task), 0)
                except asyncio.TimeoutError:
                    continue

                aggreg_task = loop.run_in_executor(
                    None,
                    aggregator.aggregate,
                    AggregationData(bid_ask_ticks_map, aggreg_data.tb_data_map),
                )

                leftofver_bid_ask_ticks_map = aggreg_data.bid_ask_ticks_map
                bid_ask_ticks_map = {
                    sym_id: leftofver_bid_ask_ticks_map[sym_id] + ask_bid_ticks
                    for sym_id, ask_bid_ticks in bid_ask_ticks_map.items()
                }

            if not all(
                ticks
                for ask_bid_ticks in bid_ask_ticks_map.values()
                for ticks in ask_bid_ticks
            ):
                raise RuntimeError()

            aggreg_data = await aggreg_task
            aggreg_data = aggregator.aggregate(
                AggregationData(bid_ask_ticks_map, aggreg_data.tb_data_map)
            )
            time_of_day = delta_to_midnight(next_trading_time) / ONE_DAY
            delta_to_last = (next_trading_time - last_trading_time) / ONE_DAY
            yield MarketData(aggreg_data.tb_data_map, time_of_day, delta_to_last)

            last_trading_time = next_trading_time


class LiveMarketDataProducer(BaseMarketDataProducer):
    def __init__(
        self,
        client: TraderClient,
        schedule: ExecutionSchedule,
        start: datetime.datetime,
    ):
        super().__init__(LiveTicksProducerFactory(schedule.traded_symbols, client))
        self._schedule = schedule
        self._start = start

    def _iterate_trading_times(self) -> Iterator[datetime.datetime]:
        schedule = self._schedule
        res = datetime.timedelta.resolution
        next_trading_time = schedule.next_valid_time(self._start)

        while True:
            yield next_trading_time
            next_trading_time = schedule.next_valid_time(next_trading_time + res)


class HistoricalMarketDataProducer(BaseMarketDataProducer):
    def __init__(
        self,
        client: TraderClient,
        schedule: ExecutionSchedule,
        start: datetime.datetime,
        n_intervals: int,
    ):
        super().__init__(
            HistoricalTicksProducerFactory(client, schedule.traded_symbols)
        )
        self._schedule = schedule
        self._start = start
        self._n_intervals = n_intervals

    def _iterate_trading_times(self) -> Iterator[datetime.datetime]:
        schedule = self._schedule
        last_trading_time = schedule.next_valid_time(self._start)
        trading_times = [last_trading_time]
        res = datetime.timedelta.resolution

        for _ in range(self._n_intervals):
            last_trading_time = schedule.last_invalid_time(last_trading_time - res)
            trading_times.append(last_trading_time)

        yield from reversed(trading_times)

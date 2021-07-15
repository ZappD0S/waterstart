import datetime
from abc import ABC, abstractmethod
from collections import AsyncIterator, Iterator
from typing import Final
from waterstart.client.trader import TraderClient

from ..datetime_utils import delta_to_midnight
from ..schedule import ExecutionSchedule
from . import AggregationData, BidAskTicks, MarketData, SymbolData
from .price_aggregation import PriceAggregator
from .tick_producer import BaseTicksProducerFactory, LiveTicksProducerFactory
from .historical import HistoricalTicksProducerFactory


class BaseMarketDataProducer(ABC):
    MAX_TICKS_LEN: int = 1000
    ONE_DAY: Final[datetime.timedelta] = datetime.timedelta(days=1)

    def __init__(self, tick_producer_factory: BaseTicksProducerFactory):
        self._tick_producer_factory = tick_producer_factory
        self._symbols = tick_producer_factory.symbols
        self._traded_symbols = tick_producer_factory.traded_symbols
        self._aggregator = PriceAggregator(self._traded_symbols, self._symbols)

    @abstractmethod
    def iterate_trading_times(self) -> Iterator[datetime.datetime]:
        ...

    async def generate_market_data(self) -> AsyncIterator[MarketData[float]]:
        trading_times_it = self.iterate_trading_times()
        last_trading_time = next(trading_times_it)

        ask_bid_ticks_map = {sym.id: BidAskTicks([], []) for sym in self._symbols}
        tb_data_map = {
            sym.id: SymbolData.build_default() for sym in self._traded_symbols
        }
        aggreg_data = AggregationData(ask_bid_ticks_map, tb_data_map)

        async with self._tick_producer_factory.get_ticks_gen_starting_from(
            last_trading_time.timestamp()
        ) as tick_producer:
            for next_trading_time in trading_times_it:
                async for tick_data in tick_producer.generate_ticks_up_to(
                    next_trading_time.timestamp()
                ):
                    ask_bid_ticks = ask_bid_ticks_map[tick_data.sym_id]
                    ticks = ask_bid_ticks[tick_data.type]
                    ticks.append(tick_data.tick)

                    if len(ticks) == self.MAX_TICKS_LEN:
                        aggreg_data = self._aggregator.aggregate(aggreg_data)

                aggreg_data = self._aggregator.aggregate(aggreg_data)
                time_of_day = delta_to_midnight(next_trading_time) / self.ONE_DAY
                delta_to_last = (next_trading_time - last_trading_time) / self.ONE_DAY
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

    def iterate_trading_times(self) -> Iterator[datetime.datetime]:
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

    def iterate_trading_times(self) -> Iterator[datetime.datetime]:
        schedule = self._schedule
        last_trading_time = schedule.next_valid_time(self._start)
        trading_times = [last_trading_time]
        res = datetime.timedelta.resolution

        for _ in range(self._n_intervals):
            last_trading_time = schedule.last_invalid_time(last_trading_time + res)
            trading_times.append(last_trading_time)

        yield from reversed(trading_times)

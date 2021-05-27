import asyncio
import datetime
from collections.abc import AsyncIterator, Mapping
from typing import Final, Optional

from ..client import OpenApiClient
from ..observable import Observable
from ..symbols import SymbolInfo, TradedSymbolInfo
from . import (
    AggregationData,
    BaseTickDataGenerator,
    BidAskTicks,
    LatestMarketData,
    MarketFeatures,
    SymbolData,
)
from .price_aggregation import PriceAggregator

MAX_LEN = 1000


# TODO: rename
class LiveMarketTracker(Observable[LatestMarketData]):
    PRICE_CONV_FACTOR: Final[int] = 10 ** 5
    ONE_DAY: Final[datetime.timedelta] = datetime.timedelta(days=1)

    def __init__(
        self,
        client: OpenApiClient,
        tick_data_gen: BaseTickDataGenerator,
        price_aggregator: PriceAggregator,
    ) -> None:
        # TODO: make maxsize bigger than 1 for safety?
        super().__init__()
        self._client = client
        self._ask_bid_ticks_map: Mapping[SymbolInfo, BidAskTicks] = {
            sym: BidAskTicks([], []) for sym in tick_data_gen.symbols
        }
        self._tb_data_map: Optional[Mapping[TradedSymbolInfo, SymbolData[float]]] = None

        self._default_data_map: Mapping[TradedSymbolInfo, SymbolData[float]] = {
            sym: SymbolData.build_default() for sym in tick_data_gen.traded_symbols
        }

        self._tick_data_gen = tick_data_gen
        self._exec_schedule = tick_data_gen.exec_schedule
        self._price_aggregator = price_aggregator
        self._data_lock = asyncio.Lock()

    @staticmethod
    def _compute_time_of_day(dt: datetime.datetime) -> datetime.timedelta:
        return (
            datetime.datetime.combine(datetime.date.min, dt.time())
            - datetime.datetime.min
        )

    @staticmethod
    def _build_latest_market_data(
        tb_data_map: Mapping[TradedSymbolInfo, SymbolData[float]],
        time_of_day: float,
        delta_to_last: float,
    ) -> LatestMarketData:
        market_feat = MarketFeatures(tb_data_map, time_of_day, delta_to_last)
        sym_prices: dict[TradedSymbolInfo, float] = {}
        margin_rates: dict[TradedSymbolInfo, float] = {}

        for sym, sym_data in tb_data_map.items():
            sym_prices[sym] = sym_data.price_trendbar.close
            margin_rates[sym] = sym_data.dep_to_base_trendbar.close

        return LatestMarketData(market_feat, sym_prices, margin_rates)

    async def _generate_market_data(
        self, start_time: datetime.datetime
    ) -> AsyncIterator[LatestMarketData]:
        next_trading_time = self._exec_schedule.next_valid_time(start_time)
        last_trading_time = self._exec_schedule.last_valid_time(
            next_trading_time - self._exec_schedule.trading_interval
        )

        while True:
            tb_data_map = await self._get_tb_data_map(next_trading_time)

            time_of_day = self._compute_time_of_day(next_trading_time) / self.ONE_DAY
            delta_to_last = (next_trading_time - last_trading_time) / self.ONE_DAY
            if tb_data_map is not None:
                yield self._build_latest_market_data(
                    tb_data_map, time_of_day, delta_to_last
                )

            last_trading_time = next_trading_time
            next_trading_time = self._exec_schedule.next_valid_time(
                last_trading_time + self._exec_schedule.trading_interval
            )

    async def _get_async_iterator(self) -> AsyncIterator[LatestMarketData]:
        now = datetime.datetime.now()
        async for market_data in self._generate_market_data(now):
            yield market_data

    async def _get_tb_data_map(
        self, next_trading_time: datetime.datetime
    ) -> Optional[Mapping[TradedSymbolInfo, SymbolData[float]]]:
        now = datetime.datetime.now()
        await asyncio.sleep((next_trading_time - now).total_seconds())
        await self._update_symbols_data()
        tb_data_map = self._tb_data_map
        self._tb_data_map = None
        return tb_data_map

    async def _update_symbols_data(self) -> None:
        async with self._data_lock:
            tb_data_map = self._tb_data_map or self._default_data_map
            aggreg_data = AggregationData(self._ask_bid_ticks_map, tb_data_map)

            try:
                aggreg_data = self._price_aggregator.aggregate(aggreg_data)
            except ValueError:
                # TODO: is this ok?
                return

            self._tb_data_map = aggreg_data.tb_data_map
            self._ask_bid_ticks_map = aggreg_data.tick_data_map

    async def _track_prices(self) -> None:
        ticks_gen = self._tick_data_gen.generate_ticks()
        try:
            async for tick_data in ticks_gen:
                async with self._data_lock:
                    ask_bid_ticks = self._ask_bid_ticks_map[tick_data.sym]
                    ticks = ask_bid_ticks[tick_data.type]
                    ticks.append(tick_data.tick)
                    ticks_len = len(ticks)

                # TODO: take this from  _price_aggregator
                if ticks_len > MAX_LEN:
                    await self._update_symbols_data()
        finally:
            await ticks_gen.aclose()

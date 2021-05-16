import asyncio
import datetime
import time
from collections.abc import AsyncIterator, MutableSequence
from contextlib import asynccontextmanager
from typing import Final, Iterator, Mapping, Sequence

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
from ..symbols import SymbolInfo, TradedSymbolInfo

from . import (
    LatestMarketData,
    MarketFeatures,
    PriceSnapshot,
    SymbolData,
    AggregationData,
)
from .price_aggregation import PriceAggregator


MAX_LEN = 1000


class PriceTracker(Observable[LatestMarketData]):
    PRICE_CONV_FACTOR: Final[int] = 10 ** 5
    ONE_DAY: Final[datetime.timedelta] = datetime.timedelta(days=1)

    def __init__(
        self,
        client: OpenApiClient,
        trader: ProtoOATrader,
        exec_schedule: ExecutionSchedule,
        price_aggregator: PriceAggregator,
    ) -> None:
        # TODO: make maxsize bigger than 1 for safety?
        super().__init__()
        self._client = client
        self._trader = trader
        self._traded_symbols = list(exec_schedule.traded_symbols)
        self._symbols = list(set(self._get_all_symbols(self._traded_symbols)))
        self._raw_data_map: Mapping[SymbolInfo, MutableSequence[PriceSnapshot]] = {
            sym: [] for sym in self._symbols
        }
        self._tb_data_map: Mapping[TradedSymbolInfo, SymbolData[float]] = {
            sym: SymbolData.build_default() for sym in self._traded_symbols
        }

        self._id_to_sym = {sym.id: sym for sym in self._symbols}
        self._exec_schedule = exec_schedule
        self._price_aggregator = price_aggregator
        self._data_lock = asyncio.Lock()

    @classmethod
    def _compute_time_of_day(cls, dt: datetime.datetime) -> float:
        time_of_day = (
            datetime.datetime.combine(datetime.date.min, dt.time())
            - datetime.datetime.min
        )
        return time_of_day / cls.ONE_DAY

    @staticmethod
    def _get_all_symbols(
        traded_symbols: Sequence[TradedSymbolInfo],
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

    async def _get_async_iterator(self) -> AsyncIterator[LatestMarketData]:
        now = datetime.datetime.now()
        next_trading_time = self._exec_schedule.next_valid_time(now)
        last_trading_time = self._exec_schedule.last_valid_time(now)

        while True:
            await asyncio.sleep((next_trading_time - now).total_seconds())

            time_of_day = self._compute_time_of_day(next_trading_time)
            delta_to_last = (next_trading_time - last_trading_time) / self.ONE_DAY

            await self._update_symbols_data()

            market_feat = MarketFeatures(self._tb_data_map, time_of_day, delta_to_last)
            sym_prices: dict[TradedSymbolInfo, float] = {}
            margin_rates: dict[TradedSymbolInfo, float] = {}

            for sym, sym_data in self._tb_data_map.items():
                sym_prices[sym] = sym_data.price_trendbar.close
                margin_rates[sym] = sym_data.dep_to_base_trendbar.close

            yield LatestMarketData(market_feat, sym_prices, margin_rates)

            last_trading_time = next_trading_time
            now = datetime.datetime.now()
            next_trading_time = self._exec_schedule.next_valid_time(now)

    async def _update_symbols_data(self) -> None:
        async with self._data_lock:
            aggreg_data = AggregationData(self._raw_data_map, self._tb_data_map)

            # TODO: what do we do if this fails?
            aggreg_data = self._price_aggregator.aggregate(aggreg_data)

            self._tb_data_map = aggreg_data.tb_data_map
            self._raw_data_map = {
                sym: list(data) for sym, data in aggreg_data.raw_data_map.items()
            }

    @asynccontextmanager
    async def _spot_event_subscription(self):
        spot_sub_req = ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=self._trader.ctidTraderAccountId,
            symbolId=self._id_to_sym,
        )
        # TODO: capture also the proto error and check?
        _ = await self._client.send_and_wait_response(
            spot_sub_req, ProtoOASubscribeSpotsRes
        )

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

                    sym = self._id_to_sym[event.symbolId]

                    async with self._data_lock:
                        raw_data = self._raw_data_map[sym]
                        raw_data.append(snapshot)
                        raw_data_len = len(raw_data)

                    if raw_data_len > MAX_LEN:
                        await self._update_symbols_data()

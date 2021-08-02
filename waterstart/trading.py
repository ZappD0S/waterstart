import datetime
from collections.abc import Mapping
from math import copysign
from typing import AsyncIterator, Optional

from .client.trader import TraderClient

from .inference.engine import InferenceEngine
from .openapi import (
    BUY,
    MARKET,
    SELL,
    ProtoOAClosePositionReq,
    ProtoOAExecutionEvent,
    ProtoOANewOrderReq,
    ProtoOAPosition,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
    ProtoOATraderReq,
    ProtoOATraderRes,
)
from .price.market_data_producer import (
    HistoricalMarketDataProducer,
    LiveMarketDataProducer,
)
from .schedule import ExecutionSchedule


class Trader:
    TRADE_SIDE_TO_SIGN = {BUY: 1.0, SELL: -1.0}
    SIGN_TO_TRADE_SIDE = {v: k for k, v in TRADE_SIDE_TO_SIGN.items()}

    def __init__(
        self, engine: InferenceEngine, client: TraderClient, schedule: ExecutionSchedule
    ) -> None:
        net_modules = engine.net_modules
        self._n_traded_sym = net_modules.n_traded_sym
        self._window_size = window_size = net_modules.window_size
        self._max_trades = net_modules.max_trades
        self._market_features = net_modules.market_features

        self._client = client
        self._engine = engine

        # TODO: store the state to disk (or to /tmp) so that we can reload if the
        # app crashes. But we also have to check that it's still valid

        # TODO: check match engine.net_modules.id_to_traded_sym, schedule.traded_symbols

        start = datetime.datetime.now()
        self._historical_market_data_producer = HistoricalMarketDataProducer(
            client, schedule, start, window_size - 1
        )

        self._live_market_data_producer = LiveMarketDataProducer(
            client, schedule, start
        )

    async def _init_market_data_arr(self) -> None:
        historical_producer = self._historical_market_data_producer
        i = 0
        async for market_data in historical_producer.generate_market_data():
            self._engine.update_market_state(market_data)
            i += 1

        assert i == self._window_size - 1

    async def _get_positions_map(self) -> Mapping[int, ProtoOAPosition]:
        reconcile_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOAReconcileReq(ctidTraderAccountId=trader_id),
            ProtoOAReconcileRes,
        )

        positions = reconcile_res.position
        pos_map = {pos.tradeData.symbolId: pos for pos in positions}

        assert len(pos_map) == positions
        return pos_map

    async def _init_balance(self) -> None:
        trader_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOATraderReq(ctidTraderAccountId=trader_id),
            ProtoOATraderRes,
        )
        trader = trader_res.trader
        balance = trader.balance / 10 ** trader.moneyDigits
        self._engine.update_balance(balance)

    async def _init_account_state(self, pos_map: Mapping[int, ProtoOAPosition]) -> None:
        # TODO: use ProtoOADealListReq to retrieve the prices and sizes of
        # currently open trades
        # TODO: also check there are no pending order (or wait for the correspoding deals)

        size_price_map: dict[int, tuple[float, float]] = {}
        TRADE_SIDE_TO_SIGN = self.TRADE_SIDE_TO_SIGN
        for sym_id, pos in pos_map.items():
            trade_data = pos.tradeData
            sign = TRADE_SIDE_TO_SIGN[trade_data.tradeSide]
            size = sign * trade_data.volume / 100
            sym_id = trade_data.symbolId
            size_price_map[sym_id] = (size, pos.price)

        self._engine.update_trades(size_price_map)

    @staticmethod
    def _get_confim_order_res(res: ProtoOAExecutionEvent) -> Optional[int]:
        # if res.order.is
        ...

    async def trade(self) -> None:
        async with self._client.register_type(
            ProtoOAExecutionEvent, lambda res: self._get_confim_order_res(res) is None
        ) as exec_event_gen:
            ...

    async def _trade(
        self, exec_event_gen: AsyncIterator[ProtoOAExecutionEvent]
    ) -> None:

        SIGN_TO_TRADE_SIDE = self.SIGN_TO_TRADE_SIDE

        await self._init_market_data_arr()
        pos_map = await self._get_positions_map()
        await self._init_account_state(pos_map)
        await self._init_balance()

        # sym_to_pos_id = {sym_id: pos.positionId for sym_id, pos in pos_map.items()}

        live_producer = self._live_market_data_producer
        async for market_data in live_producer.generate_market_data():
            new_pos_sizes_map = self._engine.evaluate(market_data)

            self._client.send_requests_from_trader(
                lambda trader_id: {
                    sym_id: ProtoOANewOrderReq(
                        ctidTraderAccountId=trader_id,
                        symbolId=sym_id,
                        orderType=MARKET,
                        tradeSide=SIGN_TO_TRADE_SIDE[copysign(1, size)],
                        volume=int(size * 100),
                    )
                    if size != 0
                    else ProtoOAClosePositionReq(
                        ctidTraderAccountId=trader_id,
                        positionId=(pos := pos_map[sym_id]).positionId,
                        volume=pos.tradeData.volume,
                    )
                    for sym_id, size in new_pos_sizes_map.items()
                },
                ProtoOAExecutionEvent,
                self._get_confim_order_res,
            )
            ...
        # iterate live producer
        # transform market data object into array
        # shift market_data arr and insert new data
        # compute new_pos_sizes using the engine

        # execute trades in order to match new_pos_sizes
        # use either ProtoOANewOrderReq or ProtoOAClosePositionReq depending on the value

        # listen to ProtoOAExecutionEvent and ProtoOAOrderErrorEvent
        # until we have received something for each order executed. Also save
        # the open prices of the trades
        #  IMPORTANT: ProtoOAExecutionEvent also happens if we withdraw or deposit money. Skip those!

        # when done, update accout_state using engine
        # repeat
        ...

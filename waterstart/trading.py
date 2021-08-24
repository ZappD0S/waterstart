import datetime
from collections.abc import Mapping
from contextlib import AsyncExitStack
from math import copysign
from typing import Optional, Union
from waterstart.openapi.OpenApiModelMessages_pb2 import (
    NOT_ENOUGH_MONEY,
    ORDER_FILLED,
    ORDER_REJECTED,
)
from waterstart.price import MarketData

from .client.trader import TraderClient
from .inference.engine import InferenceEngine
from .openapi import (
    BUY,
    MARKET,
    ORDER_ACCEPTED,
    DEPOSIT_WITHDRAW,
    SELL,
    ProtoOAClosePositionReq,
    ProtoOAExecutionEvent,
    ProtoOANewOrderReq,
    ProtoOAOrderErrorEvent,
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
from .utils import ComposableAsyncIterable


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
        self._balance_version = 0

        # TODO: store the state to disk (or to /tmp) so that we can reload if the
        # app crashes. But we also have to check that it's still valid

        # TODO: check match engine.net_modules.id_to_traded_sym, schedule.traded_symbols

        start = datetime.datetime.now()
        self._historical_market_data_producer = HistoricalMarketDataProducer(
            client,
            schedule,
            start,
            window_size - 1,  # TODO: is this correct?
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

    async def _get_positions_map(self) -> dict[int, ProtoOAPosition]:
        reconcile_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOAReconcileReq(ctidTraderAccountId=trader_id),
            ProtoOAReconcileRes,
        )

        positions = reconcile_res.position
        pos_map = {pos.tradeData.symbolId: pos for pos in positions}
        assert len(pos_map) == len(positions)

        return pos_map

    async def _init_balance(self) -> None:
        trader_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOATraderReq(ctidTraderAccountId=trader_id),
            ProtoOATraderRes,
        )
        trader = trader_res.trader
        balance = trader.balance / 10 ** trader.moneyDigits
        self._engine.update_balance(balance)
        self._balance_version = trader.balanceVersion

    async def _init_account_state(self, pos_map: Mapping[int, ProtoOAPosition]) -> None:
        # TODO: use ProtoOADealListReq to retrieve the prices and sizes of
        # currently open trades
        # TODO: also check there are no pending order
        # (or wait for the correspoding deals)

        TRADE_SIDE_TO_SIGN = self.TRADE_SIDE_TO_SIGN
        for sym_id, pos in pos_map.items():
            trade_data = pos.tradeData
            sign = TRADE_SIDE_TO_SIGN[trade_data.tradeSide]
            size = sign * trade_data.volume / 100
            sym_id = trade_data.symbolId
            self._engine.update_symbol_state(sym_id, size, pos.price)

    async def _create_orders(
        self,
        new_pos_sizes_map: Mapping[int, float],
        pos_map: Mapping[int, ProtoOAPosition],
        exec_gen: ComposableAsyncIterable[
            Union[ProtoOAExecutionEvent, ProtoOAOrderErrorEvent]
        ],
    ) -> None:
        SIGN_TO_TRADE_SIDE = self.SIGN_TO_TRADE_SIDE

        def pred(
            res: Union[ProtoOAExecutionEvent, ProtoOAOrderErrorEvent]
        ) -> Optional[int]:
            if (
                isinstance(res, ProtoOAExecutionEvent)
                and res.executionType == ORDER_ACCEPTED
            ):
                return res.order.tradeData.symbolId
            else:
                return None

        def build_key_to_req(trader_id: int):
            def get_req(sym_id: int, size: float):
                if size == 0:
                    return ProtoOAClosePositionReq(
                        ctidTraderAccountId=trader_id,
                        positionId=(pos := pos_map[sym_id]).positionId,
                        volume=pos.tradeData.volume,
                    )

                return ProtoOANewOrderReq(
                    ctidTraderAccountId=trader_id,
                    symbolId=sym_id,
                    orderType=MARKET,
                    tradeSide=SIGN_TO_TRADE_SIDE[copysign(1, size)],
                    volume=int(size * 100),
                )

            return {
                sym_id: get_req(sym_id, size)
                for sym_id, size in new_pos_sizes_map.items()
            }

        async for _ in self._client.send_requests_from_trader(
            build_key_to_req, ProtoOAExecutionEvent, pred, exec_gen
        ):
            pass

    async def trade(self) -> None:
        await self._init_market_data_arr()
        pos_map = await self._get_positions_map()
        await self._init_account_state(pos_map)
        await self._init_balance()

        async with AsyncExitStack() as stack:
            res_gen = await stack.enter_async_context(
                self._client.register_type_for_trader(ProtoOAExecutionEvent)
            )
            order_err_gen = await stack.enter_async_context(
                self._client.register_type_for_trader(ProtoOAOrderErrorEvent)
            )
            market_data_gen = await stack.enter_async_context(
                ComposableAsyncIterable.from_it(
                    self._live_market_data_producer.generate_market_data()
                )
            )

            exec_gen = res_gen | order_err_gen

            async for event in exec_gen | market_data_gen:
                if isinstance(market_data := event, MarketData):
                    self._engine.update_market_state(market_data)

                    while not await self._execute(pos_map, exec_gen):
                        pass

                    self._engine.shift_market_state_arr()
                elif isinstance(exec_event := event, ProtoOAExecutionEvent):
                    if exec_event.executionType != DEPOSIT_WITHDRAW:
                        raise RuntimeError()

                    dw = exec_event.depositWithdraw
                    if dw.balanceVersion > self._balance_version:
                        balance = dw.balance / 10 ** dw.moneyDigits
                        self._engine.update_balance(balance)
                        self._balance_version = dw.balanceVersion
                else:
                    raise RuntimeError()

    async def _execute(
        self,
        pos_map: dict[int, ProtoOAPosition],
        exec_gen: ComposableAsyncIterable[
            Union[ProtoOAExecutionEvent, ProtoOAOrderErrorEvent]
        ],
    ) -> bool:
        TRADE_SIDE_TO_SIGN = self.TRADE_SIDE_TO_SIGN
        engine = self._engine

        target_pos_sizes_map = engine.evaluate()
        await self._create_orders(target_pos_sizes_map, pos_map, exec_gen)

        balance_version = self._balance_version
        balance = float("nan")

        success = True
        done = False

        async for exec_event in exec_gen:
            if isinstance(exec_event, ProtoOAOrderErrorEvent):
                raise RuntimeError()

            exec_type = exec_event.executionType

            if exec_type == DEPOSIT_WITHDRAW:
                dw = exec_event.depositWithdraw
                if dw.balanceVersion > balance_version:
                    balance = dw.balance / 10 ** dw.moneyDigits
                    balance_version = dw.balanceVersion
                continue

            if exec_type == ORDER_REJECTED:
                assert exec_event.errorCode == NOT_ENOUGH_MONEY
                sym_id = exec_event.order.tradeData.symbolId
                done = engine.skip_symbol_update(sym_id)
                success = False
            elif exec_type == ORDER_FILLED:
                deal = exec_event.deal
                sym_id = deal.symbolId

                volume = deal.volume
                assert volume == deal.filledVolume
                sign = TRADE_SIDE_TO_SIGN[deal.tradeSide]
                size = sign * volume / 10 ** deal.moneyDigits
                price = deal.executionPrice

                cpd = deal.closePositionDetail
                if cpd.IsInitialized() and cpd.balanceVersion > balance_version:
                    balance = cpd.balance / 10 ** cpd.moneyDigits
                    balance_version = cpd.balanceVersion

                done = engine.update_symbol_state(sym_id, size, price)

                pos_map[sym_id] = exec_event.position
            else:
                raise RuntimeError()

            if done:
                break

        if balance_version > self._balance_version:
            assert balance != float("nan")
            engine.update_balance(balance)
            self._balance_version = balance_version

        return success

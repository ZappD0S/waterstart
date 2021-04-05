import asyncio
from typing import TypeVar, Union

from aioitertools import next
from google.protobuf.message import Message
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    MARKET,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)
from waterstart.openapi.OpenApiMessages_pb2 import (
    ProtoOAExecutionEvent,
    ProtoOANewOrderReq,
    ProtoOAOrderErrorEvent,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
)
from waterstart.openapi.OpenApiModelMessages_pb2 import BUY, SELL

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783470

T = TypeVar("T", bound=Message)


async def main() -> None:
    async def send_and_wait(req: Message, res_type: type[T], timeout: float = 5.0) -> T:
        async with client.register(res_type) as gen:
            await client.send_message(req)
            res = await asyncio.wait_for(next(gen), timeout)

        if isinstance(res, ProtoOAErrorRes):
            raise RuntimeError(res.description)

        return res

    client = await OpenApiClient.create(HOST, PORT)

    # refresh_token => wwJXiEBoC-7Uu4NzyNf90iWIZRlFCUdW4jUWBYoDOYs

    try:
        app_auth_req = ProtoOAApplicationAuthReq(
            clientId="2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB",
            clientSecret="B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC",
        )
        app_auth_res = await send_and_wait(app_auth_req, ProtoOAApplicationAuthRes)

        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken="FpNGIMCt16aMrPRM5jiqNxxnBzAsYB8aOxY15r1_EIU",
        )
        acc_auth_res = await send_and_wait(acc_auth_req, ProtoOAAccountAuthRes)

        sym_list_req = ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT_ID)
        sym_list_res = await send_and_wait(sym_list_req, ProtoOASymbolsListRes)

        symbol_name_to_id = {
            name: sym.symbolId
            for sym in sym_list_res.symbol
            if (name := sym.symbolName.lower()) in ["btc/usd", "btc/eur"]
        }

        sym_req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=ACCOUNT_ID, symbolId=[symbol_name_to_id["btc/usd"]]
        )
        [symbol] = (await send_and_wait(sym_req, ProtoOASymbolByIdRes)).symbol

        order_reqs = [
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=BUY,
                volume=int(0.02 * symbol.lotSize),
            ),
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=BUY,
                volume=int(0.03 * symbol.lotSize),
            ),
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=BUY,
                volume=int(0.02 * symbol.lotSize),
            ),
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=SELL,
                volume=int(0.03 * symbol.lotSize),
            ),
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=SELL,
                volume=int(0.01 * symbol.lotSize),
            ),
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=SELL,
                volume=int(0.02 * symbol.lotSize),
            ),
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=SELL,
                volume=int(0.01 * symbol.lotSize),
            ),
        ]

        async with client.register(
            Union[ProtoOAExecutionEvent, ProtoOAOrderErrorEvent]
        ) as gen:
            async for exec_event in gen:

                if isinstance(exec_event, ProtoOAErrorRes):
                    raise RuntimeError(exec_event.description)

                print(exec_event)

    finally:
        await client.close()


asyncio.run(main(), debug=True)

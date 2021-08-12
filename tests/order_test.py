import asyncio
from contextlib import AsyncExitStack
from typing import TypeVar, Union
from waterstart.openapi.OpenApiMessages_pb2 import ProtoOAOrderErrorEvent

from google.protobuf.message import Message
from waterstart.client.app import AppClient
from waterstart.openapi import (
    MARKET,
    SELL,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAErrorRes,
    ProtoOAExecutionEvent,
    ProtoOANewOrderReq,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783470
CLIENT_ID = "2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB"
CLIENT_SECRET = "B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC"

T = TypeVar("T", bound=Message)


async def main() -> None:
    client = await AppClient.create(HOST, PORT, CLIENT_ID, CLIENT_SECRET)

    try:
        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken="uMkVFVFF2T8AaxyOgqyRq29hL4wKtMX2aYqRSn_ILNA",
        )
        _ = await client.send_request(acc_auth_req, ProtoOAAccountAuthRes)

        sym_list_req = ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT_ID)
        sym_list_res = await client.send_request(sym_list_req, ProtoOASymbolsListRes)

        symbol_name_to_id = {
            name: sym.symbolId
            for sym in sym_list_res.symbol
            if (name := sym.symbolName.lower()) in ["btc/usd", "btc/eur"]
        }

        sym_req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=ACCOUNT_ID, symbolId=[symbol_name_to_id["btc/usd"]]
        )
        sym_res = await client.send_request(sym_req, ProtoOASymbolByIdRes)
        [symbol] = sym_res.symbol

        exec_events: list[Union[ProtoOAExecutionEvent, ProtoOAOrderErrorEvent]] = []

        async with AsyncExitStack() as stack:
            res_gen = await stack.enter_async_context(
                client.register_type(ProtoOAExecutionEvent)
            )
            order_err_gen = await stack.enter_async_context(
                client.register_type(ProtoOAOrderErrorEvent)
            )
            err_gen = await stack.enter_async_context(
                client.register_type(ProtoOAErrorRes)
            )

            gen = res_gen | order_err_gen | err_gen

            exec_event = await client.send_request(
                ProtoOANewOrderReq(
                    ctidTraderAccountId=ACCOUNT_ID,
                    symbolId=symbol.symbolId,
                    orderType=MARKET,
                    tradeSide=SELL,
                    # positionId=...,
                    # volume=int(0.02 * symbol.lotSize),
                    volume=int(10.0 * symbol.lotSize),
                ),
                Union[ProtoOAExecutionEvent, ProtoOAOrderErrorEvent],
                gen,
            )

            print(exec_event)
            exec_events.append(exec_event)

            async for event in gen:
                if isinstance(event, ProtoOAErrorRes):
                    raise Exception()

                print(event)
                exec_events.append(event)

    finally:
        await client.aclose()


asyncio.run(main(), debug=True)

import asyncio
from typing import TypeVar

from aioitertools import next
from google.protobuf.message import Message
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    BUY,
    MARKET,
    SELL,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOAExecutionEvent,
    ProtoOANewOrderReq,
    ProtoOAOrderErrorEvent,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783470

T = TypeVar("T", bound=Message)


async def main() -> None:
    client = await OpenApiClient.create(HOST, PORT)

    # refresh_token => wwJXiEBoC-7Uu4NzyNf90iWIZRlFCUdW4jUWBYoDOYs

    try:
        app_auth_req = ProtoOAApplicationAuthReq(
            clientId="2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB",
            clientSecret="B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC",
        )
        app_auth_res = await client.send_and_wait_response(
            app_auth_req, ProtoOAApplicationAuthRes
        )

        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken="FpNGIMCt16aMrPRM5jiqNxxnBzAsYB8aOxY15r1_EIU",
        )
        acc_auth_res = await client.send_and_wait_response(
            acc_auth_req, ProtoOAAccountAuthRes
        )

        sym_list_req = ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT_ID)
        sym_list_res = await client.send_and_wait_response(
            sym_list_req, ProtoOASymbolsListRes
        )

        symbol_name_to_id = {
            name: sym.symbolId
            for sym in sym_list_res.symbol
            if (name := sym.symbolName.lower()) in ["btc/usd", "btc/eur"]
        }

        sym_req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=ACCOUNT_ID, symbolId=[symbol_name_to_id["btc/usd"]]
        )
        sym_res = await client.send_and_wait_response(sym_req, ProtoOASymbolByIdRes)
        [symbol] = sym_res.symbol

        async with client.register_types(
            (ProtoOAExecutionEvent, ProtoOAOrderErrorEvent)
        ) as gen:
            await client.send_message(
                ProtoOANewOrderReq(
                    ctidTraderAccountId=ACCOUNT_ID,
                    symbolId=symbol.symbolId,
                    orderType=MARKET,
                    tradeSide=SELL,
                    # positionId=...,
                    volume=int(0.02 * symbol.lotSize),
                ),
            )

            async for exec_event in gen:
                if isinstance(exec_event, ProtoOAErrorRes):
                    print(exec_event)
                    raise RuntimeError(exec_event.description)

                print(exec_event)

    finally:
        await client.close()


asyncio.run(main(), debug=True)

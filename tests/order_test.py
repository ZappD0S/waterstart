import asyncio
from typing import TypeVar

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
from waterstart.utils import ComposableAsyncIterator

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783470
CLIENT_ID = "2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB"
CLIENT_SECRET = "B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC"

T = TypeVar("T", bound=Message)


async def main() -> None:
    client = await AppClient.create(HOST, PORT, CLIENT_ID, CLIENT_SECRET)

    try:
        # acc_auth_req = ProtoOAAccountAuthReq(
        #     ctidTraderAccountId=ACCOUNT_ID,
        #     accessToken="tSn9WhMJ1sK79D-kreB5lAmtphSW6YDuDGh-5Tn_wl8",
        # )
        # _ = await client.send_request(acc_auth_req, ProtoOAAccountAuthRes)

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

        exec_events: list[ProtoOAExecutionEvent] = []

        exec_event = await client.send_request(
            ProtoOANewOrderReq(
                ctidTraderAccountId=ACCOUNT_ID,
                symbolId=symbol.symbolId,
                orderType=MARKET,
                tradeSide=SELL,
                # positionId=...,
                volume=int(0.02 * symbol.lotSize),
            ),
            ProtoOAExecutionEvent,
        )

        print(exec_event)
        exec_events.append(exec_event)

        res_gen_cm = client.register_type(ProtoOAExecutionEvent)
        err_gen_cm = client.register_type(ProtoOAErrorRes)

        async with res_gen_cm as res_gen, err_gen_cm as err_gen:
            gen = ComposableAsyncIterator.from_it(res_gen) | err_gen
            async for event in gen:
                if isinstance(event, ProtoOAErrorRes):
                    raise Exception()

                print(event)
                exec_events.append(event)

    finally:
        await client.close()


asyncio.run(main(), debug=True)

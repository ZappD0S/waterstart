import asyncio
import time
from typing import Optional

import numpy as np
from aioitertools.builtins import next
from google.protobuf.message import Message
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    M1,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOASpotEvent,
    ProtoOASubscribeLiveTrendbarReq,
    ProtoOASubscribeLiveTrendbarRes,
    ProtoOASubscribeSpotsReq,
    ProtoOASubscribeSpotsRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783271


async def main() -> None:
    async def send_and_wait(req: Message, res_type: type[Message]):
        async with client.register(lambda m: isinstance(m, res_type)) as gen:
            await client.send_message(req)
            res = await asyncio.wait_for(next(gen), 5.0)

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

        print(max(sym_list_res.symbol, key=lambda x: len(x.symbolName)))

        sorted_pairs = np.load("../financelab/train_data/train_data.npz")[
            "sorted_pairs"
        ]
        pairs_set = set(pair.lower() for pair in sorted_pairs)

        symbol_id_to_name = {
            sym.symbolId: name
            for sym in sym_list_res.symbol
            if (name := sym.symbolName.lower()) in pairs_set
        }

        sub_spot_req = ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=ACCOUNT_ID, symbolId=symbol_id_to_name.keys()
        )
        sub_spot_res = await send_and_wait(sub_spot_req, ProtoOASubscribeSpotsRes)

        tasks = []
        for sym_id in symbol_id_to_name.keys():
            sub_trendbar_req = ProtoOASubscribeLiveTrendbarReq(
                ctidTraderAccountId=ACCOUNT_ID, symbolId=sym_id, period=M1
            )
            task = asyncio.create_task(
                send_and_wait(sub_trendbar_req, ProtoOASubscribeLiveTrendbarRes)
            )
            tasks.append(task)

        await asyncio.wait(tasks)

        last_completed_timestamp = 0
        symbols_set = frozenset(symbol_id_to_name)
        symbols_left = set(symbols_set)
        t0: Optional[float] = None

        async with client.register(lambda m: isinstance(m, ProtoOASpotEvent)) as gen:
            async for spot_event in gen:
                if not spot_event.symbolId in symbols_left:
                    continue

                if not spot_event.trendbar:
                    continue

                [trendbar] = spot_event.trendbar

                if trendbar.utcTimestampInMinutes <= last_completed_timestamp:
                    continue

                if t0 is None:
                    t0 = time.time()

                symbol_name = symbol_id_to_name[spot_event.symbolId]
                print(f"received spot event for: {symbol_name}")
                symbols_left.remove(spot_event.symbolId)

                if not symbols_left:
                    print(f"--- time taken: {time.time() - t0} s ---")
                    t0 = None
                    symbols_left = set(symbols_set)
                    last_completed_timestamp = trendbar.utcTimestampInMinutes

    except Exception as ex:
        print(ex)
    finally:
        await client.close()


asyncio.run(main(), debug=True)

import asyncio
import time
from datetime import datetime
from typing import Optional, TypeVar

import numpy as np
from google.protobuf.message import Message
from waterstart.client.app import AppClient
from waterstart.openapi import (
    M1,
    M30,
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
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783271
CLIENT_ID = "2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB"
CLIENT_SECRET = "B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC"


T = TypeVar("T", bound=Message)


async def main() -> None:
    client = await AppClient.create(HOST, PORT, CLIENT_ID, CLIENT_SECRET)

    # refresh_token => wwJXiEBoC-7Uu4NzyNf90iWIZRlFCUdW4jUWBYoDOYs
    sorted_pairs = np.load("../financelab/train_data/train_data.npz")["sorted_pairs"]
    pairs_set = set(pair.lower() for pair in sorted_pairs)

    try:
        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken="FpNGIMCt16aMrPRM5jiqNxxnBzAsYB8aOxY15r1_EIU",
        )
        acc_auth_res = await client.send_request(
            acc_auth_req, ProtoOAAccountAuthRes
        )

        light_sym_list_req = ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT_ID)
        light_sym_list_res = await client.send_request(
            light_sym_list_req, ProtoOASymbolsListRes
        )
        symbol_id_to_name = {
            sym.symbolId: name
            for sym in light_sym_list_res.symbol
            if (name := sym.symbolName.lower()) in pairs_set
        }

        symbol_ids = symbol_id_to_name.keys()

        sym_list_req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=ACCOUNT_ID,
            symbolId=[sym.symbolId for sym in light_sym_list_res.symbol],
        )
        sym_list_res = await client.send_request(
            sym_list_req, ProtoOASymbolByIdRes
        )

        symbol_id_to_minvol = {
            sym.symbolId: sym.minVolume / 100 for sym in sym_list_res.symbol
        }

        # symbol_ids = symbol_id_to_name.keys() & symbol_id_to_minvol.keys()
        assert symbol_ids & symbol_id_to_minvol.keys()

        symbol_name_to_minvol = {
            symbol_id_to_name[sym_id]: symbol_id_to_minvol[sym_id]
            for sym_id in symbol_ids
        }
        print(symbol_name_to_minvol)
        print(max(light_sym_list_res.symbol, key=lambda x: len(x.symbolName)))

        sub_spot_req = ProtoOASubscribeSpotsReq(
            ctidTraderAccountId=ACCOUNT_ID, symbolId=symbol_ids
        )
        sub_spot_res = await client.send_request(
            sub_spot_req, ProtoOASubscribeSpotsRes
        )

        tasks = []
        for sym_id in symbol_ids:
            sub_trendbar_req = ProtoOASubscribeLiveTrendbarReq(
                ctidTraderAccountId=ACCOUNT_ID, symbolId=sym_id, period=M30
            )
            task = asyncio.create_task(
                client.send_request(
                    sub_trendbar_req, ProtoOASubscribeLiveTrendbarRes
                )
            )
            tasks.append(task)

        await asyncio.wait(tasks)

        # last_completed_timestamp = 0
        symbols_set = frozenset(symbol_id_to_name)
        # symbols_left = set(symbols_set)
        t0: Optional[float] = None

        async with client.register_type(ProtoOASpotEvent) as gen:
            async for spot_event in gen:
                if isinstance(spot_event, ProtoOAErrorRes):
                    raise RuntimeError(spot_event.description)

                # if not spot_event.symbolId in symbols_left:
                #     continue

                symbol_name = symbol_id_to_name[spot_event.symbolId]

                if symbol_name != "eurjpy":
                    continue

                utc_ts_in_mins = int(datetime.now().timestamp() / 60)
                if not spot_event.trendbar:
                    # print(f"spot event w/o trendabr for: {symbol_name}")
                    # print(f"\tutc ts: {utc_ts_in_mins}")
                    continue

                [trendbar] = spot_event.trendbar

                # if trendbar.utcTimestampInMinutes <= last_completed_timestamp:
                #     print(f"skipped event w/ trendbar for: {symbol_name}")
                #     print(f"\tutc ts: {utc_ts_in_mins}")
                #     print(f"\treceived utc ts: {trendbar.utcTimestampInMinutes}")
                #     continue

                # if t0 is None:
                #     t0 = time.time()

                print(f"spot event for: {symbol_name}")
                print(f"\ttrendbar: {trendbar}")
                print(f"\tutc ts: {utc_ts_in_mins}")
                print(f"\treceived utc ts: {trendbar.utcTimestampInMinutes}")

                # symbols_left.remove(spot_event.symbolId)

                # if not symbols_left:
                #     print(f"--- time taken: {time.time() - t0} s ---")
                #     t0 = None
                #     symbols_left = set(symbols_set)
                #     last_completed_timestamp = trendbar.utcTimestampInMinutes

    except Exception as ex:
        print(ex)
    finally:
        await client.aclose()


asyncio.run(main(), debug=True)

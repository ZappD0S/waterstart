import asyncio
from datetime import timedelta
from typing import TypeVar
from waterstart.openapi.OpenApiMessages_pb2 import ProtoOASymbolsForConversionReq

from google.protobuf.message import Message
from waterstart.client.app import AppClient
from waterstart.openapi import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOADealListReq,
    ProtoOADealListRes,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = 20783470
CLIENT_ID = "2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB"
CLIENT_SECRET = "B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC"


T = TypeVar("T", bound=Message)


async def main() -> None:
    client = await AppClient.create(HOST, PORT, CLIENT_ID, CLIENT_SECRET)

    # refresh_token => wwJXiEBoC-7Uu4NzyNf90iWIZRlFCUdW4jUWBYoDOYs

    try:
        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken="FpNGIMCt16aMrPRM5jiqNxxnBzAsYB8aOxY15r1_EIU",
        )
        acc_auth_res = await client.send_request(
            acc_auth_req, ProtoOAAccountAuthRes
        )

        # ProtoOASymbolsForConversionReq(ctidTraderAccountId=ACCOUNT_ID, firstAssetId=)

        reconcile_req = ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID)
        reconcile_res = await client.send_request(
            reconcile_req, ProtoOAReconcileRes
        )

        position = reconcile_res.position[0]

        from_timestamp = position.tradeData.openTimestamp
        to_timestamp = position.utcLastUpdateTimestamp

        assert to_timestamp - from_timestamp < timedelta(weeks=1).total_seconds()

        deal_list_req = ProtoOADealListReq(
            ctidTraderAccountId=ACCOUNT_ID,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
        )

        deal_list_res = await client.send_request(
            deal_list_req, ProtoOADealListRes
        )

        pos_deals = [
            deal
            for deal in deal_list_res.deal
            if deal.positionId == position.positionId
        ]

        for deal in pos_deals:
            print(deal)

    finally:
        await client.close()


asyncio.run(main(), debug=True)

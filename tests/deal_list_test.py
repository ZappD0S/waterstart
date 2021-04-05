import asyncio
from datetime import timedelta
from typing import TypeVar

from aioitertools import next
from google.protobuf.message import Message
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOADealListReq,
    ProtoOADealListRes,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
)

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

        reconcile_req = ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID)
        reconcile_res = await send_and_wait(reconcile_req, ProtoOAReconcileRes)

        position = reconcile_res.position[0]

        from_timestamp = position.tradeData.openTimestamp
        to_timestamp = position.utcLastUpdateTimestamp

        assert to_timestamp - from_timestamp < timedelta(weeks=1).total_seconds()

        deal_list_req = ProtoOADealListReq(
            ctidTraderAccountId=ACCOUNT_ID,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
        )

        deal_list_res = await send_and_wait(deal_list_req, ProtoOADealListRes)

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

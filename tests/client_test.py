import asyncio

from aioitertools.builtins import next
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035


async def main():
    client = await OpenApiClient.create(HOST, PORT)

    gen = client.register(lambda m: isinstance(m, ProtoOAApplicationAuthRes))

    req = ProtoOAApplicationAuthReq(
        clientId="2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB",
        clientSecret="B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC",
    )
    await client.send_message(req)

    try:
        return await asyncio.wait_for(next(gen), 3.0)
    except asyncio.TimeoutError:
        print("failed..")
    finally:
        await client.close()


res = asyncio.run(main())

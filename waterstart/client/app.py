from __future__ import annotations

import asyncio
from asyncio import StreamReader, StreamWriter
from collections import Awaitable, Callable

from ..openapi import ProtoOAApplicationAuthReq, ProtoOAApplicationAuthRes
from .base import BaseReconnectingClient, HelperClient


class AppClient(BaseReconnectingClient):
    @staticmethod
    async def create(
        host: str,
        port: int,
        client_id: str,
        client_secret: str,
    ) -> AppClient:
        return AppClient(
            lambda: asyncio.open_connection(host, port, ssl=True),
            client_id,
            client_secret,
        )

    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
        client_id: str,
        client_secret: str,
    ) -> None:
        super().__init__(open_connection)
        self._client_id = client_id
        self._client_secret = client_secret

    async def _connect(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> HelperClient:
        auth_req = ProtoOAApplicationAuthReq(
            clientId=self._client_id,
            clientSecret=self._client_secret,
        )

        while True:
            helper_client = await super()._connect(open_connection)
            try:
                _ = await helper_client.send_request(
                    auth_req, ProtoOAApplicationAuthRes
                )
            except Exception:  # TODO: correct exception...
                continue

            return helper_client

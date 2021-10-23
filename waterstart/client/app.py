from __future__ import annotations

import asyncio
from asyncio import StreamReader, StreamWriter
from collections.abc import Awaitable, Callable

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

    async def _setup_connection(self, helper_client: HelperClient) -> None:
        await helper_client.send_request(
            ProtoOAApplicationAuthReq(
                clientId=self._client_id,
                clientSecret=self._client_secret,
            ),
            ProtoOAApplicationAuthRes,
        )

from __future__ import annotations

import asyncio
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Optional, TypeVar, Union

from google.protobuf.message import Message

from ..openapi import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAAccountsTokenInvalidatedEvent,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOARefreshTokenReq,
    ProtoOARefreshTokenRes,
)
from .base import BaseReconnectingClient, HelperClient

S = TypeVar("S")
T = TypeVar("T", bound=Message)


class TraderClient(BaseReconnectingClient):
    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
        client_id: str,
        client_secret: str,
        trader_id: int,
        auth_token: str,
        refresh_token: str,
    ) -> None:
        super().__init__(open_connection)

        self._trader_id = trader_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token = auth_token
        self._refresh_token = refresh_token

        self._refresh_token_on_expiry_task = asyncio.create_task(
            self._refresh_token_on_expiry()
        )

    def _belongs_to_trader(
        self, res: T, pred: Optional[Callable[[T], bool]] = None
    ) -> bool:
        if (trader_id := getattr(res, "ctidTraderAccountId", None)) is None:
            return False

        belongs_to_trader = trader_id == self._trader_id

        if pred is None:
            return belongs_to_trader

        return belongs_to_trader and pred(res)

    async def _connect(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> HelperClient:
        account_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=self._trader_id, accessToken=self._access_token
        )
        app_auth_req = ProtoOAApplicationAuthReq(
            clientId=self._client_id, clientSecret=self._client_secret
        )

        while True:
            helper_client = await super()._connect(open_connection)

            try:
                _ = await helper_client.send_request(
                    app_auth_req, ProtoOAApplicationAuthRes
                )
                _ = await helper_client.send_request(
                    account_auth_req,
                    ProtoOAAccountAuthRes,
                    pred=lambda res: res.ctidTraderAccountId == self._trader_id,
                )
            except Exception:  # TODO: correct exception...
                continue

            return helper_client

    async def _refresh_token_on_expiry(self) -> None:
        async with self.register_type(
            ProtoOAAccountsTokenInvalidatedEvent,
            lambda event: self._trader_id in event.ctidTraderAccountIds,
        ) as gen:
            async for _ in gen:
                req = ProtoOARefreshTokenReq(refreshToken=self._refresh_token)

                async with self._global_lock:
                    res = await self.send_request(req, ProtoOARefreshTokenRes)

                self._refresh_token = res.refreshToken
                assert res.accessToken == self._access_token

    async def send_request_from_trader(
        self,
        build_req: Callable[[int], Message],
        res_type: type[T],
        gen: Optional[AsyncIterator[Union[T, ProtoOAErrorRes]]] = None,
        pred: Optional[Callable[[T], bool]] = None,
    ) -> T:
        return await self.send_request(
            build_req(self._trader_id),
            res_type,
            gen,
            lambda res: self._belongs_to_trader(res, pred),
        )

    def send_requests_from_trader(
        self,
        build_key_to_req: Callable[[int], Mapping[S, Message]],
        res_type: type[T],
        get_key: Callable[[T], Optional[S]],
        gen: Optional[AsyncIterator[Union[T, ProtoOAErrorRes]]] = None,
    ) -> AsyncIterator[tuple[S, T]]:
        def _get_key(x: T) -> Optional[S]:
            if not self._belongs_to_trader(x):
                return None

            return get_key(x)

        return self.send_requests(
            build_key_to_req(self._trader_id), res_type, _get_key, gen
        )

    async def close(self) -> None:
        self._refresh_token_on_expiry_task.cancel()
        # TODO: account logout

        await super().close()

        try:
            await self._refresh_token_on_expiry_task
        except asyncio.CancelledError:
            pass

    @staticmethod
    async def create(
        host: str,
        port: int,
        client_id: str,
        client_secret: str,
        trader_id: int,
        auth_token: str,
        refresh_token: str,
    ) -> TraderClient:
        return TraderClient(
            lambda: asyncio.open_connection(host, port, ssl=True),
            client_id,
            client_secret,
            trader_id,
            auth_token,
            refresh_token,
        )

from __future__ import annotations

import asyncio
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Optional, TypeVar

from google.protobuf.message import Message
from waterstart.utils import ComposableAsyncIterable

from ..openapi import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAAccountsTokenInvalidatedEvent,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
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

    def _belongs_to_trader(self, res: Message) -> bool:
        if (trader_id := getattr(res, "ctidTraderAccountId", None)) is None:
            return False

        return trader_id == self._trader_id

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

    @asynccontextmanager
    async def register_type_for_trader(
        self,
        message_type: type[T],
        pred: Optional[Callable[[T], bool]] = None,
        maxsize: int = 0,
    ) -> AsyncIterator[ComposableAsyncIterable[T]]:
        def get_comp_pred(pred: Callable[[T], bool]):
            def comp_pred(x: T) -> bool:
                return belongs_to_trader(x) and pred(x)

            return comp_pred

        belongs_to_trader = self._belongs_to_trader
        async with self.register_type(
            message_type,
            belongs_to_trader if pred is None else get_comp_pred(pred),
            maxsize,
        ) as gen:
            yield gen

    async def send_request_from_trader(
        self,
        build_req: Callable[[int], Message],
        res_type: type[T],
        res_gen: Optional[ComposableAsyncIterable[T]] = None,
        pred: Optional[Callable[[T], bool]] = None,
    ) -> T:
        def get_comp_pred(pred: Callable[[T], bool]):
            def comp_pred(x: T) -> bool:
                return belongs_to_trader(x) and pred(x)

            return comp_pred

        belongs_to_trader = self._belongs_to_trader
        return await self.send_request(
            build_req(self._trader_id),
            res_type,
            res_gen,
            belongs_to_trader if pred is None else get_comp_pred(pred),
        )

    def send_requests_from_trader(
        self,
        build_key_to_req: Callable[[int], Mapping[S, Message]],
        res_type: type[T],
        get_key: Callable[[T], Optional[S]],
        res_gen: Optional[ComposableAsyncIterable[T]] = None,
    ) -> AsyncIterator[tuple[S, T]]:
        belongs_to_trader = self._belongs_to_trader

        return self.send_requests(
            build_key_to_req(self._trader_id),
            res_type,
            lambda res: get_key(res) if belongs_to_trader(res) else None,
            res_gen,
        )

    async def aclose(self) -> None:
        self._refresh_token_on_expiry_task.cancel()
        # TODO: account logout

        await super().aclose()

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

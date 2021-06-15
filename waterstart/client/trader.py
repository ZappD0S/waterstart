from asyncio import StreamReader, StreamWriter
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TypeVar
from waterstart.openapi.OpenApiMessages_pb2 import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAAccountsTokenInvalidatedEvent,
    ProtoOARefreshTokenReq,
    ProtoOARefreshTokenRes,
)

from google.protobuf.message import Message

from .app import AppClient, HelperClient

S = TypeVar("S")
T = TypeVar("T", bound=Message)


# TODO: implement token refresh
class TraderClient(AppClient):
    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
        client_id: str,
        client_secret: str,
        trader_id: int,
        auth_token: str,
        refresh_token: str,
    ) -> None:
        super().__init__(open_connection, client_id, client_secret)
        self._trader_id = trader_id

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
        helper_client = await self._connect(open_connection)

        req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=self._trader_id, accessToken=self._access_token
        )

        _ = await helper_client.send_request(
            req,
            ProtoOAAccountAuthRes,
            lambda res: res.ctidTraderAccountId == self._trader_id,
        )

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

    # TODO: add "using gen" methods
    async def send_request_from_trader(
        self, build_req: Callable[[int], Message], res_type: type[T]
    ) -> T:
        return await self.send_request(
            build_req(self._trader_id),
            res_type,
            self._belongs_to_trader,
        )

    async def send_requests_from_trader(
        self,
        build_key_to_req: Callable[[int], Mapping[S, Message]],
        res_type: type[T],
        get_key: Callable[[T], S],
    ) -> AsyncIterator[tuple[S, T]]:
        async for key_res in self.send_requests(
            build_key_to_req(self._trader_id),
            res_type,
            get_key,
            self._belongs_to_trader,
        ):
            yield key_res

    async def close(self) -> None:
        self._refresh_token_on_expiry_task.cancel()

        try:
            await self._refresh_token_on_expiry_task
        except asyncio.CancelledError:
            pass

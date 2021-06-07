from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TypeVar

from google.protobuf.message import Message

from .app import AppClient

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
        refresh_token: str
    ) -> None:
        super().__init__(open_connection, client_id, client_secret)
        self._trader_id = trader_id

        # self.auth_token =
        self._refresh_token = refresh_token

    def _belongs_to_trader(self, res: Message) -> bool:
        if (trader_id := getattr(res, "ctidTraderAccountId", None)) is None:
            return False

        return trader_id == self._trader_id

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
        return await self.send_requests(
            build_key_to_req(self._trader_id),
            res_type,
            get_key,
            self._belongs_to_trader,
        )

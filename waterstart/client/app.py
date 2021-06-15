from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from asyncio import StreamReader, StreamWriter
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Set,
)
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Final, Optional, TypeVar, Union

from google.protobuf.message import Message

from ..observable import Observable
from ..openapi import (
    ProtoHeartbeatEvent,
    ProtoMessage,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    messages_dict,
)

S = TypeVar("S")
T = TypeVar("T", bound=Message)
U = TypeVar("U", bound=Message)
V = TypeVar("V", bound=Message)


class BaseClient(Observable[Message], ABC):
    def __init__(self) -> None:
        super().__init__()
        self._lock_map: dict[type[Message], asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def register_type(
        self, message_type: type[T], pred: Optional[Callable[[T], bool]] = None
    ) -> AsyncContextManager[AsyncIterator[T]]:
        def get_map_f(pred: Callable[[T], bool]):
            def map_f(x: Message):
                return x if isinstance(x, message_type) and pred(x) else None

            return map_f

        if pred is None:
            pred = lambda _: True

        return self.register(get_map_f(pred))

    async def _get_async_generator(self) -> AsyncGenerator[Message, None]:
        while True:
            yield await self._read_message()

    @abstractmethod
    async def _write_message(self, message: Message) -> None:
        ...

    @abstractmethod
    async def _read_message(self) -> Message:
        ...

    def register_types(
        self,
        message_types: tuple[type[T], type[U]],
        pred: Optional[Callable[[T], bool]] = None,
    ) -> AsyncContextManager[AsyncIterator[Union[T, U]]]:
        def get_map_f(pred: Callable[[T], bool]):
            def map_f(x: Message):
                if isinstance(x, message_types[1]) or (
                    isinstance(x, message_types[0]) and pred(x)
                ):
                    return x
                else:
                    return None

            return map_f

        if pred is None:
            pred = lambda _: True

        return self.register(get_map_f(pred))  # type: ignore

    @asynccontextmanager
    async def _type_lock(self, res_type: type[T]):
        if (lock := self._lock_map.get(res_type)) is None:
            lock = self._lock_map[res_type] = asyncio.Lock()
            found = False
        else:
            found = True

        try:
            async with lock, self._global_lock:
                yield
        finally:
            if not found:
                del self._lock_map[res_type]

    async def send_request(
        self,
        req: Message,
        res_type: type[T],
        pred: Optional[Callable[[T], bool]] = None,
    ) -> T:
        async with self.register_types((res_type, ProtoOAErrorRes), pred) as gen:
            return await self.send_request_using_gen(req, res_type, gen)

    # TODO: find better name
    async def send_request_using_gen(
        self,
        req: Message,
        res_type: type[T],
        gen: AsyncIterator[Union[T, ProtoOAErrorRes]],
    ) -> T:
        async with self._type_lock(res_type):
            return await self._send_request_using_gen(req, gen)

    async def _send_request_using_gen(
        self,
        req: Message,
        gen: AsyncIterator[Union[T, ProtoOAErrorRes]],
    ) -> T:
        await self._write_message(req)

        res: Union[T, ProtoOAErrorRes, None] = None
        async for res in gen:
            break

        if res is None:
            raise RuntimeError()

        if isinstance(res, ProtoOAErrorRes):
            raise RuntimeError(f"{res.errorCode}: {res.description}")

        return res

    async def send_requests(
        self,
        key_to_req: Mapping[S, Message],
        res_type: type[T],
        get_key: Callable[[T], S],
        pred: Optional[Callable[[T], bool]] = None,
    ) -> AsyncIterator[tuple[S, T]]:
        async with self.register_types((res_type, ProtoOAErrorRes), pred) as gen:
            async for key_res in self.send_requests_using_gen(
                key_to_req, res_type, get_key, gen
            ):
                yield key_res

    async def send_requests_using_gen(
        self,
        key_to_req: Mapping[S, Message],
        res_type: type[T],
        get_key: Callable[[T], S],
        gen: AsyncIterator[Union[T, ProtoOAErrorRes]],
    ) -> AsyncIterator[tuple[S, T]]:
        seen_keys: set[S] = set()

        async with self._type_lock(res_type):
            tasks_left: Set[asyncio.Future[T]] = {
                asyncio.create_task(self._send_request_using_gen(req, gen))
                for req in key_to_req.values()
            }

            while tasks_left:
                [done_task], tasks_left = await asyncio.wait(
                    tasks_left, return_when=asyncio.FIRST_COMPLETED
                )

                res = await done_task
                key = get_key(res)

                if key in seen_keys:
                    raise RuntimeError()

                yield key, res
                seen_keys.add(key)


class HelperClient(BaseClient):
    _payloadtype_to_messageproto: Mapping[int, type[Message]] = {
        proto.payloadType.DESCRIPTOR.default_value: proto  # type: ignore
        for proto in messages_dict.values()
        if hasattr(proto, "payloadType")
    }

    def __init__(self, reader: StreamReader, writer: StreamWriter) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer

    def _parse_message(self, data: bytes) -> Message:
        protomessage = ProtoMessage.FromString(data)
        messageproto = self._payloadtype_to_messageproto[protomessage.payloadType]
        return messageproto.FromString(protomessage.payload)

    async def _read_message(self) -> Message:
        return await self.read_message()

    async def read_message(self) -> Message:
        length_data = await self._reader.readexactly(4)
        length = int.from_bytes(length_data, byteorder="big")

        if length <= 0:
            raise RuntimeError()

        payload_data = await self._reader.readexactly(length)
        return self._parse_message(payload_data)

    async def _write_message(self, message: Message) -> None:
        await self.write_message(message)

    async def write_message(self, message: Message) -> None:
        protomessage = ProtoMessage(
            payloadType=message.payloadType,  # type: ignore
            payload=message.SerializeToString(),
        )
        payload_data = protomessage.SerializeToString()
        length_data = len(payload_data).to_bytes(4, byteorder="big")
        self._writer.write(length_data + payload_data)
        await self._writer.drain()
        self._last_sent_message_time = time.time()

    async def close(self) -> None:
        self._writer.close()
        await self._writer.wait_closed()


# TODO: add task that listens to ProtoOAClientDisconnectEvent and
# reconnects when it happens..
class AppClient(BaseClient):
    HEARTBEAT_INTERVAL: Final[float] = 10.0
    RECONNECTION_DELAY: Final[float] = 1.0

    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
        client_id: str,
        client_secret: str,
    ) -> None:
        super().__init__()

        self._client_id = client_id
        self._client_secret = client_secret

        def create_connect_task() -> asyncio.Task[HelperClient]:
            return asyncio.create_task(self._connect(open_connection))

        self._create_connect_task = create_connect_task
        self._connect_task: asyncio.Task[HelperClient] = create_connect_task()

        self._last_sent_t = 0.0
        self._heartbeat_task = asyncio.create_task(self._send_heatbeat())

    async def _connect(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> HelperClient:
        async def get_reader_writer() -> tuple[StreamReader, StreamWriter]:
            while True:
                try:
                    return await open_connection()
                except:  # TODO: correct exception
                    await asyncio.sleep(self.RECONNECTION_DELAY)

        # TODO: this functions should never fail, so here we should handle all exceptions
        reader, writer = await get_reader_writer()
        helper_client = HelperClient(reader, writer)
        auth_req = ProtoOAApplicationAuthReq(
            clientId=self._client_id,
            clientSecret=self._client_secret,
        )
        _ = await helper_client.send_request(auth_req, ProtoOAApplicationAuthRes)

        return helper_client

    async def _write_message(self, message: Message) -> None:
        while True:
            helper_client = await self._connect_task
            try:
                return await helper_client.write_message(message)
            except:  # TODO: correct exception
                self._connect_task = self._create_connect_task()

    async def _read_message(self) -> Message:
        while True:
            helper_client = await self._connect_task
            try:
                return await helper_client.read_message()
            except:  # TODO: correct exception
                self._connect_task = self._create_connect_task()

    async def close(self) -> None:
        self._heartbeat_task.cancel()
        self._connect_task.cancel()

        try:
            await self._heartbeat_task
        except asyncio.CancelledError:
            pass

        try:
            helper_client = await self._connect_task
        except asyncio.CancelledError:
            return

        await helper_client.close()

    async def _send_heatbeat(self) -> None:
        while True:
            if (delta := time.time() - self._last_sent_t) < self.HEARTBEAT_INTERVAL:
                await asyncio.sleep(delta)
                continue

            helper_client = await self._connect_task

            try:
                await helper_client.write_message(ProtoHeartbeatEvent())
            except:  # TODO: correct exception
                self._connect_task = self._create_connect_task()

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

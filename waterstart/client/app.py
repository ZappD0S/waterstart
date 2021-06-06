from __future__ import annotations

import asyncio
import time
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import contextmanager
from typing import (
    AsyncContextManager,
    Awaitable,
    Final,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

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


# TODO: helper class that takes reader and writer and has "unprotected" methods that is
# method that don't handle exceptions.
# also create a base abstract class that defines the methods we need to have in common


class AppClient(Observable[Message]):
    HEARTBEAT_INTERVAL: Final[float] = 10.0

    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
        client_id: str,
        client_secret: str,
    ) -> None:
        super().__init__()

        self._client_id = client_id
        self._client_secret = client_secret

        self._create_connect_task: Callable[
            [], asyncio.Task[tuple[StreamReader, StreamWriter]]
        ] = lambda: asyncio.create_task(self._connect(open_connection))
        self._connect_task = self._create_connect_task()

        self._payloadtype_to_messageproto: Mapping[int, type[Message]] = {
            proto.payloadType.DESCRIPTOR.default_value: proto  # type: ignore
            for proto in messages_dict.values()
            if hasattr(proto, "payloadType")
        }

        self._last_sent_message_time = 0.0
        self._heartbeat_task = asyncio.create_task(self._send_heatbeat())
        self._lock_map: dict[type[Message], asyncio.Lock] = {}

    async def _authenticate_client(
        self, reader: StreamReader, writer: StreamWriter
    ) -> None:
        def map_f(x: Message):
            if isinstance(x, (ProtoOAApplicationAuthRes, ProtoOAErrorRes)):
                return x
            else:
                return None

        auth_req = ProtoOAApplicationAuthReq(
            clientId=self._client_id,
            clientSecret=self._client_secret,
        )
        auth_res: Optional[Union[ProtoOAApplicationAuthRes, ProtoOAErrorRes]] = None
        gen: AsyncIterator[Union[ProtoOAApplicationAuthRes, ProtoOAErrorRes]]

        async with self._register_with_iterable(
            map_f, self._iterate_messages(reader)
        ) as gen:
            await self._write_message_to_writer(auth_req, writer)
            async for auth_res in gen:
                break

        if auth_res is None:
            raise RuntimeError()

        if isinstance(auth_res, ProtoOAErrorRes):
            raise RuntimeError(f"{auth_res.errorCode}: {auth_res.description}")

    async def _connect(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> tuple[StreamReader, StreamWriter]:
        while True:
            try:
                reader, writer = await open_connection()
                await self._authenticate_client(reader, writer)
            except:  # TODO: correct exception
                continue

            return reader, writer

    async def _write_message(self, message: Message) -> None:
        while True:
            _, writer = await self._connect_task
            try:
                await self._write_message_to_writer(message, writer)
            except:  # TODO: correct exception
                self._connect_task = self._create_connect_task()
                continue

            return

    async def _write_message_to_writer(
        self, message: Message, writer: StreamWriter
    ) -> None:
        protomessage = ProtoMessage(
            payloadType=message.payloadType,  # type: ignore
            payload=message.SerializeToString(),
        )
        payload_data = protomessage.SerializeToString()
        length_data = len(payload_data).to_bytes(4, byteorder="big")
        writer.write(length_data + payload_data)
        await writer.drain()
        self._last_sent_message_time = time.time()

    @contextmanager
    def _type_lock(self, res_type: type[T]) -> Iterator[asyncio.Lock]:
        if (lock := self._lock_map.get(res_type)) is None:
            lock = self._lock_map[res_type] = asyncio.Lock()
            found = False
        else:
            found = True

        try:
            yield lock
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
        res: Union[T, ProtoOAErrorRes, None] = None
        with self._type_lock(res_type):
            await self._write_message(req)
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
            return self.send_requests_using_gen(key_to_req, res_type, get_key, gen)

    async def send_requests_using_gen(
        self,
        key_to_req: Mapping[S, Message],
        res_type: type[T],
        get_key: Callable[[T], S],
        gen: AsyncIterator[Union[T, ProtoOAErrorRes]],
    ) -> AsyncIterator[tuple[S, T]]:
        keys_left = set(key_to_req)

        with self._type_lock(res_type):
            tasks = [
                asyncio.create_task(self._write_message(req))
                for req in key_to_req.values()
            ]

            await asyncio.wait(tasks)

            async for res in gen:
                if isinstance(res, ProtoOAErrorRes):
                    raise RuntimeError()

                key = get_key(res)

                if key not in keys_left:
                    raise RuntimeError()

                yield key, res
                keys_left.remove(key)

                if not keys_left:
                    break

    def _parse_message(self, data: bytes) -> Message:
        protomessage = ProtoMessage.FromString(data)
        messageproto = self._payloadtype_to_messageproto[protomessage.payloadType]
        return messageproto.FromString(protomessage.payload)

    async def _read_message_from_reader(self, reader: StreamReader) -> Message:
        length_data = await reader.readexactly(4)
        length = int.from_bytes(length_data, byteorder="big")

        if length <= 0:
            raise RuntimeError()

        payload_data = await reader.readexactly(length)
        return self._parse_message(payload_data)

    async def _iterate_messages(self, reader: StreamReader) -> AsyncIterator[Message]:
        while True:
            yield await self._read_message_from_reader(reader)

    async def _get_async_iterator(self) -> AsyncIterator[Message]:
        while True:
            reader, _ = await self._connect_task
            try:
                async for message in self._iterate_messages(reader):
                    yield message
            except:  # TODO: correct exception
                self._connect_task = self._create_connect_task()

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

    async def close(self) -> None:
        self._heartbeat_task.cancel()

        if not self._connect_task.cancel():
            _, writer = await self._connect_task
            writer.close()
            await writer.wait_closed()

    async def _send_heatbeat(self) -> None:
        while True:
            delta = time.time() - self._last_sent_message_time
            if delta < self.HEARTBEAT_INTERVAL:
                await asyncio.sleep(delta)
            else:
                await asyncio.shield(self._write_message(ProtoHeartbeatEvent()))

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

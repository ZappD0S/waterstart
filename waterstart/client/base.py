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
)
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Final, Optional, TypeVar, Union

from google.protobuf.message import Message

from ..observable import Observable
from ..openapi import ProtoHeartbeatEvent, ProtoMessage, ProtoOAErrorRes, messages_dict

S = TypeVar("S")
T = TypeVar("T", bound=Message)
U = TypeVar("U", bound=Message)
V = TypeVar("V", bound=Message)


class OpenApiErrorResException(Exception):
    def __init__(self, error_res: ProtoOAErrorRes):
        super().__init__(error_res.description)
        self.error_res = error_res


class BaseClient(Observable[Message], ABC):
    def __init__(self) -> None:
        super().__init__()
        self._lock_map: dict[type[Message], asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    @asynccontextmanager
    async def register_type(
        self, message_type: type[T], pred: Optional[Callable[[T], bool]] = None
    ) -> AsyncIterator[AsyncIterator[T]]:
        def nofilter(_: T) -> bool:
            return True

        def get_map_f(pred: Callable[[T], bool]):
            def map_f(x: Message):
                return x if isinstance(x, message_type) and pred(x) else None

            return map_f

        if pred is None:
            pred = nofilter

        async with self.register(get_map_f(pred)) as gen:
            yield gen

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
        def nofilter(_: T) -> bool:
            return True

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
            pred = nofilter

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
        res: Union[T, ProtoOAErrorRes, None] = None

        async with self._type_lock(res_type):
            await self._write_message(req)

            async for res in gen:
                break

        if res is None:
            raise RuntimeError()

        if isinstance(res, ProtoOAErrorRes):
            raise OpenApiErrorResException(res)

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
        keys_left = set(key_to_req)

        async with self._type_lock(res_type):
            tasks = {
                asyncio.create_task(self._write_message(req))
                for req in key_to_req.values()
            }

            await asyncio.wait(tasks)

            async for res in gen:
                if isinstance(res, ProtoOAErrorRes):
                    raise OpenApiErrorResException(res)

                key = get_key(res)

                if key not in keys_left:
                    raise RuntimeError()

                yield key, res
                keys_left.remove(key)

                if not keys_left:
                    break


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
class BaseReconnectingClient(BaseClient, ABC):
    HEARTBEAT_INTERVAL: Final[float] = 10.0
    RECONNECTION_DELAY: Final[float] = 1.0

    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> None:
        super().__init__()

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
                except Exception:
                    await asyncio.sleep(self.RECONNECTION_DELAY)

        reader, writer = await get_reader_writer()
        return HelperClient(reader, writer)

    async def _write_message(self, message: Message) -> None:
        while True:
            helper_client = await self._connect_task
            try:
                await helper_client.write_message(message)
                self._last_sent_t = time.time()
                return
            except Exception:
                self._connect_task = self._create_connect_task()
                raise

    async def _read_message(self) -> Message:
        while True:
            helper_client = await self._connect_task
            try:
                return await helper_client.read_message()
            except Exception:  # TODO: replace correct exception
                self._connect_task = self._create_connect_task()
                raise

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

            try:
                await self._write_message(ProtoHeartbeatEvent())
            except Exception:
                pass

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from asyncio import StreamReader, StreamWriter
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
)
from contextlib import AsyncExitStack, asynccontextmanager
from typing import (
    AsyncContextManager,
    Final,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from google.protobuf.message import Message

from ..observable import Observable
from ..openapi import ProtoHeartbeatEvent, ProtoMessage, ProtoOAErrorRes, messages_dict
from ..utils import ComposableAsyncIterable

S = TypeVar("S")
T = TypeVar("T", bound=Message)
U = TypeVar("U", bound=Message)
V = TypeVar("V", bound=Message)


class OpenApiErrorResException(Exception):
    def __init__(self, error_res: ProtoOAErrorRes):
        super().__init__(error_res.description)
        self.error_res = error_res


class BaseClient(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._lock_map: dict[type[Message], asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._exit_stack = AsyncExitStack()
        self._err_gen: Optional[ComposableAsyncIterable[ProtoOAErrorRes]] = None

    async def _get_err_gen(self) -> ComposableAsyncIterable[ProtoOAErrorRes]:
        if (err_gen := self._err_gen) is None:
            err_gen = await self._exit_stack.enter_async_context(
                self.register_type(ProtoOAErrorRes)
            )
            self._err_gen = err_gen

        return err_gen  # type: ignore

    @abstractmethod
    def register_type(
        self,
        message_type: type[T],
        pred: Optional[Callable[[T], bool]] = None,
        maxsize: int = 0,
    ) -> AsyncContextManager[ComposableAsyncIterable[T]]:
        ...

    async def _get_async_generator(self) -> AsyncGenerator[Message, None]:
        while True:
            yield await self._read_message()

    @abstractmethod
    async def _write_message(self, message: Message) -> None:
        ...

    @abstractmethod
    async def _read_message(self) -> Message:
        ...

    @asynccontextmanager
    async def _type_lock(self, res_type: type[T]):
        types = get_args(res_type) if get_origin(res_type) is Union else (res_type,)
        lock_map = self._lock_map

        new_lock_types: set[type[Message]] = set()
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self._global_lock)  # type: ignore

            for typ in types:
                if (lock := lock_map.get(typ)) is None:
                    new_lock_types.add(typ)
                    lock = lock_map[typ] = asyncio.Lock()

                await stack.enter_async_context(lock)  # type: ignore

            try:
                yield
            finally:
                for typ in new_lock_types:
                    del lock_map[typ]

    async def send_request(
        self,
        req: Message,
        res_type: type[T],
        res_gen: Optional[ComposableAsyncIterable[T]] = None,
        pred: Optional[Callable[[T], bool]] = None,
    ) -> T:
        err_gen = await self._get_err_gen()

        async with AsyncExitStack() as stack:
            if res_gen is None:
                res_gen = await stack.enter_async_context(
                    self.register_type(res_type, pred)
                )

            gen = res_gen | err_gen  # type: ignore
            res = await self._send_request(req, res_type, gen)

        return res

    async def _send_request(
        self,
        req: Message,
        res_type: type[T],
        gen: AsyncIterable[Union[T, ProtoOAErrorRes]],
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
        get_key: Callable[[T], Optional[S]],
        res_gen: Optional[ComposableAsyncIterable[T]] = None,
    ) -> AsyncIterator[tuple[S, T]]:
        err_gen = await self._get_err_gen()

        async with AsyncExitStack() as stack:
            if res_gen is None:
                res_gen = await stack.enter_async_context(self.register_type(res_type))

            gen = res_gen | err_gen  # type: ignore

            async for key_res in self._send_requests(
                key_to_req, res_type, get_key, gen
            ):
                yield key_res

    async def _send_requests(
        self,
        key_to_req: Mapping[S, Message],
        res_type: type[T],
        get_key: Callable[[T], Optional[S]],
        gen: AsyncIterable[Union[T, ProtoOAErrorRes]],
    ) -> AsyncIterator[tuple[S, T]]:
        if not key_to_req:
            return

        keys_left = set(key_to_req)

        async with self._type_lock(res_type):
            aws = (self._write_message(req) for req in key_to_req.values())
            await asyncio.gather(*aws)

            async for res in gen:
                if isinstance(res, ProtoOAErrorRes):
                    raise OpenApiErrorResException(res)

                key = get_key(res)

                if key is None:
                    continue

                if key not in keys_left:
                    raise RuntimeError()

                yield key, res
                keys_left.remove(key)

                if not keys_left:
                    break

    async def aclose(self) -> None:
        await self._exit_stack.aclose()


class HelperClient(BaseClient, Observable[Message]):
    _payloadtype_to_messageproto: Mapping[int, type[Message]] = {
        proto.payloadType.DESCRIPTOR.default_value: proto  # type: ignore
        for proto in messages_dict.values()
        if hasattr(proto, "payloadType")
    }

    def __init__(self, reader: StreamReader, writer: StreamWriter) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer

    @asynccontextmanager
    async def register_type(
        self,
        message_type: type[T],
        pred: Optional[Callable[[T], bool]] = None,
        maxsize: int = 0,
    ) -> AsyncIterator[ComposableAsyncIterable[T]]:
        def get_map_f_with_pred(pred: Callable[[T], bool]):
            def map_f(x: Message) -> Optional[T]:
                return x if isinstance(x, types) and pred(x) else None  # type: ignore

            return map_f

        def map_f_without_pred(x: Message) -> Optional[T]:
            return x if isinstance(x, types) else None  # type: ignore

        types = (
            get_args(message_type)
            if get_origin(message_type) is Union
            else (message_type,)
        )

        map_f = map_f_without_pred if pred is None else get_map_f_with_pred(pred)

        async with AsyncExitStack() as stack:
            gen: AsyncIterator[T] = await stack.enter_async_context(
                self.register(map_f, maxsize)
            )

            comp_gen = await stack.enter_async_context(
                ComposableAsyncIterable.from_it(gen)
            )

            yield comp_gen

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

    async def aclose(self) -> None:
        await super().aclose()
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

    @asynccontextmanager
    async def register_type(
        self,
        message_type: type[T],
        pred: Optional[Callable[[T], bool]] = None,
        maxsize: int = 0,
    ) -> AsyncIterator[ComposableAsyncIterable[T]]:
        helper_client = await self._connect_task
        async with helper_client.register_type(message_type, pred, maxsize) as gen:
            yield gen

    async def _write_message(self, message: Message) -> None:
        while True:
            helper_client = await self._connect_task
            try:
                await helper_client.write_message(message)
                self._last_sent_t = time.time()
                return
            except Exception:
                self._connect_task = self._create_connect_task()

    async def _read_message(self) -> Message:
        while True:
            helper_client = await self._connect_task
            try:
                return await helper_client.read_message()
            except Exception:  # TODO: replace correct exception
                self._connect_task = self._create_connect_task()

    async def aclose(self) -> None:
        await super().aclose()

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

        await helper_client.aclose()

    async def _send_heatbeat(self) -> None:
        while True:
            if (delta := time.time() - self._last_sent_t) < self.HEARTBEAT_INTERVAL:
                await asyncio.sleep(delta)
                continue

            try:
                await self._write_message(ProtoHeartbeatEvent())
            except Exception:
                pass

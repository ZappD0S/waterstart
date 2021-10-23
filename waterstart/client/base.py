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


class BaseClient(Observable[Message], ABC):
    _payloadtype_to_messageproto: Mapping[int, type[Message]] = {
        proto.payloadType.DESCRIPTOR.default_value: proto  # type: ignore
        for proto in messages_dict.values()
        if hasattr(proto, "payloadType")
    }

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

        return err_gen

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
            gen = await stack.enter_async_context(self.register(map_f, maxsize))

            yield await stack.enter_async_context(ComposableAsyncIterable.from_it(gen))

    async def _get_async_generator(self) -> AsyncGenerator[Message, None]:
        while True:
            yield await self._read_message()

    @abstractmethod
    async def _write_message(self, message: Message) -> None:
        ...

    @abstractmethod
    async def _read_message(self) -> Message:
        ...

    @classmethod
    def _parse_message(cls, data: bytes) -> Message:
        protomessage = ProtoMessage.FromString(data)
        messageproto = cls._payloadtype_to_messageproto[protomessage.payloadType]
        return messageproto.FromString(protomessage.payload)

    @classmethod
    async def _read_message_from_reader(cls, reader: StreamReader) -> Message:
        length_data = await reader.readexactly(4)
        length = int.from_bytes(length_data, byteorder="big")

        if length <= 0:
            raise RuntimeError()

        payload_data = await reader.readexactly(length)
        return cls._parse_message(payload_data)

    @staticmethod
    async def _write_message_to_writer(writer: StreamWriter, message: Message) -> None:
        protomessage = ProtoMessage(
            payloadType=message.payloadType,  # type: ignore
            payload=message.SerializeToString(),
        )
        payload_data = protomessage.SerializeToString()
        length_data = len(payload_data).to_bytes(4, byteorder="big")
        writer.write(length_data + payload_data)
        await writer.drain()

    @asynccontextmanager
    async def _type_lock(self, res_type: type[T]):
        types = get_args(res_type) if get_origin(res_type) is Union else (res_type,)
        lock_map = self._lock_map

        new_lock_types: set[type[Message]] = set()
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(self._global_lock)  # type: ignore

            for t in types:
                if (lock := lock_map.get(t)) is None:
                    new_lock_types.add(t)
                    lock = lock_map[t] = asyncio.Lock()

                await stack.enter_async_context(lock)  # type: ignore

            try:
                yield
            finally:
                for t in new_lock_types:
                    del lock_map[t]

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

            gen = res_gen | err_gen
            return await self._send_request(req, res_type, gen)

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

            gen = res_gen | err_gen

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
        await super().aclose()


class HelperClient(BaseClient):
    def __init__(self, reader: StreamReader, writer: StreamWriter) -> None:
        super().__init__()
        self._reader = reader
        self._writer = writer

    async def _write_message(self, message: Message) -> None:
        await self._write_message_to_writer(self._writer, message)

    async def _read_message(self) -> Message:
        return await self._read_message_from_reader(self._reader)


# TODO: add task that listens to ProtoOAClientDisconnectEvent and
# reconnects when it happens..
class BaseReconnectingClient(BaseClient):
    HEARTBEAT_INTERVAL: Final[float] = 10.0
    RECONNECTION_DELAY: Final[float] = 1.0

    def __init__(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> None:
        super().__init__()

        def create_connect_task() -> asyncio.Task[tuple[StreamReader, StreamWriter]]:
            return asyncio.create_task(
                self._connect(open_connection), name="get_reader_writer"
            )

        self._create_connect_task = create_connect_task
        self._connect_task: asyncio.Task[tuple[StreamReader, StreamWriter]]
        self._connect_task = create_connect_task()

        self._last_sent_t = 0.0
        self._heartbeat_task = asyncio.create_task(
            self._send_heatbeat(), name="heartbeat"
        )

    @abstractmethod
    async def _setup_connection(self, helper_client: HelperClient) -> None:
        ...

    async def _connect(
        self,
        open_connection: Callable[[], Awaitable[tuple[StreamReader, StreamWriter]]],
    ) -> tuple[StreamReader, StreamWriter]:
        while True:
            try:
                reader, writer = await open_connection()
            except Exception:
                await asyncio.sleep(self.RECONNECTION_DELAY)
                continue

            helper_client = HelperClient(reader, writer)

            failed = False

            try:
                await self._setup_connection(helper_client)
            except Exception:  # TODO: correct exception...
                failed = True
                continue
            finally:
                await helper_client.aclose()

                if failed:
                    writer.close()
                    await writer.wait_closed()
                    await asyncio.sleep(self.RECONNECTION_DELAY)

            return reader, writer

    async def _write_message(self, message: Message) -> None:
        while True:
            _, writer = await self._connect_task
            try:
                await self._write_message_to_writer(writer, message)
                self._last_sent_t = time.time()
                return
            except Exception:
                self._connect_task = self._create_connect_task()

    async def _read_message(self) -> Message:
        while True:
            reader, _ = await self._connect_task

            try:
                return await self._read_message_from_reader(reader)
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
            _, writer = await self._connect_task
        except asyncio.CancelledError:
            return

        writer.close()
        await writer.wait_closed()

    async def _send_heatbeat(self) -> None:
        while True:
            if (delta := time.time() - self._last_sent_t) < self.HEARTBEAT_INTERVAL:
                await asyncio.sleep(delta)
                continue

            try:
                await self._write_message(ProtoHeartbeatEvent())
            except Exception:
                pass

from __future__ import annotations

import asyncio
import time
from asyncio import Future, StreamReader, StreamWriter
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Optional, TypeVar, Union

from google.protobuf.message import Message

from .openapi import ProtoHeartbeatEvent, ProtoMessage, ProtoOAErrorRes, messages_dict

T = TypeVar("T", bound=Message)


class OpenApiClient:
    def __init__(
        self,
        reader: StreamReader,
        writer: StreamWriter,
        heartbeat_interval: float = 10.0,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self._last_sent_message_time = 0.0
        self._payloadtype_to_messageproto = {
            proto.payloadType.DESCRIPTOR.default_value: proto  # type: ignore
            for proto in messages_dict.values()
            if hasattr(proto, "payloadType")
        }
        self._writer_lock = asyncio.Lock()
        self._setters: list[Callable[[Message], Coroutine]] = []
        self._heartbeat_task = asyncio.create_task(
            self._send_heatbeat(heartbeat_interval)
        )
        self._dispatch_task = asyncio.create_task(self._dispatch_messages())

    async def send_message(self, message: Message) -> None:
        protomessage = ProtoMessage(
            payloadType=message.payloadType,  # type: ignore
            payload=message.SerializeToString(),
        )
        payload_data = protomessage.SerializeToString()
        length_data = len(payload_data).to_bytes(4, byteorder="big")
        async with self._writer_lock:
            self.writer.write(length_data + payload_data)
        await self.writer.drain()
        self._last_sent_message_time = time.time()

    def _parse_message(self, data: bytes) -> Message:
        protomessage = ProtoMessage.FromString(data)
        messageproto = self._payloadtype_to_messageproto[protomessage.payloadType]
        return messageproto.FromString(protomessage.payload)

    async def _read_message(self) -> Optional[Message]:
        length_data = await self.reader.readexactly(4)
        length = int.from_bytes(length_data, byteorder="big")
        if length <= 0:
            return None

        payload_data = await self.reader.readexactly(length)
        return self._parse_message(payload_data)

    async def _dispatch_messages(self) -> None:
        while True:
            message = await self._read_message()
            if message is None:
                continue

            if not self._setters:
                continue

            tasks = [asyncio.create_task(setter(message)) for setter in self._setters]
            await asyncio.wait(tasks)

    @staticmethod
    def _build_generator_and_setter(
        message_type: type[T], pred: Callable[[T], bool]
    ) -> tuple[
        Callable[[Message], Coroutine], AsyncGenerator[Union[T, ProtoOAErrorRes], None]
    ]:
        loop = asyncio.get_running_loop()
        future: Future[Message] = loop.create_future()
        sem = asyncio.Semaphore()

        async def set_result(val: Message):
            await sem.acquire()
            future.set_result(val)

        async def generator():
            nonlocal future

            while True:
                val = await future

                future = loop.create_future()
                sem.release()

                if isinstance(val, ProtoOAErrorRes) or (
                    isinstance(val, message_type) and pred(val)
                ):
                    yield val

        return set_result, generator()

    def register(
        self, message_type: type[T], pred: Optional[Callable[[T], bool]] = None
    ) -> AsyncContextManager[AsyncIterator[Union[T, ProtoOAErrorRes]]]:
        @asynccontextmanager
        async def _get_contextmanager():
            self._setters.append(setter)
            try:
                yield gen
            finally:
                await gen.aclose()
                self._setters.remove(setter)

        if pred is None:
            pred = lambda _: True

        setter, gen = self._build_generator_and_setter(message_type, pred)
        return _get_contextmanager()

    async def close(self) -> None:
        self._heartbeat_task.cancel()
        self._dispatch_task.cancel()
        self.writer.close()
        await self.writer.wait_closed()

    async def _send_heatbeat(self, heartbeat_interval: float) -> None:
        while True:
            delta = time.time() - self._last_sent_message_time
            if delta < heartbeat_interval:
                await asyncio.sleep(delta)
            else:
                await asyncio.shield(self.send_message(ProtoHeartbeatEvent()))

    @staticmethod
    async def create(host: str, port: int) -> OpenApiClient:
        reader, writer = await asyncio.open_connection(host, port, ssl=True)
        return OpenApiClient(reader, writer)

from __future__ import annotations

import asyncio
import time
from asyncio import StreamReader, StreamWriter
from typing import AsyncIterator, Callable, Coroutine, List, Optional, Tuple

from google.protobuf.message import Message

from .openapi import ProtoHeartbeatEvent, ProtoMessage, messages_dict


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
            proto.payloadType.DESCRIPTOR.default_value: proto
            for proto in messages_dict.values()
            if hasattr(proto, "payloadType")
        }
        self._writer_lock = asyncio.Lock()
        self._setters: List[Callable[[Message], Coroutine]] = []
        self._heartbeat_task = asyncio.create_task(
            self._send_heatbeat(heartbeat_interval)
        )
        self._dispatch_task = asyncio.create_task(self._dispatch_messages())

    async def send_message(self, message: Message) -> None:
        protomessage = ProtoMessage(
            payloadType=message.payloadType, payload=message.SerializeToString()
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
        pred: Callable[[Message], bool]
    ) -> Tuple[Callable[[Message], Coroutine], AsyncIterator[Message]]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        sem = asyncio.Semaphore()

        async def set_result(val: Message) -> None:
            await sem.acquire()
            future.set_result(val)

        async def generator() -> AsyncIterator[Message]:
            nonlocal future

            while True:
                val = await future

                future = loop.create_future()
                sem.release()

                if pred(val):
                    yield val

        return set_result, generator()

    def register(self, cond: Callable[[Message], bool]):
        setter, gen = self._build_generator_and_setter(cond)
        self._setters.append(setter)
        return gen

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

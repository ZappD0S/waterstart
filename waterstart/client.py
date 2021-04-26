from __future__ import annotations

import asyncio
import time
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncIterator, Callable, Mapping
from typing import AsyncContextManager, Optional, TypeVar, Union, overload

from google.protobuf.message import Message

from .observable import Observable
from .openapi import ProtoHeartbeatEvent, ProtoMessage, ProtoOAErrorRes, messages_dict

S = TypeVar("S")
T = TypeVar("T", bound=Message)
U = TypeVar("U", bound=Message)
V = TypeVar("V", bound=Message)


class OpenApiClient(Observable[Message]):
    def __init__(
        self,
        reader: StreamReader,
        writer: StreamWriter,
        heartbeat_interval: float = 10.0,
    ) -> None:
        super().__init__()
        self.reader = reader
        self.writer = writer
        self._last_sent_message_time = 0.0
        self._payloadtype_to_messageproto = {
            proto.payloadType.DESCRIPTOR.default_value: proto  # type: ignore
            for proto in messages_dict.values()
            if hasattr(proto, "payloadType")
        }
        self._heartbeat_task = asyncio.create_task(
            self._send_heatbeat(heartbeat_interval)
        )

    async def send_message(self, message: Message) -> None:
        protomessage = ProtoMessage(
            payloadType=message.payloadType,  # type: ignore
            payload=message.SerializeToString(),
        )
        payload_data = protomessage.SerializeToString()
        length_data = len(payload_data).to_bytes(4, byteorder="big")
        self.writer.write(length_data + payload_data)
        await self.writer.drain()
        self._last_sent_message_time = time.time()

    async def send_and_wait_response(self, req: Message, res_type: type[T]) -> T:
        res: Union[T, ProtoOAErrorRes, None] = None
        async with self.register_types((res_type, ProtoOAErrorRes)) as gen:
            await self.send_message(req)
            async for res in gen:
                break

        if res is None:
            raise RuntimeError()

        if isinstance(res, ProtoOAErrorRes):
            raise RuntimeError(f"{res.errorCode}: {res.description}")

        return res

    async def send_and_wait_responses(
        self,
        id_to_req: Mapping[S, Message],
        res_type: type[T],
        get_key: Callable[[T], S],
    ) -> AsyncIterator[tuple[S, T]]:
        keys_left = set(id_to_req)

        async with self.register_types((res_type, ProtoOAErrorRes)) as gen:
            tasks = [
                asyncio.create_task(self.send_message(req))
                for req in id_to_req.values()
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

    async def _read_message(self) -> Optional[Message]:
        length_data = await self.reader.readexactly(4)
        length = int.from_bytes(length_data, byteorder="big")
        if length <= 0:
            return None

        payload_data = await self.reader.readexactly(length)
        return self._parse_message(payload_data)

    async def _get_async_iterator(self) -> AsyncIterator[Message]:
        while True:
            message = await self._read_message()
            if message is not None:
                yield message

    @overload
    def register_types(
        self, message_type: type[T]
    ) -> AsyncContextManager[AsyncIterator[T]]:
        ...

    @overload
    def register_types(
        self,
        message_type: tuple[type[T], type[U]],
    ) -> AsyncContextManager[AsyncIterator[Union[T, U]]]:
        ...

    @overload
    def register_types(
        self,
        message_type: tuple[type[T], type[U], type[V]],
    ) -> AsyncContextManager[AsyncIterator[Union[T, U, V]]]:
        ...

    def register_types(self, message_type) -> AsyncContextManager[AsyncIterator]:
        def func(x: Message):
            return x if isinstance(x, message_type) else None

        return self.register(func)

    async def close(self) -> None:
        await super().close()

        self._heartbeat_task.cancel()
        try:
            await self._heartbeat_task
        except asyncio.CancelledError:
            pass

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

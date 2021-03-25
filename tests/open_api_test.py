from __future__ import annotations

import asyncio
import time
from asyncio import StreamReader, StreamWriter
from typing import Mapping, Optional

import openapi.OpenApiCommonMessages_pb2
import openapi.OpenApiCommonModelMessages_pb2
import openapi.OpenApiMessages_pb2
import openapi.OpenApiModelMessages_pb2
from google.protobuf import descriptor_pb2, message_factory
from google.protobuf.message import Message
from openapi.OpenApiCommonMessages_pb2 import ProtoMessage

HOST = "demo.ctraderapi.com"
PORT = 5035
# HEARTBEAT_INTERVAL = 10.0


def get_payloadtype_to_messageproto() -> Mapping[int, Message]:
    pb2_list = [
        openapi.OpenApiCommonMessages_pb2,
        openapi.OpenApiCommonModelMessages_pb2,
        openapi.OpenApiMessages_pb2,
        openapi.OpenApiModelMessages_pb2,
    ]
    fd_protos = [descriptor_pb2.FileDescriptorProto.FromString(pb2.DESCRIPTOR.serialized_pb) for pb2 in pb2_list]
    messages_dict = message_factory.GetMessages(fd_protos)
    return {
        proto.payloadType.DESCRIPTOR.default_value: proto
        for proto in messages_dict.values()
        if hasattr(proto, "payloadType")
    }


class App:
    def __init__(self, reader: StreamReader, writer: StreamWriter, heartbeat_interval: float = 10.0) -> None:
        self.reader = reader
        self.writer = writer
        self._stop_heartbeat: bool = False
        self._heartbeat_task = asyncio.create_task(self._send_heatbeat(heartbeat_interval))
        self._last_sent_message_time: float = 0.0
        self._payloadtype_to_messageproto = get_payloadtype_to_messageproto()

    def send_message(self, message: Message) -> None:
        protomessage = ProtoMessage(payloadType=message.payloadType, payload=message.SerializeToString())
        payload_data = protomessage.SerializeToString()
        length_data = len(payload_data).to_bytes(4, byteorder="big")
        self.writer.write(length_data + payload_data)
        self._last_sent_message_time = time.time()

    def parse_message(self, data: bytes) -> Message:
        protomessage = ProtoMessage.FromString(data)
        messageproto = self._payloadtype_to_messageproto[protomessage.payloadType]
        return messageproto.FromString(protomessage.payload)

    async def read_message(self) -> Optional[Message]:
        length_data = await self.reader.readexactly(4)
        length = int.from_bytes(length_data, byteorder="big")
        if length <= 0:
            return None

        payload_data = await self.reader.readexactly(length)
        return self.parse_message(payload_data)

    async def close(self) -> None:
        self._stop_heartbeat = True
        await self._heartbeat_task
        self.writer.drain()
        self.writer.close()

    async def _send_heatbeat(self, heartbeat_interval) -> None:
        while not self.stop_heartbeat:
            delta = time.time() - self._last_sent_message_time
            if delta < heartbeat_interval:
                await asyncio.sleep(delta)
            else:
                self.send_message(openapi.OpenApiCommonMessages_pb2.ProtoHeartbeatEvent())

    @staticmethod
    async def create(host: str, port: int) -> App:
        reader, writer = await asyncio.open_connection(host, port, ssl=True)
        return App(reader, writer)

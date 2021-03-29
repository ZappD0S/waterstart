from typing import Mapping

from google.protobuf import message_factory
from google.protobuf.descriptor_pb2 import FileDescriptorProto
from google.protobuf.message import Message

from . import (
    OpenApiCommonMessages_pb2,
    OpenApiCommonModelMessages_pb2,
    OpenApiMessages_pb2,
    OpenApiModelMessages_pb2,
)


def _get_messages_dict() -> Mapping[int, Message]:
    pb2_list = [
        OpenApiCommonMessages_pb2,
        OpenApiCommonModelMessages_pb2,
        OpenApiMessages_pb2,
        OpenApiModelMessages_pb2,
    ]
    return message_factory.GetMessages(
        FileDescriptorProto.FromString(pb2.DESCRIPTOR.serialized_pb) for pb2 in pb2_list
    )


messages_dict = _get_messages_dict()
globals().update(messages_dict)

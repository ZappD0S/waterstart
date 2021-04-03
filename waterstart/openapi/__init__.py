from google.protobuf.message import Message

from .OpenApiCommonMessages_pb2 import *
from .OpenApiCommonModelMessages_pb2 import *
from .OpenApiMessages_pb2 import *
from .OpenApiModelMessages_pb2 import *


messages_dict = {
    k: v for k, v in vars().items() if isinstance(v, type) and issubclass(v, Message)
}

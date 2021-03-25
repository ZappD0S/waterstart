from google.protobuf.message_factory import MessageFactory
from google.protobuf.message import Message
import openapi.OpenApiMessages_pb2
import openapi.OpenApiModelMessages_pb2
import openapi.OpenApiCommonModelMessages_pb2
import openapi.OpenApiCommonMessages_pb2


def get_payloadType(proto):
    return proto.payloadType.DESCRIPTOR.default_value


message_types = {
    get_payloadType(v): v
    for v in vars(openapi.OpenApiMessages_pb2).values()
    if isinstance(v, type) and issubclass(v, Message)
}

# message_types_descriptors = openapi.OpenApiMessages_pb2.DESCRIPTOR.message_types_by_name.values()
# message_factory = MessageFactory()
# message_types = {
#     get_payloadType(proto := message_factory.GetPrototype(descriptor)): proto
#     for descriptor in message_types_descriptors
# }

trendbar_req_in = openapi.OpenApiMessages_pb2.ProtoOAGetTrendbarsReq()
trendbar_req_in.ctidTraderAccountId = 1234
trendbar_req_in.symbolId = 1
trendbar_req_in.fromTimestamp = 110232
trendbar_req_in.toTimestamp = 120243
trendbar_req_in.period = 1
trendbar_req_in.payloadType = trendbar_req_in.payloadType
# trendbar_req.ClearField("payloadType")

data = trendbar_req_in.SerializeToString()

message = openapi.OpenApiCommonMessages_pb2.ProtoMessage.FromString(data)
messsage_type = message_types[message.payloadType]

trendbar_req_out = messsage_type.FromString(data)

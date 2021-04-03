protoc \
    --plugin=protoc-gen-mypy=$(which protoc-gen-mypy) \
    -I=Open-API-2.0-protobuf-messages/7.1 \
    --python_out=waterstart/openapi \
    --mypy_out=waterstart/openapi \
    Open-API-2.0-protobuf-messages/7.1/*.proto

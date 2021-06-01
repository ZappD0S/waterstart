#!/bin/bash

files=(Open-API-2.0-protobuf-messages/*/)
proto_path="${files[0]}"

protoc \
    --plugin=protoc-gen-mypy="$(which protoc-gen-mypy)" \
    -I="$proto_path" \
    --python_out=waterstart/openapi \
    --mypy_out=waterstart/openapi \
    "$proto_path"/*.proto

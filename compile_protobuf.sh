#!/bin/bash

proto_path="openapi-proto-messages/"

protoc \
    --plugin=protoc-gen-mypy="$(which protoc-gen-mypy)" \
    -I="$proto_path" \
    --python_out=waterstart/openapi \
    --mypy_out=waterstart/openapi \
    "$proto_path"/*.proto

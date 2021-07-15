from collections import Iterator, Mapping, Set
from typing import TypeVar, Any, overload

from .base_mapper import BaseArrayMapper, FieldData


T = TypeVar("T")
U = TypeVar("U")


class DictBasedArrayMapper(BaseArrayMapper[Mapping[T, float]]):
    def __init__(self, blueprint_map: Mapping[T, FieldData]) -> None:
        super().__init__(set(blueprint_map.values()))
        self._inds_map = {key: field.index for key, field in blueprint_map.items()}

    @property
    def keys(self) -> Set[T]:
        return self._inds_map.keys()

    @overload
    def iterate_index_to_value(
        self, value: Mapping[T, float]
    ) -> Iterator[tuple[int, float]]:
        ...

    @overload
    def iterate_index_to_value(
        self,
        value: Mapping[T, U],
    ) -> Iterator[tuple[int, U]]:
        ...

    def iterate_index_to_value(
        self, value: Mapping[T, Any]
    ) -> Iterator[tuple[int, Any]]:
        for key, index in self._inds_map.items():
            yield index, value[key]

    @overload
    def iterate_index_to_value_partial(
        self, value: Mapping[T, float]
    ) -> Iterator[tuple[int, float]]:
        ...

    @overload
    def iterate_index_to_value_partial(
        self,
        value: Mapping[T, U],
    ) -> Iterator[tuple[int, U]]:
        ...

    def iterate_index_to_value_partial(
        self, value: Mapping[T, Any]
    ) -> Iterator[tuple[int, Any]]:
        for key, index in self._inds_map.items():
            if key in value:
                yield index, value[key]

    @overload
    def build_from_index_to_value_map(
        self, mapping: Mapping[int, float]
    ) -> Mapping[T, float]:
        ...

    @overload
    def build_from_index_to_value_map(self, mapping: Mapping[int, U]) -> Mapping[T, U]:
        ...

    def build_from_index_to_value_map(
        self, mapping: Mapping[int, Any]
    ) -> Mapping[T, Any]:
        return {key: mapping[index] for key, index in self._inds_map.items()}

    @overload
    def build_from_index_to_value_map_partial(
        self, mapping: Mapping[int, float]
    ) -> Mapping[T, float]:
        ...

    @overload
    def build_from_index_to_value_map_partial(
        self, mapping: Mapping[int, U]
    ) -> Mapping[T, U]:
        ...

    def build_from_index_to_value_map_partial(
        self, mapping: Mapping[int, Any]
    ) -> Mapping[T, Any]:
        return {
            key: mapping[index]
            for key, index in self._inds_map.items()
            if index in mapping
        }

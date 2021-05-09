from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Set
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class FieldData:
    index: int

class BaseArrayMapper(ABC, Generic[T]):
    def __init__(self, fields_set: Set[FieldData]) -> None:
        super().__init__()

        self._fields_set: Set[FieldData] = fields_set
        indices = [field.index for field in fields_set]

        if not indices:
            raise ValueError()

        if indices[0] != 0:
            raise ValueError()

        if not all(b - a == 1 for a, b in zip(indices[:-1], indices[1:])):
            raise ValueError()

    @property
    def n_fields(self) -> int:
        return len(self._fields_set)

    @abstractmethod
    def iterate_index_to_value(self, value: T) -> Iterator[tuple[int, float]]:
        ...

    @abstractmethod
    def build_from_index_to_value_map(self, mapping: Mapping[int, float]) -> T:
        ...

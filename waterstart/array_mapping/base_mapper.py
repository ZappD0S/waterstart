from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..utils import is_contiguous

T = TypeVar("T")


@dataclass
class FieldData:
    index: int


class BaseArrayMapper(ABC, Generic[T]):
    def __init__(self, fields: Sequence[FieldData]) -> None:
        super().__init__()

        if not fields:
            raise ValueError()

        n_fields = len(fields)
        indices = sorted({field.index for field in fields})

        if len(indices) < n_fields:
            raise ValueError()

        if indices[0] != 0:
            raise ValueError()

        if not is_contiguous(indices):
            raise ValueError()

        self._n_fields = n_fields
        self._fields = fields

    @property
    def n_fields(self) -> int:
        return self._n_fields

    @abstractmethod
    def iterate_index_to_value(self, value: T) -> Iterator[tuple[int, float]]:
        ...

    @abstractmethod
    def build_from_index_to_value_map(self, mapping: Mapping[int, float]) -> T:
        ...

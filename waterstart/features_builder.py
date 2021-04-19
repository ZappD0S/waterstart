from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableSet, Set
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar, Union, cast


@dataclass(frozen=True)
class Field:
    index: int


@dataclass(frozen=True)
class PriceField(Field):
    is_close: bool = False
    price_group: Optional[str] = None


T = TypeVar("T", bound=Enum)


# class ValuesMapBuilder(ABC, Generic[T]):
#     @abstractmethod
#     def build(self) -> Mapping[T, float]:
#         ...


class FeauturesSpec(Generic[T]):
    def __init__(self, enum_type: type[T], fields_map: Mapping[T, Field]) -> None:
        super().__init__()

        # TODO: without the cast it would be Set[Enum]. Is this a bug?
        self._enum_values = cast(Set[T], set(enum_type))
        if not self._enum_values:
            raise ValueError()

        if not fields_map.keys() == self._enum_values:
            raise ValueError()

        self._fields_map = fields_map

        fields = fields_map.values()
        fields_set = set(fields)

        if fields_set != fields:
            raise ValueError()

        # TODO: make a property for this
        self._price_groups = list(self._get_price_groups(fields_set))

        indices = sorted(field.index for field in fields)

        if indices[0] != 0:
            raise ValueError()

        if len(indices) < 2:
            return

        if not all(b - a == 1 for a, b in zip(indices[:-1], indices[1:])):
            raise ValueError()

    @property
    def price_groups(self):
        return self._price_groups

    @staticmethod
    def _get_price_groups(
        fields_set: Set[Field],
    ) -> Iterator[tuple[PriceField, Set[PriceField]]]:

        builder: dict[
            Union[str, Field], tuple[Optional[PriceField], MutableSet[PriceField]]
        ] = {}

        for field in fields_set:
            if not isinstance(field, PriceField):
                continue

            key = field.price_group or field
            close_price, group_prices = builder.get(key, (None, set()))
            if field.is_close:
                if close_price is not None:
                    raise ValueError()

                close_price = field

            if field in group_prices:
                raise ValueError()

            group_prices.add(field)
            builder[key] = (close_price, group_prices)

        for close_price, group_prices in builder.values():
            if close_price is None:
                try:
                    [close_price] = group_prices
                except ValueError:
                    raise ValueError()

            assert close_price in group_prices
            yield close_price, group_prices

    def get_inds_and_values(
        self, values_map: Mapping[T, float]
    ) -> Iterator[tuple[int, float]]:
        for enum_val in self._enum_values:
            index = self._fields_map[enum_val].index

            try:
                value = values_map[enum_val]
            except KeyError:
                raise ValueError()

            yield index, value

from collections.abc import Iterator, Mapping, MutableSet, Set
from dataclasses import dataclass, field
from typing import Generic, Optional, Sequence, TypeVar, Union

# TODO: rename file


@dataclass(frozen=True)
class Field:
    index: int


T = TypeVar("T")
U = TypeVar("U")


class KeyIndexMapper(Generic[T]):
    def __init__(self, fields_map: Mapping[T, Field]) -> None:
        super().__init__()

        self._fields_map = fields_map

        fields = fields_map.values()
        fields_set = set(fields)

        if fields_set != fields:
            raise ValueError()

        self._key_to_index_map = {key: field.index for key, field in fields_map.items()}

        indices = list(self._key_to_index_map.values())

        if not indices:
            raise ValueError()

        if indices[0] != 0:
            raise ValueError()

        if not all(b - a == 1 for a, b in zip(indices[:-1], indices[1:])):
            raise ValueError()

    @property
    def keys(self) -> Set[T]:
        return self._fields_map.keys()

    def map_inds_to_keys(self, values_map: Mapping[int, U]) -> Iterator[tuple[T, U]]:
        for key, index in self._key_to_index_map.items():

            try:
                value = values_map[index]
            except KeyError:
                raise ValueError()

            yield key, value

    def map_keys_to_inds(self, values_map: Mapping[T, U]) -> Iterator[tuple[int, U]]:
        for key, index in self._key_to_index_map.items():

            try:
                value = values_map[key]
            except KeyError:
                raise ValueError()

            yield index, value


@dataclass(frozen=True)
class PriceField(Field):
    is_close: bool = field(default=False, hash=False)
    price_group: Optional[str] = field(default=None, hash=False)


class FeautureVectorMapper(KeyIndexMapper[T]):
    def __init__(self, fields_map: Mapping[T, Field]) -> None:
        super().__init__(fields_map)

        self._price_index_groups = [
            (close_price_field.index, [field.index for field in fields_group])
            for close_price_field, fields_group in self._get_price_field_groups(
                set(fields_map.values())
            )
        ]

    @property
    def price_index_groups(self) -> Sequence[tuple[int, list[int]]]:
        return self._price_index_groups

    @staticmethod
    def _get_price_field_groups(
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

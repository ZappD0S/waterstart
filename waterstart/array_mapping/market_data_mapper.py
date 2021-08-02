from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, TypeVar

from ..price import MarketData, SymbolData, TrendBar
from .base_mapper import BaseArrayMapper, FieldData


@dataclass
class PriceFieldData(FieldData):
    price_group_id: int
    is_close_price: bool


# TODO: maybe instead of using FieldData we take MarketData[int], where int represent
# the inds and we have an internal method that returns an object that contains
# the information we need for each field (symbol info, is price, is close price..)
# this method would replace _flatten_fields

T = TypeVar("T")


class MarketDataArrayMapper(BaseArrayMapper[MarketData[float]]):
    def __init__(self, blueprint: MarketData[FieldData]) -> None:
        super().__init__(list(self._flatten_fields(blueprint)))
        self._blueprint = blueprint
        self._scaling_idxs = list(self._build_scaling_inds(self._fields))

    # TODO: find better name
    @property
    def scaling_idxs(self) -> Sequence[tuple[int, list[int]]]:
        return self._scaling_idxs

    @staticmethod
    def _to_tuple(
        sym_data: SymbolData[T],
    ) -> tuple[TrendBar[T], TrendBar[T], TrendBar[T], TrendBar[T]]:
        return (
            sym_data.price_trendbar,
            sym_data.spread_trendbar,
            sym_data.base_to_dep_trendbar,
            sym_data.quote_to_dep_trendbar,
        )

    def iterate_index_to_value(
        self, value: MarketData[float]
    ) -> Iterator[tuple[int, float]]:
        for sym_id, blueprint_sym_data in self._blueprint.symbols_data_map.items():
            sym_data = value.symbols_data_map[sym_id]

            it = zip(self._to_tuple(blueprint_sym_data), self._to_tuple(sym_data))

            for blueprint_tb, tb in it:
                yield blueprint_tb.high.index, tb.high
                yield blueprint_tb.low.index, tb.low
                yield blueprint_tb.close.index, tb.close

        yield self._blueprint.time_of_day.index, value.time_of_day
        yield self._blueprint.delta_to_last.index, value.delta_to_last

    def build_from_index_to_value_map(
        self, mapping: Mapping[int, float]
    ) -> MarketData[float]:
        sym_data_map: dict[int, SymbolData[float]] = {}

        for sym_id, blueprint_sym_data in self._blueprint.symbols_data_map.items():
            blueprint_tbs = self._to_tuple(blueprint_sym_data)

            tbs = [
                TrendBar(
                    high=mapping[blueprint_tb.high.index],
                    low=mapping[blueprint_tb.low.index],
                    close=mapping[blueprint_tb.close.index],
                )
                for blueprint_tb in blueprint_tbs
            ]
            sym_data_map[sym_id] = SymbolData(*tbs)

        return MarketData(
            sym_data_map,
            time_of_day=mapping[self._blueprint.time_of_day.index],
            delta_to_last=mapping[self._blueprint.delta_to_last.index],
        )

    @classmethod
    def _flatten_fields(cls, blueprint: MarketData[FieldData]) -> Iterator[FieldData]:
        for blueprint_sym_data in blueprint.symbols_data_map.values():
            blueprint_tbs = cls._to_tuple(blueprint_sym_data)

            for blueprint_tb in blueprint_tbs:
                yield blueprint_tb.high
                yield blueprint_tb.low
                yield blueprint_tb.close

        yield blueprint.time_of_day
        yield blueprint.delta_to_last

    @staticmethod
    def _build_scaling_inds(
        fields: Iterable[FieldData],
    ) -> Iterator[tuple[int, list[int]]]:
        builder: dict[int, tuple[Optional[int], set[int]]] = {}

        for field_data in fields:
            if not isinstance(field_data, PriceFieldData):
                continue

            index = field_data.index
            price_group_id = field_data.price_group_id

            close_price_idx, price_field_idxs = builder.get(
                price_group_id, (None, set())
            )

            if index in price_field_idxs:
                raise ValueError()

            if field_data.is_close_price:
                if close_price_idx is not None:
                    raise ValueError()

                close_price_idx = index

            price_field_idxs.add(index)
            builder[price_group_id] = (close_price_idx, price_field_idxs)

        for close_price_idx, price_group_idxs in builder.values():
            if close_price_idx is None:
                raise ValueError()

            assert close_price_idx in price_group_idxs
            yield close_price_idx, list(price_group_idxs)

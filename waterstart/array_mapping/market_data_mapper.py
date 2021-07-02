from collections import Iterable, Iterator, Mapping, Sequence
from dataclasses import astuple, dataclass
from typing import Optional

from ..price import MarketData, SymbolData, TrendBar
from .base_mapper import BaseArrayMapper, FieldData


@dataclass
class PriceFieldData(FieldData):
    sym_id: int
    is_close_price: bool


# TODO: maybe instead of using FieldData we take MarketData[int], where int represent
# the inds and we have an internal method that returns an object that contains
# the information we need for each field (symbol info, is price, is close price..)
# this method would replace _flatten_fields


class MarketDataArrayMapper(BaseArrayMapper[MarketData[float]]):
    def __init__(self, blueprint: MarketData[FieldData]) -> None:
        super().__init__(set(self._flatten_fields(blueprint)))
        self._blueprint = blueprint
        self._scaling_idxs = list(self._build_scaling_inds(self._fields_set))

    # TODO: find better name
    @property
    def scaling_idxs(self) -> Sequence[tuple[int, list[int]]]:
        return self._scaling_idxs

    def iterate_index_to_value(
        self, value: MarketData[float]
    ) -> Iterator[tuple[int, float]]:
        for sym_id, blueprint_sym_data in self._blueprint.symbols_data_map.items():
            sym_data = value.symbols_data_map[sym_id]

            it: Iterator[tuple[TrendBar[FieldData], TrendBar[float]]] = zip(
                astuple(blueprint_sym_data), astuple(sym_data)
            )

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
            blueprint_tbs: tuple[TrendBar[FieldData], ...] = astuple(blueprint_sym_data)

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

    @staticmethod
    def _flatten_fields(blueprint: MarketData[FieldData]) -> Iterator[FieldData]:
        for blueprint_sym_data in blueprint.symbols_data_map.values():
            blueprint_tbs: tuple[TrendBar[FieldData], ...] = astuple(blueprint_sym_data)

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
            sym_id = field_data.sym_id

            close_price_idx, price_field_idxs = builder.get(sym_id, (None, set()))

            if index in price_field_idxs:
                raise ValueError()

            if field_data.is_close_price:
                if close_price_idx is not None:
                    raise ValueError()

                close_price_idx = index

            price_field_idxs.add(index)
            builder[sym_id] = (close_price_idx, price_field_idxs)

        for close_price_idx, group_prices_idxs in builder.values():
            if close_price_idx is None:
                raise ValueError()

            assert close_price_idx in group_prices_idxs
            yield close_price_idx, list(group_prices_idxs)

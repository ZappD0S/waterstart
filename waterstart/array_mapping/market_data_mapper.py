from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import astuple, dataclass
from typing import Optional, TypeVar

from ..symbols import TradedSymbolInfo

from ..price import MarketFeatures, SymbolData, TrendBar
from .base_mapper import BaseArrayMapper, FieldData

T = TypeVar("T", float, FieldData)


@dataclass
class PriceFieldData(FieldData):
    sym_info: TradedSymbolInfo
    is_close_price: bool


# TODO: maybe instead of using FieldData we take MarketData[int], where int represent
# the inds and we have an internal method that returns an object that contains
# the information we need for each field (symbol info, is price, is close price..)
# this method would replace _flatten_fields


class MarketDataArrayMapper(BaseArrayMapper[MarketFeatures[float]]):
    def __init__(self, blueprint: MarketFeatures[FieldData]) -> None:
        super().__init__(set(self._flatten_fields(blueprint)))
        self._blueprint = blueprint
        self._scaling_inds = list(self._build_scaling_inds(self._fields_set))

    # TODO: find better name
    @property
    def scaling_inds(self) -> Sequence[tuple[int, list[int]]]:
        return self._scaling_inds

    def iterate_index_to_value(
        self, value: MarketFeatures[float]
    ) -> Iterator[tuple[int, float]]:
        for sym_info, blueprint_sym_data in self._blueprint.symbols_data_map.items():
            sym_data = value.symbols_data_map[sym_info]

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
    ) -> MarketFeatures[float]:
        sym_info_map: Mapping[TradedSymbolInfo, SymbolData[float]] = {}

        for sym_info, blueprint_sym_data in self._blueprint.symbols_data_map.items():
            blueprint_tbs: tuple[TrendBar[FieldData], ...] = astuple(blueprint_sym_data)

            tbs: list[TrendBar[float]] = [
                TrendBar(
                    high=mapping[blueprint_tb.high.index],
                    low=mapping[blueprint_tb.low.index],
                    close=mapping[blueprint_tb.close.index],
                )
                for blueprint_tb in blueprint_tbs
            ]
            sym_info_map[sym_info] = SymbolData(*tbs)

        market_data: MarketFeatures[float] = MarketFeatures(
            sym_info_map,
            time_of_day=mapping[self._blueprint.time_of_day.index],
            delta_to_last=mapping[self._blueprint.delta_to_last.index],
        )
        return market_data

    @staticmethod
    def _flatten_fields(blueprint: MarketFeatures[FieldData]) -> Iterator[FieldData]:
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
        builder: dict[TradedSymbolInfo, tuple[Optional[int], set[int]]] = {}

        for field_data in fields:
            if not isinstance(field_data, PriceFieldData):
                continue

            index = field_data.index
            sym_info = field_data.sym_info

            close_price_ind, price_field_inds = builder.get(sym_info, (None, set()))

            if index in price_field_inds:
                raise ValueError()

            if field_data.is_close_price:
                if close_price_ind is not None:
                    raise ValueError()

                close_price_ind = index

            price_field_inds.add(index)
            builder[sym_info] = (close_price_ind, price_field_inds)

        for close_price_ind, group_prices_inds in builder.values():
            if close_price_ind is None:
                raise ValueError()

            assert close_price_ind in group_prices_inds
            yield close_price_ind, list(group_prices_inds)

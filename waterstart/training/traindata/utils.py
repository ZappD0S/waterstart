from dataclasses import dataclass, InitVar
from typing import Collection, Iterator, Mapping, Optional
import torch

from ...array_mapping.base_mapper import FieldData
from ...array_mapping.dict_based_mapper import DictBasedArrayMapper
from ...array_mapping.market_data_mapper import MarketDataArrayMapper
from ...price import MarketData
from ...symbols import TradedSymbolInfo


@dataclass
class TrainingData:
    market_data: torch.Tensor
    midpoint_prices: torch.Tensor
    spreads: torch.Tensor
    base_to_dep_rates: torch.Tensor
    quote_to_dep_rates: torch.Tensor
    market_data_blueprint: InitVar[MarketData[FieldData]]
    traded_sym_blueprint_map: InitVar[Mapping[int, FieldData]]
    traded_symbols: Collection[TradedSymbolInfo]

    def __post_init__(
        self,
        market_data_blueprint: MarketData[FieldData],
        traded_sym_blueprint_map: Mapping[int, FieldData],
    ):
        if (
            not (n_timestemps := self.market_data.shape[0])
            == self.midpoint_prices.shape[0]
            == self.spreads.shape[0]
            == self.base_to_dep_rates.shape[0]
            == self.quote_to_dep_rates.shape[0]
        ):
            raise ValueError()

        market_data_arr_mapper = MarketDataArrayMapper(market_data_blueprint)
        traded_sym_arr_mapper = DictBasedArrayMapper(traded_sym_blueprint_map)

        if (
            not (n_market_features := market_data_arr_mapper.n_fields)
            == self.market_data.shape[1]
        ):
            raise ValueError()

        if (
            not (n_traded_sym := traded_sym_arr_mapper.n_fields)
            == self.midpoint_prices.shape[1]
            == self.spreads.shape[1]
            == self.base_to_dep_rates.shape[1]
            == self.quote_to_dep_rates.shape[1]
            == len(market_data_blueprint.symbols_data_map)
            == len(self.traded_symbols)
        ):
            raise ValueError()

        self._market_data_arr_mapper = market_data_arr_mapper
        self._traded_sym_arr_mapper = traded_sym_arr_mapper
        self._n_timestemps = n_timestemps
        self._n_traded_sym = n_traded_sym
        self._n_market_features = n_market_features

    @property
    def n_timestemps(self) -> int:
        return self._n_timestemps

    @property
    def n_traded_sym(self) -> int:
        return self._n_traded_sym

    @property
    def n_market_features(self) -> int:
        return self._n_market_features

    @property
    def market_data_arr_mapper(self) -> MarketDataArrayMapper:
        return self._market_data_arr_mapper

    @property
    def traded_sym_arr_mapper(self) -> DictBasedArrayMapper[int]:
        return self._traded_sym_arr_mapper


@dataclass
class TrainingState:
    batch_inds_it: Iterator[torch.Tensor]
    next_batch_inds: Optional[torch.Tensor]

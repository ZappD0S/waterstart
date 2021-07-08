from __future__ import annotations

from collections import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import torch

from ..price import MarketData


@dataclass
class TradesState:
    trades_sizes: torch.Tensor
    trades_prices: torch.Tensor

    # TODO: make the naming convention uniform
    @cached_property
    def pos_size(self) -> torch.Tensor:
        return self.trades_sizes.sum(dim=0)


@dataclass
class MarketState:
    prev_market_data_arr: torch.Tensor
    latest_market_data: MarketData[float]


@dataclass
class RawMarketState:
    market_data: torch.Tensor
    midpoint_prices: torch.Tensor
    spreads: torch.Tensor
    # TODO: maybe rename to base_to_dep?
    margin_rate: torch.Tensor
    quote_to_dep_rate: torch.Tensor


T = TypeVar("T", MarketState, RawMarketState)


@dataclass
class ModelInput(Generic[T]):
    market_state: T
    # TODO: it would be useful to have the three below in a dataclass
    trades_state: TradesState
    hidden_state: torch.Tensor
    balance: torch.Tensor


# TODO: make abstract?
@dataclass
class ModelInference:
    pos_sizes: torch.Tensor


# TODO: we need a better name..
@dataclass
class ModelInferenceWithMap(ModelInference):
    pos_sizes_map: Mapping[int, float]
    market_data_arr: torch.Tensor
    hidden_state: torch.Tensor


@dataclass
class RawModelOutput:
    cnn_output: torch.Tensor
    z_sample: torch.Tensor
    z_logprob: torch.Tensor
    exec_samples: torch.Tensor
    exec_logprobs: torch.Tensor
    fractions: torch.Tensor

    @cached_property
    def exec_mask(self) -> torch.Tensor:
        return self.exec_samples == 1


@dataclass
class ModelInferenceWithRawOutput(ModelInference):
    raw_model_output: RawModelOutput

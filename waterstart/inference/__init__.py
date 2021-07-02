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

    # TODO: make the naming convetion uniform
    @cached_property
    def pos_size(self) -> torch.Tensor:
        return self.trades_sizes.sum(dim=0)


@dataclass
class ModelState:
    trades_state: TradesState
    balance: torch.Tensor
    hidden_state: torch.Tensor


@dataclass
class MarketState:
    prev_market_data_arr: torch.Tensor
    latest_market_data: MarketData[float]


@dataclass
class RawMarketState:
    market_data: torch.Tensor
    sym_prices: torch.Tensor
    margin_rate: torch.Tensor


T = TypeVar("T", MarketState, RawMarketState)


@dataclass
class ModelInput(Generic[T]):
    market_state: T
    trades_state: TradesState
    hidden_state: torch.Tensor
    balance: torch.Tensor


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
    z_sample: torch.Tensor
    z_logprob: torch.Tensor
    exec_samples: torch.Tensor
    exec_logprobs: torch.Tensor
    fractions: torch.Tensor


@dataclass
class ModelInferenceWithRawOutput(ModelInference):
    raw_model_output: RawModelOutput

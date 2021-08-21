from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch


@dataclass
class AccountState:
    trades_sizes: torch.Tensor
    trades_prices: torch.Tensor
    balance: torch.Tensor

    # TODO: make the naming convention uniform
    @cached_property
    def pos_size(self) -> torch.Tensor:
        return self.trades_sizes.sum(0)

    @cached_property
    def available_trades_mask(self) -> torch.Tensor:
        # NOTE: if the last trade is 0 it means that there
        # is at least one trade can be opened
        return self.trades_sizes[0] == 0


@dataclass
class RawMarketState:
    market_data: torch.Tensor
    midpoint_prices: torch.Tensor
    spreads: torch.Tensor
    # TODO: maybe rename to base_to_dep?
    margin_rate: torch.Tensor
    quote_to_dep_rate: torch.Tensor


@dataclass
class ModelInput:
    market_state: RawMarketState
    account_state: AccountState
    hidden_state: torch.Tensor


@dataclass
class RawModelOutput:
    cnn_output: torch.Tensor
    z_sample: torch.Tensor
    z_logprob: torch.Tensor
    exec_samples: torch.Tensor
    exec_logprobs: torch.Tensor
    fracs: torch.Tensor
    fracs_logprobs: torch.Tensor

    @cached_property
    def exec_mask(self) -> torch.Tensor:
        return self.exec_samples == 1


@dataclass
class ModelOutput:
    pos_sizes: torch.Tensor
    raw_model_output: RawModelOutput

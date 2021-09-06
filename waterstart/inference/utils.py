# from functools import cached_property

import torch


@torch.jit.script  # type: ignore
class AccountState:
    def __init__(
        self,
        trades_sizes: torch.Tensor,
        trades_prices: torch.Tensor,
        balance: torch.Tensor,
    ) -> None:
        self.trades_sizes = trades_sizes
        self.trades_prices = trades_prices
        self.balance = balance

    # TODO: make the naming convention uniform
    # @cached_property
    @property
    def pos_size(self) -> torch.Tensor:
        return self.trades_sizes.sum(0)

    # @cached_property
    @property
    def available_trades_mask(self) -> torch.Tensor:
        # return self.trades_sizes[0] == 0
        return self.trades_sizes[-1] == 0


@torch.jit.script  # type: ignore
class MarketState:
    def __init__(
        self,
        midpoint_prices: torch.Tensor,
        spreads: torch.Tensor,
        margin_rate: torch.Tensor,
        quote_to_dep_rate: torch.Tensor,
    ) -> None:
        self.midpoint_prices = midpoint_prices
        self.spreads = spreads
        self.margin_rate = margin_rate
        self.quote_to_dep_rate = quote_to_dep_rate

    # @cached_property
    @property
    def bid_ask_prices(self) -> tuple[torch.Tensor, torch.Tensor]:
        midpoint_prices = self.midpoint_prices
        half_spreads = self.spreads / 2
        bid_prices = midpoint_prices - half_spreads
        ask_prices = midpoint_prices + half_spreads

        return bid_prices, ask_prices


@torch.jit.script  # type: ignore
class ModelInput:
    def __init__(
        self,
        account_state: AccountState,
        market_state: MarketState,
        raw_market_data: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        self.market_state = market_state
        self.account_state = account_state
        self.raw_market_data = raw_market_data
        self.hidden_state = hidden_state


@torch.jit.script  # type: ignore
class RawModelOutput:
    def __init__(
        self,
        trades_data: torch.Tensor,
        market_features: torch.Tensor,
        # z_sample: torch.Tensor,
        # fracs: torch.Tensor,
        logprob: torch.Tensor,
        # exec_samples: torch.Tensor,
        # exec_logprobs: torch.Tensor,
        # fracs_logprobs: torch.Tensor,
    ) -> None:
        self.trades_data = trades_data
        # self.z_sample = z_sample
        self.market_features = market_features
        # self.fracs = fracs
        self.logprob = logprob
        # self.exec_samples = exec_samples
        # self.exec_logprobs = exec_logprobs
        # self.fracs_logprobs = fracs_logprobs


class ModelOutput:
    def __init__(
        self, pos_sizes: torch.Tensor, raw_model_output: RawModelOutput
    ) -> None:
        self.pos_sizes = pos_sizes
        self.raw_model_output = raw_model_output

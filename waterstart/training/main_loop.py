from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch.optim import Optimizer

from ..inference import ModelInput, RawMarketState, TradesState
from .loss import LossEvaluator


@dataclass
class TrainingData:
    market_data: torch.Tensor
    midpoint_prices: torch.Tensor
    spreads: torch.Tensor
    base_to_dep_rates: torch.Tensor
    quote_to_dep_rates: torch.Tensor

    # TODO: this will also contain a Mapping[int, FieldData] and a MarketData[FieldData]

    def __post_init__(self):
        if (
            not (n_timestemps := self.market_data.shape[0])
            == self.midpoint_prices.shape[0]
            == self.spreads.shape[0]
            == self.base_to_dep_rates.shape[0]
            == self.quote_to_dep_rates.shape[0]
        ):
            raise ValueError()

        if (
            not (n_traded_sym := self.market_data.shape[1])
            == self.midpoint_prices.shape[1]
            == self.spreads.shape[1]
            == self.base_to_dep_rates.shape[1]
            == self.quote_to_dep_rates.shape[1]
        ):
            raise ValueError()

        self._n_timestemps = n_timestemps
        self._n_traded_sym = n_traded_sym
        self._market_features = self.market_data.shape[2]

    @property
    def n_timestemps(self) -> int:
        return self._n_timestemps

    @property
    def n_traded_sym(self) -> int:
        return self._n_traded_sym

    @property
    def market_features(self) -> int:
        return self._market_features


class TrainDataManager:
    def __init__(
        self,
        training_data: TrainingData,
        # TODO: maybe pass NetworkModules instead?
        batch_size: int,
        n_samples: int,
        window_size: int,
        max_trades: int,
        hidden_state_size: int,
    ) -> None:
        self._batch_size = batch_size
        n_timestemps = training_data.n_timestemps

        self._training_data = training_data
        # NOTE: the -1 is to make sure that whe can save the data at
        # position batch_ind + 1
        self._n_batches = (n_timestemps - 1) // batch_size
        n_traded_sym = training_data.n_traded_sym
        self._n_samples = n_samples

        # TODO: initial balance needs to be parametric
        self._balances = torch.full((self._n_batches, n_samples), 1000.0)
        self._trade_sizes = torch.zeros(
            (self._n_batches, max_trades, n_traded_sym, n_samples)
        )
        self._trade_prices = torch.zeros(
            (self._n_batches, max_trades, n_traded_sym, n_samples)
        )
        self._hidden_states = torch.zeros(
            (self._n_batches, hidden_state_size, n_samples)
        )
        self._windowed_market_data = training_data.market_data.unfold(
            0, window_size, step=1
        )

        self._batch_inds_it: Iterator[torch.Tensor] = self._build_batch_inds_it()
        self._next_batch_inds: Optional[torch.Tensor] = next(self._batch_inds_it)
        self._save_pending: bool = False

    def _build_batch_inds_it(self) -> Iterator[torch.Tensor]:
        batch_size = self._batch_size
        n_batches = self._n_batches

        batch_inds = torch.randperm(n_batches * batch_size).view(n_batches, batch_size)
        return iter(batch_inds)

    def load(self) -> ModelInput[RawMarketState]:
        def build_batch(storage: torch.Tensor) -> torch.Tensor:
            return storage[batch_inds].moveaxis(0, -1).flatten(-2, -1)

        if self._save_pending:
            raise RuntimeError()

        self._save_pending = True

        if (batch_inds := self._next_batch_inds) is None:
            self._batch_inds_it = self._build_batch_inds_it()

        market_state = RawMarketState(
            market_data=build_batch(self._windowed_market_data),
            midpoint_prices=build_batch(self._training_data.midpoint_prices),
            spreads=build_batch(self._training_data.spreads),
            margin_rate=build_batch(self._training_data.base_to_dep_rates),
            quote_to_dep_rate=build_batch(self._training_data.quote_to_dep_rates),
        )

        trades_state = TradesState(
            trades_sizes=build_batch(self._trade_sizes),
            trades_prices=build_batch(self._trade_prices),
        )

        return ModelInput(
            market_state,
            trades_state,
            hidden_state=build_batch(self._hidden_states),
            balance=build_batch(self._balances),
        )

    def save(
        self,
        trades_state: TradesState,
        hidden_state: torch.Tensor,
        balance: torch.Tensor,
    ) -> None:
        if not self._save_pending:
            raise RuntimeError()

        if (batch_inds := self._next_batch_inds) is None:
            raise RuntimeError()

        def transform_batch(batch: torch.Tensor) -> torch.Tensor:
            return batch.unflatten(  # type: ignore
                -1, (self._n_samples, self._batch_size)
            ).moveaxis(-1, 0)

        shifted_batch_inds = batch_inds + 1
        self._trade_sizes[shifted_batch_inds] = transform_batch(
            trades_state.trades_sizes
        )
        self._trade_prices[shifted_batch_inds] = transform_batch(
            trades_state.trades_prices
        )
        self._hidden_states[shifted_batch_inds] = transform_batch(hidden_state)
        self._balances[shifted_batch_inds] = transform_batch(balance)

        self._next_batch_inds = next(self._batch_inds_it, None)
        self._save_pending = False


class Trainer:
    def __init__(self, loss_eval: LossEvaluator, optimizer: Optimizer) -> None:
        pass

    def step(self):
        # 1. load data
        # 2. compute loss and backprop
        # 3. log data (tensorboard..)
        # 4. save output
        pass

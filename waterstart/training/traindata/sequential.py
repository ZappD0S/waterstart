from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch
from numpy.lib.stride_tricks import sliding_window_view

from ...inference import AccountState, MarketState, ModelInput
from .abc import BaseTrainDataManager, TrainingData, TrainingState
from .batch_manager import ReadonlyBatchManager


@dataclass
class SequentialTrainingState(TrainingState):
    pass


class SequentialTrainDataManager(BaseTrainDataManager):
    def __init__(
        self,
        training_data: TrainingData,
        training_state: Optional[TrainingState],
        batch_size: int,
        seq_len: int,
        window_size: int,
        max_trades: int,
        hidden_state_size: int,
        initial_balance: float,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            training_data, training_state, batch_size, seq_len, window_size, device
        )

        # TODO: check that training_data and training_state shapes match

        self._market_data_batch_manager = ReadonlyBatchManager(
            sliding_window_view(training_data.market_data, window_size, axis=0),
            batch_dims=1,
            load_lag=window_size - 1,
            batch_dims_last=False,
            device=device,
        )

        self._midpoint_prices_batch_manager = ReadonlyBatchManager(
            training_data.midpoint_prices,
            batch_dims=1,
            load_lag=0,
            batch_dims_last=False,
            device=device,
        )
        self._spreads_batch_manager = ReadonlyBatchManager(
            training_data.spreads,
            batch_dims=1,
            load_lag=0,
            batch_dims_last=False,
            device=device,
        )
        self._base_to_dep_rates_batch_manager = ReadonlyBatchManager(
            training_data.base_to_dep_rates,
            batch_dims=1,
            load_lag=0,
            batch_dims_last=False,
            device=device,
        )
        self._quote_to_dep_rates_batch_manager = ReadonlyBatchManager(
            training_data.quote_to_dep_rates,
            batch_dims=1,
            load_lag=0,
            batch_dims_last=False,
            device=device,
        )

        n_traded_sym = training_data.n_traded_sym
        trades_sizes = torch.zeros((max_trades, n_traded_sym, batch_size))
        trades_prices = torch.zeros((max_trades, n_traded_sym, batch_size))
        balance = torch.full((batch_size,), initial_balance)

        self._default_account_state = AccountState(
            trades_sizes=trades_sizes.to(device),
            trades_prices=trades_prices.to(device),
            balance=balance.to(device),
        )

        default_hidden_state = torch.zeros((batch_size, hidden_state_size))
        self._default_hidden_state = default_hidden_state.to(device)

    def _build_batch_inds_it(self) -> Iterator[npt.NDArray[Any]]:
        batch_size = self._batch_size
        seq_len = self._seq_len
        batch_inds = np.arange(self._window_size - 1, self._n_timestemps - seq_len)

        n_samples = batch_inds.shape[0]
        n_batches = n_samples // batch_size
        rand_perm = np.random.permutation(n_batches * batch_size)
        rand_perm.shape = (n_batches, batch_size)

        batch_inds = batch_inds[rand_perm, None] + np.arange(seq_len)[:, None]
        return iter(batch_inds)

    def load_data(self) -> ModelInput:
        batch_inds = self._get_next_batch_inds()

        market_state = MarketState(
            midpoint_prices=self._midpoint_prices_batch_manager.build_batch(batch_inds),
            spreads=self._spreads_batch_manager.build_batch(batch_inds),
            margin_rate=self._base_to_dep_rates_batch_manager.build_batch(batch_inds),
            quote_to_dep_rate=self._quote_to_dep_rates_batch_manager.build_batch(
                batch_inds
            ),
        )

        return ModelInput(
            self._default_account_state,
            market_state,
            self._market_data_batch_manager.build_batch(batch_inds),
            self._default_hidden_state,
        )

    def save_state(self) -> SequentialTrainingState:
        return SequentialTrainingState(self._batch_inds_it, self._next_batch_inds)

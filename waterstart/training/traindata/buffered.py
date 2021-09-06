from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import torch

from ...inference import AccountState, MarketState, ModelInput
from .abc import BaseTrainDataManager, TrainingData, TrainingState
from .batch_manager import BatchManager, ReadonlyBatchManager
from .inds_perm import compute_perm


@dataclass
class BufferData:
    balances: torch.Tensor
    trades_sizes: torch.Tensor
    trades_prices: torch.Tensor
    hidden_states: torch.Tensor


@dataclass
class BufferedTrainingState(TrainingState):
    buffer_data: BufferData


class BufferedTrainDataManager(BaseTrainDataManager):
    def __init__(
        self,
        training_data: TrainingData,
        training_state: Optional[BufferedTrainingState],
        buffer_data: Optional[BufferData],
        batch_size: int,
        seq_len: int,
        window_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            training_data, training_state, batch_size, seq_len, window_size, device
        )

        if buffer_data is None:
            if training_state is None:
                raise ValueError()

            buffer_data = training_state.buffer_data
        elif training_state is not None:
            raise ValueError()

        # TODO: check that training_data and buffer_data shapes match

        self._balances_batch_manager = BatchManager(
            buffer_data.balances,
            batch_dims=1,
            load_lag=1,
            device=device,
        )
        self._trades_sizes_batch_manager = BatchManager(
            buffer_data.trades_sizes,
            batch_dims=1,
            load_lag=1,
            device=device,
        )
        self._trades_prices_batch_manager = BatchManager(
            buffer_data.trades_prices,
            batch_dims=1,
            load_lag=1,
            device=device,
        )
        self._hidden_states_batch_manager = BatchManager(
            buffer_data.hidden_states,
            batch_dims=1,
            load_lag=1,
            batch_dims_last=False,
            device=device,
        )
        self._market_data_batch_manager = ReadonlyBatchManager(
            training_data.market_data.unfold(0, window_size, step=1),
            batch_dims=1,
            load_lag=window_size - 1,
            batch_dims_last=False,
            device=device,
        )

        self._midpoint_prices_batch_manager = ReadonlyBatchManager(
            training_data.midpoint_prices,
            batch_dims=1,
            load_lag=0,
            device=device,
        )
        self._spreads_batch_manager = ReadonlyBatchManager(
            training_data.spreads,
            batch_dims=1,
            load_lag=0,
            device=device,
        )
        self._base_to_dep_rates_batch_manager = ReadonlyBatchManager(
            training_data.base_to_dep_rates,
            batch_dims=1,
            load_lag=0,
            device=device,
        )
        self._quote_to_dep_rates_batch_manager = ReadonlyBatchManager(
            training_data.quote_to_dep_rates,
            batch_dims=1,
            load_lag=0,
            device=device,
        )

        self._save_pending: bool = False

    def _build_batch_inds_it(self) -> Iterator[torch.Tensor]:
        batch_size = self._batch_size
        seq_len = self._seq_len
        batch_inds = torch.arange(self._window_size - 1, self._n_timestemps - seq_len)

        n_samples = batch_inds.shape[0]
        n_batches = n_samples // batch_size
        rand_perm = torch.from_numpy(  # type: ignore
            compute_perm(n_batches, batch_size, seq_len)
        )

        batch_inds = batch_inds[rand_perm].unsqueeze(1) + torch.arange(
            seq_len
        ).unsqueeze(-1)

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

        account_state = AccountState(
            trades_sizes=self._trades_sizes_batch_manager.build_batch(batch_inds),
            trades_prices=self._trades_prices_batch_manager.build_batch(batch_inds),
            balance=self._balances_batch_manager.build_batch(batch_inds),
        )

        return ModelInput(
            account_state,
            market_state,
            raw_market_data=self._market_data_batch_manager.build_batch(batch_inds),
            hidden_state=self._hidden_states_batch_manager.build_batch(batch_inds),
        )

    def store_data(
        self, account_state: AccountState, hidden_state: torch.Tensor
    ) -> tuple[AccountState, torch.Tensor]:
        account_state = AccountState(
            self._trades_sizes_batch_manager.store_batch(account_state.trades_sizes),
            self._trades_prices_batch_manager.store_batch(account_state.trades_prices),
            self._balances_batch_manager.store_batch(account_state.balance),
        )
        hidden_state = self._hidden_states_batch_manager.store_batch(hidden_state)

        return account_state, hidden_state

    def save_state(self) -> BufferedTrainingState:
        return BufferedTrainingState(
            self._batch_inds_it,
            self._next_batch_inds,
            BufferData(
                self._balances_batch_manager.storage,
                self._trades_sizes_batch_manager.storage,
                self._trades_prices_batch_manager.storage,
                self._hidden_states_batch_manager.storage,
            ),
        )

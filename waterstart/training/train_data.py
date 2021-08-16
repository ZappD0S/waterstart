from collections.abc import Collection, Iterator, Mapping
from dataclasses import InitVar, dataclass
from typing import Optional

import torch

from ..array_mapping.base_mapper import FieldData
from ..array_mapping.dict_based_mapper import DictBasedArrayMapper
from ..array_mapping.market_data_mapper import MarketDataArrayMapper
from ..inference import AccountState, ModelInput, RawMarketState
from ..price import MarketData
from ..symbols import TradedSymbolInfo


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


class ReadonlyBatchManager:
    def __init__(
        self,
        storage: torch.Tensor,
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if batch_dims > storage.ndim:
            raise ValueError()

        if load_lag < 0:
            raise ValueError()

        self._storage = storage
        self._batch_dims = batch_dims
        self._load_lag = load_lag
        self._batch_dims_last = batch_dims_last
        self._device = device

    @property
    def storage(self) -> torch.Tensor:
        return self._storage

    def build_batch(self, inds: torch.Tensor) -> torch.Tensor:
        batch: torch.Tensor = self._storage[inds - self._load_lag]
        batch = self._transform_batch(batch, self._batch_dims + inds.ndim - 1)
        return batch.to(self._device)

    def _transform_batch(self, batch: torch.Tensor, batch_dims: int) -> torch.Tensor:
        if not self._batch_dims_last:
            return batch

        return batch.permute(*range(batch_dims, batch.ndim), *range(batch_dims))


class ExpandableBatchManagager(ReadonlyBatchManager):
    def __init__(
        self,
        storage: torch.Tensor,
        expand_size: int,
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            storage.unsqueeze(batch_dims),
            batch_dims + 1,
            load_lag,
            batch_dims_last,
            device,
        )
        self._expand_size = expand_size

    def _transform_batch(self, batch: torch.Tensor, batch_dims: int) -> torch.Tensor:
        batch = batch.expand(
            *(-1,) * (batch_dims - 1),
            self._expand_size,
            *(-1,) * (batch.ndim - batch_dims),
        )

        return super()._transform_batch(batch, batch_dims)


class BatchManager(ReadonlyBatchManager):
    def __init__(
        self,
        storage: torch.Tensor,
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(storage, batch_dims, load_lag, batch_dims_last, device)
        self._inds: Optional[torch.Tensor] = None

    def build_batch(self, inds: torch.Tensor) -> torch.Tensor:
        self._inds = inds
        return super().build_batch(inds)

    def store_batch(self, batch: torch.Tensor) -> None:
        if (inds := self._inds) is None:
            raise RuntimeError()

        assert not batch.requires_grad
        batch = batch.cpu()
        batch = self._inverse_transform_batch(batch, self._batch_dims + inds.ndim - 1)
        self._inds = None
        self._storage[inds] = batch

    def _inverse_transform_batch(
        self, batch: torch.Tensor, batch_dims: int
    ) -> torch.Tensor:
        if not self._batch_dims_last:
            return batch

        ndim = batch.ndim
        return batch.permute(*range(ndim - batch_dims, ndim), *range(ndim - batch_dims))


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
        initial_balance: float,
        device: Optional[torch.device] = None,
    ) -> None:
        n_timestemps = training_data.n_timestemps
        n_traded_sym = training_data.n_traded_sym

        self._balances_batch_manager = BatchManager(
            torch.full((n_timestemps, n_samples), initial_balance),
            batch_dims=2,
            load_lag=1,
            device=device,
        )
        self._trades_sizes_batch_manager = BatchManager(
            torch.zeros((n_timestemps, n_samples, max_trades, n_traded_sym)),
            batch_dims=2,
            load_lag=1,
            device=device,
        )
        self._trades_prices_batch_manager = BatchManager(
            torch.zeros((n_timestemps, n_samples, max_trades, n_traded_sym)),
            batch_dims=2,
            load_lag=1,
            device=device,
        )
        self._hidden_states_batch_manager = BatchManager(
            torch.zeros((n_timestemps, n_samples, hidden_state_size)),
            batch_dims=2,
            load_lag=1,
            batch_dims_last=False,
            device=device,
        )
        self._market_data_batch_manager = ExpandableBatchManagager(
            training_data.market_data.unfold(0, window_size, step=1),
            expand_size=n_samples,
            batch_dims=1,
            load_lag=window_size - 1,
            batch_dims_last=False,
            device=device,
        )

        self._midpoint_prices_batch_manager = ExpandableBatchManagager(
            training_data.midpoint_prices,
            expand_size=n_samples,
            batch_dims=1,
            load_lag=0,
            device=device,
        )
        self._spreads_batch_manager = ExpandableBatchManagager(
            training_data.spreads,
            expand_size=n_samples,
            batch_dims=1,
            load_lag=0,
            device=device,
        )
        self._base_to_dep_rates_batch_manager = ExpandableBatchManagager(
            training_data.base_to_dep_rates,
            expand_size=n_samples,
            batch_dims=1,
            load_lag=0,
            device=device,
        )
        self._quote_to_dep_rates_batch_manager = ExpandableBatchManagager(
            training_data.quote_to_dep_rates,
            expand_size=n_samples,
            batch_dims=1,
            load_lag=0,
            device=device,
        )

        self._n_timestemps = n_timestemps
        self._batch_size = batch_size
        self._window_size = window_size
        self._batch_inds_it: Iterator[torch.Tensor] = self._build_batch_inds_it()
        self._next_batch_inds: Optional[torch.Tensor] = next(self._batch_inds_it)
        self._save_pending: bool = False

    @property
    def balances(self) -> torch.Tensor:
        return self._balances_batch_manager.storage

    @property
    def trades_sizes(self) -> torch.Tensor:
        return self._trades_sizes_batch_manager.storage

    @property
    def trades_prices(self) -> torch.Tensor:
        return self._trades_prices_batch_manager.storage

    def _build_batch_inds_it(self) -> Iterator[torch.Tensor]:
        batch_size = self._batch_size
        batch_inds = torch.arange(
            self._window_size - 1, self._n_timestemps - batch_size
        ).unsqueeze(-1) + torch.arange(batch_size)

        rand_perm = torch.randperm(batch_inds.shape[0])
        return iter(batch_inds[rand_perm])

    def load(self) -> ModelInput:
        if (batch_inds := self._next_batch_inds) is None:
            self._batch_inds_it = self._build_batch_inds_it()
            batch_inds = self._next_batch_inds = next(self._batch_inds_it)

        self._next_batch_inds = next(self._batch_inds_it, None)

        market_state = RawMarketState(
            market_data=self._market_data_batch_manager.build_batch(batch_inds),
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
            market_state,
            account_state,
            hidden_state=self._hidden_states_batch_manager.build_batch(batch_inds),
        )

    # TODO: maybe return the transformed balance batch from here (smth else?)
    def save(self, account_state: AccountState, hidden_state: torch.Tensor) -> None:
        self._trades_sizes_batch_manager.store_batch(account_state.trades_sizes)
        self._trades_prices_batch_manager.store_batch(account_state.trades_prices)
        self._balances_batch_manager.store_batch(account_state.balance)

        self._hidden_states_batch_manager.store_batch(hidden_state)

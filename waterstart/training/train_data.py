from dataclasses import dataclass
from typing import Collection, Iterator, Mapping, Optional
from ..array_mapping.dict_based_mapper import DictBasedArrayMapper
from ..array_mapping.market_data_mapper import MarketDataArrayMapper

import torch

from ..array_mapping.base_mapper import FieldData
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
    market_data_blueprint: MarketData[FieldData]
    traded_sym_blueprint_map: Mapping[int, FieldData]
    traded_symbols: Collection[TradedSymbolInfo]

    def __post_init__(self):
        if (
            not (n_timestemps := self.market_data.shape[0])
            == self.midpoint_prices.shape[0]
            == self.spreads.shape[0]
            == self.base_to_dep_rates.shape[0]
            == self.quote_to_dep_rates.shape[0]
        ):
            raise ValueError()

        market_data_arr_mapper = MarketDataArrayMapper(self.market_data_blueprint)
        traded_sym_arr_mapper = DictBasedArrayMapper(self.traded_sym_blueprint_map)

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
            == len(self.market_data_blueprint.symbols_data_map)
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
        self._batch_size = batch_size
        n_timestemps = training_data.n_timestemps

        self._training_data = training_data
        # NOTE: the -1 is to make sure that whe can save the data at
        # position batch_ind + 1
        self._n_batches = (n_timestemps - 1) // batch_size
        n_traded_sym = training_data.n_traded_sym
        self._n_samples = n_samples
        self._device = device

        self._balances = torch.full((self._n_batches, n_samples), initial_balance)
        self._trade_sizes = torch.zeros(
            (self._n_batches, n_samples, max_trades, n_traded_sym)
        )
        self._trade_prices = torch.zeros(
            (self._n_batches, n_samples, max_trades, n_traded_sym)
        )
        self._hidden_states = torch.zeros(
            (self._n_batches, n_samples, hidden_state_size)
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
        def build_batch(
            storage: torch.Tensor,
            add_samples_dim: bool,
            move_batch_dims_to_last: bool = True,
        ) -> torch.Tensor:
            # TODO: for the market data we need to subtract a window length (- 1?)
            batch: torch.Tensor = storage[batch_inds]

            if add_samples_dim:
                first, *rest = batch.shape
                batch = batch.unsqueeze(1).expand(first, self._n_samples, *rest)

            if move_batch_dims_to_last:
                batch = batch.permute(*range(2, batch.ndim), 0, 1)

            return batch.to(self._device)

        if self._save_pending:
            raise RuntimeError()

        self._save_pending = True

        if (batch_inds := self._next_batch_inds) is None:
            self._batch_inds_it = self._build_batch_inds_it()
            batch_inds = self._next_batch_inds = next(self._batch_inds_it)

        market_state = RawMarketState(
            market_data=build_batch(self._windowed_market_data, True, False),
            midpoint_prices=build_batch(self._training_data.midpoint_prices, True),
            spreads=build_batch(self._training_data.spreads, True),
            margin_rate=build_batch(self._training_data.base_to_dep_rates, True),
            quote_to_dep_rate=build_batch(self._training_data.quote_to_dep_rates, True),
        )

        account_state = AccountState(
            trades_sizes=build_batch(self._trade_sizes, False),
            trades_prices=build_batch(self._trade_prices, False),
            balance=build_batch(self._balances, False),
        )

        return ModelInput(
            market_state,
            account_state,
            hidden_state=build_batch(self._hidden_states, False, False),
        )

    def save(self, account_state: AccountState, hidden_state: torch.Tensor) -> None:
        if not self._save_pending:
            raise RuntimeError()

        if (batch_inds := self._next_batch_inds) is None:
            raise RuntimeError()

        def transform_batch(batch: torch.Tensor) -> torch.Tensor:
            return batch.cpu().permute(-2, -1, *range(batch.ndim - 2))
            # return batch.cpu().moveaxis(-1, 0)

        shifted_batch_inds = batch_inds + 1
        self._trade_sizes[shifted_batch_inds] = transform_batch(
            account_state.trades_sizes
        )
        self._trade_prices[shifted_batch_inds] = transform_batch(
            account_state.trades_prices
        )
        self._hidden_states[shifted_batch_inds] = hidden_state
        self._balances[shifted_batch_inds] = transform_batch(account_state.balance)

        self._next_batch_inds = next(self._batch_inds_it, None)
        self._save_pending = False

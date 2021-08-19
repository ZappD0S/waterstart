from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import torch
import torch.distributions as dist
import torch.jit as jit
import torch.nn as nn

from ..array_mapping.dict_based_mapper import DictBasedArrayMapper
from ..array_mapping.market_data_mapper import MarketDataArrayMapper
from ..array_mapping.utils import map_to_arrays
from ..inference.model import CNN, Emitter, GatedTransition
from ..symbols import TradedSymbolInfo
from . import AccountState, ModelInput, ModelOutput, RawModelOutput


class NetworkModules(nn.Module):
    def __init__(
        self,
        cnn: CNN,
        gated_trans: GatedTransition,
        iafs: Sequence[nn.Module],
        emitter: Emitter,
        market_data_arr_mapper: MarketDataArrayMapper,
        traded_sym_arr_mapper: DictBasedArrayMapper[int],
        traded_symbols: Collection[TradedSymbolInfo],
        leverage: float,
    ):
        super().__init__()

        if (n_traded_sym := cnn.n_traded_sym) != emitter.n_traded_sym:
            raise ValueError()

        if (market_features := cnn.market_features) != market_data_arr_mapper.n_fields:
            raise ValueError()

        id_to_traded_sym = {sym.id: sym for sym in traded_symbols}

        if len(traded_symbols) < len(id_to_traded_sym):
            raise ValueError()

        if id_to_traded_sym.keys() != traded_sym_arr_mapper.keys:
            raise ValueError()

        self._cnn = cnn
        self._gated_trans = gated_trans
        self._iafs = iafs
        self._iafs_module_list = nn.ModuleList(iafs)
        self._emitter = emitter
        self._market_data_arr_mapper = market_data_arr_mapper
        self._traded_sym_arr_mapper = traded_sym_arr_mapper
        self._id_to_traded_sym = id_to_traded_sym

        self._n_traded_sym = n_traded_sym
        self._market_features = market_features
        self._max_trades = cnn.max_trades
        self._window_size = cnn.window_size
        self._hidden_state_size = gated_trans.z_dim
        self._leverage = leverage

    @property
    def n_traded_sym(self) -> int:
        return self._n_traded_sym

    @property
    def market_features(self) -> int:
        return self._market_features

    @property
    def max_trades(self) -> int:
        return self._max_trades

    @property
    def hidden_state_size(self) -> int:
        return self._hidden_state_size

    @property
    def leverage(self) -> float:
        return self._leverage

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def cnn(self) -> CNN:
        return self._cnn

    @property
    def gated_trans(self) -> GatedTransition:
        return self._gated_trans

    @property
    def iafs(self) -> Sequence[nn.Module]:
        return self._iafs

    @property
    def emitter(self) -> Emitter:
        return self._emitter

    @property
    def market_data_arr_mapper(self) -> MarketDataArrayMapper:
        return self._market_data_arr_mapper

    @property
    def traded_sym_arr_mapper(self) -> DictBasedArrayMapper[int]:
        return self._traded_sym_arr_mapper

    @property
    def id_to_traded_sym(self) -> Mapping[int, TradedSymbolInfo]:
        return self._id_to_traded_sym


class MinStepMax(nn.Module):
    def __init__(self, min: torch.Tensor, step: torch.Tensor, max: torch.Tensor):
        super().__init__()

        if not min.ndim == step.ndim == max.ndim == 1:
            raise ValueError()

        if not (size := min.numel()) == step.numel() == max.numel():
            raise ValueError()

        self._min: torch.Tensor
        self._step: torch.Tensor
        self._max: torch.Tensor
        self.register_buffer("_min", min)
        self.register_buffer("_step", step)
        self.register_buffer("_max", max)

        self._size = size

    @property
    def size(self) -> int:
        return self._size

    @property
    def min(self) -> torch.Tensor:
        return self._min

    @property
    def step(self) -> torch.Tensor:
        return self._step

    @property
    def max(self) -> torch.Tensor:
        return self._max


# TODO: maybe rename to ModelEvaluator?
class LowLevelInferenceEngine(nn.Module):
    def __init__(self, net_modules: NetworkModules) -> None:
        super().__init__()
        self._net_modules = net_modules
        self._min_step_max = self._compute_min_step_max_arr(net_modules)
        self._scaling_idx = net_modules.market_data_arr_mapper.scaling_idxs
        self._leverage = net_modules.leverage
        self._n_traded_sym = net_modules.n_traded_sym

    @property
    def net_modules(self) -> NetworkModules:
        return self._net_modules

    @staticmethod
    def _compute_min_step_max_arr(net_modules: NetworkModules) -> MinStepMax:
        min_step_max_map = {
            sym_id: (
                syminfo.min_volume / 100,
                syminfo.step_volume / 100,
                syminfo.max_volume / 100,
            )
            for sym_id, syminfo in net_modules.id_to_traded_sym.items()
        }

        min, step, max = map(
            torch.from_numpy,  # type: ignore
            map_to_arrays(
                net_modules.traded_sym_arr_mapper, min_step_max_map, dtype=np.float32
            ),
        )

        return MinStepMax(min=min, step=step, max=max)

    @jit.export  # type: ignore
    def _compute_new_pos_sizes(
        self,
        fractions: torch.Tensor,
        exec_mask: torch.Tensor,
        pos_sizes: torch.Tensor,
        margin_rate: torch.Tensor,
        unused_margin: torch.Tensor,
    ) -> torch.Tensor:
        new_pos_sizes = torch.empty_like(pos_sizes)  # type: ignore
        leverage = self._leverage
        min_step_max = self._min_step_max

        for i in range(self._n_traded_sym):
            fraction = fractions[i]
            pos_size = pos_sizes[i]

            unused_size = unused_margin * margin_rate[i] * leverage
            max_pos_size = pos_size.abs() + unused_size

            new_pos_size = fraction * (
                max_pos_size
                - torch.where(
                    pos_size * fraction > 0, unused_size, max_pos_size
                ).minimum(max_pos_size.new_zeros([]))
            )

            abs_new_pos_size = new_pos_size.abs()
            new_pos_sign = new_pos_size.sign()

            step = min_step_max.step[i]
            abs_new_pos_size = torch.floor(abs_new_pos_size / step) * step
            abs_new_pos_size = abs_new_pos_size.new_zeros([]).where(
                abs_new_pos_size < min_step_max.min[i], abs_new_pos_size
            )
            abs_new_pos_size = abs_new_pos_size.minimum(min_step_max.max[i])

            new_pos_sizes[i] = torch.where(
                exec_mask[i], new_pos_sign * abs_new_pos_size, pos_size
            )
            unused_margin = (max_pos_size - abs_new_pos_size) / (
                margin_rate[i] * leverage
            )

        return new_pos_sizes

    def evaluate(self, model_input: ModelInput) -> ModelOutput:
        market_state = model_input.market_state

        # TODO: check input?

        scaled_market_data = market_state.market_data.clone()
        for src_ind, dst_inds in self._scaling_idx:
            scaled_market_data[..., dst_inds, :] /= scaled_market_data[  # type: ignore
                ..., src_ind, None, -1, None
            ]

        assert scaled_market_data.isfinite().all()

        account_state = model_input.account_state
        rel_prices = account_state.trades_prices / market_state.midpoint_prices

        trades_used_margins = account_state.trades_sizes / (
            market_state.margin_rate * self._leverage
        )
        pos_used_margins = trades_used_margins.sum(0)

        balance = account_state.balance
        unused_margin = balance - pos_used_margins.abs().sum(0)
        # unused_margin = unused_margin.maximum(unused_margin.new_zeros([]))

        rel_margins = (
            torch.cat((trades_used_margins.flatten(0, 1), unused_margin.unsqueeze(0)))
            / balance
        )
        trades_data = torch.cat((rel_margins, rel_prices.flatten(0, 1))).movedim(0, -1)

        model_output = self._evaluate(
            trades_data, scaled_market_data, model_input.hidden_state
        )

        new_pos_sizes = self._compute_new_pos_sizes(
            model_output.fractions,
            model_output.exec_mask,
            account_state.pos_size,
            market_state.margin_rate,
            unused_margin,
        )

        assert (account_state.pos_size == new_pos_sizes)[~model_output.exec_mask].all()

        return ModelOutput(new_pos_sizes, model_output)

    # TODO: find better name
    def _evaluate(
        self,
        trades_data: torch.Tensor,
        scaled_market_data: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> RawModelOutput:
        net_modules = self.net_modules

        cnn_output: torch.Tensor = net_modules.cnn(
            scaled_market_data.flatten(0, -3), trades_data.flatten(0, -2)
        )
        batch_shape = trades_data[..., 0].shape
        cnn_output = cnn_output.unflatten(0, batch_shape)  # type: ignore

        z_loc: torch.Tensor
        z_scale: torch.Tensor
        z_loc, z_scale = net_modules.gated_trans(cnn_output, hidden_state)

        # z_dist = dist.TransformedDistribution(
        #     dist.Normal(z_loc, z_scale), net_modules.iafs
        # )0
        z_dist = dist.Independent(dist.Normal(z_loc, z_scale), 1)
        z_sample: torch.Tensor = z_dist.rsample()  # type: ignore
        z_logprob: torch.Tensor = z_dist.log_prob(z_sample)  # type: ignore

        exec_logits: torch.Tensor
        fractions: torch.Tensor
        exec_logits, fractions = net_modules.emitter(z_sample)

        exec_logits = exec_logits.movedim(-1, 0)
        fractions = fractions.movedim(-1, 0)

        exec_dist = dist.Bernoulli(logits=exec_logits)
        exec_samples: torch.Tensor = exec_dist.sample()  # type: ignore
        exec_logprobs: torch.Tensor = exec_dist.log_prob(exec_samples)  # type: ignore

        return RawModelOutput(
            cnn_output=cnn_output,
            z_sample=z_sample,
            z_logprob=z_logprob,
            exec_samples=exec_samples,
            exec_logprobs=exec_logprobs,
            fractions=fractions,
        )

    @staticmethod
    def _get_validity_mask(sizes_or_trades: torch.Tensor) -> torch.Tensor:
        trade_open_mask = sizes_or_trades != 0

        mask_cumsum = trade_open_mask.cumsum(0)
        open_trades_counts = mask_cumsum[-1] - mask_cumsum

        expected_open_trades_counts = torch.arange(
            open_trades_counts.shape[0] - 1, -1, -1, device=open_trades_counts.device
        )[(slice(None), *(None,) * (open_trades_counts.ndim - 1))]

        all_following_trades_open = open_trades_counts == expected_open_trades_counts
        return torch.all(~trade_open_mask | all_following_trades_open, dim=0)

    @classmethod
    def open_trades(
        cls,
        account_state: AccountState,
        new_pos_size: torch.Tensor,
        open_price: torch.Tensor,
    ) -> tuple[AccountState, torch.Tensor]:
        pos_size = account_state.pos_size
        trades_sizes = account_state.trades_sizes
        available_trade_mask = account_state.available_trades_mask

        open_trade_size = new_pos_size - pos_size
        new_pos_mask = (pos_size == 0) & (open_trade_size != 0)
        valid_trade_mask = new_pos_mask | (pos_size * open_trade_size > 0)
        assert torch.all(valid_trade_mask | (open_trade_size == 0))
        open_trade_mask = available_trade_mask & valid_trade_mask

        new_trades_sizes = trades_sizes.clone()
        new_trades_sizes[:-1] = trades_sizes[1:]
        new_trades_sizes[-1] = open_trade_size
        new_trades_sizes = new_trades_sizes.where(open_trade_mask, trades_sizes)

        trades_prices = account_state.trades_prices
        new_trades_prices = trades_prices.clone()
        new_trades_prices[:-1] = trades_prices[1:]
        new_trades_prices[-1] = open_price
        new_trades_prices = new_trades_prices.where(open_trade_mask, trades_prices)

        assert not torch.any((new_trades_sizes == 0) != (new_trades_prices == 0))
        assert torch.all(
            torch.all(new_trades_sizes >= 0, dim=0)
            | torch.all(new_trades_sizes <= 0, dim=0)
        )
        assert cls._get_validity_mask(new_trades_sizes).all()

        new_account_state = AccountState(
            new_trades_sizes, new_trades_prices, account_state.balance
        )
        assert (new_account_state.pos_size == new_pos_size)[open_trade_mask].all()

        return new_account_state, open_trade_mask

    @classmethod
    def close_or_reduce_trades(
        cls,
        account_state: AccountState,
        new_pos_size: torch.Tensor,
        # close_price: torch.Tensor,
        # update_balance: bool = True,
    ) -> tuple[AccountState, torch.Tensor]:
        trades_sizes = account_state.trades_sizes
        pos_size = account_state.pos_size

        close_trade_size = new_pos_size - pos_size
        right_diffs = close_trade_size + trades_sizes.cumsum(0)

        left_diffs = torch.empty_like(right_diffs)
        left_diffs[1:] = right_diffs[:-1]
        left_diffs[0] = close_trade_size

        close_trade_mask = (pos_size != 0) & (pos_size * right_diffs <= 0)
        reduce_trade_mask = left_diffs * right_diffs < 0

        closed_trades_sizes = torch.zeros_like(trades_sizes)

        assert not torch.any(close_trade_mask & reduce_trade_mask)
        assert torch.all(reduce_trade_mask.sum(0) <= 1)

        new_trades_sizes = trades_sizes.clone()
        new_trades_sizes[close_trade_mask] = 0.0
        new_trades_sizes[reduce_trade_mask] = right_diffs[reduce_trade_mask]

        closed_trades_sizes[close_trade_mask] = trades_sizes[close_trade_mask]
        closed_trades_sizes[reduce_trade_mask] = -left_diffs[reduce_trade_mask]

        trades_prices = account_state.trades_prices
        new_trades_prices = trades_prices.clone()
        new_trades_prices[close_trade_mask] = 0.0

        assert not torch.any((new_trades_sizes == 0) != (new_trades_prices == 0))
        assert torch.all(
            torch.all(new_trades_sizes >= 0, dim=0)
            | torch.all(new_trades_sizes <= 0, dim=0)
        )
        assert cls._get_validity_mask(new_trades_sizes).all()

        new_account_state = AccountState(
            new_trades_sizes, new_trades_prices, account_state.balance
        )

        close_or_reduce_mask = torch.any(close_trade_mask | reduce_trade_mask, dim=0)
        close_pos_size = closed_trades_sizes.sum(0)
        assert torch.all(new_account_state.pos_size == pos_size - close_pos_size)

        assert (
            close_pos_size
            == torch.where(
                close_trade_size.abs() > pos_size.abs(), pos_size, -close_trade_size
            )
        )[close_or_reduce_mask].all()

        return new_account_state, closed_trades_sizes

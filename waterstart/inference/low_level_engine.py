from __future__ import annotations

from collections.abc import Collection, Sequence

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
from .utils import AccountState, ModelOutput, RawModelOutput, MarketState


class NetworkModules(nn.Module):
    def __init__(
        self,
        cnn: CNN,
        gated_trans: GatedTransition,
        emitter: Emitter,
        market_data_arr_mapper: MarketDataArrayMapper,
        traded_sym_arr_mapper: DictBasedArrayMapper[int],
        traded_symbols: Collection[TradedSymbolInfo],
        max_trades: int,
        leverage: float,
    ):
        super().__init__()

        raw_market_data_size = market_data_arr_mapper.n_fields

        if raw_market_data_size != cnn.raw_market_data_size:
            raise ValueError()

        id_to_traded_sym = {sym.id: sym for sym in traded_symbols}

        if len(traded_symbols) < len(id_to_traded_sym):
            raise ValueError()

        if id_to_traded_sym.keys() != traded_sym_arr_mapper.keys:
            raise ValueError()

        self.cnn = cnn
        self.gated_trans = gated_trans
        self.emitter = emitter
        self.market_data_arr_mapper = market_data_arr_mapper
        self.traded_sym_arr_mapper = traded_sym_arr_mapper
        self.traded_symbols = traded_symbols

        self._n_traded_sym = emitter.n_traded_sym
        self._raw_market_data_size = raw_market_data_size
        self._max_trades = max_trades
        self._window_size = cnn.window_size
        self._hidden_state_size = gated_trans.z_dim
        self._leverage = leverage

    @property
    def n_traded_sym(self) -> int:
        return self._n_traded_sym

    @property
    def raw_market_data_size(self) -> int:
        return self._raw_market_data_size

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

    # @property
    # def cnn(self) -> CNN:
    #     return self._cnn

    # @property
    # def gated_trans(self) -> GatedTransition:
    #     return self._gated_trans

    # @property
    # def emitter(self) -> Emitter:
    #     return self._emitter

    # @property
    # def market_data_arr_mapper(self) -> MarketDataArrayMapper:
    #     return self._market_data_arr_mapper

    # @property
    # def traded_sym_arr_mapper(self) -> DictBasedArrayMapper[int]:
    #     return self._traded_sym_arr_mapper

    # @property
    # def id_to_traded_sym(self) -> Mapping[int, TradedSymbolInfo]:
    #     return self._id_to_traded_sym


class MinStepMax(nn.Module):
    def __init__(self, min: torch.Tensor, step: torch.Tensor, max: torch.Tensor):
        super().__init__()

        if not min.ndim == step.ndim == max.ndim == 1:
            raise ValueError()

        if not (size := min.numel()) == step.numel() == max.numel():
            raise ValueError()

        min_step_max = torch.stack((min, step, max))

        self._min_step_max: torch.Tensor
        self.register_buffer("_min_step_max", min_step_max)

        self._size = size

    @property
    def size(self) -> int:
        return self._size

    @property
    def min(self) -> torch.Tensor:
        return self._min_step_max[0]

    @property
    def step(self) -> torch.Tensor:
        return self._min_step_max[1]

    @property
    def max(self) -> torch.Tensor:
        return self._min_step_max[2]


# TODO: maybe rename to ModelEvaluator?
class LowLevelInferenceEngine(nn.Module):
    def __init__(self, net_modules: NetworkModules) -> None:
        super().__init__()
        self._net_modules = net_modules
        self._min_step_max = self._compute_min_step_max_arr(net_modules)
        scaling_idxs = net_modules.market_data_arr_mapper.scaling_idxs
        self._scaling_idxs_arr = self._build_scaling_idxs_arr(scaling_idxs)

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

    @staticmethod
    def _build_scaling_idxs_arr(
        scaling_idxs: Sequence[tuple[int, list[int]]]
    ) -> torch.Tensor:
        flat_idxs = [
            ([src_ind] * len(dst_inds), dst_inds) for src_ind, dst_inds in scaling_idxs
        ]

        return torch.tensor(flat_idxs).movedim(1, 0).flatten(1, 2)

    @jit.export  # type: ignore
    def _compute_new_pos_sizes(
        self,
        fractions: torch.Tensor,
        exec_samples: torch.Tensor,
        pos_sizes: torch.Tensor,
        margin_rates: torch.Tensor,
        unused_margin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_pos_sizes = torch.empty_like(pos_sizes)  # type: ignore
        leverage = self._leverage
        min_step_max = self._min_step_max

        dtype = unused_margin.dtype
        unused_margin = unused_margin.double()

        for i in range(self._n_traded_sym):
            fraction = fractions[i]
            exec_sample = exec_samples[i]
            pos_size = pos_sizes[i]
            margin_rate = margin_rates[i]

            unused_size = unused_margin * margin_rate * leverage
            max_pos_size = pos_size.abs() + unused_size

            new_pos_size = fraction * (
                max_pos_size
                - torch.where(
                    pos_size * fraction > 0, unused_size, max_pos_size
                ).minimum(max_pos_size.new_zeros(()))
            )

            abs_new_pos_size = new_pos_size.abs()
            new_pos_sign = new_pos_size.sign()

            step = min_step_max.step[i]
            abs_new_pos_size = torch.floor(abs_new_pos_size / step) * step
            abs_new_pos_size = abs_new_pos_size.new_zeros(()).where(
                abs_new_pos_size < min_step_max.min[i], abs_new_pos_size
            )
            abs_new_pos_size = abs_new_pos_size.minimum(min_step_max.max[i])
            # max = min_step_max.max[i]
            # new_pos_size = new_pos_size.clamp(-max, max)

            new_pos_sizes[i] = (  # type: ignore
                exec_sample * (new_pos_sign * abs_new_pos_size)
                # exec_sample * new_pos_size
                + (1 - exec_sample) * pos_size
            )
            unused_margin = (max_pos_size - abs_new_pos_size) / (margin_rate * leverage)
            # unused_margin = (max_pos_size - new_pos_size.abs()) / (
            #     margin_rate * leverage
            # )

        return new_pos_sizes, unused_margin.to(dtype)

    def extract_market_features(self, raw_market_data: torch.Tensor) -> torch.Tensor:
        scaled_market_data = raw_market_data.clone()

        src_idxs, dst_idxs = self._scaling_idxs_arr.unbind()

        # scaled_market_data[..., dst_idxs, :] = raw_market_data[..., dst_idxs, :].log()
        scaled_market_data[..., dst_idxs, :] = (
            scaled_market_data[..., dst_idxs, :]
            / scaled_market_data[..., src_idxs, -1, None]
            - 1
        )

        assert scaled_market_data.isfinite().all()

        market_features: torch.Tensor = self._net_modules.cnn(
            scaled_market_data.flatten(0, -3)
        )
        batch_shape = scaled_market_data[..., 0, 0].shape
        market_features = market_features.unflatten(0, batch_shape)  # type: ignore

        return market_features

    def _compute_trades_data(
        self,
        market_state: MarketState,
        account_state: AccountState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bid_prices, ask_prices = market_state.bid_ask_prices

        trades_sizes = account_state.trades_sizes
        pos_size = account_state.pos_size
        balance = account_state.balance

        close_price = torch.where(
            pos_size > 0,
            bid_prices,
            torch.where(pos_size < 0, ask_prices, bid_prices.new_zeros(())),
        )
        profits_losses = torch.sum(
            trades_sizes
            * (close_price - account_state.trades_prices)
            / market_state.quote_to_dep_rate,
            dim=0,
        )

        pos_used_margins = pos_size / (market_state.margin_rate * self._leverage)
        unused_margin = balance - pos_used_margins.abs().sum(0)
        tot_profit_loss = profits_losses.sum(0)

        used_trades_fracs = torch.sum(trades_sizes != 0, dim=0) / trades_sizes.shape[0]

        trades_data = torch.stack(
            (profits_losses / balance, pos_used_margins / balance, used_trades_fracs)
        )
        trades_data = torch.cat(
            (
                trades_data.flatten(0, 1),
                torch.unsqueeze(unused_margin / balance, dim=0),
                torch.unsqueeze(tot_profit_loss / balance, dim=0),
            )
        ).movedim(0, -1)

        return trades_data, tot_profit_loss, unused_margin

    def evaluate(
        self,
        market_features: torch.Tensor,
        # hidden_state: torch.Tensor,
        market_state: MarketState,
        account_state: AccountState,
    ) -> ModelOutput:
        trades_data, tot_profit_loss, unused_margin = self._compute_trades_data(
            market_state, account_state
        )

        balance = account_state.balance
        # TODO: what if unused_magin is negative?
        account_value = balance + tot_profit_loss

        used_margin = balance - unused_margin
        closeout_mask = account_value < 0.5 * used_margin

        # z_sample, exec_sample, fracs, logprob = self._evaluate(
        # z_sample, fracs, logprob = self._evaluate(
        # signs, abs_fracs, logprob = self._evaluate(
        fracs, logprob = self._evaluate(
            trades_data,
            market_features,
            # hidden_state,
        )

        # exec_sample = exec_sample.new_ones(()).where(closeout_mask, exec_sample)
        # fracs = fracs.new_zeros(()).where(closeout_mask, fracs)

        fracs = fracs.movedim(-1, 0)
        new_pos_margins = fracs * balance
        new_pos_sizes = new_pos_margins * market_state.margin_rate * self._leverage

        new_pos_sizes = new_pos_sizes.movedim(0, -1)

        abs_new_pos_sizes = new_pos_sizes.abs()
        new_pos_signs = new_pos_sizes.sign()

        min_step_max = self._min_step_max
        step = min_step_max.step
        abs_new_pos_sizes = torch.floor(new_pos_sizes / step) * step
        abs_new_pos_sizes = abs_new_pos_sizes.new_zeros(()).where(
            abs_new_pos_sizes < min_step_max.min, abs_new_pos_sizes
        )
        abs_new_pos_sizes = abs_new_pos_sizes.minimum(min_step_max.max)

        new_pos_sizes = new_pos_signs * abs_new_pos_sizes
        new_pos_sizes = new_pos_sizes.movedim(-1, 0)

        # new_pos_sizes: torch.Tensor = (  # type: ignore
        #     exec_sample * new_pos_sizes + (1 - exec_sample) * account_state.pos_size
        # )

        new_pos_sizes = new_pos_sizes.new_zeros(()).where(closeout_mask, new_pos_sizes)

        # new_pos_sizes, new_unused_margin = self._compute_new_pos_sizes(
        #     fracs,
        #     exec_samples,
        #     account_state.pos_size,
        #     market_state.margin_rate,
        #     unused_margin,
        # )

        # assert torch.all((unused_margin > 0) | (new_unused_margin >= unused_margin))
        # assert (account_state.pos_size == new_pos_sizes)[~exec_mask].all()

        return ModelOutput(
            new_pos_sizes,
            RawModelOutput(
                trades_data=trades_data,
                # z_sample=z_sample,
                market_features=market_features,
                # fracs=fracs,
                logprob=logprob,
            ),
        )

    # TODO: find better name
    def _evaluate(
        self,
        trades_data: torch.Tensor,
        market_features: torch.Tensor,
        # z0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        net_modules = self._net_modules

        # z_loc: torch.Tensor
        # z_scale: torch.Tensor
        # z_loc, z_scale = net_modules.gated_trans(
        #     market_features.flatten(0, -2), z0.flatten(0, -2)
        # )
        # batch_shape = market_features[..., 0].shape
        # z_loc = z_loc.unflatten(0, batch_shape)  # type: ignore
        # z_scale = z_scale.unflatten(0, batch_shape)  # type: ignore

        # # z_dist = dist.TransformedDistribution(
        # #     dist.Independent(dist.Normal(z_loc, z_scale), 1),
        # #     dist.transforms.TanhTransform(cache_size=1),
        # # )
        # z_dist = dist.Independent(dist.Normal(z_loc, z_scale), 1)
        # z_sample: torch.Tensor = z_dist.rsample()  # type: ignore

        # exec_logit: torch.Tensor
        # sign_logits: torch.Tensor
        # frac_conc: torch.Tensor
        frac_loc: torch.Tensor
        frac_scale: torch.Tensor
        # exec_logit, sign_logits, frac_conc = net_modules.emitter(z_sample, trades_data)
        # exec_logit, sign_logits, frac_conc = net_modules.emitter(
        frac_loc, frac_scale = net_modules.emitter(market_features, trades_data)

        # exec_dist = dist.Bernoulli(logits=exec_logit)
        # exec_sample: torch.Tensor = exec_dist.sample()  # type: ignore
        # exec_logprob: torch.Tensor = exec_dist.log_prob(exec_sample)  # type: ignore

        # sign_dist = dist.TransformedDistribution(
        #     dist.Bernoulli(logits=sign_logits),
        #     dist.transforms.AffineTransform(-1, 2, event_dim=1),
        # )
        # sign_samples: torch.Tensor = sign_dist.sample()  # type: ignore
        # sign_logprob: torch.Tensor = sign_dist.log_prob(sign_samples)  # type: ignore

        # frac_dist = dist.Independent(dist.Dirichlet(frac_conc), 1)
        frac_dist = dist.Independent(dist.Normal(frac_loc, frac_scale), 1)
        frac_samples: torch.Tensor = frac_dist.sample()  # type: ignore
        frac_logprob: torch.Tensor = frac_dist.log_prob(frac_samples)  # type: ignore

        fracs = frac_samples / frac_samples.abs().sum(-1, keepdim=True)

        # logprob = exec_logprob + sign_logprob + frac_logprob
        # logprob = sign_logprob + frac_logprob
        logprob = frac_logprob
        # fracs = sign_samples * frac_samples[..., :-1]

        # return z_sample, exec_sample, fracs, logprob
        # return z_sample, fracs, logprob
        # return sign_samples, frac_samples[..., :-1], logprob
        return fracs[..., :-1], logprob

    # TODO: make the following methods part of the AccountState class?
    @staticmethod
    def _get_validity_mask(sizes_or_trades: torch.Tensor) -> torch.Tensor:
        trade_open_mask = sizes_or_trades != 0

        mask_cumsum = trade_open_mask.cumsum(0)
        open_trades_counts = mask_cumsum

        expected_open_trades_counts = torch.arange(
            1, open_trades_counts.shape[0] + 1, device=open_trades_counts.device
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
        trades_prices = account_state.trades_prices
        available_trade_mask = account_state.available_trades_mask

        open_trade_size = new_pos_size - pos_size
        new_pos_mask = (pos_size == 0) & (open_trade_size != 0)
        # new_pos_mask = (pos_size == 0) & (new_pos_size != 0)
        # new_pos_mask = pos_size.isclose(pos_size.new_zeros(())) & ~new_pos_size.isclose(
        #     new_pos_size.new_zeros(())
        # )

        valid_trade_mask = new_pos_mask | (pos_size * open_trade_size > 0)
        assert torch.all(valid_trade_mask | (open_trade_size == 0))
        # assert torch.all(valid_trade_mask | torch.isclose(new_pos_size, pos_size))
        # assert torch.all(
        #     valid_trade_mask | open_trade_size.isclose(open_trade_size.new_zeros(()))
        # )
        open_trade_mask = available_trade_mask & valid_trade_mask

        new_trades_sizes = trades_sizes.clone()
        # new_trades_sizes[:-1] = trades_sizes[1:]
        # new_trades_sizes[-1] = open_trade_size
        new_trades_sizes[1:] = trades_sizes[:-1]
        new_trades_sizes[0] = open_trade_size

        new_trades_sizes = new_trades_sizes.where(open_trade_mask, trades_sizes)

        new_trades_prices = trades_prices.clone()
        # new_trades_prices[:-1] = trades_prices[1:]
        # new_trades_prices[-1] = open_price
        new_trades_prices[1:] = trades_prices[:-1]
        new_trades_prices[0] = open_price

        new_trades_prices = new_trades_prices.where(open_trade_mask, trades_prices)

        assert not new_trades_prices.isnan().any()
        assert not torch.any((new_trades_sizes == 0) != (new_trades_prices == 0))
        assert torch.all(
            torch.all(new_trades_sizes >= 0, dim=0)
            | torch.all(new_trades_sizes <= 0, dim=0)
        )
        assert cls._get_validity_mask(new_trades_sizes).all()

        new_account_state = AccountState(
            new_trades_sizes, new_trades_prices, account_state.balance
        )
        assert torch.all(
            ~open_trade_mask
            | (new_account_state.pos_size == new_pos_size)
            # ~open_trade_mask
            # | torch.isclose(new_account_state.pos_size, new_pos_size)
        )
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
        trades_prices = account_state.trades_prices
        pos_size = account_state.pos_size

        # close_trade_size = new_pos_size - pos_size
        # right_diffs = close_trade_size + trades_sizes.cumsum(0)
        right_diffs = trades_sizes.cumsum(0) - new_pos_size

        left_diffs = torch.empty_like(right_diffs)
        left_diffs[1:] = right_diffs[:-1]
        # left_diffs[0] = close_trade_size
        left_diffs[0] = -new_pos_size

        # close_trade_mask = (pos_size != 0) & (pos_size * right_diffs <= 0)
        close_trade_mask = (pos_size != 0) & (pos_size * left_diffs >= 0)
        reduce_trade_mask = left_diffs * right_diffs < 0

        closed_trades_sizes = torch.zeros_like(trades_sizes)

        assert not torch.any(close_trade_mask & reduce_trade_mask)
        assert torch.all(reduce_trade_mask.sum(0) <= 1)

        new_trades_sizes = trades_sizes.clone()
        new_trades_sizes[close_trade_mask] = 0.0
        # new_trades_sizes[reduce_trade_mask] = right_diffs[reduce_trade_mask]
        new_trades_sizes[reduce_trade_mask] = -left_diffs[reduce_trade_mask]

        closed_trades_sizes[close_trade_mask] = trades_sizes[close_trade_mask]
        # closed_trades_sizes[reduce_trade_mask] = -left_diffs[reduce_trade_mask]
        closed_trades_sizes[reduce_trade_mask] = right_diffs[reduce_trade_mask]

        new_trades_prices = trades_prices.clone()
        new_trades_prices[close_trade_mask] = 0.0

        assert not torch.any((new_trades_sizes == 0) != (new_trades_prices == 0))
        assert torch.all(
            torch.all((new_trades_sizes >= 0) & (closed_trades_sizes >= 0), dim=0)
            | torch.all((new_trades_sizes <= 0) & (closed_trades_sizes <= 0), dim=0)
        )
        assert cls._get_validity_mask(new_trades_sizes).all()

        new_account_state = AccountState(
            new_trades_sizes, new_trades_prices, account_state.balance
        )

        close_or_reduce_mask = torch.any(close_trade_mask | reduce_trade_mask, dim=0)
        close_pos_size = closed_trades_sizes.sum(0)
        assert torch.all(pos_size == new_account_state.pos_size + close_pos_size)
        # assert torch.isclose(
        #     pos_size, new_account_state.pos_size + close_pos_size
        # ).all()

        assert torch.all(
            ~close_or_reduce_mask
            | (pos_size * new_pos_size > 0)
            | (new_account_state.pos_size == 0)
        )

        assert torch.all(
            ~close_or_reduce_mask
            | (pos_size * new_pos_size < 0)
            | (new_account_state.pos_size == new_pos_size)
            # | torch.isclose(new_account_state.pos_size, new_pos_size)
        )

        assert torch.all(new_account_state.pos_size * new_pos_size >= 0)

        # assert torch.all(
        #     ~close_or_reduce_mask
        #     | (
        #         close_pos_size
        #         == torch.where(
        #             close_trade_size.abs() > pos_size.abs(),
        #             pos_size,
        #             -close_trade_size,
        #         )
        #     )
        # )

        return new_account_state, closed_trades_sizes

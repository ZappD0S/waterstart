from dataclasses import dataclass
import torch
import torch.nn as nn

from ..inference import ModelInput, AccountState
from ..inference.low_level_engine import LowLevelInferenceEngine


class Critic(nn.Module):
    def __init__(self, n_features: int, z_dim: int, hidden_dim: int, n_traded_sym: int):
        super().__init__()
        self._gru = nn.GRUCell(n_features, z_dim)
        self._lin1 = nn.Linear(z_dim, hidden_dim)
        self._lin2 = nn.Linear(hidden_dim, 1)
        # self.lin3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self._gru(x, h)
        out = self._lin1(out).relu()

        return self._lin2(out).squeeze(-1)


@dataclass
class LossOutput:
    loss: torch.Tensor
    surrogate_loss: torch.Tensor
    account_state: AccountState
    hidden_state: torch.Tensor


class LossEvaluator(nn.Module):
    def __init__(self, engine: LowLevelInferenceEngine, critic: Critic) -> None:
        super().__init__()
        self._engine = engine
        self._net_modules = engine.net_modules
        self._critic = critic

    def forward(self, model_input: ModelInput) -> LossOutput:  # type: ignore
        market_state = model_input.market_state
        old_account_state = model_input.account_state

        model_infer = self._engine.evaluate(model_input)
        new_pos_size = model_infer.pos_sizes
        raw_model_output = model_infer.raw_model_output

        midpoint_prices = market_state.midpoint_prices
        half_spreads = market_state.spreads / 2
        bid_prices = midpoint_prices - half_spreads
        ask_prices = midpoint_prices + half_spreads

        trade_size = new_pos_size - old_account_state.pos_size
        trade_price = torch.where(
            trade_size > 0,
            ask_prices,
            torch.where(trade_size < 0, bid_prices, bid_prices.new_zeros([])),
        )

        account_state, closed_trades_sizes = self._engine.close_or_reduce_trades(
            old_account_state, new_pos_size
        )

        closed_size = closed_trades_sizes.sum(0)
        close_price = torch.where(
            closed_size > 0,
            ask_prices,
            torch.where(closed_size < 0, bid_prices, bid_prices.new_zeros([])),
        )
        profit_loss = torch.sum(
            closed_trades_sizes
            * (close_price - old_account_state.trades_prices)
            / market_state.quote_to_dep_rate,
            dim=[0, 1],
        )

        old_balance = old_account_state.balance
        ratio = profit_loss / old_balance
        assert torch.all(ratio > -1)
        rewards = torch.log1p(ratio)

        exec_logprobs = raw_model_output.exec_logprobs
        tot_logprob = raw_model_output.z_logprob + exec_logprobs.sum(0)

        hidden_state = model_input.hidden_state
        batch_shape = hidden_state[..., 0].shape
        values: torch.Tensor = self._critic(
            raw_model_output.cnn_output,
            model_input.hidden_state.flatten(0, -2),
        )
        values = values.unflatten(0, batch_shape)  # type: ignore

        loss = -rewards.detach()
        deltas = rewards[:-1] + torch.detach(0.99 * values[1:] - values[:-1])

        surrogate_loss = tot_logprob[:-1] * deltas.detach() + deltas
        baseline_loss = (values[:-1] - (rewards[:-1].detach() + 0.99 * values[1:])) ** 2

        new_balance = old_balance + profit_loss
        assert torch.all(new_balance > 0)

        open_size = new_pos_size - account_state.pos_size
        open_price = torch.where(
            open_size > 0,
            ask_prices,
            torch.where(open_size < 0, bid_prices, bid_prices.new_zeros([])),
        )
        assert not open_price.requires_grad

        account_state, _ = self._engine.open_trades(
            AccountState(
                account_state.trades_sizes.detach(),
                account_state.trades_prices,
                new_balance.detach(),
            ),
            new_pos_size.detach(),
            open_price,
        )

        return LossOutput(
            loss,
            -surrogate_loss + 0.5 * baseline_loss,
            account_state,
            raw_model_output.z_sample.detach(),
        )

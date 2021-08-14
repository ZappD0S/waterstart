from dataclasses import dataclass
import torch
import torch.nn as nn

from ..inference import ModelInput, AccountState
from ..inference.low_level_engine import LowLevelInferenceEngine


# TODO: maybe use 3 layers?
class NeuralBaseline(nn.Module):
    def __init__(self, n_features: int, z_dim: int, hidden_dim: int, n_traded_sym: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features + z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_traded_sym)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.lin1(x).relu_()
        out = self.lin2(out)

        return out


@dataclass
class LossOutput:
    loss: torch.Tensor
    account_state: AccountState
    hidden_state: torch.Tensor


class LossEvaluator(nn.Module):
    def __init__(
        self, engine: LowLevelInferenceEngine, nn_baseline: NeuralBaseline
    ) -> None:
        super().__init__()
        self._engine = engine
        self._net_modules = engine.net_modules
        self._nn_baseline = nn_baseline

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

        profits_losses = torch.sum(
            closed_trades_sizes.abs()
            * (trade_price - old_account_state.trades_prices)
            / market_state.quote_to_dep_rate,
            dim=0,
        )

        old_balance = old_account_state.balance
        cum_balances = old_balance + profits_losses.cumsum(0)

        shifted_cum_balances = torch.empty_like(cum_balances)
        shifted_cum_balances[0] = old_balance
        shifted_cum_balances[1:] = cum_balances[:-1]

        ratio = profits_losses / shifted_cum_balances
        assert torch.all(ratio > -1)
        costs = torch.log1p_(ratio)

        exec_logprobs = raw_model_output.exec_logprobs
        exec_logprobs = exec_logprobs.where(
            torch.any(closed_trades_sizes != 0, dim=0),
            exec_logprobs.new_zeros([]),
        )

        tot_logprobs = raw_model_output.z_logprob + exec_logprobs

        nn_baseline_input = torch.cat(
            [raw_model_output.cnn_output.detach(), model_input.hidden_state], dim=-1
        )
        assert not nn_baseline_input.requires_grad
        baselines: torch.Tensor = self._nn_baseline(nn_baseline_input).movedim(-1, 0)

        # TODO: compute both regular loss a surrogate loss
        surrogate_loss = tot_logprobs * torch.detach_(costs - baselines) + costs
        baseline_loss = (costs.detach() - baselines) ** 2

        new_balance = cum_balances[-1].detach()
        assert torch.all(new_balance > 0)

        account_state, _ = self._engine.open_trades(
            AccountState(
                account_state.trades_sizes.detach(),
                account_state.trades_prices,
                new_balance,
            ),
            new_pos_size.detach(),
            trade_price,
        )

        return LossOutput(
            -surrogate_loss + baseline_loss,
            account_state,
            raw_model_output.z_sample.detach(),
        )

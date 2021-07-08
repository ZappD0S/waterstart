import torch
import torch.nn as nn

from ..inference import ModelInput, RawMarketState
from ..inference.low_level_engine import LowLevelInferenceEngine


# TODO: maybe use 3 layers?
class NeuralBaseline(nn.Module):
    def __init__(self, n_features: int, z_dim: int, hidden_dim: int, n_cur: int):
        super().__init__()
        self.lin1 = nn.Linear(n_features + z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_cur)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # x: (..., 2 * n_cur * (max_trades + 1) + z_dim + 1)

        out = self.lin1(x).relu_()
        out = self.lin2(out)

        return out


class LossEvaluator(nn.Module):
    def __init__(
        self, engine: LowLevelInferenceEngine, nn_baseline: NeuralBaseline
    ) -> None:
        super().__init__()
        self._engine = engine
        self._net_modules = engine.net_modules
        self._nn_baseline = nn_baseline

    def forward(self, model_input: ModelInput[RawMarketState]):  # type: ignore
        model_infer = self._engine.evaluate(model_input)
        raw_model_output = model_infer.raw_model_output
        old_trades_state = model_input.trades_state

        new_pos_sizes = model_infer.pos_sizes
        market_state = model_input.market_state

        midpoint_prices = market_state.midpoint_prices
        half_spreads = market_state.spreads / 2
        bid_prices = midpoint_prices - half_spreads
        ask_prices = midpoint_prices + half_spreads

        first_trade_size = old_trades_state.trades_sizes[-1]

        close_prices = torch.where(
            first_trade_size > 0,
            bid_prices,
            torch.where(first_trade_size < 0, ask_prices, midpoint_prices.new_ones([])),
        )

        open_prices = torch.where(
            new_pos_sizes > 0,
            ask_prices,
            torch.where(new_pos_sizes < 0, bid_prices, midpoint_prices.new_zeros([])),
        )

        trades_state = self._engine.close_or_reduce_trades(
            old_trades_state, new_pos_sizes
        )

        # TODO: is this always correct?
        closed_trades_sizes = trades_state.trades_sizes - old_trades_state.trades_sizes
        closed_mask = closed_trades_sizes > 0

        profits_losses = (
            closed_trades_sizes.abs()
            * (close_prices - old_trades_state.trades_prices)
            / market_state.quote_to_dep_rate
        )

        cum_balances = model_input.balance + profits_losses.cumsum(0)
        costs = torch.log1p_(profits_losses / cum_balances)

        exec_logprobs = raw_model_output.exec_logprobs
        exec_logprobs = exec_logprobs.where(closed_mask, exec_logprobs.new_zeros([]))

        tot_logprobs = raw_model_output.z_logprob + exec_logprobs

        nn_baseline_input = torch.cat(
            [raw_model_output.cnn_output.detach(), model_input.hidden_state], dim=-1
        )
        assert not nn_baseline_input.requires_grad
        baselines = self._nn_baseline(nn_baseline_input).movedim(3, 0)

        surrogate_loss = tot_logprobs * torch.detach_(costs - baselines) + costs
        baseline_loss = (costs.detach() - baselines) ** 2

        trades_state = self._engine.open_trades(
            old_trades_state, new_pos_sizes, open_prices
        )

        # TODO: make a dataclass?
        new_balance = cum_balances[-1]
        z_sample = raw_model_output.z_sample
        # TODO: return also z_samples and balance
        return surrogate_loss + baseline_loss, trades_state

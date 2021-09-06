import torch

from ...inference import ModelInput, AccountState
from ...inference.low_level_engine import LowLevelInferenceEngine
from .abc import BaseLossEvaluator
from .utils import LossOutput, Critic


class BufferedLossOutput(LossOutput):
    def __init__(
        self,
        loss: torch.Tensor,
        surrogate_loss: torch.Tensor,
        account_state: AccountState,
        hidden_state: torch.Tensor,
    ) -> None:
        super().__init__(loss, surrogate_loss, account_state)
        self.hidden_state = hidden_state


class BufferedLossEvaluator(BaseLossEvaluator):
    def __init__(
        self,
        engine: LowLevelInferenceEngine,
        critic: Critic,
        seq_len: int,
        gamma: float,
    ) -> None:
        super().__init__(seq_len, gamma)
        self._engine = engine
        self._net_modules = engine.net_modules
        self._critic = critic

    def evaluate(self, model_input: ModelInput) -> BufferedLossOutput:
        market_state = model_input.market_state
        old_account_state = model_input.account_state

        market_features = self._engine.extract_market_features(
            model_input.raw_market_data
        )
        model_infer = self._engine.evaluate(
            market_features, model_input.hidden_state, market_state, old_account_state
        )

        new_pos_size = model_infer.pos_sizes
        raw_model_output = model_infer.raw_model_output

        bid_prices, ask_prices = market_state.bid_ask_prices

        account_state, closed_trades_sizes = self._engine.close_or_reduce_trades(
            old_account_state, new_pos_size
        )

        closed_size = closed_trades_sizes.sum(0)
        close_price = torch.where(
            closed_size > 0,
            bid_prices,
            torch.where(closed_size < 0, ask_prices, bid_prices.new_zeros(())),
        )

        assert torch.all((close_price != 0) | (closed_size == 0))
        profit_loss = torch.sum(
            closed_trades_sizes
            * (close_price - old_account_state.trades_prices)
            / market_state.quote_to_dep_rate,
            dim=[0, 1],
        )

        balance = old_account_state.balance
        new_balance = balance + profit_loss
        assert torch.all(new_balance > 0)
        rewards = new_balance.log() - balance.log()

        values: torch.Tensor = self._critic(
            raw_model_output.z_sample, raw_model_output.trades_data
        )

        loss = -rewards.detach()

        advantages = self._compute_advantages(rewards, values.detach())

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = raw_model_output.fracs_logprobs[:-1].sum(0)
        action_loss = logprobs * advantages.detach() + advantages

        returns = torch.detach(advantages + values[:-1])
        baseline_loss = (values[:-1] - returns) ** 2
        surrogate_loss = -action_loss + 0.5 * baseline_loss

        open_size = new_pos_size - account_state.pos_size
        open_price = torch.where(
            open_size > 0,
            ask_prices,
            torch.where(
                open_size < 0, bid_prices, bid_prices.new_full((), float("nan"))
            ),
        )

        account_state, _ = self._engine.open_trades(
            AccountState(
                account_state.trades_sizes.detach(),
                account_state.trades_prices,
                new_balance.detach(),
            ),
            new_pos_size.detach(),
            open_price,
        )

        return BufferedLossOutput(
            loss, surrogate_loss, account_state, raw_model_output.z_sample.detach()
        )

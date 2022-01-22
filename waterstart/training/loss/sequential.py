from typing import Any, Callable, Optional
import torch
from torch.functional import Tensor
import torch.jit as jit

from ...inference import AccountState, MarketState, ModelInput
from ...inference import LowLevelInferenceEngine
from ..loss import Critic
from .abc import BaseLossEvaluator
from .utils import LossOutput


class SequentialLossEvaluator(BaseLossEvaluator):
    def __init__(
        self,
        engine: LowLevelInferenceEngine,
        critic: Critic,
        seq_len: int,
        gamma: float,
        trace: bool = False,
    ) -> None:
        super().__init__(seq_len, gamma)
        self._engine = engine
        self._net_modules = engine.net_modules

        self._net_modules.hidden_state_size
        self._net_modules.max_trades
        self._net_modules.n_traded_sym
        self._critic = critic
        self._trace = trace

        # self._traced_evaluate: Optional[Callable[[dict[str, Any]], dict[str, Any]]]
        self._traced_evaluate = None

    def _traceable_evaluate(
        self,
        account_state_map: dict[str, torch.Tensor],
        market_state_map: dict[str, torch.Tensor],
        raw_market_data: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        account_state = AccountState(
            trades_sizes=account_state_map["trades_sizes"],
            trades_prices=account_state_map["trades_prices"],
            balance=account_state_map["balance"],
        )

        market_state = MarketState(
            midpoint_prices=market_state_map["midpoint_prices"],
            spreads=market_state_map["spreads"],
            margin_rate=market_state_map["margin_rate"],
            quote_to_dep_rate=market_state_map["quote_to_dep_rate"],
        )

        loss_out = self.evaluate(
            ModelInput(
                account_state,
                market_state,
                raw_market_data,
                hidden_state,
            )
        )

        account_state = loss_out.account_state

        return (
            loss_out.loss,
            loss_out.surrogate_loss,
            {
                "trades_sizes": account_state.trades_sizes,
                "trades_prices": account_state.trades_prices,
                "balance": account_state.balance,
            },
        )

    def evaluate(self, model_input: ModelInput) -> LossOutput:
        if not self._trace:
            return self._evaluate_impl(model_input)

        account_state = model_input.account_state
        market_state = model_input.market_state

        input = (
            {
                "trades_sizes": account_state.trades_sizes,
                "trades_prices": account_state.trades_prices,
                "balance": account_state.balance,
            },
            {
                "midpoint_prices": market_state.midpoint_prices,
                "spreads": market_state.spreads,
                "margin_rate": market_state.margin_rate,
                "quote_to_dep_rate": market_state.quote_to_dep_rate,
            },
            model_input.raw_market_data,
            model_input.hidden_state,
        )

        if self._traced_evaluate is None:
            traced: SequentialLossEvaluator
            traced = jit.trace_module(  # type: ignore
                self, {"_traceable_evaluate": input}, check_trace=False
            )
            self._traced_evaluate = traced._traceable_evaluate

        loss, surrogate_loss, account_state_map = self._traced_evaluate(input)

        return LossOutput(
            loss=loss,
            surrogate_loss=surrogate_loss,
            account_state=AccountState(
                trades_sizes=account_state_map["trades_sizes"],
                trades_prices=account_state_map["trades_prices"],
                balance=account_state_map["balance"],
            ),
        )

    def _evaluate_impl(self, model_input: ModelInput) -> LossOutput:
        market_state = model_input.market_state
        account_state = model_input.account_state
        # z0 = model_input.hidden_state
        bid_prices, ask_prices = market_state.bid_ask_prices

        market_features = self._engine.extract_market_features(
            model_input.raw_market_data
        )

        midpoint_prices = market_state.midpoint_prices.movedim(1, -1)
        spreads = market_state.spreads.movedim(1, -1)
        margin_rate = market_state.margin_rate.movedim(1, -1)
        quote_to_dep_rate = market_state.quote_to_dep_rate.movedim(1, -1)
        bid_prices = bid_prices.movedim(1, -1)
        ask_prices = ask_prices.movedim(1, -1)

        batch_size = market_features.shape[1]
        n_traded_sym = self._net_modules.n_traded_sym
        max_trades = self._net_modules.max_trades
        seq_len = self._seq_len

        rewards = market_features.new_empty((seq_len, batch_size))
        values = market_features.new_empty((seq_len, batch_size))
        logprobs = market_features.new_empty((seq_len, batch_size))

        balances = market_features.new_empty((seq_len, batch_size))
        trades_sizes = market_features.new_empty(
            (seq_len, max_trades, n_traded_sym, batch_size)
        )
        trades_prices = market_features.new_empty(
            (seq_len, max_trades, n_traded_sym, batch_size)
        )

        for i in range(seq_len):
            model_infer = self._engine.evaluate(
                market_features[i],
                # z0,
                MarketState(
                    midpoint_prices=midpoint_prices[i],
                    spreads=spreads[i],
                    margin_rate=margin_rate[i],
                    quote_to_dep_rate=quote_to_dep_rate[i],
                ),
                account_state,
            )

            new_pos_size = model_infer.pos_sizes
            raw_model_output = model_infer.raw_model_output

            (
                new_account_state,
                closed_trades_sizes,
            ) = self._engine.close_or_reduce_trades(account_state, new_pos_size)

            closed_size = closed_trades_sizes.sum(0)
            close_price = torch.where(
                closed_size > 0,
                bid_prices[i],
                torch.where(closed_size < 0, ask_prices[i], bid_prices.new_zeros(())),
            )

            assert torch.all((close_price != 0) | (closed_size == 0))
            profit_loss = torch.sum(
                closed_trades_sizes
                * (close_price - account_state.trades_prices)
                / quote_to_dep_rate[i],
                dim=[0, 1],
            )

            balance = account_state.balance
            new_balance = balance + profit_loss
            assert torch.all(new_balance > 0)
            reward = new_balance.log() - balance.log()
            # reward = profit_loss

            value: torch.Tensor = self._critic(
                # raw_model_output.z_sample, raw_model_output.trades_data
                raw_model_output.market_features, raw_model_output.trades_data
            )

            values[i] = value
            rewards[i] = reward

            logprobs[i] = raw_model_output.logprob

            assert torch.all(new_balance > 0)

            open_size = new_pos_size - new_account_state.pos_size
            open_price = torch.where(
                open_size > 0,
                ask_prices[i],
                torch.where(
                    open_size < 0, bid_prices[i], bid_prices.new_full((), float("nan"))
                ),
            )

            new_account_state, _ = self._engine.open_trades(
                AccountState(
                    new_account_state.trades_sizes.detach(),
                    new_account_state.trades_prices,
                    new_balance.detach(),
                ),
                new_pos_size.detach(),
                open_price,
            )

            balances[i] = new_account_state.balance
            trades_sizes[i] = new_account_state.trades_sizes
            trades_prices[i] = new_account_state.trades_prices

            account_state = new_account_state
            # z0 = raw_model_output.z_sample

        loss = -rewards.detach()

        # this is: r_i + gamma * V(s_{i+1}) - V(s_i)
        deltas = torch.detach(rewards[:-1] + self._gae_lambda * values[1:]) - values[:-1]

        # we want to maximize this, hence the minus
        actor_loss = -logprobs[:-1] * deltas.detach()
        # we want V(s_i) to match r_i + gamma * V(s_{i+1})
        critic_loss = 0.5 * deltas ** 2

        surrogate_loss = actor_loss + critic_loss

        return LossOutput(
            loss,
            surrogate_loss,
            AccountState(
                trades_sizes.movedim(-1, 1), trades_prices.movedim(-1, 1), balances
            ),
        )

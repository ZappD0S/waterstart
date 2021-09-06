import torch
import torch.nn as nn

from ...inference import AccountState


class LossOutput:
    def __init__(
        self,
        loss: torch.Tensor,
        surrogate_loss: torch.Tensor,
        account_state: AccountState,
    ) -> None:
        self.loss = loss
        self.surrogate_loss = surrogate_loss
        self.account_state = account_state


class Critic(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, n_traded_sym: int):
        super().__init__()
        self.prev_step_features = 3 * n_traded_sym + 2

        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(self.prev_step_features, hidden_dim)
        self.lin3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.lin4 = nn.Linear(hidden_dim, 1)

    def forward(  # type: ignore
        self, z0: torch.Tensor, trades_data: torch.Tensor
    ) -> torch.Tensor:
        out1 = self.lin1(z0)
        out2 = self.lin2(trades_data)

        out = torch.cat((out1, out2), dim=-1).relu()
        out = self.lin3(out).relu()

        return self.lin4(out).squeeze(-1)

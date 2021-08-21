from math import log2
from statistics import fmean

import torch
import torch.nn as nn


def get_std(log_std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    below_threshold = log_std.exp()
    above_threshold = torch.log1p(log_std + eps) + 1.0
    return torch.where(log_std > 0, above_threshold, below_threshold)


class GatedTransition(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self._gru = nn.GRUCell(input_dim, z_dim)
        self._lin_mean = nn.Linear(z_dim, z_dim)
        self._lin_log_std = nn.Linear(z_dim, z_dim)

    def forward(  # type: ignore
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._gru(x, h)

        mean = self._lin_mean(out)
        log_std = self._lin_log_std(out)
        return mean, get_std(log_std)


class Emitter(nn.Module):
    def __init__(self, z_dim: int, n_traded_sym: int, hidden_dim: int):
        super().__init__()
        self.n_traded_sym = n_traded_sym
        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_exec_logits = nn.Linear(hidden_dim, n_traded_sym)
        self.lin_frac_mean = nn.Linear(hidden_dim, n_traded_sym)
        self.lin_frac_log_std = nn.Linear(hidden_dim, n_traded_sym)

    def forward(  # type: ignore
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (..., z_dim)

        out = self.lin1(z).relu()
        out = self.lin2(out).relu()

        exec_logits = self.lin_exec_logits(out)
        frac_mean = self.lin_frac_mean(out)
        frac_log_std = self.lin_frac_log_std(out)

        return exec_logits, frac_mean, get_std(frac_log_std)


# TODO: should we keep this name?
class CNN(nn.Module):
    def __init__(
        self,
        batch_size: int,
        window_size: int,
        market_features: int,
        out_features: int,
        n_traded_sym: int,
        max_trades: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.window_size = window_size
        self.market_features = market_features
        self.kernel_size = 3
        self.n_traded_sym = n_traded_sym
        self.max_trades = max_trades
        self.prev_step_features = 2 * n_traded_sym * max_trades + 1

        hidden_dim = 2 ** max(
            5, round(log2(fmean((market_features, self.prev_step_features))))
        )

        self.conv1 = nn.Conv1d(
            market_features, market_features, kernel_size=self.kernel_size
        )
        self.conv2 = nn.Conv1d(
            market_features,
            hidden_dim,
            kernel_size=window_size + 1 - self.kernel_size,
        )
        self.lin1 = nn.Linear(self.prev_step_features, hidden_dim)
        self.lin2 = nn.Linear(2 * hidden_dim, out_features)

    def forward(  # type: ignore
        self,
        market_data: torch.Tensor,
        prev_step_data: torch.Tensor,
    ) -> torch.Tensor:
        # market_data: (batch_size, market_features, window_size)
        # prev_step_data: (batch_size, prev_step_features)

        out1: torch.Tensor = self.conv1(market_data).relu()
        out1 = self.conv2(out1).squeeze(2)

        out2: torch.Tensor = self.lin1(prev_step_data)

        out = torch.cat((out1, out2), dim=1).relu()
        out = self.lin2(out).relu()

        return out

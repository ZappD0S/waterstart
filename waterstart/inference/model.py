from math import log2

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

        self.gru = nn.GRUCell(input_dim, z_dim)
        self.lin_mean = nn.Linear(z_dim, z_dim)
        self.lin_log_std = nn.Linear(z_dim, z_dim)

    def forward(  # type: ignore
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.gru(x, h)

        mean = self.lin_mean(out)
        log_std = self.lin_log_std(out)
        return mean, get_std(log_std)


class Emitter(nn.Module):
    def __init__(self, z_dim: int, n_traded_sym: int, hidden_dim: int):
        super().__init__()
        self.n_traded_sym = n_traded_sym
        self.prev_step_features = 3 * n_traded_sym + 2

        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(self.prev_step_features, hidden_dim)
        self.lin3 = nn.Linear(2 * hidden_dim, hidden_dim)

        # self.lin_exec_logit = nn.Linear(hidden_dim, 1)
        # self.lin_sign_logits = nn.Linear(hidden_dim, n_traded_sym)

        # self.lin_frac_log_conc = nn.Linear(hidden_dim, n_traded_sym + 1)
        self.lin_frac_loc = nn.Linear(hidden_dim, n_traded_sym + 1)
        self.lin_frac_log_scale = nn.Linear(hidden_dim, n_traded_sym + 1)

    def forward(  # type: ignore
        self, z0: torch.Tensor, trades_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        out1 = self.lin1(z0)
        out2 = self.lin2(trades_data)

        out = torch.cat((out1, out2), dim=-1).relu()
        out = self.lin3(out).relu()

        # exec_logit = self.lin_exec_logit(out).squeeze(-1)
        # sign_logits: torch.Tensor = self.lin_sign_logits(out)

        # frac_log_conc: torch.Tensor = self.lin_frac_log_conc(out)
        frac_loc: torch.Tensor = self.lin_frac_loc(out)
        frac_log_scale: torch.Tensor = self.lin_frac_log_scale(out)

        return frac_loc, get_std(frac_log_scale)


class CNN(nn.Module):
    def __init__(self, window_size: int, raw_market_data_size: int, out_features: int):
        super().__init__()
        self.window_size = window_size
        self.raw_market_data_size = raw_market_data_size
        self.kernel_size = 3

        hidden_dim = 2 ** max(5, round(log2(raw_market_data_size)))

        self.conv1 = nn.Conv1d(
            raw_market_data_size, raw_market_data_size, kernel_size=self.kernel_size
        )
        self.conv2 = nn.Conv1d(
            raw_market_data_size,
            hidden_dim,
            kernel_size=window_size + 1 - self.kernel_size,
        )
        self.lin = nn.Linear(hidden_dim, out_features)

    def forward(self, market_data: torch.Tensor) -> torch.Tensor:  # type: ignore
        # market_data: (batch_size, market_features, window_size)

        out: torch.Tensor = self.conv1(market_data).relu()
        out = self.conv2(out).squeeze(2).relu()
        out = self.lin(out).relu()

        return out

from typing import Sequence
import numpy as np

import torch
import torch.distributions as dist
import torch.nn as nn


class Model:
    def __init__(
        self,
        cnn: nn.Module,
        gated_trans: nn.Module,
        iafs: Sequence[nn.Module],
        emitter: nn.Module,
    ) -> None:
        self._cnn = cnn
        self._gated_trans = gated_trans
        self._iafs = iafs
        self._emitter = emitter
        # TODO: these need to be taken from the modules
        self.win_len: int = ...
        self.max_trades: int = ...
        self.hidden_dim: int = ...

    @torch.no_grad()
    def __call__(
        self,
        trades_data: np.ndarray,
        market_data: np.ndarray,
        hidden_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        trades_data_tensor = torch.from_numpy(trades_data).astype(torch.float32)  # type: ignore
        market_data_tensor = torch.from_numpy(market_data).astype(torch.float32)  # type: ignore
        hidden_state_tensor = torch.from_numpy(hidden_state).astype(torch.float32)  # type: ignore

        out = self._cnn(market_data_tensor, trades_data_tensor)
        # TODO: reshape..
        # .view(self.n_samples, self.seq_len, self.batch_size, -1)
        z_loc: torch.Tensor
        z_scale: torch.Tensor
        z_loc, z_scale = self._gated_trans(out, hidden_state_tensor)

        z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self._iafs)
        z_sample: torch.Tensor = z_dist.rsample()  # type: ignore

        exec_logits: torch.Tensor
        fractions: torch.Tensor
        exec_logits, fractions = self._emitter(z_sample)
        exec_logits = exec_logits.movedim(3, 0)
        fractions = fractions.movedim(3, 0)

        exec_dist = dist.Bernoulli(logits=exec_logits)
        exec_samples: torch.Tensor = exec_dist.sample()  # type: ignore
        exec_mask = exec_samples == 1

        return exec_mask.numpy(), fractions.numpy(), z_sample.numpy()

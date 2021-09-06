from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.jit as jit

from ...inference import ModelInput
from .utils import LossOutput


class BaseLossEvaluator(ABC, nn.Module):
    def __init__(self, seq_len: int, gamma: float, gae_lambda: float = 0.95) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._gamma = gamma
        self._gae_lambda = gae_lambda

    @jit.export  # type: ignore
    def _compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        last_gae_lam = rewards.new_zeros(())
        gamma = self._gamma
        gae_lambda = self._gae_lambda

        # for step in reversed(range(self._seq_len - 1)):
        for step in range(self._seq_len - 2, -1, -1):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            advantages[step] = last_gae_lam

        return advantages[:-1]

    @abstractmethod
    def evaluate(self, model_input: ModelInput) -> LossOutput:
        ...

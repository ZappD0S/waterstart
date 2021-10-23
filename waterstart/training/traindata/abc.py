from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import torch
import numpy.typing as npt

from ...inference import ModelInput
from .utils import TrainingData, TrainingState


class BaseTrainDataManager(ABC):
    def __init__(
        self,
        training_data: TrainingData,
        training_state: Optional[TrainingState],
        batch_size: int,
        seq_len: int,
        window_size: int,
        device: Optional[torch.device] = None,
    ) -> None:

        self._n_timestemps = training_data.n_timestemps
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._window_size = window_size

        self._next_batch_inds: Optional[npt.NDArray[Any]]

        if training_state is None:
            self._batch_inds_it = self._build_batch_inds_it()
            self._next_batch_inds = next(self._batch_inds_it)
        else:
            self._batch_inds_it = training_state.batch_inds_it
            self._next_batch_inds = training_state.next_batch_inds

        self._device = device

    @abstractmethod
    def _build_batch_inds_it(self) -> Iterator[npt.NDArray[Any]]:
        ...

    def _get_next_batch_inds(self) -> npt.NDArray[Any]:
        if (batch_inds := self._next_batch_inds) is None:
            self._batch_inds_it = self._build_batch_inds_it()
            batch_inds = self._next_batch_inds = next(self._batch_inds_it)

        self._next_batch_inds = next(self._batch_inds_it, None)
        return batch_inds

    @abstractmethod
    def load_data(self) -> ModelInput:
        ...

    @abstractmethod
    def save_state(self) -> TrainingState:
        ...

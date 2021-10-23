from typing import Any, Optional
import torch
import numpy as np
import numpy.typing as npt


class ReadonlyBatchManager:
    def __init__(
        self,
        storage: npt.NDArray[Any],
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if batch_dims > storage.ndim:
            raise ValueError()

        if load_lag < 0:
            raise ValueError()

        self._storage = storage
        self._batch_dims = batch_dims
        self._load_lag = load_lag
        self._batch_dims_last = batch_dims_last
        self._device = device

    @property
    def storage(self) -> npt.NDArray[Any]:
        return self._storage

    def build_batch(self, inds: npt.NDArray[Any]) -> torch.Tensor:
        batch = self._storage[inds - self._load_lag]
        batch = self._transform_batch(batch, self._batch_dims + inds.ndim - 1)
        return torch.from_numpy(batch).to(self._device)

    def _transform_batch(
        self, batch: npt.NDArray[Any], batch_dims: int
    ) -> npt.NDArray[Any]:
        if not self._batch_dims_last:
            return batch

        return batch.transpose(*range(batch_dims, batch.ndim), *range(batch_dims))


class ExpandableBatchManagager(ReadonlyBatchManager):
    def __init__(
        self,
        storage: npt.NDArray[Any],
        expand_size: int,
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            np.expand_dims(storage, batch_dims),
            batch_dims + 1,
            load_lag,
            batch_dims_last,
            device,
        )
        self._expand_size = expand_size

    def _transform_batch(
        self, batch: npt.NDArray[Any], batch_dims: int
    ) -> npt.NDArray[Any]:
        batch = batch.repeat(self._expand_size, axis=batch_dims)
        return super()._transform_batch(batch, batch_dims)


class BatchManager(ReadonlyBatchManager):
    def __init__(
        self,
        storage: npt.NDArray[Any],
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(storage, batch_dims, load_lag, batch_dims_last, device)
        self._inds: Optional[npt.NDArray[Any]] = None

    def build_batch(self, inds: npt.NDArray[Any]) -> torch.Tensor:
        self._inds = inds
        return super().build_batch(inds)

    def store_batch(self, gpu_batch: torch.Tensor) -> npt.NDArray[Any]:
        if (inds := self._inds) is None:
            raise RuntimeError()

        assert not gpu_batch.requires_grad
        batch = gpu_batch.cpu().numpy()
        batch = self._inverse_transform_batch(batch, self._batch_dims + inds.ndim - 1)
        self._inds = None
        self._storage[inds] = batch
        return batch

    def _inverse_transform_batch(
        self, batch: npt.NDArray[Any], batch_dims: int
    ) -> npt.NDArray[Any]:
        if not self._batch_dims_last:
            return batch

        ndim = batch.ndim
        return batch.transpose(
            *range(ndim - batch_dims, ndim), *range(ndim - batch_dims)
        )

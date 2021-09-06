from typing import Optional
import torch


class ReadonlyBatchManager:
    def __init__(
        self,
        storage: torch.Tensor,
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
    def storage(self) -> torch.Tensor:
        return self._storage

    def build_batch(self, inds: torch.Tensor) -> torch.Tensor:
        batch: torch.Tensor = self._storage[inds - self._load_lag]
        batch = self._transform_batch(batch, self._batch_dims + inds.ndim - 1)
        return batch.to(self._device)

    def _transform_batch(self, batch: torch.Tensor, batch_dims: int) -> torch.Tensor:
        if not self._batch_dims_last:
            return batch

        return batch.permute(*range(batch_dims, batch.ndim), *range(batch_dims))


class ExpandableBatchManagager(ReadonlyBatchManager):
    def __init__(
        self,
        storage: torch.Tensor,
        expand_size: int,
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            storage.unsqueeze(batch_dims),
            batch_dims + 1,
            load_lag,
            batch_dims_last,
            device,
        )
        self._expand_size = expand_size

    def _transform_batch(self, batch: torch.Tensor, batch_dims: int) -> torch.Tensor:
        batch = batch.expand(
            *(-1,) * (batch_dims - 1),
            self._expand_size,
            *(-1,) * (batch.ndim - batch_dims),
        )

        return super()._transform_batch(batch, batch_dims)


class BatchManager(ReadonlyBatchManager):
    def __init__(
        self,
        storage: torch.Tensor,
        batch_dims: int,
        load_lag: int,
        batch_dims_last: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(storage, batch_dims, load_lag, batch_dims_last, device)
        self._inds: Optional[torch.Tensor] = None

    def build_batch(self, inds: torch.Tensor) -> torch.Tensor:
        self._inds = inds
        return super().build_batch(inds)

    def store_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if (inds := self._inds) is None:
            raise RuntimeError()

        assert not batch.requires_grad
        batch = batch.cpu()
        batch = self._inverse_transform_batch(batch, self._batch_dims + inds.ndim - 1)
        self._inds = None
        self._storage[inds] = batch
        return batch

    def _inverse_transform_batch(
        self, batch: torch.Tensor, batch_dims: int
    ) -> torch.Tensor:
        if not self._batch_dims_last:
            return batch

        ndim = batch.ndim
        return batch.permute(*range(ndim - batch_dims, ndim), *range(ndim - batch_dims))

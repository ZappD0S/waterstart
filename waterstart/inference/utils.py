from collections import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeVar, Any
from waterstart.array_mapping.base_mapper import BaseArrayMapper

import numpy as np
import numpy.typing as npt
import torch
from waterstart.array_mapping.dict_based_mapper import DictBasedArrayMapper


@dataclass
class MaskedTensor:
    arr: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if not self.arr.ndim == self.mask.ndim == 1:
            raise ValueError()

        if self.arr.numel() != self.mask.numel():
            raise ValueError()

        if self.mask.dtype != torch.bool:
            raise ValueError()


T = TypeVar("T")


def obj_to_tensor(mapper: BaseArrayMapper[T], obj: T) -> torch.Tensor:
    rec_arr: npt.NDArray[Any] = np.fromiter(  # type: ignore
        mapper.iterate_index_to_value(obj), dtype=[("inds", int), ("vals", float)]
    )

    arr: torch.Tensor = torch.full(rec_arr.shape, np.nan, dtype=float)  # type: ignore
    inds = torch.from_numpy(rec_arr["inds"])  # type: ignore
    vals = torch.from_numpy(rec_arr["vals"])  # type: ignore
    arr[inds] = vals

    if arr.isnan().any():
        raise ValueError()

    return arr


def tensor_to_obj(mapper: BaseArrayMapper[T], arr: torch.Tensor) -> T:
    l: list[float] = arr.tolist()  # type: ignore
    return mapper.build_from_index_to_value_map(dict(enumerate(l)))


def map_to_tensors(
    mapper: DictBasedArrayMapper[int], mapping: Mapping[int, tuple[float, ...]]
) -> Iterator[torch.Tensor]:
    for masked_tens in partial_map_to_masked_tensors(mapper, mapping):
        if not masked_tens.mask.all():
            raise ValueError()

        yield masked_tens.arr


def partial_map_to_masked_tensors(
    mapper: DictBasedArrayMapper[int], mapping: Mapping[int, tuple[float, ...]]
) -> Iterator[MaskedTensor]:
    tuple_len = len(next(iter(mapping.values())))

    rec_arr: npt.NDArray[Any] = np.fromiter(  # type: ignore
        mapper.iterate_index_to_value_partial(mapping),
        dtype=[("inds", int), ("vals", [("", float) for _ in range(tuple_len)])],
    )

    inds = rec_arr["inds"]
    vals_arr = rec_arr["vals"]

    for name in vals_arr.dtype.names:
        vals = vals_arr[name]
        arr = np.empty_like(vals)  # type: ignore
        mask = np.empty_like(vals, dtype=bool)  # type: ignore
        mask[inds] = True
        arr[inds] = vals

        yield MaskedTensor(
            torch.from_numpy(arr),  # type: ignore
            torch.from_numpy(mask),  # type: ignore
        )


def masked_tensor_to_partial_map(
    mapper: DictBasedArrayMapper[int], masked_arr: MaskedTensor
) -> Mapping[int, float]:
    arr, mask = masked_arr.arr, masked_arr.mask
    inds_list: list[int] = torch.arange(arr.numel())[mask].tolist()  # type: ignore
    new_pos_sizes_list: list[float] = arr[mask].tolist()  # type: ignore

    return mapper.build_from_index_to_value_map_partial(
        dict(zip(inds_list, new_pos_sizes_list))
    )

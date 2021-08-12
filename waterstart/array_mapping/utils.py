from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions as rfn


from .base_mapper import BaseArrayMapper
from .dict_based_mapper import DictBasedArrayMapper


T = TypeVar("T")
T_float = TypeVar("T_float", bound=np.floating)


@dataclass
class MaskedArrays(Generic[T_float]):
    data: npt.NDArray[T_float]
    mask: npt.NDArray[np.bool_]

    def __post_init__(self):
        data = self.data
        mask = self.mask

        if mask.ndim != 1 or data.ndim != 2:
            raise ValueError()

        if mask.size != data.shape[0]:
            raise ValueError()

    def __iter__(self) -> Iterator[npt.NDArray[T_float]]:
        return iter(self.data.T)

    def __len__(self) -> int:
        return self.data.shape[1]


def obj_to_array(
    mapper: BaseArrayMapper[T],
    obj: T,
    dtype: type[T_float] = np.float64,  # type: ignore
) -> npt.NDArray[T_float]:
    rec_arr: npt.NDArray[Any] = np.fromiter(  # type: ignore
        mapper.iterate_index_to_value(obj), dtype=[("inds", int), ("vals", dtype)]
    )

    arr: npt.NDArray[T_float] = np.full_like(  # type: ignore
        rec_arr, np.nan, dtype=dtype
    )
    inds = rec_arr["inds"]  # type: ignore
    vals = rec_arr["vals"]  # type: ignore
    arr[inds] = vals

    if np.isnan(arr).any():
        raise ValueError()

    return arr


def array_to_obj(mapper: BaseArrayMapper[T], arr: npt.NDArray[T_float]) -> T:
    l: list[float] = arr.tolist()  # type: ignore
    return mapper.build_from_index_to_value_map(dict(enumerate(l)))


def map_to_arrays(
    mapper: DictBasedArrayMapper[int],
    mapping: Mapping[int, tuple[float, ...]],
    dtype: type[T_float]
) -> Iterator[npt.NDArray[np.float64]]:
    masked_arr = partial_map_to_masked_arrays(mapper, mapping, dtype)

    if not masked_arr.mask.all():
        raise ValueError()

    return iter(masked_arr)


def partial_map_to_masked_arrays(
    mapper: DictBasedArrayMapper[int],
    mapping: Mapping[int, tuple[float, ...]],
    dtype: type[T_float]
) -> MaskedArrays[T_float]:
    tuple_len = len(next(iter(mapping.values())))
    keys_len = len(mapper.keys)

    rec_arr: npt.NDArray[Any] = np.fromiter(  # type: ignore
        mapper.iterate_index_to_value_partial(mapping),
        dtype=[("inds", int), ("vals", [("", dtype) for _ in range(tuple_len)])],
    )

    inds: npt.NDArray[np.int64] = rec_arr["inds"]
    vals: npt.NDArray[T_float]
    vals = rfn.structured_to_unstructured(rec_arr["vals"])  # type: ignore

    mask: npt.NDArray[np.bool_] = np.zeros(keys_len, dtype=bool)  # type: ignore
    mask[inds] = True

    data: npt.NDArray[T_float]
    data = np.zeros((keys_len, tuple_len), dtype=dtype)  # type: ignore

    data[inds] = vals
    return MaskedArrays(data, mask)


def masked_array_to_partial_map(
    mapper: DictBasedArrayMapper[int], masked_arrs: MaskedArrays[T_float]
) -> Mapping[int, float]:
    res: dict[int, float] = {}

    if len(masked_arrs) != 1:
        raise ValueError()

    for idx, vals in masked_arrays_to_partial_map(mapper, masked_arrs).items():
        [val] = vals
        res[idx] = val

    return res


def masked_arrays_to_partial_map(
    mapper: DictBasedArrayMapper[int], masked_arrs: MaskedArrays[T_float]
) -> Mapping[int, Sequence[float]]:
    mask = masked_arrs.mask
    inds_list: list[int] = np.arange(len(mask))[mask].tolist()  # type: ignore
    vals_list: list[list[float]] = masked_arrs.data[mask].tolist()  # type: ignore

    return mapper.build_from_index_to_value_map_partial(dict(zip(inds_list, vals_list)))

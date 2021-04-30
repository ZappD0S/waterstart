from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, TypeVar

import numpy as np

from waterstart.symbols import SymbolInfo

from .features_builder import FeautureVectorMapper, KeyIndexMapper

T = TypeVar("T", bound=IntEnum)


@dataclass
class ExecutionData:
    exec_sample: float
    position_size: float
    # TODO: else?

    def __post_init__(self):
        if self.exec_sample not in (0.0, 1.0):
            raise ValueError()


# TODO: We need a class that takes an instance of the one below that actually
# executes the orders. Eventually this class will implement Observer and will
# produce a Summary of the current state of the account and the trades that were
# opened or closed


class Executor(Generic[T]):
    def __init__(
        self,
        # TODO: make this SymbolInfoWithConvChains?
        exec_data_mapper: KeyIndexMapper[SymbolInfo],
        feat_vec_mapper: FeautureVectorMapper[T],
        win_len: int,
        max_trades: int,
        # TODO: take network as parameter
    ) -> None:
        self._exec_data_mapper = exec_data_mapper
        self._feat_vec_mapper = feat_vec_mapper

        self.n_sym = len(exec_data_mapper.keys)
        self.n_feat = len(feat_vec_mapper.keys)

        # TODO: add batch dim before passing to the model
        self._market_data: np.ndarray = np.zeros(
            (self.n_feat, self.n_sym, win_len), dtype=np.float32
        )
        self._pos_data: np.ndarray = np.zeros(
            (self.n_sym, max_trades), dtype=np.float32
        )

    def get_inds(
        self, prices_map: Mapping[SymbolInfo, Mapping[T, float]]
    ) -> Iterator[tuple[int, int, float]]:
        for sym_ind, data_map in self._exec_data_mapper.map_keys_to_inds(prices_map):
            for feat_ind, value in self._feat_vec_mapper.map_keys_to_inds(data_map):
                yield sym_ind, feat_ind, value

    async def execute(
        self, prices_map: Mapping[SymbolInfo, Mapping[T, float]]
    ) -> Mapping[SymbolInfo, ExecutionData]:
        rec_arr = np.fromiter(
            self.get_inds(prices_map),
            [
                ("sym_ind", np.int64),
                ("feat_ind", np.int64),
                ("value", np.float32),
            ],
        )

        sym_inds = rec_arr["sym_ind"]
        feat_inds = rec_arr["feat_ind"]
        vals = rec_arr["value"]

        self._market_data = np.roll(self._market_data, shift=-1, axis=-1)
        latest_market_data = self._market_data[..., -1]
        latest_market_data[...] = np.nan
        latest_market_data[feat_inds, sym_inds] = vals

        if np.isnan(latest_market_data).any():
            raise ValueError()

        # TODO: we need to normalize the market_data with self._feat_vec_mapper.price_index_groups
        # before passing it to the network. Also the pos_data needs to be normalized

        # function that takes in the market_data and the pos_data and returns
        # an exec_mask and a new_pos_sizes

        # TODO: the new_pos_sizes that are below the minimum pos size should be turned
        # into position close orders

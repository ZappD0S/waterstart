import asyncio
from dataclasses import fields
from typing import Sequence

import numpy as np
import psycopg2
from numpy.lib import recfunctions as rfn

from ...array_mapping.base_mapper import FieldData
from ...array_mapping.market_data_mapper import PriceFieldData
from ...client.trader import TraderClient
from ...openapi import ProtoOAAssetListReq, ProtoOAAssetListRes
from ...price import MarketData, SymbolData, TrendBar
from ...symbols import SymbolsList, TradedSymbolInfo


async def get_asset_id(client: TraderClient, asset_name: str):
    asset_list_res = await client.send_request_from_trader(
        lambda trader_id: ProtoOAAssetListReq(ctidTraderAccountId=trader_id),
        ProtoOAAssetListRes,
    )
    asset_name_to_id = {asset.name: asset.assetId for asset in asset_list_res.asset}
    return asset_name_to_id[asset_name]


async def get_traded_symbols(sym_ids: Sequence[int]) -> Sequence[TradedSymbolInfo]:
    client = await TraderClient.create(
        "demo.ctraderapi.com",
        5035,
        "2396_zKg1chyHLMkfP4ahuqh5924VjbWaz4m0YPW3jlIrFc1j8cf7TB",
        "B9ExeJTkUHnNbJb13Pi1POmUwgKG0YpOiVzswE0QI1g5rXhNwC",
        20783271,
        "3_Uld7UwUnTPC7E_3MkWG2NeZaUHBrzo5jShxINrHiQ",
        "LVZl2nqEQNBMXhU7IjFicMTIYvUdmKc6h44sI8Rl",
    )

    sym_list = SymbolsList(client, await get_asset_id(client, "EUR"))
    traded_symbols = [
        sym_info
        async for sym_info in sym_list.get_traded_sym_infos_from_ids(set(sym_ids))
    ]
    await client.aclose()
    return traded_symbols


def build_trendbar(offset: int) -> tuple[TrendBar[FieldData], int]:
    tb = TrendBar(FieldData(offset + 0), FieldData(offset + 1), FieldData(offset + 2))

    return tb, offset + 3


def build_price_trendbar(
    price_group_id: int, offset: int
) -> tuple[TrendBar[PriceFieldData], int]:
    tb = TrendBar(
        PriceFieldData(offset + 0, price_group_id, is_close_price=False),
        PriceFieldData(offset + 1, price_group_id, is_close_price=False),
        PriceFieldData(offset + 2, price_group_id, is_close_price=True),
    )

    return tb, offset + 3


def build_symbol_data(sym_id: int, offset: int) -> tuple[SymbolData[FieldData], int]:
    sym_data_map: dict[str, TrendBar[FieldData]] = {}

    for i, field in enumerate(fields(SymbolData)):
        if field.name == "spread_trendbar":
            tb_data, offset = build_trendbar(offset)
        else:
            tb_data, offset = build_price_trendbar(hash((sym_id, i)), offset)

        sym_data_map[field.name] = tb_data

    return SymbolData(**sym_data_map), offset


def build_market_data(sym_ids: Sequence[int]) -> MarketData[FieldData]:
    sym_data_map: dict[int, SymbolData[FieldData]] = {}
    offset = 0
    for sym_id in sym_ids:
        sym_data, offset = build_symbol_data(sym_id, offset)
        sym_data_map[sym_id] = sym_data

    return MarketData(sym_data_map, FieldData(offset + 0), FieldData(offset + 1))


conn = psycopg2.connect(database="waterstart", user="postgres", host="localhost")
cur = conn.cursor()

columns = {
    "timestamp": "datetime64[s]",
    "sym_id": int,
    "price_high": float,
    "price_low": float,
    "price_close": float,
    "spread_high": float,
    "spread_low": float,
    "spread_close": float,
    "base_conv_high": float,
    "base_conv_low": float,
    "base_conv_close": float,
    "quote_conv_high": float,
    "quote_conv_low": float,
    "quote_conv_close": float,
    "time_of_day": float,
    "delta_to_last": float,
}

query = "SELECT " + ",".join(columns) + " FROM market_data;"

cur.execute(query)

data = np.fromiter(cur, dtype=list(columns.items()))
del cur

timestamps, sym_ids = data["timestamp"], data["sym_id"]
data = rfn.drop_fields(data, ("timestamp", "sym_id"))

unique_timestamps, timestamps_idxs = np.unique(timestamps, return_inverse=True)
unique_sym_ids, sym_idxs = np.unique(sym_ids, return_inverse=True)
n_timesteps = len(unique_timestamps)
n_traded_sym = len(unique_sym_ids)

col_to_idx = {col: i for i, col in enumerate(data.dtype.names)}

arr = np.full((n_timesteps, n_traded_sym, len(data.dtype)), np.nan, dtype=np.float32)
timestamps_arr = np.full((n_timesteps, n_traded_sym), np.nan, dtype=np.float32)

arr[timestamps_idxs, sym_idxs] = rfn.structured_to_unstructured(data)
timestamps_arr[timestamps_idxs, sym_idxs] = timestamps
del data
assert not np.isnan(arr).any()
assert not np.isnan(timestamps_arr).any()
assert np.all(timestamps_arr[:, 0, None] == timestamps_arr)
timestamps_arr = timestamps_arr[:, 0]

sym_prices = arr[..., col_to_idx["price_close"]]
spreads = arr[..., col_to_idx["spread_close"]]
margin_rates = arr[..., col_to_idx["base_conv_close"]]
quote_to_dep_rates = arr[..., col_to_idx["quote_conv_close"]]

part2_cols = {"time_of_day", "delta_to_last"}

part1_idx = [idx for col, idx in col_to_idx.items() if col not in part2_cols]
part2_idx = [idx for col, idx in col_to_idx.items() if col in part2_cols]

assert part1_idx == sorted(part1_idx)
assert part2_idx == sorted(part2_idx)

market_data_part1 = arr[..., part1_idx]
market_data_part2 = arr[..., part2_idx]

assert np.all(market_data_part2 == market_data_part2[:, 0, None])
market_data_part2 = market_data_part2[:, 0]
market_data_arr = np.concatenate(
    (market_data_part1.reshape(n_timesteps, -1), market_data_part2), axis=1
)

market_data_blueprint = build_market_data(unique_sym_ids)
traded_sym_blueprint_map = {
    sym_id: FieldData(i) for i, sym_id in enumerate(unique_sym_ids)
}
traded_symbols = asyncio.run(get_traded_symbols(unique_sym_ids))

np.savez_compressed(
    "train_data.npz",
    timestamps=timestamps_arr,
    sym_prices=sym_prices,
    spreads=spreads,
    margin_rates=margin_rates,
    quote_to_dep_rates=quote_to_dep_rates,
    market_data_arr=market_data_arr,
    market_data_blueprint=market_data_blueprint,
    traded_sym_blueprint_map=traded_sym_blueprint_map,
    traded_symbols=traded_symbols,
)

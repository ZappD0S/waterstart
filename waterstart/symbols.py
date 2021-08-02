from __future__ import annotations

import datetime
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence, Set
from dataclasses import InitVar, dataclass
from typing import Counter, Optional, TypeVar
from zoneinfo import ZoneInfo

from .client.trader import TraderClient
from .openapi import (
    ProtoOALightSymbol,
    ProtoOASymbol,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsForConversionReq,
    ProtoOASymbolsForConversionRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)
from .openapi.OpenApiModelMessages_pb2 import ProtoOAHoliday
from .utils import is_sorted


@dataclass
class Holiday:
    holiday: InitVar[ProtoOAHoliday]

    def __post_init__(self, holiday: ProtoOAHoliday) -> None:
        self._tz = ZoneInfo(holiday.scheduleTimeZone)
        epoch = datetime.datetime.fromtimestamp(0, self._tz)

        start = self._start = epoch + datetime.timedelta(
            days=holiday.holidayDate, seconds=holiday.startSecond
        )
        end = self._end = epoch + datetime.timedelta(
            days=holiday.holidayDate, seconds=holiday.endSecond
        )

        if start > end:
            raise ValueError()

        if start.date() != end.date():
            raise ValueError()

        self._is_recurring = holiday.isRecurring

    @property
    def tz(self) -> datetime.tzinfo:
        return self._tz

    @property
    def start(self) -> datetime.datetime:
        return self._start

    @property
    def end(self) -> datetime.datetime:
        return self._end

    @property
    def is_recurring(self) -> bool:
        return self._is_recurring


@dataclass
class Schedule:
    symbol: InitVar[ProtoOASymbol]

    def __post_init__(self, symbol: ProtoOASymbol) -> None:
        if not symbol.schedule:
            raise ValueError()

        timetable: list[datetime.timedelta] = []
        for interval in symbol.schedule:
            start = datetime.timedelta(seconds=interval.startSecond)
            end = datetime.timedelta(seconds=interval.endSecond)
            timetable += (start, end)

        if len(timetable) % 2 != 0:
            raise ValueError()

        if not is_sorted(timetable):
            raise ValueError()

        self._tz = ZoneInfo(symbol.scheduleTimeZone)
        self._timetable = timetable

    @property
    def tz(self) -> datetime.tzinfo:
        return self._tz

    @property
    def timetable(self) -> Sequence[datetime.timedelta]:
        return self._timetable


@dataclass
class SymbolInfo:
    light_symbol: InitVar[ProtoOALightSymbol]
    symbol: InitVar[ProtoOASymbol]

    def __post_init__(
        self, light_symbol: ProtoOALightSymbol, symbol: ProtoOASymbol
    ) -> None:
        self._id = symbol.symbolId
        self._name = light_symbol.symbolName.lower()
        self._base_asset_id = light_symbol.baseAssetId
        self._quote_asset_id = light_symbol.quoteAssetId
        self._holidays = [Holiday(holiday) for holiday in symbol.holiday]
        self._schedule = Schedule(symbol)

        self._min_volume = symbol.minVolume
        self._step_volume = symbol.stepVolume
        self._max_volume = symbol.maxVolume
        self._lot_size = symbol.lotSize

    # TODO: use NewType to define a SymbolId type instead of using bare int?
    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_asset_id(self) -> int:
        return self._base_asset_id

    @property
    def quote_asset_id(self) -> int:
        return self._quote_asset_id

    @property
    def holidays(self) -> Sequence[Holiday]:
        return self._holidays

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    @property
    def min_volume(self) -> int:
        return self._min_volume

    @property
    def step_volume(self) -> int:
        return self._step_volume

    @property
    def max_volume(self) -> int:
        return self._max_volume

    @property
    def lot_size(self) -> int:
        return self._lot_size


@dataclass
class ChainSymbolInfo(SymbolInfo):
    reciprocal: bool


@dataclass(frozen=True)
class ConvChain:
    asset_id: int
    syminfos: Sequence[ChainSymbolInfo]

    def __len__(self) -> int:
        return len(self.syminfos)

    def __iter__(self) -> Iterator[ChainSymbolInfo]:
        return iter(self.syminfos)

    def __getitem__(self, y: int) -> ChainSymbolInfo:
        return self.syminfos[y]


@dataclass(frozen=True)
class DepositConvChains:
    base_asset: ConvChain
    quote_asset: ConvChain


@dataclass
class TradedSymbolInfo(SymbolInfo):
    conv_chains: DepositConvChains


T = TypeVar("T")
U = TypeVar("U")
T_SymInfo = TypeVar("T_SymInfo", bound=SymbolInfo)


# TODO: have a loop that subscribes to ProtoOASymbolChangedEvent and updates the
# changed symbols
class SymbolsList:
    def __init__(self, client: TraderClient, dep_asset_id: int) -> None:
        self._client = client
        self._dep_asset_id = dep_asset_id
        self._id_to_sym_info_map: dict[int, SymbolInfo] = {}
        self._id_to_full_symbol_map: dict[int, ProtoOASymbol] = {}

        self._sym_name_to_id: Optional[dict[str, int]] = None
        self._id_to_lightsym: Optional[dict[int, ProtoOALightSymbol]] = None
        self._asset_pair_to_symbol_map: Optional[
            Mapping[tuple[int, int], ProtoOALightSymbol]
        ] = None

    async def get_sym_infos_from_names(
        self, subset: Optional[Set[str]] = None
    ) -> AsyncIterator[SymbolInfo]:
        ids_subset: Optional[Set[int]] = None

        if subset is not None:
            sym_name_to_id = await self._get_sym_name_to_id_map()
            ids_subset = {sym_name_to_id[name] for name in subset}

        async for sym_info in self.get_sym_infos_from_ids(ids_subset):
            yield sym_info

    async def get_sym_infos_from_ids(
        self, subset: Optional[Set[int]] = None
    ) -> AsyncIterator[SymbolInfo]:

        if subset is None:
            id_to_lightsym = await self._get_id_to_lightsym_map()
            subset = id_to_lightsym.keys()

        found_sym_infos, missing_sym_ids = self._get_saved_and_missing(
            self._id_to_sym_info_map, subset
        )

        for sym_info in found_sym_infos.values():
            yield sym_info

        async for sym_info in self._build_sym_info(missing_sym_ids):
            self._id_to_sym_info_map[sym_info.id] = sym_info
            yield sym_info

    async def get_traded_sym_infos_from_names(
        self, subset: Optional[Set[str]] = None
    ) -> AsyncIterator[SymbolInfo]:
        ids_subset: Optional[Set[int]] = None

        if subset is not None:
            sym_name_to_id = await self._get_sym_name_to_id_map()
            ids_subset = {sym_name_to_id[name] for name in subset}

        async for sym_info in self.get_traded_sym_infos_from_ids(ids_subset):
            yield sym_info

    async def get_traded_sym_infos_from_ids(
        self, subset: Optional[Set[int]] = None
    ) -> AsyncIterator[TradedSymbolInfo]:

        if subset is None:
            id_to_lightsym = await self._get_id_to_lightsym_map()
            subset = id_to_lightsym.keys()

        found_sym_infos, missing_sym_ids = self._get_saved_and_missing(
            {
                name: sym_info
                for name, sym_info in self._id_to_sym_info_map.items()
                if isinstance(sym_info, TradedSymbolInfo)
            },
            subset,
        )

        for sym_info in found_sym_infos.values():
            yield sym_info

        async for sym_info in self._build_traded_sym_info(missing_sym_ids):
            self._id_to_sym_info_map[sym_info.id] = sym_info
            yield sym_info

    @staticmethod
    def _get_saved_and_missing(
        saved_map: Mapping[T, U],
        keys: Set[T],
    ) -> tuple[Mapping[T, U], Set[T]]:
        saved_keys = saved_map.keys()
        found_keys = keys & saved_keys

        found_map = {key: saved_map[key] for key in found_keys}
        missing_keys = keys - found_keys

        return found_map, missing_keys

    async def _build_sym_info(self, sym_ids: Set[int]) -> AsyncIterator[SymbolInfo]:
        id_to_lightsym = await self._get_id_to_lightsym_map()

        async for sym_id, sym in self._get_full_symbols(sym_ids):
            yield SymbolInfo(id_to_lightsym[sym_id], sym)

    async def _build_traded_sym_info(
        self, sym_ids: Set[int]
    ) -> AsyncIterator[TradedSymbolInfo]:
        id_to_lightsym = await self._get_id_to_lightsym_map()
        id_to_conv_chain = {
            sym_id: conv_chain
            async for sym_id, conv_chain in self._build_conv_chains(sym_ids)
        }

        async for sym_id, sym in self._get_full_symbols(sym_ids):
            yield TradedSymbolInfo(
                id_to_lightsym[sym_id], sym, id_to_conv_chain[sym_id]
            )

    async def _get_full_symbols(
        self, sym_ids: Set[int]
    ) -> AsyncIterator[tuple[int, ProtoOASymbol]]:
        found_syms, missing_sym_ids = self._get_saved_and_missing(
            self._id_to_full_symbol_map, sym_ids
        )

        for sym_id, sym in found_syms.items():
            yield sym_id, sym

        sym_list_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOASymbolByIdReq(
                ctidTraderAccountId=trader_id,
                symbolId=missing_sym_ids,
            ),
            ProtoOASymbolByIdRes,
        )

        for sym in sym_list_res.symbol:
            sym_id = sym.symbolId
            self._id_to_full_symbol_map[sym_id] = sym
            yield sym_id, sym

    def _build_chainsyminfos(
        self,
        lightsym_chain: Sequence[ProtoOALightSymbol],
        id_to_sym: Mapping[int, ProtoOASymbol],
    ) -> Iterator[ChainSymbolInfo]:
        if not lightsym_chain:
            return

        first = lightsym_chain[0]
        reciprocal = self._dep_asset_id != first.baseAssetId
        yield ChainSymbolInfo(first, id_to_sym[first.symbolId], reciprocal)

        for first, second in zip(lightsym_chain[:-1], lightsym_chain[1:]):
            reciprocal = first.quoteAssetId != second.baseAssetId
            yield ChainSymbolInfo(second, id_to_sym[second.symbolId], reciprocal)

    async def _build_conv_chains(
        self, sym_ids: Set[int]
    ) -> AsyncIterator[tuple[int, DepositConvChains]]:
        def get_key(res: ProtoOASymbolsForConversionRes) -> int:
            counter = Counter(
                asset_id
                for sym in res.symbol
                for asset_id in (sym.baseAssetId, sym.quoteAssetId)
            )

            asset_id_set = {
                asset_id for asset_id, count in counter.items() if count != 2
            }
            asset_id_set.remove(dep_asset_id)

            [asset_id] = asset_id_set
            return asset_id

        # def get_key(res: ProtoOASymbolsForConversionRes) -> int:
        #     symbols = res.symbol
        #     first, second = symbols[0], symbols[1]
        #     first_assets = (first.baseAssetId, first.quoteAssetId)
        #     second_assets = {second.baseAssetId, second.quoteAssetId}

        #     for i, asset_id in enumerate(first_assets):
        #         if asset_id in second_assets:
        #             other = first_assets[1 - i]
        #             assert other not in second_assets
        #             return other

        #     raise ValueError()

        id_to_lightsym = await self._get_id_to_lightsym_map()
        lightsyms = [id_to_lightsym[sym_id] for sym_id in sym_ids]

        assets_set = {
            asset_id
            for sym in lightsyms
            for asset_id in (sym.baseAssetId, sym.quoteAssetId)
        }

        asset_id_to_convchain: dict[int, Sequence[ProtoOALightSymbol]] = {}
        dep_asset_id = self._dep_asset_id

        if dep_asset_id in assets_set:
            asset_id_to_convchain[dep_asset_id] = []
            assets_set.remove(dep_asset_id)

        asset_pair_to_symbol_map = await self._get_asset_pair_to_symbol_map()
        convchain_to_download_asset_ids: set[int] = set()

        for asset_id in assets_set:
            asset_pair = (asset_id, dep_asset_id)
            if asset_pair in asset_pair_to_symbol_map:
                asset_id_to_convchain[asset_id] = [asset_pair_to_symbol_map[asset_pair]]
            else:
                convchain_to_download_asset_ids.add(asset_id)

        def get_asset_id_to_convlist_req(trader_id: int):
            return {
                asset_id: ProtoOASymbolsForConversionReq(
                    ctidTraderAccountId=trader_id,
                    firstAssetId=dep_asset_id,
                    lastAssetId=asset_id,
                )
                for asset_id in convchain_to_download_asset_ids
            }

        asset_id_to_convchain |= {
            asset_id: res.symbol
            async for asset_id, res in self._client.send_requests_from_trader(
                get_asset_id_to_convlist_req,
                ProtoOASymbolsForConversionRes,
                get_key,
            )
        }

        conv_chains_sym_ids = {
            sym.symbolId for chain in asset_id_to_convchain.values() for sym in chain
        }

        id_to_sym = {
            sym_id: sym
            async for sym_id, sym in self._get_full_symbols(conv_chains_sym_ids)
        }

        asset_id_to_syminfo_convchain = {
            asset_id: list(self._build_chainsyminfos(convchain, id_to_sym))
            for asset_id, convchain in asset_id_to_convchain.items()
        }

        for sym in lightsyms:
            yield sym.symbolId, DepositConvChains(
                base_asset=ConvChain(
                    sym.baseAssetId, asset_id_to_syminfo_convchain[sym.baseAssetId]
                ),
                quote_asset=ConvChain(
                    sym.quoteAssetId, asset_id_to_syminfo_convchain[sym.quoteAssetId]
                ),
            )

    async def _get_sym_name_to_id_map(self) -> Mapping[str, int]:
        if (sym_name_to_id := self._sym_name_to_id) is not None:
            return sym_name_to_id

        id_to_lightsym = await self._get_id_to_lightsym_map()

        sym_name_to_id = self._sym_name_to_id = {
            sym.symbolName.lower(): sym_id for sym_id, sym in id_to_lightsym.items()
        }

        return sym_name_to_id

    async def _get_id_to_lightsym_map(self) -> Mapping[int, ProtoOALightSymbol]:
        if (id_to_lightsym := self._id_to_lightsym) is not None:
            return id_to_lightsym

        lightsym_list_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOASymbolsListReq(ctidTraderAccountId=trader_id),
            ProtoOASymbolsListRes,
        )

        id_to_lightsym = self._light_symbol_map = {
            sym.symbolId: sym for sym in lightsym_list_res.symbol
        }

        return id_to_lightsym

    async def _get_asset_pair_to_symbol_map(
        self,
    ) -> Mapping[tuple[int, int], ProtoOALightSymbol]:
        if (asset_pair_to_symbol_map := self._asset_pair_to_symbol_map) is not None:
            return asset_pair_to_symbol_map

        light_symbol_map = await self._get_id_to_lightsym_map()

        asset_pair_to_symbol_map = self._asset_pair_to_symbol_map = {
            asset_pair: sym
            for sym in light_symbol_map.values()
            for asset_pair in (
                (sym.baseAssetId, sym.quoteAssetId),
                (sym.quoteAssetId, sym.baseAssetId),
            )
        }

        return asset_pair_to_symbol_map

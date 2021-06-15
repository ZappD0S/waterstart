from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Collection, Counter, Iterator, Optional, TypeVar, Union

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


@dataclass(frozen=True)
class SymbolInfo:
    light_symbol: ProtoOALightSymbol = field(hash=False)
    symbol: ProtoOASymbol = field(hash=False)
    id: int = field(init=False, hash=True)

    def __post_init__(self):
        super().__setattr__("id", self.symbol.symbolId)

    @property
    def name(self):
        return self.light_symbol.symbolName.lower()


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TradedSymbolInfo(SymbolInfo):
    conv_chains: DepositConvChains = field(hash=False)


T = TypeVar("T")
U = TypeVar("U")
T_SymInfo = TypeVar("T_SymInfo", bound=SymbolInfo)

# TODO: have a loop that subscribes to ProtoOASymbolChangedEvent and updates the
# changed symbols
class SymbolsList:
    def __init__(self, client: TraderClient, dep_asset_id: int) -> None:
        self._client = client
        # self._trader = trader
        self._dep_asset_id = dep_asset_id
        self._name_to_sym_info_map: dict[str, SymbolInfo] = {}
        self._id_to_full_symbol_map: dict[int, ProtoOASymbol] = {}

        self._light_symbol_map: Optional[
            dict[Union[int, str], ProtoOALightSymbol]
        ] = None
        self._asset_pair_to_symbol_map: Optional[
            Mapping[tuple[int, int], ProtoOALightSymbol]
        ] = None

    async def get_sym_infos(
        self, subset: Optional[Set[str]] = None
    ) -> AsyncIterator[SymbolInfo]:
        found_sym_infos, missing_syms = await self._get_saved_sym_infos(
            self._name_to_sym_info_map, subset
        )

        for sym_info in found_sym_infos:
            yield sym_info

        async for sym_info in self._build_sym_info(missing_syms):
            self._name_to_sym_info_map[sym_info.name] = sym_info
            yield sym_info

    async def get_traded_sym_infos(
        self, subset: Optional[Set[str]] = None
    ) -> AsyncIterator[TradedSymbolInfo]:
        found_sym_infos, missing_syms = await self._get_saved_sym_infos(
            {
                name: sym_info
                for name, sym_info in self._name_to_sym_info_map.items()
                if isinstance(sym_info, TradedSymbolInfo)
            },
            subset,
        )

        for sym_info in found_sym_infos:
            yield sym_info

        async for sym_info in self._build_traded_sym_info(missing_syms):
            self._name_to_sym_info_map[sym_info.name] = sym_info
            yield sym_info

    async def _get_saved_sym_infos(
        self,
        saved_sym_info_map: Mapping[str, T_SymInfo],
        subset: Optional[Set[str]],
    ) -> tuple[Collection[T_SymInfo], Set[ProtoOALightSymbol]]:
        light_symbol_map = await self._get_light_symbol_map()

        if subset is None:
            subset = {name for name in light_symbol_map if isinstance(name, str)}

        found_sym_infos, missing_names = self._get_saved_and_missing(
            saved_sym_info_map, subset
        )

        missing_symbols = {light_symbol_map[name] for name in missing_names}
        return found_sym_infos.values(), missing_symbols

    @staticmethod
    def _get_saved_and_missing(
        saved_map: Mapping[T, U],
        keys: Set[T],
    ) -> tuple[Mapping[T, U], Set[T]]:
        missing_keys = keys - saved_map.keys()
        saved_map = {key: saved_map[key] for key in keys}
        return saved_map, missing_keys

    async def _build_sym_info(
        self, light_syms: Set[ProtoOALightSymbol]
    ) -> AsyncIterator[SymbolInfo]:
        async for light_sym, sym in self._get_full_symbols(light_syms):
            yield SymbolInfo(light_sym, sym)

    async def _build_traded_sym_info(
        self, light_syms: Set[ProtoOALightSymbol]
    ) -> AsyncIterator[TradedSymbolInfo]:
        conv_chains = {
            sym: conv_chain
            async for sym, conv_chain in self._build_conv_chains(light_syms)
        }

        async for light_sym, sym in self._get_full_symbols(light_syms):
            yield TradedSymbolInfo(light_sym, sym, conv_chains[light_sym])

    async def _get_full_symbols(
        self, light_syms: Set[ProtoOALightSymbol]
    ) -> AsyncIterator[tuple[ProtoOALightSymbol, ProtoOASymbol]]:
        sym_ids = {sym.symbolId for sym in light_syms}

        found_syms, missing_sym_ids = self._get_saved_and_missing(
            self._id_to_full_symbol_map, sym_ids
        )

        light_symbol_map = await self._get_light_symbol_map()

        for sym_id, sym in found_syms.items():
            yield light_symbol_map[sym_id], sym

        sym_list_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOASymbolByIdReq(
                ctidTraderAccountId=trader_id,
                symbolId=missing_sym_ids,
            ),
            ProtoOASymbolByIdRes,
        )

        for sym in sym_list_res.symbol:
            self._id_to_full_symbol_map[sym.symbolId] = sym
            yield light_symbol_map[sym.symbolId], sym

    def _build_chainsyminfos(
        self,
        lightsym_chain: Sequence[ProtoOALightSymbol],
        lightsym_to_sym: Mapping[ProtoOALightSymbol, ProtoOASymbol],
    ) -> Iterator[ChainSymbolInfo]:
        first = lightsym_chain[0]
        reciprocal = self._dep_asset_id != first.quoteAssetId
        yield ChainSymbolInfo(first, lightsym_to_sym[first], reciprocal)

        for first, second in zip(lightsym_chain[:-1], lightsym_chain[1:]):
            reciprocal = first.baseAssetId != second.quoteAssetId
            yield ChainSymbolInfo(second, lightsym_to_sym[second], reciprocal)

    async def _build_conv_chains(
        self, light_syms: Set[ProtoOALightSymbol]
    ) -> AsyncIterator[tuple[ProtoOALightSymbol, DepositConvChains]]:
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

        def get_key2(res: ProtoOASymbolsForConversionRes) -> int:
            symbols = res.symbol
            first, second = symbols[0], symbols[1]
            first_assets = (first.baseAssetId, first.quoteAssetId)
            second_assets = {second.baseAssetId, second.quoteAssetId}

            for i, asset_id in enumerate(first_assets):
                if asset_id in second_assets:
                    other = first_assets[1 - i]
                    assert other not in second_assets
                    return other

            raise ValueError()

        asset_pair_to_symbol_map = await self._get_asset_pair_to_symbol_map()
        dep_asset_id = self._dep_asset_id

        convchain_to_download_asset_ids: set[int] = set()
        asset_id_to_convchain: dict[int, Sequence[ProtoOALightSymbol]] = {}

        for sym in light_syms:
            for asset_id in (sym.baseAssetId, sym.quoteAssetId):
                asset_pair = (asset_id, dep_asset_id)
                if asset_pair in asset_pair_to_symbol_map:
                    asset_id_to_convchain[asset_id] = [
                        asset_pair_to_symbol_map[asset_pair]
                    ]
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

        conv_chains_lightsyms = {
            sym for chain in asset_id_to_convchain.values() for sym in chain
        }

        lightsym_to_sym = {
            light_sym: sym
            async for light_sym, sym in self._get_full_symbols(conv_chains_lightsyms)
        }

        asset_id_to_syminfo_convchain = {
            asset_id: list(self._build_chainsyminfos(convchain, lightsym_to_sym))
            for asset_id, convchain in asset_id_to_convchain.items()
        }

        for sym in light_syms:
            yield sym, DepositConvChains(
                base_asset=ConvChain(
                    sym.baseAssetId, asset_id_to_syminfo_convchain[sym.baseAssetId]
                ),
                quote_asset=ConvChain(
                    sym.quoteAssetId, asset_id_to_syminfo_convchain[sym.quoteAssetId]
                ),
            )

    async def _get_light_symbol_map(
        self,
    ) -> Mapping[Union[int, str], ProtoOALightSymbol]:
        light_symbol_map: Optional[dict[Union[int, str], ProtoOALightSymbol]]

        if (light_symbol_map := self._light_symbol_map) is not None:
            return light_symbol_map

        light_sym_list_res = await self._client.send_request_from_trader(
            lambda trader_id: ProtoOASymbolsListReq(ctidTraderAccountId=trader_id),
            ProtoOASymbolsListRes,
        )
        light_symbols = light_sym_list_res.symbol

        light_symbol_map = self._light_symbol_map = {}
        light_symbol_map.update((sym.symbolId, sym) for sym in light_symbols)
        light_symbol_map.update((sym.symbolName.lower(), sym) for sym in light_symbols)

        # TODO: mypy complains about this, but it's correct
        # light_symbol_map = self._light_symbol_map = {
        #     id_or_name: sym
        #     for sym in light_sym_list_res.symbol
        #     for id_or_name in (sym.symbolId, sym.symbolName.lower())
        # }

        return light_symbol_map

    async def _get_asset_pair_to_symbol_map(
        self,
    ) -> Mapping[tuple[int, int], ProtoOALightSymbol]:
        if (asset_pair_to_symbol_map := self._asset_pair_to_symbol_map) is not None:
            return asset_pair_to_symbol_map

        light_symbol_map = await self._get_light_symbol_map()

        asset_pair_to_symbol_map = self._asset_pair_to_symbol_map = {
            asset_pair: sym
            for sym in light_symbol_map.values()
            for asset_pair in (
                (sym.baseAssetId, sym.quoteAssetId),
                (sym.quoteAssetId, sym.baseAssetId),
            )
        }

        return asset_pair_to_symbol_map

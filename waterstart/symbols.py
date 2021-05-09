from collections.abc import AsyncIterator, Mapping, MutableMapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Collection, Optional, TypeVar, Union, cast

from waterstart.client import OpenApiClient
from waterstart.openapi import (
    ProtoOALightSymbol,
    ProtoOASymbol,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsForConversionReq,
    ProtoOASymbolsForConversionRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
    ProtoOATrader,
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
class ConvChains:
    base_asset: Sequence[SymbolInfo]
    quote_asset: Sequence[SymbolInfo]


@dataclass(frozen=True)
class TradedSymbolInfo(SymbolInfo):
    conv_chains: ConvChains = field(hash=False)


T = TypeVar("T")
U = TypeVar("U")
T_SymInfo = TypeVar("T_SymInfo", bound=SymbolInfo)

# TODO: have a loop that subscribes to ProtoOASymbolChangedEvent and updates the
# changed symbols
class SymbolsList:
    def __init__(self, client: OpenApiClient, trader: ProtoOATrader) -> None:
        self.client = client
        self._trader = trader
        self._light_symbol_map: Optional[
            Mapping[Union[int, str], ProtoOALightSymbol]
        ] = None
        self._name_to_sym_info_map: MutableMapping[str, SymbolInfo] = {}
        self._id_to_full_symbol_map: MutableMapping[int, ProtoOASymbol] = {}

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

        sym_list_req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=self._trader.ctidTraderAccountId,
            symbolId=missing_sym_ids,
        )
        sym_list_res = await self.client.send_and_wait_response(
            sym_list_req, ProtoOASymbolByIdRes
        )

        for sym in sym_list_res.symbol:
            self._id_to_full_symbol_map[sym.symbolId] = sym
            yield light_symbol_map[sym.symbolId], sym

    async def _build_conv_chains(
        self, light_syms: Set[ProtoOALightSymbol]
    ) -> AsyncIterator[tuple[ProtoOALightSymbol, ConvChains]]:
        id_to_convlist_req = {
            asset_id: ProtoOASymbolsForConversionReq(
                # it's firstAssetId / lastAssetId
                ctidTraderAccountId=self._trader.ctidTraderAccountId,
                firstAssetId=asset_id,
                lastAssetId=self._trader.depositAssetId,
            )
            for sym in light_syms
            for asset_id in (sym.baseAssetId, sym.quoteAssetId)
        }

        id_to_convchain = {
            asset_id: res.symbol
            async for asset_id, res in self.client.send_and_wait_responses(
                id_to_convlist_req,
                ProtoOASymbolsForConversionRes,
                # TODO: verify this is correct
                lambda res: res.symbol[0].quoteAssetId,
            )
        }

        conv_chains_sym_names = {
            sym.symbolName for chain in id_to_convchain.values() for sym in chain
        }

        light_sym_to_sym_info = {
            sym_info.light_symbol: sym_info
            async for sym_info in self.get_sym_infos(conv_chains_sym_names)
        }

        id_to_sym_info_convchain = {
            asset_id: [light_sym_to_sym_info[sym] for sym in convchain]
            for asset_id, convchain in id_to_convchain.items()
        }

        for sym in light_syms:
            yield sym, ConvChains(
                base_asset=id_to_sym_info_convchain[sym.baseAssetId],
                quote_asset=id_to_sym_info_convchain[sym.quoteAssetId],
            )

    async def _get_light_symbol_map(
        self,
    ) -> Mapping[Union[int, str], ProtoOALightSymbol]:
        if self._light_symbol_map is None:
            light_sym_list_req = ProtoOASymbolsListReq(
                ctidTraderAccountId=self._trader.ctidTraderAccountId
            )
            light_sym_list_res = await self.client.send_and_wait_response(
                light_sym_list_req, ProtoOASymbolsListRes
            )
            self._light_symbol_map = {
                cast(Union[int, str], id_or_name): sym
                for sym in light_sym_list_res.symbol
                for id_or_name in (sym.symbolId, sym.symbolName)
            }

        return self._light_symbol_map

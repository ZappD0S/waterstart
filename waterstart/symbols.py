from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence, Set
from dataclasses import dataclass
from typing import Optional

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
class ConvChains:
    # TODO: maybe make this sequences of SymbolInfo?
    base_asset: Sequence[ProtoOALightSymbol]
    quote_asset: Sequence[ProtoOALightSymbol]


# TODO: remove conv_chains from here and make a subclass that has it
@dataclass(frozen=True)
class SymbolInfo:
    light_symbol: ProtoOALightSymbol
    symbol: ProtoOASymbol
    conv_chains: Optional[ConvChains] = None

    @property
    def name(self):
        return self.light_symbol.symbolName.lower()

    @property
    def id(self):
        return self.symbol.symbolId


# TODO: here we pass the Trader object to the constructor instead of passing
# the id to the methods below
class SymbolsList:
    def __init__(self, client: OpenApiClient) -> None:
        self.client = client
        self._light_symbols_map: Optional[Mapping[int, ProtoOALightSymbol]] = None
        self._symbols_map: MutableMapping[int, ProtoOASymbol] = {}

    # TODO: we keep a dict of SymbolInfo, and we create only the one that are missing
    # We keep two methods, one that returns a sequence of instances of SymbolInfo, while
    # the other the subclass with conv_chains. If we already have a SymbolInfo that doesn't
    # have conv_chains we update the saved one with the newly created
    async def get_symbols(
        self,
        trader: ProtoOATrader,
        subset: Optional[Iterable[str]] = None,
        conv_chains: bool = False,
    ) -> Sequence[SymbolInfo]:
        if subset is None:
            return await self._get_symbols(trader, conv_chains=conv_chains)

        return await self._get_symbols(
            trader,
            {name.lower() for name in subset},
            conv_chains=conv_chains,
        )

    @staticmethod
    def _get_symbols_subset(
        symbols: Iterable[ProtoOALightSymbol], subset: Set[str]
    ) -> Iterator[ProtoOALightSymbol]:
        subset = set(subset)
        for sym in symbols:
            if (name := sym.symbolName.lower()) in subset:
                yield sym
                subset.remove(name)

        if subset:
            raise ValueError(
                "The following symbols could not found: " + ", ".join(subset)
            )

    async def _get_light_symbols_map(
        self, account_id: int
    ) -> Mapping[int, ProtoOALightSymbol]:
        light_sym_list_req = ProtoOASymbolsListReq(ctidTraderAccountId=account_id)
        light_sym_list_res = await self.client.send_and_wait_response(
            light_sym_list_req, ProtoOASymbolsListRes
        )
        return {sym.symbolId: sym for sym in light_sym_list_res.symbol}

    async def _get_symbols_map(
        self, account_id: int, symbol_ids: Iterable[int]
    ) -> Mapping[int, ProtoOASymbol]:
        sym_list_req = ProtoOASymbolByIdReq(
            ctidTraderAccountId=account_id,
            symbolId=symbol_ids,
        )
        sym_list_res = await self.client.send_and_wait_response(
            sym_list_req, ProtoOASymbolByIdRes
        )
        return {sym.symbolId: sym for sym in sym_list_res.symbol}

    async def _get_symbols(
        self,
        trader: ProtoOATrader,
        subset: Optional[Set[str]] = None,
        conv_chains: bool = False,
    ) -> Sequence[SymbolInfo]:
        account_id = trader.ctidTraderAccountId

        if self._light_symbols_map is None:
            self._light_symbols_map = await self._get_light_symbols_map(account_id)

        light_symbols_map: Mapping[int, ProtoOALightSymbol] = self._light_symbols_map

        if subset is not None:
            light_symbols_map = {
                sym.symbolId: sym
                for sym in self._get_symbols_subset(light_symbols_map.values(), subset)
            }

        symbol_ids = light_symbols_map.keys()

        missing_ids = symbol_ids - self._symbols_map.keys()

        if missing_ids:
            self._symbols_map.update(
                await self._get_symbols_map(account_id, missing_ids)
            )

        assert symbol_ids <= self._symbols_map.keys()

        if not conv_chains:
            return [
                SymbolInfo(
                    sym,
                    self._symbols_map[sym_id],
                )
                for sym_id, sym in light_symbols_map.items()
            ]

        id_to_convlist_req = {
            asset_id: ProtoOASymbolsForConversionReq(
                # it's firstAssetId / lastAssetId
                ctidTraderAccountId=account_id,
                firstAssetId=asset_id,
                lastAssetId=trader.depositAssetId,
            )
            for sym in light_symbols_map.values()
            for asset_id in (sym.baseAssetId, sym.quoteAssetId)
        }

        id_to_convlist = {
            asset_id: res.symbol
            async for asset_id, res in self.client.send_and_wait_responses(
                id_to_convlist_req,
                ProtoOASymbolsForConversionRes,
                lambda res: res.symbol[0].quoteAssetId,
            )
        }

        return [
            SymbolInfo(
                sym,
                self._symbols_map[sym_id],
                ConvChains(
                    id_to_convlist[sym.baseAssetId], id_to_convlist[sym.quoteAssetId]
                ),
            )
            for sym_id, sym in light_symbols_map.items()
        ]

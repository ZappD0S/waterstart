import datetime
from .schedule import ExecutionSchedule
from .price.market_data_producer import (
    HistoricalMarketDataProducer,
    LiveMarketDataProducer,
)
from .client.trader import TraderClient
from .inference.engine import InferenceEngine


class Executor:
    def __init__(
        self, engine: InferenceEngine, client: TraderClient, schedule: ExecutionSchedule
    ) -> None:
        # create Account state and hidden state array
        # for the Account state we need to make a request for the balance
        start = datetime.datetime.now()

        historical_market_data_producer = HistoricalMarketDataProducer(
            client,
            schedule,
            start,
            engine.net_modules.window_size - 1,
        )

        # TODO: for now we send ProtoOAReconcileReq and if there are any one positions at
        # start we throw an exception
        # Later we will use ProtoOADealListReq to retrieve the prices and sizes of
        # currently open trades

        self._live_market_data_producer = LiveMarketDataProducer(
            client, schedule, start
        )

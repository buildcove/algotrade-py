"""
51.52% chance of having a consecutive bar in the same direction.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pprint import pprint
from typing import Tuple

import holidays
import ib_async
import pandas as pd
import pytz
from algotrade.core import brokers, engine, nz, repository, types
from algotrade.core.utils import adjust_daily_dates, simulate_stream
from algotrade.schedules import CME_TZ
from structlog import get_logger

logger = get_logger()


@dataclass
class TwoBarExtra:
    prev_action: types.Action = None
    price_17_00: float = 0.0
    price_17_00_dt: datetime = None

    drawdown: float = 0.0
    drawdown_price: float = 0.0
    drawdown_dt: datetime = None

    stoploss: float = 0.0


@dataclass
class TwoBarParams:
    symbol: ib_async.Contract | str

    open_trade: bool = False
    missed_trade: bool = False
    has_trade_today: bool = False
    end_of_day_dt: datetime = None
    tz: str = CME_TZ


class TwoBarStrategy(engine.BaseEngine):
    """
    WARNINGS:
    - The strategy used US/Centeral timezone, but when you submit an order
    you have to change the timezone base on the IB Gateway timezone.
    """

    def __init__(
        self,
        broker: brokers.BacktestingBroker | brokers.IBKRBroker,
        param: TwoBarParams,
        extra: TwoBarExtra,
    ):
        super().__init__(broker)

        self.broker = broker
        # Basics
        self.open_time = time(17, 0)
        self.close_time = time(15, 59)
        self.close_time_1h = time(15, 0)

        self.previous_stream = None

        self.default_data: TwoBarExtra = extra
        self.d: TwoBarExtra = deepcopy(self.default_data)

        self.param_default: TwoBarParams = param
        self.param: TwoBarParams = deepcopy(self.param_default)

        self.tick_size = 1 / 64
        self.fatal = False

    def get_closing_datetime(self, _date):
        if type(_date) is not date:
            _date = _date.date()

        dt = datetime.combine(_date, self.close_time)

        if _date in holidays.NYSE():
            self.get_closing_datetime(dt + timedelta(days=1))

        dt = dt + timedelta(days=1)
        tz = pytz.timezone(self.param.tz)
        return tz.localize(dt)

    def get_opening_price_and_dt_from_historical(self, end_dt="", previous_bars=[]) -> Tuple[float, datetime, datetime]:
        """
        Will only be called if dt > 17:00. It's a fallback mechanism when the opening price is missed.
        IBKR
        """
        bars = self.broker.ib.reqHistoricalData(
            self.param.symbol,
            endDateTime=end_dt,
            durationStr="2 D",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=False,
        )
        bars = [asdict(bar) for bar in bars]
        bars += previous_bars
        bars = sorted(bars, key=lambda x: x["date"], reverse=True)

        # find the first 15:00 bar, the previous bar is the opening price
        for index, row in enumerate(bars):
            if row["date"].time() == self.close_time_1h:
                prev = bars[index - 1]
                open_, open_dt = prev["open"], prev["date"]
                close_dt = self.get_closing_datetime(bars[0]["date"].date())
                return open_, open_dt, close_dt

        end_dt = bars[-1]["date"]
        return self.get_opening_price_and_dt_from_historical(end_dt, bars)

    def new_day_order_checks(self):
        orders = self.broker.get_orders(self.param.symbol)
        logger.debug("New day: Current open orders", orders=orders)

        if orders:
            self.broker.cancel_orders(orders)

        # Important to put it below after canceling the orders
        # Previous day might fail to close the trade
        positions = self.broker.get_positions(self.param.symbol)
        logger.debug("New day: Current positions", positions=positions)

        if positions:
            self.param.has_trade_today = True
            logger.error("FATAL! There is an open positions today.")
            raise Exception("FATAL! There is an open positions today.")

    def prerequisite(self, stream: types.Stream):
        if self.fatal:
            raise Exception("Fatal is true. Exiting the strategy.")

        # Will not process the same stream twice. Speed up the process.
        if (
            isinstance(self.broker, brokers.IBKRBroker)
            and self.previous_stream is not None
            and self.previous_stream.bid == stream.bid
            and self.previous_stream.ask == stream.ask
        ):
            return

        if isinstance(self.broker, brokers.IBKRBroker):
            if self.param.end_of_day_dt is None or stream.timestamp > self.param.end_of_day_dt:
                self.new_day_order_checks()

                self.param = deepcopy(self.param_default)
                self.d = deepcopy(self.default_data)

                logger.debug("Getting opening price from historical data")
                self.d.price_17_00, self.d.price_17_00_dt, self.param.end_of_day_dt = (
                    self.get_opening_price_and_dt_from_historical()
                )
                logger.debug(f"Open price: {self.d.price_17_00} | {self.d.price_17_00_dt} | {self.param.end_of_day_dt}")

                # TODO: previous action

        if isinstance(self.broker, brokers.BacktestingBroker):
            if (
                self.previous_stream is not None
                and self.previous_stream.timestamp.time() == self.close_time
                and stream.timestamp.time() == self.open_time
                # and stream.timestamp.date() not in holidays.NYSE()
            ):
                # WARNING: The first record should be less than 17:00
                self.new_day_order_checks()

                self.param = deepcopy(self.param_default)
                self.d = deepcopy(self.default_data)

                self.d.price_17_00 = stream.bid
                self.d.price_17_00_dt = stream.timestamp
                self.param.end_of_day_dt = self.get_closing_datetime(stream.timestamp.date())
                logger.debug(f"Open price: {self.d.price_17_00} | {self.d.price_17_00_dt} | {self.param.end_of_day_dt}")

                bar = self.broker.get_previous_day_candle(self.param.symbol, end_date=self.d.price_17_00_dt.date())
                self.d.prev_action = types.Action.LONG if bar["a_close"] > bar["a_open"] else types.Action.SHORT
                logger.debug(f"Previous action: {self.d.prev_action}")

                # if self.d.prev_action == types.Action.LONG:
                #     distance_from_ = nz.decimal_ticks_diff(bar["a_open"], bar["a_low"])
                #     half_distance = distance_from_ / 2
                #     self.d.stoploss = nz.decimal_subtraction_by_ticks(bar["a_open"], half_distance)

                # elif self.d.prev_action == types.Action.SHORT:
                #     distance_from_ = nz.decimal_ticks_diff(bar["a_high"], bar["a_open"])
                #     half_distance = distance_from_ / 2
                #     self.d.stoploss = nz.decimal_addition_by_ticks(bar["a_open"], half_distance)

                self.d.stoploss = bar["a_low"] if self.d.prev_action == types.Action.LONG else bar["a_high"]

                logger.debug(f"Stoploss: {self.d.stoploss}", bar=bar)

    def next(self, stream: types.Stream, historical: bool = False):
        self.prerequisite(stream)

        self.historical: bool = historical
        self.previous_stream = stream

        # It's important it's written after the previous block
        if self.param.missed_trade or self.param.has_trade_today or self.d.price_17_00 == 0.0:
            return

        if not self.param.open_trade:
            if self.historical:
                logger.debug("missed trade", data=asdict(self.d))
                self.param.missed_trade = True
                return

            order = types.MarketOrder(
                symbol=self.param.symbol,
                quantity=1,
                action=self.d.prev_action,
                timestamp=stream.timestamp,
                extra=self.d,
                params=self.param,
            )
            stop_order = types.StopOrder(
                symbol=self.param.symbol,
                quantity=1,
                action=types.Action.SHORT if self.d.prev_action == types.Action.LONG else types.Action.LONG,
                price=self.d.stoploss,
                timestamp=stream.timestamp,
                extra=self.d,
                params=self.param,
            )
            bracket = types.BracketOrder(
                symbol=self.param.symbol,
                order=order,
                stop_loss=stop_order,
                take_profit=None,
            )
            self.broker.order(bracket)
            self.param.open_trade = True

        else:
            self.exit(stream)

    def exit(self, stream: types.Stream):
        if stream.timestamp >= self.param.end_of_day_dt:
            positions = self.broker.get_positions(self.param.symbol)
            # The stoploss has been hit if the positions are empty
            if not positions:
                return

            number_of_positions = len(positions)
            if number_of_positions > 1:
                logger.error(
                    "FATAL! More than one position. Forcing to close all positions.", number_of_positions=number_of_positions
                )
                self.fatal = True

            orders = self.broker.get_orders(self.param.symbol)
            if orders:
                self.broker.cancel_orders(orders)

            group_by_action = {types.Action.LONG: [], types.Action.SHORT: []}
            for pos in positions:
                group_by_action[pos.action].append(pos)

            for action, positions in group_by_action.items():
                if not positions:
                    continue

                action = types.Action.LONG if action == types.Action.SHORT else types.Action.SHORT
                quantity = sum([pos.quantity for pos in positions])

                logger.debug(f"Closing {action}: {stream.timestamp}")
                order = types.MarketOrder(
                    symbol=self.param.symbol,
                    quantity=quantity,
                    action=action,
                    timestamp=stream.timestamp,
                    extra=self.d,
                    params=self.param,
                )
                self.broker.order(order)
                self.close_trade_post_action()

    def close_trade_post_action(self):
        logger.debug("Closed trade post actions")
        self.d = deepcopy(self.default_data)
        self.param.has_trade_today = True


def backtest() -> pd.DataFrame:
    # Stream data
    d = pd.read_csv("~/desktop/test.csv")
    d["dt"] = pd.to_datetime(d["dt"])
    d["a_dt"] = pd.to_datetime(d["a_dt"])

    first_row_a_dt = d.iloc[0]["a_dt"]
    d = d[d["a_dt"] > first_row_a_dt]

    # first_row_a_dt = pd.to_datetime("2024-06-9 15:00:00")
    a_dt = pd.to_datetime("2024-06-06 15:00:00")
    d = d[d["a_dt"] >= a_dt]

    # d.to_csv("~/desktop/1d.csv", index=False)
    logger.debug(f"Total rows: {len(d)}")

    # The first row has been cleaned up and it's the opening price of the day.
    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo)
    param = TwoBarParams(symbol="ZN")
    two_wave_strategy = TwoBarStrategy(broker, param, extra=TwoBarExtra())

    for _, row in d.iterrows():
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]
        timestamp = row["dt"].tz_localize(param.tz)

        # Simulate the stream bid and ask
        stream = simulate_stream(open_price, timestamp)
        two_wave_strategy.next(stream)
        broker.next(stream)

        stream = simulate_stream(high_price, timestamp)
        two_wave_strategy.next(stream)
        broker.next(stream)

        stream = simulate_stream(low_price, timestamp)
        two_wave_strategy.next(stream)
        broker.next(stream)

        stream = simulate_stream(close_price, timestamp)
        two_wave_strategy.next(stream)
        broker.next(stream)

    trades = broker.closed_positions
    trades = [asdict(trade) for trade in trades]
    return pd.json_normalize(trades), two_wave_strategy
    # plogger.debug(trades)
    # df = pd.json_normalize(trades)
    # logger.debug(df[["timestamp", "trade_type", "price", "extra.stoploss"]])
    # logger.debug(df.columns)


class Live:
    """
    TODO: rolling window for parameters
    """

    def __init__(self) -> None:
        logger.debug("Running live")
        self.contract = ib_async.Future(symbol="ZN", exchange="CBOT", lastTradeDateOrContractMonth="202409")

        self.repo = repository.Repository()
        self.param = TwoBarParams(symbol=self.contract)
        self.extra = TwoBarExtra()

    def run(self):
        self.broker = brokers.IBKRBroker(self.repo)
        self.strategy = TwoBarStrategy(
            broker=self.broker,
            param=self.param,
            extra=self.extra,
        )

        duration = "1 D"
        days_elaps_since_open = 0
        open_dt = self.strategy.d.price_17_00_dt
        if open_dt is not None:
            now = datetime.now().astimezone(pytz.timezone(self.param.tz))
            days_elaps_since_open = (now - open_dt).days
            duration = f"{days_elaps_since_open + 1} D"

        historical = self.broker.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow="BID_ASK",
            useRTH=False,
        )
        for row in historical:
            stream = types.Stream(row.open, row.open, row.date)
            self.strategy.next(stream, historical=True)

            stream = types.Stream(row.high, row.high, row.date)
            self.strategy.next(stream, historical=True)

            stream = types.Stream(row.low, row.low, row.date)
            self.strategy.next(stream, historical=True)

            stream = types.Stream(row.close, row.close, row.date)
            self.strategy.next(stream, historical=True)

        self.broker.ib.reqTickByTickData(self.contract, "BidAsk", numberOfTicks=1)
        self.broker.ib.pendingTickersEvent += self.on_event

        self.broker.ib.run()

    def on_event(self, event):
        try:
            ticker = event.pop()
            ticker.time = ticker.time.astimezone(pytz.timezone(self.param.tz))
            stream = types.Stream(
                bid=ticker.bid,
                ask=ticker.ask,
                timestamp=ticker.time,
            )
            # logger.debug(str(stream.timestamp))
            self.two_wave_strategy.next(stream)
        except Exception as e:
            logger.exception(e)

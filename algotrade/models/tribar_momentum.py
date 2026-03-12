"""
This module aims to fully automate and backtest the Tribar Momentum strategy.

Tribar momentum
LONG - SMA200 Follow
- Current bar closed is above the highs of the last 3 bars
- Current bar closed is above 9 EMA
- Current bar closed is above 200 SMA.

Short - SMA200 reversal
- Current bar closed is below the lows of the last 3 bars
- Current bar closed is below 9 EMA
- Current bar closed is above 200 SMA.
- Previous signal has SMA200_reversal

WARNING:
I used TradingView when I was developing this strategy manually.
And the (day) candle starts at 6PM central time.

Meanwhile, the IBKR candle starts at 12AM for 4 hour timeframe.
You have to manually reconstruct the data from 1 hour to 4 hour, and match how trading view prints the data.

Apr. 5, 2025
- IBKR futures hourly data doesn't match with TradingView, but Databento does.

Apr. 7, 2025
- I have figured out how to pulldata from TradingView.
"""

import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum

import ib_async
import pandas as pd
import pytz
import retry
from algotrade.core import brokers, engine, repository, types, utils
from algotrade.core.safety import require_live_trading_enabled
from algotrade.schedules import CME_SCHEDULE, IB_TZ
from algotrade.settings import settings
from structlog import get_logger
from tvDatafeed import Interval, TvDatafeed

logger = get_logger()


class Variation(Enum):
    ABOVE_200SMA_LONG = "ABOVE_200SMA_LONG"
    ABOVE_200SMA_REVERSAL_SHORT = "ABOVE_200SMA_REVERSAL_SHORT"  # Above 200SMA, enough room to SMA200

    BELOW_200SMA_SHORT = "BELOW_200SMA_SHORT"
    BELOW_200SMA_REVERSAL_LONG = "BELOW_200SMA_REVERSAL_LONG"  # Below 200SMA, enough room to SMA200


@dataclass
class Candle(types.Candle):
    sma200: float = None
    ema9: float = None


@dataclass
class Stream(types.Stream):
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    timestamp: datetime = None
    sma200: float = None
    ema9: float = None


@dataclass
class Extra:
    variation: Variation = None
    drawdown: float = 0.0
    drawdown_price: float = 0.0
    drawdown_dt: datetime = None

    realized_rr: float = 0.0


@dataclass
class Param:
    symbol: ib_async.Contract | str
    tick_size: float
    tick_value: float
    tz: str = IB_TZ
    max_trade_per_week: int = 2
    order_offset = 1
    risk_reward: float = 3.0

    capital: float = 10000.0
    min_quantity: int = 1
    max_quantity: int = 1


@dataclass
class Var:
    week_opening_price: float = None
    week_opening_dt: datetime = None
    week_high: float = None
    week_low: float = None

    open_trade: bool = False
    missed_trade: bool = False

    # Rule: 1 winning setup or maximum of (1 loss) 2 trades per week per ticker
    first_setup_this_week: types.BracketOrder = None
    first_setup_this_week_winning: bool = False
    second_setup_this_week: types.BracketOrder = None
    last_week_setup: types.BracketOrder = None

    previous_fourth_bar: Candle = None
    previous_third_bar: Candle = None
    previous_second_bar: Candle = None
    previous_first_bar: Candle = None

    previous_above_200sma: float = None
    previous_above_9ema: float = None
    previous_long_three_bar: Candle = None
    previous_short_three_bar: Candle = None


class PrererequisiteFailed(Exception):
    pass


class TribarMomentumStrategy(engine.BaseEngine):
    def __init__(
        self,
        broker: brokers.BacktestingBroker | brokers.IBKRBroker,
        param: Param,
        var: Var,
        extra: Extra,
    ):
        super().__init__(broker)
        self.broker = broker

        # Basics
        self.previous_stream: Candle = None

        self.default_extra: Extra = extra
        self.extra: Extra = deepcopy(self.default_extra)

        self.param_default: Param = param
        self.param: Param = deepcopy(self.param_default)

        self.var_default: Var = var
        self.var: Var = deepcopy(self.var_default)

        self.fatal = False

        self.setups = []
        self.last_historical_pull = None

    @retry.retry(tries=10, max_delay=10, delay=1, backoff=3)
    def pull_historical_data(self):
        start = time.time()
        tv = TvDatafeed()
        tv.token = settings.tradingview_key

        start = time.time()
        n_bars = 200 + (31 * 2)  # 200 SMA + 1 week (28.75 bars)
        data = tv.get_hist(
            symbol="MGC",
            exchange="COMEX_MINI",
            interval=Interval.in_4_hour,
            n_bars=n_bars,
            fut_contract=1,
            extended_session=True,
        )
        data = data.reset_index()
        data["datetime"] = data["datetime"].dt.tz_localize("UTC")

        logger.debug("Data fetch time:", time=time.time() - start)

        start = time.time()
        data["200sma"] = data["close"].rolling(window=200).mean()
        data["9ema"] = data["close"].ewm(span=9, adjust=False).mean()
        data = data.dropna()
        data = data.reset_index()

        logger.debug("Indicator calculation time:", time=time.time() - start)

        start = time.time()
        for index, row in data.iterrows():
            open_price = row["open"]
            high_price = row["high"]
            low_price = row["low"]
            close_price = row["close"]
            timestamp = row["datetime"]

            stream = Candle(
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                timestamp=timestamp,
                sma200=row["200sma"],
                ema9=row["9ema"],
            )

            # WARNING: the last bar is the current bar. It will be use to determine an entry.
            if index == data.shape[0] - 1:
                self.next(stream, historical=False)
                continue

            self.next(stream, historical=True)

        logger.debug("Historical data processing time:", time=time.time() - start)
        self.last_historical_pull = datetime.now(pytz.timezone(self.param.tz))

    def get_quantity(self, stoploss_ticks: int) -> int:
        # 1% of the account size
        risk_per_contract = stoploss_ticks * self.param.tick_value
        if risk_per_contract <= 0:
            raise ValueError("Stop loss must be greater than 0.")

        risk_amount = self.param.capital * 0.01  # 1% of account
        raw_quantity = risk_amount / risk_per_contract
        quantity = int(raw_quantity)

        quantity = max(self.param.min_quantity, min(quantity, self.param.max_quantity))
        # logger.debug("Quantity", quantity=quantity, risk_amount=risk_amount, risk_per_contract=risk_per_contract)
        return quantity

    def is_ready(self, stream: Candle) -> bool:
        if self.fatal:
            raise Exception("Fatal is true. Exiting the strategy.")

        if not isinstance(stream, Candle):
            raise TypeError("Stream must be of type Candle.")

        # Check if the date is Sunday, if so, reset the variables
        # Except the previous bars high and low
        tz = utils.get_eastern_timezone_label(stream.timestamp)
        open_time = CME_SCHEDULE.open_times_utc[tz]
        if stream.timestamp.weekday() == 6 and stream.timestamp.time() == open_time:
            previous_fourth_bar = self.var.previous_fourth_bar
            previous_third_bar = self.var.previous_third_bar
            previous_second_bar = self.var.previous_second_bar
            previous_first_bar = self.var.previous_first_bar

            last_week_setup = None
            if self.var.open_trade:
                last_week_setup = self.var.second_setup_this_week or self.var.first_setup_this_week

            self.param = deepcopy(self.param_default)
            self.var = deepcopy(self.var_default)
            self.extra = deepcopy(self.default_extra)

            self.var.previous_fourth_bar = previous_fourth_bar
            self.var.previous_third_bar = previous_third_bar
            self.var.previous_second_bar = previous_second_bar
            self.var.previous_first_bar = previous_first_bar

            self.var.week_opening_price = stream.open
            self.var.week_opening_dt = stream.timestamp
            self.var.week_high = stream.high
            self.var.week_low = stream.low

            if last_week_setup:
                self.var.last_week_setup = last_week_setup
                self.var.open_trade = True

            logger.debug(f"Week opening price: {self.var.week_opening_price} | {self.var.week_opening_dt}")

        if self.var.week_high is None or self.var.week_high < stream.high:
            self.var.week_high = stream.high

        if self.var.week_low is None or self.var.week_low > stream.low:
            self.var.week_low = stream.low

        # Push the previous bars to the next bar
        if (
            self.var.previous_fourth_bar
            and self.var.previous_third_bar
            and self.var.previous_second_bar
            and self.var.previous_first_bar
        ):
            self.var.previous_fourth_bar = self.var.previous_third_bar
            self.var.previous_third_bar = self.var.previous_second_bar
            self.var.previous_second_bar = self.var.previous_first_bar
            self.var.previous_first_bar = self.previous_stream
            self.previous_stream = stream

            return True
        elif not self.var.previous_first_bar:
            self.var.previous_first_bar = self.previous_stream
            self.previous_stream = stream
            return
        elif not self.var.previous_second_bar:
            self.var.previous_second_bar = self.var.previous_first_bar
            self.var.previous_first_bar = self.previous_stream
            self.previous_stream = stream
            return
        elif not self.var.previous_third_bar:
            self.var.previous_third_bar = self.var.previous_second_bar
            self.var.previous_second_bar = self.var.previous_first_bar
            self.var.previous_first_bar = self.previous_stream
            self.previous_stream = stream
            return
        elif not self.var.previous_fourth_bar:
            self.var.previous_fourth_bar = self.var.previous_third_bar
            self.var.previous_third_bar = self.var.previous_second_bar
            self.var.previous_second_bar = self.var.previous_first_bar
            self.var.previous_first_bar = self.previous_stream
            self.previous_stream = stream
            return
        else:
            raise PrererequisiteFailed("Previous bars are not set.")

    def monitor_ongoing_trade(self, stream: Candle, long_three_bar: bool, short_three_bar: bool):
        if not self.var.open_trade:
            logger.debug("No ongoing trade to monitor", data=stream)
            return

        if (
            self.var.first_setup_this_week
            and self.var.first_setup_this_week.take_profit.timestamp is None
            and self.var.first_setup_this_week.take_profit.timestamp is None
        ):
            order = self.var.first_setup_this_week
        elif (
            self.var.second_setup_this_week
            and self.var.second_setup_this_week.take_profit.timestamp is None
            and self.var.second_setup_this_week.stop_loss.timestamp is None
        ):
            order = self.var.second_setup_this_week
        elif (
            self.var.last_week_setup
            and self.var.last_week_setup.take_profit.timestamp is None
            and self.var.last_week_setup.stop_loss.timestamp is None
        ):
            order = self.var.last_week_setup
        else:
            return

        # TODO: create unit tests that last_week_setup being handled correctly
        if order.order.action == types.Action.LONG:
            if stream.low <= order.stop_loss.price:
                logger.debug("Stop loss hit", data=stream)
                order.stop_loss.timestamp = stream.timestamp
                self.var.open_trade = False

                if self.var.last_week_setup:
                    self.var.last_week_setup = None
                    logger.debug("Last week setup closed", data=stream)

                realized_rr = utils.compute_rr(
                    entry=order.order.price,
                    stop_loss=order.stop_loss.price,
                    exit=order.stop_loss.price,
                )
                logger.debug("Realized RR", realized_rr=realized_rr)
                order.extra.realized_rr = realized_rr

                self.setups.append(order)
            elif stream.high >= order.take_profit.price:
                logger.debug("Take profit hit", data=stream)
                order.take_profit.timestamp = stream.timestamp
                self.var.open_trade = False

                if self.var.last_week_setup:
                    self.var.last_week_setup = None
                    logger.debug("Last week setup closed", data=stream)

                realized_rr = utils.compute_rr(
                    entry=order.order.price,
                    stop_loss=order.stop_loss.price,
                    exit=order.take_profit.price,
                )
                logger.debug("Realized RR", realized_rr=realized_rr)
                order.extra.realized_rr = realized_rr

                self.setups.append(order)

            elif self.var.last_week_setup and short_three_bar:
                logger.debug("Last week setup closed - against the setup", data=stream)
                realized_rr = utils.compute_rr(
                    entry=order.order.price,
                    stop_loss=order.stop_loss.price,
                    exit=stream.close,
                )
                logger.debug("Realized RR", realized_rr=realized_rr)
                order.extra.realized_rr = realized_rr
                # Set the stoploss/take profit to the current price
                if stream.close >= order.take_profit.price:
                    order.take_profit.timestamp = stream.timestamp
                    order.take_profit.price = stream.close
                else:
                    order.stop_loss.price = stream.close
                    order.stop_loss.timestamp = stream.timestamp

                self.var.last_week_setup = None
                self.var.open_trade = False

                self.setups.append(order)
                self.broker.close_position(order.symbol)

            return

        elif order.order.action == types.Action.SHORT:
            if stream.high >= order.stop_loss.price:
                logger.debug("Stop loss hit", data=stream)
                order.stop_loss.timestamp = stream.timestamp
                self.var.open_trade = False

                if self.var.last_week_setup:
                    self.var.last_week_setup = None
                    logger.debug("Last week setup closed", data=stream)

                realized_rr = utils.compute_rr(
                    entry=order.order.price,
                    stop_loss=order.stop_loss.price,
                    exit=order.stop_loss.price,
                )
                logger.debug("Realized RR", realized_rr=realized_rr)
                order.extra.realized_rr = realized_rr
                self.setups.append(order)
            elif stream.low <= order.take_profit.price:
                logger.debug("Take profit hit", data=stream)
                order.take_profit.timestamp = stream.timestamp
                self.var.open_trade = False

                if self.var.last_week_setup:
                    self.var.last_week_setup = None
                    logger.debug("Last week setup closed", data=stream)

                realized_rr = utils.compute_rr(
                    entry=order.order.price,
                    stop_loss=order.stop_loss.price,
                    exit=order.take_profit.price,
                )
                logger.debug("Realized RR", realized_rr=realized_rr)
                order.extra.realized_rr = realized_rr
                self.setups.append(order)

            elif self.var.last_week_setup and long_three_bar:
                logger.debug("Last week setup closed - against the setup", data=stream)
                realized_rr = utils.compute_rr(
                    entry=order.order.price,
                    stop_loss=order.stop_loss.price,
                    exit=stream.close,
                )
                logger.debug("Realized RR", realized_rr=realized_rr)
                order.extra.realized_rr = realized_rr

                # Set the stoploss/take profit to the current price
                if stream.close <= order.take_profit.price:
                    order.take_profit.timestamp = stream.timestamp
                    order.take_profit.price = stream.close
                else:
                    order.stop_loss.price = stream.close
                    order.stop_loss.timestamp = stream.timestamp

                self.var.last_week_setup = None
                self.var.open_trade = False

                self.setups.append(order)
                self.broker.close_position(order.symbol)

            return

    def create_bracket(self, action: types.Action, stream: Candle):
        if action == types.Action.LONG:
            entry_price = self.var.previous_first_bar.close - (self.param.order_offset * self.param.tick_size)
            stop_price = self.var.previous_first_bar.low
            stoploss_ticks = int((entry_price - stop_price) / self.param.tick_size)
            quantity = self.get_quantity(stoploss_ticks)

            order = types.LimitOrder(
                symbol=self.param.symbol,
                quantity=quantity,
                price=entry_price,
                action=types.Action.LONG,
                timestamp=None,
            )

            stop_loss = types.StopOrder(
                symbol=self.param.symbol,
                price=stop_price,
                action=types.Action.SHORT,
                quantity=quantity,
                timestamp=None,
            )

            take_profit_price = entry_price + (entry_price - stop_price) * self.param.risk_reward
            take_profit = types.LimitOrder(
                symbol=self.param.symbol,
                quantity=quantity,
                price=take_profit_price,
                action=types.Action.SHORT,
                timestamp=None,
            )
            bracket = types.BracketOrder(
                symbol=self.param.symbol,
                order=order,
                stop_loss=stop_loss,
                take_profit=take_profit,
                extra=deepcopy(self.extra),
                params=deepcopy(self.param),
                var=deepcopy(self.var),
                timestamp=stream.timestamp,
            )
        elif action == types.Action.SHORT:
            entry_price = self.var.previous_first_bar.close + (self.param.order_offset * self.param.tick_size)
            stop_loss_price = self.var.previous_first_bar.high
            stoploss_ticks = int((stop_loss_price - entry_price) / self.param.tick_size)
            quantity = self.get_quantity(stoploss_ticks)

            order = types.LimitOrder(
                symbol=self.param.symbol, quantity=quantity, price=entry_price, action=types.Action.SHORT, timestamp=None
            )

            stop_loss = types.StopOrder(
                symbol=self.param.symbol,
                price=stop_loss_price,
                action=types.Action.LONG,
                quantity=quantity,
                timestamp=None,
            )

            take_profit_price = entry_price - (stop_loss_price - entry_price) * self.param.risk_reward
            take_profit = types.LimitOrder(
                symbol=self.param.symbol,
                quantity=quantity,
                price=take_profit_price,
                action=types.Action.LONG,
                timestamp=None,
            )

            bracket = types.BracketOrder(
                symbol=self.param.symbol,
                order=order,
                stop_loss=stop_loss,
                take_profit=take_profit,
                extra=deepcopy(self.extra),
                params=deepcopy(self.param),
                var=deepcopy(self.var),
                timestamp=stream.timestamp,
            )

        return bracket

    def next(self, stream: Candle, historical: bool = False):
        # logger.debug("Next", data=stream, historical=historical)

        if isinstance(stream, Candle):
            if not self.is_ready(stream):
                return

        if (
            self.var.first_setup_this_week and self.var.first_setup_this_week.take_profit.timestamp
        ) or self.var.second_setup_this_week:
            # logger.warning("Maximum number of trades reached for the week.")
            return

        if not (self.var.previous_first_bar and self.var.previous_second_bar and self.var.previous_third_bar):
            # logger.debug("Waiting for previous bars to be set")
            return

        above_200_sma = self.var.previous_first_bar.close > self.var.previous_first_bar.sma200
        above_9_ema = self.var.previous_first_bar.close > self.var.previous_first_bar.ema9
        crossing_200_sma = (
            self.var.previous_first_bar.low <= self.var.previous_first_bar.sma200 <= self.var.previous_first_bar.high
        )

        long_three_bar = (
            self.var.previous_first_bar.close > self.var.previous_second_bar.high
            and self.var.previous_first_bar.close > self.var.previous_third_bar.high
            and self.var.previous_first_bar.close > self.var.previous_fourth_bar.high
            and above_9_ema
        )
        short_three_bar = (
            self.var.previous_first_bar.close < self.var.previous_second_bar.low
            and self.var.previous_first_bar.close < self.var.previous_third_bar.low
            and self.var.previous_first_bar.close < self.var.previous_fourth_bar.low
            and not above_9_ema
        )
        if not (long_three_bar or short_three_bar):
            # logger.debug("No signal", data=stream)
            return

        if self.var.open_trade:
            # logger.warning("Open trade already exists.")
            self.monitor_ongoing_trade(stream, long_three_bar, short_three_bar)
            return

        bracket = None
        if long_three_bar and above_9_ema and above_200_sma:
            self.extra.variation = Variation.ABOVE_200SMA_LONG
            logger.debug("Long three bar", data=stream)
            bracket = self.create_bracket(types.Action.LONG, stream)
        elif short_three_bar and not above_200_sma and not above_9_ema:
            self.extra.variation = Variation.BELOW_200SMA_SHORT
            bracket = self.create_bracket(types.Action.SHORT, stream)
        elif short_three_bar and self.var.previous_short_three_bar and above_200_sma:
            self.extra.variation = Variation.ABOVE_200SMA_REVERSAL_SHORT
            bracket = self.create_bracket(types.Action.SHORT, stream)
        elif long_three_bar and self.var.previous_long_three_bar and not above_200_sma:
            self.extra.variation = Variation.BELOW_200SMA_REVERSAL_LONG
            bracket = self.create_bracket(types.Action.LONG, stream)

        if bracket and not self.var.first_setup_this_week:
            self.var.first_setup_this_week = bracket
            self.var.open_trade = True

            logger.debug("Setup", variation=self.extra.variation, crossing_200_sma=crossing_200_sma)
            logger.debug("First setup executed", data=stream, historical=historical, bracket=bracket)

            if not historical:
                logger.debug("Placing order", historical=historical)
                self.broker.order(bracket)
                self.var.first_setup_this_week = bracket
                self.var.open_trade = True

        elif (
            self.var.first_setup_this_week
            and self.var.first_setup_this_week.stop_loss.timestamp is not None
            and self.var.second_setup_this_week is None
        ):
            self.var.second_setup_this_week = bracket
            self.var.open_trade = True

            logger.debug("Setup", variation=self.extra.variation, crossing_200_sma=crossing_200_sma)
            logger.debug("Second setup executed", data=stream, historical=historical, bracket=bracket)

            if not historical:
                self.broker.order(bracket)
                logger.debug("Placing order")

        # Warning: this should be at the end of the function
        if long_three_bar or short_three_bar:
            self.var.previous_long_three_bar = long_three_bar
            self.var.previous_short_three_bar = short_three_bar
            self.var.previous_above_200sma = above_200_sma
            self.var.previous_above_9ema = above_200_sma


def backtest():
    d = pd.read_csv("~/Desktop/data.csv")
    d["time"] = pd.to_datetime(d["time"])
    # 2025 and onwards
    d = d[d["time"] >= "2025-01-01"]
    d["time"] = d["time"].dt.tz_convert("UTC")
    d["200sma"] = d["close"].rolling(window=200).mean()
    d["9ema"] = d["close"].ewm(span=9, adjust=False).mean()
    d = d.dropna()

    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo)
    param = Param(
        symbol="MGC",
        tick_size=0.1,
        tick_value=1,
        tz=IB_TZ,
    )
    var = Var()
    extra = Extra()
    strategy = TribarMomentumStrategy(
        broker=broker,
        param=param,
        var=var,
        extra=extra,
    )

    # strategy.pull_historical_data()

    for _, row in d.iterrows():
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]
        timestamp = row["time"]

        stream = Candle(
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            timestamp=timestamp,
            sma200=row["200sma"],
            ema9=row["9ema"],
        )
        strategy.next(stream, historical=True)

        stream = utils.simulate_stream(open_price, timestamp)
        broker.next(stream)

        stream = utils.simulate_stream(high_price, timestamp)
        broker.next(stream)

        stream = utils.simulate_stream(low_price, timestamp)
        broker.next(stream)

        stream = utils.simulate_stream(close_price, timestamp)
        broker.next(stream)

    setups = strategy.setups
    realized_rr = sum(setup.extra.realized_rr for setup in setups)
    logger.debug("total Realized RR", realized_rr=realized_rr)
    logger.debug("Total setups", total=len(setups))
    for i in setups:
        logger.debug("Setup", realized_rr=i.extra.realized_rr, datetime=i.timestamp)

    trades = [asdict(trade) for trade in setups]
    df = pd.json_normalize(trades)
    return df


class Live:
    def __init__(self) -> None:
        logger.debug("Running live")
        require_live_trading_enabled(
            is_live=settings.is_live,
            live_confirm=settings.live_confirm,
            operation="tribar_momentum.Live.__init__",
        )
        self.contract = ib_async.Future(symbol="MGC", exchange="COMEX", localSymbol="MGCM5")
        self.tick_size = 0.1

        self.repo = repository.Repository()
        self.param = Param(symbol=self.contract, tick_size=self.tick_size, tick_value=1)
        self.extra = Extra()
        self.var = Var()

        ib_async.util.startLoop()
        self.broker = brokers.IBKRBroker(self.repo, port=4001, account=settings.ib_account)
        self.broker.connect()

    def run(self):
        self.strategy = TribarMomentumStrategy(
            broker=self.broker,
            param=self.param,
            var=self.var,
            extra=self.extra,
        )
        self.strategy.pull_historical_data()

        # self.broker.ib.reqTickByTickData(self.contract, "BidAsk", numberOfTicks=1)
        # self.broker.ib.pendingTickersEvent += self.on_event
        # self.broker.ib.run()

    def on_event(self, event):
        try:
            ticker = event.pop()
            ticker.time = ticker.time.astimezone(pytz.timezone(self.param.tz))
            stream = Stream(
                bid=ticker.bid,
                ask=ticker.ask,
                open=ticker.ask,
                high=ticker.ask,
                low=ticker.ask,
                close=ticker.ask,
                timestamp=ticker.time,
                sma200=self.strategy.var.previous_first_bar.sma200,
                ema9=self.strategy.var.previous_first_bar.ema9,
            )
            self.strategy.next(stream)
        except Exception as e:
            logger.exception(e)


class Paper:
    def __init__(self) -> None:
        logger.debug("Running paper")
        self.contract = ib_async.Future(symbol="MGC", exchange="COMEX", localSymbol="MGCM5")
        self.tick_size = 0.1

        self.repo = repository.Repository()
        self.param = Param(symbol=self.contract, tick_size=self.tick_size, tick_value=1)
        self.extra = Extra()
        self.var = Var()

        ib_async.util.startLoop()
        self.broker = brokers.IBKRBroker(self.repo, port=4002)
        self.broker.connect()

    def run(self):
        self.strategy = TribarMomentumStrategy(
            broker=self.broker,
            param=self.param,
            var=self.var,
            extra=self.extra,
        )
        self.strategy.pull_historical_data()

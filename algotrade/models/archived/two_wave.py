"""
52% chance of having two consecutive bars in the same direction
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
from algotrade import constants as const
from algotrade.core import brokers, engine, nz, repository, types
from algotrade.core.utils import adjust_daily_dates, simulate_stream
from structlog import get_logger

logger = get_logger()


@dataclass
class TwoWaveData:
    price_17_00: float = 0.0
    price_17_00_dt: datetime = None

    f_signal: bool = False
    f_signal_dt: datetime = None
    f_signal_price: float = 0.0
    f_signal_dir: types.Action = None
    f_signal_dst: float = 0.0
    f_signal_dst_price: float = 0.0
    f_signal_dst_dt: datetime = None
    f_signal_dst_max: float = 0.0
    f_signal_dst_max_price: float = 0.0
    f_signal_dst_max_dt: datetime = None

    s_signal: bool = False
    s_signal_dt: datetime = None
    s_signal_price: float = 0.0
    s_signal_dir: types.Action = None
    s_signal_dst: float = 0.0
    s_signal_dst_price: float = 0.0
    s_signal_dst_dt: datetime = None
    s_signal_dst_max: float = 0.0
    s_signal_dst_max_price: float = 0.0
    s_signal_dst_max_dt: datetime = None

    trigger_entry: bool = False
    trigger_entry_dt: datetime = None

    drawdown: float = 0.0
    drawdown_price: float = 0.0
    drawdown_dt: datetime = None

    stoploss: float = 0.0


@dataclass
class TwoWaveParams:
    symbol: ib_async.Contract | str
    rolling_window: int

    f_signal_ticks: int = None
    s_signal_ticks: int = None
    tick_stoploss: int = None
    open_trade: bool = False
    missed_trade: bool = False
    has_trade_today: bool = False
    end_of_day_dt: datetime = None
    tz: str = const.cme_tz
    has_time_closed: bool = False


class TwoWaveStrategy(engine.BaseEngine):
    """
    WARNINGS:
    - The strategy used US/Centeral timezone, but when you submit an order
    you have to change the timezone base on the IB Gateway timezone.
    """

    def __init__(
        self,
        broker: brokers.BacktestingBroker | brokers.IBKRBroker,
        param: TwoWaveParams,
        extra: TwoWaveData,
    ):
        super().__init__(broker)

        self.broker = broker
        # Basics
        self.open_time = time(17, 0)
        self.close_time = time(15, 59)
        self.close_time_1h = time(15, 0)

        self.previous_stream = None

        self.default_data: TwoWaveData = extra
        self.d: TwoWaveData = deepcopy(self.default_data)

        self.param_default: TwoWaveParams = param
        self.param: TwoWaveParams = deepcopy(self.param_default)

        self.tick_size = 1 / 64
        self.fatal = False

        self.seed_data = []
        self.current_stream = {"dt": None, "open": None, "high": None, "low": None, "close": None}
        self.current_seed_days = 0
        self.current_OHLC_part = 0  # 0: open, 1: high, 2: low, 3: close

    def indicator(self):
        # dt, a_dt, a_dir, a_open, a_high, a_low, a_close
        df = pd.DataFrame(self.seed_data)
        df["dt"] = pd.to_datetime(df["dt"])
        df = adjust_daily_dates(df)
        df.dropna(inplace=True)

        logger.debug(f"self.seed_data: {len(df.index)}")

        # rows should be equal or greater than window
        group = df.groupby("a_dt")
        logger.debug(f"group: {len(group)}")

        if len(group) < self.param.rolling_window:
            raise ValueError("The number of rows should be equal or greater than the rolling window size")

        # print(df)
        # import pdb

        # pdb.set_trace()
        # print(df)
        data = []

        d_f_ticks = None
        s_lowest_touch = False
        s_lowest_touch_price = None
        d_drawdown = None
        d_drawdown_price = None
        d_drawdown_dt = None
        has_touch_open = False

        rows = df.to_dict(orient="records")
        rows_len = len(rows)
        for i, row in enumerate(rows):
            ticks = None
            if row["a_dir"] == "LONG":
                if row["high"] > row["a_open"]:
                    ticks = nz.decimal_ticks_diff(row["high"], row["a_open"])

                if not d_f_ticks:
                    d_f_ticks = ticks

                if ticks and ticks > d_f_ticks and not s_lowest_touch:
                    d_f_ticks = ticks

                if row["a_low"] == row["low"]:
                    s_lowest_touch = True
                    s_lowest_touch_price = row["low"]

                if s_lowest_touch and not has_touch_open:
                    if row["high"] >= row["a_open"]:
                        has_touch_open = True

                if has_touch_open and row["a_open"] > row["low"]:
                    # find the drawdown from a_open
                    drawdown = abs(nz.decimal_ticks_diff(row["low"], row["a_open"]))
                    if not d_drawdown:
                        d_drawdown = drawdown
                        d_drawdown_price = row["low"]
                        d_drawdown_dt = row["dt"]

                    if drawdown > d_drawdown:
                        d_drawdown = drawdown
                        d_drawdown_price = row["low"]
                        d_drawdown_dt = row["dt"]

                if i + 1 == rows_len or rows[i + 1]["a_dt"] != row["a_dt"]:
                    s = {
                        "a_dt": row["a_dt"],
                        "a_dir": row["a_dir"],
                        "f_ticks": d_f_ticks,
                        "s_ticks": nz.decimal_ticks_diff(row["a_low"], row["a_open"]),
                        # "s_lowest_touch": s_lowest_touch_price,
                        "dd": d_drawdown,
                        # "drawdown_price": d_drawdown_price,
                        # "a_open": row["a_open"],
                        # "a_high": row["a_high"],
                        # "a_low": row["a_low"],
                        # "a_close": row["a_close"],
                        "gain": nz.decimal_ticks_diff(row["a_close"], row["a_open"]),
                    }
                    data.append(s)

                    d_f_ticks = None
                    s_lowest_touch = False
                    d_drawdown = None
                    has_touch_open = False

            elif row["a_dir"] == "SHORT":
                if row["low"] < row["a_open"]:
                    ticks = nz.decimal_ticks_diff(row["a_open"], row["low"])

                if not d_f_ticks:
                    d_f_ticks = ticks

                if ticks and ticks > d_f_ticks and not s_lowest_touch:
                    d_f_ticks = ticks

                if row["a_high"] == row["high"]:
                    s_lowest_touch = True
                    s_lowest_touch_price = row["high"]

                if s_lowest_touch and not has_touch_open:
                    if row["low"] <= row["a_open"]:
                        has_touch_open = True

                if has_touch_open and row["a_open"] < row["high"]:
                    # find the drawdown from a_open
                    drawdown = abs(nz.decimal_ticks_diff(row["high"], row["a_open"]))
                    if not d_drawdown:
                        d_drawdown = drawdown
                        d_drawdown_price = row["high"]
                        d_drawdown_dt = row["dt"]

                    if drawdown > d_drawdown:
                        d_drawdown = drawdown
                        d_drawdown_price = row["high"]
                        d_drawdown_dt = row["dt"]

                if i + 1 == rows_len or rows[i + 1]["a_dt"] != row["a_dt"]:
                    s = {
                        "a_dt": row["a_dt"],
                        "a_dir": row["a_dir"],
                        "f_ticks": d_f_ticks,
                        "s_ticks": nz.decimal_ticks_diff(row["a_open"], row["high"]),
                        # "s_lowest_touch": s_lowest_touch_price,
                        "dd": d_drawdown,
                        # "drawdown_price": d_drawdown_price,
                        # "a_open": row["a_open"],
                        # "a_high": row["a_high"],
                        # "a_low": row["a_low"],
                        # "a_close": row["a_close"],
                        "gain": nz.decimal_ticks_diff(row["a_open"], row["a_close"]),
                    }
                    data.append(s)

                    d_f_ticks = None
                    s_lowest_touch = False
                    d_drawdown = None
                    has_touch_open = False

        df = pd.DataFrame(data)
        df.fillna(0, inplace=True)

        # Rolling window f_ticks and s_ticks median, mean, max, min, std
        f_ticks_rolling = df["f_ticks"].rolling(window=self.param.rolling_window)
        df["f_ticks_median"] = f_ticks_rolling.median()
        df["f_ticks_mean"] = f_ticks_rolling.mean()
        df["f_ticks_max"] = f_ticks_rolling.max()
        df["f_ticks_min"] = f_ticks_rolling.min()
        # df["f_ticks_std"] = f_ticks_rolling.std()

        s_ticks_rolling = df["s_ticks"].rolling(window=self.param.rolling_window)
        df["s_ticks_median"] = s_ticks_rolling.median()
        df["s_ticks_mean"] = s_ticks_rolling.mean()
        df["s_ticks_max"] = s_ticks_rolling.max()
        df["s_ticks_min"] = s_ticks_rolling.min()
        # df["s_ticks_std"] = s_ticks_rolling.std()

        dd_rolling = df["dd"].rolling(window=self.param.rolling_window)
        df["dd_median"] = dd_rolling.median()
        df["dd_mean"] = dd_rolling.mean()
        df["dd_max"] = dd_rolling.max()
        df["dd_min"] = dd_rolling.min()
        # df["dd_std"] = dd_rolling.std()

        # df = df[
        #     [
        #         "a_dt",
        #         "a_dir",
        #         "gain",
        #         "f_ticks_median",
        #         "s_ticks_median",
        #         "dd_median",
        #     ]
        # ]
        logger.debug(f"rolling df length: {len(df.index)}")
        try:
            f_ticks_median = df["f_ticks_median"].iloc[-1] + 1
            if f_ticks_median < 4:
                f_ticks_median = 4

            s_ticks_median = df["s_ticks_median"].iloc[-1]
            dd_median = df["dd_mean"].iloc[-1] + 2
            if dd_median >= 15:
                dd_median = 15

            return f_ticks_median, f_ticks_median, dd_median
        except IndexError:
            return None, None, None

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

    def prerequisite(self, stream: types.Stream, historical: bool = False):
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

        self.historical: bool = historical
        # print(stream.timestamp)

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

                # TODO: seed_data should be saved in the database

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

                if self.current_seed_days >= self.param.rolling_window:
                    d = self.indicator()
                    self.param.f_signal_ticks, self.param.s_signal_ticks, self.param.tick_stoploss = d

                # pprint(self.current_stream)
                self.current_seed_days += 1
                logger.debug(f"Seed days: {self.current_seed_days}")

            # Since the stream is OHLC
            if self.current_OHLC_part == 0:
                self.current_stream["dt"] = stream.timestamp
                self.current_stream["open"] = stream.bid
                self.current_OHLC_part += 1
            elif self.current_OHLC_part == 1:
                self.current_stream["high"] = stream.bid
                self.current_OHLC_part += 1
            elif self.current_OHLC_part == 2:
                self.current_stream["low"] = stream.bid
                self.current_OHLC_part += 1
            elif self.current_OHLC_part == 3:
                self.current_stream["close"] = stream.bid
                self.seed_data.append(self.current_stream)

                self.current_stream = {"dt": None, "open": None, "high": None, "low": None, "close": None}
                self.current_OHLC_part = 0

    def check_f_signal(self, stream: types.Stream):
        # f_signal_dst_max
        if self.d.f_signal and not self.param.open_trade and not self.d.s_signal:
            dst = 0
            dst_price = None
            if stream.ask > self.d.price_17_00:
                dst_price = stream.ask
            elif stream.bid < self.d.price_17_00:
                dst_price = stream.bid

            if dst_price is not None:
                dst = nz.decimal_ticks_diff(self.d.price_17_00, dst_price)

            if dst > self.d.f_signal_dst_max:
                self.d.f_signal_dst_max = dst
                self.d.f_signal_dst_max_price = dst_price
                self.d.f_signal_dst_max_dt = stream.timestamp

        if self.d.f_signal:
            return

        dst_price = None
        if stream.ask > self.d.price_17_00:
            self.d.f_signal_dir = types.Action.LONG
            dst_price = stream.ask
        elif stream.bid < self.d.price_17_00:
            self.d.f_signal_dir = types.Action.SHORT
            dst_price = stream.bid

        if dst_price is not None:
            self.d.f_signal_dst = nz.decimal_ticks_diff(self.d.price_17_00, dst_price)
            self.d.f_signal_dst_price = dst_price
            self.d.f_signal_dst_dt = stream.timestamp

        if self.d.f_signal_dst >= self.param.f_signal_ticks:
            self.d.f_signal = True
            self.d.f_signal_price = dst_price
            self.d.f_signal_dt = stream.timestamp

            self.d.f_signal_dst_max = self.d.f_signal_dst
            self.d.f_signal_dst_max_price = dst_price
            self.d.f_signal_dst_max_dt = stream.timestamp

            logger.debug("f_signal triggered", data=asdict(self.d))

    def check_s_signal(self, stream: types.Stream):
        # s_signal_dst_max
        if self.d.f_signal and self.d.s_signal and not self.param.open_trade:
            dst = 0
            dst_price = None

            if self.d.s_signal_dir == types.Action.SHORT and stream.ask < self.d.f_signal_price:
                dst_price = stream.ask
            elif self.d.s_signal_dir == types.Action.LONG and stream.bid > self.d.f_signal_price:
                dst_price = stream.bid

            if dst_price is not None:
                dst = nz.decimal_ticks_diff(self.d.price_17_00, dst_price)

            if dst > self.d.s_signal_dst_max:
                self.d.s_signal_dst_max = dst
                self.d.s_signal_dst_max_price = dst_price
                self.d.s_signal_dst_max_dt = stream.timestamp

        if self.d.s_signal:
            return

        dst_price = None
        if self.d.f_signal_dir == types.Action.LONG and stream.ask < self.d.price_17_00:
            dst_price = stream.ask
        elif self.d.f_signal_dir == types.Action.SHORT and stream.bid > self.d.price_17_00:
            dst_price = stream.bid

        if dst_price is not None:
            self.d.s_signal_dst = nz.decimal_ticks_diff(self.d.price_17_00, dst_price)
            self.d.s_signal_dst_price = dst_price
            self.d.s_signal_dst_dt = stream.timestamp

        if self.d.s_signal_dst >= self.param.s_signal_ticks:
            self.d.s_signal = True
            self.d.s_signal_price = dst_price
            self.d.s_signal_dt = stream.timestamp

            if self.d.f_signal_dir == types.Action.LONG:
                self.d.s_signal_dir = types.Action.SHORT
            elif self.d.f_signal_dir == types.Action.SHORT:
                self.d.s_signal_dir = types.Action.LONG

            self.d.s_signal_dst_max = self.d.s_signal_dst
            self.d.s_signal_dst_max_price = dst_price
            self.d.s_signal_dst_max_dt = stream.timestamp

            logger.debug("s_signal triggered", data=asdict(self.d))

    def check_drawdown(self, stream: types.Stream):
        drawdown = 0
        drawdown_price = None
        if self.d.f_signal_dir == types.Action.LONG and stream.ask < self.d.price_17_00:
            drawdown_price = stream.ask
        elif self.d.f_signal_dir == types.Action.SHORT and stream.bid > self.d.price_17_00:
            drawdown_price = stream.bid

        if drawdown_price is not None:
            drawdown = nz.decimal_ticks_diff(self.d.price_17_00, drawdown_price)

        if drawdown > self.d.drawdown:
            self.d.drawdown = drawdown
            self.d.drawdown_price = drawdown_price
            self.d.drawdown_dt = stream.timestamp

    def next(self, stream: types.Stream, historical: bool = False):
        """ "
        If SHORT, bid; if LONG, ask
        """
        self.prerequisite(stream, historical)
        self.previous_stream = stream

        # It's important it's written after the previous block
        if self.param.missed_trade or self.param.has_trade_today or self.d.price_17_00 == 0.0:
            return

        if self.param.f_signal_ticks is None or self.param.s_signal_ticks is None or self.param.tick_stoploss is None:
            return

        if not self.param.open_trade:
            self.check_f_signal(stream)
            self.check_s_signal(stream)
            self.entry(stream)
        else:
            self.check_drawdown(stream)
            self.exit(stream)

    def entry(self, stream: types.Stream):
        if not self.d.s_signal or not self.d.f_signal:
            return

        if self.d.f_signal_dir == types.Action.LONG:
            self.param.open_trade = True
            self.d.stoploss = nz.decimal_subtraction_by_ticks(self.d.price_17_00, self.param.tick_stoploss)

            if self.historical:
                logger.debug("missed trade", data=self.d)
                self.param.missed_trade = True
                return

            logger.debug(f"LONG: {stream.timestamp}")

            parent = types.LimitOrder(
                symbol=self.param.symbol,
                quantity=1,
                action=types.Action.LONG,
                price=self.d.price_17_00,
                timestamp=stream.timestamp,
                extra=self.d,
                params=self.param,
            )
            stoploss = types.StopOrder(
                symbol=self.param.symbol,
                quantity=1,
                action=types.Action.SHORT,
                timestamp=stream.timestamp,
                price=self.d.stoploss,
                extra=self.d,
                params=self.param,
            )
            bracket = types.BracketOrder(symbol=self.param.symbol, order=parent, stop_loss=stoploss, take_profit=None)
            self.broker.order(bracket)

        elif self.d.f_signal_dir == types.Action.SHORT:
            self.param.open_trade = True
            self.d.stoploss = nz.decimal_addition_by_ticks(self.d.price_17_00, self.param.tick_stoploss)

            if self.historical:
                logger.debug("missed trade", data=asdict(self.d))
                self.param.missed_trade = True
                return

            logger.debug(f"SHORT: {stream.timestamp}")
            parent = types.LimitOrder(
                symbol=self.param.symbol,
                quantity=1,
                action=types.Action.SHORT,
                price=self.d.price_17_00,
                timestamp=stream.timestamp,
                extra=self.d,
                params=self.param,
            )

            stoploss = types.StopOrder(
                symbol=self.param.symbol,
                quantity=1,
                action=types.Action.LONG,
                timestamp=stream.timestamp,
                price=self.d.stoploss,
                extra=self.d,
                params=self.param,
            )

            bracket = types.BracketOrder(symbol=self.param.symbol, order=parent, stop_loss=stoploss, take_profit=None)
            self.broker.order(bracket)

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

        self.current_seed_days = 0


def backtest() -> pd.DataFrame:
    # Stream data
    d = pd.read_csv("~/desktop/test.csv")
    d["dt"] = pd.to_datetime(d["dt"])
    d["a_dt"] = pd.to_datetime(d["a_dt"])

    # dt = pd.to_datetime("2024-04-12 15:00:00")
    # a_dt = pd.to_datetime("2024-06-14")
    # d = d[d["dt"] >= dt]

    # d.to_csv("~/desktop/1d.csv", index=False)
    logger.debug(f"Total rows: {len(d)}")

    # The first row has been cleaned up and it's the opening price of the day.
    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo)
    param = TwoWaveParams(symbol="ZN", rolling_window=5)
    two_wave_strategy = TwoWaveStrategy(broker, param, extra=TwoWaveData())

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
        self.param = TwoWaveParams(f_signal_ticks=4, s_signal_ticks=4, tick_stoploss=5, symbol=self.contract)
        self.extra = TwoWaveData()

    def run(self):
        self.broker = brokers.IBKRBroker(self.repo)
        self.two_wave_strategy = TwoWaveStrategy(
            broker=self.broker,
            param=self.param,
            extra=self.extra,
        )

        duration = "1 D"
        days_elaps_since_open = 0
        open_dt = self.two_wave_strategy.d.price_17_00_dt
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
            self.two_wave_strategy.next(stream, historical=True)

            stream = types.Stream(row.high, row.high, row.date)
            self.two_wave_strategy.next(stream, historical=True)

            stream = types.Stream(row.low, row.low, row.date)
            self.two_wave_strategy.next(stream, historical=True)

            stream = types.Stream(row.close, row.close, row.date)
            self.two_wave_strategy.next(stream, historical=True)

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

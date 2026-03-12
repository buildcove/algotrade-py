"""
This module aims to aid the execution instead of manually placing orders.
"""

import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

import ib_async
import numpy as np
import pandas as pd
import pandas_ta
import pytz
import retry
from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, engine, indicators, repository, types, utils
from algotrade.core.safety import require_live_trading_enabled
from algotrade.schedules import IB_TZ, TIMEFRAME_TO_TV_INTERVAL
from algotrade.settings import settings
from algotrade.tv import get_token as get_tv_auth
from structlog import get_logger
from tvDatafeed import Interval, TvDatafeed

logger = get_logger()


class Variation(str, Enum):
    ABOVE_200SMA_LONG = "ABOVE_200SMA_LONG"
    ABOVE_200SMA_REVERSAL_SHORT = "ABOVE_200SMA_REVERSAL_SHORT"  # Above 200SMA, enough room to SMA200

    BELOW_200SMA_SHORT = "BELOW_200SMA_SHORT"
    BELOW_200SMA_REVERSAL_LONG = "BELOW_200SMA_REVERSAL_LONG"  # Below 200SMA, enough room to SMA200


class TargetType(str, Enum):
    ATR = "ATR"
    ATR_MULTIPLE = "ATR_MULTIPLE"
    RR = "RR"


@dataclass
class Candle(types.Candle):
    custom_high: float = None  # max(open, close)
    custom_low: float = None
    sma200: float = None
    ema9: float = None
    adx: float = None
    atr: float = None
    weekly_open: float = None
    weekly_high: float = None
    weekly_low: float = None
    weekly_atr: float = None
    week_key: str = None
    wicks_diff: float = None
    wicks_diff_sma14: float = None
    ext: float = None
    ext_sma14: float = None


@dataclass
class Stream(types.Stream):
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    custom_high: float = None  # max(open, close)
    custom_low: float = None
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


class AccountRisk(str, Enum):  # inherit from `str`
    ONE_PERCENT = "ONE_PERCENT"
    TWO_PERCENT = "TWO_PERCENT"

    @property
    def pct(self) -> float:  # handy numeric form
        return 0.01 if self is AccountRisk.ONE_PERCENT else 0.02


class RiskProfile(str, Enum):  # inherit from `str`
    ROUND = "ROUND"
    LEAST = "LEAST"


@dataclass
class Param:
    contract: types.ContractSpec
    tz: str = IB_TZ
    allowed_setups: List[types.SupportedContract] = field(default_factory=list)
    not_allowed_setups: List[types.SupportedContract] = field(default_factory=list)
    max_trade_per_week: int = 2
    minimum_risk_reward: float = 1.2
    risk_reward: float = 3.0
    account_risk: AccountRisk = AccountRisk.ONE_PERCENT  # 1% of the account risk per trade

    capital: float = 10000.0
    min_quantity: int = 1
    max_quantity: int = 1
    risk_profile: RiskProfile = RiskProfile.ROUND

    target_atr: Literal["full", "third"] = "full"  # full or third
    target_type: TargetType = TargetType.ATR_MULTIPLE
    target_atr_multiple: float = 3

    timeframe: types.Timeframe = types.Timeframe.h4

    gt: Optional[float] = None
    gte: Optional[float] = None
    lt: Optional[float] = None
    lte: Optional[float] = None

    gt_target: Optional[float] = None
    gte_target: Optional[float] = None
    lt_target: Optional[float] = None
    lte_target: Optional[float] = None

    long_gt: Optional[float] = None
    long_gte: Optional[float] = None
    short_gt: Optional[float] = None
    short_gte: Optional[float] = None

    target_bar_dt: Optional[datetime] = None
    last_bar_as_signal: bool = False

    with_crossing_sma: bool = False
    enable_wicks: bool = False


@dataclass
class Var:
    open_trade: bool = False
    missed_trade: bool = False

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


class DataFetchError(Exception):
    pass


class TribarExecutor(engine.BaseEngine):
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

        self.precheck_failed = False
        self.tv_instance = TvDatafeed()

    def get_tv(self):
        if utils.is_jwt_expired_no_sig(self.tv_instance.token):
            self.tv_instance.token = get_tv_auth(settings.tradingview_sessionid)

        logger.debug(
            "TradingView token ready",
            token_present=bool(self.tv_instance.token),
            token_expired=utils.is_jwt_expired_no_sig(self.tv_instance.token) if self.tv_instance.token else None,
        )
        return self.tv_instance

    def check_last_bar(self):
        tv = self.get_tv()

        data = tv.get_hist(
            symbol=self.param.contract.symbol,
            exchange=self.param.contract.exchange,
            interval=TIMEFRAME_TO_TV_INTERVAL[self.param.timeframe],
            n_bars=1,
            fut_contract=1,
            extended_session=True,
        )
        print(data)
        logger.warning("Please check if trading view is using live data")

    @retry.retry(exceptions=DataFetchError, tries=10, max_delay=10, delay=2, backoff=5)
    def pull_historical_data(self):
        start = time.time()
        tv = self.get_tv()
        # n_bars = 200 + (31 * 2)  # 200 SMA + 1 week (28.75 bars)
        n_bars = 700
        interval = TIMEFRAME_TO_TV_INTERVAL[self.param.timeframe]
        main_tf = tv.get_hist(
            symbol=self.param.contract.symbol,
            exchange=self.param.contract.exchange,
            interval=interval,
            n_bars=n_bars,
            fut_contract=1,
            extended_session=True,
        )

        if main_tf.empty:
            raise DataFetchError("No data fetched from TradingView")

        main_tf.index = main_tf.index.tz_localize("UTC")
        logger.debug("Data fetch time:", time=time.time() - start, timeframe=interval)

        df = self.get_df(main_tf)

        start = time.time()
        df.reset_index(inplace=True)

        print(df.tail())

        def _create_candle(row):
            return Candle(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                custom_high=row["custom_high"],
                custom_low=row["custom_low"],
                timestamp=row["datetime"],
                sma200=row["200sma"],
                ema9=row["9ema"],
                adx=row["adx"],
                atr=row["atr"],
                weekly_open=row.get("wk_open"),
                weekly_high=row.get("wk_high_to_date"),
                weekly_low=row.get("wk_low_to_date"),
                weekly_atr=row.get("wk_atr"),
                week_key=row.get("week_start"),
                wicks_diff=row.get("wicks_diff"),
                wicks_diff_sma14=row.get("wicks_diff_sma14"),
                ext=row.get("ext"),
                ext_sma14=row.get("ext_sma14"),
            )

        last_row_candle = df.iloc[-1]
        last_row_candle = _create_candle(last_row_candle)
        # If the last bar doesn't matches the target bar datetime, it means the next bar doesn't exist yet.
        # We need to fetch the data again until the target bar datetime matches the last bar.
        if (
            self.param.target_bar_dt
            and self.param.target_bar_dt != last_row_candle.timestamp
            and not self.param.last_bar_as_signal
        ):
            logger.debug(
                "Target bar datetime does not match the last bar. Fetching data again",
                target=self.param.target_bar_dt,
                last=last_row_candle.timestamp,
            )
            raise DataFetchError("Target bar datetime does not match the last bar.")

        for index, row in df.iterrows():
            stream = _create_candle(row)

            # WARNING: the last bar is the current bar. It will be use to determine an entry.
            if index == df.shape[0] - 1:
                logger.debug("Last bar as live", candle=asdict(stream))
                self.next(stream, historical=False)
                continue

            self.next(stream, historical=True)

        # WARNING: this is a workaround to treat the last bar as signal bar
        # Useful if you want to do a dry run or when you execute the trade when the market is closed.
        if self.param.last_bar_as_signal:
            logger.debug("Treating last bar as signal bar", candle=asdict(stream))
            self.next(stream)

        logger.debug("Historical data processing time:", time=time.time() - start)
        self.last_historical_pull = datetime.now(pytz.timezone(self.param.tz))

    def get_df(self, df: pd.DataFrame) -> pd.DataFrame:
        timeframe = self.param.timeframe

        pd.set_option("display.max_columns", None)  # None means "no limit"
        pd.set_option("display.width", None)  # prevents line‑wrapping
        start = time.time()

        df = df.sort_index()
        df["custom_high"] = df[["open", "close"]].max(axis=1)
        df["custom_low"] = df[["open", "close"]].min(axis=1)
        df["200sma"] = pandas_ta.sma(df["close"], length=200)
        df["9ema"] = pandas_ta.ema(df["close"], length=9)
        # h4["adx"] = indicators.adx(h4)
        df["adx"] = pandas_ta.adx(df["high"], df["low"], df["close"], length=14, mamode="rma")["ADX_14"]
        df["atr"] = pandas_ta.atr(df["high"], df["low"], df["close"], length=14, mamode="rma")

        # Calculate wicks difference
        # Green bars (close > open): open - low
        # Red bars (close < open): high - open
        df["is_green"] = df["close"] > df["open"]
        df["wicks_diff"] = np.where(df["is_green"], df["open"] - df["low"], df["high"] - df["open"])
        df["wicks_diff_sma14"] = pandas_ta.sma(df["wicks_diff"], length=14)

        # Calculate average extension
        # Green bars: high - open
        # Red bars: open - low
        df["ext"] = np.where(df["is_green"], df["high"] - df["open"], df["open"] - df["low"])
        df["ext_sma14"] = pandas_ta.sma(df["ext"], length=14)

        # Drop the temporary helper column
        # df = df.drop(columns=["is_green"])

        df = df.dropna()

        def choose_anchor(ts):
            ANCHOR_CDT = pd.Timestamp("1970-01-04 22:00:00", tz="UTC")
            ANCHOR_CST = pd.Timestamp("1970-01-04 23:00:00", tz="UTC")
            return ANCHOR_CDT if ts.tz_convert("America/Chicago").dst() else ANCHOR_CST

        anchor_vec = df.index.map(choose_anchor)
        week_number = ((df.index - anchor_vec) // pd.Timedelta(weeks=1)).astype("int64")
        df["week_start"] = anchor_vec + week_number * pd.Timedelta(weeks=1)
        df["wk_open"] = df.groupby("week_start")["open"].transform("first")
        df["wk_high_to_date"] = df.groupby("week_start")["high"].cummax()
        df["wk_low_to_date"] = df.groupby("week_start")["low"].cummin()

        if timeframe == types.Timeframe.h4:
            weekly = (
                df.assign(idx=df.index - df["week_start"])  # helper to keep order
                .groupby("week_start", sort=True)
                .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
            )
            weekly["atr"] = pandas_ta.atr(weekly["high"], weekly["low"], weekly["close"], length=14, mamode="EMA")
            # weekly["atr"] = indicators.atr(weekly, length=14)
            weekly["wk_atr"] = weekly["atr"].shift(1)  # ← previous week’s ATR
            weekly.dropna(inplace=True)
            # print("---weekly @ TribarExecutor")
            # print(weekly)

            df = df.merge(weekly["wk_atr"], left_on="week_start", right_index=True, how="left")

        df.dropna(inplace=True)
        logger.debug("Indicator calculation time:", time=time.time() - start)

        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

        return df

    def get_quantity(self, stoploss_ticks: int) -> int:
        # 1% of the account size
        risk_per_contract = stoploss_ticks * self.param.contract.tick_value
        if risk_per_contract <= 0:
            raise ValueError("Stop loss must be greater than 0.")

        risk_amount = self.param.capital * self.param.account_risk.pct
        raw_quantity = risk_amount / risk_per_contract

        if self.param.risk_profile == RiskProfile.ROUND:
            quantity = round(raw_quantity)  # Warning: +- 0.5
        else:  # LEAST
            quantity = int(raw_quantity)  # Floor operation, no rounding up

        quantity = max(self.param.min_quantity, min(quantity, self.param.max_quantity))
        # logger.debug("Quantity", quantity=quantity, risk_amount=risk_amount, risk_per_contract=risk_per_contract)
        return quantity

    def run_precheck(self):
        existing_position = self.broker.get_positions(self.param.contract.ib)
        if existing_position:
            logger.error("Existing position found. Exiting strategy.")
            self.precheck_failed = True
            raise ValueError("Existing position found. Exiting strategy.")

        existing_orders = self.broker.get_orders(self.param.contract.ib)
        if existing_orders:
            logger.error("Existing orders found. Exiting strategy.")
            self.precheck_failed = True
            raise ValueError("Existing orders found. Exiting strategy.")

        expiry = self.broker.get_contract_expiration(self.param.contract.ib)
        logger.warning("Contract expires on", expiry=expiry.strftime("%B"), symbol=self.param.contract.ib.localSymbol)

    def tribar_ready(self, stream: Candle) -> bool:
        if self.fatal:
            raise Exception("Fatal is true. Exiting the strategy.")

        if not isinstance(stream, Candle):
            raise TypeError("Stream must be of type Candle.")

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

    def create_bracket(self, action: types.Action, stream: Candle) -> types.BracketOrder:
        remaining_points = 0.0
        third_remaining_points = 0.0
        if self.param.timeframe == types.Timeframe.h4:
            remaining_points = self.get_full_remaining_points()
            third_remaining_points = remaining_points * 0.75 if remaining_points > 0 else 0
            remaining_points = remaining_points if self.param.target_atr == "full" else third_remaining_points

        # if (
        #     self.param.target_type == TargetType.ATR
        #     and remaining_points <= 0
        #     and self.extra.variation in (Variation.ABOVE_200SMA_LONG, Variation.BELOW_200SMA_SHORT)
        # ):
        #     logger.warning(
        #         "skipping: remaining points <= 0",
        #         remaining_points=remaining_points,
        #         candle=asdict(self.var.previous_first_bar),
        #     )
        #     return

        if action == types.Action.LONG:
            entry_price = self.var.previous_first_bar.close - (self.param.contract.tick_spread * self.param.contract.tick_size)
            entry_price = utils.round_to_tick(entry_price, self.param.contract.tick_size)

            # Original stop loss based on signal bar low
            signal_bar_stop_price = self.var.previous_first_bar.low

            # ATR-based stop loss
            atr_stop_price = entry_price - self.var.previous_first_bar.atr

            # Convert both to ticks and choose the lower one (closer to entry)
            signal_bar_stop_ticks = int((entry_price - signal_bar_stop_price) / self.param.contract.tick_size)
            atr_stop_ticks = int((entry_price - atr_stop_price) / self.param.contract.tick_size)

            # Choose the lower stop loss in ticks (smaller risk)
            if signal_bar_stop_ticks <= atr_stop_ticks:
                stop_price = signal_bar_stop_price
            else:
                stop_price = atr_stop_price

            stop_price = utils.round_to_tick(stop_price, self.param.contract.tick_size)

            stoploss_ticks = int((entry_price - stop_price) / self.param.contract.tick_size)
            quantity = self.get_quantity(stoploss_ticks)

            order = types.LimitOrder(
                symbol=self.param.contract.ib,
                quantity=quantity,
                price=entry_price,
                action=types.Action.LONG,
                timestamp=None,
            )

            stop_loss = types.StopOrder(
                symbol=self.param.contract.ib,
                price=stop_price,
                action=types.Action.SHORT,
                quantity=quantity,
                timestamp=None,
            )

            if self.param.target_type == TargetType.RR:
                take_profit_price = entry_price + (entry_price - stop_price) * self.param.risk_reward
            elif self.param.target_type == TargetType.ATR:
                take_profit_price = self.var.previous_first_bar.weekly_high + remaining_points
            elif self.param.target_type == TargetType.ATR_MULTIPLE:
                take_profit_price = self.var.previous_first_bar.close + (
                    self.var.previous_first_bar.atr * self.param.target_atr_multiple
                )
            else:
                raise ValueError("Invalid target type")

            take_profit_price = utils.round_to_tick(take_profit_price, self.param.contract.tick_size)

            take_profit = types.LimitOrder(
                symbol=self.param.contract.ib,
                quantity=quantity,
                price=take_profit_price,
                action=types.Action.SHORT,
                timestamp=None,
            )
            bracket = types.BracketOrder(
                symbol=self.param.contract.ib,
                order=order,
                stop_loss=stop_loss,
                take_profit=take_profit,
                extra=deepcopy(self.extra),
                params=deepcopy(self.param),
                var=deepcopy(self.var),
                timestamp=stream.timestamp,
            )
        elif action == types.Action.SHORT:
            entry_price = self.var.previous_first_bar.close + (self.param.contract.tick_spread * self.param.contract.tick_size)
            entry_price = utils.round_to_tick(entry_price, self.param.contract.tick_size)

            # Original stop loss based on signal bar high
            signal_bar_stop_price = self.var.previous_first_bar.high

            # ATR-based stop loss
            atr_stop_loss_price = entry_price + self.var.previous_first_bar.atr

            # Convert both to ticks and choose the lower one (closer to entry)
            signal_bar_stop_ticks = int((signal_bar_stop_price - entry_price) / self.param.contract.tick_size)
            atr_stop_ticks = int((atr_stop_loss_price - entry_price) / self.param.contract.tick_size)

            # Choose the lower stop loss in ticks (smaller risk)
            if signal_bar_stop_ticks <= atr_stop_ticks:
                stop_loss_price = signal_bar_stop_price
            else:
                stop_loss_price = atr_stop_loss_price

            stop_loss_price = utils.round_to_tick(stop_loss_price, self.param.contract.tick_size)

            stoploss_ticks = int((stop_loss_price - entry_price) / self.param.contract.tick_size)
            quantity = self.get_quantity(stoploss_ticks)

            order = types.LimitOrder(
                symbol=self.param.contract.ib, quantity=quantity, price=entry_price, action=types.Action.SHORT, timestamp=None
            )

            stop_loss = types.StopOrder(
                symbol=self.param.contract.ib,
                price=stop_loss_price,
                action=types.Action.LONG,
                quantity=quantity,
                timestamp=None,
            )

            if self.param.target_type == TargetType.RR:
                take_profit_price = entry_price - (stop_loss_price - entry_price) * self.param.risk_reward
            elif self.param.target_type == TargetType.ATR:
                take_profit_price = self.var.previous_first_bar.weekly_low - remaining_points
            elif self.param.target_type == TargetType.ATR_MULTIPLE:
                take_profit_price = self.var.previous_first_bar.close - (
                    self.var.previous_first_bar.atr * self.param.target_atr_multiple
                )
            else:
                raise ValueError("Invalid target type")

            take_profit_price = utils.round_to_tick(take_profit_price, self.param.contract.tick_size)

            take_profit = types.LimitOrder(
                symbol=self.param.contract.ib,
                quantity=quantity,
                price=take_profit_price,
                action=types.Action.LONG,
                timestamp=None,
            )

            bracket = types.BracketOrder(
                symbol=self.param.contract.ib,
                order=order,
                stop_loss=stop_loss,
                take_profit=take_profit,
                extra=deepcopy(self.extra),
                params=deepcopy(self.param),
                var=deepcopy(self.var),
                timestamp=stream.timestamp,
            )

        rr = utils.compute_rr(
            entry=bracket.order.price,
            stop_loss=bracket.stop_loss.price,
            exit=bracket.take_profit.price,
        )

        logger.debug(
            "Creating bracket order",
            rr=rr,
            entry=bracket.order.price,
            stop_loss=bracket.stop_loss.price,
            target=bracket.take_profit.price,
            setup=self.extra.variation.name,
            qty=bracket.order.quantity,
            atr=stream.atr,
            atr_product=stream.atr * self.param.target_atr_multiple,
        )
        logger.debug("candle", candle=asdict(self.var.previous_first_bar))
        if rr < self.param.minimum_risk_reward:
            logger.debug("skipped RR is less than minimum", rr=rr, min_rr=self.param.minimum_risk_reward)
            return None

        if True or stream.adx > 20 or self.var.previous_first_bar.adx > 20:
            self.setups.append(
                {
                    "date": self.var.previous_first_bar.timestamp,
                    "entry": bracket.order.price,
                    "target": bracket.take_profit.price,
                    "stop_loss": bracket.stop_loss.price,
                    "rr": rr,
                    "atr": stream.atr,
                    "w_atr": self.var.previous_first_bar.weekly_atr,
                    "adx": self.var.previous_first_bar.adx,
                    "c_adx": stream.adx,
                    "c_range": self.var.previous_first_bar.weekly_high - self.var.previous_first_bar.weekly_low,
                    "remaining_p": remaining_points,
                    "week_key": self.var.previous_first_bar.week_key,
                }
            )

        return bracket

    def next(self, stream: Candle, historical: bool = False):
        # logger.debug("Next", data=stream, historical=historical)

        if self.precheck_failed:
            logger.error("Precheck failed. Exiting")
            return

        if isinstance(stream, Candle):
            if not self.tribar_ready(stream):
                logger.debug("Waiting for tribar to be ready")
                return

        above_200_sma = self.var.previous_first_bar.close > self.var.previous_first_bar.sma200
        above_9_ema = self.var.previous_first_bar.close > self.var.previous_first_bar.ema9
        crossing_200_sma = (
            self.var.previous_first_bar.low <= self.var.previous_first_bar.sma200 <= self.var.previous_first_bar.high
        )

        if crossing_200_sma and self.param.with_crossing_sma:
            logger.debug("skipped Crossing 200 SMA", candle=asdict(self.var.previous_first_bar))
            return

        long_three_bar = (
            self.var.previous_first_bar.close > self.var.previous_first_bar.open
            and self.var.previous_first_bar.close > self.var.previous_second_bar.custom_high
            and self.var.previous_first_bar.close > self.var.previous_third_bar.custom_high
            # and self.var.previous_first_bar.close > self.var.previous_fourth_bar.custom_high
            and above_9_ema
        )
        short_three_bar = (
            self.var.previous_first_bar.close < self.var.previous_first_bar.open
            and self.var.previous_first_bar.close < self.var.previous_second_bar.custom_low
            and self.var.previous_first_bar.close < self.var.previous_third_bar.custom_low
            # and self.var.previous_first_bar.close < self.var.previous_fourth_bar.custom_low
            and not above_9_ema
        )
        if not (long_three_bar or short_three_bar):
            logger.debug("No signal", data=self.var.previous_first_bar)
            return

        if self.var.open_trade:
            logger.warning("Open trade already exists.")

        bracket = None
        if long_three_bar and above_9_ema and above_200_sma:
            self.extra.variation = Variation.ABOVE_200SMA_LONG
            bracket = self.create_bracket(types.Action.LONG, stream)
        elif short_three_bar and not above_200_sma and not above_9_ema:
            self.extra.variation = Variation.BELOW_200SMA_SHORT
            bracket = self.create_bracket(types.Action.SHORT, stream)
        elif short_three_bar and above_200_sma:
            self.extra.variation = Variation.ABOVE_200SMA_REVERSAL_SHORT
            bracket = self.create_bracket(types.Action.SHORT, stream)
        elif long_three_bar and not above_200_sma:
            self.extra.variation = Variation.BELOW_200SMA_REVERSAL_LONG
            bracket = self.create_bracket(types.Action.LONG, stream)

        if bracket and self.param.allowed_setups and self.extra.variation not in self.param.allowed_setups:
            logger.debug(
                "skipped not in Allowed setups", variation=self.extra.variation.name, candle=asdict(self.var.previous_first_bar)
            )
            return

        if bracket and self.param.not_allowed_setups and self.extra.variation in self.param.not_allowed_setups:
            logger.debug(
                "skipped in Not allowed setups", variation=self.extra.variation.name, candle=asdict(self.var.previous_first_bar)
            )
            return

        if self.param.gt and self.var.previous_first_bar.close <= self.param.gt:
            logger.debug(
                "skipped below or equal to gt",
                gt=self.param.gt,
                close=self.var.previous_first_bar.close,
                candle=asdict(self.var.previous_first_bar),
            )
            return
        if self.param.gte and self.var.previous_first_bar.close < self.param.gte:
            logger.debug(
                "skipped below gte",
                gte=self.param.gte,
                close=self.var.previous_first_bar.close,
                candle=asdict(self.var.previous_first_bar),
            )
            return
        if self.param.lt and self.var.previous_first_bar.close >= self.param.lt:
            logger.debug(
                "skipped above or equal to lt",
                lt=self.param.lt,
                close=self.var.previous_first_bar.close,
                candle=asdict(self.var.previous_first_bar),
            )
            return
        if self.param.lte and self.var.previous_first_bar.close > self.param.lte:
            logger.debug(
                "skipped above lte",
                lte=self.param.lte,
                close=self.var.previous_first_bar.close,
                candle=asdict(self.var.previous_first_bar),
            )
            return

        # Long/Short specific price validation using bracket
        if bracket:
            is_long_setup = bracket.order.action == types.Action.LONG
            is_short_setup = bracket.order.action == types.Action.SHORT

            if is_long_setup:
                if self.param.long_gt and self.var.previous_first_bar.close <= self.param.long_gt:
                    logger.debug(
                        "skipped long setup below or equal to long_gt",
                        long_gt=self.param.long_gt,
                        close=self.var.previous_first_bar.close,
                        candle=asdict(self.var.previous_first_bar),
                    )
                    return
                if self.param.long_gte and self.var.previous_first_bar.close < self.param.long_gte:
                    logger.debug(
                        "skipped long setup below long_gte",
                        long_gte=self.param.long_gte,
                        close=self.var.previous_first_bar.close,
                        candle=asdict(self.var.previous_first_bar),
                    )
                    return

            if is_short_setup:
                if self.param.short_gt and self.var.previous_first_bar.close <= self.param.short_gt:
                    logger.debug(
                        "skipped short setup below or equal to short_gt",
                        short_gt=self.param.short_gt,
                        close=self.var.previous_first_bar.close,
                        candle=asdict(self.var.previous_first_bar),
                    )
                    return
                if self.param.short_gte and self.var.previous_first_bar.close < self.param.short_gte:
                    logger.debug(
                        "skipped short setup below short_gte",
                        short_gte=self.param.short_gte,
                        close=self.var.previous_first_bar.close,
                        candle=asdict(self.var.previous_first_bar),
                    )
                    return

        # Target price validation
        if bracket:
            target_price = bracket.take_profit.price
            if self.param.gt_target and target_price <= self.param.gt_target:
                logger.debug(
                    "skipped target below or equal to gt_target",
                    gt_target=self.param.gt_target,
                    target=target_price,
                    candle=asdict(self.var.previous_first_bar),
                )
                return
            if self.param.gte_target and target_price < self.param.gte_target:
                logger.debug(
                    "skipped target below gte_target",
                    gte_target=self.param.gte_target,
                    target=target_price,
                    candle=asdict(self.var.previous_first_bar),
                )
                return
            if self.param.lt_target and target_price >= self.param.lt_target:
                logger.debug(
                    "skipped target above or equal to lt_target",
                    lt_target=self.param.lt_target,
                    target=target_price,
                    candle=asdict(self.var.previous_first_bar),
                )
                return
            if self.param.lte_target and target_price > self.param.lte_target:
                logger.debug(
                    "skipped target above lte_target",
                    lte_target=self.param.lte_target,
                    target=target_price,
                    candle=asdict(self.var.previous_first_bar),
                )
                return

        # if not (stream.adx > 20 or self.var.previous_first_bar.adx > 20):
        #     logger.debug("ADX is below 20. No signal", adx=stream.adx, candle=asdict(self.var.previous_first_bar))
        #     return

        # if (
        #     bracket
        #     and self.extra.variation == Variation.ABOVE_200SMA_REVERSAL_SHORT
        #     and stream.sma200 > bracket.take_profit.price
        # ):
        #     logger.debug(
        #         "skipped not enough to sma: ABOVE_200SMA_REVERSAL_SHORT", target=bracket.take_profit.price, sma200=stream.sma200
        #     )
        #     return
        # elif (
        #     bracket and self.extra.variation == Variation.BELOW_200SMA_REVERSAL_LONG and stream.sma200 < bracket.take_profit.price
        # ):
        #     logger.debug(
        #         "skipped not enough to sma: BELOW_200SMA_REVERSAL_LONG", target=bracket.take_profit.price, sma200=stream.sma200
        #     )
        #     return

        if bracket and not historical:
            logger.debug("Placing order", historical=historical)
            self.var.open_trade = True
            self.broker.order(bracket)

        if bracket:
            logger.debug("Valid Setup", setup=self.extra.variation.name, adx=stream.adx)

        # Warning: this should be at the end of the function
        if long_three_bar or short_three_bar:
            self.var.previous_long_three_bar = long_three_bar
            self.var.previous_short_three_bar = short_three_bar
            self.var.previous_above_200sma = above_200_sma
            self.var.previous_above_9ema = above_200_sma

    def get_third_remaining_points(self) -> float:
        third_atr = self.var.previous_first_bar.weekly_atr * 0.75
        current_range = self.var.previous_first_bar.weekly_high - self.var.previous_first_bar.weekly_low
        remaining_range = third_atr - current_range
        logger.debug(
            "get_third_remaining_points",
            third_atr=third_atr,
            current_range=current_range,
            remaining_range=remaining_range,
            atr=self.var.previous_first_bar.weekly_atr,
        )
        if remaining_range < 0:
            return 0

        return remaining_range

    def get_full_remaining_points(self) -> float:
        full_atr = self.var.previous_first_bar.weekly_atr
        current_range = self.var.previous_first_bar.weekly_high - self.var.previous_first_bar.weekly_low
        remaining_range = full_atr - current_range
        logger.debug(
            "full_remaining_points",
            full_atr=full_atr,
            current_range=current_range,
            remaining_range=remaining_range,
            atr=self.var.previous_first_bar.weekly_atr,
        )
        if remaining_range < 0:
            return 0

        return remaining_range

    def exit_end_of_week(self):
        # TODO: Implement this method
        raise NotImplementedError

    def get_max_points_via_weekly_diff(self, action: types.Action) -> float:
        if action == types.Action.LONG:
            weekly_diff = self.var.previous_first_bar.close - self.var.previous_first_bar.weekly_open
            max_potential_points = self.var.previous_first_bar.weekly_atr - weekly_diff
        elif action == types.Action.SHORT:
            weekly_diff = self.var.previous_first_bar.weekly_open - self.var.previous_first_bar.close
            max_potential_points = self.var.previous_first_bar.weekly_atr - weekly_diff

        if max_potential_points < 0:
            logger.warning("Deprecated: Max potential points is negative", max_potential_points=max_potential_points)
            return 0

        return max_potential_points


def backtest():
    main_tf = pd.read_csv("~/Desktop/mes.csv")
    main_tf.reset_index(inplace=True)
    main_tf = main_tf[["time", "open", "high", "low", "close"]]
    main_tf["time"] = pd.to_datetime(main_tf["time"], utc=True)
    main_tf = main_tf.rename(columns={"time": "datetime"})
    main_tf = main_tf[["datetime", "open", "high", "low", "close"]]
    main_tf = main_tf.set_index("datetime")
    print("---main_tf @ loader")
    print(main_tf)

    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo)
    contract = get_contract_spec(types.SupportedContract.MGC)
    param = Param(contract=contract, tz=IB_TZ, target_type=TargetType.ATR, enable_wicks=True)
    var = Var()
    extra = Extra()
    strategy = TribarExecutor(
        broker=broker,
        param=param,
        var=var,
        extra=extra,
    )

    combine_df = strategy.get_df(main_tf)
    print("---combine_df @ TribarExecutor")
    print(combine_df.tail(100))
    combine_df = combine_df.loc[pd.Timestamp("2025-06-01 00:00:00", tz="UTC") :]

    # temp_df = combine_df[["is_green", "open", "high", "low", "wicks_diff", "wicks_diff_sma14", "ext", "ext_sma14"]].tail()
    # print(temp_df)

    # return
    for index, row in combine_df.iterrows():
        open_price = row["open"]
        high_price = row["high"]
        low_price = row["low"]
        close_price = row["close"]
        timestamp = index

        stream = Candle(
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            custom_high=row["custom_high"],
            custom_low=row["custom_low"],
            timestamp=timestamp,
            sma200=row["200sma"],
            ema9=row["9ema"],
            adx=row["adx"],
            atr=row["atr"],
            weekly_atr=row["wk_atr"],
            weekly_open=row["wk_open"],
            weekly_high=row["wk_high_to_date"],
            weekly_low=row["wk_low_to_date"],
            week_key=row["week_start"],
            wicks_diff=row["wicks_diff"],
            wicks_diff_sma14=row["wicks_diff_sma14"],
            ext=row["average_ext"],
            ext_sma14=row["average_ext_sma14"],
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

    df = pd.DataFrame(strategy.setups)
    df["date"] = df["date"].dt.tz_convert("Asia/Manila")
    df["date"] = df["date"].dt.strftime("%b %d, %Y %-I:%M %p")
    pd.set_option("display.max_columns", None)  # None means "no limit"
    pd.set_option("display.width", None)  # prevents line‑wrapping
    pd.set_option("display.max_rows", None)  # None means "no limit"


def dry_run(
    use_broker=False,
    contract: types.SupportedContract = types.SupportedContract.ZM,
    timeframe: types.Timeframe = types.Timeframe.daily,
    allowed_setups=None,
    not_allowed_setups=None,
    capital: float = 10000.0,
    account_risk: AccountRisk = AccountRisk.ONE_PERCENT,
    max_quantity: int = 1,
    min_quantity: int = 1,
    risk_profile: RiskProfile = RiskProfile.ROUND,
    gt=None,
    gte=None,
    lt=None,
    lte=None,
    gt_target=None,
    gte_target=None,
    lt_target=None,
    lte_target=None,
    long_gt=None,
    long_gte=None,
    short_gt=None,
    short_gte=None,
    target_atr_mul=None,
    with_crossing_sma: bool = False,
    last_bar_as_signal: bool = False,
    ib_client_id: int = 1,
):
    contract_spec = get_contract_spec(contract)
    repo = repository.Repository()

    broker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)
    if use_broker:
        ib_async.util.startLoop()
        broker = brokers.IBKRBroker(repo, port=4002, account=settings.ib_account, ib_client_id=ib_client_id)
        broker.connect()

    param = Param(
        contract=contract_spec,
        tz=IB_TZ,
        timeframe=timeframe,
        allowed_setups=allowed_setups or [],
        not_allowed_setups=not_allowed_setups or [],
        capital=capital,
        account_risk=account_risk,
        max_quantity=max_quantity,
        min_quantity=min_quantity,
        risk_profile=risk_profile,
        gt=gt,
        gte=gte,
        lt=lt,
        lte=lte,
        gt_target=gt_target,
        gte_target=gte_target,
        lt_target=lt_target,
        lte_target=lte_target,
        long_gt=long_gt,
        long_gte=long_gte,
        short_gt=short_gt,
        short_gte=short_gte,
        with_crossing_sma=with_crossing_sma,
        last_bar_as_signal=last_bar_as_signal,
    )

    if target_atr_mul is not None:
        param.target_atr_multiple = target_atr_mul

    var = Var()
    extra = Extra()
    strategy = TribarExecutor(
        broker=broker,
        param=param,
        var=var,
        extra=extra,
    )
    strategy.pull_historical_data()
    if isinstance(broker, brokers.IBKRBroker):
        broker.disconnect()


class PositionPrices(TypedDict):
    entry_price: float
    stop_loss_price: float
    target_price: float
    action: types.Action


def move_stoploss_to_05_rr(broker: brokers.IBKRBroker) -> None:
    """
    Move a position’s stop-loss to ±0.5 R once price has travelled ≥ 0.75 ×
    (entry-to-target distance) in its favour.

    ─ Definitions ───────────────────────────────────────────────────────────
      entry      = fill price of the position
      target     = declared take-profit price
      initial SL = original stop-loss at entry
      R          = |entry − initial SL|            (risk in money / points)

    ─ LONG ──────────────────────────────────────────────────────────────────
      Trigger    : last_price ≥ entry + 0.75 × (target − entry)
      New SL     : entry + 0.5 R        (but never lower than current SL)

    ─ SHORT ─────────────────────────────────────────────────────────────────
      Trigger    : last_price ≤ entry − 0.75 × (entry − target)
      New SL     : entry − 0.5 R        (but never higher than current SL)
    """
    # ── Sanity checks ──────────────────────────────────────────────────────
    if not isinstance(broker, brokers.IBKRBroker):
        raise TypeError("broker must be an IBKRBroker instance")

    if not broker.ib.isConnected():
        raise ConnectionError("broker is not connected")

    positions: List[ib_async.Position] = broker.get_positions()
    if not positions:
        logger.debug("No open positions – nothing to trail.")
        return

    tv = utils.get_tv()  # your chart / price-feed helper

    # ── Walk through every open position ───────────────────────────────────
    for pos in positions:
        if pos.position == 0:  # just in case the broker returns flats
            continue

        contract: ib_async.Contract = pos.contract
        prices: PositionPrices = broker.get_position_prices(contract=contract)

        entry = prices["entry_price"]
        stop = prices["stop_loss_price"]
        target = prices["target_price"]  # must exist
        action = prices["action"]

        risk = abs(entry - stop)
        if risk <= 0:
            logger.warning("Risk is zero for %s – skipped.", contract.symbol)
            continue

        reward_dist = abs(target - entry)
        if reward_dist <= 0:
            logger.warning("Target not set / identical to entry for %s – skipped.", contract.symbol)
            continue

        # Grab the latest high/low (4 hour bar for a nice compromise)
        try:
            df: pd.DataFrame = tv.get_hist(
                symbol=contract.symbol,
                exchange=contract.exchange,
                interval=Interval.in_4_hour,
                n_bars=1,
                fut_contract=1,
                extended_session=True,
            )
            if df.empty:
                raise ValueError("empty dataframe")
        except Exception as exc:
            logger.error("Cannot fetch price for %s (%s).", contract.symbol, exc)
            continue

        high, low = df.iloc[-1][["high", "low"]]

        # ── LONG / SHORT trigger logic ─────────────────────────────────────
        if action == types.Action.LONG:
            trigger = entry + (0.75 * reward_dist)
            new_stop = entry + (0.5 * risk)
            moved = high >= trigger
            previous_stop_not_tightened = entry > stop  # ensure we *tighten* only once
            tighten = new_stop > stop and previous_stop_not_tightened
        elif action == types.Action.SHORT:
            trigger = entry - (0.75 * reward_dist)
            new_stop = entry - (0.5 * risk)
            moved = low <= trigger
            previous_stop_not_tightened = entry < stop  # ensure we *tighten* only once
            tighten = new_stop < stop and previous_stop_not_tightened
        else:
            logger.warning("Unknown action %s for %s – skipped.", action, contract.symbol)
            continue

        logger.debug(
            "data",
            position_type=action,
            contract=contract.symbol,
            entry=entry,
            stop=stop,
            target=target,
            new_stop=new_stop,
        )

        # ── Update if warranted ────────────────────────────────────────────
        if moved and tighten:
            logger.info(
                "[%s] %s hit 0.75×reward (trigger %.2f). Moving SL %.2f → %.2f",
                datetime.utcnow().isoformat(timespec="seconds"),
                contract.symbol,
                trigger,
                stop,
                new_stop,
            )
            broker.modify_position_stoploss(contract, new_stop)


class Live:

    def __init__(
        self,
        type_: Literal["live", "paper"],
        param: Param,
        ib_client_id: int = 1,
    ) -> None:
        logger.debug("Running", type_=type_, contract=param.contract.symbol, timeframe=param.timeframe)
        self.contract = param.contract
        self.param = param

        self.repo = repository.Repository()

        self.extra = Extra()
        self.var = Var()
        logger.debug("Param", param=self.param)
        logger.debug("Extra", extra=self.extra)
        logger.debug("Var", var=self.var)

        port = 4001 if type_ == "live" else 4002

        if type_ == "live":
            require_live_trading_enabled(
                is_live=settings.is_live,
                live_confirm=settings.live_confirm,
                operation="tribar_v2.Live.__init__",
            )

        self.broker = brokers.IBKRBroker(self.repo, port=port, account=settings.ib_account, ib_client_id=ib_client_id)
        self.broker.connect()
        logger.debug("Broker Settings", account=settings.ib_account, port=port, client_id=ib_client_id)
        self.strategy = TribarExecutor(
            broker=self.broker,
            param=self.param,
            var=self.var,
            extra=self.extra,
        )

    def check_tv_live(self):
        self.strategy.check_last_bar()

    def run_precheck(self):
        self.strategy.run_precheck()

    def run(self):
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

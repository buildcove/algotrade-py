"""
Variables:
- price_17_00
- f_wave_signal_ticks
- s_wave_signal_ticks
- tick_stoploss:

Strategy:

- Get opening price at 17:00
- Determine the entry direction based on the distance from the opening price using the tick_1_signal
- If the distance is greater than the tick_1_signal, then signal

best params: {'f_signal_ticks': 4, 's_signal_ticks': 4, 'tick_stoploss': 30}

convert this to a class
Result:
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pprint import pprint
from typing import Dict, List, Optional

import holidays
import pandas as pd
from algotrade.core import nz

nyse_holidays = holidays.NYSE()


day_open_time = time(17, 00)
day_close_time = time(15, 59)


class Type(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    a_d: date
    price_17_00_date: date
    price_17_00: float

    f_signal_dir: Optional[str]
    f_signal: bool
    f_signal_dst: float
    f_signal_dt: datetime
    f_signal_price: float

    s_signal_dir: str
    s_signal: bool
    s_signal_dst: float
    s_signal_dt: datetime
    s_signal_price: float

    open_dt: datetime
    open_price: float

    close_dt: datetime
    close_price: float

    stoploss: float
    takeprofit: Optional[float]
    type_: Type
    gain: float = 0.0


class Backtest:
    index: int = 0
    params: Dict = {}
    trades: List[Trade] = []

    def set_params(self, f_signal_ticks, s_signal_ticks, tick_stoploss):
        self.f_signal_ticks = f_signal_ticks
        self.s_signal_ticks = s_signal_ticks
        self.tick_stoploss = tick_stoploss

        self.params = {
            "f_signal_ticks": f_signal_ticks,
            "s_signal_ticks": s_signal_ticks,
            "tick_stoploss": tick_stoploss,
        }

    def init_basic_data(self):
        self._open_position = False

        self._type = ""
        self._open_dt = None
        self._open_price = 0.0

        self._close_dt = None
        self._close_price = 0.0
        self._stoploss = 0.0
        self._takeprofit = 0.0

        self.index = 0

        # This is the actual date based on trading view, it skipped weekends and holidays
        self.prev_real_date = None

    def init_strat_vars(self):
        self.price_17_00_date = None
        self.price_17_00 = None

        self.f_signal = False
        self.f_signal_dir = None
        self.f_signal_dst = 0.0
        self.f_signal_dt = None
        self.f_signal_price = 0.0

        self.s_signal = False
        self.s_signal_dir = None
        self.s_signal_dst = 0.0
        self.s_signal_dt = None
        self.s_signal_price = 0.0

        self.crossing_open = False

    def reset(self):
        self.trades = []
        self.init_basic_data()
        self.init_strat_vars()

    def close_position(self):
        self.init_basic_data()
        self.init_strat_vars()

    def prepare_data(self):
        df = pd.read_csv("~/dataset/zn-1m.csv", sep=";")
        df.columns = ["date", "time", "open", "high", "low", "close", "vol"]
        day = 1440
        year = day * 365
        quarter = int(year / 4)
        df = df.tail(day * 30)

        df["dt"] = df["date"] + " " + df["time"]
        df["dt"] = pd.to_datetime(df["dt"], dayfirst=True)

        df = df.sort_values(by=["dt"])
        df.drop(columns=["date", "time"], inplace=True)

        df = df.reset_index(drop=True)
        data = df.to_dict(orient="records")
        self.data = self.adjusted_date(data)

        df = pd.DataFrame(self.data)
        df.to_csv("~/Desktop/0_adjusted_dates.csv", index=False)

    def adjusted_date(self, records: List[Dict]):
        def _new(records):
            skipped_dates = []
            for record in records:
                if record["dt"].date() in nyse_holidays:
                    skipped_dates.append(record["dt"].date())

                if record["dt"].time() == day_open_time and record["dt"].date() not in nyse_holidays:
                    current_date = record["dt"].date() + timedelta(days=1)

                if current_date in nyse_holidays:
                    current_date += timedelta(days=1)

                record["a_d"] = current_date

            skipped_dates = set(skipped_dates)
            # for d in skipped_dates:
            # print(f"Skipping {d}")

            return records

        def _remove_recent_data_until_day_open_time(records):
            for record in records[:]:
                if record["dt"].time() == day_open_time:
                    break

                records.remove(record)

            return records

        records = _remove_recent_data_until_day_open_time(records)
        records = _new(records)
        return records

    def _debug(self, row):
        # print(f"{row['dt']}")
        # print(f"OHLC: {row['open']} | {row['high']} | {row['low']} | {row['close']}")
        # print(f"price_17_00: {self.price_17_00} | {self.price_17_00_date}")
        # print("")

        # print(f"f_signal: {self.f_signal} | {self.f_signal_dir} | {self.f_signal_dst}")
        # print(f"f_signal_dt: {self.f_signal_dt} | {self.f_signal_price}")

        # print(f"s_signal: {self.s_signal} | {self.s_signal_dst}")
        # print(f"s_signal_dt: {self.s_signal_dt} | {self.s_signal_price}")

        # print("")
        # print(f"pos: {self._open_position}")
        # print(f"open: {self._open_price} | {self._open_dt}")
        # print(f"close: {self._close_price} | {self._close_dt}")

        current = f"""
{row['dt']}
OHLC: {row['open']} | {row['high']} | {row['low']} | {row['close']}
price_17_00: {self.price_17_00} | {self.price_17_00_date}

f_signal: {self.f_signal} | {self.f_signal_dir} | {self.f_signal_dst}
f_signal_dt: {self.f_signal_dt} | {self.f_signal_price}
crossing: {self.crossing_open}

s_signal: {self.s_signal} | {self.s_signal_dst}
s_signal_dt: {self.s_signal_dt} | {self.s_signal_price}

pos: {self._open_position}
open: {self._open_price} | {self._open_dt}
close: {self._close_price} | {self._close_dt}
\n
"""
        # print if the following changes
        # price_17_00, price_17_00_date
        # f_signal, f_signal_dir, s_signal
        # open_position, open_price, close_price
        price_17_00 = None
        price_17_00_date = None
        f_signal = None
        f_signal_dir = None
        s_signal = None
        open_position = None
        open_price = None
        close_price = None

        if (
            self.price_17_00 != price_17_00
            or self.price_17_00_date != price_17_00_date
            or self.f_signal != f_signal
            or self.f_signal_dir != f_signal_dir
            or self.s_signal != s_signal
            or self._open_position != open_position
            or self._open_price != open_price
            or self._close_price != close_price
        ):
            print(current)
            price_17_00 = self.price_17_00
            price_17_00_date = self.price_17_00_date
            f_signal = self.f_signal
            f_signal_dir = self.f_signal_dir
            s_signal = self.s_signal
            open_position = self._open_position
            open_price = self._open_price
            close_price = self._close_price

    def run(self):
        # Get opening price at 17:00
        # print("len data", len(self.data))
        for i, row in enumerate(self.data):
            # It means it's the next daily candle
            if self.prev_real_date != row["a_d"] and row["dt"].time() == time(17, 0):
                self.init_strat_vars()
                self.price_17_00 = row["open"]
                self.price_17_00_date = row["dt"].date()
                self.prev_real_date = row["a_d"]

            if not self._open_position and self.price_17_00 is not None and not self.f_signal:
                if self.price_17_00 > row["high"]:
                    self.f_signal_dir = "down"
                    self.f_signal_dst = nz.decimal_ticks_diff(self.price_17_00, row["low"])

                elif self.price_17_00 < row["low"]:
                    self.f_signal_dir = "up"
                    self.f_signal_dst = nz.decimal_ticks_diff(self.price_17_00, row["high"])

                if self.f_signal_dst and self.f_signal_dst >= self.f_signal_ticks:
                    self.f_signal = True
                    self.f_signal_dt = row["dt"]

                    if self.f_signal_dir == "up":
                        self.f_signal_price = row["high"]
                    elif self.f_signal_dir == "down":
                        self.signal_price = row["low"]

            if self.f_signal and not self.s_signal:
                # Second wave, crossing the opening price
                if self.f_signal_dir == "up" and self.price_17_00 >= row["low"]:
                    self.s_signal_dst = nz.decimal_ticks_diff(self.price_17_00, row["low"])
                    self.crossing_open = True
                elif self.f_signal_dir == "down" and self.price_17_00 <= row["high"]:
                    self.s_signal_dst = nz.decimal_ticks_diff(self.price_17_00, row["high"])
                    self.crossing_open = True

                if self.s_signal_dst and self.s_signal_dst >= self.s_signal_ticks:
                    self.s_signal = True
                    self.s_signal_dt = row["dt"]

                    if self.f_signal_dir == "up":
                        self.s_signal_dir = "down"
                        self.s_signal_price = row["low"]
                    elif self.f_signal_dir == "down":
                        self.s_signal_dir = "up"
                        self.s_signal_price = row["high"]

            if self._open_position is False and self.s_signal:
                if self.f_signal_dir == "up" and row["high"] >= self.price_17_00:
                    self._type = Type.LONG
                    self._open_position = True
                    self._open_dt = row["dt"]
                    self._open_price = self.price_17_00
                    self._stoploss = nz.decimal_subtraction_by_ticks(self._open_price, self.tick_stoploss)

                elif self.f_signal_dir == "down" and row["low"] <= self.price_17_00:
                    self._type = Type.SHORT
                    self._open_position = True
                    self._open_dt = row["dt"]
                    self._open_price = self.price_17_00
                    self._stoploss = nz.decimal_addition_by_ticks(self._open_price, self.tick_stoploss)

            # Take profit when the time is 15:30am
            if row["dt"].time() == time(15, 59) and self._open_position:
                self._close_price = row["close"]
                gain = nz.decimal_ticks_diff(self._open_price, self._close_price)
                if self._type == Type.LONG and self._close_price <= self._open_price:
                    gain = -gain
                elif self._type == Type.SHORT and self._close_price >= self._open_price:
                    gain = -gain

                trade = Trade(
                    a_d=row["a_d"],
                    price_17_00_date=self.price_17_00_date,
                    price_17_00=self.price_17_00,
                    f_signal=self.f_signal,
                    f_signal_dt=self.f_signal_dt,
                    f_signal_price=self.f_signal_price,
                    f_signal_dir=self.f_signal_dir,
                    f_signal_dst=self.f_signal_dst,
                    s_signal=self.s_signal,
                    s_signal_dt=self.s_signal_dt,
                    s_signal_price=self.s_signal_price,
                    s_signal_dst=self.s_signal_dst,
                    s_signal_dir=self.s_signal_dir,
                    open_dt=self._open_dt,
                    open_price=self._open_price,
                    close_dt=row["dt"],
                    close_price=self._close_price,
                    stoploss=self._stoploss,
                    takeprofit=self._takeprofit,
                    type_=self._type,
                    gain=gain,
                )

                self.trades.append(trade)
                self.close_position()

            # Stoploss
            elif self._open_position:
                if self._type == Type.SHORT and row["high"] >= self._stoploss:
                    self._close_price = self._stoploss
                    gain = -abs(self.tick_stoploss)

                    trade = Trade(
                        a_d=row["a_d"],
                        price_17_00_date=self.price_17_00_date,
                        price_17_00=self.price_17_00,
                        f_signal=self.f_signal,
                        f_signal_dt=self.f_signal_dt,
                        f_signal_price=self.f_signal_price,
                        f_signal_dir=self.f_signal_dir,
                        f_signal_dst=self.f_signal_dst,
                        s_signal=self.s_signal,
                        s_signal_dt=self.s_signal_dt,
                        s_signal_price=self.s_signal_price,
                        s_signal_dst=self.s_signal_dst,
                        s_signal_dir=self.s_signal_dir,
                        open_dt=self._open_dt,
                        open_price=self._open_price,
                        close_dt=row["dt"],
                        close_price=self._close_price,
                        stoploss=self._stoploss,
                        takeprofit=self._takeprofit,
                        type_=self._type,
                        gain=gain,
                    )

                    self.trades.append(trade)
                    self.close_position()
                elif self._type == Type.LONG and row["low"] <= self._stoploss:
                    self._close_price = self._stoploss
                    gain = -abs(self.tick_stoploss)

                    trade = Trade(
                        a_d=row["a_d"],
                        price_17_00_date=self.price_17_00_date,
                        price_17_00=self.price_17_00,
                        f_signal=self.f_signal,
                        f_signal_dt=self.f_signal_dt,
                        f_signal_price=self.f_signal_price,
                        f_signal_dir=self.f_signal_dir,
                        f_signal_dst=self.f_signal_dst,
                        s_signal=self.s_signal,
                        s_signal_dt=self.s_signal_dt,
                        s_signal_price=self.s_signal_price,
                        s_signal_dst=self.s_signal_dst,
                        s_signal_dir=self.s_signal_dir,
                        open_dt=self._open_dt,
                        open_price=self._open_price,
                        close_dt=row["dt"],
                        close_price=self._close_price,
                        stoploss=self._stoploss,
                        takeprofit=self._takeprofit,
                        type_=self._type,
                        gain=gain,
                    )

                    self.trades.append(trade)
                    self.close_position()

            # self._debug(row)

        # remove last trade since it is not closed
        # self.trades.pop()

    def export(self):
        d = pd.json_normalize(asdict(obj) for obj in self.trades)
        d.to_csv("~/Desktop/trades.csv", index=False)

    def analyze(self):
        d = pd.json_normalize(asdict(obj) for obj in self.trades)
        if d.empty:
            return {
                "count_winner": 0,
                "count_loser": 0,
                "tick_gain_median": 0,
                "tick_gain": 0,
                "tick_loss": 0,
                "net_tick": 0,
                "dollar_gain": 0,
                "dollar_loss": 0,
                "commission": 0,
                "net_dollar": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "params": self.params,
            }

        # print(d)
        df = pd.DataFrame()
        df["Portfolio_Value"] = d["gain"].cumsum()

        # Calculate cumulative maximum portfolio value
        df["Cumulative_Max"] = df["Portfolio_Value"].cummax()

        # Calculate drawdown in dollar terms
        df["Drawdown"] = df["Cumulative_Max"] - df["Portfolio_Value"]

        # Find maximum drawdown in dollar terms
        max_drawdown = (d.gain + 1).cumprod().diff().min()

        comms_rate = 1.7
        winner = d[d["gain"] > 0]
        winner.reset_index(drop=True, inplace=True)

        loser = d[d["gain"] < 0]
        loser.reset_index(drop=True, inplace=True)

        count_winner = len(winner.index)
        count_loser = len(loser.index)

        tick_gain = winner["gain"].sum()
        tick_loss = abs(loser["gain"].sum())
        net_tick = tick_gain - tick_loss

        dollar_gain = tick_gain * 15.625
        dollar_loss = tick_loss * 15.625
        commission = round((count_winner + count_loser) * comms_rate, 2)
        net_dollar = dollar_gain - dollar_loss - commission

        win_rate = count_winner / (count_winner + count_loser)
        win_percentage = round(win_rate * 100, 2)

        profit_factor = round(tick_gain / tick_loss, 2)

        d = {
            "count_winner": count_winner,
            "count_loser": count_loser,
            "tick_gain_median": winner["gain"].median(),
            "tick_gain": tick_gain,
            "tick_loss": tick_loss,
            "net_tick": net_tick,
            "dollar_gain": dollar_gain,
            "dollar_loss": dollar_loss,
            "commission": commission,
            "net_dollar": net_dollar,
            "win_rate": win_percentage,
            "profit_factor": profit_factor,
            # "max_drawdown": max_drawdown,
            # "max_drawdown": max_drawdown,
            "params": self.params,
        }
        # pprint(d)
        return d

        # print(winner)
        # print(loser)
        # print(winner.describe())
        # print(loser.describe())

    def exportable(self):
        transformed_data = []
        for row in self.trades:
            # Determine the buy/sell actions based on type
            if row.type_ == Type.LONG:
                buy_action = {
                    "date": row.open_dt.date(),
                    "time": row.open_dt.time(),
                    "symbol": "ZN",
                    "side": "buy",
                    "currency": "USD",
                    "market": "futures",
                    "price": row.open_price,
                    "quantity": 1,
                }
                sell_action = {
                    "date": row.close_dt.date(),
                    "time": row.close_dt.time(),
                    "symbol": "ZN",
                    "side": "sell",
                    "currency": "USD",
                    "market": "futures",
                    "price": row.close_price,
                    "quantity": 1,
                }
            else:
                buy_action = {
                    "date": row.open_dt.date(),
                    "time": row.open_dt.time(),
                    "symbol": "ZN",
                    "side": "sell",
                    "currency": "USD",
                    "market": "futures",
                    "price": row.open_price,
                    "quantity": 1,
                }
                sell_action = {
                    "date": row.close_dt.date(),
                    "time": row.close_dt.time(),
                    "symbol": "ZN",
                    "side": "buy",
                    "currency": "USD",
                    "market": "futures",
                    "price": row.close_price,
                    "quantity": 1,
                }
            # Append the transformed actions to the list
            transformed_data.append(buy_action)
            transformed_data.append(sell_action)

        df = pd.DataFrame(transformed_data)
        df.to_csv("~/Desktop/transformed_data.csv", index=False)

    def exportable_tradingview(self):
        format_ = "f_trade(_long, _enter_d, _enter_m, _enter_y, _enter_price, _exit_d, _exit_m, _exit_y, _exit_price)"
        transformed_data = []
        for i in self.trades:
            is_long = True if i.type_ == Type.LONG else False
            enter_d = i.open_dt.day
            enter_m = i.open_dt.month
            enter_y = i.open_dt.year
            enter_price = i.open_price

            exit_d = i.close_dt.day
            exit_m = i.close_dt.month
            exit_y = i.close_dt.year
            exit_price = i.close_price

            x = f"f_trade({is_long}, {enter_d}, {enter_m}, {enter_y}, {enter_price}, {exit_d}, {exit_m}, {exit_y}, {exit_price})"
            print(x)

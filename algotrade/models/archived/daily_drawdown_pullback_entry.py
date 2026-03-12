"""
Strategy:
- Get opening price at 17:00
- Determine the direction of the drawdown from the opening price
- If the drawdown is greater than N ticks, it's an entry signal
- Enter trade at the opening price
- Take profit at 15:30 (30 mins before close)

Bugs:
- Skipping a day or two from the opening price (17:00) jumps the signal distance.
You can validate this bug by reviewing the trades from 2024-06-05 to 2024-06-17.
- Why open_dt and signal_dt are the same? It should be different.

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
    price_17_00_date: date
    price_17_00: float

    signal_dt: datetime
    signal_price: float
    signal_dir: str
    signal_dist: float

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

    def set_params(self, tick_signal, tick_stoploss):
        self.tick_signal = tick_signal
        self.tick_stoploss = tick_stoploss

        self.tick_signal = tick_signal
        self.tick_stoploss = tick_stoploss
        self.params = {"tick_signal": tick_signal, "tick_stoploss": tick_stoploss}

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

    def init_strat_vars(self):
        self.price_17_00_date = None
        self.price_17_00 = None

        self.distance = None  # distance from opening price
        self.dd_direction = None  # determine drawdown direction
        self.dd_signal = False
        self.signal = False  # when it hit the drawdown distance
        self.signal_price = None
        self.signal_dt = None

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
        df = df.tail(year)

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

                record["a_dt"] = current_date

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
        print(f"{row['dt']}")
        print(f"OHLC: {row['open']} | {row['high']} | {row['low']} | {row['close']}")
        print(f"price_17_00: {self.price_17_00} | {self.price_17_00_date}")
        print(f"d: {self.distance} s: {self.signal} dir: {self.dd_direction}")
        print(f"s_p: {self.signal_price} | {self.signal_dt}")
        print("")
        print(f"pos: {self._open_position}")
        print(f"open: {self._open_price} | {self._open_dt}")
        print(f"close: {self._close_price} | {self._close_dt}")

        print("\n\n")

    def run(self):
        # Get opening price at 17:00
        for i, row in enumerate(self.data):
            # self._debug(row)
            if row["dt"].time() == time(17, 0):
                self.init_strat_vars()
                self.price_17_00 = row["open"]
                self.price_17_00_date = row["dt"].date()

            if not self._open_position and self.price_17_00 is not None and not self.signal:
                if self.price_17_00 > row["high"]:
                    self.dd_direction = "down"
                    self.distance = nz.decimal_ticks_diff(self.price_17_00, row["high"])

                elif self.price_17_00 < row["low"]:
                    self.dd_direction = "up"
                    self.distance = nz.decimal_ticks_diff(self.price_17_00, row["low"])

                if self.distance and self.distance >= self.tick_signal:
                    self.signal = True
                    self.signal_dt = row["dt"]

                    if self.dd_direction == "up":
                        self.signal_price = row["high"]
                    elif self.dd_direction == "down":
                        self.signal_price = row["low"]

            if self._open_position is False and self.signal:
                if self.dd_direction == "up" and self.price_17_00 >= row["low"]:
                    self._type = Type.LONG
                    self._open_position = True
                    self._open_dt = row["dt"]
                    self._open_price = self.price_17_00
                    self._stoploss = nz.decimal_subtraction_by_ticks(self._open_price, self.tick_stoploss)

                elif self.dd_direction == "down" and self.price_17_00 <= row["high"]:
                    self._type = Type.SHORT
                    self._open_position = True
                    self._open_dt = row["dt"]
                    self._open_price = self.price_17_00
                    self._stoploss = nz.decimal_addition_by_ticks(self._open_price, self.tick_stoploss)

            # Take profit when the time is 8:30am
            if row["dt"].time() == time(15, 59) and self._open_position:
                self._close_price = row["close"]
                gain = abs(nz.decimal_ticks_diff(self._open_price, self._close_price))

                trade = Trade(
                    price_17_00_date=self.price_17_00_date,
                    price_17_00=self.price_17_00,
                    signal_dt=self.signal_dt,
                    signal_price=self.signal_price,
                    signal_dir=self.dd_direction,
                    signal_dist=self.distance,
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
                    gain = -abs(nz.decimal_ticks_diff(self._open_price, self._close_price))

                    trade = Trade(
                        price_17_00_date=self.price_17_00_date,
                        price_17_00=self.price_17_00,
                        signal_dt=self.signal_dt,
                        signal_price=self.signal_price,
                        signal_dir=self.dd_direction,
                        signal_dist=self.distance,
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
                    gain = -abs(nz.decimal_ticks_diff(self._open_price, self._close_price))

                    trade = Trade(
                        price_17_00_date=self.price_17_00_date,
                        price_17_00=self.price_17_00,
                        signal_dt=self.signal_dt,
                        signal_price=self.signal_price,
                        signal_dir=self.dd_direction,
                        signal_dist=self.distance,
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

        # remove last trade since it is not closed
        # self.trades.pop()

    def export(self):
        d = pd.json_normalize(asdict(obj) for obj in self.trades)
        d.to_csv("~/Desktop/trades.csv", index=False)

    def analyze(self):
        d = pd.json_normalize(asdict(obj) for obj in self.trades)
        # print(d)

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
            "params": self.params,
        }
        # pprint(d)
        return d

        # print(winner)
        # print(loser)
        # print(winner.describe())
        # print(loser.describe())

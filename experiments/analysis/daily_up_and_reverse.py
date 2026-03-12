"""
Find the median of the data where:

1. candle is green, then it reverses and closed red.
2. candle is red, then it reverses and closed green.

Questions
1. Find what time of day is the lowest before the reversal.
2. Find what time of day it reversed to the open and never touched the open again.
- This might be interesting data
- I used 30 minutes data to infer the reversal time,
it takes me 2 days to transform it into Daily timeframe.

3. How many ticks is the drawdown from the open to the lowest of the day.
- This might be interesting data

Constraints:
- The candle opens at 17:00:00 (UTC-5 Chicago time)
- Skip NYSE holidays and weekends

Comments
- Seems like the data is correct, I compared March 29, 2024 data with trading view
- I also considered the NYSE holidays, and skipped those dates

Problem:
- The drawdown is not calculated correctly, the price can go back and forth at the open price
- The drawdown is calculated from the open price to the low/high of the day,
it should be calculated from the open price to the low/high of the day after the reversal


Solution:
- Determine the time when it's the lowest before the reversal. This is the lowest time of the day.
- Get how many ticks the drawdown. It's essentially open - low/high on daily timeframe.
- After the "lowest time" of day drawdown, determine the time it touch the open. This is the reversal time.
- Get how many ticks the drawdown from the entry
"""

from datetime import datetime, time, timedelta
from pprint import pprint
from typing import Dict, List

import holidays
import pandas as pd
from algotrade.core import nz

nyse_holidays = holidays.NYSE()


day_open_time = time(17, 00)
day_close_time = time(15, 59)


def load_data() -> List[Dict]:
    df = pd.read_csv("~/dataset/zn-1m.csv", sep=";")
    df.columns = ["date", "time", "open", "high", "low", "close", "vol"]

    day = 1440
    year = day * 365
    quarter = int(year / 4)
    df = df.tail(year)

    df["dt"] = df["date"] + " " + df["time"]
    df["dt"] = pd.to_datetime(df["dt"], dayfirst=True)
    # df = df.sort_values(by="dt")
    # df.to_csv("~/desktop/0_raw_data.csv", index=False)
    # return df.to_dict(orient="records")
    return df


def load_data_with_tick_price(records: List[Dict]) -> List[Dict]:
    for record in records:
        record["o_open"] = record["open"]
        record["o_high"] = record["high"]
        record["o_low"] = record["low"]
        record["o_close"] = record["close"]

        record["open"] = nz.decimal_to_treasury(record["open"])
        record["high"] = nz.decimal_to_treasury(record["high"])
        record["low"] = nz.decimal_to_treasury(record["low"])
        record["close"] = nz.decimal_to_treasury(record["close"])

    return records


def adjusted_date(records: List[Dict]):
    def _new(records):
        skipped_dates = []
        for record in records:
            if record["dt"].date() in nyse_holidays:
                skipped_dates.append(record["dt"].date())

            if record["dt"].time() == day_open_time and record["dt"].date() not in nyse_holidays:
                current_date = record["dt"].date() + timedelta(days=1)

            if current_date in nyse_holidays:
                current_date += timedelta(days=1)

            record["adjusted_date"] = current_date

        skipped_dates = set(skipped_dates)
        for date in skipped_dates:
            print(f"Skipping {date}")

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


def group_by(records: List[Dict], key: str) -> Dict[str, List[Dict]]:
    # Group the data by date
    day_data = {}
    for row in records:
        date = row[key]
        if date not in day_data:
            day_data[date] = []

        row.pop(key, None)
        day_data[date].append(row)

    return day_data


def tabulate(records) -> List[Dict]:
    tabbulated = []
    for date, data in records.items():
        open_ = data[0]["open"]
        close = data[-1]["close"]

        open_date = data[0]["dt"].date()
        open_time = data[0]["dt"].time()
        close_time = data[-1]["dt"].time()

        high = None
        low = None

        low_of_day_date = None  # this considers holidays and weekends in the calculation
        low_of_day_time = None

        direction = None
        day_drawdown_tick = None
        gain = None

        for row in data:
            if high is None or row["high"] > high:
                high = row["high"]
            elif low is None or row["low"] < low:
                low = row["low"]

        if close > open_:
            direction = "up"
            low = low if low and low < open_ else open_
            day_drawdown_tick = nz.decimal_ticks_diff(open_, low)

        elif close < open_:
            direction = "down"
            high = high if high and high > open_ else open_
            day_drawdown_tick = nz.decimal_ticks_diff(open_, high)

        else:
            direction = "even"
            day_drawdown_tick = 0

        if day_drawdown_tick > 0:
            for row in data:
                if direction == "up" and row["low"] == low:
                    low_of_day_date = row["dt"].date()
                    low_of_day_time = row["dt"].time()
                    break
                elif direction == "down" and row["high"] == high:
                    low_of_day_date = row["dt"].date()
                    low_of_day_time = row["dt"].time()
                    break

        drawdown_after_reversal_price = 0.0
        drawdown_after_reversal_tick = 0
        drawdown_after_reversal_time = None
        drawdown_after_reversal_date = None

        reversal_date = None
        reversal_time = None
        for row in data:
            if not low_of_day_date or not low_of_day_time:
                break

            # skip the data before the lowest time of the day
            low_of_day_datetime = datetime.combine(low_of_day_date, low_of_day_time)
            if row["dt"] < low_of_day_datetime:
                continue

            # determine when it reversed to the open
            if direction == "up" and row["high"] > open_ and not reversal_time:
                reversal_date = row["dt"].date()
                reversal_time = row["dt"].time()
            elif direction == "down" and row["low"] < open_ and not reversal_time:
                reversal_date = row["dt"].date()
                reversal_time = row["dt"].time()

            if reversal_time == day_close_time:
                reversal_time = None
                reversal_date = None

            # after reversal, calculate the largest drawdown
            if reversal_time:
                if direction == "up" and open_ > row["low"]:
                    if drawdown_after_reversal_price > 0 and row["low"] > drawdown_after_reversal_price:
                        continue

                    drawdown_after_reversal_price = row["low"]
                    drawdown_after_reversal_date = row["dt"].date()
                    drawdown_after_reversal_time = row["dt"].time()

                elif direction == "down" and open_ < row["high"]:
                    if drawdown_after_reversal_price > 0 and row["high"] < drawdown_after_reversal_price:
                        continue

                    drawdown_after_reversal_price = row["high"]
                    drawdown_after_reversal_date = row["dt"].date()
                    drawdown_after_reversal_time = row["dt"].time()

        drawdown_after_reversal_tick = (
            nz.decimal_ticks_diff(open_, drawdown_after_reversal_price) if drawdown_after_reversal_price > 0 else 0
        )
        gain = nz.decimal_ticks_diff(open_, close)

        debug_data = {
            "b_date": date,
            "date": open_date,
            "open": nz.decimal_to_treasury(open_),
            "close": nz.decimal_to_treasury(close),
            "d_open": open_,
            "d_close": close,
            "open_time": open_time,
            "close_time": close_time,
            "high": nz.decimal_to_treasury(high),
            "low": nz.decimal_to_treasury(low),
            "d_high": high,
            "d_low": low,
            "dir": direction,
            "LoD_date": low_of_day_date,
            "LoD_time": low_of_day_time,  # the time when it's the lowest time of the day
            "r_date": reversal_date,
            "r_time": reversal_time,  # the time when it reversed to the open after the lowest time of the day
            "d_dd_tick": day_drawdown_tick,
            "dd_after_reversal_tick": drawdown_after_reversal_tick,
            "dd_after_reversal_price": nz.decimal_to_treasury(drawdown_after_reversal_price),
            "dd_after_reversal_d_price": drawdown_after_reversal_price,
            "dd_after_reversal_time": drawdown_after_reversal_time,
            "dd_after_reversal_date": drawdown_after_reversal_date,
            "gain": gain,
        }

        tabbulated.append(debug_data)
    return tabbulated


def analze(df):
    # drop rows where None or 0
    df = df.dropna()
    df = df[df["d_dd_tick"] > 0]

    # offset time by - 17 hours
    df["dt"] = df["r_date"].astype(str) + " " + df["r_time"].astype(str)
    df["dt"] = pd.to_datetime(df["dt"])

    # get the median d_dd_tick
    median = df["d_dd_tick"].median()
    print(f"Median d_dd_tick: {median}")

    # get the median dd_after_reversal_tick
    median = df["dd_after_reversal_tick"].median()
    print(f"Median dd_after_reversal_tick: {median}")

    # mean
    mean = df["d_dd_tick"].mean()
    print(f"Mean d_dd_tick: {mean}")

    mean = df["dd_after_reversal_tick"].mean()
    print(f"Mean dd_after_reversal_tick: {mean}")

    d = df[["d_dd_tick", "dd_after_reversal_tick"]]
    print(d.describe())
    return df


def backtest(df):
    d_dd_tick = 10
    dd_after_reversal_tick = 15

    comms_rate = 1.7

    df = df[["b_date", "r_date", "r_time", "d_dd_tick", "dd_after_reversal_tick", "gain"]]
    df = df[df["d_dd_tick"] >= d_dd_tick]
    winner = df[df["dd_after_reversal_tick"] < dd_after_reversal_tick]
    winner.reset_index(drop=True, inplace=True)

    loser = df[df["dd_after_reversal_tick"] >= dd_after_reversal_tick]
    loser.reset_index(drop=True, inplace=True)

    count_winner = len(winner.index)
    count_loser = len(loser.index)

    tick_gain = winner["gain"].sum()
    tick_loss = loser["gain"].count() * dd_after_reversal_tick
    net_tick = tick_gain - tick_loss

    dollar_gain = tick_gain * 15.625
    dollar_loss = tick_loss * 15.625
    commission = round((count_winner + count_loser) * comms_rate, 2)
    net_dollar = dollar_gain - dollar_loss - commission

    win_rate = round(len(winner.index) / len(df.index) * 100, 2)
    profit_factor = int(tick_gain / tick_loss)

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
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
    pprint(d)

    print(winner)
    # print(winner.describe())
    # print(loser.describe())

    # print(winner[["b_date"]])


def main():
    records = load_data()
    records = adjusted_date(records)
    pd.DataFrame(records).to_csv("~/desktop/0_adjusted_date.csv", index=False)

    day_data = group_by(records, "adjusted_date")
    tabulated = tabulate(day_data)

    df = pd.DataFrame(tabulated)
    df["r_date"] = pd.to_datetime(df["r_date"])

    df = df[::-1]  # reverse the dataframe
    df = df.iloc[1:]  # remove first row, because it's incomplete data
    df.to_csv("~/desktop/1_daily_up_and_reverse.csv", index=False)

    df = df[["r_date", "r_time", "d_dd_tick", "dd_after_reversal_tick"]]
    df = analze(df)
    df.to_csv("~/desktop/2_analysis.csv", index=False)


def post_analysis():
    df = pd.read_csv("~/desktop/1_daily_up_and_reverse.csv")
    df["r_date"] = pd.to_datetime(df["r_date"]).dt.date
    df = df[df["r_date"] >= datetime(2024, 1, 1).date()]
    df = analze(df)
    backtest(df)


def advance_bt():
    import pandas as pd

    df = load_data()

    # Set the parameters
    ticks_to_move = 10
    stop_loss_ticks = 15
    tick_size = 1 / 64

    # Convert timestamp to datetime and set it as the index
    df["dt"] = pd.to_datetime(df["dt"])
    df.set_index("dt", inplace=True)

    # Get the 17:00 opening price
    df["17:00_open"] = df[df.index.time == pd.Timestamp("17:00").time()]["open"]

    # Forward fill the 17:00 opening price
    df["17:00_open"] = df["17:00_open"].ffill()

    # Check if the price has moved 10 ticks from the 17:00 opening price
    df["price_moved_up"] = df["high"] >= df["17:00_open"] + (ticks_to_move * tick_size)
    df["price_moved_down"] = df["low"] <= df["17:00_open"] - (ticks_to_move * tick_size)

    # Detect the reversal to 17:00 opening price
    # If price_moved_up, lows < open
    df["reversal_to_17:00"] = df["price_moved_up"] & (df["low"] < df["17:00_open"])
    # If price_moved_down, highs > open
    df["reversal_to_17:00"] = df["price_moved_down"] & (df["high"] > df["17:00_open"])

    # Initialize trade signals
    df["signal"] = 0

    # Identify entry points based on the reversal to the 17:00 opening price
    df.loc[df["price_moved_up"] & df["reversal_to_17:00"], "signal"] = -1  # Short
    df.loc[df["price_moved_down"] & df["reversal_to_17:00"], "signal"] = 1  # Long

    # Remove duplicate entries
    df["signal"] = df["signal"].replace(to_replace=0, method="ffill")

    # Define stop loss and take profit
    df["stop_loss"] = df.apply(
        lambda row: (
            row["17:00_open"] + stop_loss_ticks * tick_size
            if row["signal"] == -1
            else row["17:00_open"] - stop_loss_ticks * tick_size if row["signal"] == 1 else None
        ),
        axis=1,
    )

    # Implement take profit at 15:30 the next day
    df["take_profit"] = df.index.time == pd.Timestamp("15:30").time()
    df["signal"] = df.apply(lambda row: 0 if row["take_profit"] else row["signal"], axis=1)

    # Shift the signals to account for trade execution on the next time period
    df["signal"] = df["signal"].shift()

    # Drop the helper columns
    # df.drop(columns=["price_moved_up", "price_moved_down", "reversal_to_17:00", "17:00_open", "take_profit"], inplace=True)

    # Display the DataFrame
    df.to_csv("~/desktop/advance_bt.csv", index=False)


# main()
# post_analysis()
# advance_bt()

df = pd.read_csv("~/desktop/advance_bt.csv")
df["gain"] = df.apply(lambda row: row["close"] - row["open"] if row["signal"] == 1 else row["open"] - row["close"], axis=1)
dd = df[df["signal"] == 1]
print(dd)
# winner = df[df["take_profit"] == True]
# losser = df[df["stop_loss"] > 0]
# print(winner)

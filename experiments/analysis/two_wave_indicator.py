import pandas as pd
from algotrade.core import nz


def indicator():
    d = pd.read_csv("~/desktop/test.csv")
    d["dt"] = pd.to_datetime(d["dt"])
    d["a_dt"] = pd.to_datetime(d["a_dt"])

    data = []

    d_f_ticks = None
    s_lowest_touch = False
    s_lowest_touch_price = None
    d_drawdown = None
    d_drawdown_price = None
    d_drawdown_dt = None
    has_touch_open = False

    rows = d.to_dict(orient="records")
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

    if df.empty:
        return df

    # Rolling window f_ticks and s_ticks median, mean, max, min, std
    window = 5

    f_ticks_rolling = df["f_ticks"].rolling(window=window)
    df["f_ticks_median"] = f_ticks_rolling.median()
    df["f_ticks_mean"] = f_ticks_rolling.mean()
    df["f_ticks_max"] = f_ticks_rolling.max()
    df["f_ticks_min"] = f_ticks_rolling.min()
    # df["f_ticks_std"] = f_ticks_rolling.std()

    s_ticks_rolling = df["s_ticks"].rolling(window=window)
    df["s_ticks_median"] = s_ticks_rolling.median()
    df["s_ticks_mean"] = s_ticks_rolling.mean()
    df["s_ticks_max"] = s_ticks_rolling.max()
    df["s_ticks_min"] = s_ticks_rolling.min()
    # df["s_ticks_std"] = s_ticks_rolling.std()

    dd_rolling = df["dd"].rolling(window=window)
    df["dd_median"] = dd_rolling.median()
    df["dd_mean"] = dd_rolling.mean()
    df["dd_max"] = dd_rolling.max()
    df["dd_min"] = dd_rolling.min()
    # df["dd_std"] = dd_rolling.std()

    # df.dropna(inplace=True)

    return df[
        [
            "a_dt",
            "a_dir",
            "gain",
            "f_ticks",
            "s_ticks",
            "dd",
            "f_ticks_median",
            "s_ticks_median",
            "dd_median",
            "f_ticks_mean",
            "s_ticks_mean",
            "dd_mean",
            "f_ticks_max",
            "s_ticks_max",
            "dd_max",
            "f_ticks_min",
            "s_ticks_min",
            "dd_min",
        ]
    ]

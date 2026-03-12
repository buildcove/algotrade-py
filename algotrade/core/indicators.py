import pandas as pd


def rma(series: pd.Series, length=14):
    alpha = 1.0 / length
    r = series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    r.iloc[length - 1] = series.iloc[:length].mean()  # seed = SMA
    return r


def atr(df, length=14, high="high", low="low", close="close"):
    """
    Average True Range ⟵ uses pine_rma() for smoothing.
    df      : DataFrame with H, L, C columns
    length  : look-back period (default 14)
    """
    h, l, c = df[high], df[low], df[close]
    prev_c = c.shift(1)

    # True Range (Wilder)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

    # Wilder / TradingView smoothing
    atr_series = rma(tr, length)
    atr_series.name = f"ATR_{length}"
    return atr_series


def adx(df: pd.DataFrame, dilen: int = 14, adxlen: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    dilen : int
        Length for DI smoothing (default 14).
    adxlen : int
        Length for ADX smoothing (default 14).

    Returns
    -------
    pd.Series
        The ADX values.
    """
    # 1) Directional Movements
    up = df["high"].diff()
    down = -df["low"].diff()
    plusDM = up.where((up > down) & (up > 0), 0.0)
    minusDM = down.where((down > up) & (down > 0), 0.0)

    # 2) True Range
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # 3) Smooth TR and DMs with RMA
    tr_rma = rma(tr, dilen)
    plus_rma = rma(plusDM, dilen)
    minus_rma = rma(minusDM, dilen)

    # 4) Directional Indices
    plusDI = (plus_rma * 100).div(tr_rma).fillna(0)
    minusDI = (minus_rma * 100).div(tr_rma).fillna(0)

    # 5) DX and ADX
    denom = plusDI + minusDI
    # avoid division by zero
    denom = denom.replace(0, 1)
    dx = (plusDI - minusDI).abs().div(denom)
    adx_series = rma(dx, adxlen) * 100

    return adx_series

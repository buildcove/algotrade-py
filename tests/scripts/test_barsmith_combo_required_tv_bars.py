from __future__ import annotations

import numpy as np
import pandas as pd
from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
    build_python_indicator_frame,
)


def _make_synthetic_ohlc(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    base = 100.0 + np.arange(n, dtype=float) * 0.1
    open_ = base + (np.sin(np.arange(n, dtype=float) * 0.2) * 0.05)
    close = open_ + (np.where((np.arange(n) % 2) == 0, 0.2, -0.2))
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    return pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close})


def test_barsmith_combo_requires_200_bars_for_sma200() -> None:
    required_cols = [
        "open",
        "high",
        "low",
        "close",
        "timestamp",
        "lower_high",
        "higher_low",
        "kf_atr_contracting",
        "rsi_14",
        "macd",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "wicks_diff_sma14",
        "atr",
        "200sma",
        "9ema",
    ]

    df_199 = _make_synthetic_ohlc(199)
    ind_199 = build_python_indicator_frame(df_199).sort_values("timestamp")
    ind_199["lower_high"] = ind_199["high"] < ind_199["high"].shift(1)
    ind_199["higher_low"] = ind_199["low"] > ind_199["low"].shift(1)
    ind_199["kf_atr_contracting"] = ind_199["kf_atr"].diff() < 0.0
    assert ind_199.dropna(subset=required_cols).empty

    df_200 = _make_synthetic_ohlc(200)
    ind_200 = build_python_indicator_frame(df_200).sort_values("timestamp")
    ind_200["lower_high"] = ind_200["high"] < ind_200["high"].shift(1)
    ind_200["higher_low"] = ind_200["low"] > ind_200["low"].shift(1)
    ind_200["kf_atr_contracting"] = ind_200["kf_atr"].diff() < 0.0
    assert not ind_200.dropna(subset=required_cols).empty

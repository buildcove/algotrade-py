from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ib_async")

from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, repository, types
from algotrade.models.barsmith_combo_2x_atr_tp_atr_stop import (
    BARSMITH_COMBO_REQUIRED_TV_BARS,
    BarsmithComboExecutor,
    DataFetchError,
    Extra,
    Param,
    Var,
)


def _make_executor() -> BarsmithComboExecutor:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)
    return BarsmithComboExecutor(
        broker=broker,
        param=Param(contract=contract_spec, supported_contract=types.SupportedContract.MES),
        var=Var(),
        extra=Extra(),
    )


def _make_synthetic_ohlc(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    base = 100.0 + np.arange(n, dtype=float) * 0.1
    open_ = base + (np.sin(np.arange(n, dtype=float) * 0.2) * 0.05)
    close = open_ + (np.where((np.arange(n) % 2) == 0, 0.2, -0.2))
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    return pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close})


def test_signal_candle_requires_minimum_bars_for_trend_strength() -> None:
    exe = _make_executor()

    df_199 = _make_synthetic_ohlc(BARSMITH_COMBO_REQUIRED_TV_BARS - 1)
    with pytest.raises(DataFetchError):
        exe._signal_candle_from_raw_ohlc(df_raw=df_199, expected_latest_bar_start_utc=pd.Timestamp("2024-01-05T00:00:00Z"))

    df_required = _make_synthetic_ohlc(BARSMITH_COMBO_REQUIRED_TV_BARS)
    candle, _ = exe._signal_candle_from_raw_ohlc(
        df_raw=df_required, expected_latest_bar_start_utc=pd.Timestamp("2024-01-05T00:00:00Z")
    )
    assert np.isfinite(float(candle.trend_strength))

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

pytest.importorskip("ib_async")

from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, repository, types
from algotrade.models.barsmith_combo_2x_atr_tp_atr_stop import (
    BarsmithComboExecutor,
    Extra,
    Param,
    Var,
)


def test_scan_nbars_outputs_realized_rr(monkeypatch: pytest.MonkeyPatch) -> None:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)
    exe = BarsmithComboExecutor(
        broker=broker,
        param=Param(contract=contract_spec, supported_contract=types.SupportedContract.MES),
        var=Var(),
        extra=Extra(),
    )

    # 3 bars: bar0 is the signal, bar1 hits TP, bar2 is newest/in-progress and is excluded.
    engineered = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "atr": 1.0,
                "is_tribar_green": True,
                "is_kf_breakout_potential": True,
                "kf_innovation_abs": 6.0,
                "macd_hist": -1.0,
                "trend_strength": 0.25,
                "kf_adx": 10.0,
                "kf_atr": 1.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01T00:30:00Z"),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "atr": 1.0,
                "is_tribar_green": False,
                "is_kf_breakout_potential": False,
                "kf_innovation_abs": 0.0,
                "macd_hist": 0.0,
                "trend_strength": 0.0,
                "kf_adx": 10.0,
                "kf_atr": 1.0,
            },
            {
                "timestamp": pd.Timestamp("2025-01-01T01:00:00Z"),
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "atr": 1.0,
                "is_tribar_green": False,
                "is_kf_breakout_potential": False,
                "kf_innovation_abs": 0.0,
                "macd_hist": 0.0,
                "trend_strength": 0.0,
                "kf_adx": 10.0,
                "kf_atr": 1.0,
            },
        ]
    )

    def _fake_build(_df_raw: pd.DataFrame) -> pd.DataFrame:
        return engineered

    monkeypatch.setattr(
        "algotrade.barsmith_parity.check_barsmith_indicator_parity.build_python_indicator_frame_minimal", _fake_build
    )

    df_raw = pd.DataFrame({"timestamp": [datetime(2025, 1, 1)]})
    trades = exe.simulate_realized_trades_from_raw_ohlc(df_raw)
    assert len(trades) == 1
    assert trades.iloc[0]["rr"] == pytest.approx(2.0)

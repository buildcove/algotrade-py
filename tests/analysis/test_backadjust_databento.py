"""Unit tests for tools.backadjust_databento utilities."""

from datetime import time, timedelta
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("databento")

from tools.backadjust_databento import (
    RolloverSpec,
    backadjust_databento,
    compute_adjustments,
    resample_with_session,
)


def _make_frame(rows):
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def test_compute_adjustments_uses_rollover_prices():
    roll_dt = pd.Timestamp("2023-12-12T23:00:00Z")
    frames = {
        "ESZ2023": _make_frame(
            [
                {"timestamp": "2023-12-12T22:59:00Z", "open": 100, "high": 101, "low": 99, "close": 101, "volume": 10},
            ]
        ),
        "ESH2024": _make_frame(
            [
                {"timestamp": "2023-12-12T23:00:00Z", "open": 90, "high": 91, "low": 89, "close": 90, "volume": 12},
            ]
        ),
    }
    rollovers = [RolloverSpec(roll_dt, "ESZ2023", "ESH2024")]
    adjustments = compute_adjustments(frames, rollovers, tolerance=timedelta(minutes=5))
    assert adjustments["ESH2024"] == 0.0
    assert adjustments["ESZ2023"] == -11.0


def test_resample_with_session_aligns_to_session_open():
    df = _make_frame(
        [
            {"timestamp": "2024-01-01T18:00:00Z", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 1},
            {"timestamp": "2024-01-02T00:30:00Z", "open": 2, "high": 3, "low": 2, "close": 3, "volume": 1},
        ]
    )
    result = resample_with_session(df, rule="1D", session_start=time(17, 0), session_timezone="UTC")
    assert result.index[0] == pd.Timestamp("2024-01-01T17:00:00Z")


def test_backadjust_databento_pipeline(tmp_path: Path):
    roll_dt = pd.Timestamp("2023-12-12T22:01:00Z")
    data_a = """ts_event,open,high,low,close,volume
2023-12-12T21:59:00Z,100,101,99,100,10
2023-12-12T22:00:00Z,101,102,100,101,10
"""
    data_b = """ts_event,open,high,low,close,volume
2023-12-12T22:01:00Z,90,91,89,90,12
2023-12-12T22:02:00Z,91,92,90,91,12
"""
    (tmp_path / "ESZ2023.csv").write_text(data_a)
    (tmp_path / "ESH2024.csv").write_text(data_b)

    rollovers = [RolloverSpec(roll_dt, "ESZ2023", "ESH2024")]
    df = backadjust_databento(
        databento_path=tmp_path,
        timeframe="1m",
        rollovers=rollovers,
        session_start=time(0, 0),
        session_timezone="UTC",
    )

    first_bar = df.iloc[0]
    assert first_bar["open"] == 89.0  # 100 price adjusted down by 11 ticks


def test_backadjust_with_reference_aligns_prices(tmp_path: Path):
    roll_dt = pd.Timestamp("2023-12-12T22:01:00Z")
    raw = """ts_event,open,high,low,close,volume
2023-12-12T22:01:00Z,90,91,89,90,12
2023-12-12T22:02:00Z,91,92,90,91,12
"""
    (tmp_path / "ESH2024.csv").write_text(raw)

    reference = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2023-12-12T22:01:00Z", "2023-12-12T22:02:00Z"],
                utc=True,
            ),
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100, 101],
            "volume": [12, 12],
        }
    ).set_index("timestamp")

    rollovers = [RolloverSpec(roll_dt, "ESZ2023", "ESH2024")]
    df = backadjust_databento(
        databento_path=tmp_path,
        timeframe="1m",
        rollovers=rollovers,
        session_start=time(0, 0),
        session_timezone="UTC",
        reference_df=reference,
        tick_size=1.0,
    )

    pd.testing.assert_series_equal(df["close"], reference["close"], check_freq=False, check_dtype=False)

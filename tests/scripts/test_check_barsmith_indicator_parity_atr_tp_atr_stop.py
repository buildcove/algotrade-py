from __future__ import annotations

import numpy as np
import pytest
from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
    compute_atr_tp_atr_stop_targets_and_rr,
)


def test_atr_tp_atr_stop_long_tp_is_1x_atr() -> None:
    # Entry is at close[0]=100. ATR=2 => stop=98, TP=102, risk=2.
    # Next bar hits TP => RR=(102-100)/2 = 1.
    open_ = np.array([99.0, 100.0, 100.0], dtype=float)
    high = np.array([100.5, 102.0, 100.0], dtype=float)
    low = np.array([80.0, 99.0, 100.0], dtype=float)
    close = np.array([100.0, 101.0, 100.0], dtype=float)
    atr = np.array([2.0, 2.0, 2.0], dtype=float)

    long, short, rr_long, rr_short = compute_atr_tp_atr_stop_targets_and_rr(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr,
        tick_size=None,
        resolve_end_idx=None,
    )

    assert bool(long[0]) is True
    assert bool(short[0]) is False
    assert float(rr_long[0]) == pytest.approx(1.0)
    assert np.isnan(rr_short[0])


def test_atr_tp_atr_stop_short_tp_is_1x_atr() -> None:
    # Entry is at close[0]=100 (red candle). ATR=2 => stop=102, TP=98, risk=2.
    # Next bar hits TP => RR=(100-98)/2 = 1.
    open_ = np.array([101.0, 100.0, 100.0], dtype=float)
    high = np.array([101.0, 101.0, 100.0], dtype=float)
    low = np.array([99.0, 98.0, 100.0], dtype=float)
    close = np.array([100.0, 99.0, 100.0], dtype=float)
    atr = np.array([2.0, 2.0, 2.0], dtype=float)

    long, short, rr_long, rr_short = compute_atr_tp_atr_stop_targets_and_rr(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr,
        tick_size=None,
        resolve_end_idx=None,
    )

    assert bool(long[0]) is False
    assert bool(short[0]) is True
    assert np.isnan(rr_long[0])
    assert float(rr_short[0]) == pytest.approx(1.0)


def test_atr_tp_atr_stop_skips_trade_when_atr_yields_zero_risk() -> None:
    # ATR=0 => stop=entry => invalid => no trade.
    open_ = np.array([99.0, 100.0], dtype=float)
    high = np.array([101.0, 101.0], dtype=float)
    low = np.array([99.0, 99.0], dtype=float)
    close = np.array([100.0, 100.0], dtype=float)
    atr = np.array([0.0, 0.0], dtype=float)

    long, short, rr_long, rr_short = compute_atr_tp_atr_stop_targets_and_rr(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr,
        tick_size=0.25,
        resolve_end_idx=None,
    )

    assert bool(long[0]) is False
    assert bool(short[0]) is False
    assert np.isnan(rr_long[0])
    assert np.isnan(rr_short[0])

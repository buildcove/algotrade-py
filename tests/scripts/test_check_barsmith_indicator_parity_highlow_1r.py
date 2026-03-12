from __future__ import annotations

import numpy as np
import pytest
from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
    compute_highlow_1r_targets_and_rr,
)


def test_highlow_1r_long_tp_is_1r() -> None:
    # Entry is at close[0]=100. Stop=low[0]=99 => risk=1, TP=101.
    # Next bar hits TP => RR=1.
    open_ = np.array([99.0, 100.0, 100.0], dtype=float)
    high = np.array([101.0, 101.0, 100.0], dtype=float)
    low = np.array([99.0, 99.5, 100.0], dtype=float)
    close = np.array([100.0, 100.5, 100.0], dtype=float)
    atr = np.array([5.0, 5.0, 5.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_1r_targets_and_rr(
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


def test_highlow_1r_gap_tp_fill_can_exceed_1r() -> None:
    # Entry is at close[0]=100. Stop=low[0]=99 => risk=1, TP=101.
    # Next bar opens above TP => fill at open=111 => RR=(111-100)/1 = 11.
    open_ = np.array([99.0, 111.0], dtype=float)
    high = np.array([101.0, 111.0], dtype=float)
    low = np.array([99.0, 110.0], dtype=float)
    close = np.array([100.0, 111.0], dtype=float)
    atr = np.array([5.0, 5.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_1r_targets_and_rr(
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
    assert float(rr_long[0]) == pytest.approx(11.0)
    assert np.isnan(rr_short[0])


def test_highlow_1r_skips_trade_when_stop_not_below_entry() -> None:
    # Low equals entry => stop quantizes to entry => invalid (risk=0) => no trade.
    open_ = np.array([99.0, 100.0], dtype=float)
    high = np.array([100.25, 100.25], dtype=float)
    low = np.array([100.0, 100.0], dtype=float)
    close = np.array([100.0, 100.0], dtype=float)
    atr = np.array([1.0, 1.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_1r_targets_and_rr(
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


def test_highlow_1r_short_tp_is_1r() -> None:
    # Entry is at close[0]=100 (red candle). Stop=high[0]=101 => risk=1, TP=99.
    # Next bar hits TP => RR=1.
    open_ = np.array([101.0, 100.0, 100.0], dtype=float)
    high = np.array([101.0, 100.75, 100.0], dtype=float)
    low = np.array([99.0, 99.0, 100.0], dtype=float)
    close = np.array([100.0, 99.5, 100.0], dtype=float)
    atr = np.array([5.0, 5.0, 5.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_1r_targets_and_rr(
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


def test_highlow_1r_short_gap_tp_fill_can_exceed_1r() -> None:
    # Entry is at close[0]=100 (red candle). Stop=high[0]=101 => risk=1, TP=99.
    # Next bar opens below TP => fill at open=89 => RR=(100-89)/1 = 11.
    open_ = np.array([101.0, 89.0], dtype=float)
    high = np.array([101.0, 100.0], dtype=float)
    low = np.array([99.0, 89.0], dtype=float)
    close = np.array([100.0, 90.0], dtype=float)
    atr = np.array([5.0, 5.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_1r_targets_and_rr(
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
    assert float(rr_short[0]) == pytest.approx(11.0)

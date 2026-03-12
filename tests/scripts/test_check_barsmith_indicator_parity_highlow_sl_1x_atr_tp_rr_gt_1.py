from __future__ import annotations

import numpy as np
import pytest
from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
    compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr,
)


def test_highlow_sl_1x_atr_tp_rr_gt_1_long_hits_tp_when_rr_at_tp_gt_1() -> None:
    # Entry=close[0]=100 (green). Stop=low[0]=99 => risk=1. ATR=2 => TP=102.
    open_ = np.array([99.0, 100.0, 100.0], dtype=float)
    high = np.array([100.5, 102.0, 100.0], dtype=float)
    low = np.array([99.0, 99.25, 100.0], dtype=float)
    close = np.array([100.0, 101.0, 100.0], dtype=float)
    atr = np.array([2.0, 2.0, 2.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
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
    assert float(rr_long[0]) == pytest.approx(2.0)
    assert np.isnan(rr_short[0])


def test_highlow_sl_1x_atr_tp_rr_gt_1_long_rejects_when_rr_at_tp_is_1() -> None:
    # Entry=100 (green). Stop=99 => risk=1. ATR=1 => TP=101.
    # RR_at_tp = (101-100)/1 = 1.0 => strict gate (>1) rejects.
    open_ = np.array([99.0, 100.0, 100.0], dtype=float)
    high = np.array([100.5, 102.0, 100.0], dtype=float)
    low = np.array([99.0, 99.25, 100.0], dtype=float)
    close = np.array([100.0, 101.0, 100.0], dtype=float)
    atr = np.array([1.0, 1.0, 1.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
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


def test_highlow_sl_1x_atr_tp_rr_gt_1_short_hits_tp_when_rr_at_tp_gt_1() -> None:
    # Entry=close[0]=100 (red). Stop=high[0]=101 => risk=1. ATR=2 => TP=98.
    open_ = np.array([101.0, 100.0, 100.0], dtype=float)
    high = np.array([101.0, 100.75, 100.0], dtype=float)
    low = np.array([99.0, 98.0, 100.0], dtype=float)
    close = np.array([100.0, 99.0, 100.0], dtype=float)
    atr = np.array([2.0, 2.0, 2.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
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
    assert float(rr_short[0]) == pytest.approx(2.0)


def test_highlow_sl_1x_atr_tp_rr_gt_1_short_rejects_when_rr_at_tp_is_1() -> None:
    # Entry=100 (red). Stop=101 => risk=1. ATR=1 => TP=99.
    # RR_at_tp = (100-99)/1 = 1.0 => strict gate (>1) rejects.
    open_ = np.array([101.0, 100.0, 100.0], dtype=float)
    high = np.array([101.0, 101.0, 100.0], dtype=float)
    low = np.array([99.0, 98.0, 100.0], dtype=float)
    close = np.array([100.0, 99.0, 100.0], dtype=float)
    atr = np.array([1.0, 1.0, 1.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
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

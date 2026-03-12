from __future__ import annotations

import numpy as np
import pytest
from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
    compute_highlow_or_atr_tightest_stop_targets_and_rr,
)


def test_highlow_or_atr_tightest_stop_uses_tighter_stop() -> None:
    # Entry is at close[0]=100. ATR=1. With the tightest-stop variant:
    # - stop = max(low=99.5, entry-atr=99.0) = 99.5
    # - tp = entry + 2*atr = 102.0
    # Next bar hits TP => reward=2.0, risk=0.5 => RR=4.0.
    open_ = np.array([99.0, 100.0, 100.0], dtype=float)
    high = np.array([100.5, 102.0, 100.0], dtype=float)
    low = np.array([99.5, 99.75, 100.0], dtype=float)
    close = np.array([100.0, 101.0, 100.0], dtype=float)
    atr = np.array([1.0, 1.0, 1.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_or_atr_tightest_stop_targets_and_rr(
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
    assert float(rr_long[0]) == pytest.approx(4.0)
    assert np.isnan(rr_short[0])

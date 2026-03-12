from __future__ import annotations

import numpy as np
import pytest
from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
    compute_highlow_or_atr_targets_and_rr,
)


def test_highlow_or_atr_cutoff_keeps_post_period_entries() -> None:
    # Mirrors the Rust unit test:
    # highlow_or_atr_populates_rr_after_cutoff_horizon_for_post_period_entries
    open_ = np.array([99.0, 100.0, 99.0, 100.0], dtype=float)
    high = np.array([100.5, 101.0, 100.5, 102.0], dtype=float)
    low = np.array([99.0, 99.5, 99.0, 99.5], dtype=float)
    close = np.array([100.0, 100.5, 100.0, 101.0], dtype=float)
    atr = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    long, short, rr_long, rr_short = compute_highlow_or_atr_targets_and_rr(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr,
        tick_size=None,
        resolve_end_idx=1,
    )

    assert bool(long[0]) is False
    assert bool(short[0]) is False
    assert float(rr_long[0]) == pytest.approx(0.5)
    assert np.isnan(rr_short[0])

    assert bool(long[1]) is False
    assert np.isnan(rr_long[1])
    assert np.isnan(rr_short[1])

    assert bool(long[2]) is True
    assert bool(short[2]) is False
    assert float(rr_long[2]) == pytest.approx(2.0)
    assert np.isnan(rr_short[2])

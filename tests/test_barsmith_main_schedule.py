from __future__ import annotations

from datetime import datetime, timezone

import pytest
from algotrade.core import types
from main import compute_barsmith_next_run


def test_compute_barsmith_next_run_m30() -> None:
    now = datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
    next_bar, run_at = compute_barsmith_next_run(
        timeframe=types.Timeframe.m30,
        contract=types.SupportedContract.MES,
        now=now,
        close_delay_seconds=3.0,
    )
    assert next_bar == datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    assert run_at == datetime(2026, 1, 1, 13, 0, 3, tzinfo=timezone.utc)


def test_compute_barsmith_next_run_rejects_negative_delay() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        compute_barsmith_next_run(
            timeframe=types.Timeframe.m30,
            contract=types.SupportedContract.MES,
            now=now,
            close_delay_seconds=-1.0,
        )

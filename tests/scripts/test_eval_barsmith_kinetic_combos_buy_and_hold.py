import runpy

import pandas as pd
import pytest


def _load_compute_buy_and_hold_benchmark():
    mod = runpy.run_path("tools/eval_barsmith_kinetic_combos.py")
    return mod["compute_buy_and_hold_benchmark"]


def test_compute_buy_and_hold_benchmark_reports_return_and_max_dd() -> None:
    compute_buy_and_hold_benchmark = _load_compute_buy_and_hold_benchmark()

    df = pd.DataFrame(
        {
            "timestamp": [
                "2021-01-01 00:00:00+00:00",
                "2021-04-01 00:00:00+00:00",
                "2021-07-01 00:00:00+00:00",
                "2022-01-01 00:00:00+00:00",
            ],
            "close": [100.0, 120.0, 90.0, 130.0],
        }
    )
    mask = pd.Series([True, True, True, True], index=df.index)

    got = compute_buy_and_hold_benchmark(df, mask, capital_dollar=100_000.0)

    assert got["start"] == "2021-01-01 00:00:00"
    assert got["end"] == "2022-01-01 00:00:00"
    assert got["start_close"] == pytest.approx(100.0, abs=1e-12)
    assert got["end_close"] == pytest.approx(130.0, abs=1e-12)
    assert got["total_return_pct"] == pytest.approx(30.0, rel=1e-12)
    assert got["max_drawdown_pct"] == pytest.approx(25.0, rel=1e-12)

    assert float(got["cagr_pct"]) > 0.0
    assert got["calmar"] == pytest.approx(float(got["cagr_pct"]) / 25.0, rel=1e-12)
    assert got["final_capital"] == pytest.approx(130_000.0, rel=1e-12)

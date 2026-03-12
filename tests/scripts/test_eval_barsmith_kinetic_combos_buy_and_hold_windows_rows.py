import runpy

import pandas as pd
import pytest


def _load_impl():
    mod = runpy.run_path("tools/eval_barsmith_kinetic_combos.py")
    return mod["build_calendar_year_windows"], mod["build_buy_and_hold_window_rows"]


def test_build_buy_and_hold_window_rows_emits_one_row_per_year() -> None:
    build_calendar_year_windows, build_buy_and_hold_window_rows = _load_impl()

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-01-01 00:00:00+00:00",
                "2020-06-01 00:00:00+00:00",
                "2020-12-31 00:00:00+00:00",
                "2021-01-01 00:00:00+00:00",
                "2021-06-01 00:00:00+00:00",
                "2021-12-31 00:00:00+00:00",
            ],
            "close": [100.0, 120.0, 110.0, 200.0, 160.0, 180.0],
        }
    )
    all_mask = pd.Series([True] * len(df), index=df.index)
    windows = build_calendar_year_windows(df["timestamp"], all_mask)

    rows = build_buy_and_hold_window_rows(df, windows, scope="all", capital_dollar=100.0)
    assert len(rows) == 2

    years = sorted(int(r["year"]) for r in rows)
    assert years == [2020, 2021]

    rows_by_year = {int(r["year"]): r for r in rows}
    y2020 = rows_by_year[2020]
    assert y2020["expr"] == "__BUY_AND_HOLD__"
    assert float(y2020["total_return_pct"]) == pytest.approx(10.0, rel=1e-12)
    assert float(y2020["final_capital"]) == pytest.approx(110.0, rel=1e-12)
    assert float(y2020["max_drawdown_pct_equity"]) == pytest.approx((120.0 - 110.0) / 120.0 * 100.0, rel=1e-12)

    y2021 = rows_by_year[2021]
    assert float(y2021["total_return_pct"]) == pytest.approx(-10.0, rel=1e-12)
    assert float(y2021["final_capital"]) == pytest.approx(90.0, rel=1e-12)
    assert float(y2021["max_drawdown_pct_equity"]) == pytest.approx(20.0, rel=1e-12)

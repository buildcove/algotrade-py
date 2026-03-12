from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
from experiments.analysis.entry_rotation import (
    EntryRotationConfig,
    SharadarPaths,
    _annual_allocation_whole_shares,
    _annual_contribution_dates,
    _annual_sim_end_date,
    _bench_date_on_or_after,
    _benchmark_buy_hold_calendar_year_returns,
    _calendar_yearly_stats,
    _count_opened_new_names_for_year,
    _count_reserved_new_names_for_year,
    _dedupe_latest_lastupdated,
    _ensure_tickers_cache,
    _implied_div_cash_per_share_from_closeadj,
    _issuer_key_from_candidates,
    _monthly_sim_end_date,
    _next_select_date_after,
    _pick_first_unheld,
    _position_mae_pct_from_cache,
    _rebalance_periods_stats,
    _reentry_dca_init_from_cash,
    _reentry_dca_release,
    _release_reserved_new_name_for_year,
    _reserve_new_name_for_year,
    _resolve_sell_fill,
    _select_top_n,
    _sell_price_no_forward,
    _stopout_resume_month_idx,
    _twr_from_equity_curve,
    _whole_shares_for_spendable_cash,
    _xirr,
    parse_cli_number,
    select_snapshot_for_date,
)


def test_pick_first_unheld_skips_held() -> None:
    ranked = pd.DataFrame(
        {
            "symbol_key": ["SFP:AAA", "SFP:BBB", "SFP:CCC"],
            "entry_open_adj": [10.0, 20.0, 30.0],
        }
    )
    sym, px = _pick_first_unheld(ranked, {"SFP:AAA", "SFP:BBB"})
    assert sym == "SFP:CCC"
    assert px == 30.0


def test_select_top_n_applies_gates_and_sorts() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=1,
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B", "SFP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [20_000.0, float("nan"), 20_000.0],
            "marketcap_missing": [False, True, False],
            "dollar_volume_median": [0.0, 200.0, 0.0],
            "cagr_5y": [0.20, 0.30, 0.10],
            "worst_past_60m_mae_12m": [-0.30, -0.20, -0.40],
            "n_past_mae_12m": [60.0, 60.0, 60.0],
            "worst_past_5y_mae_12m": [-0.30, -0.20, -0.40],
            "n_past_5y_mae_12m": [5.0, 5.0, 5.0],
            "adverse_mag": [0.30, 0.20, 0.40],
            "cagr_to_mae_5y": [0.20 / 0.30, 0.30 / 0.20, 0.10 / 0.40],
            "last_n_years_green": [True, True, False],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:B", "SFP:A"]


def test_implied_div_cash_per_share_from_closeadj_infers_dividend_step() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "close_raw": [100.0, 100.0],
            "close_adj": [100.0, 101.0],
        }
    )
    div = _implied_div_cash_per_share_from_closeadj(df)
    assert div.tolist() == pytest.approx([0.0, 1.0])


def test_implied_div_cash_per_share_from_closeadj_clips_negative_to_zero() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "close_raw": [100.0, 100.0],
            "close_adj": [100.0, 99.0],
        }
    )
    div = _implied_div_cash_per_share_from_closeadj(df)
    assert div.tolist() == [0.0, 0.0]


def test_dividend_withholding_requires_raw_price_mode() -> None:
    with pytest.raises(ValueError, match="dividend-withholding-rate requires --price-mode raw"):
        EntryRotationConfig(
            mode="annual-lumpsum",
            price_mode="adj",
            start_year=2020,
            end_year=2021,
            sharadar_dir=Path("tmp"),
            cache_dir=Path("tmp"),
            output_dir=Path("tmp"),
            dividend_withholding_rate=0.25,
        )


def test_calendar_yearly_stats_includes_dividend_sums() -> None:
    equity_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-12-31"]),
            "equity": [100.0, 110.0],
            "equity_low": [95.0, 105.0],
            "equity_low_pre_actions": [98.0, 108.0],
            "div_gross": [1.0, 2.0],
            "div_tax": [0.25, 0.50],
            "div_net": [0.75, 1.50],
        }
    )
    cal = _calendar_yearly_stats(equity_df)
    row = cal[cal["year"] == 2020].iloc[0]
    assert row["div_gross_sum"] == pytest.approx(3.0)
    assert row["div_tax_sum"] == pytest.approx(0.75)
    assert row["div_net_sum"] == pytest.approx(2.25)


def test_xirr_matches_simple_annual_growth() -> None:
    # -100 today, +110 one year later -> 10% IRR
    d0 = pd.Timestamp("2020-01-01")
    d1 = pd.Timestamp("2021-01-01")
    irr = _xirr(dates=[d0, d1], amounts=[-100.0, 110.0])
    expected = (1.1 ** (365.25 / 366.0)) - 1.0  # 2020 is a leap year
    assert irr == pytest.approx(expected, abs=1e-6)


def test_twr_ignores_contribution_and_matches_return() -> None:
    # Day0 equity 100.
    # Day1: add 100 at start-of-day, no performance -> equity 200.
    # Day2: 10% performance -> equity 220.
    eq = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "equity": [100.0, 200.0, 220.0],
        }
    )
    cash_flows = pd.DataFrame({"date": [pd.Timestamp("2020-01-02")], "amount": [-100.0]})  # investor contribution
    twr_total, twr_cagr = _twr_from_equity_curve(dates=eq["date"], equity=eq["equity"], cash_flows=cash_flows)
    assert twr_total == pytest.approx(0.10, abs=1e-9)
    # Annualization over 2 calendar days yields a very large CAGR; check it matches the formula.
    years = 2.0 / 365.25
    expected_cagr = (1.0 + twr_total) ** (1.0 / years) - 1.0
    assert twr_cagr == pytest.approx(expected_cagr, rel=1e-9)


def test_select_top_n_calmar_rank_metric_changes_sorting() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=0,
        rank_metric="calmar",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B", "SFP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [20_000.0, 20_000.0, 20_000.0],
            "marketcap_missing": [False, False, False],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "cagr_5y": [0.20, 0.20, 0.20],
            "worst_past_5y_mae_12m": [-0.30, -0.30, -0.30],
            "n_past_5y_mae_12m": [5.0, 5.0, 5.0],
            "adverse_mag": [0.30, 0.30, 0.30],
            "cagr_to_mae_5y": [0.20 / 0.30, 0.20 / 0.30, 0.20 / 0.30],
            "max_dd_5y": [-0.50, -0.25, -0.40],
            "calmar_5y": [0.40, 0.80, 0.50],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:B", "SFP:C"]


def test_select_top_n_rank_lookback_years_3_uses_3y_columns() -> None:
    base = dict(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=1,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        require_last_n_years_green=0,
        rank_metric="calmar",
    )
    cfg3 = EntryRotationConfig(**base, rank_lookback_years=3)
    cfg5 = EntryRotationConfig(**base, rank_lookback_years=5)

    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [True, True],
            "is_sfp_blacklisted": [False, False],
            "marketcap_entry": [20_000.0, 20_000.0],
            "marketcap_missing": [False, False],
            "dollar_volume_median": [0.0, 0.0],
            # 3y signals say A is better, 5y signals say B is better.
            "cagr_3y": [0.30, 0.30],
            "max_dd_3y": [-0.20, -0.20],
            "calmar_3y": [1.50, 1.00],
            "cagr_5y": [0.30, 0.30],
            "max_dd_5y": [-0.20, -0.20],
            "calmar_5y": [0.50, 2.00],
            "last_n_years_green": [True, True],
        }
    )
    assert list(_select_top_n(df, cfg3)["symbol_key"]) == ["SFP:A"]
    assert list(_select_top_n(df, cfg5)["symbol_key"]) == ["SFP:B"]


def test_select_top_n_rank_lookback_years_3_requires_3y_columns() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=1,
        require_last_n_years_green=0,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        rank_metric="calmar",
        rank_lookback_years=3,
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A"],
            "instrument_id": ["PT1"],
            "instrument_id_is_fallback": [False],
            "permaticker": pd.Series([1], dtype="Int64"),
            "entry_open_adj": [10.0],
            "has_open": [True],
            "is_equity_asof": [True],
            "is_sfp_blacklisted": [False],
            "marketcap_entry": [20_000.0],
            "marketcap_missing": [False],
            "dollar_volume_median": [0.0],
            # Only 5y columns exist; 3y lookback should raise a clear error.
            "cagr_5y": [0.30],
            "max_dd_5y": [-0.20],
            "calmar_5y": [1.0],
            "last_n_years_green": [True],
        }
    )
    with pytest.raises(ValueError, match="rank-lookback-years 3"):
        _select_top_n(df, cfg)


def test_select_top_n_cagr_to_mae_median_rank_metric_changes_sorting() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=0,
        rank_metric="cagr_to_mae_median",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B", "SFP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [20_000.0, 20_000.0, 20_000.0],
            "marketcap_missing": [False, False, False],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "cagr_5y": [0.20, 0.20, 0.20],
            "worst_past_5y_mae_12m": [-0.50, -0.20, -0.30],
            "median_past_5y_mae_12m": [-0.10, -0.20, -0.30],
            "n_past_5y_mae_12m": [5.0, 5.0, 5.0],
            "adverse_mag": [0.50, 0.20, 0.30],
            "adverse_median_mag": [0.10, 0.20, 0.30],
            "cagr_to_mae_5y": [0.20 / 0.50, 0.20 / 0.20, 0.20 / 0.30],
            "cagr_to_mae_median_5y": [0.20 / 0.10, 0.20 / 0.20, 0.20 / 0.30],
            "max_dd_5y": [-0.50, -0.50, -0.50],
            "calmar_5y": [0.40, 0.40, 0.40],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:A", "SFP:B"]


def test_select_top_n_prev_12m_return_rank_metric_changes_sorting() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=0,
        rank_metric="prev_12m_return",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B", "SFP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [20_000.0, 20_000.0, 20_000.0],
            "marketcap_missing": [False, False, False],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "prev_12m_return": [0.10, 0.30, -0.05],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:B", "SFP:A"]


def test_select_top_n_cagr_5y_rank_metric_changes_sorting() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=0,
        rank_metric="cagr_5y",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B", "SFP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [20_000.0, 20_000.0, 20_000.0],
            "marketcap_missing": [False, False, False],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "cagr_5y": [0.20, 0.30, 0.10],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:B", "SFP:A"]


def test_select_top_n_cagr_to_trailing_low_5y_rank_metric_changes_sorting() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        require_last_n_years_green=0,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        rank_metric="cagr_to_trailing_low_5y",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SEP:A", "SEP:B", "SEP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "category_asof": ["Domestic Common Stock", "Domestic Common Stock", "Domestic Common Stock"],
            "name_asof": ["A Inc", "B Inc", "C Inc"],
            "marketcap_entry": [10_000.0, 10_000.0, 10_000.0],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "cagr_5y_asof_close": [0.20, 0.10, 0.30],
            "trailing_low_dd_mag_5y": [0.40, 0.10, 0.60],
            "cagr_to_trailing_low_5y": [0.20 / 0.40, 0.10 / 0.10, 0.30 / 0.60],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SEP:B", "SEP:A"]


def test_select_top_n_cagr_to_trailing_low_5y_daily_rank_metric_changes_sorting() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=2,
        require_last_n_years_green=0,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        rank_metric="cagr_to_trailing_low_5y_daily",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SEP:A", "SEP:B", "SEP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "category_asof": ["Domestic Common Stock", "Domestic Common Stock", "Domestic Common Stock"],
            "name_asof": ["A Inc", "B Inc", "C Inc"],
            "marketcap_entry": [10_000.0, 10_000.0, 10_000.0],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "cagr_5y_asof_close": [0.20, 0.10, 0.30],
            "trailing_low_dd_mag_5y": [0.40, 0.10, 0.60],
            "cagr_to_trailing_low_5y": [0.20 / 0.40, 0.10 / 0.10, 0.30 / 0.60],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SEP:B", "SEP:A"]


def test_select_top_n_excludes_lp_by_default_and_allows_override() -> None:
    df = pd.DataFrame(
        {
            "symbol_key": ["SEP:LP1", "SEP:CS1"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [True, True],
            "is_sfp_blacklisted": [False, False],
            "category_asof": ["Domestic Common Stock", "Domestic Common Stock"],
            "name_asof": ["Example Holdings, L.P.", "Example Holdings Inc"],
            "dollar_volume_median": [100_000_000.0, 100_000_000.0],
            "marketcap_entry": [10_000.0, 10_000.0],
            "cagr_5y": [0.50, 0.10],
            "worst_past_5y_mae_12m": [-0.25, -0.25],
            "n_past_5y_mae_12m": [5.0, 5.0],
            "adverse_mag": [0.25, 0.25],
            "cagr_to_mae_5y": [0.50 / 0.25, 0.10 / 0.25],
            "last_n_years_green": [True, True],
        }
    )

    cfg_default = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=1,
        require_last_n_years_green=0,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
    )
    selected_default = _select_top_n(df, cfg_default)
    assert list(selected_default["symbol_key"]) == ["SEP:CS1"]

    cfg_allow = dataclasses.replace(cfg_default, exclude_lp=False)
    selected_allow = _select_top_n(df, cfg_allow)
    assert list(selected_allow["symbol_key"]) == ["SEP:LP1"]


def test_exclude_lp_does_not_filter_sfp_etfs_with_lp_in_name() -> None:
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:USO", "SEP:LP_EQ"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "table": ["SFP", "SEP"],
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [False, True],
            "is_sfp_blacklisted": [False, False],
            "category_asof": ["ETF", "Domestic Common Stock"],
            "name_asof": ["UNITED STATES OIL FUND LP", "Example Midstream Holdings, L.P."],
            "dollar_volume_median": [100_000_000.0, 100_000_000.0],
            "marketcap_entry": [float("nan"), 10_000.0],
            "marketcap_missing": [True, False],
            "cagr_5y": [0.50, 0.40],
            "worst_past_5y_mae_12m": [-0.25, -0.25],
            "n_past_5y_mae_12m": [5.0, 5.0],
            "adverse_mag": [0.25, 0.25],
            "cagr_to_mae_5y": [2.0, 1.6],
            "last_n_years_green": [True, True],
        }
    )
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        require_last_n_years_green=0,
        universe="all",
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        exclude_lp=True,
    )
    selected = _select_top_n(df, cfg)
    assert "SFP:USO" in set(selected["symbol_key"])
    assert "SEP:LP_EQ" not in set(selected["symbol_key"])


def test_issuer_key_uses_name_then_fallback_to_base_ticker() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["TEN-PE", "TEN-PF", "BRK-B", "BRK-A", "NO_NAME-A"],
            "name_asof": [
                "Tsakos Energy Navigation Ltd",
                "TSAKOS ENERGY NAVIGATION LTD",
                "Berkshire Hathaway Inc",
                "Berkshire Hathaway Inc",
                "",
            ],
        }
    )
    key = _issuer_key_from_candidates(df)
    assert key.iloc[0] == key.iloc[1]
    assert key.iloc[2] == key.iloc[3]
    assert key.iloc[4] == "NO_NAME"


def test_select_top_n_dedupes_issuer_by_default_and_allows_override() -> None:
    base = {
        "symbol_key": ["SEP:BRK-A", "SEP:BRK-B", "SEP:FIX"],
        "ticker": ["BRK-A", "BRK-B", "FIX"],
        "table": ["SEP", "SEP", "SEP"],
        "instrument_id": ["PT1", "PT2", "PT3"],
        "instrument_id_is_fallback": [False, False, False],
        "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
        "entry_open_adj": [10.0, 10.0, 10.0],
        "has_open": [True, True, True],
        "is_equity_asof": [True, True, True],
        "is_sfp_blacklisted": [False, False, False],
        "category_asof": ["Domestic Common Stock", "Domestic Common Stock", "Domestic Common Stock"],
        "name_asof": ["BERKSHIRE HATHAWAY INC", "BERKSHIRE HATHAWAY INC", "COMFORT SYSTEMS USA INC"],
        "marketcap_entry": [10_000.0, 10_000.0, 10_000.0],
        "dollar_volume_median": [0.0, 0.0, 0.0],
        "cagr_5y": [0.20, 0.19, 0.10],
        "worst_past_5y_mae_12m": [-0.20, -0.20, -0.20],
        "n_past_5y_mae_12m": [5.0, 5.0, 5.0],
        "adverse_mag": [0.20, 0.20, 0.20],
        "cagr_to_mae_5y": [1.0, 0.95, 0.5],
        "last_n_years_green": [True, True, True],
    }
    df = pd.DataFrame(base)
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=3,
        require_last_n_years_green=0,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        rank_metric="cagr_to_mae",
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["ticker"]) == ["BRK-A", "FIX"]
    assert "issuer_key" in selected.columns

    cfg2 = dataclasses.replace(cfg, dedupe_issuer=False)
    selected2 = _select_top_n(df, cfg2)
    assert list(selected2["ticker"]) == ["BRK-A", "BRK-B", "FIX"]


def test_select_top_n_excludes_preferred_quasi_and_adr_common_by_default() -> None:
    df = pd.DataFrame(
        {
            "symbol_key": ["SEP:PREF", "SEP:DEP", "SEP:ADR", "SEP:COM"],
            "ticker": ["PREF-A", "DEP-A", "ADR", "COM"],
            "table": ["SEP", "SEP", "SEP", "SEP"],
            "instrument_id": ["PT1", "PT2", "PT3", "PT4"],
            "instrument_id_is_fallback": [False, False, False, False],
            "permaticker": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0, 10.0],
            "has_open": [True, True, True, True],
            "is_equity_asof": [True, True, True, True],
            "is_sfp_blacklisted": [False, False, False, False],
            "category_asof": ["ADR Preferred Stock", "Domestic Depositary Shares", "ADR Common Stock", "Domestic Common Stock"],
            "name_asof": ["PREF CO", "DEP CO", "ADR CO", "COM CO"],
            "marketcap_entry": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            "dollar_volume_median": [0.0, 0.0, 0.0, 0.0],
            "cagr_5y": [0.50, 0.40, 0.30, 0.10],
            "worst_past_5y_mae_12m": [-0.20, -0.20, -0.20, -0.20],
            "n_past_5y_mae_12m": [5.0, 5.0, 5.0, 5.0],
            "adverse_mag": [0.20, 0.20, 0.20, 0.20],
            "cagr_to_mae_5y": [2.5, 2.0, 1.5, 0.5],
            "last_n_years_green": [True, True, True, True],
        }
    )
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        require_last_n_years_green=0,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        rank_metric="cagr_to_mae",
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["ticker"]) == ["COM"]

    cfg2 = dataclasses.replace(cfg, exclude_preferred=False, exclude_quasi_preferred=False, exclude_adr_common=False)
    selected2 = _select_top_n(df, cfg2)
    assert list(selected2["ticker"]) == ["PREF-A", "DEP-A", "ADR", "COM"]


def test_calendar_yearly_stats_uses_pre_actions_low_when_present() -> None:
    equity_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-12-31",
                    "2021-01-04",
                    "2021-12-31",
                ]
            ),
            "equity": [100.0, 110.0, 120.0, 120.0, 150.0],
            "equity_low": [95.0, 100.0, 110.0, 118.0, 130.0],
            # pre-actions low is worse in 2020 than post-actions, and equal in 2021
            "equity_low_pre_actions": [90.0, 95.0, 105.0, 118.0, 130.0],
            "bm_equity": [100.0, 105.0, 115.0, 115.0, 140.0],
            "bm_equity_low": [98.0, 102.0, 110.0, 113.0, 120.0],
            "bm_equity_low_pre_actions": [97.0, 101.0, 109.0, 113.0, 120.0],
        }
    )
    out = _calendar_yearly_stats(equity_df)
    row2020 = out[out["year"] == 2020].iloc[0]
    # start equity for 2020 is 100, pre-actions min low is 90 => -10%
    assert float(row2020["mae_pre_actions_pct"]) == pytest.approx(-0.10)
    # post-actions min low is 95 => -5%
    assert float(row2020["mae_post_actions_pct"]) == pytest.approx(-0.05)


def test_stopout_resume_month_idx_zero_means_next_month_bucket() -> None:
    assert _stopout_resume_month_idx(130, 0) == 131
    assert _stopout_resume_month_idx(130, 1) == 131
    assert _stopout_resume_month_idx(130, 12) == 142


def test_reentry_dca_init_from_cash_and_release_sums_to_cash() -> None:
    remaining, months_left, budget = _reentry_dca_init_from_cash(1200.0, 12)
    assert remaining == pytest.approx(1200.0)
    assert months_left == 12
    assert budget == pytest.approx(0.0)

    released_total = 0.0
    for _ in range(12):
        released, remaining, months_left = _reentry_dca_release(remaining, months_left)
        released_total += released
    assert months_left == 0
    assert remaining == pytest.approx(0.0)
    assert released_total == pytest.approx(1200.0)


def test_reentry_dca_init_disabled_returns_infinite_budget() -> None:
    remaining, months_left, budget = _reentry_dca_init_from_cash(1000.0, 0)
    assert remaining == 0.0
    assert months_left == 0
    assert budget == float("inf")


def test_stopout_reentry_dca_months_rejected_for_annual_mode() -> None:
    with pytest.raises(ValueError, match="stopout-reentry-dca-months"):
        EntryRotationConfig(
            mode="annual-lumpsum",
            price_mode="adj",
            start_year=2020,
            end_year=2021,
            sharadar_dir=Path("tmp"),
            cache_dir=Path("tmp"),
            output_dir=Path("tmp"),
            stopout_reentry_dca_months=12,
        )


def test_initial_lump_flags_rejected_for_annual_mode() -> None:
    with pytest.raises(ValueError, match="initial-lump"):
        EntryRotationConfig(
            mode="annual-lumpsum",
            price_mode="adj",
            start_year=2020,
            end_year=2021,
            sharadar_dir=Path("tmp"),
            cache_dir=Path("tmp"),
            output_dir=Path("tmp"),
            initial_lump_sum=1000.0,
        )
    with pytest.raises(ValueError, match="initial-lump"):
        EntryRotationConfig(
            mode="annual-lumpsum",
            price_mode="adj",
            start_year=2020,
            end_year=2021,
            sharadar_dir=Path("tmp"),
            cache_dir=Path("tmp"),
            output_dir=Path("tmp"),
            initial_lump_dca_months=12,
        )


def test_select_top_n_applies_min_filters_and_combines_with_and() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=0,
        rank_metric="prev_12m_return",
        min_filters=(("prev_12m_return", 0.05), ("cagr_5y", 0.10)),
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B", "SFP:C"],
            "instrument_id": ["PT1", "PT2", "PT3"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 2, 3], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [20_000.0, 20_000.0, 20_000.0],
            "marketcap_missing": [False, False, False],
            "dollar_volume_median": [0.0, 0.0, 0.0],
            "prev_12m_return": [0.06, 0.10, 0.20],
            "cagr_5y": [0.09, 0.11, 0.20],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:C", "SFP:B"]


def test_select_top_n_min_filter_max_dd_works_as_drawdown_cap() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        require_last_n_years_green=0,
        rank_metric="calmar",
        rank_lookback_years=3,
        min_filters=(("max_dd_3y", -0.20),),
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:OK", "SFP:BAD"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [True, True],
            "is_sfp_blacklisted": [False, False],
            "marketcap_entry": [20_000.0, 20_000.0],
            "marketcap_missing": [False, False],
            "dollar_volume_median": [0.0, 0.0],
            "cagr_3y": [0.30, 0.30],
            "max_dd_3y": [-0.10, -0.30],  # BAD fails: -30% < -20%
            "calmar_3y": [3.0, 3.0],
            "last_n_years_green": [True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert set(selected["symbol_key"]) == {"SFP:OK"}


def test_parse_cli_number_accepts_underscores_and_commas() -> None:
    assert parse_cli_number("10_000_000") == 10_000_000.0
    assert parse_cli_number("10,000,000") == 10_000_000.0
    assert parse_cli_number("1e7") == 10_000_000.0
    assert parse_cli_number(123) == 123.0


def test_select_top_n_applies_global_dollar_volume_floor() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=0.0,
        dollar_volume_floor=100.0,
        dollar_volume_floor_missing_mcap=0.0,
        require_last_n_years_green=0,
        rank_metric="prev_12m_return",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:B"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [True, True],
            "is_sfp_blacklisted": [False, False],
            "marketcap_entry": [float("nan"), float("nan")],
            "marketcap_missing": [True, True],
            "dollar_volume_median": [50.0, 150.0],
            "prev_12m_return": [0.10, 0.05],
            "last_n_years_green": [True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:B"]


def test_select_top_n_mcap_floor_zero_disables_missing_mcap_fallback() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
        # Very high fallback should not matter when mcap_floor==0.
        dollar_volume_floor_missing_mcap=1_000_000_000.0,
        require_last_n_years_green=0,
        rank_metric="prev_12m_return",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A"],
            "instrument_id": ["PT1"],
            "instrument_id_is_fallback": [False],
            "permaticker": pd.Series([1], dtype="Int64"),
            "entry_open_adj": [10.0],
            "has_open": [True],
            "is_equity_asof": [True],
            "is_sfp_blacklisted": [False],
            "marketcap_entry": [float("nan")],
            "marketcap_missing": [True],
            "dollar_volume_median": [1.0],
            "prev_12m_return": [0.10],
            "last_n_years_green": [True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:A"]


def test_select_top_n_mcap_floor_uses_missing_mcap_fallback() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=10_000_000_000.0,
        dollar_volume_floor=0.0,
        dollar_volume_floor_missing_mcap=100.0,
        require_last_n_years_green=0,
        rank_metric="prev_12m_return",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:LOWDV", "SFP:HIDV"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [True, True],
            "is_sfp_blacklisted": [False, False],
            "marketcap_entry": [float("nan"), float("nan")],
            "marketcap_missing": [True, True],
            "dollar_volume_median": [50.0, 150.0],
            "prev_12m_return": [0.20, 0.10],
            "last_n_years_green": [True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:HIDV"]


def test_annual_contribution_dates_excludes_start_includes_end() -> None:
    month_index = pd.DataFrame(
        {
            "select_date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-02-03",
                    "2025-03-03",
                    "2025-04-01",
                    "2025-05-01",
                    "2025-06-02",
                    "2025-07-01",
                    "2025-08-01",
                    "2025-09-02",
                    "2025-10-01",
                    "2025-11-03",
                    "2025-12-01",
                    "2026-01-02",
                ]
            )
        }
    )
    start = pd.Timestamp("2025-01-02")
    end = pd.Timestamp("2026-01-02")
    contrib = _annual_contribution_dates(month_index, start_date=start, end_date=end)
    assert len(contrib) == 12
    assert pd.Timestamp("2025-01-02") not in set(contrib)
    assert pd.Timestamp("2026-01-02") in set(contrib)


def test_config_allows_annual_monthly_contribution_but_rejects_negative() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        monthly_dca_amount=100.0,
    )
    assert cfg.monthly_dca_amount == 100.0
    with pytest.raises(ValueError, match="monthly-dca-amount must be >= 0"):
        EntryRotationConfig(
            mode="annual-lumpsum",
            price_mode="adj",
            start_year=2020,
            end_year=2021,
            sharadar_dir=Path("tmp"),
            cache_dir=Path("tmp"),
            output_dir=Path("tmp"),
            monthly_dca_amount=-1.0,
        )


def test_calendar_yearly_stats_exposes_pre_and_post_actions_mae() -> None:
    equity_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "equity": [100.0, 101.0, 102.0],
            # Post-actions low path (e.g., after stopout/cash sweep).
            "equity_low": [95.0, 96.0, 94.0],
            # Pre-actions low path (before MAE-triggered liquidations).
            "equity_low_pre_actions": [80.0, 85.0, 82.0],
        }
    )
    out = _calendar_yearly_stats(equity_df)
    row = out.iloc[0]
    assert row["year"] == 2020
    assert row["mae_pre_actions_pct"] == pytest.approx(80.0 / 100.0 - 1.0)
    assert row["mae_post_actions_pct"] == pytest.approx(94.0 / 100.0 - 1.0)


def test_rebalance_periods_stats_exposes_pre_and_post_actions_mae() -> None:
    equity_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "equity": [100.0, 101.0, 102.0],
            "equity_low": [95.0, 96.0, 94.0],
            "equity_low_pre_actions": [80.0, 85.0, 82.0],
        }
    )
    rebalance_dates = {2020: pd.Timestamp("2020-01-02")}
    out = _rebalance_periods_stats(equity_df, rebalance_dates, start_year=2020, end_year=2020)
    row = out.iloc[0]
    assert row["period_id"] == 2020
    assert row["mae_pre_actions_pct"] == pytest.approx(80.0 / 100.0 - 1.0)
    assert row["mae_post_actions_pct"] == pytest.approx(94.0 / 100.0 - 1.0)


def test_annual_allocation_whole_shares_is_deterministic() -> None:
    ranked = pd.DataFrame(
        {
            "symbol_key": ["SFP:AAA", "SFP:BBB"],
            "entry_open_adj": [10.0, 20.0],
        }
    )
    buys = _annual_allocation_whole_shares(ranked, cash_available=100.0, commission_per_trade=0.0, slippage_bps=0.0)
    assert buys == {"SFP:AAA": 6, "SFP:BBB": 2}


def test_dedupe_latest_lastupdated_keeps_latest_row() -> None:
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A", "SFP:A", "SFP:A"],
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-02"]),
            "lastupdated": pd.to_datetime(["2020-01-10", "2020-01-20", "2020-01-15"]),
            "value": [1, 2, 3],
        }
    )
    out = _dedupe_latest_lastupdated(df, keys=["symbol_key", "date"], lastupdated_col="lastupdated")
    assert len(out) == 1
    assert int(out.iloc[0]["value"]) == 2


def test_sell_price_no_forward_never_looks_ahead() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-10"]),
            "open_adj": [10.0, 12.0],
            "close_adj": [10.5, 12.5],
        }
    )
    assert _sell_price_no_forward(df, pd.Timestamp("2020-01-05"), open_col="open_adj", close_col="close_adj") is None


def test_resolve_sell_fill_no_forward_cash_date_is_requested_date() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-10"]),
            "open_adj": [10.0, 12.0],
            "close_adj": [10.5, 12.5],
        }
    )
    fill = _resolve_sell_fill(df, pd.Timestamp("2020-01-10"), policy="no-forward", open_col="open_adj", close_col="close_adj")
    assert fill is not None
    assert fill.cash_date == pd.Timestamp("2020-01-10")
    assert fill.price_date == pd.Timestamp("2020-01-10")
    assert fill.raw_price == 12.0


def test_resolve_sell_fill_no_forward_delist_fallback_uses_last_close() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "open_adj": [10.0],
            "close_adj": [10.5],
        }
    )
    fill = _resolve_sell_fill(df, pd.Timestamp("2020-01-10"), policy="no-forward", open_col="open_adj", close_col="close_adj")
    assert fill is not None
    assert fill.cash_date == pd.Timestamp("2020-01-10")
    assert fill.price_date == pd.Timestamp("2020-01-02")
    assert fill.raw_price == 10.5


def test_resolve_sell_fill_defer_forward_executes_on_future_close() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-10"]),
            "open_adj": [10.0, float("nan")],
            "close_adj": [10.5, 12.5],
        }
    )
    fill = _resolve_sell_fill(df, pd.Timestamp("2020-01-05"), policy="defer-forward", open_col="open_adj", close_col="close_adj")
    assert fill is not None
    assert fill.cash_date == pd.Timestamp("2020-01-10")
    assert fill.price_date == pd.Timestamp("2020-01-10")
    assert fill.raw_price == 12.5


def test_resolve_sell_fill_defer_forward_clamps_after_final_day() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-10"]),
            "open_adj": [10.0, float("nan")],
            "close_adj": [10.5, 12.5],
        }
    )
    fill = _resolve_sell_fill(
        df,
        pd.Timestamp("2020-01-05"),
        policy="defer-forward",
        open_col="open_adj",
        close_col="close_adj",
        final_day=pd.Timestamp("2020-01-05"),
    )
    assert fill is None


def test_select_top_n_dedupes_by_instrument_id() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=0.0,
        dollar_volume_floor_missing_mcap=0.0,
        require_last_n_years_green=0,
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:A1", "SFP:A2", "SFP:B"],
            "instrument_id": ["PT1", "PT1", "PT2"],
            "instrument_id_is_fallback": [False, False, False],
            "permaticker": pd.Series([1, 1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0, 10.0],
            "has_open": [True, True, True],
            "is_equity_asof": [True, True, True],
            "is_sfp_blacklisted": [False, False, False],
            "marketcap_entry": [100.0, 100.0, 100.0],
            "marketcap_missing": [False, False, False],
            "dollar_volume_median": [1.0, 1.0, 1.0],
            "cagr_5y": [0.30, 0.20, 0.10],
            "worst_past_60m_mae_12m": [-0.30, -0.30, -0.30],
            "n_past_mae_12m": [60.0, 60.0, 60.0],
            "worst_past_5y_mae_12m": [-0.30, -0.30, -0.30],
            "n_past_5y_mae_12m": [5.0, 5.0, 5.0],
            "adverse_mag": [0.30, 0.30, 0.30],
            "cagr_to_mae_5y": [1.0, 0.666, 0.333],
            "last_n_years_green": [True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["instrument_id"]) == ["PT1", "PT2"]
    assert list(selected["symbol_key"])[0] == "SFP:A1"


def test_select_top_n_equities_universe_excludes_non_equities() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        mcap_floor=0.0,
        dollar_volume_floor_missing_mcap=0.0,
        require_last_n_years_green=0,
        universe="equities",
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:ETF1", "SFP:EQ1"],
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "entry_open_adj": [10.0, 10.0],
            "has_open": [True, True],
            "is_equity_asof": [False, True],
            "is_sfp_blacklisted": [False, False],
            "marketcap_entry": [100.0, 100.0],
            "marketcap_missing": [False, False],
            "dollar_volume_median": [1.0, 1.0],
            "cagr_5y": [0.50, 0.10],
            "worst_past_60m_mae_12m": [-0.30, -0.30],
            "n_past_mae_12m": [60.0, 60.0],
            "worst_past_5y_mae_12m": [-0.30, -0.30],
            "n_past_5y_mae_12m": [5.0, 5.0],
            "adverse_mag": [0.30, 0.30],
            "cagr_to_mae_5y": [1.666, 0.333],
            "last_n_years_green": [True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    assert list(selected["symbol_key"]) == ["SFP:EQ1"]


def test_select_top_n_exclude_bond_like_etfs_filters_sfp_etf_names_only() -> None:
    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        top_n=10,
        require_last_n_years_green=0,
        universe="all",
        exclude_bond_like_etfs=True,
        mcap_floor=0.0,
        dollar_volume_floor=0.0,
    )
    df = pd.DataFrame(
        {
            "symbol_key": ["SFP:TLT", "SFP:GLD", "SFP:JPST", "SFP:FTSM", "SEP:BONDCO"],
            "instrument_id": ["PT1", "PT2", "PT3", "PT4", "PT5"],
            "instrument_id_is_fallback": [False, False, False, False, False],
            "permaticker": pd.Series([1, 2, 3, 4, 5], dtype="Int64"),
            "table": ["SFP", "SFP", "SFP", "SFP", "SEP"],
            "ticker": ["TLT", "GLD", "JPST", "FTSM", "BONDCO"],
            "category_asof": ["ETF", "ETF", "ETF", "ETF", "Domestic Common Stock"],
            "name_asof": [
                "ISHARES 20+ YEAR TREASURY BOND ETF",
                "SPDR GOLD TRUST",
                "JPMORGAN ULTRA-SHORT INCOME ETF",
                "FIRST TRUST ENHANCED SHORT MATURITY ETF",
                "ACME BOND COMPANY INC",
            ],
            "entry_open_adj": [10.0, 10.0, 10.0, 10.0, 10.0],
            "has_open": [True, True, True, True, True],
            "is_equity_asof": [False, False, False, False, True],
            "is_sfp_blacklisted": [False, False, False, False, False],
            "marketcap_entry": [float("nan"), float("nan"), float("nan"), float("nan"), 100.0],
            "marketcap_missing": [True, True, True, True, False],
            "dollar_volume_median": [1e9, 1e9, 1e9, 1e9, 1e9],
            "cagr_5y": [0.50, 0.40, 0.35, 0.34, 0.30],
            "worst_past_5y_mae_12m": [-0.20, -0.20, -0.20, -0.20, -0.20],
            "n_past_5y_mae_12m": [5.0, 5.0, 5.0, 5.0, 5.0],
            "adverse_mag": [0.20, 0.20, 0.20, 0.20, 0.20],
            "cagr_to_mae_5y": [2.5, 2.0, 1.75, 1.70, 1.5],
            "last_n_years_green": [True, True, True, True, True],
        }
    )
    selected = _select_top_n(df, cfg)
    # TLT/JPST/FTSM are excluded by the bond-like ETF filter; GLD remains eligible in universe=all.
    assert "SFP:TLT" not in set(selected["symbol_key"])
    assert "SFP:JPST" not in set(selected["symbol_key"])
    assert "SFP:FTSM" not in set(selected["symbol_key"])
    assert "SFP:GLD" in set(selected["symbol_key"])
    # SEP equities should not be filtered by this rule even if the name contains 'bond'.
    assert "SEP:BONDCO" in set(selected["symbol_key"])


def test_select_snapshot_for_date_loads_year_month(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    ym_dir = cache_dir / "candidates_monthly" / "year_month=2025-12"
    ym_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "instrument_id": ["PT1", "PT2"],
            "instrument_id_is_fallback": [False, False],
            "permaticker": pd.Series([1, 2], dtype="Int64"),
            "symbol_key": ["SEP:AAA", "SEP:BBB"],
            "table": ["SEP", "SEP"],
            "ticker": ["AAA", "BBB"],
            "year_month": ["2025-12", "2025-12"],
            "month_idx": [0, 0],
            "select_date": pd.to_datetime(["2025-12-01", "2025-12-01"]),
            "has_open": [True, True],
            "is_equity_asof": [True, True],
            "is_sfp_blacklisted": [False, False],
            "marketcap_entry": [20_000.0, 20_000.0],  # USD millions
            "marketcap_missing": [False, False],
            "dollar_volume_median": [100_000_000.0, 100_000_000.0],
            "cagr_5y": [0.30, 0.10],
            "worst_past_60m_mae_12m": [-0.30, -0.30],
            "n_past_mae_12m": [60.0, 60.0],
            "worst_past_5y_mae_12m": [-0.30, -0.30],
            "n_past_5y_mae_12m": [5.0, 5.0],
            "adverse_mag": [0.30, 0.30],
            "cagr_to_mae_5y": [1.0, 0.333],
            "last_n_years_green": [True, True],
        }
    )
    df.to_parquet(ym_dir / "part.parquet", index=False)

    cfg = EntryRotationConfig(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2025,
        end_year=2025,
        sharadar_dir=tmp_path,
        cache_dir=cache_dir,
        output_dir=tmp_path,
        top_n=1,
        mcap_floor=10_000_000_000.0,
        require_last_n_years_green=0,
        universe="equities",
    )
    snap = select_snapshot_for_date(cfg, select_date=pd.Timestamp("2025-12-29"))
    assert list(snap["symbol_key"]) == ["SEP:AAA"]


def test_tickers_cache_filters_to_price_tables_only(tmp_path: Path) -> None:
    sharadar_dir = tmp_path / "sharadar"
    sharadar_dir.mkdir(parents=True, exist_ok=True)
    tickers_csv = sharadar_dir / "SHARADAR_TICKERS_test.csv"
    tickers_csv.write_text(
        "table,ticker,permaticker,firstpricedate,lastpricedate,lastupdated,category,name\n"
        "SEP,AAA,1,2000-01-01,2026-01-01,2026-01-02,Domestic Common Stock,AAA INC\n"
        "SFP,BBB,2,2000-01-01,2026-01-01,2026-01-02,Domestic Common Stock,BBB INC\n"
        "SF1,CCC,3,2000-01-01,2026-01-01,2026-01-02,Domestic Common Stock,CCC INC\n"
    )
    paths = SharadarPaths(tickers_csv=tickers_csv, sep_csv=Path("missing"), sfp_csv=Path("missing"), daily_csv=Path("missing"))
    cache_dir = tmp_path / "cache" / "tickers"
    _ensure_tickers_cache(paths, cache_dir, rebuild=False)

    df = pd.read_parquet(cache_dir / "tickers.parquet")
    assert set(df["table"].unique()) == {"SEP", "SFP"}
    assert all(~df["symbol_key"].str.startswith("SF1:"))


def test_annual_sim_end_date_prefers_exit_rebalance_date_when_present() -> None:
    bench_dates = pd.DatetimeIndex(pd.date_range("2020-02-03", "2021-02-03", freq="B"))
    rebalance_dates = {2020: pd.Timestamp("2020-02-03"), 2021: pd.Timestamp("2021-02-03")}
    end_date, exit_reb = _annual_sim_end_date(bench_dates=bench_dates, rebalance_dates=rebalance_dates, end_year=2020)
    assert end_date == pd.Timestamp("2021-02-03")
    assert exit_reb == pd.Timestamp("2021-02-03")


def test_monthly_sim_end_date_extends_to_cover_maturity_month() -> None:
    month_index = pd.DataFrame(
        {
            "year_month": pd.period_range("2020-01", "2021-12", freq="M").astype(str),
            "month_idx": list(range(24)),
            "select_date": pd.date_range("2020-01-01", periods=24, freq="MS"),
        }
    )
    sim, start_d, end_d = _monthly_sim_end_date(month_index=month_index, start_year=2020, end_year=2020, hold_period_months=12)
    assert len(sim) == 24
    assert start_d == pd.Timestamp("2020-01-01")
    assert end_d == pd.Timestamp("2021-12-01")


def test_config_mae_stop_pct_validation() -> None:
    base = dict(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
    )
    EntryRotationConfig(**base, mae_stop_pct=0.0)
    EntryRotationConfig(**base, mae_stop_pct=0.2)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, mae_stop_pct=-0.01)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, mae_stop_pct=1.0)


def test_config_position_mae_stop_pct_validation() -> None:
    base = dict(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
    )
    EntryRotationConfig(**base, position_mae_stop_pct=0.0)
    EntryRotationConfig(**base, position_mae_stop_pct=0.2)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, position_mae_stop_pct=-0.01)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, position_mae_stop_pct=1.0)


def test_config_monthly_sell_policy_validation() -> None:
    base = dict(
        mode="monthly-dca",
        price_mode="adj",
        start_year=2020,
        end_year=2020,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
        monthly_dca_amount=1000.0,
    )
    EntryRotationConfig(**base, monthly_sell_policy="maturity", max_new_tickers_per_year=12)
    EntryRotationConfig(**base, monthly_sell_policy="never", max_new_tickers_per_year=12)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, monthly_sell_policy="never", max_new_tickers_per_year=0)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, monthly_sell_policy="wat", max_new_tickers_per_year=12)  # type: ignore[arg-type]

    annual = dict(
        mode="annual-lumpsum",
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
    )
    with pytest.raises(ValueError):
        EntryRotationConfig(**annual, monthly_sell_policy="never")


def test_yearly_new_name_cap_helpers_track_opened_and_reserved_counts() -> None:
    opened: dict[int, set[str]] = {}
    reserved: dict[int, set[str]] = {}
    assert _count_opened_new_names_for_year(opened, year=2025) == 0
    assert _count_reserved_new_names_for_year(reserved, year=2025) == 0

    _reserve_new_name_for_year(reserved, year=2025, instrument_id="PT1")
    _reserve_new_name_for_year(reserved, year=2025, instrument_id="PT2")
    assert _count_reserved_new_names_for_year(reserved, year=2025) == 2

    _release_reserved_new_name_for_year(reserved, year=2025, instrument_id="PT1")
    assert _count_reserved_new_names_for_year(reserved, year=2025) == 1

    opened.setdefault(2025, set()).add("PT2")
    assert _count_opened_new_names_for_year(opened, year=2025) == 1


def test_whole_shares_for_spendable_cash_respects_buffer_and_commission() -> None:
    assert _whole_shares_for_spendable_cash(cash=100.0, effective_price=10.0, commission=1.0, cash_buffer_pct=0.0) == 9
    assert _whole_shares_for_spendable_cash(cash=100.0, effective_price=10.0, commission=1.0, cash_buffer_pct=0.10) == 8
    assert _whole_shares_for_spendable_cash(cash=10.0, effective_price=10.0, commission=1.0, cash_buffer_pct=0.0) == 0
    assert _whole_shares_for_spendable_cash(cash=0.0, effective_price=10.0, commission=1.0, cash_buffer_pct=0.0) == 0


def test_position_mae_pct_from_cache_uses_low_else_close() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"]).to_numpy(dtype="datetime64[ns]")
    price_cache = {
        "PT1": (
            dates,
            pd.Series([10.0, 10.0]).to_numpy(dtype="float64"),
            pd.Series([8.0, float("nan")]).to_numpy(dtype="float64"),
            pd.Series([9.0, 7.0]).to_numpy(dtype="float64"),
        )
    }
    mae0 = _position_mae_pct_from_cache(
        price_cache=price_cache, instrument_id="PT1", date=pd.Timestamp("2020-01-02"), entry_effective_price=10.0
    )
    assert mae0 == pytest.approx(-0.2)
    mae1 = _position_mae_pct_from_cache(
        price_cache=price_cache, instrument_id="PT1", date=pd.Timestamp("2020-01-03"), entry_effective_price=10.0
    )
    assert mae1 == pytest.approx(-0.3)


def test_bench_date_on_or_after_picks_next_or_none() -> None:
    bench = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-01-10", "2020-01-20"]))
    assert _bench_date_on_or_after(bench, pd.Timestamp("2020-01-10")) == pd.Timestamp("2020-01-10")
    assert _bench_date_on_or_after(bench, pd.Timestamp("2020-01-11")) == pd.Timestamp("2020-01-20")
    assert _bench_date_on_or_after(bench, pd.Timestamp("2020-02-01")) is None


def test_benchmark_buy_hold_calendar_year_returns_uses_open_to_close() -> None:
    calendar = pd.DatetimeIndex(pd.to_datetime(["2003-01-02", "2003-12-31", "2004-01-03", "2004-12-30"]))
    dates = calendar.to_numpy(dtype="datetime64[ns]")
    open_px = pd.Series([100.0, 110.0, 120.0, 130.0]).to_numpy(dtype="float64")
    low_px = pd.Series([90.0, 105.0, 100.0, 120.0]).to_numpy(dtype="float64")
    close_px = pd.Series([105.0, 120.0, 125.0, 140.0]).to_numpy(dtype="float64")
    price_cache = {"PTSPY": (dates, open_px, low_px, close_px)}

    out = _benchmark_buy_hold_calendar_year_returns(calendar=calendar, price_cache=price_cache, instrument_id="PTSPY")
    assert list(out["year"]) == [2003, 2004]
    assert float(out[out["year"] == 2003]["bm_buy_hold_return_pct"].iloc[0]) == pytest.approx(120.0 / 100.0 - 1.0)
    assert float(out[out["year"] == 2004]["bm_buy_hold_return_pct"].iloc[0]) == pytest.approx(140.0 / 120.0 - 1.0)


def test_monthly_cycle_maturity_is_select_date_plus_12_months() -> None:
    select_date_by_month_idx = {i: pd.Timestamp(d) for i, d in enumerate(pd.date_range("2020-01-01", periods=24, freq="MS"))}
    maturity = select_date_by_month_idx.get(1 + 12)
    assert maturity == pd.Timestamp("2021-02-01")


def test_next_select_date_after_picks_next_or_none() -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"]))
    assert _next_select_date_after(dates, pd.Timestamp("2020-01-02")) == pd.Timestamp("2020-02-03")
    assert _next_select_date_after(dates, pd.Timestamp("2020-01-15")) == pd.Timestamp("2020-02-03")
    assert _next_select_date_after(dates, pd.Timestamp("2020-03-02")) is None


def test_config_stopout_cooldown_months_validation() -> None:
    base = dict(
        price_mode="adj",
        start_year=2020,
        end_year=2021,
        sharadar_dir=Path("tmp"),
        cache_dir=Path("tmp"),
        output_dir=Path("tmp"),
    )
    EntryRotationConfig(**base, mode="monthly-dca", monthly_dca_amount=1.0, stopout_cooldown_months=0)
    EntryRotationConfig(**base, mode="monthly-dca", monthly_dca_amount=1.0, stopout_cooldown_months=12)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, mode="monthly-dca", monthly_dca_amount=1.0, stopout_cooldown_months=-1)
    with pytest.raises(ValueError):
        EntryRotationConfig(**base, mode="annual-lumpsum", stopout_cooldown_months=1)

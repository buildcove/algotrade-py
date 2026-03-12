import runpy

import numpy as np
import pytest


def _load_impl():
    mod = runpy.run_path("tools/eval_barsmith_kinetic_combos.py")
    return mod["compute_equity_curve_dollar_contracts"], mod["compute_rust_style_stats"]


def test_compute_equity_curve_dollar_contracts_compounds_and_scales_contracts() -> None:
    compute_equity_curve_dollar_contracts, _compute_rust_style_stats = _load_impl()

    rr = np.array([1.0, -1.0], dtype=float)
    rpc = np.array([50.0, 50.0], dtype=float)

    equity = compute_equity_curve_dollar_contracts(rr, rpc, 1000.0, 10.0, min_contracts=1, max_contracts=None)
    assert equity.tolist() == pytest.approx([1100.0, 1000.0], rel=1e-12)


def test_compute_rust_style_stats_contracts_reports_equity_dd_pct() -> None:
    _compute_equity_curve_dollar_contracts, compute_rust_style_stats = _load_impl()

    rr = np.array([1.0, -1.0], dtype=float)
    labels = np.array([False, False], dtype=bool)
    rpc = np.array([50.0, 50.0], dtype=float)

    stats = compute_rust_style_stats(
        rr,
        labels,
        1000.0,
        10.0,
        1.0,
        position_sizing="contracts",
        risk_per_contract_dollar=rpc,
        min_contracts=1,
        max_contracts=None,
    )

    assert stats["final_capital"] == pytest.approx(1000.0, rel=1e-12)
    assert stats["total_return_pct"] == pytest.approx(0.0, abs=1e-12)
    # Peak 1100 then 1000 => 9.0909% drawdown.
    assert stats["max_drawdown_pct_equity"] == pytest.approx((1100.0 - 1000.0) / 1100.0 * 100.0, rel=1e-9)


def test_compute_equity_curve_dollar_contracts_reflects_per_contract_costs_via_rr() -> None:
    compute_equity_curve_dollar_contracts, _compute_rust_style_stats = _load_impl()

    # Gross RR=1.0 with rpc=$50, and a $10/contract round-trip cost implies
    # rr_net = 1.0 - (10/50) = 0.8 (Barsmith Rust contracts mode).
    rr_net = np.array([0.8], dtype=float)
    rpc = np.array([50.0], dtype=float)
    equity = compute_equity_curve_dollar_contracts(rr_net, rpc, 1000.0, 10.0, min_contracts=1, max_contracts=None)
    # Risk budget = $100 => contracts=floor(100/50)=2 => pnl=0.8*50*2=$80
    assert equity.tolist() == pytest.approx([1080.0], rel=1e-12)


def test_compute_equity_curve_dollar_contracts_applies_margin_cap() -> None:
    compute_equity_curve_dollar_contracts, _compute_rust_style_stats = _load_impl()

    # capital=10k, risk=10% => risk budget=1k, rpc=100 => 10 contracts by risk
    # margin cap: 10k / 2500 = 4 contracts
    rr = np.array([1.0], dtype=float)
    rpc = np.array([100.0], dtype=float)
    equity = compute_equity_curve_dollar_contracts(
        rr,
        rpc,
        10_000.0,
        10.0,
        min_contracts=1,
        max_contracts=None,
        margin_per_contract_dollar=2500.0,
    )
    assert equity.tolist() == pytest.approx([10_400.0], rel=1e-12)

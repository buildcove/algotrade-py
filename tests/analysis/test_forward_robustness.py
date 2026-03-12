from __future__ import annotations

import pytest
from experiments.analysis.forward_robustness import compute_frs


def test_compute_frs_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        compute_frs([1.0, 2.0], [1.0], [10, 20])


def test_compute_frs_negative_median_calmar_clamps_to_zero() -> None:
    frs = compute_frs([-1.0, -1.0], [1.0, 1.0], [30, 30])
    assert frs.c_plus == 0.0
    assert frs.frs == 0.0


def test_compute_frs_tail_penalty_collapses_on_blowup() -> None:
    frs = compute_frs([1.0, 1.0, 1.0], [1.0, 1.0, 10.0], [30, 30, 30])
    assert frs.dd_min == 1.0
    assert frs.dd_max == 10.0
    assert frs.dd_median == 1.0
    assert frs.dd_mean == pytest.approx(4.0, abs=1e-12)
    assert frs.tail_penalty == pytest.approx(1.0 / 11.0, rel=1e-9)


def test_compute_frs_trade_score_penalizes_low_activity() -> None:
    frs = compute_frs([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [10, 10, 10], n_min=30)
    assert frs.trade_score == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_compute_frs_stability_collapses_when_mean_zero() -> None:
    frs = compute_frs([1.0, -1.0], [1.0, 1.0], [30, 30])
    assert frs.mu_r == pytest.approx(0.0, abs=1e-12)
    assert frs.stability < 1e-6


def test_compute_frs_reports_return_median() -> None:
    frs = compute_frs([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [30, 30, 30])
    assert frs.r_min == pytest.approx(1.0, abs=1e-12)
    assert frs.r_median == pytest.approx(2.0, abs=1e-12)
    assert frs.r_max == pytest.approx(3.0, abs=1e-12)


def test_compute_frs_stability_uses_median_and_mad() -> None:
    # Median is 1, MAD is 0 => stability should be 1 even with an outlier.
    frs = compute_frs([1.0, 1.0, 1.0, 100.0], [1.0, 1.0, 1.0, 1.0], [30, 30, 30, 30])
    assert frs.r_median == pytest.approx(1.0, abs=1e-12)
    assert frs.stability == pytest.approx(1.0, abs=1e-12)

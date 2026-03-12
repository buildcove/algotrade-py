import runpy

import numpy as np


def _load_select_trade_indices_no_stacking():
    mod = runpy.run_path("tools/eval_barsmith_kinetic_combos.py")
    return mod["select_trade_indices_no_stacking"]


def test_select_trade_indices_no_stacking_skips_until_exit_index():
    select_trade_indices_no_stacking = _load_select_trade_indices_no_stacking()

    combo_mask = np.array([True, True, True, True, True], dtype=bool)
    eligible = np.array([True, True, True, True, True], dtype=bool)
    rr_finite = np.array([True, True, True, True, True], dtype=bool)
    exit_i = np.array([3, 3, 3, 5, 5], dtype=np.int64)

    # hit 0 => next_free=3, so hits 1 and 2 are skipped; hit 3 is allowed.
    got = select_trade_indices_no_stacking(combo_mask, eligible, rr_finite, exit_i)
    assert got.tolist() == [0, 3]


def test_select_trade_indices_no_stacking_ignores_ineligible_hits():
    select_trade_indices_no_stacking = _load_select_trade_indices_no_stacking()

    combo_mask = np.array([True, True, True], dtype=bool)
    eligible = np.array([False, True, True], dtype=bool)
    rr_finite = np.array([True, True, True], dtype=bool)
    exit_i = np.array([2, 3, 3], dtype=np.int64)

    got = select_trade_indices_no_stacking(combo_mask, eligible, rr_finite, exit_i)
    assert got.tolist() == [1]


def test_select_trade_indices_no_stacking_requires_finite_rr():
    select_trade_indices_no_stacking = _load_select_trade_indices_no_stacking()

    combo_mask = np.array([True, True, True], dtype=bool)
    eligible = np.array([True, True, True], dtype=bool)
    rr_finite = np.array([False, True, True], dtype=bool)
    exit_i = np.array([2, 3, 3], dtype=np.int64)

    got = select_trade_indices_no_stacking(combo_mask, eligible, rr_finite, exit_i)
    assert got.tolist() == [1]


def test_select_trade_indices_no_stacking_treats_invalid_exit_i_as_next_bar():
    select_trade_indices_no_stacking = _load_select_trade_indices_no_stacking()

    combo_mask = np.array([True, True, True], dtype=bool)
    eligible = np.array([True, True, True], dtype=bool)
    rr_finite = np.array([True, True, True], dtype=bool)

    # exit_i == idx is treated as invalid (should advance at least 1 bar),
    # so 0 doesn't block 1.
    exit_i = np.array([0, -1, 2], dtype=np.int64)
    got = select_trade_indices_no_stacking(combo_mask, eligible, rr_finite, exit_i)
    assert got.tolist() == [0, 1, 2]

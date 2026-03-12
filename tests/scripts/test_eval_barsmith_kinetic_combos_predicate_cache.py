import runpy

import numpy as np
import pandas as pd
import pytest


def _load_impl():
    mod = runpy.run_path("tools/eval_barsmith_kinetic_combos.py")
    return mod["build_combo_mask"], mod["PredicateMaskCache"]


@pytest.mark.parametrize(
    "expr",
    [
        "flag_a",
        "x>1.5",
        "x<=1.5",
        "x==1.0",
        "x=1.0",
        "x!=1.0",
        "x>y",
        "x>0.01 && y<=2.0 && flag_a",
        "x> 0.01 && y <= 2.0 && flag_a",
    ],
)
def test_predicate_mask_cache_matches_build_combo_mask(expr: str):
    build_combo_mask, PredicateMaskCache = _load_impl()

    df = pd.DataFrame(
        {
            "flag_a": [1.0, 0.0, np.nan, 1.0],
            "x": [0.0, 1.0, 2.0, 0.0101],
            "y": [0.0, 2.0, 1.0, 2.0],
        }
    )

    # For parity with Barsmith's formatted predicate names, the expression might say
    # x>0.01 but the actual threshold should be slightly different.
    threshold_name_map = {"x>0.01": 0.01023}

    expected = build_combo_mask(df, expr, threshold_name_map=threshold_name_map).to_numpy(dtype=bool, copy=False)
    cache = PredicateMaskCache(df, threshold_name_map=threshold_name_map)
    got = cache.eval_expr(expr)

    assert np.array_equal(got, expected)


def test_predicate_mask_cache_raises_on_missing_column():
    _build_combo_mask, PredicateMaskCache = _load_impl()
    df = pd.DataFrame({"x": [1.0, 2.0]})
    cache = PredicateMaskCache(df, threshold_name_map=None)

    with pytest.raises(KeyError):
        cache.eval_expr("missing_flag")

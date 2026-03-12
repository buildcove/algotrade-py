#!/usr/bin/env python3
"""
Evaluate the “kinetic” Barsmith top‑10 combinations on a Barsmith‑engineered
dataset (`barsmith_prepared.csv`), split into:

  - In‑sample:   bars with timestamp <= cutoff date (default 2024‑12‑31)
  - Forward:     bars with timestamp  > cutoff date

The script:
  - Reads the engineered CSV produced by `custom_rs::prepare_dataset`.
  - Uses a Rust‑computed target column (default: `2x_atr_tp_atr_stop`; also supports
    `3x_atr_tp_atr_stop`, `atr_tp_atr_stop`, `highlow_or_atr`, etc.)
    and its matching RR column (default: `rr_<target>`).
  - If the engineered dataset includes `<target>_eligible`, metrics are computed
    per-trade using: `trades = combo_mask && eligible && rr is finite` (recall
    is still based on raw combo-mask hits).
  - Applies each combination as an AND‑only filter over the engineered
    columns (no tribar_ai / pandas_ta).
  - Recomputes key performance metrics for each combo in both windows:
      Win rate, expectancy, total R, max DD, Calmar, PF, Sharpe, Sortino,
      median / p05 / p95 RR, avg loss, density, recall, and streak stats.

Usage example:

    uv run python tools/eval_barsmith_kinetic_combos.py \
    --prepared tmp/barsmith/2x_atr_tp_atr_stop/es_30m_official_v1_long/barsmith_prepared.csv \
    --cutoff 2024-12-31 --asset MES --target 2x_atr_tp_atr_stop \
    --max-drawdown 25 --min-calmar 1 \
    > test.log 
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.analysis.forward_robustness import compute_frs  # noqa: E402


def infer_stop_distance_column(target: str) -> str | None:
    target = str(target or "").strip()
    if target in {"2x_atr_tp_atr_stop", "3x_atr_tp_atr_stop", "atr_tp_atr_stop", "atr_stop"}:
        return "atr"
    return None


def build_calendar_year_windows(
    timestamps: pd.Series,
    forward_mask: pd.Series,
) -> list[dict]:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dt.tz_convert(None)
    years = ts.dt.year
    candidate_years = sorted(set(int(y) for y in years[forward_mask].dropna().unique()))

    windows: list[dict] = []
    for year in candidate_years:
        mask = forward_mask & (years == year)
        if not bool(mask.any()):
            continue
        ts_sub = ts[mask].dropna()
        windows.append(
            {
                "year": int(year),
                "label": f"{int(year)}–{int(year) + 1}",
                "mask": mask,
                "rows": int(mask.sum()),
                "start": ts_sub.min() if not ts_sub.empty else None,
                "end": ts_sub.max() if not ts_sub.empty else None,
            }
        )
    return windows


def _percentile_sorted(samples: np.ndarray, quantile: float) -> float:
    n = int(samples.size)
    if n == 0:
        return float("nan")
    rank = float(quantile) * float(n - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return float(samples[lower])
    weight = rank - float(lower)
    return float(samples[lower] + weight * (samples[upper] - samples[lower]))


def _round_to_nice_multiple(value: float, down: bool) -> float:
    value = float(value)
    if value == 0.0:
        return 0.0

    magnitude = 10.0 ** float(np.floor(np.log10(abs(value))))
    candidates = [magnitude, 2.0 * magnitude, 5.0 * magnitude, 10.0 * magnitude]

    if down:
        if value > 0.0:
            le = [c for c in candidates if c <= value]
            return float(max(le)) if le else 0.0

        le = [c for c in candidates if c <= abs(value)]
        return float(-min(le)) if le else float(-(magnitude * 10.0))

    if value > 0.0:
        ge = [c for c in candidates if c >= value]
        return float(min(ge)) if ge else float(candidates[-1])

    ge = [c for c in candidates if c >= abs(value)]
    return float(-max(ge)) if ge else 0.0


def _calculate_optimal_increment(min_val: float, max_val: float, operator_count: int, target_values: int) -> float:
    range_size = float(max_val) - float(min_val)
    if range_size <= 0.0:
        return 0.0

    target_per_operator = float(target_values + operator_count) / float(operator_count)
    ideal_increment = range_size / target_per_operator
    standard_increments = [
        0.00001,
        0.00002,
        0.00005,
        0.0001,
        0.0002,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        2000.0,
        5000.0,
    ]

    best_increment: float | None = None
    best_diff = float("inf")
    for inc in standard_increments:
        values_per_op = int(np.floor(range_size / float(inc))) + 1
        total_conditions = (values_per_op * operator_count) - operator_count
        if 5 <= total_conditions <= 15:
            diff = abs(float(total_conditions) - float(target_values))
            if diff < best_diff:
                best_diff = diff
                best_increment = float(inc)

    if best_increment is not None:
        return float(best_increment)

    return float(min(standard_increments, key=lambda x: abs(float(x) - ideal_increment)))


def _generate_threshold_values(min_val: float, max_val: float, increment: float, operator: str) -> list[float]:
    thresholds: list[float] = []
    current = float(min_val)
    max_val = float(max_val)
    increment = float(increment)
    while current <= max_val + float(np.finfo(float).eps):
        if operator == ">":
            impossible = current >= max_val
        elif operator == "<":
            impossible = current <= float(min_val)
        elif operator == ">=":
            impossible = current > max_val
        elif operator == "<=":
            impossible = current < float(min_val)
        else:
            impossible = False
        if not impossible:
            thresholds.append(float(current))
        current += increment
    return thresholds


def _format_threshold_value(value: float) -> str:
    formatted = f"{float(value):.2f}"
    if "." in formatted:
        trimmed = formatted.rstrip("0")
        if trimmed.endswith("."):
            trimmed += "0"
        decimals = trimmed.split(".", 1)[1]
        if len(decimals) <= 2:
            formatted = trimmed
    return formatted


def _extract_numeric_predicate_names(combos: list[str]) -> set[str]:
    needed: set[str] = set()
    for expr in combos:
        clauses = [c.strip() for c in expr.split("&&") if c.strip()]
        for clause in clauses:
            op = None
            for candidate in ("<=", ">=", "==", "!=", "<", ">", "="):
                if candidate in clause:
                    op = candidate
                    break
            if op is None:
                continue
            left, right = [p.strip() for p in clause.split(op, 1)]
            try:
                float(right)
            except ValueError:
                continue
            needed.add(f"{left}{op}{right}")
    return needed


def build_barsmith_threshold_name_map(
    df: pd.DataFrame,
    needed_names: set[str],
    *,
    feature_ranges_path: Path,
) -> dict[str, float]:
    """
    Rebuild Barsmith's feature-vs-constant threshold names -> float thresholds.

    Barsmith formats thresholds to 2dp in predicate *names* (e.g. "...<=0.01")
    but evaluates using the underlying float threshold. This mapping lets
    Python parity evaluation match Barsmith's true predicate values.
    """
    if not needed_names or not feature_ranges_path.exists():
        return {}

    ranges = json.loads(feature_ranges_path.read_text())
    name_to_threshold: dict[str, float] = {}

    for feature, cfg in ranges.items():
        if cfg.get("enabled") is False:
            continue
        if feature not in df.columns:
            continue
        if not any(name.startswith(f"{feature}>") or name.startswith(f"{feature}<") for name in needed_names):
            continue

        series = pd.to_numeric(df[feature], errors="coerce")
        values = series.to_numpy(dtype=float, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            continue

        finite.sort()
        p1 = _percentile_sorted(finite, 0.01)
        p99 = _percentile_sorted(finite, 0.99)
        min_override = _round_to_nice_multiple(p1, down=True)
        max_override = _round_to_nice_multiple(p99, down=False)

        min_val = min_override if cfg.get("min") == "auto" else float(cfg.get("min", min_override))
        max_val = max_override if cfg.get("max") == "auto" else float(cfg.get("max", max_override))
        if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val >= max_val:
            continue

        operators = list(cfg.get("operators") or [])
        operator_count = len(operators) if operators else 1
        if cfg.get("increment") == "auto":
            increment = _calculate_optimal_increment(min_val, max_val, operator_count, 10)
        else:
            increment = float(cfg.get("increment") or 0.0)
        if increment <= 0.0:
            continue

        for operator in operators:
            thresholds = _generate_threshold_values(min_val, max_val, increment, operator)
            for threshold in thresholds:
                name = f"{feature}{operator}{_format_threshold_value(threshold)}"
                if name not in needed_names:
                    continue
                if name in name_to_threshold:
                    continue
                name_to_threshold[name] = float(threshold)

    return name_to_threshold


def compute_equity_curve_dollar(
    rr: np.ndarray,
    capital_dollar: float,
    risk_pct_per_trade: float,
) -> np.ndarray:
    rr = rr.astype(float)
    capital = float(capital_dollar)
    risk_factor = float(risk_pct_per_trade) / 100.0
    out = np.full(rr.shape[0], np.nan, dtype=float)
    for i, r in enumerate(rr):
        risk_i = capital * risk_factor
        pnl = r * risk_i
        capital = capital + pnl
        out[i] = capital
    return out


def compute_equity_curve_dollar_contracts(
    rr: np.ndarray,
    risk_per_contract_dollar: np.ndarray,
    capital_dollar: float,
    risk_pct_per_trade: float,
    *,
    min_contracts: int = 1,
    max_contracts: int | None = None,
    margin_per_contract_dollar: float | None = None,
) -> np.ndarray:
    rr = rr.astype(float)
    rpc = risk_per_contract_dollar.astype(float)
    capital = float(capital_dollar)
    risk_factor = float(risk_pct_per_trade) / 100.0
    min_contracts_i = max(int(min_contracts), 1)
    max_contracts_i = int(max_contracts) if max_contracts is not None else None

    out = np.full(rr.shape[0], np.nan, dtype=float)
    for i, (r, rpc_i) in enumerate(zip(rr, rpc, strict=False)):
        if not np.isfinite(r) or not np.isfinite(rpc_i) or rpc_i <= 0.0:
            out[i] = capital
            continue
        risk_budget = capital * risk_factor
        raw = np.floor(risk_budget / rpc_i) if rpc_i > 0.0 else 0.0
        contracts = int(raw) if np.isfinite(raw) and raw >= 0.0 else 0
        if contracts < min_contracts_i:
            contracts = min_contracts_i
        if max_contracts_i is not None:
            contracts = min(contracts, max_contracts_i)
        if (
            margin_per_contract_dollar is not None
            and np.isfinite(float(margin_per_contract_dollar))
            and float(margin_per_contract_dollar) > 0.0
        ):
            cap = np.floor(capital / float(margin_per_contract_dollar))
            if np.isfinite(cap) and cap >= 0.0:
                contracts = min(contracts, int(cap))
        if contracts <= 0:
            out[i] = capital
            continue
        pnl = float(r) * float(rpc_i) * float(contracts)
        capital = capital + pnl
        out[i] = capital
    return out


def compute_equity_curves_for_window(
    df: pd.DataFrame,
    window_mask: pd.Series,
    strategies: list[dict],
    top_k: int,
    *,
    capital_dollar: float,
    risk_pct_per_trade: float,
    cost_per_trade_dollar: float | None,
    position_sizing: str,
    stop_distance_column: str | None,
    stop_distance_unit: str,
    point_value: float | None,
    tick_value: float | None,
    min_contracts: int = 1,
    max_contracts: int | None = None,
    margin_per_contract_dollar: float | None = None,
    target_col: str,
    rr_col: str,
    eligible_col: str | None,
    stacking_mode: str,
    exit_i_col: str | None,
    window_label: str,
    rank_by: str,
    threshold_name_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    if top_k <= 0:
        return pd.DataFrame()

    sub = df.loc[window_mask]
    if sub.empty:
        return pd.DataFrame()

    if rr_col not in sub.columns:
        raise KeyError(f"RR column '{rr_col}' not found in engineered CSV.")

    rr_series = sub[rr_col].astype(float)
    rr_values = rr_series.to_numpy(dtype=float, copy=False)
    rr_finite_np = np.isfinite(rr_values)

    position_sizing_norm = str(position_sizing or "fractional").strip().lower()
    if position_sizing_norm not in {"fractional", "contracts"}:
        raise ValueError(f"Unknown position_sizing: {position_sizing!r}")
    rpc_values = compute_risk_per_contract_dollar_series(
        sub,
        position_sizing=position_sizing_norm,
        stop_distance_column=stop_distance_column,
        stop_distance_unit=stop_distance_unit,
        point_value=point_value,
        tick_value=tick_value,
    )
    if rpc_values is not None:
        rr_finite_np = rr_finite_np & np.isfinite(rpc_values) & (rpc_values > 0.0)

    if eligible_col is not None and eligible_col in sub.columns:
        eligible_series = sub[eligible_col]
        if eligible_series.dtype == bool:
            eligible_mask = eligible_series.fillna(False)
        else:
            eligible_mask = eligible_series.fillna(0.0).astype(float) > 0.5
    else:
        eligible_mask = pd.Series(True, index=sub.index)

    eligible_np = eligible_mask.to_numpy(dtype=bool, copy=False)
    entry_abs_indices = pd.to_numeric(sub.index.to_series(), errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
    if np.any(entry_abs_indices < 0):
        entry_abs_indices = np.arange(int(len(sub)), dtype=np.int64)
    exit_indices: np.ndarray | None = None
    if stacking_mode == "no-stacking":
        if not exit_i_col or exit_i_col not in sub.columns:
            raise SystemExit(
                f"--stacking-mode no-stacking requires an exit index column in the engineered CSV (expected '{target_col}_exit_i'). "
                "Re-generate the prepared CSV using a newer Barsmith custom_rs, or rerun with --stacking-mode stacking."
            )
        exit_indices = _parse_exit_indices(sub[exit_i_col], expected_len=len(sub))

    ts_all = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    ts_values = ts_all.to_numpy()

    dollars_per_r = capital_dollar * (risk_pct_per_trade / 100.0) if capital_dollar > 0.0 and risk_pct_per_trade > 0.0 else 0.0
    cost_per_trade_r = None
    if position_sizing_norm == "fractional" and cost_per_trade_dollar is not None and dollars_per_r > 0.0:
        cost_per_trade_r = float(cost_per_trade_dollar) / float(dollars_per_r)

    frames: list[pd.DataFrame] = []
    predicate_cache = PredicateMaskCache(sub, threshold_name_map=threshold_name_map)
    eligible_strategies = [s for s in strategies if s.get("metrics") is not None]
    for rank, strategy in enumerate(eligible_strategies[:top_k], start=1):
        expr = strategy["expr"]
        combo_np = predicate_cache.eval_expr(expr)
        if stacking_mode == "no-stacking":
            if exit_indices is None:
                raise RuntimeError("exit_indices not initialized for no-stacking mode")
            trade_indices = select_trade_indices_no_stacking(
                combo_np,
                eligible_np,
                rr_finite_np,
                exit_indices,
                entry_abs_indices=entry_abs_indices,
            )
        else:
            trade_mask_np = combo_np & eligible_np & rr_finite_np
            trade_indices = np.flatnonzero(trade_mask_np).astype(np.int64, copy=False)
        trades = int(trade_indices.size)
        if trades == 0:
            continue

        rr_raw = rr_values[trade_indices]
        if position_sizing_norm == "contracts":
            if rpc_values is None:
                raise RuntimeError("rpc_values must be computed for contracts sizing mode")
            rpc_trade = rpc_values[trade_indices]
            if cost_per_trade_dollar is not None and float(cost_per_trade_dollar) != 0.0:
                rr_net = rr_raw - (float(cost_per_trade_dollar) / rpc_trade)
            else:
                rr_net = rr_raw
        else:
            if cost_per_trade_r is not None and cost_per_trade_r > 0.0:
                rr_net = rr_raw - float(cost_per_trade_r)
            else:
                rr_net = rr_raw

        ts = ts_values[trade_indices]
        equity_r = np.cumsum(rr_net)

        equity_dollar = None
        if capital_dollar > 0.0 and risk_pct_per_trade > 0.0:
            if position_sizing_norm == "contracts":
                if rpc_values is None:
                    raise RuntimeError("rpc_values must be computed for contracts sizing mode")
                rpc_trade = rpc_values[trade_indices]
                equity_dollar = compute_equity_curve_dollar_contracts(
                    rr_net,
                    rpc_trade,
                    capital_dollar,
                    risk_pct_per_trade,
                    min_contracts=min_contracts,
                    max_contracts=max_contracts,
                    margin_per_contract_dollar=margin_per_contract_dollar,
                )
            else:
                equity_dollar = compute_equity_curve_dollar(rr_net, capital_dollar, risk_pct_per_trade)

        out = pd.DataFrame(
            {
                "rank_by": rank_by,
                "window": window_label,
                "rank": rank,
                "formula": expr,
                "timestamp": ts,
                "trade_index": np.arange(1, trades + 1, dtype=int),
                "rr": rr_net,
                "equity_r": equity_r,
            }
        )
        if equity_dollar is not None:
            out["equity_dollar"] = equity_dollar
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_equity_curves(
    curves: pd.DataFrame,
    *,
    out_path: Path,
    x_axis: str,
    metric: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"Plotting requested but matplotlib is unavailable ({exc}). " "Install matplotlib or rerun with --no-plot."
        ) from exc

    if curves.empty:
        raise SystemExit("No equity curve rows available to plot.")

    x_col = "timestamp" if x_axis == "timestamp" else "trade_index"
    y_col = "equity_dollar" if metric == "dollar" else "equity_r"
    if y_col not in curves.columns:
        raise SystemExit(
            f"Requested plot metric '{metric}' but column '{y_col}' is unavailable. "
            "Provide --capital and --risk-pct-per-trade, or use --plot-metric r."
        )

    fig, ax = plt.subplots(figsize=(14, 8))
    for (rank_by, window, rank), group in curves.groupby(["rank_by", "window", "rank"], sort=True):
        label = f"{rank_by} {window} Rank {rank}"
        ax.plot(group[x_col], group[y_col], linewidth=1.6, label=label)

    ax.set_title("Barsmith Top Strategies Equity Curves")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)


def plot_equity_curves_individual(
    curves: pd.DataFrame,
    *,
    out_dir: Path,
    x_axis: str,
    metric: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"Plotting requested but matplotlib is unavailable ({exc}). " "Install matplotlib or rerun with --no-plot."
        ) from exc

    if curves.empty:
        raise SystemExit("No equity curve rows available to plot.")

    x_col = "timestamp" if x_axis == "timestamp" else "trade_index"
    y_col = "equity_dollar" if metric == "dollar" else "equity_r"
    if y_col not in curves.columns:
        raise SystemExit(
            f"Requested plot metric '{metric}' but column '{y_col}' is unavailable. "
            "Provide --capital and --risk-pct-per-trade, or use --plot-metric r."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for (rank_by, window, rank), group in curves.groupby(["rank_by", "window", "rank"], sort=True):
        formula = str(group["formula"].iloc[0]) if "formula" in group.columns and not group.empty else ""
        title = f"Equity Curve (rank_by={rank_by}, window={window}) Rank {rank}"
        if formula:
            title = f"{title}\n{formula}"

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(group[x_col], group[y_col], linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        filename = f"{rank_by}_{window}_rank_{int(rank):02d}.png"
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def _parse_ranked_combos_from_text(text: str) -> list[tuple[int, str]]:
    ranked: list[tuple[int, str]] = []
    pattern = re.compile(r"^Rank\s+(?P<rank>\d+)\s*:\s*(?P<expr>.+?)\s*$", re.IGNORECASE)
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        match = pattern.match(raw)
        if not match:
            continue
        rank = int(match.group("rank"))
        expr = match.group("expr").strip()
        if expr:
            ranked.append((rank, expr))
    ranked.sort(key=lambda item: item[0])
    return ranked


FORMULAS_PATH = REPO_ROOT / "scripts" / "kinetic_formulas_ranked.txt"

try:
    _formulas_text = FORMULAS_PATH.read_text()
except FileNotFoundError:  # pragma: no cover
    _formulas_text = ""
RANKED_COMBOS = _parse_ranked_combos_from_text(_formulas_text)
COMBOS = [expr for _rank, expr in RANKED_COMBOS]


def fmt_int(x: int) -> str:
    return f"{x:,}"


_OPS = ("<=", ">=", "==", "!=", "<", ">", "=")


@dataclass(frozen=True)
class ParsedClause:
    left: str
    op: str | None
    op_original: str | None
    right: str | None
    rhs_kind: str
    rhs_value: float | None = None


class PredicateMaskCache:
    """
    Cache clause-level boolean masks for AND-only Barsmith expressions.

    This speeds up evaluation when many expressions share the same predicates.
    """

    def __init__(self, df: pd.DataFrame, *, threshold_name_map: dict[str, float] | None = None) -> None:
        self._df = df
        self._threshold_name_map = threshold_name_map or {}
        self._float_col_cache: dict[str, np.ndarray] = {}
        self._clause_mask_cache: dict[ParsedClause, np.ndarray] = {}
        self._expr_cache: dict[str, list[ParsedClause]] = {}

    def _get_float_col(self, col: str) -> np.ndarray:
        cached = self._float_col_cache.get(col)
        if cached is not None:
            return cached
        if col not in self._df.columns:
            raise KeyError(f"Column '{col}' not found")
        values = pd.to_numeric(self._df[col], errors="coerce").to_numpy(dtype=float, copy=False)
        self._float_col_cache[col] = values
        return values

    def _parse_clause(self, clause: str) -> ParsedClause:
        clause = clause.strip()
        if not clause:
            raise ValueError("Empty clause")

        op_found: str | None = None
        for candidate in _OPS:
            if candidate in clause:
                op_found = candidate
                break

        if op_found is None:
            col = clause
            if col not in self._df.columns:
                raise KeyError(f"Column '{col}' not found for clause '{clause}'")
            return ParsedClause(left=col, op=None, op_original=None, right=None, rhs_kind="flag", rhs_value=None)

        left, right = [p.strip() for p in clause.split(op_found, 1)]
        if left not in self._df.columns:
            raise KeyError(f"Column '{left}' not found for clause '{clause}'")

        op_eval = "==" if op_found in ("==", "=") else op_found
        if right in self._df.columns:
            return ParsedClause(left=left, op=op_eval, op_original=op_found, right=right, rhs_kind="col", rhs_value=None)

        rhs_val = self._threshold_name_map.get(f"{left}{op_found}{right}")
        if rhs_val is None:
            try:
                rhs_val = float(right)
            except ValueError as exc:
                raise ValueError(f"Unsupported RHS '{right}' in clause '{clause}'") from exc

        return ParsedClause(left=left, op=op_eval, op_original=op_found, right=right, rhs_kind="const", rhs_value=float(rhs_val))

    def _parse_expr(self, expr: str) -> list[ParsedClause]:
        cached = self._expr_cache.get(expr)
        if cached is not None:
            return cached
        clauses = [c.strip() for c in expr.split("&&") if c.strip()]
        parsed = [self._parse_clause(c) for c in clauses]
        self._expr_cache[expr] = parsed
        return parsed

    @staticmethod
    def _apply_op(left: np.ndarray, right: np.ndarray | float, op: str) -> np.ndarray:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise ValueError(f"Unsupported operator '{op}'")

    def _mask_for_clause(self, clause: ParsedClause) -> np.ndarray:
        cached = self._clause_mask_cache.get(clause)
        if cached is not None:
            return cached

        left = self._get_float_col(clause.left)
        if clause.rhs_kind == "flag":
            mask = left > 0.5
        elif clause.rhs_kind == "col":
            if clause.right is None:
                raise ValueError("Missing RHS column for feature-vs-feature predicate")
            right = self._get_float_col(clause.right)
            if clause.op is None:
                raise ValueError("Missing operator for feature-vs-feature predicate")
            mask = self._apply_op(left, right, clause.op)
        elif clause.rhs_kind == "const":
            if clause.rhs_value is None:
                raise ValueError("Missing RHS constant for feature-vs-const predicate")
            if clause.op is None:
                raise ValueError("Missing operator for feature-vs-const predicate")
            mask = self._apply_op(left, float(clause.rhs_value), clause.op)
        else:
            raise ValueError(f"Unsupported rhs_kind '{clause.rhs_kind}'")

        mask = np.asarray(mask, dtype=bool)
        self._clause_mask_cache[clause] = mask
        return mask

    def eval_expr(self, expr: str) -> np.ndarray:
        parsed = self._parse_expr(expr)
        if not parsed:
            return np.ones((int(len(self._df)),), dtype=bool)

        out: np.ndarray | None = None
        for clause in parsed:
            clause_mask = self._mask_for_clause(clause)
            if out is None:
                out = clause_mask.copy()
            else:
                out &= clause_mask
                if not bool(out.any()):
                    break
        return out if out is not None else np.ones((int(len(self._df)),), dtype=bool)


def build_combo_mask(df: pd.DataFrame, expr: str, *, threshold_name_map: dict[str, float] | None = None) -> pd.Series:
    """
    Build boolean mask for an AND-only Barsmith-style expression.

    Supports:
      - Boolean flags: 'higher_low', 'is_kf_consolidation', 'consecutive_red_2', ...
      - Feature vs constant: 'upper_shadow_ratio>0.6', 'kf_atr>3.0', ...
      - Feature vs feature: '20ema>kf_smooth', '50ema>close', ...
    """
    mask = pd.Series(True, index=df.index)
    clauses = [c.strip() for c in expr.split("&&") if c.strip()]

    for clause in clauses:
        op = None
        for candidate in ("<=", ">=", "==", "!=", "<", ">", "="):
            if candidate in clause:
                op = candidate
                break

        if op is None:
            col = clause
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found for clause '{clause}'")
            cond = df[col] > 0.5
        else:
            left, right = [p.strip() for p in clause.split(op, 1)]
            if left not in df.columns:
                raise KeyError(f"Column '{left}' not found for clause '{clause}'")

            if right in df.columns:
                rhs = df[right]
            else:
                rhs_val = None
                if threshold_name_map is not None:
                    rhs_val = threshold_name_map.get(f"{left}{op}{right}")
                if rhs_val is None:
                    try:
                        rhs_val = float(right)
                    except ValueError as exc:
                        raise ValueError(f"Unsupported RHS '{right}' in clause '{clause}'") from exc
                rhs = rhs_val

            series = df[left]
            if op in ("==", "="):
                cond = series == rhs
            elif op == "!=":
                cond = series != rhs
            elif op == "<":
                cond = series < rhs
            elif op == "<=":
                cond = series <= rhs
            elif op == ">":
                cond = series > rhs
            elif op == ">=":
                cond = series >= rhs
            else:
                raise ValueError(f"Unsupported operator '{op}' in clause '{clause}'")

        mask &= cond

    return mask


def compute_streaks(rr: np.ndarray) -> tuple[int, int, float, float]:
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    win_lengths: list[int] = []
    loss_lengths: list[int] = []

    for v in rr:
        if v > 0:
            cur_win += 1
            if cur_loss > 0:
                loss_lengths.append(cur_loss)
                max_loss = max(max_loss, cur_loss)
                cur_loss = 0
        elif v < 0:
            cur_loss += 1
            if cur_win > 0:
                win_lengths.append(cur_win)
                max_win = max(max_win, cur_win)
                cur_win = 0
        else:
            if cur_win > 0:
                win_lengths.append(cur_win)
                max_win = max(max_win, cur_win)
                cur_win = 0
            if cur_loss > 0:
                loss_lengths.append(cur_loss)
                max_loss = max(max_loss, cur_loss)
                cur_loss = 0

    if cur_win > 0:
        win_lengths.append(cur_win)
        max_win = max(max_win, cur_win)
    if cur_loss > 0:
        loss_lengths.append(cur_loss)
        max_loss = max(max_loss, cur_loss)

    avg_win = float(np.mean(win_lengths)) if win_lengths else 0.0
    avg_loss = float(np.mean(loss_lengths)) if loss_lengths else 0.0
    return max_win, max_loss, avg_win, avg_loss


def resolve_exit_i_column(df: pd.DataFrame, target: str) -> str | None:
    for candidate in (f"{target}_exit_i", f"{target}_exit_i_long", f"{target}_exit_i_short"):
        if candidate in df.columns:
            return candidate
    return None


def _parse_exit_indices(values: pd.Series, *, expected_len: int) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full((expected_len,), -1, dtype=np.int64)
    valid = np.isfinite(raw) & (raw >= 0.0)
    if bool(valid.any()):
        out[valid] = raw[valid].astype(np.int64)
    return out


def select_trade_indices_no_stacking(
    combo_mask: np.ndarray,
    eligible_mask: np.ndarray,
    rr_finite: np.ndarray,
    exit_indices: np.ndarray,
    *,
    entry_abs_indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Enforce Barsmith Rust's `--stacking-mode no-stacking` semantics.
    """
    hits = np.flatnonzero(combo_mask)
    if hits.size == 0:
        return np.empty((0,), dtype=np.int64)

    selected: list[int] = []
    if entry_abs_indices is None:
        entry_abs_indices = np.arange(int(combo_mask.size), dtype=np.int64)

    next_free_abs_idx = int(entry_abs_indices[0]) if entry_abs_indices.size else 0
    for idx in hits:
        idx = int(idx)
        abs_idx = int(entry_abs_indices[idx])
        if abs_idx < next_free_abs_idx:
            continue
        if not eligible_mask[idx] or not rr_finite[idx]:
            continue

        selected.append(idx)
        exit_i = int(exit_indices[idx])
        if exit_i < 0 or exit_i <= abs_idx:
            candidate = abs_idx + 1
        else:
            candidate = exit_i
        if candidate > next_free_abs_idx:
            next_free_abs_idx = candidate

    if not selected:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(selected, dtype=np.int64)


def compute_risk_per_contract_dollar_series(
    sub: pd.DataFrame,
    *,
    position_sizing: str,
    stop_distance_column: str | None,
    stop_distance_unit: str,
    point_value: float | None,
    tick_value: float | None,
) -> np.ndarray | None:
    position_sizing_norm = str(position_sizing or "fractional").strip().lower()
    if position_sizing_norm != "contracts":
        return None

    if not stop_distance_column:
        raise ValueError("--stop-distance-column is required for --position-sizing contracts")
    if stop_distance_column not in sub.columns:
        raise KeyError(f"Stop distance column '{stop_distance_column}' not found in engineered CSV.")

    unit = str(stop_distance_unit or "points").strip().lower()
    if unit not in {"points", "ticks"}:
        raise ValueError(f"Unknown stop_distance_unit: {stop_distance_unit!r}")

    multiplier: float | None
    if unit == "points":
        multiplier = point_value
    else:
        multiplier = tick_value
    if multiplier is None or not np.isfinite(float(multiplier)) or float(multiplier) <= 0.0:
        raise ValueError(f"Invalid multiplier for stop_distance_unit={unit}: {multiplier}")

    stop = pd.to_numeric(sub[stop_distance_column], errors="coerce").to_numpy(dtype=float, copy=False)
    rpc = stop * float(multiplier)
    rpc[~np.isfinite(rpc) | (rpc <= 0.0)] = np.nan
    return rpc


def compute_window_results(
    df: pd.DataFrame,
    mask: pd.Series,
    capital_dollar: float,
    risk_pct_per_trade: float,
    cost_per_trade_dollar: float | None,
    *,
    position_sizing: str,
    stop_distance_column: str | None,
    stop_distance_unit: str,
    point_value: float | None,
    tick_value: float | None,
    min_contracts: int = 1,
    max_contracts: int | None = None,
    margin_per_contract_dollar: float | None = None,
    target_col: str,
    rr_col: str,
    eligible_col: str | None,
    stacking_mode: str,
    exit_i_col: str | None,
    max_drawdown_limit: float | None = None,
    min_calmar: float | None = None,
    threshold_name_map: dict[str, float] | None = None,
) -> tuple[int, list[dict]]:
    sub = df.loc[mask]
    if sub.empty:
        return 0, []

    if target_col not in sub.columns:
        raise KeyError(f"Target column '{target_col}' not found in engineered CSV.")
    if rr_col not in sub.columns:
        raise KeyError(f"RR column '{rr_col}' not found in engineered CSV.")

    rr_series = sub[rr_col].astype(float)
    target_series = sub[target_col]
    if target_series.dtype == bool:
        target_bool = target_series.fillna(False)
    else:
        target_bool = target_series.fillna(0.0).astype(float) > 0.5

    if eligible_col is not None and eligible_col in sub.columns:
        eligible_series = sub[eligible_col]
        if eligible_series.dtype == bool:
            eligible_mask = eligible_series.fillna(False)
        else:
            eligible_mask = eligible_series.fillna(0.0).astype(float) > 0.5
    else:
        eligible_mask = pd.Series(True, index=sub.index)

    exit_indices: np.ndarray | None = None
    if stacking_mode == "no-stacking":
        if not exit_i_col or exit_i_col not in sub.columns:
            raise SystemExit(
                f"--stacking-mode no-stacking requires an exit index column in the engineered CSV (expected '{target_col}_exit_i'). "
                "Re-generate the prepared CSV using a newer Barsmith custom_rs, or rerun with --stacking-mode stacking."
            )
        exit_indices = _parse_exit_indices(sub[exit_i_col], expected_len=len(sub))
    dataset_bars = len(sub)

    # Infer time horizon in years for this window from timestamps.
    try:
        ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        ts_valid = ts.dropna()
        if len(ts_valid) >= 2:
            delta = ts_valid.max() - ts_valid.min()
            years = max(delta.total_seconds() / (365.25 * 24 * 3600), 1e-9)
        else:
            years = 1.0
    except Exception:
        years = 1.0

    # Map 1R to dollars via capital and risk%.
    dollars_per_r = capital_dollar * (risk_pct_per_trade / 100.0) if capital_dollar > 0.0 and risk_pct_per_trade > 0.0 else 0.0
    position_sizing_norm = str(position_sizing or "fractional").strip().lower()
    if position_sizing_norm not in {"fractional", "contracts"}:
        raise ValueError(f"Unknown position_sizing: {position_sizing!r}")

    cost_per_trade_r = None
    if position_sizing_norm == "fractional" and cost_per_trade_dollar is not None and dollars_per_r > 0.0:
        cost_per_trade_r = cost_per_trade_dollar / dollars_per_r

    rr_values = rr_series.to_numpy(dtype=float, copy=False)
    rr_finite_np = np.isfinite(rr_values)

    rpc_values = compute_risk_per_contract_dollar_series(
        sub,
        position_sizing=position_sizing_norm,
        stop_distance_column=stop_distance_column,
        stop_distance_unit=stop_distance_unit,
        point_value=point_value,
        tick_value=tick_value,
    )
    if rpc_values is not None:
        rr_finite_np = rr_finite_np & np.isfinite(rpc_values) & (rpc_values > 0.0)
    eligible_np = eligible_mask.to_numpy(dtype=bool, copy=False)
    target_np = target_bool.to_numpy(dtype=bool, copy=False)
    entry_abs_indices = pd.to_numeric(sub.index.to_series(), errors="coerce").fillna(-1).to_numpy(dtype=np.int64, copy=False)
    if np.any(entry_abs_indices < 0):
        entry_abs_indices = np.arange(int(len(sub)), dtype=np.int64)

    predicate_cache = PredicateMaskCache(sub, threshold_name_map=threshold_name_map)
    combo_results: list[dict] = []

    for rank, expr in RANKED_COMBOS:
        combo_np = predicate_cache.eval_expr(expr)
        base_count = int(combo_np.sum())
        if base_count == 0:
            combo_results.append(
                {
                    "rank": rank,
                    "expr": expr,
                    "base_count": base_count,
                    "trades": 0,
                    "metrics": None,
                }
            )
            continue

        if stacking_mode == "no-stacking":
            if exit_indices is None:
                raise RuntimeError("exit_indices not initialized for no-stacking mode")
            trade_indices = select_trade_indices_no_stacking(
                combo_np,
                eligible_np,
                rr_finite_np,
                exit_indices,
                entry_abs_indices=entry_abs_indices,
            )
        else:
            trade_mask_np = combo_np & eligible_np & rr_finite_np
            trade_indices = np.flatnonzero(trade_mask_np).astype(np.int64, copy=False)

        trades = int(trade_indices.size)
        if trades == 0:
            combo_results.append(
                {
                    "rank": rank,
                    "expr": expr,
                    "base_count": base_count,
                    "trades": trades,
                    "metrics": None,
                }
            )
            continue

        rr_raw = rr_values[trade_indices]
        labels = target_np[trade_indices]

        if position_sizing_norm == "contracts":
            if rpc_values is None:
                raise RuntimeError("rpc_values must be computed for contracts sizing mode")
            rpc_trade = rpc_values[trade_indices]
            if cost_per_trade_dollar is not None and float(cost_per_trade_dollar) != 0.0:
                rr = rr_raw - (float(cost_per_trade_dollar) / rpc_trade)
            else:
                rr = rr_raw
            metrics = compute_rust_style_stats(
                rr,
                labels,
                capital_dollar,
                risk_pct_per_trade,
                years,
                position_sizing="contracts",
                risk_per_contract_dollar=rpc_trade,
                min_contracts=min_contracts,
                max_contracts=max_contracts,
                margin_per_contract_dollar=margin_per_contract_dollar,
            )
        else:
            # Apply per-trade cost in R, as Barsmith Rust does (rr_net = rr_raw - cost_per_trade_r).
            if cost_per_trade_r is not None and cost_per_trade_r > 0.0:
                rr = rr_raw - float(cost_per_trade_r)
            else:
                rr = rr_raw
            metrics = compute_rust_style_stats(
                rr,
                labels,
                capital_dollar,
                risk_pct_per_trade,
                years,
                position_sizing="fractional",
            )

        # Optional R-space max drawdown filter: mirror Barsmith Rust's
        # --max-drawdown pruning. If a limit is provided and the combo exceeds
        # it, skip this combo entirely in the window results.
        if max_drawdown_limit is not None and metrics["max_drawdown"] > max_drawdown_limit:
            continue
        # Optional minimum equity Calmar filter: require equity Calmar
        # (CAGR / max_drawdown_pct_equity) to be at least min_calmar when provided.
        if min_calmar is not None and metrics["calmar_equity"] < min_calmar:
            continue
        metrics["dollars_per_r"] = dollars_per_r
        metrics["cost_per_trade_dollar"] = cost_per_trade_dollar
        metrics["cost_per_trade_r"] = cost_per_trade_r
        metrics["position_sizing"] = position_sizing_norm
        if position_sizing_norm == "contracts":
            metrics["stop_distance_column"] = stop_distance_column
            metrics["stop_distance_unit"] = stop_distance_unit
            metrics["min_contracts"] = int(max(int(min_contracts), 1))
            metrics["max_contracts"] = int(max_contracts) if max_contracts is not None else None
        wins_net = metrics["profitable_bars"]
        win_rate = metrics["win_rate"]

        median_rr = metrics["median_rr"]
        p05_rr = metrics["p05_rr"]
        p95_rr = metrics["p95_rr"]
        avg_loss = metrics["avg_losing_rr"]
        avg_win = metrics["avg_winning_rr"]
        max_win = metrics["largest_win"]
        max_loss = metrics["largest_loss"]
        max_w_streak, max_l_streak, avg_w_streak, avg_l_streak = compute_streaks(rr)
        density = trades / (dataset_bars / 1000.0) if dataset_bars > 0 else 0.0
        recall_pct = base_count / dataset_bars * 100.0 if dataset_bars > 0 else 0.0

        combo_results.append(
            {
                "rank": rank,
                "expr": expr,
                "base_count": base_count,
                "trades": trades,
                "mask_hits": base_count,
                "wins_net": wins_net,
                "win_rate": win_rate,
                "label_hits": metrics["label_hits"],
                "label_misses": metrics["label_misses"],
                "label_hit_rate": metrics["label_hit_rate"],
                "metrics": metrics,
                "median_rr": median_rr,
                "p05_rr": p05_rr,
                "p95_rr": p95_rr,
                "avg_loss": avg_loss,
                "avg_win": avg_win,
                "max_win": max_win,
                "max_loss": max_loss,
                "max_w_streak": max_w_streak,
                "max_l_streak": max_l_streak,
                "avg_w_streak": avg_w_streak,
                "avg_l_streak": avg_l_streak,
                "density": density,
                "recall_pct": recall_pct,
            }
        )

    return dataset_bars, combo_results


def _classify_sample(total_bars: int) -> str:
    """
    Mirror the Rust classify_sample thresholds.
    """
    if total_bars >= 100:
        return "excellent"
    if total_bars >= 50:
        return "good"
    if total_bars >= 30:
        return "fair"
    return "poor"


def _percentile_triplet(rr: np.ndarray) -> tuple[float, float, float]:
    """
    Approximate Rust's percentile_triplet (median, p05, p95) using NumPy.
    """
    if rr.size == 0:
        return 0.0, 0.0, 0.0
    sorted_rr = np.sort(rr)
    n = sorted_rr.size
    mid = n // 2
    if n % 2 == 0:
        mid_val = sorted_rr[mid]
        max_left = float(sorted_rr[:mid].max())
        median = (max_left + mid_val) / 2.0
    else:
        median = float(sorted_rr[mid])

    if n == 1:
        return median, float(sorted_rr[0]), float(sorted_rr[0])

    len_minus_one = n - 1
    idx_p05 = int(round(len_minus_one * 0.05))
    idx_p95 = int(round(len_minus_one * 0.95))
    idx_p05 = max(0, min(idx_p05, n - 1))
    idx_p95 = max(0, min(idx_p95, n - 1))
    p05 = float(sorted_rr[idx_p05])
    p95 = float(sorted_rr[idx_p95])
    return float(median), p05, p95


def compute_rust_style_stats(
    rr: np.ndarray,
    labels: np.ndarray,
    capital_dollar: float,
    risk_pct_per_trade: float,
    equity_time_years: float,
    *,
    position_sizing: str = "fractional",
    risk_per_contract_dollar: np.ndarray | None = None,
    min_contracts: int = 1,
    max_contracts: int | None = None,
    margin_per_contract_dollar: float | None = None,
) -> dict:
    """
    Compute statistics aligned with barsmith_rs::compute_full_statistics for R-space metrics.
    """
    n = int(rr.size)
    stats = {
        "total_bars": n,
        "total_return": 0.0,
        "expectancy": 0.0,
        "profit_factor": 0.0,
        "avg_winning_rr": 0.0,
        "avg_losing_rr": 0.0,
        "win_loss_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown": 0.0,
        "ulcer_index": 0.0,
        "pain_ratio": 0.0,
        "median_rr": 0.0,
        "p05_rr": 0.0,
        "p95_rr": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "profitable_bars": 0,
        "unprofitable_bars": 0,
        "win_rate": 0.0,
        "label_hits": 0,
        "label_misses": 0,
        "label_hit_rate": 0.0,
        "sample_quality": _classify_sample(n),
        # Equity-curve metrics.
        "final_capital": 0.0,
        "total_return_pct": 0.0,
        "cagr_pct": 0.0,
        "max_drawdown_pct_equity": 0.0,
        "calmar_equity": 0.0,
        "sharpe_equity": 0.0,
        "sortino_equity": 0.0,
    }

    if n == 0:
        return stats

    rr = rr.astype(float)
    total_return = float(rr.sum())
    rr_sq = rr * rr
    sum_sq = float(rr_sq.sum())

    profit_mask = rr > 0.0
    loss_mask = rr < 0.0
    profit_sum = float(rr[profit_mask].sum())
    loss_sum_abs = float((-rr[loss_mask]).sum())
    profit_count = int(profit_mask.sum())
    loss_count = int(loss_mask.sum())

    downside_sum_sq = float(rr_sq[loss_mask].sum())
    downside_count = loss_count

    # Drawdown-related accumulators in R.
    equity = 0.0
    equity_peak = 0.0
    max_drawdown = 0.0
    dd_sum = 0.0
    dd_pct_sq_sum = 0.0
    dd_count = 0
    have_nonzero_peak = False

    # Streak-related accumulators.
    current_win_streak = 0
    current_loss_streak = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    total_win_streak_len = 0
    total_loss_streak_len = 0
    win_streak_count = 0
    loss_streak_count = 0

    largest_win = 0.0
    largest_loss = 0.0

    eps = np.finfo(float).eps

    for r in rr:
        if r > 0.0:
            if r > largest_win:
                largest_win = r
        elif r < 0.0:
            abs_r = -r
            if abs_r > largest_loss:
                largest_loss = abs_r

        equity += r
        if equity > equity_peak:
            equity_peak = equity
            if abs(equity_peak) >= eps:
                have_nonzero_peak = True
        dd = equity - equity_peak
        dd_sum += dd
        dd_count += 1
        if dd < max_drawdown:
            max_drawdown = dd
        if abs(equity_peak) >= eps:
            pct = (dd / (equity_peak + eps)) * 100.0
            dd_pct_sq_sum += pct * pct

        # Streak tracking: wins > 0, losses < 0, zero breaks both.
        if r > 0.0:
            if current_loss_streak > 0:
                total_loss_streak_len += current_loss_streak
                loss_streak_count += 1
                if current_loss_streak > max_consecutive_losses:
                    max_consecutive_losses = current_loss_streak
                current_loss_streak = 0
            current_win_streak += 1
        elif r < 0.0:
            if current_win_streak > 0:
                total_win_streak_len += current_win_streak
                win_streak_count += 1
                if current_win_streak > max_consecutive_wins:
                    max_consecutive_wins = current_win_streak
                current_win_streak = 0
            current_loss_streak += 1
        else:
            if current_win_streak > 0:
                total_win_streak_len += current_win_streak
                win_streak_count += 1
                if current_win_streak > max_consecutive_wins:
                    max_consecutive_wins = current_win_streak
                current_win_streak = 0
            if current_loss_streak > 0:
                total_loss_streak_len += current_loss_streak
                loss_streak_count += 1
                if current_loss_streak > max_consecutive_losses:
                    max_consecutive_losses = current_loss_streak
                current_loss_streak = 0

    # Flush trailing streaks.
    if current_win_streak > 0:
        total_win_streak_len += current_win_streak
        win_streak_count += 1
        if current_win_streak > max_consecutive_wins:
            max_consecutive_wins = current_win_streak
    if current_loss_streak > 0:
        total_loss_streak_len += current_loss_streak
        loss_streak_count += 1
        if current_loss_streak > max_consecutive_losses:
            max_consecutive_losses = current_loss_streak

    expectancy_raw = total_return / n if n > 0 else 0.0
    stats["expectancy"] = float(expectancy_raw)

    # Profit factor / win-loss geometry.
    if loss_sum_abs > 0.0:
        profit_factor_raw = profit_sum / loss_sum_abs
    elif profit_sum > 0.0:
        profit_factor_raw = float("inf")
    else:
        profit_factor_raw = 0.0
    stats["profit_factor"] = float(profit_factor_raw)

    if profit_count > 0:
        avg_winning_rr = profit_sum / profit_count
    else:
        avg_winning_rr = 0.0
    if loss_count > 0:
        avg_loss_abs = loss_sum_abs / loss_count
        avg_losing_rr = -avg_loss_abs
    else:
        avg_loss_abs = 0.0
        avg_losing_rr = 0.0
    if avg_loss_abs > 0.0:
        win_loss_ratio_raw = avg_winning_rr / avg_loss_abs
    elif avg_winning_rr > 0.0:
        win_loss_ratio_raw = float("inf")
    else:
        win_loss_ratio_raw = 0.0

    stats["avg_winning_rr"] = float(avg_winning_rr)
    stats["avg_losing_rr"] = float(avg_losing_rr)
    stats["win_loss_ratio"] = float(win_loss_ratio_raw)

    # Sharpe and Sortino (R-based).
    if n > 0:
        mean = expectancy_raw
        mean_sq = sum_sq / n
        var = max(mean_sq - mean * mean, 0.0)
        returns_std = var**0.5
    else:
        returns_std = 0.0

    if returns_std > 0.0:
        sharpe_ratio = expectancy_raw / returns_std
    else:
        sharpe_ratio = 0.0

    if downside_count > 0:
        downside_std = (downside_sum_sq / downside_count) ** 0.5
    else:
        downside_std = 0.0

    if downside_std > 0.0:
        sortino_ratio = expectancy_raw / downside_std
    elif expectancy_raw > 0.0 and downside_count == 0:
        sortino_ratio = float("inf")
    else:
        sortino_ratio = 0.0

    # Drawdown-derived metrics.
    max_drawdown_abs = abs(max_drawdown)
    if not have_nonzero_peak or dd_count == 0:
        ulcer_index = 0.0
    else:
        ulcer_index = (dd_pct_sq_sum / dd_count) ** 0.5
    avg_drawdown = dd_sum / dd_count if dd_count > 0 else 0.0
    if avg_drawdown < 0.0:
        pain_ratio = total_return / abs(avg_drawdown)
    else:
        pain_ratio = 0.0
    if max_drawdown_abs > 0.0:
        calmar_ratio_raw = total_return / max_drawdown_abs
    elif total_return > 0.0:
        calmar_ratio_raw = float("inf")
    else:
        calmar_ratio_raw = 0.0

    stats["max_drawdown"] = float(max_drawdown_abs)
    stats["ulcer_index"] = float(ulcer_index)
    stats["pain_ratio"] = float(pain_ratio)
    stats["calmar_ratio"] = float(calmar_ratio_raw)

    # Distribution shape.
    median_rr, p05_rr, p95_rr = _percentile_triplet(rr)
    stats["median_rr"] = float(median_rr)
    stats["p05_rr"] = float(p05_rr)
    stats["p95_rr"] = float(p95_rr)
    stats["largest_win"] = float(largest_win)
    stats["largest_loss"] = float(largest_loss)

    # Win rate (net-R) and label hit-rate.
    profitable_bars = profit_count
    unprofitable_bars = loss_count
    win_rate = (profitable_bars / n) * 100.0 if n > 0 else 0.0
    labels = labels.astype(bool)
    label_hits = int(labels.sum())
    label_misses = n - label_hits
    label_hit_rate = (label_hits / n) * 100.0 if n > 0 else 0.0

    stats["total_return"] = float(total_return)
    stats["profitable_bars"] = profitable_bars
    stats["unprofitable_bars"] = unprofitable_bars
    stats["win_rate"] = float(win_rate)
    stats["label_hits"] = label_hits
    stats["label_misses"] = label_misses
    stats["label_hit_rate"] = float(label_hit_rate)

    # Equity-curve metrics driven by capital and risk%.
    capital_0 = float(capital_dollar)
    risk_pct = float(risk_pct_per_trade)
    if capital_0 > 0.0 and risk_pct > 0.0 and n > 0:
        capital = capital_0
        peak_capital = capital_0
        eq_ret_sum = 0.0
        eq_ret_sq_sum = 0.0
        downside_sq_sum_eq = 0.0
        downside_count_eq = 0
        max_drawdown_pct_equity = 0.0

        position_sizing_norm = str(position_sizing or "fractional").strip().lower()
        if position_sizing_norm not in {"fractional", "contracts"}:
            raise ValueError(f"Unknown position_sizing: {position_sizing!r}")

        rpc = None
        min_contracts_i = max(int(min_contracts), 1)
        max_contracts_i = int(max_contracts) if max_contracts is not None else None
        if position_sizing_norm == "contracts":
            if risk_per_contract_dollar is None:
                raise ValueError("risk_per_contract_dollar is required for position_sizing='contracts'")
            rpc = np.asarray(risk_per_contract_dollar, dtype=float)
            if int(rpc.size) != n:
                raise ValueError(f"risk_per_contract_dollar length mismatch: expected {n}, got {int(rpc.size)}")

        for i, r in enumerate(rr):
            if position_sizing_norm == "fractional":
                risk_i = capital * (risk_pct / 100.0)
                pnl = float(r) * float(risk_i)
            else:
                rpc_i = float(rpc[i]) if rpc is not None else float("nan")
                if not np.isfinite(rpc_i) or rpc_i <= 0.0 or not np.isfinite(float(r)):
                    pnl = 0.0
                else:
                    risk_budget = capital * (risk_pct / 100.0)
                    raw = np.floor(risk_budget / rpc_i)
                    contracts = int(raw) if np.isfinite(raw) and raw >= 0.0 else 0
                    if contracts < min_contracts_i:
                        contracts = min_contracts_i
                    if max_contracts_i is not None:
                        contracts = min(contracts, max_contracts_i)
                    if (
                        margin_per_contract_dollar is not None
                        and np.isfinite(float(margin_per_contract_dollar))
                        and float(margin_per_contract_dollar) > 0.0
                    ):
                        cap = np.floor(capital / float(margin_per_contract_dollar))
                        if np.isfinite(cap) and cap >= 0.0:
                            contracts = min(contracts, int(cap))
                    if contracts <= 0:
                        pnl = 0.0
                    else:
                        pnl = float(r) * rpc_i * float(contracts)

            next_capital = capital + pnl
            if capital > 0.0:
                ret = (next_capital / capital) - 1.0
            else:
                ret = 0.0

            eq_ret_sum += ret
            eq_ret_sq_sum += ret * ret
            if ret < 0.0:
                downside_sq_sum_eq += ret * ret
                downside_count_eq += 1

            capital = next_capital
            if capital > peak_capital:
                peak_capital = capital
            if peak_capital > 0.0:
                dd_pct = ((capital - peak_capital) / peak_capital) * 100.0
                dd_mag = -dd_pct
                if dd_mag > max_drawdown_pct_equity:
                    max_drawdown_pct_equity = dd_mag

        final_capital = capital
        total_return_pct = 0.0
        if capital_0 > 0.0:
            total_return_pct = ((final_capital / capital_0) - 1.0) * 100.0

        years = float(equity_time_years)
        if years <= 0.0:
            years = 1e-9
        growth = final_capital / capital_0 if capital_0 > 0.0 else 1.0
        if years > 0.0 and np.isfinite(growth) and growth > 0.0:
            cagr_pct = (growth ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr_pct = total_return_pct

        if max_drawdown_pct_equity > 0.0:
            calmar_equity = cagr_pct / max_drawdown_pct_equity
        elif cagr_pct > 0.0:
            calmar_equity = float("inf")
        else:
            calmar_equity = 0.0

        n_returns = float(n)
        mean_eq = eq_ret_sum / n_returns
        var_eq = max((eq_ret_sq_sum / n_returns) - mean_eq * mean_eq, 0.0)
        std_eq = var_eq**0.5

        if downside_count_eq > 0:
            downside_std_eq = (downside_sq_sum_eq / float(downside_count_eq)) ** 0.5
        else:
            downside_std_eq = 0.0

        trades_per_year = max(n_returns / years, 1e-9)
        annual_scale = trades_per_year**0.5

        if std_eq > 0.0:
            sharpe_equity = (mean_eq / std_eq) * annual_scale
        else:
            sharpe_equity = 0.0

        if downside_std_eq > 0.0:
            sortino_equity = (mean_eq / downside_std_eq) * annual_scale
        elif downside_std_eq == 0.0 and mean_eq > 0.0:
            sortino_equity = float("inf")
        else:
            sortino_equity = 0.0

        stats["final_capital"] = float(final_capital)
        stats["total_return_pct"] = float(total_return_pct)
        stats["cagr_pct"] = float(cagr_pct)
        stats["max_drawdown_pct_equity"] = float(max_drawdown_pct_equity)
        stats["calmar_equity"] = float(calmar_equity)
        stats["sharpe_equity"] = float(sharpe_equity)
        stats["sortino_equity"] = float(sortino_equity)

    return stats


def sort_combo_results(
    results: list[dict],
    prev_ranks: dict[str, int] | None = None,
    *,
    primary_metric: str = "calmar_equity",
) -> list[dict]:
    """
    Sort combos primarily by a chosen metric (descending), and secondarily by
    previous rank (ascending) when provided.
    """

    def sort_key(result: dict) -> tuple[float, int]:
        if primary_metric == "frs":
            primary = float(result.get("frs")) if result.get("frs") is not None else float("-inf")
        elif primary_metric == "calmar_equity":
            metrics = result["metrics"]
            if metrics is None:
                primary = float("-inf")
            else:
                calmar_eq = float(metrics.get("calmar_equity", 0.0))
                if np.isinf(calmar_eq):
                    primary = float("inf")
                else:
                    primary = calmar_eq
        else:
            raise ValueError(f"Unknown primary_metric: {primary_metric}")

        # Negative for descending sort on primary.
        primary_key = -primary

        if prev_ranks is not None:
            secondary_key = prev_ranks.get(result["expr"], 10**9)
        else:
            secondary_key = 0

        return (primary_key, secondary_key)

    return sorted(results, key=sort_key)


def compute_buy_and_hold_benchmark(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    close_col: str = "close",
    timestamp_col: str = "timestamp",
    capital_dollar: float | None = None,
) -> dict[str, float | str | None]:
    sub = df.loc[mask]
    if sub.empty:
        return {"rows": 0, "start": None, "end": None}

    if close_col not in sub.columns:
        raise KeyError(f"Buy & hold requires '{close_col}' column in prepared CSV.")
    if timestamp_col not in sub.columns:
        raise KeyError(f"Buy & hold requires '{timestamp_col}' column in prepared CSV.")

    closes = pd.to_numeric(sub[close_col], errors="coerce")
    ts = pd.to_datetime(sub[timestamp_col], utc=True, errors="coerce").dt.tz_convert(None)
    valid = closes.notna() & ts.notna()
    if int(valid.sum()) < 2:
        return {"rows": int(len(sub)), "start": None, "end": None}

    closes_valid = closes[valid].to_numpy(dtype=float, copy=False)
    ts_valid = ts[valid]

    start_close = float(closes_valid[0])
    end_close = float(closes_valid[-1])
    if not np.isfinite(start_close) or start_close <= 0.0:
        return {"rows": int(len(sub)), "start": None, "end": None}

    equity = closes_valid / start_close
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / np.maximum(peak, np.finfo(float).eps) * 100.0

    total_return_pct = float((equity[-1] - 1.0) * 100.0)
    max_drawdown_pct = float(np.max(dd_pct)) if dd_pct.size else 0.0

    delta = ts_valid.iloc[-1] - ts_valid.iloc[0]
    years = float(delta.total_seconds() / (365.25 * 24 * 3600))
    years = max(years, 1e-9)

    growth = float(equity[-1])
    if np.isfinite(growth) and growth > 0.0:
        cagr_pct = float((growth ** (1.0 / years) - 1.0) * 100.0)
    else:
        cagr_pct = total_return_pct

    if max_drawdown_pct > 0.0:
        calmar = float(cagr_pct / max_drawdown_pct)
    elif cagr_pct > 0.0:
        calmar = float("inf")
    else:
        calmar = 0.0

    final_capital = None
    if capital_dollar is not None and float(capital_dollar) > 0.0:
        final_capital = float(capital_dollar) * float(growth)

    return {
        "rows": int(len(sub)),
        "start": str(ts_valid.iloc[0]),
        "end": str(ts_valid.iloc[-1]),
        "start_close": start_close,
        "end_close": end_close,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar": calmar,
        "final_capital": final_capital,
    }


def build_buy_and_hold_window_rows(
    df: pd.DataFrame,
    windows: list[dict],
    *,
    scope: str,
    capital_dollar: float | None,
) -> list[dict]:
    rows: list[dict] = []
    for w in windows:
        mask = w.get("mask")
        if mask is None:
            continue
        bh = compute_buy_and_hold_benchmark(df, mask, capital_dollar=capital_dollar)
        if bh.get("start") is None:
            continue
        rows.append(
            {
                "scope": scope,
                "expr": "__BUY_AND_HOLD__",
                "window_label": w.get("label"),
                "year": int(w.get("year")) if w.get("year") is not None else None,
                "rows": int(w.get("rows") or 0),
                "start": w.get("start"),
                "end": w.get("end"),
                # R-space columns are not applicable for buy & hold.
                "total_return_r": None,
                "max_drawdown_r": None,
                "trades": None,
                "calmar_r": None,
                "expectancy_r": None,
                # Equity columns.
                "total_return_pct": float(bh.get("total_return_pct") or 0.0),
                "cagr_pct": float(bh.get("cagr_pct") or 0.0),
                "max_drawdown_pct_equity": float(bh.get("max_drawdown_pct") or 0.0),
                "calmar_equity": float(bh.get("calmar") or 0.0),
                "final_capital": bh.get("final_capital"),
                "start_close": bh.get("start_close"),
                "end_close": bh.get("end_close"),
            }
        )
    return rows


def print_window_results(
    label: str,
    dataset_bars: int,
    combo_results: list[dict],
    prev_ranks: dict[str, int] | None = None,
    *,
    stacking_mode: str,
    buy_and_hold: dict[str, float | str | None] | None = None,
) -> None:
    print(f"\n=== Window: {label} (rows={dataset_bars}) ===")
    if dataset_bars == 0 or not combo_results:
        print("No rows in this window.")
        return

    if buy_and_hold is not None and buy_and_hold.get("start") is not None:
        bh_total = float(buy_and_hold.get("total_return_pct") or 0.0)
        bh_cagr = float(buy_and_hold.get("cagr_pct") or 0.0)
        bh_dd = float(buy_and_hold.get("max_drawdown_pct") or 0.0)
        bh_calmar = float(buy_and_hold.get("calmar") or 0.0)
        bh_start = str(buy_and_hold.get("start") or "")
        bh_end = str(buy_and_hold.get("end") or "")
        bh_final = buy_and_hold.get("final_capital")
        bh_final_str = None
        if isinstance(bh_final, (int, float)) and np.isfinite(float(bh_final)) and float(bh_final) > 0.0:
            bh_final_str = fmt_int(int(round(float(bh_final))))
        calmar_str = "Inf" if np.isinf(bh_calmar) else f"{bh_calmar:.2f}"

        print(
            f"Buy & Hold (close): {bh_start} -> {bh_end} | "
            f"Total {bh_total:.1f}% | CAGR {bh_cagr:.2f}% | "
            f"Max DD {bh_dd:.1f}% | Calmar {calmar_str}" + (f" | Final ${bh_final_str}" if bh_final_str else "")
        )

    for idx, result in enumerate(combo_results, start=1):
        expr = result["expr"]
        base_count = result["base_count"]
        trades = result["trades"]
        metrics = result["metrics"]

        print("\n" + "-" * 80)
        if prev_ranks is not None:
            prev_rank = prev_ranks.get(expr)
            if prev_rank is not None:
                print(f"Rank {idx} (prev {prev_rank}): {expr}")
            else:
                print(f"Rank {idx}: {expr}")
        else:
            print(f"Rank {idx}: {expr}")
        print(f"Bars matching combo mask: {base_count} ({base_count / dataset_bars * 100:.2f}% of window)")
        print(f"Trades (eligible & finite RR, {stacking_mode}): {fmt_int(trades)}")

        frs = result.get("frs")
        frs_components = result.get("frs_components")
        if frs is not None and isinstance(frs_components, dict):
            print(
                "  FRS: "
                f"{float(frs):.3f} | "
                f"P {float(frs_components.get('p', 0.0)):.2f} | "
                f"C+ {float(frs_components.get('c_plus', 0.0)):.2f} | "
                f"Tail {float(frs_components.get('tail_penalty', 0.0)):.2f} | "
                f"Stable {float(frs_components.get('stability', 0.0)):.2f} | "
                f"DD {float(frs_components.get('dd_min', 0.0)):.2f}/"
                f"{float(frs_components.get('dd_median', 0.0)):.2f}/"
                f"{float(frs_components.get('dd_mean', 0.0)):.2f}/"
                f"{float(frs_components.get('dd_max', 0.0)):.2f} | "
                f"R {float(frs_components.get('r_min', 0.0)):.2f}/"
                f"{float(frs_components.get('r_median', 0.0)):.2f}/"
                f"{float(frs_components.get('mu_r', 0.0)):.2f}/"
                f"{float(frs_components.get('r_max', 0.0)):.2f} | "
                f"Trade {float(frs_components.get('trade_score', 0.0)):.2f} "
                f"(k={int(frs_components.get('k', 0))}, Nmed={float(frs_components.get('n_med', 0.0)):.1f})"
            )

        if metrics is None or trades == 0:
            print("No trades after eligibility/RR filtering; skipping.")
            continue

        wins_net = result["wins_net"]
        win_rate = result["win_rate"]
        label_hits = result["label_hits"]
        label_hit_rate = result["label_hit_rate"]
        median_rr = result["median_rr"]
        p05_rr = result["p05_rr"]
        p95_rr = result["p95_rr"]
        avg_loss = result["avg_loss"]
        avg_win = result["avg_win"]
        max_win = result["max_win"]
        max_loss = result["max_loss"]
        max_w_streak = result["max_w_streak"]
        max_l_streak = result["max_l_streak"]
        avg_w_streak = result["avg_w_streak"]
        avg_l_streak = result["avg_l_streak"]
        density = result["density"]
        recall_pct = result["recall_pct"]

        pf_str = "Inf" if np.isinf(metrics["profit_factor"]) else f"{metrics['profit_factor']:.3f}"
        winloss_str = "Inf" if np.isinf(metrics["win_loss_ratio"]) else f"{metrics['win_loss_ratio']:.2f}"

        # Core R-space block (match Barsmith Rust formatting).
        print(f"  Win Rate: {win_rate:.2f}% " f"({fmt_int(wins_net)}/{fmt_int(trades)} bars)")
        print(f"  Target hit-rate: {label_hit_rate:.2f}% " f"({fmt_int(label_hits)}/{fmt_int(trades)} bars)")
        print(f"  Expectancy: {metrics['expectancy']:.3f}R | " f"Avg win: {avg_win:.3f}R | Avg loss: {avg_loss:.3f}R")
        print(
            f"  Total R: {metrics['total_return']:.1f}R | "
            f"Max DD: {metrics['max_drawdown']:.1f}R | "
            f"Profit factor: {pf_str}"
        )
        print(f"  R-dist: median {median_rr:.3f}R | " f"p05 {p05_rr:.3f}R | p95 {p95_rr:.3f}R | avg loss {avg_loss:.3f}R")

        # Cost model (if capital and cost are provided).
        position_sizing = str(metrics.get("position_sizing") or "fractional").strip().lower()
        dollars_per_r = float(metrics.get("dollars_per_r") or 0.0)
        cost_per_trade_r = metrics.get("cost_per_trade_r")
        cost_per_trade_dollar = metrics.get("cost_per_trade_dollar")
        if position_sizing == "contracts":
            if cost_per_trade_dollar is not None and float(cost_per_trade_dollar) > 0.0:
                print(f"  Cost model: ${float(cost_per_trade_dollar):.2f}/contract round-trip")
        else:
            if dollars_per_r > 0.0 and cost_per_trade_r is not None and cost_per_trade_r > 0.0:
                cost_dollar = cost_per_trade_r * dollars_per_r
                print(f"  Cost model: {cost_per_trade_r:.3f}R/trade (~${cost_dollar:.2f})")

        # Equity metrics (if capital-aware simulation is meaningful).
        final_capital = float(metrics.get("final_capital") or 0.0)
        total_return_pct = float(metrics.get("total_return_pct") or 0.0)
        if final_capital > 0.0 and total_return_pct != 0.0:
            cagr_pct = float(metrics.get("cagr_pct") or 0.0)
            max_dd_pct_eq = float(metrics.get("max_drawdown_pct_equity") or 0.0)
            calmar_equity = float(metrics.get("calmar_equity") or 0.0)
            sharpe_equity = float(metrics.get("sharpe_equity") or 0.0)
            sortino_equity = float(metrics.get("sortino_equity") or 0.0)
            final_capital_str = fmt_int(int(round(final_capital)))
            print(f"  Equity: Final ${final_capital_str} | Total {total_return_pct:.1f}% | CAGR {cagr_pct:.2f}%")
            print(f"  Equity DD: Max {max_dd_pct_eq:.1f}% | Calmar (equity): {calmar_equity:.2f}")
            print(f"  Equity Sharpe/Sortino: {sharpe_equity:.2f} / {sortino_equity:.2f}")

        # Remaining R-space summary (equity Calmar is printed above; here we show Win/Loss and drawdown-derived metrics).
        print(f"  Win/Loss: {winloss_str}")
        print(f"  Drawdown shape: Pain {metrics['pain_ratio']:.2f} | " f"Ulcer {metrics['ulcer_index']:.2f}")
        print(f"  Recall: {fmt_int(base_count)} / {fmt_int(dataset_bars)} bars " f"({recall_pct:.2f}% of dataset)")
        print(f"  Density: {density:.2f} trades/1000 bars")
        print(
            f"  Streaks W/L: {fmt_int(max_w_streak)}/{fmt_int(max_l_streak)} "
            f"(avg {avg_w_streak:.2f}/{avg_l_streak:.2f}) | "
            f"Largest win/loss: {max_win:.2f}R / {max_loss:.2f}R"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Barsmith kinetic combos on engineered dataset (pre vs forward).")
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Path to barsmith_prepared.csv produced by custom_rs::prepare_dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="2x_atr_tp_atr_stop",
        help="Target column name in the engineered CSV (e.g., 2x_atr_tp_atr_stop, 3x_atr_tp_atr_stop, atr_tp_atr_stop, highlow_or_atr).",
    )
    parser.add_argument(
        "--rr-column",
        type=str,
        default=None,
        help="Override RR column name (defaults to rr_<target>).",
    )
    parser.add_argument(
        "--stacking-mode",
        choices=["no-stacking", "stacking"],
        default="no-stacking",
        help="Trade stacking behavior. Default is no-stacking (Barsmith Rust default).",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2024-12-31",
        help="Cutoff date (inclusive) for in-sample window, e.g. 2024-12-31",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Starting capital in USD for equity simulation (matches Barsmith CLI default).",
    )
    parser.add_argument(
        "--risk-pct-per-trade",
        type=float,
        default=1.0,
        help="Risk percentage of current equity per trade for equity metrics.",
    )
    parser.add_argument(
        "--equity-curves-top-k",
        type=int,
        default=10,
        help="Export equity curves for the top K strategies per window (default: 10; set to 0 to disable).",
    )
    parser.add_argument(
        "--equity-curves-rank-by",
        choices=["post", "pre"],
        default="post",
        help="Which ranking list to select the top K strategies from (default: post).",
    )
    parser.add_argument(
        "--equity-curves-window",
        choices=["post", "pre", "both"],
        default="both",
        help="Which ranked window to export curves for (default: both).",
    )
    parser.add_argument(
        "--equity-curves-out",
        type=Path,
        default=None,
        help="Output CSV path for equity curves (default: <prepared_dir>/equity_curves_top10.csv).",
    )
    parser.add_argument(
        "--rank-by",
        choices=["calmar_equity", "frs"],
        default="frs",
        help="Primary ranking metric for the forward (post) window (default: calmar_equity).",
    )
    parser.add_argument(
        "--frs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Compute Forward Robustness Score (FRS) over anchored calendar-year windows "
        "(default: true). Use --no-frs to disable.",
    )
    parser.add_argument(
        "--frs-scope",
        "--frs-period",
        choices=["window", "post", "pre", "all"],
        default="all",
        help="Which rows to use for building the anchored calendar-year windows used by FRS. "
        "'window' computes FRS separately inside each printed window (pre vs post). (default: window).",
    )
    parser.add_argument(
        "--frs-nmin",
        type=int,
        default=30,
        help="Trade-count sanity floor N_min (median trades per forward window) for FRS TradeScore (default: 30).",
    )
    parser.add_argument(
        "--frs-alpha",
        type=float,
        default=2.0,
        help="FRS exponent alpha for consistency P (default: 2.0).",
    )
    parser.add_argument(
        "--frs-beta",
        type=float,
        default=2.0,
        help="FRS exponent beta for TailPenalty (default: 2.0).",
    )
    parser.add_argument(
        "--frs-gamma",
        type=float,
        default=1.0,
        help="FRS exponent gamma for Stability (default: 1.0).",
    )
    parser.add_argument(
        "--frs-delta",
        type=float,
        default=1.0,
        help="FRS exponent delta for TradeScore (default: 1.0).",
    )
    parser.add_argument(
        "--frs-out",
        type=Path,
        default=None,
        help="Optional output CSV path for FRS summary (one row per combo).",
    )
    parser.add_argument(
        "--frs-windows-out",
        type=Path,
        default=None,
        help="Optional output CSV path for FRS per-window breakdown (one row per combo per forward window).",
    )
    parser.add_argument(
        "--barsmith-threshold-parity",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resolve Barsmith feature-vs-constant predicate thresholds using Barsmith's feature_ranges.json so "
        "expressions like 'price_vs_200sma_dev<=0.01' match Barsmith's true threshold values (default: true).",
    )
    parser.add_argument(
        "--plot",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Plot the exported equity curves (default: true). Use --no-plot to disable.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["individual", "combined"],
        default="individual",
        help="Plot mode for equity curves (default: individual).",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Output PNG path for combined plot (default: <prepared_dir>/equity_curves_top10.png).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Output directory for individual plots (default: <prepared_dir>/equity_curves_top10_plots).",
    )
    parser.add_argument(
        "--plot-x",
        choices=["timestamp", "trade_index"],
        default="timestamp",
        help="X-axis for the equity curve plot (default: timestamp).",
    )
    parser.add_argument(
        "--plot-metric",
        choices=["dollar", "r"],
        default="dollar",
        help="Metric for plotting equity curves (default: dollar).",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help="Optional R-space max drawdown filter in R units (mirror Barsmith --max-drawdown).",
    )
    parser.add_argument(
        "--min-calmar",
        type=float,
        default=None,
        help="Optional minimum equity Calmar ratio (CAGR/max_drawdown_pct_equity) filter for the forward window.",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default="MES",
        help="Asset code for cost model (e.g., ES, MES). Defaults to MES to mirror Barsmith CLI.",
    )
    parser.add_argument(
        "--position-sizing",
        choices=["fractional", "contracts"],
        default="contracts",
        help="Equity simulation sizing mode: fractional (legacy) or contracts (futures-style).",
    )
    parser.add_argument(
        "--stop-distance-column",
        type=str,
        default=None,
        help="Per-trade stop distance column for contracts sizing (e.g., atr). When omitted, certain targets infer it.",
    )
    parser.add_argument(
        "--stop-distance-unit",
        choices=["points", "ticks"],
        default="points",
        help="Unit for --stop-distance-column (default: points).",
    )
    parser.add_argument(
        "--min-contracts",
        type=int,
        default=1,
        help="Minimum contracts for contracts sizing (default: 1).",
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="Optional maximum contracts cap for contracts sizing.",
    )
    parser.add_argument(
        "--margin-per-contract-dollar",
        type=float,
        default=None,
        help="Initial/overnight margin per contract in USD for contracts sizing (default: from --asset).",
    )
    parser.add_argument(
        "--commission-per-trade-dollar",
        type=float,
        default=None,
        help="Override commission per round-trip trade in USD (optional).",
    )
    parser.add_argument(
        "--slippage-per-trade-dollar",
        type=float,
        default=None,
        help="Override slippage per round-trip trade in USD (optional).",
    )
    parser.add_argument(
        "--cost-per-trade-dollar",
        type=float,
        default=None,
        help="Round-trip total cost per trade in USD (when set, overrides asset/commission/slippage).",
    )
    args = parser.parse_args()

    # Resolve an optional cost-per-trade model that mirrors the Barsmith CLI
    # semantics: when an asset is provided, use IBKR-style defaults for
    # commission and slippage, overridden by any explicit per-trade values.
    #
    # Asset profiles are kept intentionally small here and should mirror
    # barsmith_rs::asset::AssetProfile for ES/MES.
    asset_cost: float | None = None
    asset_point_value: float | None = None
    asset_tick_value: float | None = None
    asset_margin_per_contract: float | None = None
    if args.cost_per_trade_dollar is not None:
        asset_cost = args.cost_per_trade_dollar
    else:
        commission = args.commission_per_trade_dollar
        slippage = args.slippage_per_trade_dollar
        if args.asset:
            asset_code = args.asset.upper()
            if asset_code == "ES":
                # ES: ~ $4.50 round turn, no default slippage.
                base_commission = 2.0 * 2.25
                base_slippage = 0.0
                asset_point_value = 50.0
                asset_tick_value = 12.50
                asset_margin_per_contract = 25_000.0
            elif asset_code == "MES":
                # MES: ~ $1.14 round turn, no default slippage.
                base_commission = 2.0 * 0.57
                base_slippage = 0.0
                asset_point_value = 5.0
                asset_tick_value = 1.25
                asset_margin_per_contract = 2_500.0
            else:
                raise SystemExit(f"Unknown asset code '{args.asset}' for cost model; expected ES or MES.")
            commission = commission if commission is not None else base_commission
            slippage = slippage if slippage is not None else base_slippage
            asset_cost = commission + slippage
        elif commission is not None or slippage is not None:
            commission = commission if commission is not None else 0.0
            slippage = slippage if slippage is not None else 0.0
            asset_cost = commission + slippage

    # If an asset is supplied but cost model is disabled via explicit per-trade cost=None,
    # still populate point/tick values for contracts sizing.
    if args.asset and (asset_point_value is None or asset_tick_value is None):
        asset_code = args.asset.upper()
        if asset_code == "ES":
            asset_point_value = 50.0
            asset_tick_value = 12.50
            asset_margin_per_contract = 25_000.0
        elif asset_code == "MES":
            asset_point_value = 5.0
            asset_tick_value = 1.25
            asset_margin_per_contract = 2_500.0

    if not args.prepared.exists():
        print(f"ERROR: engineered CSV not found: {args.prepared}")
        return 1

    print("=" * 80)
    print("Barsmith kinetic combos evaluation using barsmith_prepared.csv")
    print("=" * 80)
    print(f"Prepared CSV: {args.prepared}")
    print(f"Cutoff date (in-sample <= cutoff): {args.cutoff}")
    print(f"Target: {args.target}")
    print(f"Stacking mode: {args.stacking_mode}")
    print(f"Position sizing: {args.position_sizing}")
    if args.max_drawdown is not None:
        print(f"Max DD filter (R-space, forward only): {args.max_drawdown:.1f}R")
    if args.min_calmar is not None:
        print(f"Min Calmar filter (equity, forward only): {args.min_calmar:.2f}")
    if args.frs:
        print(f"FRS (anchored calendar-year windows): enabled (scope={args.frs_scope}, N_min={args.frs_nmin})")
    else:
        print("FRS (anchored calendar-year windows): disabled")
    print()

    df = pd.read_csv(args.prepared)
    print(f"Loaded engineered frame with {len(df)} rows and {df.shape[1]} columns")

    rr_col = args.rr_column if args.rr_column else f"rr_{args.target}"
    eligible_col = f"{args.target}_eligible" if f"{args.target}_eligible" in df.columns else None
    exit_i_col: str | None = None
    if args.stacking_mode == "no-stacking":
        exit_i_col = resolve_exit_i_column(df, args.target)
        if exit_i_col is None:
            raise SystemExit(
                f"--stacking-mode no-stacking requires an exit index column (expected '{args.target}_exit_i'). "
                "Re-generate the prepared CSV using a newer Barsmith custom_rs, or rerun with --stacking-mode stacking."
            )
    missing: list[str] = []
    if args.target not in df.columns:
        missing.append(args.target)
    if rr_col not in df.columns:
        missing.append(rr_col)
    if missing:
        cols = sorted(df.columns.tolist())
        preview = ", ".join(cols[:30]) + (" ..." if len(cols) > 30 else "")
        raise SystemExit(
            "Engineered CSV missing required columns: " + ", ".join(f"'{c}'" for c in missing) + f"\nFirst columns: {preview}"
        )

    position_sizing = str(args.position_sizing or "fractional").strip().lower()
    stop_distance_column = args.stop_distance_column
    stop_distance_unit = str(args.stop_distance_unit or "points").strip().lower()
    min_contracts = int(args.min_contracts)
    max_contracts = int(args.max_contracts) if args.max_contracts is not None else None
    margin_per_contract_dollar = (
        float(args.margin_per_contract_dollar) if args.margin_per_contract_dollar is not None else asset_margin_per_contract
    )

    if position_sizing == "contracts":
        if stop_distance_column is None:
            stop_distance_column = infer_stop_distance_column(args.target)
        if stop_distance_column is None:
            raise SystemExit("--position-sizing contracts requires --stop-distance-column (or use a target that infers it).")
        if stop_distance_column not in df.columns:
            raise SystemExit(f"Stop distance column '{stop_distance_column}' not found in engineered CSV.")
        if stop_distance_unit not in {"points", "ticks"}:
            raise SystemExit(f"Unknown --stop-distance-unit: {args.stop_distance_unit}")
        if stop_distance_unit == "points" and (asset_point_value is None or asset_point_value <= 0.0):
            raise SystemExit("--position-sizing contracts with --stop-distance-unit points requires ES/MES point value.")
        if stop_distance_unit == "ticks" and (asset_tick_value is None or asset_tick_value <= 0.0):
            raise SystemExit("--position-sizing contracts with --stop-distance-unit ticks requires ES/MES tick value.")
        print(
            f"Contracts sizing stop model: col={stop_distance_column} unit={stop_distance_unit} "
            f"min={min_contracts} max={max_contracts} margin={margin_per_contract_dollar}"
        )
    else:
        stop_distance_column = None

    # Use the engineered timestamp column from Barsmith.
    dt = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    cutoff_date = pd.to_datetime(args.cutoff).date()
    pre_mask = dt.dt.date <= cutoff_date
    post_mask = dt.dt.date > cutoff_date
    all_mask = pd.Series(True, index=df.index)

    all_bh = compute_buy_and_hold_benchmark(df, all_mask, capital_dollar=float(args.capital))
    if all_bh.get("start") is not None:
        bh_total = float(all_bh.get("total_return_pct") or 0.0)
        bh_cagr = float(all_bh.get("cagr_pct") or 0.0)
        bh_dd = float(all_bh.get("max_drawdown_pct") or 0.0)
        bh_calmar = float(all_bh.get("calmar") or 0.0)
        bh_start = str(all_bh.get("start") or "")
        bh_end = str(all_bh.get("end") or "")
        bh_final = all_bh.get("final_capital")
        bh_final_str = None
        if isinstance(bh_final, (int, float)) and np.isfinite(float(bh_final)) and float(bh_final) > 0.0:
            bh_final_str = fmt_int(int(round(float(bh_final))))
        calmar_str = "Inf" if np.isinf(bh_calmar) else f"{bh_calmar:.2f}"

        print(
            f"Buy & Hold (close, all): {bh_start} -> {bh_end} | "
            f"Total {bh_total:.1f}% | CAGR {bh_cagr:.2f}% | "
            f"Max DD {bh_dd:.1f}% | Calmar {calmar_str}" + (f" | Final ${bh_final_str}" if bh_final_str else "")
        )
        print()

    if args.rank_by == "frs" and not args.frs:
        raise SystemExit("--rank-by frs requires --frs (FRS computation must be enabled).")

    threshold_name_map: dict[str, float] | None = None
    if args.barsmith_threshold_parity:
        needed_names = _extract_numeric_predicate_names(COMBOS)
        feature_ranges_path = REPO_ROOT / "barsmith" / "custom_rs" / "feature_ranges.json"
        threshold_name_map = build_barsmith_threshold_name_map(df, needed_names, feature_ranges_path=feature_ranges_path)
        if threshold_name_map:
            print(f"Barsmith threshold parity: mapped {len(threshold_name_map):,} predicate names to true thresholds")
        else:
            print("Barsmith threshold parity: no predicate thresholds mapped (feature_ranges.json missing or no matches)")

    def warn_if_no_finite_rr(window_label: str, mask: pd.Series) -> None:
        if not bool(mask.any()):
            return
        rr_raw = df.loc[mask, rr_col]
        rr_num = pd.to_numeric(rr_raw, errors="coerce")
        finite = rr_num.notna() & np.isfinite(rr_num.to_numpy())
        finite_count = int(finite.sum())
        nonnull_count = int(rr_num.notna().sum())
        if finite_count == 0:
            print(
                f"WARNING: RR column '{rr_col}' has 0 finite values in window {window_label} "
                f"(nonnull={nonnull_count:,}, rows={int(mask.sum()):,}). "
                "All combos will have 0 eligible trades in this window."
            )
            if args.target in {
                "highlow_or_atr",
                "highlow_or_atr_tightest_stop",
                "highlow_or_atr_tighest_stop",
                "highlow_1r",
                "2x_atr_tp_atr_stop",
                "3x_atr_tp_atr_stop",
                "atr_tp_atr_stop",
            }:
                print(
                    "WARNING: For target=highlow_or_atr*, Barsmith's Rust pipeline can intentionally leave RR as NaN "
                    "after a configured --date-end (include_date_end) to prevent leakage in older prepared CSVs. "
                    "Re-generate the prepared CSV using the updated Barsmith custom_rs (or without --date-end) to score forward windows."
                )

    warn_if_no_finite_rr(f"<= {args.cutoff}", pre_mask)
    warn_if_no_finite_rr(f">  {args.cutoff}", post_mask)

    frs_windows_rows: list[dict] = []

    def compute_frs_for_scope(scope: str, scope_mask: pd.Series) -> dict[str, dict]:
        windows = build_calendar_year_windows(df["timestamp"], scope_mask)
        if not windows:
            print(f"\nFRS({scope}): no calendar-year windows found; skipping.")
            return {}

        print(f"\nFRS({scope}): computing over {len(windows)} anchored calendar-year windows:")
        for w in windows:
            start_str = str(w["start"]) if w["start"] is not None else "N/A"
            end_str = str(w["end"]) if w["end"] is not None else "N/A"
            print(f"  - {w['label']}: rows={w['rows']:,} start={start_str} end={end_str}")

        per_expr: dict[str, dict[str, list]] = {}
        for w in windows:
            dataset_rows, window_results = compute_window_results(
                df,
                w["mask"],
                float(args.capital),
                float(args.risk_pct_per_trade),
                asset_cost,
                position_sizing=position_sizing,
                stop_distance_column=stop_distance_column,
                stop_distance_unit=stop_distance_unit,
                point_value=asset_point_value,
                tick_value=asset_tick_value,
                min_contracts=min_contracts,
                max_contracts=max_contracts,
                margin_per_contract_dollar=margin_per_contract_dollar,
                target_col=args.target,
                rr_col=rr_col,
                eligible_col=eligible_col,
                stacking_mode=args.stacking_mode,
                exit_i_col=exit_i_col,
                threshold_name_map=threshold_name_map,
            )
            w["rows"] = int(dataset_rows)
            w_start = w.get("start")
            w_end = w.get("end")

            for result in window_results:
                expr = result["expr"]
                metrics = result.get("metrics")
                trades = int(result.get("trades") or 0)
                total_return_r = float(metrics["total_return"]) if metrics is not None and trades > 0 else 0.0
                max_dd_r = float(metrics["max_drawdown"]) if metrics is not None and trades > 0 else 0.0
                expectancy_r = float(metrics["expectancy"]) if metrics is not None and trades > 0 else 0.0

                acc = per_expr.setdefault(expr, {"R": [], "DD": [], "N": [], "E": []})
                acc["R"].append(total_return_r)
                acc["DD"].append(max_dd_r)
                acc["N"].append(trades)
                acc["E"].append(expectancy_r)

                if args.frs_windows_out is not None:
                    equity = metrics if metrics is not None and trades > 0 else None
                    frs_windows_rows.append(
                        {
                            "scope": scope,
                            "expr": expr,
                            "window_label": w["label"],
                            "year": int(w["year"]),
                            "rows": int(w.get("rows") or 0),
                            "start": w_start,
                            "end": w_end,
                            "total_return_r": total_return_r,
                            "max_drawdown_r": max_dd_r,
                            "trades": trades,
                            "calmar_r": total_return_r / (max_dd_r + 1e-9),
                            "expectancy_r": expectancy_r,
                            "total_return_pct": float(equity.get("total_return_pct", 0.0)) if equity is not None else None,
                            "cagr_pct": float(equity.get("cagr_pct", 0.0)) if equity is not None else None,
                            "max_drawdown_pct_equity": (
                                float(equity.get("max_drawdown_pct_equity", 0.0)) if equity is not None else None
                            ),
                            "calmar_equity": float(equity.get("calmar_equity", 0.0)) if equity is not None else None,
                            "final_capital": float(equity.get("final_capital", 0.0)) if equity is not None else None,
                        }
                    )

        if args.frs_windows_out is not None:
            frs_windows_rows.extend(build_buy_and_hold_window_rows(df, windows, scope=scope, capital_dollar=float(args.capital)))

        out: dict[str, dict] = {}
        for expr, series in per_expr.items():
            components = compute_frs(
                series["R"],
                series["DD"],
                series["N"],
                n_min=int(args.frs_nmin),
                alpha=float(args.frs_alpha),
                beta=float(args.frs_beta),
                gamma=float(args.frs_gamma),
                delta=float(args.frs_delta),
            )
            out[expr] = asdict(components)
        return out

    def attach_frs(results: list[dict], frs_by_expr: dict[str, dict] | None) -> None:
        if not frs_by_expr:
            return
        for result in results:
            expr = result["expr"]
            components = frs_by_expr.get(expr) if frs_by_expr is not None else None
            if components is None:
                continue
            result["frs"] = float(components.get("frs", 0.0))
            result["frs_components"] = components

    frs_by_expr_pre: dict[str, dict] | None = None
    frs_by_expr_post: dict[str, dict] | None = None
    frs_by_expr_all: dict[str, dict] | None = None
    if args.frs:
        if args.frs_scope == "window":
            frs_by_expr_pre = compute_frs_for_scope("pre", pre_mask)
            frs_by_expr_post = compute_frs_for_scope("post", post_mask)
        elif args.frs_scope == "pre":
            frs_by_expr_pre = compute_frs_for_scope("pre", pre_mask)
        elif args.frs_scope == "post":
            frs_by_expr_post = compute_frs_for_scope("post", post_mask)
        elif args.frs_scope == "all":
            frs_by_expr_all = compute_frs_for_scope("all", pd.Series(True, index=df.index))
            frs_by_expr_pre = frs_by_expr_all
            frs_by_expr_post = frs_by_expr_all
        else:
            raise SystemExit(f"Unknown --frs-scope: {args.frs_scope}")

    # In-sample window: sort by Calmar only (no max DD/Calmar filter).
    pre_dataset_bars, pre_results = compute_window_results(
        df,
        pre_mask,
        float(args.capital),
        float(args.risk_pct_per_trade),
        asset_cost,
        position_sizing=position_sizing,
        stop_distance_column=stop_distance_column,
        stop_distance_unit=stop_distance_unit,
        point_value=asset_point_value,
        tick_value=asset_tick_value,
        min_contracts=min_contracts,
        max_contracts=max_contracts,
        margin_per_contract_dollar=margin_per_contract_dollar,
        target_col=args.target,
        rr_col=rr_col,
        eligible_col=eligible_col,
        stacking_mode=args.stacking_mode,
        exit_i_col=exit_i_col,
        max_drawdown_limit=None,
        min_calmar=None,
        threshold_name_map=threshold_name_map,
    )
    attach_frs(pre_results, frs_by_expr_pre)
    pre_sorted = sort_combo_results(pre_results, prev_ranks=None, primary_metric="calmar_equity")
    pre_bh = compute_buy_and_hold_benchmark(df, pre_mask, capital_dollar=float(args.capital))
    print_window_results(
        f"<= {args.cutoff}",
        pre_dataset_bars,
        pre_sorted,
        stacking_mode=args.stacking_mode,
        buy_and_hold=pre_bh,
    )

    # Build mapping from expression to in-sample rank so the forward window
    # can use previous rank as a secondary sort key and for display.
    prev_ranks: dict[str, int] = {result["expr"]: rank for rank, result in enumerate(pre_sorted, start=1)}

    # Forward window: sort by Calmar (forward) and previous in-sample rank, with optional max DD/Calmar filters.
    post_dataset_bars, post_results = compute_window_results(
        df,
        post_mask,
        float(args.capital),
        float(args.risk_pct_per_trade),
        asset_cost,
        position_sizing=position_sizing,
        stop_distance_column=stop_distance_column,
        stop_distance_unit=stop_distance_unit,
        point_value=asset_point_value,
        tick_value=asset_tick_value,
        min_contracts=min_contracts,
        max_contracts=max_contracts,
        margin_per_contract_dollar=margin_per_contract_dollar,
        target_col=args.target,
        rr_col=rr_col,
        eligible_col=eligible_col,
        stacking_mode=args.stacking_mode,
        exit_i_col=exit_i_col,
        max_drawdown_limit=args.max_drawdown,
        min_calmar=args.min_calmar,
        threshold_name_map=threshold_name_map,
    )
    attach_frs(post_results, frs_by_expr_post)
    post_sorted = sort_combo_results(post_results, prev_ranks=prev_ranks, primary_metric=args.rank_by)
    post_bh = compute_buy_and_hold_benchmark(df, post_mask, capital_dollar=float(args.capital))
    print_window_results(
        f">  {args.cutoff}",
        post_dataset_bars,
        post_sorted,
        prev_ranks=prev_ranks,
        stacking_mode=args.stacking_mode,
        buy_and_hold=post_bh,
    )

    if args.frs and (frs_by_expr_pre or frs_by_expr_post):
        if args.frs_out is not None:
            rows: list[dict] = []
            if frs_by_expr_pre is not None and args.frs_scope in ("window", "pre", "all"):
                rows.extend([{"scope": "pre", "expr": expr, **components} for expr, components in frs_by_expr_pre.items()])
            if frs_by_expr_post is not None and args.frs_scope in ("window", "post", "all"):
                rows.extend([{"scope": "post", "expr": expr, **components} for expr, components in frs_by_expr_post.items()])
            out = pd.DataFrame(rows).sort_values(["scope", "frs"], ascending=[True, False])
            args.frs_out.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.frs_out, index=False)
            print(f"\nFRS summary written: {args.frs_out}")

        if args.frs_windows_out is not None and frs_windows_rows:
            out = pd.DataFrame(frs_windows_rows)
            args.frs_windows_out.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.frs_windows_out, index=False)
            print(f"FRS windows written: {args.frs_windows_out}")

    if args.equity_curves_top_k > 0:
        prepared_dir = args.prepared.parent
        out_csv = args.equity_curves_out or (prepared_dir / f"equity_curves_top{args.equity_curves_top_k}.csv")
        out_png = args.plot_out or (prepared_dir / f"equity_curves_top{args.equity_curves_top_k}.png")
        out_dir = args.plot_dir or (prepared_dir / f"equity_curves_top{args.equity_curves_top_k}_plots")

        rank_source = post_sorted if args.equity_curves_rank_by == "post" else pre_sorted

        frames: list[pd.DataFrame] = []
        if args.equity_curves_window in ("pre", "both"):
            frames.append(
                compute_equity_curves_for_window(
                    df,
                    pre_mask,
                    rank_source,
                    args.equity_curves_top_k,
                    capital_dollar=args.capital,
                    risk_pct_per_trade=args.risk_pct_per_trade,
                    cost_per_trade_dollar=asset_cost,
                    position_sizing=position_sizing,
                    stop_distance_column=stop_distance_column,
                    stop_distance_unit=stop_distance_unit,
                    point_value=asset_point_value,
                    tick_value=asset_tick_value,
                    min_contracts=min_contracts,
                    max_contracts=max_contracts,
                    margin_per_contract_dollar=margin_per_contract_dollar,
                    target_col=args.target,
                    rr_col=rr_col,
                    eligible_col=eligible_col,
                    stacking_mode=args.stacking_mode,
                    exit_i_col=exit_i_col,
                    window_label="pre",
                    rank_by=args.equity_curves_rank_by,
                    threshold_name_map=threshold_name_map,
                )
            )
        if args.equity_curves_window in ("post", "both"):
            frames.append(
                compute_equity_curves_for_window(
                    df,
                    post_mask,
                    rank_source,
                    args.equity_curves_top_k,
                    capital_dollar=args.capital,
                    risk_pct_per_trade=args.risk_pct_per_trade,
                    cost_per_trade_dollar=asset_cost,
                    position_sizing=position_sizing,
                    stop_distance_column=stop_distance_column,
                    stop_distance_unit=stop_distance_unit,
                    point_value=asset_point_value,
                    tick_value=asset_tick_value,
                    min_contracts=min_contracts,
                    max_contracts=max_contracts,
                    margin_per_contract_dollar=margin_per_contract_dollar,
                    target_col=args.target,
                    rr_col=rr_col,
                    eligible_col=eligible_col,
                    stacking_mode=args.stacking_mode,
                    exit_i_col=exit_i_col,
                    window_label="post",
                    rank_by=args.equity_curves_rank_by,
                    threshold_name_map=threshold_name_map,
                )
            )

        non_empty = [f for f in frames if not f.empty]
        curves = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        if curves.empty:
            print("\nNo equity curve rows to export (no eligible trades in top strategies).")
        else:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            curves.to_csv(out_csv, index=False)
            print(f"\nEquity curves written: {out_csv}")

            if args.plot:
                if args.plot_mode == "combined":
                    plot_equity_curves(curves, out_path=out_png, x_axis=args.plot_x, metric=args.plot_metric)
                    print(f"Equity curve plot written: {out_png}")
                else:
                    plot_equity_curves_individual(curves, out_dir=out_dir, x_axis=args.plot_x, metric=args.plot_metric)
                    print(f"Equity curve plots written: {out_dir}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

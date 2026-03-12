#!/usr/bin/env python3
"""
Official Barsmith indicator parity checker (Rust vs standalone Python port).

This script:
  - Loads the engineered Barsmith dataset (`barsmith_prepared.csv`) produced by
    `custom_rs::prepare_dataset`.
  - Recomputes the same numeric indicators directly in Python using a port of
    the logic from `barsmith/custom_rs/src/engineer.rs` (no tribar_ai).
  - Aligns rows by timestamp and compares shared columns column-by-column.

Usage (example for ES 30m run v6):

    uv run python tools/check_barsmith_indicator_parity.py \\
        --csv barsmith/tests/data/es_30m_sample.csv \\
        --prepared barsmith/tmp/barsmith_run_es_30m_official_v7/barsmith_prepared.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Constants matching Rust `engineer.rs`
SMALL_DIVISOR: float = 1e-9
NON_ZERO_RANGE_EPS: float = float(np.finfo(float).eps)
NEXT_BAR_SL_MULTIPLIER: float = 1.5


def infer_timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Return a timezone-naive timestamp Series from common time columns."""
    lower = {c.lower(): c for c in df.columns}
    for candidate in ("timestamp", "datetime", "time", "date"):
        if candidate in lower:
            col = lower[candidate]
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if not isinstance(ts, pd.Series):
                ts = pd.Series(ts, index=df.index)
            try:
                ts = ts.dt.tz_convert(None)
            except AttributeError:
                ts = pd.Series(ts.values, index=df.index)
                ts = ts.dt.tz_localize(None)
            return ts
    raise KeyError("No timestamp-like column found (expected one of timestamp/datetime/time/date)")


class PriceSeries:
    def __init__(self, df: pd.DataFrame) -> None:
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)

    @property
    def length(self) -> int:
        return self.close.shape[0]


def diff_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def elementwise_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def upper_wick_vec(open_: np.ndarray, close: np.ndarray, high: np.ndarray) -> np.ndarray:
    return high - np.maximum(open_, close)


def lower_wick_vec(open_: np.ndarray, close: np.ndarray, low: np.ndarray) -> np.ndarray:
    return np.minimum(open_, close) - low


def range_vec(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    return high - low


def ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = np.abs(denom) >= np.finfo(float).eps
    out[mask] = num[mask] / denom[mask]
    return out


def ratio_with_eps(num: np.ndarray, denom: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = np.abs(denom) >= eps
    out[mask] = num[mask] / denom[mask]
    return out


def deviation(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    mask = np.abs(reference) >= np.finfo(float).eps
    out[mask] = (values[mask] - reference[mask]) / reference[mask]
    return out


def add_scalar(values: np.ndarray, scalar: float) -> np.ndarray:
    return values + scalar


def vector_abs(values: np.ndarray) -> np.ndarray:
    return np.abs(values)


def rma(values: np.ndarray, period: int) -> np.ndarray:
    length = values.shape[0]
    out = np.full(length, np.nan, dtype=float)
    if period == 0 or length == 0:
        return out
    alpha = 1.0 / float(period)
    prev = np.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            if np.isfinite(prev):
                out[i] = prev
            continue
        if np.isfinite(prev):
            nxt = alpha * v + (1.0 - alpha) * prev
        else:
            nxt = v
        out[i] = nxt
        prev = nxt
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if period == 0:
        return np.full_like(values, np.nan, dtype=float)
    length = values.shape[0]
    out = np.full(length, np.nan, dtype=float)
    if length == 0 or length < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = float(values[:period].mean())
    out[period - 1] = seed
    prev = seed
    for i in range(period, length):
        v = float(values[i])
        prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def ema_with_start(values: np.ndarray, period: int, start: int) -> np.ndarray:
    length = values.shape[0]
    out = np.full(length, np.nan, dtype=float)
    if start >= length:
        return out
    partial = ema(values[start:], period)
    out[start : start + partial.shape[0]] = partial
    return out


def sma(values: np.ndarray, period: int) -> np.ndarray:
    length = values.shape[0]
    out = np.full(length, np.nan, dtype=float)
    if period == 0 or period > length:
        return out
    weight = 1.0 / float(period)
    conv_len = length + period - 1
    conv = np.zeros(conv_len, dtype=float)
    conv_invalid = np.zeros(conv_len, dtype=bool)
    for k in range(period):
        for j in range(length):
            idx = k + j
            v = values[j]
            if not np.isfinite(v):
                conv_invalid[idx] = True
            elif not conv_invalid[idx]:
                conv[idx] += v * weight
    conv[conv_invalid] = np.nan
    out[period - 1 :] = conv[period - 1 : period - 1 + (length - (period - 1))]
    return out


def quantize_distance_to_tick(distance: float, tick_size: float) -> float:
    """Ceil-round a positive distance to the tick grid, mirroring Rust."""
    if not np.isfinite(distance) or tick_size <= 0.0:
        return distance
    if abs(distance) < np.finfo(float).eps:
        return 0.0
    ticks = distance / tick_size
    raw_rounded = math.ceil(ticks)
    ticks_final = max(raw_rounded, 1.0)
    return float(ticks_final * tick_size)


def quantize_price_to_tick(price: float, tick_size: float, mode: str) -> float:
    """Snap a price to the tick grid, mirroring Rust quantize_price_to_tick."""
    if not np.isfinite(price) or tick_size <= 0.0:
        return price
    if abs(price) < np.finfo(float).eps:
        return 0.0
    ticks = price / tick_size
    if mode == "floor":
        rounded = math.floor(ticks)
    elif mode == "ceil":
        rounded = math.ceil(ticks)
    else:
        raise ValueError(f"unknown tick rounding mode: {mode}")
    return float(rounded * tick_size)


def _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
    stop_mode: str = "wide",
    tp_mode: str = "atr",
    tp_multiple: float = 2.0,
    min_tp_rr: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python port of Rust compute_highlow_or_atr_targets_and_rr.

    - Entry at current bar close.
    - Long only on green candles; short only on red candles; doji = no trade.
    - Scan forward until TP, where TP is selected by tp_mode:
        - atr:  entry +/- (tp_multiple x ATR)
        - risk: entry +/- (tp_multiple x risk), where risk is the distance to the (tick-quantized) stop
      Stop is selected by stop_mode:
        - wide (original): long stop = min(low, entry-ATR), short stop = max(high, entry+ATR)
        - tightest: long stop = max(low, entry-ATR) subject to stop < entry;
                    short stop = min(high, entry+ATR) subject to stop > entry
        - highlow_only: long stop = low, short stop = high
        - atr_only: long stop = entry-ATR, short stop = entry+ATR
    - Conservative ordering: SL dominates if both touched in the same bar.
    - If resolve_end_idx is provided, cap the forward scan for entries at/before
      that index and force-exit remaining open trades at close[resolve_end_idx]
      (label=false). Entries after the cutoff still resolve normally to the end
      of the dataset, mirroring the Rust implementation.
    """
    if stop_mode not in {"wide", "tightest", "highlow_only", "atr_only"}:
        raise ValueError(f"Unknown stop_mode: {stop_mode}")
    if tp_mode not in {"atr", "risk"}:
        raise ValueError(f"Unknown tp_mode: {tp_mode}")
    if not np.isfinite(float(tp_multiple)) or float(tp_multiple) <= 0.0:
        raise ValueError(f"Invalid tp_multiple: {tp_multiple}")
    if min_tp_rr is not None and (not np.isfinite(float(min_tp_rr)) or float(min_tp_rr) <= 0.0):
        raise ValueError(f"Invalid min_tp_rr: {min_tp_rr}")

    requires_atr = stop_mode in {"wide", "tightest", "atr_only"} or tp_mode == "atr"
    length = min(open_.shape[0], high.shape[0], low.shape[0], close.shape[0], atr_14.shape[0])
    long = np.zeros(length, dtype=bool)
    short = np.zeros(length, dtype=bool)
    rr_long = np.full(length, np.nan, dtype=float)
    rr_short = np.full(length, np.nan, dtype=float)
    if length < 2:
        return long, short, rr_long, rr_short

    cutoff_horizon = resolve_end_idx if resolve_end_idx is not None else (length - 1)
    cutoff_horizon = int(min(max(cutoff_horizon, 0), length - 1))

    for idx in range(length - 1):
        cap_to_cutoff = resolve_end_idx is not None and idx <= cutoff_horizon
        local_horizon = cutoff_horizon if cap_to_cutoff else (length - 1)
        if idx >= local_horizon:
            continue

        o = float(open_[idx])
        c = float(close[idx])
        h0 = float(high[idx])
        l0 = float(low[idx])
        a = float(atr_14[idx])
        if not (np.isfinite(o) and np.isfinite(c) and np.isfinite(h0) and np.isfinite(l0)):
            continue
        if requires_atr and not np.isfinite(a):
            continue

        body = c - o
        if abs(body) <= np.finfo(float).eps:
            continue

        entry = c
        if body > 0.0:
            if stop_mode == "wide":
                stop_raw = min(l0, entry - a)
            elif stop_mode == "highlow_only":
                stop_raw = l0
            elif stop_mode == "atr_only":
                stop_raw = entry - a
            else:
                stop_raw = np.nan
                if l0 < entry:
                    stop_raw = l0
                atr_stop = entry - a
                if atr_stop < entry and (not np.isfinite(stop_raw) or atr_stop > stop_raw):
                    stop_raw = atr_stop

            stop = quantize_price_to_tick(stop_raw, tick_size, "floor") if tick_size is not None else stop_raw
            if not np.isfinite(stop) or stop >= entry:
                continue
            risk = entry - stop
            if risk <= SMALL_DIVISOR:
                continue

            tp_raw = entry + (float(tp_multiple) * a if tp_mode == "atr" else float(tp_multiple) * risk)
            tp = quantize_price_to_tick(tp_raw, tick_size, "ceil") if tick_size is not None else tp_raw
            if not np.isfinite(tp):
                continue
            if min_tp_rr is not None:
                rr_at_tp = (tp - entry) / risk
                if not np.isfinite(rr_at_tp) or rr_at_tp <= float(min_tp_rr):
                    continue

            rr = np.nan
            hit_tp = False
            for j in range(idx + 1, local_horizon + 1):
                oj = float(open_[j])
                hj = float(high[j])
                lj = float(low[j])
                if not (np.isfinite(hj) and np.isfinite(lj)):
                    continue

                # Gap-aware fills: if a bar opens beyond our stop/TP, assume the
                # fill happens at the open price (RR can be < -1 or > 2).
                if np.isfinite(oj):
                    if oj <= stop:
                        rr = (oj - entry) / risk
                        hit_tp = False
                        break
                    if oj >= tp:
                        rr = (oj - entry) / risk
                        hit_tp = True
                        break

                if lj <= stop:
                    rr = -1.0
                    hit_tp = False
                    break
                if hj >= tp:
                    rr = (tp - entry) / risk
                    hit_tp = True
                    break

            if not np.isfinite(rr) and cap_to_cutoff:
                exit_ = float(close[local_horizon])
                if np.isfinite(exit_):
                    rr = (exit_ - entry) / risk
                    hit_tp = False

            if np.isfinite(rr):
                rr_long[idx] = rr
                long[idx] = hit_tp
        else:
            if stop_mode == "wide":
                stop_raw = max(h0, entry + a)
            elif stop_mode == "highlow_only":
                stop_raw = h0
            elif stop_mode == "atr_only":
                stop_raw = entry + a
            else:
                stop_raw = np.nan
                if h0 > entry:
                    stop_raw = h0
                atr_stop = entry + a
                if atr_stop > entry and (not np.isfinite(stop_raw) or atr_stop < stop_raw):
                    stop_raw = atr_stop
            stop = quantize_price_to_tick(stop_raw, tick_size, "ceil") if tick_size is not None else stop_raw
            if not np.isfinite(stop) or stop <= entry:
                continue
            risk = stop - entry
            if risk <= SMALL_DIVISOR:
                continue

            tp_raw = entry - (float(tp_multiple) * a if tp_mode == "atr" else float(tp_multiple) * risk)
            tp = quantize_price_to_tick(tp_raw, tick_size, "floor") if tick_size is not None else tp_raw
            if not np.isfinite(tp):
                continue
            if min_tp_rr is not None:
                rr_at_tp = (entry - tp) / risk
                if not np.isfinite(rr_at_tp) or rr_at_tp <= float(min_tp_rr):
                    continue

            rr = np.nan
            hit_tp = False
            for j in range(idx + 1, local_horizon + 1):
                oj = float(open_[j])
                hj = float(high[j])
                lj = float(low[j])
                if not (np.isfinite(hj) and np.isfinite(lj)):
                    continue

                # Gap-aware fills: if a bar opens beyond our stop/TP, assume the
                # fill happens at the open price (RR can be < -1 or > 2).
                if np.isfinite(oj):
                    if oj >= stop:
                        rr = (entry - oj) / risk
                        hit_tp = False
                        break
                    if oj <= tp:
                        rr = (entry - oj) / risk
                        hit_tp = True
                        break

                if hj >= stop:
                    rr = -1.0
                    hit_tp = False
                    break
                if lj <= tp:
                    rr = (entry - tp) / risk
                    hit_tp = True
                    break

            if not np.isfinite(rr) and cap_to_cutoff:
                exit_ = float(close[local_horizon])
                if np.isfinite(exit_):
                    rr = (entry - exit_) / risk
                    hit_tp = False

            if np.isfinite(rr):
                rr_short[idx] = rr
                short[idx] = hit_tp

    return long, short, rr_long, rr_short


def compute_highlow_or_atr_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="wide",
        tp_mode="atr",
        tp_multiple=2.0,
    )


def compute_highlow_or_atr_tightest_stop_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="tightest",
        tp_mode="atr",
        tp_multiple=2.0,
    )


def compute_highlow_1r_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="highlow_only",
        tp_mode="risk",
        tp_multiple=1.0,
    )


def compute_atr_stop_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="atr_only",
        tp_mode="atr",
        tp_multiple=2.0,
    )


def compute_2x_atr_tp_atr_stop_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return compute_atr_stop_targets_and_rr(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
    )


def compute_atr_tp_atr_stop_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="atr_only",
        tp_mode="atr",
        tp_multiple=1.0,
    )


def compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="highlow_only",
        tp_mode="atr",
        tp_multiple=2.0,
        min_tp_rr=1.0,
    )


def compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_14: np.ndarray,
    tick_size: float | None = None,
    resolve_end_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_highlow_or_atr_targets_and_rr_with_stop_mode(
        open_=open_,
        high=high,
        low=low,
        close=close,
        atr_14=atr_14,
        tick_size=tick_size,
        resolve_end_idx=resolve_end_idx,
        stop_mode="highlow_only",
        tp_mode="atr",
        tp_multiple=1.0,
        min_tp_rr=1.0,
    )


def compute_next_bar_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    wicks_diff_sma14: np.ndarray,
    sl_multiplier: float = NEXT_BAR_SL_MULTIPLIER,
    tick_size: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python port of Rust compute_next_bar_targets_and_rr using CURRENT bar wicks
    for stop sizing and NEXT bar OHLC for trade outcome.
    """
    length = min(open_.shape[0], high.shape[0], low.shape[0], close.shape[0], wicks_diff_sma14.shape[0])
    long = np.zeros(length, dtype=bool)
    short = np.zeros(length, dtype=bool)
    rr_long = np.full(length, np.nan, dtype=float)
    rr_short = np.full(length, np.nan, dtype=float)
    if length < 2:
        return long, short, rr_long, rr_short

    for idx in range(length - 1):
        nxt = idx + 1
        entry = float(open_[nxt])
        high_next = float(high[nxt])
        low_next = float(low[nxt])
        close_next = float(close[nxt])
        wick = float(wicks_diff_sma14[idx])
        if not (
            np.isfinite(entry)
            and np.isfinite(high_next)
            and np.isfinite(low_next)
            and np.isfinite(close_next)
            and np.isfinite(wick)
        ):
            continue
        sl_distance_raw = abs(wick * sl_multiplier)
        sl_distance = quantize_distance_to_tick(sl_distance_raw, tick_size) if tick_size is not None else sl_distance_raw
        if sl_distance <= SMALL_DIVISOR:
            continue

        long_sl = entry - sl_distance
        short_sl = entry + sl_distance

        long_sl_hit = low_next <= long_sl
        short_sl_hit = high_next >= short_sl

        long[idx] = close_next > entry and not long_sl_hit
        short[idx] = close_next < entry and not short_sl_hit

        long_exit = long_sl if long_sl_hit else close_next
        rr_long[idx] = (long_exit - entry) / sl_distance

        short_exit = short_sl if short_sl_hit else close_next
        rr_short[idx] = (entry - short_exit) / sl_distance

    return long, short, rr_long, rr_short


def compute_wicks_kf_targets_and_rr(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    kf_wicks_smooth: np.ndarray,
    sl_multiplier: float = NEXT_BAR_SL_MULTIPLIER,
    tick_size: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python port of Rust compute_wicks_kf_targets_and_rr using CURRENT bar kf_wicks_smooth
    for stop sizing and NEXT bar OHLC for trade outcome.
    """
    length = min(open_.shape[0], high.shape[0], low.shape[0], close.shape[0], kf_wicks_smooth.shape[0])
    long = np.zeros(length, dtype=bool)
    short = np.zeros(length, dtype=bool)
    rr_long = np.full(length, np.nan, dtype=float)
    rr_short = np.full(length, np.nan, dtype=float)
    if length < 2:
        return long, short, rr_long, rr_short

    for idx in range(length - 1):
        nxt = idx + 1
        entry = float(open_[nxt])
        high_next = float(high[nxt])
        low_next = float(low[nxt])
        close_next = float(close[nxt])
        wick = float(kf_wicks_smooth[idx])
        if not (
            np.isfinite(entry)
            and np.isfinite(high_next)
            and np.isfinite(low_next)
            and np.isfinite(close_next)
            and np.isfinite(wick)
        ):
            continue

        sl_distance_raw = abs(wick * sl_multiplier)
        sl_distance = quantize_distance_to_tick(sl_distance_raw, tick_size) if tick_size is not None else sl_distance_raw
        if sl_distance <= SMALL_DIVISOR:
            continue

        long_sl = entry - sl_distance
        short_sl = entry + sl_distance

        long_sl_hit = low_next <= long_sl
        short_sl_hit = high_next >= short_sl

        long[idx] = close_next > entry and not long_sl_hit
        short[idx] = close_next < entry and not short_sl_hit

        long_exit = long_sl if long_sl_hit else close_next
        rr_long[idx] = (long_exit - entry) / sl_distance

        short_exit = short_sl if short_sl_hit else close_next
        rr_short[idx] = (entry - short_exit) / sl_distance

    return long, short, rr_long, rr_short


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    length = values.shape[0]
    if period == 0:
        return np.full(length, np.nan, dtype=float)
    if period == 1:
        return np.zeros(length, dtype=float)
    mean = sma(values, period)
    out = np.full(length, np.nan, dtype=float)
    for i in range(length):
        if i + 1 < period:
            continue
        start = i + 1 - period
        window = values[start : i + 1]
        if np.any(~np.isfinite(window)):
            continue
        mean_val = mean[i]
        if not np.isfinite(mean_val):
            continue
        var = float(((window - mean_val) ** 2).sum()) / float(period - 1)
        out[i] = np.sqrt(var)
    return out


def momentum(values: np.ndarray, period: int) -> np.ndarray:
    length = values.shape[0]
    out = np.zeros(length, dtype=float)
    for i in range(length):
        if i < period:
            out[i] = 0.0
        else:
            out[i] = float(values[i] - values[i - period])
    return out


def roc(values: np.ndarray, period: int) -> np.ndarray:
    length = values.shape[0]
    out = np.full(length, np.nan, dtype=float)
    for i in range(length):
        if i < period:
            continue
        prev = values[i - period]
        if abs(prev) < np.finfo(float).eps:
            continue
        out[i] = float(values[i] / prev - 1.0)
    return out


def derivative(values: np.ndarray, lag: int) -> np.ndarray:
    length = values.shape[0]
    out = np.zeros(length, dtype=float)
    for i in range(length):
        if i < lag:
            out[i] = 0.0
        else:
            out[i] = float(values[i] - values[i - lag])
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    length = close.shape[0]
    tr = np.zeros(length, dtype=float)
    for i in range(length):
        high_low = high[i] - low[i]
        if i == 0:
            high_close = high_low
            low_close = high_low
        else:
            high_close = abs(high[i] - close[i - 1])
            low_close = abs(low[i] - close[i - 1])
        tr[i] = max(high_low, high_close, low_close)
    return rma(tr, period)


def atr_close_to_close_core(close: np.ndarray, period: int) -> np.ndarray:
    length = close.shape[0]
    result = np.full(length, np.nan, dtype=float)
    if length == 0 or period == 0:
        return result
    ranges = np.full(length, np.nan, dtype=float)
    for i in range(1, length):
        cur = close[i]
        prev = close[i - 1]
        if np.isfinite(cur) and np.isfinite(prev):
            ranges[i] = abs(cur - prev)
    for i in range(length):
        start = 0 if i + 1 < period else i + 1 - period
        window = ranges[start : i + 1]
        valid = np.isfinite(window)
        if not valid.any():
            continue
        result[i] = float(window[valid].mean())
    if length > period:
        alpha = 2.0 / (period + 1.0)
        for i in range(period, length):
            value = ranges[i]
            prev = result[i - 1]
            if np.isfinite(value) and np.isfinite(prev):
                result[i] = alpha * value + (1.0 - alpha) * prev
    return result


def atr_close_to_close(close: np.ndarray, period: int) -> np.ndarray:
    # Mirror Rust: use the core implementation directly and let its natural
    # NaN warmup be handled later by dropping rows with NaNs.
    return atr_close_to_close_core(close, period)


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    length = close.shape[0]
    plus_dm = np.zeros(length, dtype=float)
    minus_dm = np.zeros(length, dtype=float)
    tr = np.zeros(length, dtype=float)
    for i in range(1, length):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0
        high_low = high[i] - low[i]
        high_close = abs(high[i] - close[i - 1])
        low_close = abs(low[i] - close[i - 1])
        tr[i] = max(high_low, high_close, low_close)
    atr_values = rma(tr, period)
    plus_smoothed = rma(plus_dm, period)
    minus_smoothed = rma(minus_dm, period)
    plus_di = np.zeros(length, dtype=float)
    minus_di = np.zeros(length, dtype=float)
    mask = np.abs(atr_values) >= np.finfo(float).eps
    plus_di[mask] = (plus_smoothed[mask] / atr_values[mask]) * 100.0
    minus_di[mask] = (minus_smoothed[mask] / atr_values[mask]) * 100.0
    dx = np.zeros(length, dtype=float)
    denom = plus_di + minus_di
    mask_dx = np.abs(denom) >= np.finfo(float).eps
    dx[mask_dx] = (np.abs(plus_di[mask_dx] - minus_di[mask_dx]) / denom[mask_dx]) * 100.0
    dx_clean = np.where(np.isfinite(dx), dx, 0.0)
    return rma(dx_clean, period)


def rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    length = close.shape[0]
    gains = np.full(length, np.nan, dtype=float)
    losses = np.full(length, np.nan, dtype=float)
    for i in range(1, length):
        change = close[i] - close[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)
    avg_gain = rma(gains, period)
    avg_loss = rma(losses, period)
    out = np.zeros(length, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
    out[avg_loss == 0.0] = 100.0
    return out


def rsi(close: np.ndarray, period: int, start: int) -> np.ndarray:
    length = close.shape[0]
    out = np.full(length, np.nan, dtype=float)
    if start >= length:
        return out
    partial = rsi_core(close[start:], period)
    out[start : start + partial.shape[0]] = partial
    return out


def macd_core(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fast = 12
    slow = 26
    signal_len = 9
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_line[: slow - 1] = np.nan
    signal = np.full_like(macd_line, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(macd_line))[0]
    if finite_idx.size > 0:
        first_valid = int(finite_idx[0])
        slice_vals = macd_line[first_valid:]
        ema_vals = ema(slice_vals, signal_len)
        signal[first_valid : first_valid + ema_vals.shape[0]] = ema_vals
    hist = np.where(np.isfinite(macd_line) & np.isfinite(signal), macd_line - signal, np.nan)
    return macd_line, signal, hist


def macd(close: np.ndarray, start: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = close.shape[0]
    macd_full = np.full(length, np.nan, dtype=float)
    signal_full = np.full(length, np.nan, dtype=float)
    hist_full = np.full(length, np.nan, dtype=float)
    if start >= length:
        return macd_full, signal_full, hist_full
    macd_slice, signal_slice, hist_slice = macd_core(close[start:])
    macd_full[start : start + macd_slice.shape[0]] = macd_slice
    signal_full[start : start + signal_slice.shape[0]] = signal_slice
    hist_full[start : start + hist_slice.shape[0]] = hist_slice
    return macd_full, signal_full, hist_full


def kalman_filter(values: np.ndarray, process_var: float, obs_var: float) -> Tuple[np.ndarray, np.ndarray]:
    length = values.shape[0]
    filtered = np.full(length, np.nan, dtype=float)
    innovations = np.zeros(length, dtype=float)
    if length == 0:
        return filtered, innovations
    finite_vals = values[np.isfinite(values)]
    x = float(finite_vals[0]) if finite_vals.size > 0 else 0.0
    p = 1.0
    for i, value in enumerate(values):
        x_pred = x
        p_pred = p + process_var
        if np.isfinite(value):
            k = p_pred / (p_pred + obs_var)
            innovation = value - x_pred
            x = x_pred + k * innovation
            p = (1.0 - k) * p_pred
            filtered[i] = x
            innovations[i] = innovation
        else:
            filtered[i] = x_pred
            innovations[i] = 0.0
            x = x_pred
            p = p_pred
    return filtered, innovations


def kalman_filter_with_start(values: np.ndarray, start: int, process_var: float, obs_var: float) -> Tuple[np.ndarray, np.ndarray]:
    length = values.shape[0]
    filtered = np.full(length, np.nan, dtype=float)
    innovations = np.zeros(length, dtype=float)
    if start >= length:
        return filtered, innovations
    valid_indices = [idx for idx in range(start, length) if np.isfinite(values[idx])]
    if not valid_indices:
        return filtered, innovations
    compacted = np.array([values[idx] for idx in valid_indices], dtype=float)
    slice_filtered, slice_innov = kalman_filter(compacted, process_var, obs_var)
    for offset, target_idx in enumerate(valid_indices):
        filtered[target_idx] = slice_filtered[offset]
        innovations[target_idx] = slice_innov[offset]
    return filtered, innovations


def stochastic_core(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int, signal: int
) -> Tuple[np.ndarray, np.ndarray]:
    length = close.shape[0]
    highest = np.full(length, np.nan, dtype=float)
    lowest = np.full(length, np.nan, dtype=float)
    ranges = np.full(length, np.nan, dtype=float)
    for i in range(length):
        if i + 1 < period:
            continue
        slice_high = high[i + 1 - period : i + 1]
        slice_low = low[i + 1 - period : i + 1]
        high_val = float(np.max(slice_high))
        low_val = float(np.min(slice_low))
        highest[i] = high_val
        lowest[i] = low_val
        ranges[i] = high_val - low_val
    if np.any((ranges == 0.0) & np.isfinite(ranges)):
        ranges = np.where(np.isfinite(ranges) & (ranges == 0.0), ranges + NON_ZERO_RANGE_EPS, ranges)
    raw_k = np.full(length, np.nan, dtype=float)
    for i in range(length):
        r = ranges[i]
        lv = lowest[i]
        if not np.isfinite(r) or not np.isfinite(lv):
            continue
        denom = r
        if abs(denom) < NON_ZERO_RANGE_EPS:
            denom += NON_ZERO_RANGE_EPS
        if abs(denom) < NON_ZERO_RANGE_EPS:
            continue
        raw_k[i] = float(np.clip((close[i] - lv) / denom * 100.0, 0.0, 100.0))
    smooth_k = np.full(length, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(raw_k))[0]
    if finite_idx.size > 0:
        first_valid = int(finite_idx[0])
        slice_vals = raw_k[first_valid:]
        slice_sma = sma(slice_vals, signal)
        smooth_k[first_valid : first_valid + slice_sma.shape[0]] = slice_sma
    d = np.full(length, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(smooth_k))[0]
    if finite_idx.size > 0:
        first_valid = int(finite_idx[0])
        slice_vals = smooth_k[first_valid:]
        slice_sma = sma(slice_vals, signal)
        d[first_valid : first_valid + slice_sma.shape[0]] = slice_sma
    return smooth_k, d


def stochastic(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int, signal: int, start: int
) -> Tuple[np.ndarray, np.ndarray]:
    length = close.shape[0]
    k_full = np.full(length, np.nan, dtype=float)
    d_full = np.full(length, np.nan, dtype=float)
    if start >= length:
        return k_full, d_full
    k_slice, d_slice = stochastic_core(close[start:], high[start:], low[start:], period, signal)
    k_full[start : start + k_slice.shape[0]] = k_slice
    d_full[start : start + d_slice.shape[0]] = d_slice
    return k_full, d_full


def bollinger_core(close: np.ndarray, period: int, std_mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(close, period)
    std = rolling_std(close, period)
    upper = mid + std * std_mult
    lower = mid - std * std_mult
    return mid, upper, lower, std


def bollinger(
    close: np.ndarray, period: int, std_mult: float, start: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = close.shape[0]
    mid_full = np.full(length, np.nan, dtype=float)
    upper_full = np.full(length, np.nan, dtype=float)
    lower_full = np.full(length, np.nan, dtype=float)
    std_full = np.full(length, np.nan, dtype=float)
    if start >= length:
        return mid_full, upper_full, lower_full, std_full
    mid, upper, lower, std = bollinger_core(close[start:], period, std_mult)
    mid_full[start : start + mid.shape[0]] = mid
    upper_full[start : start + upper.shape[0]] = upper
    lower_full[start : start + lower.shape[0]] = lower
    std_full[start : start + std.shape[0]] = std
    return mid_full, upper_full, lower_full, std_full


def rolling_coeff_var(values: np.ndarray, period: int) -> np.ndarray:
    mean = sma(values, period)
    std = rolling_std(values, period)
    out = np.zeros_like(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(mean) < np.finfo(float).eps, 0.0, std / mean)
    return out


def extension(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    length = high.shape[0]
    out = np.zeros(length, dtype=float)
    for i in range(length):
        if i + 1 < period:
            out[i] = 0.0
        else:
            start = i + 1 - period
            out[i] = float(high[i] - np.min(low[start : i + 1]))
    return out


def percentile_rank(values: np.ndarray) -> np.ndarray:
    length = values.shape[0]
    idx_val = [(i, float(v)) for i, v in enumerate(values) if np.isfinite(v)]
    count = len(idx_val)
    if count == 0:
        return np.full(length, np.nan, dtype=float)
    idx_val.sort(key=lambda x: x[1])
    ranks = np.full(length, np.nan, dtype=float)
    denom = float(count)
    i = 0
    while i < count:
        j = i + 1
        while j < count and np.isclose(idx_val[j][1], idx_val[i][1]):
            j += 1
        avg_rank = ((i + j - 1) / 2.0 + 1.0) / denom
        for k in range(i, j):
            ranks[idx_val[k][0]] = avg_rank
        i = j
    return ranks


def momentum_score_vec(rsi14: np.ndarray, roc5: np.ndarray, roc10: np.ndarray) -> np.ndarray:
    rsi_rank = percentile_rank(rsi14)
    roc5_rank = percentile_rank(roc5)
    roc10_rank = percentile_rank(roc10)
    out = np.full_like(rsi14, np.nan, dtype=float)
    mask = np.isfinite(rsi_rank) & np.isfinite(roc5_rank) & np.isfinite(roc10_rank)
    out[mask] = (rsi_rank[mask] + roc5_rank[mask] + roc10_rank[mask]) / 3.0
    return out


class DerivedMetrics:
    def __init__(self, prices: PriceSeries) -> None:
        close = prices.close
        open_ = prices.open
        high = prices.high
        low = prices.low

        body = diff_vec(close, open_)
        abs_body = np.abs(body)
        up_w = upper_wick_vec(open_, close, high)
        lo_w = lower_wick_vec(open_, close, low)
        max_w = elementwise_max(up_w, lo_w)

        ema9 = ema(close, 9)
        ema20 = ema(close, 20)
        ema50 = ema(close, 50)
        ema200 = ema(close, 200)
        sma200 = sma(close, 200)

        kf_smooth, kf_innovation = kalman_filter(close, 0.01, 0.1)
        kf_trend, _ = kalman_filter(close, 0.001, 0.5)
        kf_ma = kf_smooth.copy()
        kf_close_mom = derivative(kf_smooth, 1)
        kf_slope_5 = derivative(kf_smooth, 5) / 5.0

        momentum_14 = momentum(close, 14)
        roc5 = roc(close, 5)
        roc10 = roc(close, 10)

        adx_vals = adx(high, low, close, 14)

        atr_vals = atr(high, low, close, 14)
        atr_c2c = atr_close_to_close(close, 14)
        atr_pct = ratio(atr_vals, close)
        atr_c2c_pct = ratio(atr_c2c, close)
        bar_range = range_vec(high, low)
        bar_range_pct = ratio(bar_range, close)
        vol_20_cv = rolling_coeff_var(close, 20)
        body_size_pct = ratio(abs_body, close)
        upper_shadow_ratio = ratio(up_w, bar_range)
        lower_shadow_ratio = ratio(lo_w, bar_range)

        wicks_diff = np.array(
            [(o - l) if c > o else (h - o) for o, c, l, h in zip(open_, close, low, high)],
            dtype=float,
        )
        wicks_diff_sma14 = sma(wicks_diff, 14)
        kf_wicks_smooth, _ = kalman_filter(wicks_diff, 0.01, 0.1)
        total_wick = up_w + lo_w
        body_to_total_wick = ratio_with_eps(abs_body, total_wick, SMALL_DIVISOR)

        price_vs_200sma_dev = deviation(close, sma200)
        price_vs_9ema_dev = deviation(close, ema9)
        nine_to_two_hundred = deviation(ema9, ema200)

        rsi14 = rsi(close, 14, 0)
        rsi7 = rsi(close, 7, 0)
        rsi21 = rsi(close, 21, 0)
        mom_score = momentum_score_vec(rsi14, roc5, roc10)
        atr_mean50 = sma(atr_vals, 50)
        atr_c2c_mean50 = sma(atr_c2c, 50)

        macd_line, macd_signal, macd_hist = macd(close, 0)
        stoch_k, stoch_d = stochastic(close, high, low, 14, 3, 0)

        bb_mid, bb_upper, bb_lower, bb_std = bollinger(close, 20, 2.0, 0)
        ext = extension(high, low, 20)
        ext_sma14 = sma(ext, 14)

        kf_close_minus = diff_vec(close, kf_smooth)
        kf_price_deviation = ratio(kf_close_minus, close)
        kf_vs_9ema = ratio(diff_vec(kf_smooth, ema9), ema9)
        kf_vs_200sma = ratio(diff_vec(kf_smooth, sma200), sma200)
        kf_innovation_abs = vector_abs(kf_innovation)

        kf_adx, kf_adx_innovation = kalman_filter(adx_vals, 0.005, 0.2)
        kf_adx_slope = derivative(kf_adx, 1)
        kf_adx_dev = ratio_with_eps(
            diff_vec(adx_vals, kf_adx),
            add_scalar(kf_adx, SMALL_DIVISOR),
            SMALL_DIVISOR,
        )
        kf_adx_innov_abs = vector_abs(kf_adx_innovation)
        kf_adx_mom5 = derivative(kf_adx, 5)

        kf_atr, kf_atr_innovation = kalman_filter(atr_vals, 0.01, 0.15)
        kf_atr_c2c, kf_atr_c2c_innovation = kalman_filter_with_start(atr_c2c, 0, 0.01, 0.15)
        kf_atr_pct = ratio(kf_atr, close)
        kf_atr_c2c_pct = ratio(kf_atr_c2c, close)
        kf_atr_vs_c2c = ratio_with_eps(kf_atr, add_scalar(kf_atr_c2c, SMALL_DIVISOR), SMALL_DIVISOR)
        kf_atr_dev = ratio_with_eps(
            diff_vec(atr_vals, kf_atr),
            add_scalar(kf_atr, SMALL_DIVISOR),
            SMALL_DIVISOR,
        )
        kf_atr_mom5 = derivative(kf_atr, 5)
        kf_atr_c2c_mom5 = derivative(kf_atr_c2c, 5)
        kf_trend_mom = derivative(kf_trend, 5)
        denom_vol = kf_atr_pct * 100.0 + SMALL_DIVISOR
        kf_trend_vol_ratio = ratio_with_eps(kf_adx, denom_vol, SMALL_DIVISOR)
        body_atr_ratio = ratio_with_eps(abs_body, atr_vals, SMALL_DIVISOR)

        self.abs_body = abs_body
        self.upper_wick = up_w
        self.lower_wick = lo_w
        self.max_wick = max_w
        self.body_to_total_wick = body_to_total_wick
        self.body_atr_ratio = body_atr_ratio
        self.ema9 = ema9
        self.ema20 = ema20
        self.ema50 = ema50
        self.sma200 = sma200
        self.kf_ma = kf_ma
        self.momentum_14 = momentum_14
        self.momentum_score = mom_score
        self.roc5 = roc5
        self.roc10 = roc10
        self.adx = adx_vals
        self.atr = atr_vals
        self.atr_c2c = atr_c2c
        self.atr_pct = atr_pct
        self.atr_c2c_pct = atr_c2c_pct
        self.bar_range_pct = bar_range_pct
        self.volatility_20_cv = vol_20_cv
        self.body_size_pct = body_size_pct
        self.upper_shadow_ratio = upper_shadow_ratio
        self.lower_shadow_ratio = lower_shadow_ratio
        self.wicks_diff = wicks_diff
        self.wicks_diff_sma14 = wicks_diff_sma14
        self.price_vs_200sma_dev = price_vs_200sma_dev
        self.price_vs_9ema_dev = price_vs_9ema_dev
        self.nine_to_two_hundred = nine_to_two_hundred
        self.rsi14 = rsi14
        self.rsi7 = rsi7
        self.rsi21 = rsi21
        self.atr_mean50 = atr_mean50
        self.atr_c2c_mean50 = atr_c2c_mean50
        self.macd = macd_line
        self.macd_signal = macd_signal
        self.macd_hist = macd_hist
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.bb_mid = bb_mid
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_std = bb_std
        self.ext = ext
        self.ext_sma14 = ext_sma14
        self.kf_smooth = kf_smooth
        self.kf_innovation = kf_innovation
        self.kf_close_momentum = kf_close_mom
        self.kf_slope_5 = kf_slope_5
        self.kf_trend = kf_trend
        self.kf_adx = kf_adx
        self.kf_atr = kf_atr
        self.kf_atr_c2c = kf_atr_c2c
        self.kf_adx_slope = kf_adx_slope
        self.kf_trend_momentum = kf_trend_mom
        self.kf_trend_volatility_ratio = kf_trend_vol_ratio
        self.kf_price_deviation = kf_price_deviation
        self.kf_vs_9ema = kf_vs_9ema
        self.kf_vs_200sma = kf_vs_200sma
        self.kf_innovation_abs = kf_innovation_abs
        self.kf_adx_deviation = kf_adx_dev
        self.kf_adx_innovation_abs = kf_adx_innov_abs
        self.kf_adx_momentum_5 = kf_adx_mom5
        self.kf_atr_pct = kf_atr_pct
        self.kf_atr_c2c_pct = kf_atr_c2c_pct
        self.kf_atr_vs_c2c = kf_atr_vs_c2c
        self.kf_atr_deviation = kf_atr_dev
        self.kf_atr_momentum_5 = kf_atr_mom5
        self.kf_atr_c2c_momentum_5 = kf_atr_c2c_mom5
        self.kf_atr_innovation = kf_atr_innovation
        self.kf_atr_c2c_innovation = kf_atr_c2c_innovation
        self.kf_wicks_smooth = kf_wicks_smooth


def build_python_indicator_frame(
    df_raw: pd.DataFrame,
    tick_size: float | None = None,
    target: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    """Compute indicators in Python using the same formulas as Rust."""
    prices = PriceSeries(df_raw)
    derived = DerivedMetrics(prices)

    # Candle color flags used by several boolean pattern features.
    is_green = prices.close > prices.open
    is_red = prices.close < prices.open

    # Tribar variants mirrored from Rust candle_features().
    length = len(prices.close)
    custom_high = np.maximum(prices.open, prices.close)
    custom_low = np.minimum(prices.open, prices.close)
    tribar_green = np.zeros(length, dtype=bool)
    tribar_red = np.zeros(length, dtype=bool)
    tribar_hl_green = np.zeros(length, dtype=bool)
    tribar_hl_red = np.zeros(length, dtype=bool)
    for i in range(2, length):
        # "Tribar": close breaks above/below the prior two body highs/lows.
        if (
            bool(is_green[i])
            and float(prices.close[i]) > float(custom_high[i - 1])
            and float(prices.close[i]) > float(custom_high[i - 2])
        ):
            tribar_green[i] = True
        if (
            bool(is_red[i])
            and float(prices.close[i]) < float(custom_low[i - 1])
            and float(prices.close[i]) < float(custom_low[i - 2])
        ):
            tribar_red[i] = True

        # "Tribar HL": close breaks above/below the prior two full-bar highs/lows.
        if (
            bool(is_green[i])
            and float(prices.close[i]) > float(prices.high[i - 1])
            and float(prices.close[i]) > float(prices.high[i - 2])
        ):
            tribar_hl_green[i] = True
        if (
            bool(is_red[i])
            and float(prices.close[i]) < float(prices.low[i - 1])
            and float(prices.close[i]) < float(prices.low[i - 2])
        ):
            tribar_hl_red[i] = True

    normalized_target = target
    if normalized_target == "highlow_or_atr_tighest_stop":
        normalized_target = "highlow_or_atr_tightest_stop"
    if normalized_target == "atr_stop":
        normalized_target = "2x_atr_tp_atr_stop"

    if normalized_target in {
        "highlow_or_atr",
        "highlow_or_atr_tightest_stop",
        "highlow_1r",
        "2x_atr_tp_atr_stop",
        "atr_tp_atr_stop",
        "highlow_sl_2x_atr_tp_rr_gt_1",
        "highlow_sl_1x_atr_tp_rr_gt_1",
    }:
        resolve_end_idx: int | None = None
        if date_end:
            ts_raw = infer_timestamp_column(df_raw)
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            cutoff = pd.to_datetime(date_end).date()
            mask = ts.notna() & (ts.dt.date <= cutoff)
            if bool(mask.any()):
                resolve_end_idx = int(np.flatnonzero(mask.to_numpy()).max())

        if normalized_target == "highlow_or_atr":
            long_t, short_t, rr_long, rr_short = compute_highlow_or_atr_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
        elif normalized_target == "highlow_or_atr_tightest_stop":
            long_t, short_t, rr_long, rr_short = compute_highlow_or_atr_tightest_stop_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
        elif normalized_target == "highlow_1r":
            long_t, short_t, rr_long, rr_short = compute_highlow_1r_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
        elif normalized_target == "atr_tp_atr_stop":
            long_t, short_t, rr_long, rr_short = compute_atr_tp_atr_stop_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
        elif normalized_target == "highlow_sl_2x_atr_tp_rr_gt_1":
            long_t, short_t, rr_long, rr_short = compute_highlow_sl_2x_atr_tp_rr_gt_1_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
        elif normalized_target == "highlow_sl_1x_atr_tp_rr_gt_1":
            long_t, short_t, rr_long, rr_short = compute_highlow_sl_1x_atr_tp_rr_gt_1_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
        else:
            long_t, short_t, rr_long, rr_short = compute_2x_atr_tp_atr_stop_targets_and_rr(
                prices.open,
                prices.high,
                prices.low,
                prices.close,
                derived.atr,
                tick_size=tick_size,
                resolve_end_idx=resolve_end_idx,
            )
    elif target == "wicks_kf":
        long_t, short_t, rr_long, rr_short = compute_wicks_kf_targets_and_rr(
            prices.open,
            prices.high,
            prices.low,
            prices.close,
            derived.kf_wicks_smooth,
            sl_multiplier=NEXT_BAR_SL_MULTIPLIER,
            tick_size=tick_size,
        )
    else:
        long_t, short_t, rr_long, rr_short = compute_next_bar_targets_and_rr(
            prices.open,
            prices.high,
            prices.low,
            prices.close,
            derived.wicks_diff_sma14,
            sl_multiplier=NEXT_BAR_SL_MULTIPLIER,
            tick_size=tick_size,
        )

    data: Dict[str, np.ndarray] = {
        "open": prices.open,
        "high": prices.high,
        "low": prices.low,
        "close": prices.close,
        "is_green": is_green,
        "is_red": is_red,
        "is_tribar": tribar_green | tribar_red,
        "is_tribar_green": tribar_green,
        "is_tribar_red": tribar_red,
        "is_tribar_hl": tribar_hl_green | tribar_hl_red,
        "is_tribar_hl_green": tribar_hl_green,
        "is_tribar_hl_red": tribar_hl_red,
        "9ema": derived.ema9,
        "20ema": derived.ema20,
        "50ema": derived.ema50,
        "200sma": derived.sma200,
        "momentum_14": derived.momentum_14,
        "momentum_score": derived.momentum_score,
        "roc_5": derived.roc5,
        "roc_10": derived.roc10,
        "adx": derived.adx,
        "atr": derived.atr,
        "atr_c2c": derived.atr_c2c,
        "atr_pct": derived.atr_pct,
        "atr_c2c_pct": derived.atr_c2c_pct,
        "bar_range_pct": derived.bar_range_pct,
        "volatility_20_cv": derived.volatility_20_cv,
        "body_size_pct": derived.body_size_pct,
        "upper_shadow_ratio": derived.upper_shadow_ratio,
        "lower_shadow_ratio": derived.lower_shadow_ratio,
        "wicks_diff": derived.wicks_diff,
        "wicks_diff_sma14": derived.wicks_diff_sma14,
        "kf_wicks_smooth": derived.kf_wicks_smooth,
        "price_vs_200sma_dev": derived.price_vs_200sma_dev,
        "price_vs_9ema_dev": derived.price_vs_9ema_dev,
        "9ema_to_200sma": derived.nine_to_two_hundred,
        "rsi_14": derived.rsi14,
        "rsi_7": derived.rsi7,
        "rsi_21": derived.rsi21,
        "atr_pct_mean50": derived.atr_mean50,
        "atr_c2c_mean50": derived.atr_c2c_mean50,
        "macd": derived.macd,
        "macd_signal": derived.macd_signal,
        "macd_hist": derived.macd_hist,
        "stoch_k": derived.stoch_k,
        "stoch_d": derived.stoch_d,
        "bb_std": derived.bb_std,
        "ext": derived.ext,
        "ext_sma14": derived.ext_sma14,
        "kf_smooth": derived.kf_smooth,
        "kf_price_deviation": derived.kf_price_deviation,
        "kf_vs_9ema": derived.kf_vs_9ema,
        "kf_vs_200sma": derived.kf_vs_200sma,
        "kf_innovation_abs": derived.kf_innovation_abs,
        "kf_adx": derived.kf_adx,
        "kf_atr": derived.kf_atr,
        "kf_atr_c2c": derived.kf_atr_c2c,
        "kf_adx_slope": derived.kf_adx_slope,
        "kf_trend_momentum": derived.kf_trend_momentum,
        "kf_trend_volatility_ratio": derived.kf_trend_volatility_ratio,
        "kf_adx_deviation": derived.kf_adx_deviation,
        "kf_adx_innovation_abs": derived.kf_adx_innovation_abs,
        "kf_adx_momentum_5": derived.kf_adx_momentum_5,
        "kf_atr_pct": derived.kf_atr_pct,
        "kf_atr_c2c_pct": derived.kf_atr_c2c_pct,
        "kf_atr_vs_c2c": derived.kf_atr_vs_c2c,
        "kf_atr_deviation": derived.kf_atr_deviation,
        "kf_atr_momentum_5": derived.kf_atr_momentum_5,
        "kf_atr_c2c_momentum_5": derived.kf_atr_c2c_momentum_5,
        "kf_atr_innovation": derived.kf_atr_innovation,
        "kf_atr_c2c_innovation": derived.kf_atr_c2c_innovation,
    }

    if normalized_target in {
        "highlow_or_atr",
        "highlow_or_atr_tightest_stop",
        "highlow_1r",
        "2x_atr_tp_atr_stop",
        "atr_tp_atr_stop",
    }:
        prefix = normalized_target
        data[f"{prefix}_long"] = long_t
        data[f"{prefix}_short"] = short_t
        data["rr_long"] = rr_long
        data["rr_short"] = rr_short
        eligible_long = np.isfinite(prices.open) & np.isfinite(prices.close) & (prices.close > prices.open)
        eligible_short = np.isfinite(prices.open) & np.isfinite(prices.close) & (prices.close < prices.open)
        data[f"{prefix}_eligible_long"] = eligible_long
        data[f"{prefix}_eligible_short"] = eligible_short
        # Directional aliases are long-biased by default; parity checks skip them.
        data[prefix] = long_t
        data[f"rr_{prefix}"] = rr_long
        data[f"{prefix}_eligible"] = eligible_long
    elif target == "wicks_kf":
        data["wicks_kf_long"] = long_t
        data["wicks_kf_short"] = short_t
        data["rr_long"] = rr_long
        data["rr_short"] = rr_short
        # Directional aliases are long-biased by default; parity checks skip them.
        data["wicks_kf"] = long_t
        data["rr_wicks_kf"] = rr_long
    else:
        # Attach next-bar targets matching Rust semantics.
        data["next_bar_color_and_wicks_long"] = long_t
        data["next_bar_color_and_wicks_short"] = short_t
        data["rr_long"] = rr_long
        data["rr_short"] = rr_short
        # Directional aliases are long-biased by default; parity checks skip them.
        data["next_bar_color_and_wicks"] = long_t
        data["rr_next_bar_color_and_wicks"] = rr_long

    adx_sma = sma(derived.adx, 14)
    trend_strength = np.full_like(derived.adx, np.nan, dtype=float)
    for i in range(trend_strength.shape[0]):
        price = prices.close[i]
        sma200_val = derived.sma200[i]
        adx_val = derived.adx[i]
        if not (np.isfinite(price) and np.isfinite(sma200_val) and np.isfinite(adx_val)):
            continue
        if abs(sma200_val) < np.finfo(float).eps:
            continue
        adx_term = (adx_val / 100.0) * 0.4
        deviation_term = abs((price / sma200_val) - 1.0) * 10.0 * 0.6
        trend_strength[i] = adx_term + deviation_term

    data["adx_sma"] = adx_sma
    data["trend_strength"] = trend_strength

    df = pd.DataFrame(data)
    ts = infer_timestamp_column(df_raw)
    df["timestamp"] = ts
    return df


def build_python_indicator_frame_minimal(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a minimal Barsmith indicator frame for live models.

    This is a faster/lower-memory alternative to `build_python_indicator_frame` when
    you only need a small subset of columns (no targets, no wide feature export).

    Columns returned:
      - OHLC + timestamp
      - is_tribar_green
      - is_kf_breakout_potential
      - kf_innovation_abs
      - macd_hist
      - trend_strength
      - atr
      - kf_adx
      - kf_atr
    """
    prices = PriceSeries(df_raw)
    derived = DerivedMetrics(prices)

    length = len(prices.close)
    custom_high = np.maximum(prices.open, prices.close)

    tribar_green = np.zeros(length, dtype=bool)
    for i in range(2, length):
        if (
            bool(prices.close[i] > prices.open[i])
            and float(prices.close[i]) > float(custom_high[i - 1])
            and float(prices.close[i]) > float(custom_high[i - 2])
        ):
            tribar_green[i] = True

    kf_adx_increasing = np.zeros(length, dtype=bool)
    kf_atr_expanding = np.zeros(length, dtype=bool)
    if length >= 2:
        kf_adx_increasing[1:] = np.diff(derived.kf_adx) > 0.0
        kf_atr_expanding[1:] = np.diff(derived.kf_atr) > 0.0
    is_kf_breakout_potential = kf_adx_increasing & kf_atr_expanding

    trend_strength = np.full_like(derived.adx, np.nan, dtype=float)
    for i in range(trend_strength.shape[0]):
        price = prices.close[i]
        sma200_val = derived.sma200[i]
        adx_val = derived.adx[i]
        if not (np.isfinite(price) and np.isfinite(sma200_val) and np.isfinite(adx_val)):
            continue
        if abs(sma200_val) < np.finfo(float).eps:
            continue
        adx_term = (adx_val / 100.0) * 0.4
        deviation_term = abs((price / sma200_val) - 1.0) * 10.0 * 0.6
        trend_strength[i] = adx_term + deviation_term

    data = {
        "open": prices.open,
        "high": prices.high,
        "low": prices.low,
        "close": prices.close,
        "is_tribar_green": tribar_green,
        "is_kf_breakout_potential": is_kf_breakout_potential,
        "kf_innovation_abs": derived.kf_innovation_abs,
        "macd_hist": derived.macd_hist,
        "trend_strength": trend_strength,
        "atr": derived.atr,
        "kf_adx": derived.kf_adx,
        "kf_atr": derived.kf_atr,
    }

    df = pd.DataFrame(data)
    ts = infer_timestamp_column(df_raw)
    df["timestamp"] = ts
    return df


def build_aligned_frames(df_py: pd.DataFrame, df_rs: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    ts_py = infer_timestamp_column(df_py)
    ts_rs = infer_timestamp_column(df_rs)

    py = df_py.copy()
    py["__ts__"] = ts_py
    py = py.dropna(subset=["__ts__"])
    py = py.set_index("__ts__").sort_index()
    py = py.add_suffix("_py")

    rs = df_rs.copy()
    rs["__ts__"] = ts_rs
    rs = rs.dropna(subset=["__ts__"])
    rs = rs.set_index("__ts__").sort_index()
    rs = rs.add_suffix("_rs")

    merged = rs.join(py, how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping timestamps between Rust and Python frames")

    base_names: List[str] = []
    for col in rs.columns:
        if not col.endswith("_rs"):
            continue
        base = col[:-3]
        py_name = f"{base}_py"
        if py_name in merged.columns:
            base_names.append(base)
    base_names.sort()
    return merged, base_names


def coerce_boolish(series: pd.Series) -> pd.Series:
    """Best-effort conversion of bool/0-1/"true"/"false" columns into pandas nullable boolean."""
    if series.dtype == bool or str(series.dtype) == "boolean":
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if len(valid) > 0 and valid.isin([0, 1]).all():
            return valid.astype("Int64").astype("boolean").reindex(series.index)
        return pd.Series(pd.NA, index=series.index, dtype="boolean")

    # Strings/objects
    raw = series.astype("string")
    lowered = raw.str.lower().str.strip()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    true_set = {"true", "t", "1", "yes", "y"}
    false_set = {"false", "f", "0", "no", "n"}
    out[lowered.isin(true_set)] = True
    out[lowered.isin(false_set)] = False
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Barsmith indicator parity checker (Rust vs standalone Python port)")
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "barsmith/tests/data/es_30m_sample.csv",
        help="Raw input CSV used by Barsmith (default: barsmith/tests/data/es_30m_sample.csv)",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Path to barsmith_prepared.csv produced by custom_rs::prepare_dataset",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=None,
        help="Optional tick size to snap wick-based stops (ceil rounding), matching Barsmith CLI tick_size",
    )
    parser.add_argument(
        "--date-end",
        type=str,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD) used by multi-bar targets like highlow_or_atr to cap TP/SL resolution",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute difference tolerance to treat as exact match (default: 1e-9)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all shared columns, not just those exceeding tolerance",
    )

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: raw CSV not found: {args.csv}")
        return 1
    if not args.prepared.exists():
        print(f"ERROR: engineered CSV not found: {args.prepared}")
        return 1

    print("=" * 80)
    print("Barsmith Indicator Parity Check (Rust barsmith_prepared.csv vs standalone Python port)")
    print("=" * 80)
    print(f"Raw CSV:       {args.csv}")
    print(f"Prepared CSV:  {args.prepared}")
    print()

    print("Loading raw CSV...")
    df_raw = pd.read_csv(args.csv)
    print(f"  Raw rows: {len(df_raw)}")

    print("Loading Barsmith engineered dataset...")
    df_rs = pd.read_csv(args.prepared)
    print(f"  Rust frame: {df_rs.shape[0]} rows, {df_rs.shape[1]} columns")

    target = "next_bar_color_and_wicks"
    if "highlow_or_atr_tightest_stop_long" in df_rs.columns or "rr_highlow_or_atr_tightest_stop" in df_rs.columns:
        target = "highlow_or_atr_tightest_stop"
    elif "highlow_sl_2x_atr_tp_rr_gt_1_long" in df_rs.columns or "rr_highlow_sl_2x_atr_tp_rr_gt_1" in df_rs.columns:
        target = "highlow_sl_2x_atr_tp_rr_gt_1"
    elif "highlow_sl_1x_atr_tp_rr_gt_1_long" in df_rs.columns or "rr_highlow_sl_1x_atr_tp_rr_gt_1" in df_rs.columns:
        target = "highlow_sl_1x_atr_tp_rr_gt_1"
    elif "highlow_1r_long" in df_rs.columns or "rr_highlow_1r" in df_rs.columns:
        target = "highlow_1r"
    elif "atr_tp_atr_stop_long" in df_rs.columns or "rr_atr_tp_atr_stop" in df_rs.columns:
        target = "atr_tp_atr_stop"
    elif "2x_atr_tp_atr_stop_long" in df_rs.columns or "rr_2x_atr_tp_atr_stop" in df_rs.columns:
        target = "2x_atr_tp_atr_stop"
    elif "highlow_or_atr_long" in df_rs.columns or "rr_highlow_or_atr" in df_rs.columns:
        target = "highlow_or_atr"
    elif "wicks_kf_long" in df_rs.columns:
        target = "wicks_kf"
    elif "next_bar_color_and_wicks_long" in df_rs.columns:
        target = "next_bar_color_and_wicks"
    else:
        print("WARNING: could not infer target from prepared dataset columns; defaulting to next_bar_color_and_wicks")

    print(f"Computing Python indicators (standalone port of engineer.rs, target={target})...")
    if (
        target
        in {
            "highlow_or_atr",
            "highlow_or_atr_tightest_stop",
            "highlow_1r",
            "highlow_sl_2x_atr_tp_rr_gt_1",
            "highlow_sl_1x_atr_tp_rr_gt_1",
            "2x_atr_tp_atr_stop",
            "atr_tp_atr_stop",
        }
        and args.date_end is None
    ):
        print(
            "WARNING: target=highlow_or_atr* but --date-end not provided; parity may mismatch if the prepared dataset was built with --date-end"
        )
    df_py = build_python_indicator_frame(df_raw, tick_size=args.tick_size, target=target, date_end=args.date_end)
    print(f"  Python frame: {df_py.shape[0]} rows, {df_py.shape[1]} columns")

    print("Aligning frames by timestamp...")
    merged, base_names = build_aligned_frames(df_py, df_rs)
    aligned_rows = merged.shape[0]
    print(f"  Overlapping rows (by timestamp): {aligned_rows}")
    print(f"  Shared columns (candidate indicators): {len(base_names)}")

    skip_prefixes: Tuple[str, ...] = ()
    # Skip direction-aliased columns whose value depends on the Barsmith CLI direction.
    skip_exact = {
        "next_bar_color_and_wicks",
        "rr_next_bar_color_and_wicks",
        "wicks_kf",
        "rr_wicks_kf",
        "highlow_or_atr",
        "rr_highlow_or_atr",
        "highlow_or_atr_eligible",
        "highlow_or_atr_tightest_stop",
        "rr_highlow_or_atr_tightest_stop",
        "highlow_or_atr_tightest_stop_eligible",
        "highlow_1r",
        "rr_highlow_1r",
        "highlow_1r_eligible",
        "2x_atr_tp_atr_stop",
        "rr_2x_atr_tp_atr_stop",
        "2x_atr_tp_atr_stop_eligible",
        "atr_tp_atr_stop",
        "rr_atr_tp_atr_stop",
        "atr_tp_atr_stop_eligible",
        "highlow_sl_2x_atr_tp_rr_gt_1",
        "rr_highlow_sl_2x_atr_tp_rr_gt_1",
        "highlow_sl_2x_atr_tp_rr_gt_1_eligible",
        "highlow_sl_1x_atr_tp_rr_gt_1",
        "rr_highlow_sl_1x_atr_tp_rr_gt_1",
        "highlow_sl_1x_atr_tp_rr_gt_1_eligible",
    }

    results = []
    for name in base_names:
        if name in skip_exact or any(name.startswith(p) for p in skip_prefixes):
            continue

        col_rs = f"{name}_rs"
        col_py = f"{name}_py"
        rs_vals = pd.to_numeric(merged[col_rs], errors="coerce").to_numpy(dtype=float)
        py_vals = pd.to_numeric(merged[col_py], errors="coerce").to_numpy(dtype=float)

        rs_nan = ~np.isfinite(rs_vals)
        py_nan = ~np.isfinite(py_vals)
        valid_mask = ~(rs_nan | py_nan)
        valid_count = int(valid_mask.sum())

        if valid_count > 0:
            diff = rs_vals[valid_mask] - py_vals[valid_mask]
            abs_diff = np.abs(diff)
            max_abs = float(abs_diff.max())
            mean_abs = float(abs_diff.mean())
            mismatched = int((abs_diff > args.tolerance).sum())
        else:
            rs_bool = coerce_boolish(merged[col_rs])
            py_bool = coerce_boolish(merged[col_py])
            valid_bool = rs_bool.notna() & py_bool.notna()
            valid_count = int(valid_bool.sum())
            if valid_count == 0:
                continue
            mismatched = int((rs_bool[valid_bool] != py_bool[valid_bool]).sum())
            max_abs = 1.0 if mismatched > 0 else 0.0
            mean_abs = mismatched / float(valid_count)

        results.append(
            {
                "name": name,
                "valid": valid_count,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "mismatched": mismatched,
            }
        )

    if not results:
        print("No comparable indicator columns found after filtering.")
        return 0

    results.sort(key=lambda r: r["max_abs"], reverse=True)

    print("\n" + "=" * 80)
    print("COLUMN-BY-COLUMN DIFFERENCES")
    print("=" * 80)
    print(f"{'Column':<32} {'valid':>8} {'mismatched':>11} " f"{'max_abs':>14} {'mean_abs':>14}")
    print("-" * 80)

    for r in results:
        if not args.show_all and r["max_abs"] <= args.tolerance and r["mismatched"] == 0:
            continue
        print(f"{r['name']:<32} {r['valid']:>8} {r['mismatched']:>11} " f"{r['max_abs']:>14.6g} {r['mean_abs']:>14.6g}")

    worst = results[0]
    avg_max = float(np.mean([r["max_abs"] for r in results]))
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Worst column: {worst['name']} (max_abs={worst['max_abs']:.6g})")
    print(f"Average max_abs over {len(results)} columns: {avg_max:.6g}")
    print(f"Tolerance: {args.tolerance}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

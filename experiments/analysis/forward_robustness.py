from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ForwardRobustnessComponents:
    frs: float
    k: int
    p: float
    c: float
    c_plus: float
    dd_min: float
    dd_median: float
    dd_mean: float
    dd_max: float
    t: float
    tail_penalty: float
    r_min: float
    r_median: float
    r_max: float
    mu_r: float
    sigma_r: float
    stability: float
    n_med: float
    n_min: int
    trade_score: float


def compute_frs(
    returns_r: Iterable[float],
    max_drawdowns_r: Iterable[float],
    trades: Iterable[int],
    *,
    eps: float = 1e-9,
    n_min: int = 30,
    alpha: float = 2.0,
    beta: float = 2.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> ForwardRobustnessComponents:
    """
    Compute a Forward Robustness Score (FRS) over anchored forward windows.

    Inputs are per-forward-window values:
      - returns_r: total return in R
      - max_drawdowns_r: max drawdown in R (non-negative)
      - trades: number of trades
    """
    r = np.asarray(list(returns_r), dtype=float)
    dd = np.asarray(list(max_drawdowns_r), dtype=float)
    n = np.asarray(list(trades), dtype=float)

    if r.size != dd.size or r.size != n.size:
        raise ValueError("FRS input lengths must match")

    k = int(r.size)
    if k == 0:
        return ForwardRobustnessComponents(
            frs=0.0,
            k=0,
            p=0.0,
            c=0.0,
            c_plus=0.0,
            dd_min=0.0,
            dd_median=0.0,
            dd_mean=0.0,
            dd_max=0.0,
            t=0.0,
            tail_penalty=0.0,
            r_min=0.0,
            r_median=0.0,
            r_max=0.0,
            mu_r=0.0,
            sigma_r=0.0,
            stability=0.0,
            n_med=0.0,
            n_min=int(n_min),
            trade_score=0.0,
        )

    eps = float(eps)
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    dd = np.maximum(dd, 0.0)
    p = float(np.mean(r > 0.0))

    calmar_i = r / (dd + eps)
    c = float(np.median(calmar_i))
    c_plus = float(max(0.0, c))

    dd_min = float(np.min(dd))
    dd_median = float(np.median(dd))
    dd_mean = float(np.mean(dd))
    dd_max = float(np.max(dd))
    t = float(dd_max / (dd_median + eps))
    tail_penalty = float(1.0 / (1.0 + t))

    r_min = float(np.min(r))
    r_median = float(np.median(r))
    r_max = float(np.max(r))
    abs_dev = np.abs(r - r_median)
    mad_r = float(np.median(abs_dev))

    mu_r = float(np.mean(r))
    sigma_r = float(np.std(r, ddof=0))

    # Robust stability: prefer median/MAD so one-off outlier windows don't dominate
    # this component (tail risk is handled separately by tail_penalty).
    stability = float(1.0 / (1.0 + (mad_r / (abs(r_median) + eps))))

    n_med = float(np.median(n))
    n_min_int = int(n_min)
    if n_min_int <= 0:
        trade_score = 1.0
    else:
        trade_score = float(min(1.0, n_med / float(n_min_int)))

    frs = float(
        (p ** float(alpha)) * c_plus * (tail_penalty ** float(beta)) * (stability ** float(gamma)) * (trade_score ** float(delta))
    )

    return ForwardRobustnessComponents(
        frs=frs,
        k=k,
        p=p,
        c=c,
        c_plus=c_plus,
        dd_min=dd_min,
        dd_median=dd_median,
        dd_mean=dd_mean,
        dd_max=dd_max,
        t=t,
        tail_penalty=tail_penalty,
        r_min=r_min,
        r_median=r_median,
        r_max=r_max,
        mu_r=mu_r,
        sigma_r=sigma_r,
        stability=stability,
        n_med=n_med,
        n_min=n_min_int,
        trade_score=trade_score,
    )

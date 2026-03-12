from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)

Mode = Literal["annual-lumpsum", "monthly-dca"]
Universe = Literal["equities", "all"]
PriceMode = Literal["adj", "raw"]
SellPricePolicy = Literal["no-forward", "defer-forward"]
RankMetric = Literal[
    "cagr_to_mae",
    "cagr_to_mae_median",
    "calmar",
    "cagr_5y",
    "prev_12m_return",
    "cagr_to_trailing_low_5y",
    "cagr_to_trailing_low_5y_daily",
]
MonthlySellPolicy = Literal["maturity", "never"]

# pandas hashing requires a 16-byte (encoded) key; keep this exactly 16 characters for determinism.
HASH_KEY = "entry_rotation_1"
CACHE_VERSION = 5


def parse_cli_number(raw: str | int | float) -> float:
    """
    Parse a human-friendly numeric CLI value.

    Accepts:
    - underscores: 10_000_000
    - commas: 10,000,000
    - scientific notation: 1e7
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    s = s.replace("_", "").replace(",", "")
    return float(s)


def _years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    days = (end - start).days + (end - start).seconds / 86400.0
    return float(days / 365.25) if days > 0 else 0.0


def _xnpv(rate: float, *, dates: np.ndarray, amounts: np.ndarray) -> float:
    if not (np.isfinite(rate) and rate > -1.0):
        return np.nan
    if len(dates) == 0:
        return np.nan
    t0 = dates[0]
    years = (dates - t0).astype("timedelta64[D]").astype("float64") / 365.25
    denom = np.power(1.0 + float(rate), years, dtype="float64")
    if not np.all(np.isfinite(denom)) or np.any(denom <= 0):
        return np.nan
    return float(np.sum(amounts / denom))


def _xirr(*, dates: list[pd.Timestamp], amounts: list[float]) -> float:
    """
    Compute annualized IRR for irregular cash flows (XIRR).

    Convention:
    - contributions (investor pays in) are negative
    - terminal withdrawal (final equity) is positive
    """
    if len(dates) != len(amounts) or len(dates) < 2:
        return float("nan")
    d = np.array([np.datetime64(pd.Timestamp(x).normalize()) for x in dates], dtype="datetime64[D]")
    a = np.array([float(x) for x in amounts], dtype="float64")
    # Need at least one negative and one positive flow for a meaningful IRR.
    if not (np.any(a < 0) and np.any(a > 0)):
        return float("nan")

    order = np.argsort(d, kind="mergesort")
    d = d[order]
    a = a[order]

    low = -0.9999
    high = 1.0
    f_low = _xnpv(low, dates=d, amounts=a)
    if not np.isfinite(f_low):
        return float("nan")
    f_high = _xnpv(high, dates=d, amounts=a)
    # Expand high until we bracket a root or give up.
    tries = 0
    while (not np.isfinite(f_high) or np.sign(f_low) == np.sign(f_high)) and high < 1e6 and tries < 60:
        high *= 2.0
        f_high = _xnpv(high, dates=d, amounts=a)
        tries += 1
    if not np.isfinite(f_high) or np.sign(f_low) == np.sign(f_high):
        return float("nan")

    lo = low
    hi = high
    flo = f_low
    fhi = f_high
    for _ in range(120):
        mid = (lo + hi) / 2.0
        fmid = _xnpv(mid, dates=d, amounts=a)
        if not np.isfinite(fmid):
            return float("nan")
        if abs(fmid) < 1e-10:
            return float(mid)
        if np.sign(fmid) == np.sign(flo):
            lo = mid
            flo = fmid
        else:
            hi = mid
            fhi = fmid
    return float((lo + hi) / 2.0)


def _twr_from_equity_curve(
    *,
    dates: pd.Series,
    equity: pd.Series,
    cash_flows: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compute time-weighted return from an equity curve and external cash flows.

    Cash flow timing convention: treat flows recorded on date D as occurring at the START of date D.
    The cash_flows.csv uses investor convention (contribution negative), so for TWR we convert to portfolio convention (CF_portfolio = -amount).
    """
    if equity.empty or len(equity) != len(dates):
        return float("nan"), float("nan")
    dates = pd.to_datetime(dates)
    equity = pd.to_numeric(equity, errors="coerce").astype("float64")
    if len(equity) < 2:
        return float("nan"), float("nan")
    start = pd.Timestamp(dates.iloc[0])
    end = pd.Timestamp(dates.iloc[-1])
    years = _years_between(start, end)
    if years <= 0:
        return float("nan"), float("nan")

    cf = cash_flows.copy() if cash_flows is not None and not cash_flows.empty else pd.DataFrame(columns=["date", "amount"])
    if not cf.empty:
        cf["date"] = pd.to_datetime(cf["date"])
        cf["amount"] = pd.to_numeric(cf["amount"], errors="coerce").astype("float64")
        cf["date_norm"] = cf["date"].dt.normalize()
        cf = cf.groupby("date_norm", sort=True, as_index=False)["amount"].sum()
    cf_map = {pd.Timestamp(r["date_norm"]): float(r["amount"]) for _, r in cf.iterrows()} if not cf.empty else {}

    log_sum = 0.0
    for i in range(1, len(equity)):
        v_prev = float(equity.iloc[i - 1])
        v = float(equity.iloc[i])
        if not (np.isfinite(v_prev) and v_prev > 0 and np.isfinite(v)):
            return float("nan"), float("nan")
        d = pd.Timestamp(dates.iloc[i]).normalize()
        # Convert investor sign to portfolio sign.
        cf_port = -float(cf_map.get(d, 0.0))
        r = ((v - cf_port) / v_prev) - 1.0
        if not np.isfinite(r) or r <= -1.0:
            return float("nan"), float("nan")
        log_sum += float(np.log1p(r))

    twr_total = float(np.expm1(log_sum))
    twr_cagr = float(np.power(1.0 + twr_total, 1.0 / years) - 1.0) if twr_total > -1.0 else float("nan")
    return twr_total, twr_cagr


def _write_summary_stats(
    *,
    out_root: Path,
    equity_df: pd.DataFrame,
    cash_flows: list[dict[str, Any]],
    cfg: EntryRotationConfig,
) -> None:
    if equity_df.empty or "date" not in equity_df.columns or "equity" not in equity_df.columns:
        return
    dates = pd.to_datetime(equity_df["date"])
    start = pd.Timestamp(dates.iloc[0])
    end = pd.Timestamp(dates.iloc[-1])
    years = _years_between(start, end)

    def cagr(v0: float, v1: float) -> float:
        if years <= 0 or not (np.isfinite(v0) and v0 > 0 and np.isfinite(v1) and v1 > 0):
            return float("nan")
        return float(np.power(v1 / v0, 1.0 / years) - 1.0)

    cf_df = pd.DataFrame(cash_flows) if cash_flows else pd.DataFrame(columns=["date", "amount"])
    if not cf_df.empty:
        cf_df["date"] = pd.to_datetime(cf_df["date"])
        cf_df["amount"] = pd.to_numeric(cf_df["amount"], errors="coerce").astype("float64")

    s0 = float(equity_df["equity"].iloc[0])
    s1 = float(equity_df["equity"].iloc[-1])
    s_cagr = cagr(s0, s1)
    s_twr_total, s_twr_cagr = _twr_from_equity_curve(dates=equity_df["date"], equity=equity_df["equity"], cash_flows=cf_df)
    s_irr = _xirr(
        dates=[start, *([pd.Timestamp(x) for x in cf_df["date"].tolist()] if not cf_df.empty else []), end],
        amounts=[
            -float(cfg.initial_equity),
            *([float(x) for x in cf_df["amount"].tolist()] if not cf_df.empty else []),
            float(s1),
        ],
    )

    out: dict[str, Any] = {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "years": years,
        "final_equity": s1,
        "cagr": s_cagr,
        "twr_total": s_twr_total,
        "twr_cagr": s_twr_cagr,
        "irr": s_irr,
    }

    if not cfg.no_benchmark and "bm_equity" in equity_df.columns:
        b0 = float(equity_df["bm_equity"].iloc[0])
        b1 = float(equity_df["bm_equity"].iloc[-1])
        b_cagr = cagr(b0, b1)
        b_twr_total, b_twr_cagr = _twr_from_equity_curve(dates=equity_df["date"], equity=equity_df["bm_equity"], cash_flows=cf_df)
        b_irr = _xirr(
            dates=[start, *([pd.Timestamp(x) for x in cf_df["date"].tolist()] if not cf_df.empty else []), end],
            amounts=[
                -float(cfg.initial_equity),
                *([float(x) for x in cf_df["amount"].tolist()] if not cf_df.empty else []),
                float(b1),
            ],
        )
        out |= {
            "bm_final_equity": b1,
            "bm_cagr": b_cagr,
            "bm_twr_total": b_twr_total,
            "bm_twr_cagr": b_twr_cagr,
            "bm_irr": b_irr,
        }

    pd.DataFrame([out]).to_csv(out_root / "summary_stats.csv", index=False)


@dataclass(frozen=True)
class SharadarPaths:
    tickers_csv: Path
    sep_csv: Path
    sfp_csv: Path
    daily_csv: Path


@dataclass(frozen=True)
class EntryRotationConfig:
    mode: Mode
    start_year: int
    end_year: int
    sharadar_dir: Path
    cache_dir: Path
    output_dir: Path
    price_mode: PriceMode = "raw"
    top_n: int = 12
    entry_month: int = 1
    hold_period_months: int = 12
    universe: Universe = "all"
    exclude_lp: bool = True
    dedupe_issuer: bool = True
    exclude_preferred: bool = True
    exclude_quasi_preferred: bool = True
    exclude_adr_common: bool = True
    exclude_bond_like_etfs: bool = True
    initial_equity: float = 100_000.0
    monthly_dca_amount: float = 0.0
    initial_lump_sum: float = 0.0
    initial_lump_dca_months: int = 0
    cash_sweep_ticker: str = "BIL"
    cash_sweep_enabled: bool = True
    monthly_sell_policy: MonthlySellPolicy = "maturity"
    max_new_tickers_per_year: int = 12
    mcap_floor: float = 0.0
    dollar_volume_floor: float = 0.0
    min_history_years: int = 5
    min_history_bars: int = 900
    dollar_volume_floor_missing_mcap: float = 50_000_000.0
    dollar_volume_lookback_bars: int = 252
    require_last_n_years_green: int = 3
    rank_metric: RankMetric = "cagr_to_mae"
    rank_lookback_years: int = 5
    min_filters: tuple[tuple[str, float], ...] = ()
    selection_benchmark_ticker: str = "SPY"
    benchmark_ticker: str = "SPY"
    no_benchmark: bool = False
    no_sfp_blacklist: bool = False
    sfp_blacklist_extra: tuple[str, ...] = ()
    cash_buffer_pct: float = 0.0
    commission_per_trade: float = 0.35
    slippage_bps: float = 10.0
    dividend_withholding_rate: float = 0.0
    sell_price_policy: SellPricePolicy = "defer-forward"
    allow_overlap_on_sell_block: bool = False
    mae_stop_pct: float = 0.0
    position_mae_stop_pct: float = 0.0
    stopout_cooldown_months: int = 0
    stopout_reentry_dca_months: int = 0
    num_buckets: int = 32
    jobs: int = 8

    def __post_init__(self) -> None:
        if self.price_mode not in ("adj", "raw"):
            raise ValueError("--price-mode must be one of: adj, raw")
        if not (1 <= self.entry_month <= 12):
            raise ValueError("--entry-month must be in [1..12]")
        if self.hold_period_months != 12:
            raise ValueError("hold_period_months is fixed at 12 for this strategy family")
        if self.end_year < self.start_year:
            raise ValueError("--end-year must be >= --start-year")
        if self.top_n <= 0:
            raise ValueError("--top-n must be > 0")
        if self.require_last_n_years_green < 0:
            raise ValueError("--require-last-n-years-green must be >= 0")
        if not (0.0 <= self.cash_buffer_pct < 1.0):
            raise ValueError("--cash-buffer-pct must be in [0,1)")
        if not (0.0 <= float(self.dividend_withholding_rate) < 1.0):
            raise ValueError("--dividend-withholding-rate must be in [0,1)")
        if float(self.dividend_withholding_rate) > 0 and self.price_mode != "raw":
            raise ValueError(
                "--dividend-withholding-rate requires --price-mode raw (price-only) so dividends can be modeled explicitly."
            )
        if not (0.0 <= self.mae_stop_pct < 1.0):
            raise ValueError("--mae-stop-pct must be in [0,1)")
        if not (0.0 <= self.position_mae_stop_pct < 1.0):
            raise ValueError("--position-mae-stop-pct must be in [0,1)")
        if self.stopout_cooldown_months < 0:
            raise ValueError("--stopout-cooldown-months must be >= 0")
        if self.stopout_reentry_dca_months < 0:
            raise ValueError("--stopout-reentry-dca-months must be >= 0")
        if self.monthly_dca_amount < 0:
            raise ValueError("--monthly-dca-amount must be >= 0")
        if self.mcap_floor < 0:
            raise ValueError("--mcap-floor must be >= 0")
        if self.dollar_volume_floor < 0:
            raise ValueError("--dollar-volume-floor must be >= 0")
        if self.dollar_volume_floor_missing_mcap < 0:
            raise ValueError("--dollar-volume-floor-missing-mcap must be >= 0")
        if self.mode != "monthly-dca" and self.stopout_cooldown_months != 0:
            raise ValueError("--stopout-cooldown-months applies to monthly-dca only (set to 0 otherwise)")
        if self.mode != "monthly-dca" and self.stopout_reentry_dca_months != 0:
            raise ValueError("--stopout-reentry-dca-months applies to monthly-dca only (set to 0 otherwise)")
        if self.mode != "monthly-dca" and (self.initial_lump_sum != 0.0 or self.initial_lump_dca_months != 0):
            raise ValueError(
                "--initial-lump-* applies to monthly-dca only (set --initial-lump-sum 0 and --initial-lump-dca-months 0)"
            )
        if self.mode == "monthly-dca" and self.monthly_dca_amount <= 0:
            raise ValueError("--monthly-dca-amount must be > 0 in monthly-dca mode")
        if self.mode == "monthly-dca":
            if self.monthly_sell_policy not in ("maturity", "never"):
                raise ValueError("--sell-policy must be one of: maturity, never")
            if self.max_new_tickers_per_year <= 0:
                raise ValueError("--max-new-tickers-per-year must be > 0")
        else:
            if self.monthly_sell_policy != "maturity":
                raise ValueError("--sell-policy applies to monthly-dca only (use maturity otherwise)")
            if self.max_new_tickers_per_year != 12:
                raise ValueError("--max-new-tickers-per-year applies to monthly-dca only")
        if self.sell_price_policy not in ("no-forward", "defer-forward"):
            raise ValueError("--sell-price-policy must be one of: no-forward, defer-forward")
        if self.universe not in ("equities", "all"):
            raise ValueError("--universe must be one of: equities, all")
        if self.rank_metric not in (
            "cagr_to_mae",
            "cagr_to_mae_median",
            "calmar",
            "cagr_5y",
            "prev_12m_return",
            "cagr_to_trailing_low_5y",
            "cagr_to_trailing_low_5y_daily",
        ):
            raise ValueError(
                "--rank-metric must be one of: cagr_to_mae, cagr_to_mae_median, calmar, cagr_5y, prev_12m_return, cagr_to_trailing_low_5y, cagr_to_trailing_low_5y_daily"
            )
        if int(self.rank_lookback_years) not in (3, 5):
            raise ValueError("--rank-lookback-years must be 3 or 5")
        if self.min_filters:
            allowed = {
                "cagr_3y",
                "cagr_5y",
                "prev_12m_return",
                "cagr_to_mae_3y",
                "cagr_to_mae_median_3y",
                "cagr_to_mae_5y",
                "cagr_to_mae_median_5y",
                "calmar_3y",
                "calmar_5y",
                "max_dd_3y",
                "max_dd_5y",
                "cagr_to_trailing_low_3y",
                "cagr_to_trailing_low_5y",
            }
            for metric, min_value in self.min_filters:
                if metric not in allowed:
                    raise ValueError(f"--min-filter metric must be one of: {', '.join(sorted(allowed))} (got {metric})")
                if not (isinstance(min_value, (int, float)) and np.isfinite(float(min_value))):
                    raise ValueError(f"--min-filter value must be finite for {metric}")


def _whole_shares_for_spendable_cash(*, cash: float, effective_price: float, commission: float, cash_buffer_pct: float) -> int:
    """
    Compute whole-share quantity using (almost) all available cash, respecting a cash buffer and commission.

    This is used for cash sweep buys (e.g., BIL) where we want to deploy leftover cash, not equal-weight it.
    """
    if not (np.isfinite(cash) and cash > 0):
        return 0
    if not (np.isfinite(effective_price) and effective_price > 0):
        return 0
    if not (np.isfinite(commission) and commission >= 0):
        return 0
    if not (0.0 <= cash_buffer_pct < 1.0):
        return 0
    spendable = float(cash) * (1.0 - float(cash_buffer_pct))
    if spendable <= commission:
        return 0
    shares = int(math.floor((spendable - commission) / float(effective_price)))
    return max(shares, 0)


def _position_mae_pct_from_cache(
    *,
    price_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    instrument_id: str,
    date: pd.Timestamp,
    entry_effective_price: float,
) -> float | None:
    """
    Per-position MAE, computed from entry_effective_price to the day's low price.

    Uses the same "low fallback to close" rule as portfolio equity_low.
    """
    if not np.isfinite(entry_effective_price) or entry_effective_price <= 0:
        return None
    cached = price_cache.get(instrument_id)
    if cached is None:
        return None
    dates, _open_px, low_px, close_px = cached
    ix = np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side="right") - 1
    if ix < 0:
        return None
    c = float(close_px[ix])
    l = float(low_px[ix])
    used = l if np.isfinite(l) and l > 0 else c
    if not np.isfinite(used) or used <= 0:
        return None
    return (used / float(entry_effective_price)) - 1.0


def _next_select_date_after(select_dates: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    ix = select_dates.searchsorted(pd.Timestamp(date).to_datetime64(), side="right")
    if ix < len(select_dates):
        return pd.Timestamp(select_dates[ix])
    return None


def _reserve_new_name_for_year(reserved: dict[int, set[str]], *, year: int, instrument_id: str) -> None:
    reserved.setdefault(int(year), set()).add(str(instrument_id))


def _release_reserved_new_name_for_year(reserved: dict[int, set[str]], *, year: int, instrument_id: str) -> None:
    y = int(year)
    iid = str(instrument_id)
    s = reserved.get(y)
    if not s:
        return
    s.discard(iid)
    if not s:
        reserved.pop(y, None)


def _count_opened_new_names_for_year(opened: dict[int, set[str]], *, year: int) -> int:
    s = opened.get(int(year))
    return len(s) if s is not None else 0


def _count_reserved_new_names_for_year(reserved: dict[int, set[str]], *, year: int) -> int:
    s = reserved.get(int(year))
    return len(s) if s is not None else 0


def _equity_mask_from_categories(categories: np.ndarray) -> np.ndarray:
    """
    Define the "equities only" universe from Sharadar TICKERS.category.

    We keep common/preferred equity and exclude warrants/derivative-like categories.
    """
    cats = np.asarray(categories, dtype="U")
    cats = np.char.strip(np.char.lower(cats))
    has_common = np.char.find(cats, "common stock") >= 0
    has_preferred = np.char.find(cats, "preferred stock") >= 0
    has_ordinary = np.char.find(cats, "ordinary share") >= 0
    has_warrant = np.char.find(cats, "warrant") >= 0
    return (has_common | has_preferred | has_ordinary) & (~has_warrant)


def _year_month_from_date(d: pd.Timestamp) -> str:
    ts = pd.Timestamp(d)
    return f"{int(ts.year):04d}-{int(ts.month):02d}"


def discover_sharadar_files(sharadar_dir: Path) -> SharadarPaths:
    sharadar_dir = sharadar_dir.expanduser().resolve()
    if not sharadar_dir.exists():
        raise FileNotFoundError(f"Sharadar dir not found: {sharadar_dir}")

    def pick_latest(pattern: str) -> Path:
        matches = sorted(sharadar_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing Sharadar file matching {pattern} in {sharadar_dir}")
        return matches[-1]

    return SharadarPaths(
        tickers_csv=pick_latest("SHARADAR_TICKERS_*.csv"),
        sep_csv=pick_latest("SHARADAR_SEP_*.csv"),
        sfp_csv=pick_latest("SHARADAR_SFP_*.csv"),
        daily_csv=pick_latest("SHARADAR_DAILY_*.csv"),
    )


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint_file(path: Path, compute_sha256: bool = True) -> dict[str, Any]:
    st = path.stat()
    fp: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }
    if compute_sha256:
        fp["sha256"] = _sha256_file(path)
    return fp


def _fingerprint_file_cached(path: Path, prev: dict[str, Any] | None) -> dict[str, Any]:
    """
    Avoid re-hashing huge input files on every run.

    If the previous manifest fingerprint matches (path/size/mtime) and includes sha256, reuse it.
    Otherwise compute sha256 once and return the full fingerprint.
    """
    st = path.stat()
    quick = {"path": str(path.resolve()), "size": int(st.st_size), "mtime": float(st.st_mtime)}
    if (
        prev
        and prev.get("path") == quick["path"]
        and int(prev.get("size", -1)) == quick["size"]
        and float(prev.get("mtime", -1.0)) == quick["mtime"]
        and "sha256" in prev
    ):
        return dict(prev)
    quick["sha256"] = _sha256_file(path)
    return quick


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        for child in path.glob("**/*"):
            if child.is_file():
                child.unlink()
        for child in sorted(path.glob("**/*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        path.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _resolve_columns(header: list[str], required: Iterable[str]) -> dict[str, str]:
    lower_to_orig = {c.strip().lower(): c.strip() for c in header}
    mapping: dict[str, str] = {}
    for need in required:
        key = need.lower()
        if key not in lower_to_orig:
            raise KeyError(f"Missing column '{need}' in CSV header")
        mapping[need] = lower_to_orig[key]
    return mapping


def _dedupe_latest_lastupdated(df: pd.DataFrame, keys: list[str], lastupdated_col: str = "lastupdated") -> pd.DataFrame:
    if df.empty:
        return df
    if lastupdated_col not in df.columns:
        return df.drop_duplicates(keys, keep="last")
    out = df.copy()
    out[lastupdated_col] = pd.to_datetime(out[lastupdated_col], errors="coerce")
    out = out.sort_values(keys + [lastupdated_col], kind="mergesort").drop_duplicates(keys, keep="last")
    return out


def _bucket_for_symbol_keys(symbol_keys: pd.Series, num_buckets: int) -> pd.Series:
    hashed = pd.util.hash_pandas_object(symbol_keys, index=False, hash_key=HASH_KEY).astype("uint64")
    return (hashed % np.uint64(num_buckets)).astype("int16")


def _bucket_for_symbol_key(symbol_key: str, num_buckets: int) -> int:
    s = pd.Series([symbol_key], dtype="string")
    return int(_bucket_for_symbol_keys(s, num_buckets).iloc[0])


def _parquet_write_partitioned(
    df: pd.DataFrame,
    out_root: Path,
    partition_cols: tuple[str, ...],
    part_counter: dict[tuple[Any, ...], int],
    compression: str = "zstd",
) -> None:
    if df.empty:
        return
    group_cols = list(partition_cols)
    for keys, sub in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_tuple = tuple(keys)
        part = part_counter.get(key_tuple, 0)
        part_counter[key_tuple] = part + 1

        dir_path = out_root
        for col, key in zip(partition_cols, key_tuple, strict=True):
            dir_path = dir_path / f"{col}={key}"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"part-{part:06d}.parquet"

        # Avoid duplicating partition columns in the parquet file; pyarrow can infer them from hive-style directories.
        sub = sub.drop(columns=group_cols, errors="ignore")
        table = pa.Table.from_pandas(sub.reset_index(drop=True), preserve_index=False)
        pq.write_table(table, file_path, compression=compression)


def _year_bin_5y(date_series: pd.Series) -> pd.Series:
    years = pd.to_datetime(date_series).dt.year.astype("int32")
    return (years // 5) * 5


def ensure_sharadar_caches(cfg: EntryRotationConfig, *, rebuild: bool = False) -> SharadarPaths:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_sharadar_files(cfg.sharadar_dir)

    tickers_dir = cfg.cache_dir / "tickers"
    bars_dir = cfg.cache_dir / "bars"
    daily_dir = cfg.cache_dir / "daily"
    bars_pt_dir = cfg.cache_dir / "bars_permaticker"
    daily_pt_dir = cfg.cache_dir / "daily_permaticker"
    month_index_dir = cfg.cache_dir / "month_index"
    candidates_dir = cfg.cache_dir / "candidates_monthly"

    _ensure_tickers_cache(paths, tickers_dir, rebuild=rebuild)
    _ensure_bars_cache(paths, bars_dir, num_buckets=cfg.num_buckets, rebuild=rebuild)
    _ensure_daily_cache(paths, daily_dir, num_buckets=cfg.num_buckets, rebuild=rebuild)
    _ensure_bars_permaticker_cache(bars_dir, tickers_dir, bars_pt_dir, cfg, rebuild=rebuild)
    _ensure_daily_permaticker_cache(daily_dir, tickers_dir, daily_pt_dir, cfg, rebuild=rebuild)
    _ensure_month_index_cache(bars_dir, tickers_dir, month_index_dir, cfg, rebuild=rebuild)
    _ensure_candidates_cache(bars_pt_dir, daily_pt_dir, tickers_dir, month_index_dir, candidates_dir, cfg, rebuild=rebuild)

    return paths


def _manifest_matches(
    manifest: dict[str, Any], *, cache_version: int, inputs: dict[str, Any], config_key: dict[str, Any]
) -> bool:
    if not manifest:
        return False
    if int(manifest.get("cache_version", -1)) != cache_version:
        return False
    if manifest.get("inputs") != inputs:
        return False
    if manifest.get("config_key") != config_key:
        return False
    return True


def _ensure_tickers_cache(paths: SharadarPaths, tickers_dir: Path, *, rebuild: bool) -> None:
    tickers_parquet = tickers_dir / "tickers.parquet"
    manifest_path = tickers_dir / "_manifest.json"
    config_key = {"schema": "tickers_v4_price_tables_only_with_category_name"}

    manifest = _read_manifest(manifest_path)
    prev_inputs = (manifest or {}).get("inputs", {}) if manifest else {}
    inputs = {"tickers_csv": _fingerprint_file_cached(paths.tickers_csv, prev_inputs.get("tickers_csv"))}
    if (
        not rebuild
        and tickers_parquet.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building tickers cache: %s", tickers_dir)
    _ensure_empty_dir(tickers_dir)

    header = _read_csv_header(paths.tickers_csv)
    cols = _resolve_columns(
        header,
        required=[
            "table",
            "ticker",
            "permaticker",
            "firstpricedate",
            "lastpricedate",
            "lastupdated",
        ],
    )
    # Optional universe metadata (present in standard Sharadar TICKERS exports).
    lower_to_orig = {c.strip().lower(): c.strip() for c in header}
    if "category" in lower_to_orig:
        cols["category"] = lower_to_orig["category"]
    if "name" in lower_to_orig:
        cols["name"] = lower_to_orig["name"]
    df = pd.read_csv(
        paths.tickers_csv,
        usecols=list(cols.values()),
        parse_dates=[cols["firstpricedate"], cols["lastpricedate"], cols["lastupdated"]],
        dtype={
            cols["table"]: "string",
            cols["ticker"]: "string",
            **({cols["category"]: "string"} if "category" in cols else {}),
            **({cols["name"]: "string"} if "name" in cols else {}),
        },
    )
    df = df.rename(columns={v: k for k, v in cols.items()})
    df["table"] = df["table"].str.upper()
    df["ticker"] = df["ticker"].str.upper()
    # Sharadar TICKERS includes SF1 (fundamentals) rows for the same permaticker; we only want price tables.
    df = df[df["table"].isin(["SEP", "SFP"])].copy()
    df["symbol_key"] = df["table"] + ":" + df["ticker"]
    df["permaticker"] = df["permaticker"].astype("Int64")
    if "category" in df.columns:
        df["category"] = df["category"].astype("string")
    if "name" in df.columns:
        df["name"] = df["name"].astype("string")

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tickers_parquet, compression="zstd")
    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
            "rows": int(len(df)),
        },
    )


def _ensure_bars_cache(paths: SharadarPaths, bars_dir: Path, *, num_buckets: int, rebuild: bool) -> None:
    manifest_path = bars_dir / "_manifest.json"
    config_key = {"schema": "bars", "num_buckets": int(num_buckets)}
    manifest = _read_manifest(manifest_path)
    prev_inputs = (manifest or {}).get("inputs", {}) if manifest else {}
    inputs = {
        "sep_csv": _fingerprint_file_cached(paths.sep_csv, prev_inputs.get("sep_csv")),
        "sfp_csv": _fingerprint_file_cached(paths.sfp_csv, prev_inputs.get("sfp_csv")),
    }
    if (
        not rebuild
        and bars_dir.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building bars cache: %s", bars_dir)
    _ensure_empty_dir(bars_dir)

    required = ["date", "ticker", "open", "high", "low", "close", "closeadj", "closeunadj", "volume"]

    part_counter: dict[tuple[Any, ...], int] = {}
    for table_name, csv_path in [("SEP", paths.sep_csv), ("SFP", paths.sfp_csv)]:
        header = _read_csv_header(csv_path)
        cols = _resolve_columns(header, required=required + ["lastupdated"])
        usecols = list(cols.values())
        date_col = cols["date"]

        for chunk in tqdm(
            pd.read_csv(
                csv_path,
                usecols=usecols,
                parse_dates=[date_col],
                chunksize=1_000_000,
                dtype={cols["ticker"]: "string"},
            ),
            desc=f"bars:{table_name}",
        ):
            chunk = chunk.rename(columns={v: k for k, v in cols.items()})
            chunk["table"] = table_name
            chunk["ticker"] = chunk["ticker"].str.upper()
            chunk["symbol_key"] = chunk["table"] + ":" + chunk["ticker"]
            # Best-effort within-chunk dedupe; full dedupe is enforced at read time.
            chunk = _dedupe_latest_lastupdated(chunk, keys=["symbol_key", "date"], lastupdated_col="lastupdated")

            # Sharadar SEP/SFP: open/high/low/close are split-adjusted prices; closeadj is total-return adjusted close; closeunadj is raw close.
            # To make OHLC consistent with closeadj, apply only the dividend/total-return adjustment factor: closeadj / close.
            close = pd.to_numeric(chunk["close"], errors="coerce").astype("float64")
            close_adj = pd.to_numeric(chunk["closeadj"], errors="coerce").astype("float64")
            close_unadj = pd.to_numeric(chunk["closeunadj"], errors="coerce").astype("float64")
            div_factor = close_adj / close

            valid = (close_adj > 0) & (close > 0) & (close_unadj > 0) & np.isfinite(div_factor) & (div_factor > 0)
            chunk = chunk.loc[valid].copy()
            if chunk.empty:
                continue

            close_adj = close_adj.loc[valid]
            close_unadj = close_unadj.loc[valid]
            div_factor = div_factor.loc[valid]

            for col in ["open", "high", "low", "close", "volume"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float64")

            # Raw/split-adjusted OHLC (dividends not included).
            chunk["open_raw"] = chunk["open"]
            chunk["high_raw"] = chunk["high"]
            chunk["low_raw"] = chunk["low"]
            chunk["close_raw"] = chunk["close"]

            chunk["close_adj"] = close_adj
            chunk["close_unadj"] = close_unadj
            chunk["div_factor"] = div_factor
            chunk["open_adj"] = chunk["open"] * div_factor
            chunk["high_adj"] = chunk["high"] * div_factor
            chunk["low_adj"] = chunk["low"] * div_factor

            chunk["year"] = chunk["date"].dt.year.astype("int16")
            chunk["bucket"] = _bucket_for_symbol_keys(chunk["symbol_key"], num_buckets)

            keep = [
                "table",
                "ticker",
                "symbol_key",
                "date",
                "open_raw",
                "high_raw",
                "low_raw",
                "close_raw",
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
                "close_unadj",
                "volume",
                "lastupdated",
                "year",
                "bucket",
            ]
            _parquet_write_partitioned(
                chunk[keep], bars_dir, partition_cols=("year", "bucket"), part_counter=part_counter, compression="zstd"
            )

    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
        },
    )


def _ensure_daily_cache(paths: SharadarPaths, daily_dir: Path, *, num_buckets: int, rebuild: bool) -> None:
    manifest_path = daily_dir / "_manifest.json"
    config_key = {"schema": "daily", "num_buckets": int(num_buckets)}
    manifest = _read_manifest(manifest_path)
    prev_inputs = (manifest or {}).get("inputs", {}) if manifest else {}
    inputs = {"daily_csv": _fingerprint_file_cached(paths.daily_csv, prev_inputs.get("daily_csv"))}
    if (
        not rebuild
        and daily_dir.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building daily cache: %s", daily_dir)
    _ensure_empty_dir(daily_dir)

    header = _read_csv_header(paths.daily_csv)
    lower = [c.strip().lower() for c in header]
    has_table = "table" in lower
    has_lastupdated = "lastupdated" in lower

    required = ["date", "ticker", "marketcap"]
    if has_table:
        required.append("table")
    if has_lastupdated:
        required.append("lastupdated")
    cols = _resolve_columns(header, required=required)
    part_counter: dict[tuple[Any, ...], int] = {}
    for chunk in tqdm(
        pd.read_csv(
            paths.daily_csv,
            usecols=list(cols.values()),
            parse_dates=[cols["date"]] + ([cols["lastupdated"]] if has_lastupdated else []),
            chunksize=1_000_000,
            dtype={cols["ticker"]: "string", **({cols["table"]: "string"} if has_table else {})},
        ),
        desc="daily",
    ):
        chunk = chunk.rename(columns={v: k for k, v in cols.items()})
        if not has_table:
            chunk["table"] = "SEP"
        else:
            chunk["table"] = chunk["table"].astype("string").str.upper()
        chunk["ticker"] = chunk["ticker"].str.upper()
        chunk["symbol_key"] = chunk["table"] + ":" + chunk["ticker"]
        if "lastupdated" not in chunk.columns:
            chunk["lastupdated"] = pd.NaT
        chunk = _dedupe_latest_lastupdated(chunk, keys=["symbol_key", "date"], lastupdated_col="lastupdated")
        chunk["marketcap"] = pd.to_numeric(chunk["marketcap"], errors="coerce").astype("float64")
        chunk = chunk[np.isfinite(chunk["marketcap"]) & (chunk["marketcap"] > 0)].copy()
        if chunk.empty:
            continue
        chunk["year"] = chunk["date"].dt.year.astype("int16")
        chunk["bucket"] = _bucket_for_symbol_keys(chunk["symbol_key"], num_buckets)
        keep = ["table", "ticker", "symbol_key", "date", "marketcap", "lastupdated", "year", "bucket"]
        _parquet_write_partitioned(
            chunk[keep], daily_dir, partition_cols=("year", "bucket"), part_counter=part_counter, compression="zstd"
        )

    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
        },
    )


def _symbol_key_to_permaticker_map(tickers_dir: Path) -> dict[str, int]:
    tickers = pq.read_table(tickers_dir / "tickers.parquet", columns=["symbol_key", "permaticker", "lastupdated"]).to_pandas()
    tickers["lastupdated"] = pd.to_datetime(tickers["lastupdated"], errors="coerce")
    tickers = tickers.sort_values(["symbol_key", "lastupdated"], kind="mergesort")
    tickers = tickers.drop_duplicates(["symbol_key"], keep="last")
    tickers = tickers[pd.notna(tickers["permaticker"])].copy()
    tickers["permaticker"] = tickers["permaticker"].astype("Int64")
    return {str(k): int(v) for k, v in zip(tickers["symbol_key"], tickers["permaticker"], strict=True) if pd.notna(v)}


def _build_symbol_key_interval_map(tickers_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return mapping: symbol_key -> (starts, ends, permatickers).

    - Starts/ends are datetime64[ns] arrays.
    - Permatickers are int64 arrays.
    - Rows are ordered by increasing lastupdated so later rows overwrite earlier rows if ranges overlap.
    """
    tickers = pq.read_table(
        tickers_dir / "tickers.parquet",
        columns=["symbol_key", "permaticker", "firstpricedate", "lastpricedate", "lastupdated"],
    ).to_pandas()
    if tickers.empty:
        return {}
    tickers["firstpricedate"] = pd.to_datetime(tickers["firstpricedate"], errors="coerce")
    tickers["lastpricedate"] = pd.to_datetime(tickers["lastpricedate"], errors="coerce")
    tickers["lastupdated"] = pd.to_datetime(tickers["lastupdated"], errors="coerce")
    tickers = tickers[pd.notna(tickers["permaticker"])].copy()
    tickers["permaticker"] = tickers["permaticker"].astype("Int64")
    tickers = tickers[pd.notna(tickers["permaticker"])].copy()

    min_date = np.datetime64("1900-01-01")
    max_date = np.datetime64("2262-04-11")
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, g in tickers.groupby("symbol_key", sort=False):
        g = g.copy()
        g["firstpricedate"] = g["firstpricedate"].fillna(pd.Timestamp(min_date))
        g["lastpricedate"] = g["lastpricedate"].fillna(pd.Timestamp(max_date))
        g["lastupdated"] = g["lastupdated"].fillna(pd.Timestamp(min_date))
        g = g.sort_values(["lastupdated", "permaticker"], kind="mergesort")
        starts = g["firstpricedate"].to_numpy(dtype="datetime64[ns]")
        ends = g["lastpricedate"].to_numpy(dtype="datetime64[ns]")
        perms = g["permaticker"].astype("int64").to_numpy(dtype="int64")
        out[str(sym)] = (starts, ends, perms)
    return out


def _map_symbol_keys_dates_to_permaticker(
    symbol_keys: pd.Series,
    dates: pd.Series,
    interval_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Map (symbol_key, date) to permaticker using tickers.csv date ranges.

    If multiple permaticker rows match a date (overlap), the row with the latest lastupdated wins because the interval
    lists are ordered by increasing lastupdated and later matches overwrite earlier ones.
    """
    n = int(len(symbol_keys))
    if n == 0:
        return np.full(0, np.nan, dtype="float64")
    sym_vals = symbol_keys.astype("string").fillna("").to_numpy()
    date_vals = pd.to_datetime(dates, errors="coerce").to_numpy(dtype="datetime64[ns]")
    out = np.full(n, np.nan, dtype="float64")

    codes, uniques = pd.factorize(sym_vals, sort=False)
    for code, sym in enumerate(uniques):
        sym_s = str(sym)
        if not sym_s:
            continue
        intervals = interval_map.get(sym_s)
        if intervals is None:
            continue
        idxs = np.where(codes == code)[0]
        if len(idxs) == 0:
            continue
        d = date_vals[idxs]
        starts, ends, perms = intervals
        perm_out = np.full(len(idxs), np.nan, dtype="float64")
        for s, e, p in zip(starts, ends, perms, strict=True):
            mask = (d >= s) & (d <= e)
            if np.any(mask):
                perm_out[mask] = float(p)
        out[idxs] = perm_out
    return out


def _ensure_bars_permaticker_cache(
    bars_dir: Path, tickers_dir: Path, out_dir: Path, cfg: EntryRotationConfig, *, rebuild: bool
) -> None:
    manifest_path = out_dir / "_manifest.json"
    bars_manifest = _read_manifest(bars_dir / "_manifest.json") or {}
    tickers_manifest = _read_manifest(tickers_dir / "_manifest.json") or {}
    inputs = {"bars_manifest": bars_manifest, "tickers_manifest": tickers_manifest}
    config_key = {"schema": "bars_permaticker", "num_buckets": int(cfg.num_buckets), "year_bin_years": 5}
    manifest = _read_manifest(manifest_path)
    if (
        not rebuild
        and out_dir.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building bars_permaticker cache: %s", out_dir)
    _ensure_empty_dir(out_dir)
    interval_map = _build_symbol_key_interval_map(tickers_dir)

    files = sorted((bars_dir).glob("year=*/bucket=*/*.parquet"))
    part_counter: dict[tuple[Any, ...], int] = {}
    missing = 0
    written = 0
    for path in tqdm(files, desc="bars_permaticker"):
        try:
            df = pq.read_table(
                str(path),
                columns=[
                    "symbol_key",
                    "date",
                    "open_raw",
                    "high_raw",
                    "low_raw",
                    "close_raw",
                    "open_adj",
                    "high_adj",
                    "low_adj",
                    "close_adj",
                    "close_unadj",
                    "volume",
                    "lastupdated",
                ],
            ).to_pandas()
        except Exception:
            # Backwards-compatible read for older caches that don't have raw OHLC.
            df = pq.read_table(
                str(path),
                columns=[
                    "symbol_key",
                    "date",
                    "open_adj",
                    "high_adj",
                    "low_adj",
                    "close_adj",
                    "close_unadj",
                    "volume",
                    "lastupdated",
                ],
            ).to_pandas()
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["permaticker"] = _map_symbol_keys_dates_to_permaticker(df["symbol_key"], df["date"], interval_map)
        miss = int(np.sum(~np.isfinite(df["permaticker"].to_numpy(dtype="float64"))))
        if miss:
            missing += miss
        df = df[np.isfinite(df["permaticker"])].copy()
        if df.empty:
            continue
        df["permaticker"] = df["permaticker"].astype("int64")
        df["perm_bucket"] = (df["permaticker"] % cfg.num_buckets).astype("int16")
        df["year_bin"] = _year_bin_5y(df["date"]).astype("int16")
        keep = [
            "permaticker",
            "date",
            *([c for c in ["open_raw", "high_raw", "low_raw", "close_raw"] if c in df.columns]),
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "close_unadj",
            "volume",
            "lastupdated",
            "perm_bucket",
            "year_bin",
        ]
        _parquet_write_partitioned(
            df[keep], out_dir, partition_cols=("perm_bucket", "year_bin"), part_counter=part_counter, compression="zstd"
        )
        written += len(df)

    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
            "rows_written": int(written),
            "rows_missing_permaticker_dropped": int(missing),
        },
    )


def _ensure_daily_permaticker_cache(
    daily_dir: Path, tickers_dir: Path, out_dir: Path, cfg: EntryRotationConfig, *, rebuild: bool
) -> None:
    manifest_path = out_dir / "_manifest.json"
    daily_manifest = _read_manifest(daily_dir / "_manifest.json") or {}
    tickers_manifest = _read_manifest(tickers_dir / "_manifest.json") or {}
    inputs = {"daily_manifest": daily_manifest, "tickers_manifest": tickers_manifest}
    config_key = {"schema": "daily_permaticker", "num_buckets": int(cfg.num_buckets), "year_bin_years": 5}
    manifest = _read_manifest(manifest_path)
    if (
        not rebuild
        and out_dir.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building daily_permaticker cache: %s", out_dir)
    _ensure_empty_dir(out_dir)
    interval_map = _build_symbol_key_interval_map(tickers_dir)

    files = sorted((daily_dir).glob("year=*/bucket=*/*.parquet"))
    part_counter: dict[tuple[Any, ...], int] = {}
    missing = 0
    written = 0
    for path in tqdm(files, desc="daily_permaticker"):
        df = pq.read_table(str(path), columns=["symbol_key", "date", "marketcap", "lastupdated"]).to_pandas()
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["permaticker"] = _map_symbol_keys_dates_to_permaticker(df["symbol_key"], df["date"], interval_map)
        miss = int(np.sum(~np.isfinite(df["permaticker"].to_numpy(dtype="float64"))))
        if miss:
            missing += miss
        df = df[np.isfinite(df["permaticker"])].copy()
        if df.empty:
            continue
        df["permaticker"] = df["permaticker"].astype("int64")
        df["perm_bucket"] = (df["permaticker"] % cfg.num_buckets).astype("int16")
        df["year_bin"] = _year_bin_5y(df["date"]).astype("int16")
        keep = ["permaticker", "date", "marketcap", "lastupdated", "perm_bucket", "year_bin"]
        _parquet_write_partitioned(
            df[keep], out_dir, partition_cols=("perm_bucket", "year_bin"), part_counter=part_counter, compression="zstd"
        )
        written += len(df)

    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
            "rows_written": int(written),
            "rows_missing_permaticker_dropped": int(missing),
        },
    )


def _resolve_benchmark_symbol_key(tickers_dir: Path, ticker: str, *, prefer_table: str = "SFP") -> str:
    tickers = pq.read_table(tickers_dir / "tickers.parquet", columns=["table", "ticker", "symbol_key"]).to_pandas()
    ticker = ticker.upper()
    matches = tickers[tickers["ticker"] == ticker]
    if matches.empty:
        raise ValueError(f"Benchmark ticker not found in tickers cache: {ticker}")
    preferred = matches[matches["table"] == prefer_table]
    if not preferred.empty:
        return str(preferred.iloc[0]["symbol_key"])
    return str(matches.iloc[0]["symbol_key"])


def _load_bars_for_symbol_key(bars_dir: Path, symbol_key: str, *, num_buckets: int) -> pd.DataFrame:
    bucket = _bucket_for_symbol_key(symbol_key, num_buckets)
    files = sorted(bars_dir.glob(f"year=*/bucket={bucket}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No bars parquet files found for bucket={bucket} under {bars_dir}")
    cols = [
        "symbol_key",
        "date",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "close_unadj",
        "volume",
        "table",
        "ticker",
        "lastupdated",
    ]
    # Best-effort include raw OHLC when present.
    try:
        table = pq.read_table(
            [str(p) for p in files],
            columns=["open_raw", "high_raw", "low_raw", "close_raw", *cols],
            filters=[("symbol_key", "==", symbol_key)],
        )
    except Exception:
        table = pq.read_table([str(p) for p in files], columns=cols, filters=[("symbol_key", "==", symbol_key)])
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = _dedupe_latest_lastupdated(df, keys=["symbol_key", "date"], lastupdated_col="lastupdated")
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def _ensure_month_index_cache(
    bars_dir: Path,
    tickers_dir: Path,
    month_index_dir: Path,
    cfg: EntryRotationConfig,
    *,
    rebuild: bool,
) -> None:
    manifest_path = month_index_dir / "_manifest.json"
    month_index_parquet = month_index_dir / "month_index.parquet"
    bars_manifest = _read_manifest(bars_dir / "_manifest.json") or {}
    tickers_manifest = _read_manifest(tickers_dir / "_manifest.json") or {}
    inputs = {"bars_manifest": bars_manifest, "tickers_manifest": tickers_manifest}
    config_key = {
        "schema": "month_index",
        "selection_benchmark_ticker": cfg.selection_benchmark_ticker,
        "num_buckets": cfg.num_buckets,
    }
    manifest = _read_manifest(manifest_path)
    if (
        not rebuild
        and month_index_parquet.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building month index cache: %s", month_index_dir)
    _ensure_empty_dir(month_index_dir)

    symbol_key = _resolve_benchmark_symbol_key(tickers_dir, cfg.selection_benchmark_ticker, prefer_table="SFP")
    bench = _load_bars_for_symbol_key(bars_dir, symbol_key, num_buckets=cfg.num_buckets)
    if bench.empty:
        raise ValueError(f"No bars found for selection benchmark {cfg.selection_benchmark_ticker} ({symbol_key})")

    bench["year_month"] = bench["date"].dt.to_period("M").astype(str)
    select = bench.groupby("year_month", sort=True)["date"].min().reset_index()
    select = select.sort_values("date", kind="mergesort").reset_index(drop=True)
    select["month_idx"] = np.arange(len(select), dtype=np.int32)
    select["select_date"] = select["date"]
    select = select[["year_month", "month_idx", "select_date"]]

    pq.write_table(pa.Table.from_pandas(select, preserve_index=False), month_index_parquet, compression="zstd")
    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
            "rows": int(len(select)),
            "symbol_key": symbol_key,
        },
    )


def _default_sfp_blacklist() -> set[str]:
    # A conservative starter set; configurable via --no-sfp-blacklist and --sfp-blacklist-extra.
    return {
        "BIL",
        "SGOV",
        "SHV",
        "SHY",
        "TLT",
        "TQQQ",
        "SQQQ",
        "UPRO",
        "SPXU",
        "SSO",
        "SDS",
        "QLD",
        "QID",
        "SPXL",
        "SPXS",
        "SOXL",
        "SOXS",
        "LABU",
        "LABD",
        "TNA",
        "TZA",
    }


def _compute_year_green_flags(dates: np.ndarray, open_adj: np.ndarray, close_adj: np.ndarray) -> dict[int, bool] | None:
    if len(dates) == 0:
        return None
    ts = pd.to_datetime(dates)
    years = ts.year.to_numpy()
    months = ts.month.to_numpy()
    out: dict[int, bool] = {}
    for y in np.unique(years):
        idx_y = np.where(years == y)[0]
        months_y = months[idx_y]
        dates_y = ts[idx_y]
        open_y = open_adj[idx_y]
        close_y = close_adj[idx_y]

        jan_idx = idx_y[months_y == 1]
        if len(jan_idx) == 0:
            out[y] = False
            continue
        entry_pos = jan_idx[np.argmin(dates_y[months_y == 1].to_numpy())]
        entry_px = open_adj[entry_pos]
        if not np.isfinite(entry_px) or entry_px <= 0:
            out[y] = False
            continue

        dec_idx = idx_y[months_y == 12]
        if len(dec_idx) > 0:
            exit_pos = dec_idx[np.argmax(dates_y[months_y == 12].to_numpy())]
            exit_px = close_adj[exit_pos]
        else:
            # Fallback: last available bar on/before Dec 31 of that year == last bar in the year.
            exit_pos = idx_y[np.argmax(dates_y.to_numpy())]
            exit_px = close_adj[exit_pos]
        if not np.isfinite(exit_px) or exit_px <= 0:
            out[y] = False
            continue

        out[y] = (exit_px / entry_px) - 1.0 > 0.0
    return out


def _rolling_forward_min(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values, dtype="float64")
    return s.rolling(window, min_periods=window).min().shift(-(window - 1)).to_numpy(dtype="float64")


def _compute_cagr_years(
    dates: np.ndarray,
    close_adj: np.ndarray,
    select_dates: np.ndarray,
    *,
    years: int,
    min_history_bars: int,
) -> np.ndarray:
    n = len(select_dates)
    if len(dates) == 0 or n == 0:
        return np.full(n, np.nan, dtype="float64")

    ts = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    closes = close_adj.astype("float64", copy=False)

    # Strictly before select_date (avoid using the entry bar's close).
    end_ix = np.searchsorted(ts, select_dates, side="left") - 1
    out = np.full(n, np.nan, dtype="float64")
    valid_end = end_ix >= 0
    if not np.any(valid_end):
        return out

    end_ix_v = end_ix[valid_end]
    end_dates = ts[end_ix_v]
    end_px = closes[end_ix_v]
    valid_px = np.isfinite(end_px) & (end_px > 0)
    if not np.any(valid_px):
        return out

    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[np.where(valid_end)[0][valid_px]] = True

    end_dates_v = pd.to_datetime(end_dates[valid_px])
    start_targets = (end_dates_v - pd.DateOffset(years=int(years))).to_numpy(dtype="datetime64[ns]")
    # Strict/no-lookahead anchor: use last bar on/before the target 5y-ago date (never look forward).
    start_ix_v = np.searchsorted(ts, start_targets, side="right") - 1
    start_ix = np.full(n, -1, dtype=int)
    start_ix[valid_mask] = start_ix_v

    end_ix_full = end_ix.copy()
    start_ix_full = start_ix
    ok_idx = (start_ix_full >= 0) & (start_ix_full <= end_ix_full)
    ok_idx &= (end_ix_full - start_ix_full + 1) >= min_history_bars
    if not np.any(ok_idx):
        return out

    start_px = closes[start_ix_full[ok_idx]]
    end_px_ok = closes[end_ix_full[ok_idx]]
    ok_px = np.isfinite(start_px) & (start_px > 0) & np.isfinite(end_px_ok) & (end_px_ok > 0)
    if not np.any(ok_px):
        return out

    idxs = np.where(ok_idx)[0][ok_px]
    start_dates = ts[start_ix_full[idxs]]
    end_dates2 = ts[end_ix_full[idxs]]
    years = (end_dates2 - start_dates).astype("timedelta64[D]").astype("int64") / 365.25
    ok_years = years > 0
    if not np.any(ok_years):
        return out

    idxs = idxs[ok_years]
    years = years[ok_years]
    start_px2 = closes[start_ix_full[idxs]]
    end_px2 = closes[end_ix_full[idxs]]
    out[idxs] = np.power(end_px2 / start_px2, 1.0 / years) - 1.0
    return out


def _annual_sim_end_date(
    *,
    bench_dates: pd.DatetimeIndex,
    rebalance_dates: dict[int, pd.Timestamp],
    end_year: int,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """
    Return (end_date_inclusive, exit_rebalance_date).

    If rebalance_date(end_year+1) exists in the benchmark calendar, treat that as the natural exit for the last rolling-year.
    Otherwise fall back to last benchmark day on/before Dec 31 of end_year.
    """
    data_end = pd.Timestamp(bench_dates.max())
    exit_rebalance = rebalance_dates.get(end_year + 1)
    if exit_rebalance is not None and exit_rebalance <= data_end:
        return pd.Timestamp(exit_rebalance), pd.Timestamp(exit_rebalance)
    cutoff = pd.Timestamp(year=end_year, month=12, day=31)
    end_date = bench_dates[bench_dates <= cutoff].max()
    if pd.isna(end_date):
        end_date = bench_dates.max()
    return pd.Timestamp(end_date), None


def _max_drawdown_pct_window(close_prices: np.ndarray) -> float:
    """
    Compute max drawdown for a single window of close prices.

    Returns a negative value (e.g. -0.25 for -25%).
    """
    close_prices = np.asarray(close_prices, dtype="float64")
    if close_prices.size == 0 or not np.all(np.isfinite(close_prices)) or np.any(close_prices <= 0):
        return float("nan")
    peak = np.maximum.accumulate(close_prices)
    dd = (close_prices / peak) - 1.0
    return float(np.nanmin(dd))


def _monthly_sim_end_date(
    *,
    month_index: pd.DataFrame,
    start_year: int,
    end_year: int,
    hold_period_months: int,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Return (simulation_month_index, start_select_date, end_select_date).

    Contributions/buys occur only in months with select_date.year in [start_year..end_year].
    Simulation extends through the maturity month for the last contribution month (end_contrib_month_idx + hold_period_months),
    if present in month_index; otherwise it ends at the last available select_date.
    """
    mi = month_index.copy()
    mi["select_date"] = pd.to_datetime(mi["select_date"])
    contrib = mi[(mi["select_date"].dt.year >= start_year) & (mi["select_date"].dt.year <= end_year)].copy()
    if contrib.empty:
        raise ValueError("No months in requested year range")
    contrib = contrib.sort_values("select_date", kind="mergesort").reset_index(drop=True)
    start_month_idx = int(contrib["month_idx"].iloc[0])
    end_contrib_month_idx = int(contrib["month_idx"].iloc[-1])

    end_sim_month_idx = min(int(mi["month_idx"].max()), end_contrib_month_idx + hold_period_months)
    sim = mi[(mi["month_idx"] >= start_month_idx) & (mi["month_idx"] <= end_sim_month_idx)].copy()
    sim = sim.sort_values("select_date", kind="mergesort").reset_index(drop=True)
    return sim, pd.Timestamp(contrib["select_date"].iloc[0]), pd.Timestamp(sim["select_date"].iloc[-1])


def _annual_contribution_dates(
    month_index: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """
    Return monthly contribution dates for annual-lumpsum.

    Convention:
    - Contributions land on the benchmark month schedule (`month_index.select_date`).
    - Exclude the start rebalance date (start_date), but include the end boundary (end_date) if it is a select_date.
    - This yields exactly 12 contributions in a full rebalance->rebalance cycle when end_date is the next rebalance date.
    """
    if month_index.empty:
        return pd.DatetimeIndex([])
    mi = month_index[["select_date"]].copy()
    mi["select_date"] = pd.to_datetime(mi["select_date"])
    dates = pd.DatetimeIndex(mi["select_date"]).sort_values()
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    return dates[(dates > start_date) & (dates <= end_date)]


def _ensure_candidates_cache(
    bars_dir: Path,
    daily_dir: Path,
    tickers_dir: Path,
    month_index_dir: Path,
    candidates_dir: Path,
    cfg: EntryRotationConfig,
    *,
    rebuild: bool,
) -> None:
    manifest_path = candidates_dir / "_manifest.json"
    bars_manifest = _read_manifest(bars_dir / "_manifest.json") or {}
    daily_manifest = _read_manifest(daily_dir / "_manifest.json") or {}
    tickers_manifest = _read_manifest(tickers_dir / "_manifest.json") or {}
    month_index_manifest = _read_manifest(month_index_dir / "_manifest.json") or {}
    inputs = {
        "bars_manifest": bars_manifest,
        "daily_manifest": daily_manifest,
        "tickers_manifest": tickers_manifest,
        "month_index_manifest": month_index_manifest,
    }
    config_key = {
        "schema": "candidates_monthly_v12_permaticker_add_trailing_low_rank",
        "layout": "year_month",
        "segmentation": "calendar_month",
        "num_buckets": int(cfg.num_buckets),
        "min_history_bars": int(cfg.min_history_bars),
        "dollar_volume_lookback_bars": int(cfg.dollar_volume_lookback_bars),
        "hold_period_months": int(cfg.hold_period_months),
        "require_last_n_years_green": int(cfg.require_last_n_years_green),
        "selection_benchmark_ticker": cfg.selection_benchmark_ticker,
    }
    manifest = _read_manifest(manifest_path)
    if (
        not rebuild
        and candidates_dir.exists()
        and manifest
        and _manifest_matches(manifest, cache_version=CACHE_VERSION, inputs=inputs, config_key=config_key)
    ):
        return

    logger.info("Building candidates cache: %s", candidates_dir)
    _ensure_empty_dir(candidates_dir)

    tickers_df = pq.read_table(tickers_dir / "tickers.parquet").to_pandas()
    tickers_df["permaticker"] = tickers_df["permaticker"].astype("Int64")
    tickers_df = tickers_df[pd.notna(tickers_df["permaticker"])].copy()
    tickers_df["firstpricedate"] = pd.to_datetime(tickers_df["firstpricedate"], errors="coerce")
    tickers_df["lastpricedate"] = pd.to_datetime(tickers_df["lastpricedate"], errors="coerce")
    tickers_df["lastupdated"] = pd.to_datetime(tickers_df["lastupdated"], errors="coerce")
    if "category" not in tickers_df.columns:
        tickers_df["category"] = ""
    if "name" not in tickers_df.columns:
        tickers_df["name"] = ""

    month_index = pq.read_table(month_index_dir / "month_index.parquet").to_pandas()
    month_index["select_date"] = pd.to_datetime(month_index["select_date"])
    select_dates = month_index["select_date"].to_numpy(dtype="datetime64[ns]")
    year_months = month_index["year_month"].astype("string").to_numpy()
    month_idxs = month_index["month_idx"].to_numpy(dtype="int32")
    bench_months = month_index["select_date"].to_numpy(dtype="datetime64[M]")
    n_months = len(month_index)

    sfp_blacklist = set()
    if not cfg.no_sfp_blacklist:
        sfp_blacklist |= _default_sfp_blacklist()
    sfp_blacklist |= {t.upper() for t in cfg.sfp_blacklist_extra}

    perm_buckets = [int(p.name.split("=", 1)[1]) for p in sorted(bars_dir.glob("perm_bucket=*")) if p.is_dir()]

    def flush_year_month(frames: list[pd.DataFrame], *, perm_bucket: int, counter: dict[str, int]) -> None:
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        for year_month, sub in df.groupby("year_month", sort=True):
            ym = str(year_month)
            out_dir = candidates_dir / f"year_month={ym}"
            out_dir.mkdir(parents=True, exist_ok=True)
            part = counter.get(ym, 0)
            counter[ym] = part + 1
            file_path = out_dir / f"bucket{perm_bucket:02d}-part-{part:06d}.parquet"
            # Avoid duplicating the hive partition column inside the file.
            sub = sub.drop(columns=["year_month"], errors="ignore")
            pq.write_table(pa.Table.from_pandas(sub.reset_index(drop=True), preserve_index=False), file_path, compression="zstd")

    def labels_asof_months(perm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g = tickers_df[tickers_df["permaticker"] == perm].copy()
        # Extra safety: ignore non-price tables even if an older tickers cache is present.
        if not g.empty:
            g = g[g["table"].isin(["SEP", "SFP"])].copy()
        if g.empty:
            n = n_months
            return (
                np.array([""] * n, dtype=object),
                np.array([""] * n, dtype=object),
                np.array([""] * n, dtype=object),
                np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
                np.array([""] * n, dtype=object),
                np.array([""] * n, dtype=object),
            )
        g = g.sort_values(["firstpricedate", "lastpricedate", "lastupdated", "table", "ticker"], kind="mergesort")
        n = n_months
        out_symbol = np.array([""] * n, dtype=object)
        out_table = np.array([""] * n, dtype=object)
        out_ticker = np.array([""] * n, dtype=object)
        out_last = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        out_upd = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        out_cat = np.array([""] * n, dtype=object)
        out_name = np.array([""] * n, dtype=object)

        for _, row in g.iterrows():
            fp = row["firstpricedate"]
            if pd.isna(fp):
                continue
            lp = row["lastpricedate"]
            lu = row["lastupdated"]
            lu64 = lu.to_datetime64() if pd.notna(lu) else np.datetime64("NaT")
            end = lp.to_datetime64() if pd.notna(lp) else np.datetime64("2262-04-11")
            in_range = (select_dates >= fp.to_datetime64()) & (select_dates <= end)
            if not np.any(in_range):
                continue
            better = np.isnat(out_upd) | ((~np.isnat(lu64)) & (lu64 > out_upd))
            take = in_range & better
            if not np.any(take):
                continue
            out_symbol[take] = str(row["symbol_key"])
            out_table[take] = str(row["table"])
            out_ticker[take] = str(row["ticker"])
            out_last[take] = lp.to_datetime64() if pd.notna(lp) else np.datetime64("NaT")
            out_upd[take] = lu64
            cat = row.get("category", "")
            if pd.isna(cat):
                cat = ""
            out_cat[take] = str(cat)
            name = row.get("name", "")
            if pd.isna(name):
                name = ""
            out_name[take] = str(name)
        return out_symbol, out_table, out_ticker, out_last, out_cat, out_name

    def build_one_perm_bucket(perm_bucket: int) -> int:
        bars_files = sorted((bars_dir).glob(f"perm_bucket={perm_bucket}/year_bin=*/*.parquet"))
        if not bars_files:
            return 0
        try:
            bars = pq.read_table(
                [str(p) for p in bars_files],
                columns=[
                    "permaticker",
                    "date",
                    "open_raw",
                    "low_raw",
                    "close_raw",
                    "open_adj",
                    "low_adj",
                    "close_adj",
                    "close_unadj",
                    "volume",
                    "lastupdated",
                ],
            ).to_pandas()
        except Exception:
            # Backwards-compatible read for older caches without raw OHLC.
            bars = pq.read_table(
                [str(p) for p in bars_files],
                columns=["permaticker", "date", "open_adj", "low_adj", "close_adj", "close_unadj", "volume", "lastupdated"],
            ).to_pandas()
        if bars.empty:
            return 0
        bars["date"] = pd.to_datetime(bars["date"])
        bars = _dedupe_latest_lastupdated(bars, keys=["permaticker", "date"], lastupdated_col="lastupdated")
        bars = bars.sort_values(["permaticker", "date"], kind="mergesort").reset_index(drop=True)

        daily_files = sorted((daily_dir).glob(f"perm_bucket={perm_bucket}/year_bin=*/*.parquet"))
        daily = (
            pq.read_table([str(p) for p in daily_files], columns=["permaticker", "date", "marketcap", "lastupdated"]).to_pandas()
            if daily_files
            else pd.DataFrame()
        )
        daily_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if not daily.empty:
            daily["date"] = pd.to_datetime(daily["date"])
            daily = _dedupe_latest_lastupdated(daily, keys=["permaticker", "date"], lastupdated_col="lastupdated")
            daily = daily.sort_values(["permaticker", "date"], kind="mergesort").reset_index(drop=True)
            for perm, g in daily.groupby("permaticker", sort=True):
                daily_map[int(perm)] = (g["date"].to_numpy(dtype="datetime64[ns]"), g["marketcap"].to_numpy(dtype="float64"))

        out_rows: list[pd.DataFrame] = []
        counter: dict[str, int] = {}
        processed = 0
        for perm, df_sym in tqdm(bars.groupby("permaticker", sort=True), desc=f"candidates:perm_bucket={perm_bucket}"):
            perm_i = int(perm)
            processed += 1

            instrument_id = f"PT{perm_i}"
            symbol_key_asof, table_asof, ticker_asof, lastpricedate_asof, category_asof, name_asof = labels_asof_months(perm_i)
            is_sfp_blacklisted = (table_asof == "SFP") & np.isin(ticker_asof, list(sfp_blacklist))
            is_equity_asof = _equity_mask_from_categories(category_asof)

            dates = df_sym["date"].to_numpy(dtype="datetime64[ns]")
            open_adj = df_sym["open_adj"].to_numpy(dtype="float64")
            low_adj = df_sym["low_adj"].to_numpy(dtype="float64")
            close_adj = df_sym["close_adj"].to_numpy(dtype="float64")
            open_raw = (
                df_sym["open_raw"].to_numpy(dtype="float64")
                if "open_raw" in df_sym.columns
                else np.full(len(df_sym), np.nan, dtype="float64")
            )
            low_raw = (
                df_sym["low_raw"].to_numpy(dtype="float64")
                if "low_raw" in df_sym.columns
                else np.full(len(df_sym), np.nan, dtype="float64")
            )
            close_raw = (
                df_sym["close_raw"].to_numpy(dtype="float64")
                if "close_raw" in df_sym.columns
                else np.full(len(df_sym), np.nan, dtype="float64")
            )
            close_unadj = df_sym["close_unadj"].to_numpy(dtype="float64")
            volume_unadj = df_sym["volume"].to_numpy(dtype="float64")

            # Calendar-month segmentation: each bar belongs to its actual calendar month, using the benchmark month_index set.
            bar_months = dates.astype("datetime64[M]")
            month_ix = np.searchsorted(bench_months, bar_months, side="left")
            in_range = (month_ix >= 0) & (month_ix < n_months)
            valid_month = in_range.copy()
            if np.any(in_range):
                inr = np.where(in_range)[0]
                valid_month[inr] &= bench_months[month_ix[inr]] == bar_months[inr]
            if not np.any(valid_month):
                continue
            month_ix_v = month_ix[valid_month]
            dates_v = dates[valid_month]
            starts = np.r_[0, np.flatnonzero(np.diff(month_ix_v)) + 1]
            ends = np.r_[starts[1:], len(month_ix_v)]

            def compute_segments(
                *, open_px: np.ndarray, low_px: np.ndarray, close_px: np.ndarray
            ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                open_v = open_px[valid_month]
                low_v = low_px[valid_month]
                close_v = close_px[valid_month]
                low_used_v = np.where(np.isfinite(low_v) & (low_v > 0), low_v, close_v)

                seg_min = np.full(n_months, np.nan, dtype="float64")
                seg_last_close = np.full(n_months, np.nan, dtype="float64")
                seg_first_date = np.full(n_months, np.datetime64("NaT"), dtype="datetime64[ns]")
                seg_first_open = np.full(n_months, np.nan, dtype="float64")
                seg_first_close = np.full(n_months, np.nan, dtype="float64")
                for s, e in zip(starts, ends, strict=True):
                    mi = int(month_ix_v[s])
                    seg_first_date[mi] = dates_v[s]
                    seg_first_open[mi] = float(open_v[s])
                    seg_first_close[mi] = float(close_v[s])
                    with np.errstate(all="ignore"):
                        seg_min[mi] = float(np.nanmin(low_used_v[s:e]))
                    seg_last_close[mi] = float(close_v[e - 1])
                return pd.to_datetime(seg_first_date), seg_first_open, seg_first_close, seg_min, seg_last_close

            entry_date_adj, entry_open_adj, entry_close_adj, seg_min_adj, seg_last_close_adj = compute_segments(
                open_px=open_adj, low_px=low_adj, close_px=close_adj
            )
            entry_date_raw, entry_open_raw_m, entry_close_raw_m, seg_min_raw, seg_last_close_raw = compute_segments(
                open_px=open_raw, low_px=low_raw, close_px=close_raw
            )

            # Keep the original variable names for the adjusted mode (this is what the legacy caches stored).
            entry_date = entry_date_adj
            entry_open = entry_open_adj
            entry_close = entry_close_adj
            seg_min = seg_min_adj
            seg_last_close = seg_last_close_adj

            # Previous rolling 12-month return (open-to-open): entry_open[t] / entry_open[t-12] - 1.
            prev_12m_return = np.full(n_months, np.nan, dtype="float64")
            prev_open = np.full(n_months, np.nan, dtype="float64")
            if n_months > cfg.hold_period_months:
                prev_open[cfg.hold_period_months :] = entry_open[: n_months - cfg.hold_period_months]
                ok_prev = np.isfinite(entry_open) & np.isfinite(prev_open) & (entry_open > 0) & (prev_open > 0)
                if np.any(ok_prev):
                    np.divide(entry_open, prev_open, out=prev_12m_return, where=ok_prev)
                    prev_12m_return = prev_12m_return - 1.0

            window_min_12m = _rolling_forward_min(seg_min, window=cfg.hold_period_months)
            mae_ratio_12m = np.full(n_months, np.nan, dtype="float64")
            mae_ok = np.isfinite(window_min_12m) & np.isfinite(entry_open) & (entry_open > 0)
            if np.any(mae_ok):
                np.divide(window_min_12m, entry_open, out=mae_ratio_12m, where=mae_ok)
            mae_12m = mae_ratio_12m - 1.0

            # "5y MAE" convention: for entry month t, look at the MAE of the five prior rolling-year windows
            # starting at t-12, t-24, ..., t-60 (each MAE is entry_open -> worst monthly low over the next 12 months).
            # This is a strict, completed-window measure at time t: all these windows end on/before t.
            offsets = [12, 24, 36, 48, 60]
            past_mae = np.full((len(offsets), n_months), np.nan, dtype="float64")
            for j, off in enumerate(offsets):
                if off < n_months:
                    past_mae[j, off:] = mae_12m[: n_months - off]
            n_past_5y_mae = np.sum(np.isfinite(past_mae), axis=0).astype("float64")
            worst_past_5y_mae_12m = np.full(n_months, np.nan, dtype="float64")
            median_past_5y_mae_12m = np.full(n_months, np.nan, dtype="float64")
            ok_past = n_past_5y_mae > 0
            if np.any(ok_past):
                worst_past_5y_mae_12m[ok_past] = np.nanmin(past_mae[:, ok_past], axis=0)
                with np.errstate(all="ignore"):
                    median_past_5y_mae_12m[ok_past] = np.nanmedian(past_mae[:, ok_past], axis=0)
            adverse_mag = -worst_past_5y_mae_12m
            adverse_median_mag = -median_past_5y_mae_12m

            # Keep the older "worst over past 60 months of completed 12m windows" for diagnostics (not used for selection).
            completed_mae = pd.Series(mae_12m, dtype="float64").shift(cfg.hold_period_months)
            worst_past_60m = completed_mae.rolling(60, min_periods=60).min().to_numpy(dtype="float64")
            n_past_mae = completed_mae.rolling(60, min_periods=60).count().to_numpy(dtype="float64")

            entry_dates64 = pd.to_datetime(entry_date).to_numpy(dtype="datetime64[ns]")
            cagr_3y = np.full(n_months, np.nan, dtype="float64")
            cagr_5y = np.full(n_months, np.nan, dtype="float64")
            valid_entry = ~np.isnat(entry_dates64)
            if np.any(valid_entry):
                cagr_3y[valid_entry] = _compute_cagr_years(
                    dates,
                    close_adj,
                    entry_dates64[valid_entry],
                    years=3,
                    min_history_bars=min(cfg.min_history_bars, 3 * 252),
                )
                cagr_5y[valid_entry] = _compute_cagr_years(
                    dates,
                    close_adj,
                    entry_dates64[valid_entry],
                    years=5,
                    min_history_bars=cfg.min_history_bars,
                )

            # 3y max drawdown (Calmar denominator): computed on monthly close_adj for the 36 months ending strictly before the entry month.
            max_dd_3y = np.full(n_months, np.nan, dtype="float64")
            calmar_3y = np.full(n_months, np.nan, dtype="float64")
            for t in range(36, n_months):
                window = seg_last_close[t - 36 : t]
                dd = _max_drawdown_pct_window(window)
                if np.isfinite(dd) and dd < 0 and np.isfinite(cagr_3y[t]):
                    max_dd_3y[t] = dd
                    calmar_3y[t] = float(cagr_3y[t]) / abs(float(dd))

            # 5y max drawdown (Calmar denominator): computed on monthly close_adj for the 60 months ending strictly before the entry month.
            # For entry month index t, use seg_last_close[t-60 : t] which ends at month t-1.
            max_dd_5y = np.full(n_months, np.nan, dtype="float64")
            calmar_5y = np.full(n_months, np.nan, dtype="float64")
            for t in range(60, n_months):
                window = seg_last_close[t - 60 : t]
                dd = _max_drawdown_pct_window(window)
                if np.isfinite(dd) and dd < 0 and np.isfinite(cagr_5y[t]):
                    max_dd_5y[t] = dd
                    calmar_5y[t] = float(cagr_5y[t]) / abs(float(dd))

            # TC2000-style rank (monthly, no-lookahead):
            # C = last close strictly before entry month (seg_last_close[t-1])
            # C60 = close 60 months before that (seg_last_close[t-61])
            # MINL60 = minimum low over prior 60 months ending at t-1 (min(seg_min[t-60:t]))
            # score = ((C/C60)**0.2 - 1) / abs(MINL60/C - 1)
            cagr_3y_asof_close = np.full(n_months, np.nan, dtype="float64")
            trailing_low_dd_mag_3y = np.full(n_months, np.nan, dtype="float64")
            cagr_to_trailing_low_3y = np.full(n_months, np.nan, dtype="float64")
            cagr_5y_asof_close = np.full(n_months, np.nan, dtype="float64")
            trailing_low_dd_mag_5y = np.full(n_months, np.nan, dtype="float64")
            cagr_to_trailing_low_5y = np.full(n_months, np.nan, dtype="float64")
            asof_close = np.full(n_months, np.nan, dtype="float64")
            if n_months > 1:
                asof_close[1:] = seg_last_close[:-1]
            asof_close_36 = np.full(n_months, np.nan, dtype="float64")
            if n_months > 36:
                asof_close_36[36:] = asof_close[: n_months - 36]
            ok_cagr_3 = np.isfinite(asof_close) & np.isfinite(asof_close_36) & (asof_close > 0) & (asof_close_36 > 0)
            if np.any(ok_cagr_3):
                with np.errstate(all="ignore"):
                    ratio = np.divide(asof_close, asof_close_36, out=np.full(n_months, np.nan, dtype="float64"), where=ok_cagr_3)
                    cagr_3y_asof_close[ok_cagr_3] = np.power(ratio[ok_cagr_3], 1.0 / 3.0) - 1.0
            trailing_min_36_end = pd.Series(seg_min, dtype="float64").rolling(36, min_periods=36).min().to_numpy(dtype="float64")
            min_low_36 = np.full(n_months, np.nan, dtype="float64")
            if n_months > 1:
                min_low_36[1:] = trailing_min_36_end[:-1]
            ok_dd_3 = np.isfinite(min_low_36) & np.isfinite(asof_close) & (asof_close > 0) & (min_low_36 > 0)
            if np.any(ok_dd_3):
                with np.errstate(all="ignore"):
                    ratio2 = np.divide(min_low_36, asof_close, out=np.full(n_months, np.nan, dtype="float64"), where=ok_dd_3)
                    trailing_low_dd_mag_3y[ok_dd_3] = np.abs(ratio2[ok_dd_3] - 1.0)
            ok_score_3 = np.isfinite(cagr_3y_asof_close) & np.isfinite(trailing_low_dd_mag_3y) & (trailing_low_dd_mag_3y > 0)
            if np.any(ok_score_3):
                np.divide(cagr_3y_asof_close, trailing_low_dd_mag_3y, out=cagr_to_trailing_low_3y, where=ok_score_3)
            asof_close_60 = np.full(n_months, np.nan, dtype="float64")
            if n_months > 60:
                asof_close_60[60:] = asof_close[: n_months - 60]
            ok_cagr = np.isfinite(asof_close) & np.isfinite(asof_close_60) & (asof_close > 0) & (asof_close_60 > 0)
            if np.any(ok_cagr):
                with np.errstate(all="ignore"):
                    ratio = np.divide(asof_close, asof_close_60, out=np.full(n_months, np.nan, dtype="float64"), where=ok_cagr)
                    cagr_5y_asof_close[ok_cagr] = np.power(ratio[ok_cagr], 0.2) - 1.0
            trailing_min_60_end = pd.Series(seg_min, dtype="float64").rolling(60, min_periods=60).min().to_numpy(dtype="float64")
            min_low_60 = np.full(n_months, np.nan, dtype="float64")
            if n_months > 1:
                min_low_60[1:] = trailing_min_60_end[:-1]
            ok_dd = np.isfinite(min_low_60) & np.isfinite(asof_close) & (asof_close > 0) & (min_low_60 > 0)
            if np.any(ok_dd):
                with np.errstate(all="ignore"):
                    ratio2 = np.divide(min_low_60, asof_close, out=np.full(n_months, np.nan, dtype="float64"), where=ok_dd)
                    trailing_low_dd_mag_5y[ok_dd] = np.abs(ratio2[ok_dd] - 1.0)
            ok_score = np.isfinite(cagr_5y_asof_close) & np.isfinite(trailing_low_dd_mag_5y) & (trailing_low_dd_mag_5y > 0)
            if np.any(ok_score):
                np.divide(cagr_5y_asof_close, trailing_low_dd_mag_5y, out=cagr_to_trailing_low_5y, where=ok_score)

            # 3y MAE (past 3 completed 12m windows).
            offsets3 = [12, 24, 36]
            past_mae_3y = np.full((len(offsets3), n_months), np.nan, dtype="float64")
            for j, off in enumerate(offsets3):
                if off < n_months:
                    past_mae_3y[j, off:] = mae_12m[: n_months - off]
            n_past_3y_mae = np.sum(np.isfinite(past_mae_3y), axis=0).astype("float64")
            worst_past_3y_mae_12m = np.full(n_months, np.nan, dtype="float64")
            median_past_3y_mae_12m = np.full(n_months, np.nan, dtype="float64")
            ok_past_3 = n_past_3y_mae > 0
            if np.any(ok_past_3):
                worst_past_3y_mae_12m[ok_past_3] = np.nanmin(past_mae_3y[:, ok_past_3], axis=0)
                with np.errstate(all="ignore"):
                    median_past_3y_mae_12m[ok_past_3] = np.nanmedian(past_mae_3y[:, ok_past_3], axis=0)
            adverse_mag_3y = -worst_past_3y_mae_12m
            adverse_median_mag_3y = -median_past_3y_mae_12m

            cagr_to_mae_3y = np.full(n_months, np.nan, dtype="float64")
            ratio_ok_3 = np.isfinite(cagr_3y) & np.isfinite(adverse_mag_3y) & (adverse_mag_3y > 0)
            if np.any(ratio_ok_3):
                np.divide(cagr_3y, adverse_mag_3y, out=cagr_to_mae_3y, where=ratio_ok_3)
            cagr_to_mae_median_3y = np.full(n_months, np.nan, dtype="float64")
            ratio_ok2_3 = np.isfinite(cagr_3y) & np.isfinite(adverse_median_mag_3y) & (adverse_median_mag_3y > 0)
            if np.any(ratio_ok2_3):
                np.divide(cagr_3y, adverse_median_mag_3y, out=cagr_to_mae_median_3y, where=ratio_ok2_3)

            cagr_to_mae = np.full(n_months, np.nan, dtype="float64")
            ratio_ok = np.isfinite(cagr_5y) & np.isfinite(adverse_mag) & (adverse_mag > 0)
            if np.any(ratio_ok):
                np.divide(cagr_5y, adverse_mag, out=cagr_to_mae, where=ratio_ok)
            cagr_to_mae_median = np.full(n_months, np.nan, dtype="float64")
            ratio_ok2 = np.isfinite(cagr_5y) & np.isfinite(adverse_median_mag) & (adverse_median_mag > 0)
            if np.any(ratio_ok2):
                np.divide(cagr_5y, adverse_median_mag, out=cagr_to_mae_median, where=ratio_ok2)

            # --- Raw (split-adjusted, price-only) metrics (used when cfg.price_mode == "raw") ---
            entry_open_raw = entry_open_raw_m
            entry_close_raw = entry_close_raw_m
            entry_dates64_raw = pd.to_datetime(entry_date_raw).to_numpy(dtype="datetime64[ns]")

            # Previous rolling 12-month return (open-to-open), raw mode.
            prev_12m_return_raw = np.full(n_months, np.nan, dtype="float64")
            prev_open_raw = np.full(n_months, np.nan, dtype="float64")
            if n_months > cfg.hold_period_months:
                prev_open_raw[cfg.hold_period_months :] = entry_open_raw[: n_months - cfg.hold_period_months]
                ok_prev_raw = (
                    np.isfinite(entry_open_raw) & np.isfinite(prev_open_raw) & (entry_open_raw > 0) & (prev_open_raw > 0)
                )
                if np.any(ok_prev_raw):
                    np.divide(entry_open_raw, prev_open_raw, out=prev_12m_return_raw, where=ok_prev_raw)
                    prev_12m_return_raw = prev_12m_return_raw - 1.0

            window_min_12m_raw = _rolling_forward_min(seg_min_raw, window=cfg.hold_period_months)
            mae_ratio_12m_raw = np.full(n_months, np.nan, dtype="float64")
            mae_ok_raw = np.isfinite(window_min_12m_raw) & np.isfinite(entry_open_raw) & (entry_open_raw > 0)
            if np.any(mae_ok_raw):
                np.divide(window_min_12m_raw, entry_open_raw, out=mae_ratio_12m_raw, where=mae_ok_raw)
            mae_12m_raw = mae_ratio_12m_raw - 1.0

            offsets = [12, 24, 36, 48, 60]
            past_mae_raw = np.full((len(offsets), n_months), np.nan, dtype="float64")
            for j, off in enumerate(offsets):
                if off < n_months:
                    past_mae_raw[j, off:] = mae_12m_raw[: n_months - off]
            n_past_5y_mae_raw = np.sum(np.isfinite(past_mae_raw), axis=0).astype("float64")
            worst_past_5y_mae_12m_raw = np.full(n_months, np.nan, dtype="float64")
            median_past_5y_mae_12m_raw = np.full(n_months, np.nan, dtype="float64")
            ok_past_raw = n_past_5y_mae_raw > 0
            if np.any(ok_past_raw):
                worst_past_5y_mae_12m_raw[ok_past_raw] = np.nanmin(past_mae_raw[:, ok_past_raw], axis=0)
                with np.errstate(all="ignore"):
                    median_past_5y_mae_12m_raw[ok_past_raw] = np.nanmedian(past_mae_raw[:, ok_past_raw], axis=0)
            adverse_mag_raw = -worst_past_5y_mae_12m_raw
            adverse_median_mag_raw = -median_past_5y_mae_12m_raw

            offsets3 = [12, 24, 36]
            past_mae_3y_raw = np.full((len(offsets3), n_months), np.nan, dtype="float64")
            for j, off in enumerate(offsets3):
                if off < n_months:
                    past_mae_3y_raw[j, off:] = mae_12m_raw[: n_months - off]
            n_past_3y_mae_raw = np.sum(np.isfinite(past_mae_3y_raw), axis=0).astype("float64")
            worst_past_3y_mae_12m_raw = np.full(n_months, np.nan, dtype="float64")
            median_past_3y_mae_12m_raw = np.full(n_months, np.nan, dtype="float64")
            ok_past_3_raw = n_past_3y_mae_raw > 0
            if np.any(ok_past_3_raw):
                worst_past_3y_mae_12m_raw[ok_past_3_raw] = np.nanmin(past_mae_3y_raw[:, ok_past_3_raw], axis=0)
                with np.errstate(all="ignore"):
                    median_past_3y_mae_12m_raw[ok_past_3_raw] = np.nanmedian(past_mae_3y_raw[:, ok_past_3_raw], axis=0)
            adverse_mag_3y_raw = -worst_past_3y_mae_12m_raw
            adverse_median_mag_3y_raw = -median_past_3y_mae_12m_raw

            completed_mae_raw = pd.Series(mae_12m_raw, dtype="float64").shift(cfg.hold_period_months)
            worst_past_60m_raw = completed_mae_raw.rolling(60, min_periods=60).min().to_numpy(dtype="float64")
            n_past_mae_raw = completed_mae_raw.rolling(60, min_periods=60).count().to_numpy(dtype="float64")

            cagr_3y_raw = np.full(n_months, np.nan, dtype="float64")
            cagr_5y_raw = np.full(n_months, np.nan, dtype="float64")
            valid_entry_raw = ~np.isnat(entry_dates64_raw)
            if np.any(valid_entry_raw):
                cagr_3y_raw[valid_entry_raw] = _compute_cagr_years(
                    dates,
                    close_raw,
                    entry_dates64_raw[valid_entry_raw],
                    years=3,
                    min_history_bars=min(cfg.min_history_bars, 3 * 252),
                )
                cagr_5y_raw[valid_entry_raw] = _compute_cagr_years(
                    dates,
                    close_raw,
                    entry_dates64_raw[valid_entry_raw],
                    years=5,
                    min_history_bars=cfg.min_history_bars,
                )

            max_dd_3y_raw = np.full(n_months, np.nan, dtype="float64")
            calmar_3y_raw = np.full(n_months, np.nan, dtype="float64")
            for t in range(36, n_months):
                window = seg_last_close_raw[t - 36 : t]
                dd = _max_drawdown_pct_window(window)
                if np.isfinite(dd) and dd < 0 and np.isfinite(cagr_3y_raw[t]):
                    max_dd_3y_raw[t] = dd
                    calmar_3y_raw[t] = float(cagr_3y_raw[t]) / abs(float(dd))

            max_dd_5y_raw = np.full(n_months, np.nan, dtype="float64")
            calmar_5y_raw = np.full(n_months, np.nan, dtype="float64")
            for t in range(60, n_months):
                window = seg_last_close_raw[t - 60 : t]
                dd = _max_drawdown_pct_window(window)
                if np.isfinite(dd) and dd < 0 and np.isfinite(cagr_5y_raw[t]):
                    max_dd_5y_raw[t] = dd
                    calmar_5y_raw[t] = float(cagr_5y_raw[t]) / abs(float(dd))

            cagr_3y_asof_close_raw = np.full(n_months, np.nan, dtype="float64")
            trailing_low_dd_mag_3y_raw = np.full(n_months, np.nan, dtype="float64")
            cagr_to_trailing_low_3y_raw = np.full(n_months, np.nan, dtype="float64")
            cagr_5y_asof_close_raw = np.full(n_months, np.nan, dtype="float64")
            trailing_low_dd_mag_5y_raw = np.full(n_months, np.nan, dtype="float64")
            cagr_to_trailing_low_5y_raw = np.full(n_months, np.nan, dtype="float64")
            asof_close_raw = np.full(n_months, np.nan, dtype="float64")
            if n_months > 1:
                asof_close_raw[1:] = seg_last_close_raw[:-1]
            asof_close_36_raw = np.full(n_months, np.nan, dtype="float64")
            if n_months > 36:
                asof_close_36_raw[36:] = asof_close_raw[: n_months - 36]
            ok_cagr_3_raw = (
                np.isfinite(asof_close_raw) & np.isfinite(asof_close_36_raw) & (asof_close_raw > 0) & (asof_close_36_raw > 0)
            )
            if np.any(ok_cagr_3_raw):
                with np.errstate(all="ignore"):
                    ratio = np.divide(
                        asof_close_raw, asof_close_36_raw, out=np.full(n_months, np.nan, dtype="float64"), where=ok_cagr_3_raw
                    )
                    cagr_3y_asof_close_raw[ok_cagr_3_raw] = np.power(ratio[ok_cagr_3_raw], 1.0 / 3.0) - 1.0
            trailing_min_36_end_raw = (
                pd.Series(seg_min_raw, dtype="float64").rolling(36, min_periods=36).min().to_numpy(dtype="float64")
            )
            min_low_36_raw = np.full(n_months, np.nan, dtype="float64")
            if n_months > 1:
                min_low_36_raw[1:] = trailing_min_36_end_raw[:-1]
            ok_dd_3_raw = np.isfinite(min_low_36_raw) & np.isfinite(asof_close_raw) & (asof_close_raw > 0) & (min_low_36_raw > 0)
            if np.any(ok_dd_3_raw):
                with np.errstate(all="ignore"):
                    ratio2 = np.divide(
                        min_low_36_raw, asof_close_raw, out=np.full(n_months, np.nan, dtype="float64"), where=ok_dd_3_raw
                    )
                    trailing_low_dd_mag_3y_raw[ok_dd_3_raw] = np.abs(ratio2[ok_dd_3_raw] - 1.0)
            ok_score_3_raw = (
                np.isfinite(cagr_3y_asof_close_raw) & np.isfinite(trailing_low_dd_mag_3y_raw) & (trailing_low_dd_mag_3y_raw > 0)
            )
            if np.any(ok_score_3_raw):
                np.divide(
                    cagr_3y_asof_close_raw, trailing_low_dd_mag_3y_raw, out=cagr_to_trailing_low_3y_raw, where=ok_score_3_raw
                )

            asof_close_60_raw = np.full(n_months, np.nan, dtype="float64")
            if n_months > 60:
                asof_close_60_raw[60:] = asof_close_raw[: n_months - 60]
            ok_cagr_raw = (
                np.isfinite(asof_close_raw) & np.isfinite(asof_close_60_raw) & (asof_close_raw > 0) & (asof_close_60_raw > 0)
            )
            if np.any(ok_cagr_raw):
                with np.errstate(all="ignore"):
                    ratio = np.divide(
                        asof_close_raw,
                        asof_close_60_raw,
                        out=np.full(n_months, np.nan, dtype="float64"),
                        where=ok_cagr_raw,
                    )
                    cagr_5y_asof_close_raw[ok_cagr_raw] = np.power(ratio[ok_cagr_raw], 0.2) - 1.0
            trailing_min_60_end_raw = (
                pd.Series(seg_min_raw, dtype="float64").rolling(60, min_periods=60).min().to_numpy(dtype="float64")
            )
            min_low_60_raw = np.full(n_months, np.nan, dtype="float64")
            if n_months > 1:
                min_low_60_raw[1:] = trailing_min_60_end_raw[:-1]
            ok_dd_raw = np.isfinite(min_low_60_raw) & np.isfinite(asof_close_raw) & (asof_close_raw > 0) & (min_low_60_raw > 0)
            if np.any(ok_dd_raw):
                with np.errstate(all="ignore"):
                    ratio2 = np.divide(
                        min_low_60_raw, asof_close_raw, out=np.full(n_months, np.nan, dtype="float64"), where=ok_dd_raw
                    )
                    trailing_low_dd_mag_5y_raw[ok_dd_raw] = np.abs(ratio2[ok_dd_raw] - 1.0)
            ok_score_raw = (
                np.isfinite(cagr_5y_asof_close_raw) & np.isfinite(trailing_low_dd_mag_5y_raw) & (trailing_low_dd_mag_5y_raw > 0)
            )
            if np.any(ok_score_raw):
                np.divide(cagr_5y_asof_close_raw, trailing_low_dd_mag_5y_raw, out=cagr_to_trailing_low_5y_raw, where=ok_score_raw)

            cagr_to_mae_3y_raw = np.full(n_months, np.nan, dtype="float64")
            ratio_ok_3_raw = np.isfinite(cagr_3y_raw) & np.isfinite(adverse_mag_3y_raw) & (adverse_mag_3y_raw > 0)
            if np.any(ratio_ok_3_raw):
                np.divide(cagr_3y_raw, adverse_mag_3y_raw, out=cagr_to_mae_3y_raw, where=ratio_ok_3_raw)
            cagr_to_mae_median_3y_raw = np.full(n_months, np.nan, dtype="float64")
            ratio_ok2_3_raw = np.isfinite(cagr_3y_raw) & np.isfinite(adverse_median_mag_3y_raw) & (adverse_median_mag_3y_raw > 0)
            if np.any(ratio_ok2_3_raw):
                np.divide(cagr_3y_raw, adverse_median_mag_3y_raw, out=cagr_to_mae_median_3y_raw, where=ratio_ok2_3_raw)

            cagr_to_mae_raw = np.full(n_months, np.nan, dtype="float64")
            ratio_ok_raw = np.isfinite(cagr_5y_raw) & np.isfinite(adverse_mag_raw) & (adverse_mag_raw > 0)
            if np.any(ratio_ok_raw):
                np.divide(cagr_5y_raw, adverse_mag_raw, out=cagr_to_mae_raw, where=ratio_ok_raw)
            cagr_to_mae_median_raw = np.full(n_months, np.nan, dtype="float64")
            ratio_ok2_raw = np.isfinite(cagr_5y_raw) & np.isfinite(adverse_median_mag_raw) & (adverse_median_mag_raw > 0)
            if np.any(ratio_ok2_raw):
                np.divide(cagr_5y_raw, adverse_median_mag_raw, out=cagr_to_mae_median_raw, where=ratio_ok2_raw)

            mcap = np.full(n_months, np.nan, dtype="float64")
            if perm_i in daily_map:
                d_dates, d_mcap = daily_map[perm_i]
                idxs = np.searchsorted(d_dates, entry_dates64, side="left") - 1
                ok = idxs >= 0
                mcap[ok] = d_mcap[idxs[ok]]

            dv_full = close_unadj * volume_unadj
            dv_roll = (
                pd.Series(dv_full, dtype="float64")
                .rolling(cfg.dollar_volume_lookback_bars, min_periods=cfg.dollar_volume_lookback_bars)
                .median()
            )
            dv_median = np.full(n_months, np.nan, dtype="float64")
            asof = np.searchsorted(dates, entry_dates64, side="left") - 1
            ok = (asof >= 0) & (asof < len(dv_roll))
            dv_median[ok] = dv_roll.to_numpy(dtype="float64")[asof[ok]]

            mcap_raw = np.full(n_months, np.nan, dtype="float64")
            if perm_i in daily_map:
                d_dates, d_mcap = daily_map[perm_i]
                idxs = np.searchsorted(d_dates, entry_dates64_raw, side="left") - 1
                ok = idxs >= 0
                mcap_raw[ok] = d_mcap[idxs[ok]]

            dv_median_raw = np.full(n_months, np.nan, dtype="float64")
            asof_raw = np.searchsorted(dates, entry_dates64_raw, side="left") - 1
            ok_raw = (asof_raw >= 0) & (asof_raw < len(dv_roll))
            dv_median_raw[ok_raw] = dv_roll.to_numpy(dtype="float64")[asof_raw[ok_raw]]

            # Entry-month aligned "green years": require previous N completed rolling-years (hold_period_months windows) to be positive.
            # Use open-to-open to match the execution convention (entry uses select_date open; maturity sells target the maturity select_date).
            year_end_open = (
                pd.Series(entry_open, dtype="float64").shift(-cfg.hold_period_months).to_numpy(dtype="float64")
                if cfg.hold_period_months > 0
                else np.full(n_months, np.nan, dtype="float64")
            )
            rolling_year_ratio = np.full(n_months, np.nan, dtype="float64")
            yr_ok = np.isfinite(year_end_open) & np.isfinite(entry_open) & (entry_open > 0)
            if np.any(yr_ok):
                np.divide(year_end_open, entry_open, out=rolling_year_ratio, where=yr_ok)
            rolling_year_ret = rolling_year_ratio - 1.0
            year_green_start = np.isfinite(rolling_year_ret) & (rolling_year_ret > 0.0)
            last_n_green = np.full(n_months, True, dtype=bool)
            if cfg.require_last_n_years_green > 0:
                for k in range(1, cfg.require_last_n_years_green + 1):
                    shift = cfg.hold_period_months * k
                    prev = np.full(n_months, False, dtype=bool)
                    if shift < n_months:
                        prev[shift:] = year_green_start[: n_months - shift]
                    last_n_green &= prev

            year_end_open_raw = (
                pd.Series(entry_open_raw, dtype="float64").shift(-cfg.hold_period_months).to_numpy(dtype="float64")
                if cfg.hold_period_months > 0
                else np.full(n_months, np.nan, dtype="float64")
            )
            rolling_year_ratio_raw = np.full(n_months, np.nan, dtype="float64")
            yr_ok_raw = np.isfinite(year_end_open_raw) & np.isfinite(entry_open_raw) & (entry_open_raw > 0)
            if np.any(yr_ok_raw):
                np.divide(year_end_open_raw, entry_open_raw, out=rolling_year_ratio_raw, where=yr_ok_raw)
            rolling_year_ret_raw = rolling_year_ratio_raw - 1.0
            year_green_start_raw = np.isfinite(rolling_year_ret_raw) & (rolling_year_ret_raw > 0.0)
            last_n_green_raw = np.full(n_months, True, dtype=bool)
            if cfg.require_last_n_years_green > 0:
                for k in range(1, cfg.require_last_n_years_green + 1):
                    shift = cfg.hold_period_months * k
                    prev = np.full(n_months, False, dtype=bool)
                    if shift < n_months:
                        prev[shift:] = year_green_start_raw[: n_months - shift]
                    last_n_green_raw &= prev

            has_open = np.isfinite(entry_open) & (entry_open > 0)
            has_open_raw = np.isfinite(entry_open_raw) & (entry_open_raw > 0)
            has_symbol = np.array(symbol_key_asof, dtype=object) != ""
            cand = pd.DataFrame(
                {
                    "instrument_id": instrument_id,
                    "instrument_id_is_fallback": False,
                    "permaticker": perm_i,
                    "symbol_key": symbol_key_asof,
                    "table": table_asof,
                    "ticker": ticker_asof,
                    "category_asof": np.array(category_asof, dtype=object),
                    "name_asof": np.array(name_asof, dtype=object),
                    "is_equity_asof": np.array(is_equity_asof, dtype=bool),
                    "lastpricedate": pd.to_datetime(lastpricedate_asof),
                    "year_month": year_months,
                    "month_idx": month_idxs,
                    "select_date": select_dates,
                    "entry_date": entry_date,
                    "entry_open_adj": entry_open,
                    "entry_close_adj": entry_close,
                    "entry_date_raw": entry_date_raw,
                    "entry_open_raw": entry_open_raw,
                    "entry_close_raw": entry_close_raw,
                    "marketcap_entry": mcap,
                    "marketcap_missing": ~np.isfinite(mcap),
                    "dollar_volume_median": dv_median,
                    "marketcap_entry_raw": mcap_raw,
                    "marketcap_missing_raw": ~np.isfinite(mcap_raw),
                    "dollar_volume_median_raw": dv_median_raw,
                    "cagr_3y": cagr_3y,
                    "cagr_5y": cagr_5y,
                    "prev_12m_return": prev_12m_return,
                    "max_dd_3y": max_dd_3y,
                    "calmar_3y": calmar_3y,
                    "max_dd_5y": max_dd_5y,
                    "calmar_5y": calmar_5y,
                    "cagr_3y_asof_close": cagr_3y_asof_close,
                    "trailing_low_dd_mag_3y": trailing_low_dd_mag_3y,
                    "cagr_to_trailing_low_3y": cagr_to_trailing_low_3y,
                    "cagr_5y_asof_close": cagr_5y_asof_close,
                    "trailing_low_dd_mag_5y": trailing_low_dd_mag_5y,
                    "cagr_to_trailing_low_5y": cagr_to_trailing_low_5y,
                    "fwd_mae_12m_from_entry": mae_12m,
                    "worst_past_60m_mae_12m": worst_past_60m,
                    "n_past_mae_12m": n_past_mae,
                    "worst_past_3y_mae_12m": worst_past_3y_mae_12m,
                    "median_past_3y_mae_12m": median_past_3y_mae_12m,
                    "n_past_3y_mae_12m": n_past_3y_mae,
                    "adverse_mag_3y": adverse_mag_3y,
                    "adverse_median_mag_3y": adverse_median_mag_3y,
                    "cagr_to_mae_3y": cagr_to_mae_3y,
                    "cagr_to_mae_median_3y": cagr_to_mae_median_3y,
                    "worst_past_5y_mae_12m": worst_past_5y_mae_12m,
                    "median_past_5y_mae_12m": median_past_5y_mae_12m,
                    "n_past_5y_mae_12m": n_past_5y_mae,
                    "adverse_mag": adverse_mag,
                    "adverse_median_mag": adverse_median_mag,
                    "cagr_to_mae_5y": cagr_to_mae,
                    "cagr_to_mae_median_5y": cagr_to_mae_median,
                    "last_n_years_green": last_n_green,
                    "has_open": has_open,
                    "has_open_raw": has_open_raw,
                    "is_sfp_blacklisted": is_sfp_blacklisted,
                    "cagr_3y_raw": cagr_3y_raw,
                    "cagr_5y_raw": cagr_5y_raw,
                    "prev_12m_return_raw": prev_12m_return_raw,
                    "max_dd_3y_raw": max_dd_3y_raw,
                    "calmar_3y_raw": calmar_3y_raw,
                    "max_dd_5y_raw": max_dd_5y_raw,
                    "calmar_5y_raw": calmar_5y_raw,
                    "cagr_3y_asof_close_raw": cagr_3y_asof_close_raw,
                    "trailing_low_dd_mag_3y_raw": trailing_low_dd_mag_3y_raw,
                    "cagr_to_trailing_low_3y_raw": cagr_to_trailing_low_3y_raw,
                    "cagr_5y_asof_close_raw": cagr_5y_asof_close_raw,
                    "trailing_low_dd_mag_5y_raw": trailing_low_dd_mag_5y_raw,
                    "cagr_to_trailing_low_5y_raw": cagr_to_trailing_low_5y_raw,
                    "fwd_mae_12m_from_entry_raw": mae_12m_raw,
                    "worst_past_60m_mae_12m_raw": worst_past_60m_raw,
                    "n_past_mae_12m_raw": n_past_mae_raw,
                    "worst_past_3y_mae_12m_raw": worst_past_3y_mae_12m_raw,
                    "median_past_3y_mae_12m_raw": median_past_3y_mae_12m_raw,
                    "n_past_3y_mae_12m_raw": n_past_3y_mae_raw,
                    "adverse_mag_3y_raw": adverse_mag_3y_raw,
                    "adverse_median_mag_3y_raw": adverse_median_mag_3y_raw,
                    "cagr_to_mae_3y_raw": cagr_to_mae_3y_raw,
                    "cagr_to_mae_median_3y_raw": cagr_to_mae_median_3y_raw,
                    "worst_past_5y_mae_12m_raw": worst_past_5y_mae_12m_raw,
                    "median_past_5y_mae_12m_raw": median_past_5y_mae_12m_raw,
                    "n_past_5y_mae_12m_raw": n_past_5y_mae_raw,
                    "adverse_mag_raw": adverse_mag_raw,
                    "adverse_median_mag_raw": adverse_median_mag_raw,
                    "cagr_to_mae_5y_raw": cagr_to_mae_raw,
                    "cagr_to_mae_median_5y_raw": cagr_to_mae_median_raw,
                    "last_n_years_green_raw": last_n_green_raw,
                }
            )
            cand = cand[has_symbol & (has_open | has_open_raw)].copy()
            if cand.empty:
                continue
            cand["bucket"] = perm_bucket
            out_rows.append(cand)
            if sum(len(x) for x in out_rows) >= 200_000:
                flush_year_month(out_rows, perm_bucket=perm_bucket, counter=counter)
                out_rows = []

        if out_rows:
            flush_year_month(out_rows, perm_bucket=perm_bucket, counter=counter)
        return processed

    results = Parallel(n_jobs=cfg.jobs)(delayed(build_one_perm_bucket)(b) for b in perm_buckets)
    total_processed = int(sum(results))

    _write_manifest(
        manifest_path,
        {
            "cache_version": CACHE_VERSION,
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "inputs": inputs,
            "config_key": config_key,
            "total_permatickers_processed": int(total_processed),
        },
    )


def _stable_sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "instrument_id_is_fallback" in df.columns:
        df["_fallback_sort"] = df["instrument_id_is_fallback"].astype(bool)
    else:
        df["_fallback_sort"] = False
    if "permaticker" in df.columns:
        df["_permaticker_sort"] = df["permaticker"].astype("Int64").fillna(2**63 - 1).astype("int64")
    else:
        df["_permaticker_sort"] = 2**63 - 1
    return df.sort_values(
        ["rank_score", "marketcap_entry", "_fallback_sort", "_permaticker_sort", "instrument_id", "symbol_key"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["_fallback_sort", "_permaticker_sort"])


def _normalize_issuer_name(name: str) -> str:
    """
    Best-effort issuer normalization to dedupe share classes / preferred series.

    Goal: treat "TSAKOS ENERGY NAVIGATION LTD" as one issuer across TEN-PE/TEN-PF, etc.
    This is a heuristic and intentionally conservative (strip common legal suffixes).
    """
    s = (name or "").strip().upper()
    if not s:
        return ""
    # Replace punctuation with spaces, then collapse whitespace.
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    s = " ".join(s.split())
    if not s:
        return ""
    suffixes = {
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "LIMITED",
        "PLC",
        "AG",
        "NV",
        "SA",
        "LP",
        "L P",
    }
    parts = s.split()
    while parts and parts[-1] in suffixes:
        parts.pop()
    return " ".join(parts)


def _issuer_key_from_candidates(df: pd.DataFrame) -> pd.Series:
    """
    Compute an issuer dedupe key for candidates.

    Priority:
    - normalized name_asof (if present and non-empty)
    - else base ticker (strip series suffix after first '-')
    """
    if "name_asof" in df.columns:
        name = df["name_asof"].astype("string").fillna("")
        issuer = name.map(lambda x: _normalize_issuer_name(str(x)))
    else:
        issuer = pd.Series("", index=df.index, dtype="string")
    has_issuer = issuer.astype("string").str.len().fillna(0) > 0
    if "ticker" in df.columns:
        tick = df["ticker"].astype("string").fillna("").str.upper()
    elif "symbol_key" in df.columns:
        tick = df["symbol_key"].astype("string").fillna("").str.split(":", n=1).str[-1].str.upper()
    else:
        tick = pd.Series("", index=df.index, dtype="string")
    base = tick.str.split("-", n=1).str[0].fillna("")
    issuer = issuer.where(has_issuer, base)
    return issuer.astype("string")


def load_candidates_for_select_date(
    candidates_dir: Path, select_date: pd.Timestamp, *, bucket: int | None = None
) -> pd.DataFrame:
    year_month = pd.Timestamp(select_date).to_period("M").strftime("%Y-%m")
    if bucket is None:
        files = sorted((candidates_dir).glob(f"year_month={year_month}/*.parquet"))
    else:
        files = sorted((candidates_dir).glob(f"year_month={year_month}/bucket{bucket:02d}-part-*.parquet"))
    if not files:
        return pd.DataFrame()
    table = pq.read_table([str(p) for p in files])
    df = table.to_pandas()
    df["select_date"] = pd.to_datetime(df["select_date"])
    df = df[df["select_date"] == pd.Timestamp(select_date)]
    return df.reset_index(drop=True)


@dataclass
class Trade:
    date: pd.Timestamp
    portfolio: Literal["strategy", "benchmark"]
    side: Literal["BUY", "SELL"]
    instrument_id: str
    symbol_key: str
    ticker: str
    table: str
    shares: int
    raw_price: float
    effective_price: float
    commission: float
    slippage_bps: float
    notional: float
    cash_after: float
    price_date: pd.Timestamp | None = None


def _apply_slippage(price: float, *, side: Literal["BUY", "SELL"], slippage_bps: float) -> float:
    if side == "BUY":
        return price * (1.0 + slippage_bps / 10_000.0)
    return price * (1.0 - slippage_bps / 10_000.0)


def _max_drawdown_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def _run_id(cfg: EntryRotationConfig, inputs: SharadarPaths) -> str:
    blob = {
        "cfg": dataclasses.asdict(cfg),
        "inputs": {k: _fingerprint_file(v, compute_sha256=False) for k, v in dataclasses.asdict(inputs).items()},
    }
    raw = json.dumps(blob, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def run_backtest(cfg: EntryRotationConfig, *, rebuild_cache: bool = False) -> Path:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")

    inputs = ensure_sharadar_caches(cfg, rebuild=rebuild_cache)
    run_id = _run_id(cfg, inputs)
    out_root = cfg.output_dir.expanduser().resolve() / cfg.mode / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "mode": cfg.mode,
        "config": dataclasses.asdict(cfg),
        "inputs": {
            "sharadar_dir": str(cfg.sharadar_dir),
            "tickers_csv": str(inputs.tickers_csv),
            "sep_csv": str(inputs.sep_csv),
            "sfp_csv": str(inputs.sfp_csv),
            "daily_csv": str(inputs.daily_csv),
        },
        "policies": {
            "fractional_shares": False,
            "cash_yield_pct": 0.0,
            "execution": {
                "buys": "open_else_close",
                "sells": (
                    "no-forward: open_else_close (no stale, no forward); "
                    "defer-forward: open_else_close_else_next_open_or_close (cash_date=execution_date); "
                    "end-of-backtest: force-close at last_close_on_or_before(final_day)"
                ),
            },
            "pit": "latest_vintage_closeadj",
        },
    }
    (out_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    month_index = pq.read_table(cfg.cache_dir / "month_index" / "month_index.parquet").to_pandas()
    month_index["select_date"] = pd.to_datetime(month_index["select_date"])

    if cfg.mode == "annual-lumpsum":
        _run_mode_annual(cfg, out_root, month_index)
    else:
        _run_mode_monthly_dca(cfg, out_root, month_index)
    return out_root


def select_snapshot_for_date(cfg: EntryRotationConfig, *, select_date: pd.Timestamp, top_n: int | None = None) -> pd.DataFrame:
    """
    Return the top-N selection snapshot for an arbitrary date.

    Implementation note: this uses the precomputed monthly candidates for the calendar month containing select_date.
    The ranking/gates are therefore aligned to the monthly strategy cadence; this is intended for "what would the model pick this month?"
    style queries, not intramonth factor recomputation.
    """
    ym = _year_month_from_date(pd.Timestamp(select_date))
    candidates_root = cfg.cache_dir.expanduser().resolve() / "candidates_monthly" / f"year_month={ym}"
    files = sorted(candidates_root.glob("*.parquet"))
    if not files:
        raise ValueError(f"No candidates cache for year_month={ym} under {candidates_root.parent}")
    # Read partitions individually to avoid pyarrow dataset schema merge issues (dictionary-encoded strings vs plain strings).
    parts = [pd.read_parquet(p) for p in files]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    if "year_month" in df.columns:
        df = df[df["year_month"].astype(str) == ym].copy()
    if df.empty:
        return df

    # Apply the normal selector.
    if top_n is not None:
        cfg = dataclasses.replace(cfg, top_n=int(top_n))
    selected = _select_top_n(df, cfg)
    if selected.empty:
        return selected
    selected = selected.copy()
    selected["requested_select_date"] = pd.Timestamp(select_date)
    return selected


def _apply_price_mode_to_candidates(df: pd.DataFrame, cfg: EntryRotationConfig) -> pd.DataFrame:
    """
    Adapt candidates to the configured price_mode.

    The candidates cache stores adjusted-mode columns as the primary (historical behavior), plus *_raw columns. When cfg.price_mode=="raw",
    we copy the raw columns into the primary names so that ranking/gating logic stays unchanged.
    """
    if cfg.price_mode != "raw":
        return df

    mapping = {
        # Execution-aligned entry fields (used mostly for reporting/diagnostics).
        "entry_date": "entry_date_raw",
        "entry_open_adj": "entry_open_raw",
        "entry_close_adj": "entry_close_raw",
        # Gates.
        "has_open": "has_open_raw",
        "last_n_years_green": "last_n_years_green_raw",
        "marketcap_entry": "marketcap_entry_raw",
        "marketcap_missing": "marketcap_missing_raw",
        "dollar_volume_median": "dollar_volume_median_raw",
        # Metrics.
        "cagr_3y": "cagr_3y_raw",
        "cagr_5y": "cagr_5y_raw",
        "prev_12m_return": "prev_12m_return_raw",
        "max_dd_3y": "max_dd_3y_raw",
        "calmar_3y": "calmar_3y_raw",
        "max_dd_5y": "max_dd_5y_raw",
        "calmar_5y": "calmar_5y_raw",
        "cagr_3y_asof_close": "cagr_3y_asof_close_raw",
        "trailing_low_dd_mag_3y": "trailing_low_dd_mag_3y_raw",
        "cagr_to_trailing_low_3y": "cagr_to_trailing_low_3y_raw",
        "cagr_5y_asof_close": "cagr_5y_asof_close_raw",
        "trailing_low_dd_mag_5y": "trailing_low_dd_mag_5y_raw",
        "cagr_to_trailing_low_5y": "cagr_to_trailing_low_5y_raw",
        "fwd_mae_12m_from_entry": "fwd_mae_12m_from_entry_raw",
        "worst_past_60m_mae_12m": "worst_past_60m_mae_12m_raw",
        "n_past_mae_12m": "n_past_mae_12m_raw",
        "worst_past_3y_mae_12m": "worst_past_3y_mae_12m_raw",
        "median_past_3y_mae_12m": "median_past_3y_mae_12m_raw",
        "n_past_3y_mae_12m": "n_past_3y_mae_12m_raw",
        "adverse_mag_3y": "adverse_mag_3y_raw",
        "adverse_median_mag_3y": "adverse_median_mag_3y_raw",
        "cagr_to_mae_3y": "cagr_to_mae_3y_raw",
        "cagr_to_mae_median_3y": "cagr_to_mae_median_3y_raw",
        "worst_past_5y_mae_12m": "worst_past_5y_mae_12m_raw",
        "median_past_5y_mae_12m": "median_past_5y_mae_12m_raw",
        "n_past_5y_mae_12m": "n_past_5y_mae_12m_raw",
        "adverse_mag": "adverse_mag_raw",
        "adverse_median_mag": "adverse_median_mag_raw",
        "cagr_to_mae_5y": "cagr_to_mae_5y_raw",
        "cagr_to_mae_median_5y": "cagr_to_mae_median_5y_raw",
    }

    missing = [raw for base, raw in mapping.items() if base in df.columns and raw not in df.columns]
    if missing:
        raise ValueError(
            "price_mode=raw requires raw candidates columns in the cache. "
            "Rebuild caches with `--rebuild-cache`.\n"
            f"Missing columns: {', '.join(sorted(set(missing)))}"
        )

    df = df.copy()
    for base, raw in mapping.items():
        if base in df.columns:
            df[base] = df[raw]
    return df


def _eligibility_mask(df: pd.DataFrame, cfg: EntryRotationConfig) -> pd.Series:
    ok = pd.Series(True, index=df.index)
    ok &= df["has_open"].astype(bool)
    if not cfg.no_sfp_blacklist:
        ok &= ~df["is_sfp_blacklisted"].astype(bool)
    if cfg.universe == "equities":
        if "is_equity_asof" in df.columns:
            ok &= df["is_equity_asof"].astype(bool)
        elif "category_asof" in df.columns:
            ok &= pd.Series(_equity_mask_from_categories(df["category_asof"].to_numpy(dtype=object)), index=df.index)
        else:
            ok &= False

    table_s = df["table"].astype("string") if "table" in df.columns else pd.Series("", index=df.index, dtype="string")
    table_uc = table_s.fillna("").str.upper()

    cat_s = (
        df["category_asof"].astype("string") if "category_asof" in df.columns else pd.Series("", index=df.index, dtype="string")
    )
    if cfg.exclude_preferred:
        ok &= ~cat_s.str.contains("preferred", case=False, na=False)
    if cfg.exclude_quasi_preferred:
        quasi_terms = [
            "depositary",
            "depository",
            "unit",
            "units",
            "receipt",
            "etd",
            "trust preferred",
        ]
        quasi = pd.Series(False, index=df.index)
        for term in quasi_terms:
            quasi |= cat_s.str.contains(term, case=False, na=False)
        ok &= ~quasi
    if cfg.exclude_adr_common:
        ok &= ~(cat_s.str.contains("adr", case=False, na=False) & cat_s.str.contains("common", case=False, na=False))

    if cfg.exclude_bond_like_etfs:
        # Only apply to SFP ETFs; avoid false positives in SEP equities.
        is_sfp_etf = (table_uc == "SFP") & cat_s.str.contains(r"\betf\b", case=False, na=False, regex=True)
        if bool(is_sfp_etf.any()):
            name_s = (
                df["name_asof"].astype("string") if "name_asof" in df.columns else pd.Series("", index=df.index, dtype="string")
            )
            name_uc = name_s.fillna("").str.upper()
            # This is intentionally a name-based heuristic (no reliable instrument taxonomy in the dataset).
            # Keep it conservative: match common bond/cash keywords but avoid overly broad terms like "INCOME" alone.
            bond_like = name_uc.str.contains(
                r"(?:\bBOND\b|\bTREASUR(?:Y|IES)\b|\bT-?BILL\b|\bFIXED INCOME\b|\bMUNI\b|\bMUNICIPAL\b|\bTIPS\b|\bINFLATION\b|\bCREDIT\b|\bMORTGAGE\b|\bFLOATING RATE\b|\bAGGREGATE\b|\bDURATION\b|\bULTRA[- ]?SHORT\b|\bSHORT[- ]?TERM\b|\bSHORT MATURITY\b|\bCASH MANAGEMENT\b|\bMONEY MARKET\b)",
                regex=True,
                na=False,
            )
            ok &= ~(is_sfp_etf & bond_like)

    if cfg.exclude_lp:
        # Apply the LP/MLP heuristic only to SEP equities (true partnership units), not to SFP ETFs whose legal name can include "LP"
        # (commodity pool ETFs like USO/UNG/BNO/etc).
        is_sep_equity = (table_uc == "SEP") | (table_uc == "")
        name_s = df["name_asof"].astype("string") if "name_asof" in df.columns else pd.Series("", index=df.index, dtype="string")
        name_uc = name_s.fillna("").str.upper()
        is_lp = (
            cat_s.str.contains("limited partnership", case=False, na=False)
            | cat_s.str.contains("master limited partnership", case=False, na=False)
            | cat_s.str.contains(r"\bmlp\b", case=False, na=False, regex=True)
            | name_uc.str.contains(" LIMITED PARTNERSHIP", na=False)
            | name_uc.str.contains(" MASTER LIMITED PARTNERSHIP", na=False)
            | name_uc.str.contains(" L.P", na=False)
            | name_uc.str.endswith(" LP")
            | name_uc.str.endswith(" L.P")
            | name_uc.str.endswith(" L.P.")
        )
        ok &= ~(is_sep_equity & is_lp)

    dv = df["dollar_volume_median"].astype("float64")
    if float(cfg.dollar_volume_floor) > 0:
        ok &= np.isfinite(dv) & (dv >= float(cfg.dollar_volume_floor))

    if float(cfg.mcap_floor) > 0:
        # Sharadar DAILY.marketcap is in USD millions in the exported CSVs; treat --mcap-floor as raw USD dollars.
        mcap_dollars = df["marketcap_entry"].astype("float64") * 1_000_000.0
        mcap_ok = np.isfinite(mcap_dollars) & (mcap_dollars >= float(cfg.mcap_floor))
        fallback_ok = df["marketcap_missing"].astype(bool) & (dv >= float(cfg.dollar_volume_floor_missing_mcap))
        ok &= mcap_ok | fallback_ok
    if cfg.require_last_n_years_green > 0:
        ok &= df["last_n_years_green"].astype(bool)

    lookback_years = int(cfg.rank_lookback_years)
    cagr_col = f"cagr_{lookback_years}y"
    max_dd_col = f"max_dd_{lookback_years}y"
    calmar_col = f"calmar_{lookback_years}y"
    cagr_to_trailing_low_col = f"cagr_to_trailing_low_{lookback_years}y"
    trailing_low_dd_mag_col = f"trailing_low_dd_mag_{lookback_years}y"
    worst_past_mae_col = f"worst_past_{lookback_years}y_mae_12m"
    median_past_mae_col = f"median_past_{lookback_years}y_mae_12m"
    n_past_mae_col = f"n_past_{lookback_years}y_mae_12m"
    cagr_to_mae_col = f"cagr_to_mae_{lookback_years}y"
    cagr_to_mae_median_col = f"cagr_to_mae_median_{lookback_years}y"
    adverse_mag_col = "adverse_mag" if lookback_years == 5 else "adverse_mag_3y"
    adverse_median_mag_col = "adverse_median_mag" if lookback_years == 5 else "adverse_median_mag_3y"
    required_mae_samples = lookback_years

    def _require(col: str) -> None:
        if col not in df.columns:
            raise ValueError(
                f"Candidates cache missing required column '{col}' for --rank-lookback-years {lookback_years}. "
                "Rebuild caches with `--rebuild-cache`."
            )

    if cfg.rank_metric == "prev_12m_return":
        _require("prev_12m_return")
        ok &= np.isfinite(df["prev_12m_return"])
    elif cfg.rank_metric == "cagr_5y":
        _require(cagr_col)
        ok &= np.isfinite(df[cagr_col])
    elif cfg.rank_metric in ("cagr_to_trailing_low_5y", "cagr_to_trailing_low_5y_daily"):
        _require(cagr_to_trailing_low_col)
        _require(trailing_low_dd_mag_col)
        ok &= (
            np.isfinite(df[cagr_to_trailing_low_col])
            & np.isfinite(df[trailing_low_dd_mag_col])
            & (df[trailing_low_dd_mag_col] > 0)
        )
    elif cfg.rank_metric == "calmar":
        _require(cagr_col)
        _require(calmar_col)
        _require(max_dd_col)
        ok &= np.isfinite(df[cagr_col]) & np.isfinite(df[calmar_col]) & np.isfinite(df[max_dd_col]) & (df[max_dd_col] < 0)
    elif cfg.rank_metric == "cagr_to_mae_median":
        _require(cagr_col)
        _require(median_past_mae_col)
        _require(n_past_mae_col)
        _require(cagr_to_mae_median_col)
        _require(adverse_median_mag_col)
        ok &= (
            np.isfinite(df[cagr_col])
            & np.isfinite(df[median_past_mae_col])
            & (df[n_past_mae_col] >= required_mae_samples)
            & np.isfinite(df[cagr_to_mae_median_col])
            & np.isfinite(df[adverse_median_mag_col])
            & (df[adverse_median_mag_col] > 0)
        )
    else:
        _require(cagr_col)
        _require(worst_past_mae_col)
        _require(n_past_mae_col)
        _require(cagr_to_mae_col)
        _require(adverse_mag_col)
        ok &= (
            np.isfinite(df[cagr_col])
            & np.isfinite(df[worst_past_mae_col])
            & (df[n_past_mae_col] >= required_mae_samples)
            & np.isfinite(df[cagr_to_mae_col])
            & (df[adverse_mag_col] > 0)
        )

    # Optional user-specified min-filters (AND combined).
    for metric, min_value in cfg.min_filters:
        if metric not in df.columns:
            raise ValueError(f"--min-filter requested column not present in candidates: {metric}")
        s = df[metric].astype("float64")
        ok &= np.isfinite(s) & (s >= float(min_value))
    return ok


def _select_top_n(df: pd.DataFrame, cfg: EntryRotationConfig) -> pd.DataFrame:
    df = _apply_price_mode_to_candidates(df, cfg).copy()
    lookback_years = int(cfg.rank_lookback_years)
    cagr_col = f"cagr_{lookback_years}y"
    calmar_col = f"calmar_{lookback_years}y"
    cagr_to_trailing_low_col = f"cagr_to_trailing_low_{lookback_years}y"
    cagr_to_mae_col = f"cagr_to_mae_{lookback_years}y"
    cagr_to_mae_median_col = f"cagr_to_mae_median_{lookback_years}y"

    def _require(col: str) -> None:
        if col not in df.columns:
            raise ValueError(
                f"Candidates cache missing required column '{col}' for --rank-lookback-years {lookback_years}. "
                "Rebuild caches with `--rebuild-cache`."
            )

    if cfg.rank_metric == "calmar":
        _require(calmar_col)
        df["rank_score"] = df[calmar_col].astype("float64")
    elif cfg.rank_metric == "cagr_5y":
        _require(cagr_col)
        df["rank_score"] = df[cagr_col].astype("float64")
    elif cfg.rank_metric in ("cagr_to_trailing_low_5y", "cagr_to_trailing_low_5y_daily"):
        _require(cagr_to_trailing_low_col)
        df["rank_score"] = df[cagr_to_trailing_low_col].astype("float64")
    elif cfg.rank_metric == "cagr_to_mae_median":
        _require(cagr_to_mae_median_col)
        df["rank_score"] = df[cagr_to_mae_median_col].astype("float64")
    elif cfg.rank_metric == "prev_12m_return":
        _require("prev_12m_return")
        df["rank_score"] = df["prev_12m_return"].astype("float64")
    else:
        _require(cagr_to_mae_col)
        df["rank_score"] = df[cagr_to_mae_col].astype("float64")
    df = df[_eligibility_mask(df, cfg)].copy()
    if df.empty:
        return df
    df = _stable_sort_candidates(df)
    if "instrument_id" in df.columns:
        df = df.drop_duplicates(subset=["instrument_id"], keep="first")
    if cfg.dedupe_issuer:
        df["issuer_key"] = _issuer_key_from_candidates(df)
        # Only dedupe when we have a meaningful key; keep empty keys unique by falling back to instrument_id.
        key = df["issuer_key"].astype("string").fillna("")
        use_key = key.str.len().fillna(0) > 0
        df["_issuer_key_final"] = key.where(use_key, df.get("instrument_id", df.index.astype(str)))
        df = df.drop_duplicates(subset=["_issuer_key_final"], keep="first").drop(columns=["_issuer_key_final"])
    return df.head(cfg.top_n).reset_index(drop=True)


def _pick_first_unheld(ranked: pd.DataFrame, held_instrument_ids: set[str]) -> tuple[str | None, float | None]:
    for _, cand in ranked.iterrows():
        sym = str(cand["symbol_key"])
        instrument_id = str(cand.get("instrument_id", sym))
        if instrument_id in held_instrument_ids:
            continue
        px = float(cand["entry_open_adj"])
        return sym, px
    return None, None


def _load_symbol_series_for_valuation(cfg: EntryRotationConfig, symbols: set[str]) -> dict[str, pd.DataFrame]:
    bars_dir = cfg.cache_dir / "bars"
    series: dict[str, pd.DataFrame] = {}
    for symbol_key in symbols:
        bucket = _bucket_for_symbol_key(symbol_key, cfg.num_buckets)
        files = sorted((bars_dir).glob(f"year=*/bucket={bucket}/*.parquet"))
        if not files:
            continue
        cols = ["symbol_key", "date", "open_adj", "low_adj", "close_adj", "lastupdated"]
        try:
            table = pq.read_table(
                [str(p) for p in files],
                columns=["open_raw", "low_raw", "close_raw", *cols],
                filters=[("symbol_key", "==", symbol_key)],
            )
        except Exception:
            table = pq.read_table([str(p) for p in files], columns=cols, filters=[("symbol_key", "==", symbol_key)])
        df = table.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        df = _dedupe_latest_lastupdated(df, keys=["symbol_key", "date"], lastupdated_col="lastupdated")
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        if cfg.price_mode == "raw" and (
            "open_raw" not in df.columns or "low_raw" not in df.columns or "close_raw" not in df.columns
        ):
            raise ValueError("price_mode=raw requires raw OHLC columns in bars cache. Rebuild caches with --rebuild-cache.")
        series[symbol_key] = df
    return series


def _permaticker_from_instrument_id(instrument_id: str) -> int | None:
    if not instrument_id.startswith("PT"):
        return None
    raw = instrument_id[2:]
    if not raw.isdigit():
        return None
    return int(raw)


def _load_bars_for_instrument_id(cfg: EntryRotationConfig, instrument_id: str) -> pd.DataFrame:
    perm = _permaticker_from_instrument_id(instrument_id)
    if perm is None:
        raise ValueError(f"Unsupported instrument_id (expected PT<permaticker>): {instrument_id}")
    bars_dir = cfg.cache_dir / "bars_permaticker"
    perm_bucket = perm % cfg.num_buckets
    files = sorted(bars_dir.glob(f"perm_bucket={perm_bucket}/year_bin=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    cols = ["permaticker", "date", "open_adj", "low_adj", "close_adj", "lastupdated"]
    try:
        table = pq.read_table(
            [str(p) for p in files],
            columns=["open_raw", "low_raw", "close_raw", *cols],
            filters=[("permaticker", "==", perm)],
        )
    except Exception:
        table = pq.read_table([str(p) for p in files], columns=cols, filters=[("permaticker", "==", perm)])
    df = table.to_pandas()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = _dedupe_latest_lastupdated(df, keys=["permaticker", "date"], lastupdated_col="lastupdated")
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    if cfg.price_mode == "raw" and ("open_raw" not in df.columns or "low_raw" not in df.columns or "close_raw" not in df.columns):
        raise ValueError(
            "price_mode=raw requires raw OHLC columns in bars_permaticker cache. Rebuild caches with --rebuild-cache."
        )
    return df


def _load_series_for_instrument_ids(cfg: EntryRotationConfig, instrument_ids: set[str]) -> dict[str, pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    for iid in instrument_ids:
        series[iid] = _load_bars_for_instrument_id(cfg, iid)
    return series


@dataclass(frozen=True)
class SellFill:
    cash_date: pd.Timestamp
    price_date: pd.Timestamp
    raw_price: float


def _sell_resolver(cfg: EntryRotationConfig):
    if cfg.sell_price_policy == "defer-forward":
        return _sell_price_with_fallback
    return _sell_price_no_forward


def _price_cols_for_mode(price_mode: PriceMode) -> tuple[str, str, str]:
    if price_mode == "raw":
        return "open_raw", "low_raw", "close_raw"
    return "open_adj", "low_adj", "close_adj"


def _price_at(df: pd.DataFrame, date: pd.Timestamp, col: str) -> float | None:
    dates = df["date"].to_numpy(dtype="datetime64[ns]")
    ix = np.searchsorted(dates, np.datetime64(date), side="left")
    if ix < len(dates) and dates[ix] == np.datetime64(date):
        px = float(df.iloc[ix][col])
        if np.isfinite(px) and px > 0:
            return px
    return None


def _sell_price_with_fallback(
    df: pd.DataFrame, date: pd.Timestamp, *, open_col: str, close_col: str
) -> tuple[pd.Timestamp, float] | None:
    px_open = _price_at(df, date, open_col)
    if px_open is not None:
        return date, px_open
    px_close = _price_at(df, date, close_col)
    if px_close is not None:
        return date, px_close

    dates = df["date"].to_numpy(dtype="datetime64[ns]")
    ix = np.searchsorted(dates, np.datetime64(date), side="right")
    while ix < len(dates):
        d = pd.Timestamp(dates[ix])
        px_open2 = float(df.iloc[ix][open_col])
        if np.isfinite(px_open2) and px_open2 > 0:
            return d, px_open2
        px_close2 = float(df.iloc[ix][close_col])
        if np.isfinite(px_close2) and px_close2 > 0:
            return d, px_close2
        ix += 1
    return None


def _sell_price_no_forward(
    df: pd.DataFrame, date: pd.Timestamp, *, open_col: str, close_col: str
) -> tuple[pd.Timestamp, float] | None:
    """
    Sell price resolver that never looks forward in time.

    Preference order:
    1) open on date
    2) close on date
    """
    px_open = _price_at(df, date, open_col)
    if px_open is not None:
        return date, px_open
    px_close = _price_at(df, date, close_col)
    if px_close is not None:
        return date, px_close
    return None


def _resolve_sell_fill(
    df: pd.DataFrame,
    requested_date: pd.Timestamp,
    *,
    policy: SellPricePolicy,
    open_col: str,
    close_col: str,
    final_day: pd.Timestamp | None = None,
) -> SellFill | None:
    """
    Resolve a sell into a fill with separate cash date vs price observation date.

    - policy="defer-forward": cash_date == price_date == the execution date (requested_date if open/close exists, else first future open/close).
    - policy="no-forward": cash_date == requested_date, price_date == requested_date using open else close (no forward).
      - If the series ends strictly before requested_date (likely delisted / dataset end), fall back to the last available close
        on/before the final bar date as the price observation (still no forward).

    If final_day is provided, a deferred-forward fill that would occur after final_day falls back to no-forward on requested_date.
    """
    requested_date = pd.Timestamp(requested_date)
    if policy == "defer-forward":
        res = _sell_price_with_fallback(df, requested_date, open_col=open_col, close_col=close_col)
        if res is None:
            return None
        d, raw_px = res
        cash_date = pd.Timestamp(d)
        if final_day is not None and cash_date > pd.Timestamp(final_day):
            # Clamp to no-forward at the requested_date.
            res2 = _sell_price_no_forward(df, requested_date, open_col=open_col, close_col=close_col)
            if res2 is None:
                return None
            price_date, raw_px2 = res2
            return SellFill(cash_date=requested_date, price_date=pd.Timestamp(price_date), raw_price=float(raw_px2))
        return SellFill(cash_date=cash_date, price_date=cash_date, raw_price=float(raw_px))

    # no-forward
    res = _sell_price_no_forward(df, requested_date, open_col=open_col, close_col=close_col)
    if res is None:
        if not df.empty:
            max_date = pd.Timestamp(df["date"].max())
            if max_date < requested_date:
                last = _last_close_on_or_before(df, max_date, close_col=close_col)
                if last is None:
                    return None
                price_date, raw_px = last
                return SellFill(cash_date=requested_date, price_date=pd.Timestamp(price_date), raw_price=float(raw_px))
        return None
    price_date, raw_px = res
    return SellFill(cash_date=requested_date, price_date=pd.Timestamp(price_date), raw_price=float(raw_px))


def _force_liquidation_fill(df: pd.DataFrame, final_day: pd.Timestamp, *, close_col: str) -> SellFill | None:
    """
    End-of-backtest liquidation helper.

    - Cash is credited on final_day (the backtest ends cleanly).
    - Price uses the last available close on/before final_day (can be stale, but avoids leaving positions open forever).
    """
    res = _last_close_on_or_before(df, pd.Timestamp(final_day), close_col=close_col)
    if res is None:
        return None
    price_date, raw_px = res
    return SellFill(cash_date=pd.Timestamp(final_day), price_date=pd.Timestamp(price_date), raw_price=float(raw_px))


def _bench_date_on_or_after(bench_dates: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    ix = bench_dates.searchsorted(pd.Timestamp(d).to_datetime64(), side="left")
    if ix < len(bench_dates):
        return pd.Timestamp(bench_dates[ix])
    return None


def _last_close_on_or_before(df: pd.DataFrame, cutoff: pd.Timestamp, *, close_col: str) -> tuple[pd.Timestamp, float] | None:
    if df.empty:
        return None
    dates = df["date"].to_numpy(dtype="datetime64[ns]")
    ix = np.searchsorted(dates, np.datetime64(cutoff), side="right") - 1
    while ix >= 0:
        d = pd.Timestamp(dates[ix])
        px = float(df.iloc[ix][close_col])
        if np.isfinite(px) and px > 0:
            return d, px
        ix -= 1
    return None


def _stopout_resume_month_idx(trigger_month_idx: int, cooldown_months: int) -> int:
    """
    Compute the first month_idx where buying may resume after a stop-out.

    Semantics:
    - cooldown_months == 0 means "resume at the next monthly select_date" (1 month bucket later),
      once liquidation has completed.
    - cooldown_months > 0 means "resume after N month buckets".
    """
    return int(trigger_month_idx + (int(cooldown_months) if int(cooldown_months) > 0 else 1))


def _reentry_dca_init_from_cash(cash_balance: float, months: int) -> tuple[float, int, float]:
    """
    Initialize a post-stopout re-entry DCA ramp.

    Returns:
    - remaining: cash that is restricted and will be released over time
    - months_left: how many releases remain
    - buy_budget: how much cash is currently permitted to be invested (starts at 0)

    Note: this is a budget constraint only; the full cash balance still counts toward equity.
    """
    m = int(months)
    if m <= 0 or not (isinstance(cash_balance, (int, float)) and np.isfinite(float(cash_balance)) and float(cash_balance) > 0):
        return 0.0, 0, float("inf")
    return float(cash_balance), m, 0.0


def _reentry_dca_release(remaining: float, months_left: int) -> tuple[float, float, int]:
    """
    Release one equal tranche from the remaining cash constraint.

    Returns (released_amount, new_remaining, new_months_left).
    """
    m = int(months_left)
    rem = float(remaining)
    if m <= 0 or not (np.isfinite(rem) and rem > 0):
        return 0.0, max(0.0, rem) if np.isfinite(rem) else 0.0, 0
    release = rem / m
    rem2 = rem - release
    m2 = m - 1
    if m2 <= 0 or rem2 < 1e-9:
        return float(release), 0.0, 0
    return float(release), float(rem2), int(m2)


def _remove_lots_for_symbol(lots: list[dict[str, Any]], symbol_key: str) -> None:
    for lot in list(lots):
        if str(lot.get("symbol_key")) == symbol_key:
            lots.remove(lot)


def _remove_lots_for_instrument(lots: list[dict[str, Any]], instrument_id: str) -> None:
    for lot in list(lots):
        if str(lot.get("instrument_id")) == instrument_id:
            lots.remove(lot)


def _pick_first_unheld_instrument(ranked: pd.DataFrame, held_instrument_ids: set[str]) -> pd.Series | None:
    for _, cand in ranked.iterrows():
        iid = str(cand.get("instrument_id", ""))
        if not iid or iid in held_instrument_ids:
            continue
        return cand
    return None


def _rebalance_dates(month_index: pd.DataFrame, entry_month: int, start_year: int, end_year: int) -> dict[int, pd.Timestamp]:
    out: dict[int, pd.Timestamp] = {}
    for y in range(start_year, end_year + 2):
        ym = f"{y:04d}-{entry_month:02d}"
        row = month_index[month_index["year_month"] == ym]
        if row.empty:
            continue
        out[y] = pd.Timestamp(row.iloc[0]["select_date"])
    return out


def _annual_allocation_whole_shares(
    ranked: pd.DataFrame,
    *,
    cash_available: float,
    commission_per_trade: float,
    slippage_bps: float,
) -> dict[str, int]:
    buys: dict[str, int] = {}
    if ranked.empty or cash_available <= 0:
        return buys

    ranked = ranked.copy()
    ranked["eff_buy_px"] = ranked["entry_open_adj"].astype("float64") * (1.0 + slippage_bps / 10_000.0)
    ranked = ranked[np.isfinite(ranked["eff_buy_px"]) & (ranked["eff_buy_px"] > 0)].reset_index(drop=True)
    if ranked.empty:
        return buys

    feasible = ranked[ranked["eff_buy_px"] + commission_per_trade <= cash_available].reset_index(drop=True)
    if feasible.empty:
        return buys

    while True:
        base_budget = math.floor(cash_available / len(feasible))
        shares = np.floor((base_budget - commission_per_trade) / feasible["eff_buy_px"].to_numpy(dtype="float64")).astype(int)
        keep = shares >= 1
        if keep.all():
            feasible = feasible.reset_index(drop=True)
            shares = shares.astype(int)
            break
        feasible = feasible.loc[keep].reset_index(drop=True)
        if feasible.empty:
            return buys

    key_col = "instrument_id" if "instrument_id" in feasible.columns else "symbol_key"
    for i, row in feasible.iterrows():
        buys[str(row[key_col])] = int(shares[i])

    cash_left = cash_available - float(
        np.sum(shares * feasible["eff_buy_px"].to_numpy(dtype="float64")) + (len(buys) * commission_per_trade)
    )

    # Redistribute remaining cash deterministically in rank order.
    progressed = True
    while progressed:
        progressed = False
        for _, row in feasible.iterrows():
            sym = str(row[key_col])
            px = float(row["eff_buy_px"])
            if cash_left >= px:
                buys[sym] += 1
                cash_left -= px
                progressed = True
    return buys


def _make_price_cache(
    symbol_series: dict[str, pd.DataFrame],
    *,
    open_col: str,
    low_col: str,
    close_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, df in symbol_series.items():
        if df.empty:
            continue
        dates = pd.to_datetime(df["date"]).to_numpy(dtype="datetime64[ns]")
        open_px = df[open_col].to_numpy(dtype="float64") if open_col in df.columns else np.full(len(df), np.nan)
        low_px = df[low_col].to_numpy(dtype="float64") if low_col in df.columns else np.full(len(df), np.nan)
        close_px = df[close_col].to_numpy(dtype="float64") if close_col in df.columns else np.full(len(df), np.nan)
        cache[sym] = (dates, open_px, low_px, close_px)
    return cache


def _implied_div_cash_per_share_from_closeadj(df: pd.DataFrame) -> np.ndarray:
    """
    Best-effort cash distribution inference from Sharadar bars.

    Sharadar exports used here do not include an explicit dividend/distribution column. We infer a per-share cash distribution proxy
    using the change in (close_adj / close_raw) between bars.

    If close_adj is a total-return adjusted close and close_raw is split-adjusted price-only close, then:
      div_factor[t] = close_adj[t] / close_raw[t]
      div_factor[t] / div_factor[t-1] - 1  ~= dividend return component on day t

    We turn that return component into a per-share cash amount proxy as:
      div_cash_per_share[t] = close_raw[t] * max(0, div_factor_ratio - 1)

    Notes:
    - This is an approximation; it depends on how close_adj is constructed by Sharadar.
    - Negative/invalid implied values are clipped to 0.
    """
    if df.empty or "close_raw" not in df.columns or "close_adj" not in df.columns:
        return np.zeros(len(df), dtype="float64")
    close_raw = pd.to_numeric(df["close_raw"], errors="coerce").to_numpy(dtype="float64")
    close_adj = pd.to_numeric(df["close_adj"], errors="coerce").to_numpy(dtype="float64")
    out = np.zeros(len(df), dtype="float64")
    ok = np.isfinite(close_raw) & (close_raw > 0) & np.isfinite(close_adj) & (close_adj > 0)
    if not ok.any():
        return out
    div_factor = np.full(len(df), np.nan, dtype="float64")
    div_factor[ok] = close_adj[ok] / close_raw[ok]
    prev = np.roll(div_factor, 1)
    prev[0] = np.nan
    ratio = div_factor / prev
    implied = close_raw * (ratio - 1.0)
    implied[~np.isfinite(implied)] = 0.0
    implied[implied < 0] = 0.0
    out[:] = implied
    return out


def _implied_div_cash_per_share_on_date(
    div_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    instrument_id: str,
    df: pd.DataFrame,
    date: pd.Timestamp,
) -> float:
    """
    Return the inferred cash distribution per share for an instrument on an exact bar date.
    """
    if instrument_id not in div_cache:
        dates = (
            pd.to_datetime(df["date"]).to_numpy(dtype="datetime64[ns]") if not df.empty else np.array([], dtype="datetime64[ns]")
        )
        div_ps = _implied_div_cash_per_share_from_closeadj(df)
        div_cache[instrument_id] = (dates, div_ps)
    dates, div_ps = div_cache[instrument_id]
    if len(dates) == 0:
        return 0.0
    d64 = np.datetime64(pd.Timestamp(date))
    ix = int(np.searchsorted(dates, d64, side="left"))
    if ix < len(dates) and dates[ix] == d64:
        v = float(div_ps[ix])
        return v if np.isfinite(v) and v > 0 else 0.0
    return 0.0


def _benchmark_buy_hold_calendar_year_returns(
    *,
    calendar: pd.DatetimeIndex,
    price_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    instrument_id: str,
) -> pd.DataFrame:
    """
    Benchmark "buy & hold" calendar-year returns (no external cash flows).

    This is a pure price series stat:
    - start = open of the first available trading day in the calendar year
    - end = close of the last available trading day in the calendar year
    - return = end / start - 1

    It intentionally does NOT use the backtest benchmark portfolio equity curve, so it is not affected by
    `monthly_dca_amount`, cash sweeps, commissions, or slippage.
    """
    cached = price_cache.get(instrument_id)
    if cached is None:
        return pd.DataFrame(columns=["year", "bm_buy_hold_return_pct"])

    dates, open_px, _low_px, close_px = cached
    cal = pd.to_datetime(calendar).to_numpy(dtype="datetime64[ns]")
    cal_year = pd.to_datetime(calendar).year.to_numpy(dtype=int)

    idx = np.searchsorted(dates, cal, side="left")
    ok = (idx >= 0) & (idx < len(dates)) & (dates[idx] == cal)
    open_aligned = np.full(len(cal), np.nan, dtype="float64")
    close_aligned = np.full(len(cal), np.nan, dtype="float64")
    open_aligned[ok] = open_px[idx[ok]]
    close_aligned[ok] = close_px[idx[ok]]

    rows: list[dict[str, Any]] = []
    for y in np.unique(cal_year):
        take = cal_year == y
        o = open_aligned[take]
        c = close_aligned[take]
        if not np.isfinite(o).any() or not np.isfinite(c).any():
            rows.append({"year": int(y), "bm_buy_hold_return_pct": np.nan})
            continue
        start_ix = int(np.argmax(np.isfinite(o)))
        end_ix = int(len(c) - 1 - np.argmax(np.isfinite(c[::-1])))
        start_open = float(o[start_ix])
        end_close = float(c[end_ix])
        ret = (end_close / start_open) - 1.0 if start_open > 0 else np.nan
        rows.append({"year": int(y), "bm_buy_hold_return_pct": ret})

    return pd.DataFrame(rows).sort_values("year", kind="mergesort").reset_index(drop=True)


def _update_price_cache_symbol(
    price_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    sym: str,
    df: pd.DataFrame,
    *,
    open_col: str,
    low_col: str,
    close_col: str,
) -> None:
    if df.empty:
        return
    dates = pd.to_datetime(df["date"]).to_numpy(dtype="datetime64[ns]")
    open_px = df[open_col].to_numpy(dtype="float64") if open_col in df.columns else np.full(len(df), np.nan)
    low_px = df[low_col].to_numpy(dtype="float64") if low_col in df.columns else np.full(len(df), np.nan)
    close_px = df[close_col].to_numpy(dtype="float64") if close_col in df.columns else np.full(len(df), np.nan)
    price_cache[sym] = (dates, open_px, low_px, close_px)


def _value_portfolio(
    *,
    date: pd.Timestamp,
    cash: float,
    holdings: dict[str, int],
    price_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    val = cash
    val_low = cash
    d64 = np.datetime64(date)
    for sym, sh in holdings.items():
        if sh == 0:
            continue
        cached = price_cache.get(sym)
        if cached is None:
            continue
        dates, _open_px, low_px, close_px = cached
        ix = np.searchsorted(dates, d64, side="right") - 1
        if ix < 0:
            continue
        c = float(close_px[ix])
        if np.isfinite(c) and c > 0:
            val += sh * c
        l = float(low_px[ix])
        used = l if np.isfinite(l) and l > 0 else c
        if np.isfinite(used) and used > 0:
            val_low += sh * used
    return val, val_low


def _run_mode_annual_symbol_key_legacy(cfg: EntryRotationConfig, out_root: Path, month_index: pd.DataFrame) -> None:
    raise RuntimeError(
        "Legacy symbol_key engine disabled. Use the permaticker-based engine (def _run_mode_annual) to avoid ticker rename/reuse issues."
    )
    rebalance_dates = _rebalance_dates(month_index, cfg.entry_month, cfg.start_year, cfg.end_year + 1)
    if cfg.start_year not in rebalance_dates:
        raise ValueError("No rebalance date for start_year; check entry_month and benchmark calendar coverage.")
    rebalance_set = set(rebalance_dates.values())

    selection_bench_symbol = _resolve_benchmark_symbol_key(
        cfg.cache_dir / "tickers", cfg.selection_benchmark_ticker, prefer_table="SFP"
    )
    selection_bench_series = _load_symbol_series_for_valuation(cfg, {selection_bench_symbol}).get(selection_bench_symbol)
    if selection_bench_series is None or selection_bench_series.empty:
        raise ValueError(f"No bars for selection benchmark {cfg.selection_benchmark_ticker}")

    bench_dates = pd.DatetimeIndex(selection_bench_series["date"])
    start_date = rebalance_dates[cfg.start_year]
    end_date, exit_rebalance_date = _annual_sim_end_date(
        bench_dates=bench_dates, rebalance_dates=rebalance_dates, end_year=cfg.end_year
    )
    calendar = pd.DatetimeIndex(bench_dates[(bench_dates >= start_date) & (bench_dates <= end_date)])

    bm_symbol = _resolve_benchmark_symbol_key(cfg.cache_dir / "tickers", cfg.benchmark_ticker, prefer_table="SFP")

    symbol_series: dict[str, pd.DataFrame] = {selection_bench_symbol: selection_bench_series}

    def ensure_series(sym: str) -> pd.DataFrame:
        if sym not in symbol_series:
            symbol_series.update(_load_symbol_series_for_valuation(cfg, {sym}))
        return symbol_series[sym]

    def next_bench_date_after(d: pd.Timestamp) -> pd.Timestamp | None:
        ix = bench_dates.searchsorted(d.to_datetime64(), side="right")
        if ix < len(bench_dates):
            return pd.Timestamp(bench_dates[ix])
        return None

    def bench_date_on_or_after(d: pd.Timestamp) -> pd.Timestamp | None:
        return _bench_date_on_or_after(bench_dates, d)

    def bench_date_on_or_after(d: pd.Timestamp) -> pd.Timestamp | None:
        return _bench_date_on_or_after(bench_dates, d)

    def schedule_delist_credit(
        pending: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp]]],
        *,
        sym: str,
        shares: int,
        planned_exit: pd.Timestamp,
        lastpricedate: pd.Timestamp | None,
    ) -> None:
        df = ensure_series(sym)
        cutoff = (
            pd.Timestamp(lastpricedate)
            if lastpricedate is not None and not pd.isna(lastpricedate)
            else pd.Timestamp(df["date"].iloc[-1])
        )
        last_trade = _last_close_on_or_before(df, cutoff, close_col=close_col)
        if last_trade is None:
            return
        last_date, raw_px = last_trade
        if last_date >= planned_exit:
            return
        credit_date = next_bench_date_after(last_date) or planned_exit
        if credit_date > end_date:
            credit_date = end_date
        eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = shares * eff_px - cfg.commission_per_trade
        pending.setdefault(credit_date, []).append((sym, shares, raw_px, eff_px, proceeds, last_date))

    # Strategy state
    cash_s = cfg.initial_equity
    holdings_s: dict[str, int] = {}
    cash_by_day_s = pd.Series(index=calendar, dtype="float64")
    holdings_by_day_s: dict[pd.Timestamp, dict[str, int]] = {}
    pending_s: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp]]] = {}

    # Benchmark state
    cash_b = cfg.initial_equity
    holdings_b: dict[str, int] = {}
    cash_by_day_b = pd.Series(index=calendar, dtype="float64")
    holdings_by_day_b: dict[pd.Timestamp, dict[str, int]] = {}
    pending_b: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp]]] = {}

    trades: list[Trade] = []
    selected_rows: list[pd.DataFrame] = []

    ensure_series(bm_symbol)
    price_cache = _make_price_cache(symbol_series)

    strategy_equity = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low_pre_actions = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low_pre_actions = np.full(len(calendar), np.nan, dtype="float64")

    for i, d in enumerate(calendar):
        # Apply pending delisting liquidations (cash settles on credit day).
        for sym, shares, raw_px, eff_px, proceeds, price_date in pending_s.pop(d, []):
            if holdings_s.get(sym, 0) >= shares:
                holdings_s[sym] -= shares
                if holdings_s[sym] <= 0:
                    holdings_s.pop(sym, None)
                cash_s += proceeds
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="strategy",
                        side="SELL",
                        symbol_key=sym,
                        ticker=sym.split(":", 1)[1],
                        table=sym.split(":", 1)[0],
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_s,
                    )
                )
        for sym, shares, raw_px, eff_px, proceeds, price_date in pending_b.pop(d, []):
            if holdings_b.get(sym, 0) >= shares:
                holdings_b[sym] -= shares
                if holdings_b[sym] <= 0:
                    holdings_b.pop(sym, None)
                cash_b += proceeds
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="benchmark",
                        side="SELL",
                        symbol_key=sym,
                        ticker=sym.split(":", 1)[1],
                        table=sym.split(":", 1)[0],
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_b,
                    )
                )

        if d in rebalance_set:
            is_exit_rebalance = exit_rebalance_date is not None and d == exit_rebalance_date
            # Sell then buy (sell-first ordering).
            if d.year != cfg.start_year:
                for sym, sh in list(holdings_s.items()):
                    sell = _sell_price_with_fallback(ensure_series(sym), d)
                    if sell is None:
                        continue
                    sell_date, raw_px = sell
                    eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
                    proceeds = sh * eff_px - cfg.commission_per_trade
                    if sell_date != d:
                        pending_s.setdefault(sell_date, []).append((sym, sh, raw_px, eff_px, proceeds, sell_date))
                    else:
                        cash_s += proceeds
                        holdings_s.pop(sym, None)
                        trades.append(
                            Trade(
                                date=sell_date,
                                price_date=sell_date,
                                portfolio="strategy",
                                side="SELL",
                                symbol_key=sym,
                                ticker=sym.split(":", 1)[1],
                                table=sym.split(":", 1)[0],
                                shares=sh,
                                raw_price=raw_px,
                                effective_price=eff_px,
                                commission=cfg.commission_per_trade,
                                slippage_bps=cfg.slippage_bps,
                                notional=sh * eff_px,
                                cash_after=cash_s,
                            )
                        )

                for sym, sh in list(holdings_b.items()):
                    sell = _sell_price_with_fallback(ensure_series(sym), d)
                    if sell is None:
                        continue
                    sell_date, raw_px = sell
                    eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
                    proceeds = sh * eff_px - cfg.commission_per_trade
                    if sell_date != d:
                        pending_b.setdefault(sell_date, []).append((sym, sh, raw_px, eff_px, proceeds, sell_date))
                    else:
                        cash_b += proceeds
                        holdings_b.pop(sym, None)
                        trades.append(
                            Trade(
                                date=sell_date,
                                price_date=sell_date,
                                portfolio="benchmark",
                                side="SELL",
                                symbol_key=sym,
                                ticker=sym.split(":", 1)[1],
                                table=sym.split(":", 1)[0],
                                shares=sh,
                                raw_price=raw_px,
                                effective_price=eff_px,
                                commission=cfg.commission_per_trade,
                                slippage_bps=cfg.slippage_bps,
                                notional=sh * eff_px,
                                cash_after=cash_b,
                            )
                        )

            if not is_exit_rebalance:
                candidates = load_candidates_for_select_date(cfg.cache_dir / "candidates_monthly", d)
                ranked = _select_top_n(candidates, cfg)
                if not ranked.empty:
                    ranked = ranked.copy()
                    ranked["selected_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
                    ranked["rebalance_date"] = d
                    selected_rows.append(ranked)

                cash_avail_s = cash_s * (1.0 - cfg.cash_buffer_pct)
                buy_plan = _annual_allocation_whole_shares(
                    ranked,
                    cash_available=cash_avail_s,
                    commission_per_trade=cfg.commission_per_trade,
                    slippage_bps=cfg.slippage_bps,
                )
                next_reb = rebalance_dates.get(int(d.year) + 1, end_date)
                for sym, sh in buy_plan.items():
                    px = float(ranked.loc[ranked["symbol_key"] == sym, "entry_open_adj"].iloc[0])
                    lastpricedate = (
                        ranked.loc[ranked["symbol_key"] == sym, "lastpricedate"].iloc[0]
                        if "lastpricedate" in ranked.columns
                        else pd.NaT
                    )
                    eff_px = _apply_slippage(px, side="BUY", slippage_bps=cfg.slippage_bps)
                    cost = sh * eff_px + cfg.commission_per_trade
                    if cost > cash_s:
                        continue
                    cash_s -= cost
                    holdings_s[sym] = holdings_s.get(sym, 0) + sh
                    df_sym = ensure_series(sym)
                    _update_price_cache_symbol(price_cache, sym, df_sym)
                    trades.append(
                        Trade(
                            date=d,
                            portfolio="strategy",
                            side="BUY",
                            symbol_key=sym,
                            ticker=sym.split(":", 1)[1],
                            table=sym.split(":", 1)[0],
                            shares=sh,
                            raw_price=px,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_s,
                        )
                    )
                    schedule_delist_credit(pending_s, sym=sym, shares=sh, planned_exit=next_reb, lastpricedate=lastpricedate)

                # Benchmark: buy 100% benchmark_ticker.
                bm_px = _price_at(ensure_series(bm_symbol), d, "open_adj")
                if bm_px is not None:
                    cash_avail_b = cash_b * (1.0 - cfg.cash_buffer_pct)
                    bm_eff = _apply_slippage(bm_px, side="BUY", slippage_bps=cfg.slippage_bps)
                    bm_sh = int(math.floor((cash_avail_b - cfg.commission_per_trade) / bm_eff))
                    if bm_sh >= 1:
                        bm_cost = bm_sh * bm_eff + cfg.commission_per_trade
                        if bm_cost <= cash_b:
                            cash_b -= bm_cost
                            holdings_b[bm_symbol] = holdings_b.get(bm_symbol, 0) + bm_sh
                            trades.append(
                                Trade(
                                    date=d,
                                    portfolio="benchmark",
                                    side="BUY",
                                    symbol_key=bm_symbol,
                                    ticker=bm_symbol.split(":", 1)[1],
                                    table=bm_symbol.split(":", 1)[0],
                                    shares=bm_sh,
                                    raw_price=bm_px,
                                    effective_price=bm_eff,
                                    commission=cfg.commission_per_trade,
                                    slippage_bps=cfg.slippage_bps,
                                    notional=bm_sh * bm_eff,
                                    cash_after=cash_b,
                                )
                            )

        s_val, s_low = _value_portfolio(date=d, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
        b_val, b_low = _value_portfolio(date=d, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
        strategy_equity[i] = s_val
        strategy_equity_low[i] = s_low
        bm_equity[i] = b_val
        bm_equity_low[i] = b_low

    # End-of-backtest liquidation (never sell after final benchmark day).
    final_day = calendar[-1]
    for sym, sh in list(holdings_s.items()):
        sell = _sell_price_no_forward(ensure_series(sym), final_day)
        if sell is None:
            continue
        sell_date, raw_px = sell
        eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = sh * eff_px - cfg.commission_per_trade
        cash_s += proceeds
        holdings_s.pop(sym, None)
        trades.append(
            Trade(
                date=sell_date,
                price_date=sell_date,
                portfolio="strategy",
                side="SELL",
                symbol_key=sym,
                ticker=sym.split(":", 1)[1],
                table=sym.split(":", 1)[0],
                shares=sh,
                raw_price=raw_px,
                effective_price=eff_px,
                commission=cfg.commission_per_trade,
                slippage_bps=cfg.slippage_bps,
                notional=sh * eff_px,
                cash_after=cash_s,
            )
        )
    for sym, sh in list(holdings_b.items()):
        sell = _sell_price_no_forward(ensure_series(sym), final_day)
        if sell is None:
            continue
        sell_date, raw_px = sell
        eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = sh * eff_px - cfg.commission_per_trade
        cash_b += proceeds
        holdings_b.pop(sym, None)
        trades.append(
            Trade(
                date=sell_date,
                price_date=sell_date,
                portfolio="benchmark",
                side="SELL",
                symbol_key=sym,
                ticker=sym.split(":", 1)[1],
                table=sym.split(":", 1)[0],
                shares=sh,
                raw_price=raw_px,
                effective_price=eff_px,
                commission=cfg.commission_per_trade,
                slippage_bps=cfg.slippage_bps,
                notional=sh * eff_px,
                cash_after=cash_b,
            )
        )
    # Re-value final day after liquidation.
    price_cache = _make_price_cache(symbol_series)
    s_val, s_low = _value_portfolio(date=final_day, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
    b_val, b_low = _value_portfolio(date=final_day, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
    strategy_equity[-1] = s_val
    strategy_equity_low[-1] = s_low
    bm_equity[-1] = b_val
    bm_equity_low[-1] = b_low

    equity_df = pd.DataFrame(
        {
            "date": calendar,
            "equity": strategy_equity,
            "equity_low": strategy_equity_low,
            "equity_low_pre_actions": strategy_equity_low_pre_actions,
        }
    )
    if not cfg.no_benchmark:
        equity_df["bm_equity"] = bm_equity
        equity_df["bm_equity_low"] = bm_equity_low
        equity_df["bm_equity_low_pre_actions"] = bm_equity_low_pre_actions

    pq.write_table(pa.Table.from_pandas(equity_df, preserve_index=False), out_root / "equity_curve.parquet", compression="zstd")
    if trades:
        pd.DataFrame([dataclasses.asdict(t) for t in trades]).to_csv(out_root / "trades.csv", index=False)
    if selected_rows:
        pq.write_table(
            pa.Table.from_pandas(pd.concat(selected_rows, ignore_index=True), preserve_index=False),
            out_root / "selected.parquet",
            compression="zstd",
        )

    periods = _rebalance_periods_stats(equity_df, rebalance_dates, start_year=cfg.start_year, end_year=cfg.end_year)
    periods.to_csv(out_root / "rebalance_periods.csv", index=False)
    _calendar_yearly_stats(equity_df).to_csv(out_root / "calendar_yearly_equity.csv", index=False)


def _run_mode_annual(cfg: EntryRotationConfig, out_root: Path, month_index: pd.DataFrame) -> None:
    """
    Annual (rolling-year) rebalance engine keyed by permaticker instrument_id.

    - Holds are tracked by `instrument_id` (PT<permaticker>) so ticker renames do not create duplicate exposure.
    - Selection + pricing uses the permaticker-stitched bar series.
    - Sells follow `cfg.sell_price_policy`, but end-of-backtest liquidation never looks forward.
    """
    rebalance_dates = _rebalance_dates(month_index, cfg.entry_month, cfg.start_year, cfg.end_year + 1)
    if cfg.start_year not in rebalance_dates:
        raise ValueError("No rebalance date for start_year; check entry_month and benchmark calendar coverage.")
    rebalance_set = set(rebalance_dates.values())

    selection_bench_symbol = _resolve_benchmark_symbol_key(
        cfg.cache_dir / "tickers", cfg.selection_benchmark_ticker, prefer_table="SFP"
    )
    selection_bench_series = _load_symbol_series_for_valuation(cfg, {selection_bench_symbol}).get(selection_bench_symbol)
    if selection_bench_series is None or selection_bench_series.empty:
        raise ValueError(f"No bars for selection benchmark {cfg.selection_benchmark_ticker}")
    open_col, low_col, close_col = _price_cols_for_mode(cfg.price_mode)
    if cfg.price_mode == "raw":
        missing = [c for c in (open_col, low_col, close_col) if c not in selection_bench_series.columns]
        if missing:
            raise ValueError(
                f"price_mode=raw requires raw OHLC in the bars cache. Rebuild caches with --rebuild-cache. Missing: {missing}"
            )

    bench_dates = pd.DatetimeIndex(selection_bench_series["date"])
    start_date = rebalance_dates[cfg.start_year]
    end_date, exit_rebalance_date = _annual_sim_end_date(
        bench_dates=bench_dates, rebalance_dates=rebalance_dates, end_year=cfg.end_year
    )
    calendar = pd.DatetimeIndex(bench_dates[(bench_dates >= start_date) & (bench_dates <= end_date)])
    contrib_dates = set(
        pd.Timestamp(x) for x in _annual_contribution_dates(month_index, start_date=start_date, end_date=end_date)
    )

    candidates_dir = cfg.cache_dir / "candidates_monthly"
    sell_policy = cfg.sell_price_policy

    bm_symbol_key = _resolve_benchmark_symbol_key(cfg.cache_dir / "tickers", cfg.benchmark_ticker, prefer_table="SFP")
    sym_to_perm = _symbol_key_to_permaticker_map(cfg.cache_dir / "tickers")
    bm_perm = sym_to_perm.get(bm_symbol_key)
    if bm_perm is None:
        raise ValueError(f"Benchmark ticker missing permaticker mapping: {bm_symbol_key}")
    bm_instrument_id = f"PT{bm_perm}"

    instrument_series: dict[str, pd.DataFrame] = {}

    def ensure_series(instrument_id: str) -> pd.DataFrame:
        if instrument_id not in instrument_series:
            instrument_series.update(_load_series_for_instrument_ids(cfg, {instrument_id}))
        return instrument_series[instrument_id]

    def next_bench_date_after(d: pd.Timestamp) -> pd.Timestamp | None:
        ix = bench_dates.searchsorted(d.to_datetime64(), side="right")
        if ix < len(bench_dates):
            return pd.Timestamp(bench_dates[ix])
        return None

    cash_sweep_enabled = bool(cfg.cash_sweep_enabled) and bool(str(cfg.cash_sweep_ticker or "").strip())
    cash_sweep_symbol_key: str | None = None
    cash_sweep_instrument_id: str | None = None
    if cash_sweep_enabled:
        try:
            cash_sweep_symbol_key = _resolve_benchmark_symbol_key(
                cfg.cache_dir / "tickers", str(cfg.cash_sweep_ticker).strip().upper(), prefer_table="SFP"
            )
            sweep_perm = sym_to_perm.get(cash_sweep_symbol_key)
            if sweep_perm is None:
                raise ValueError(f"Cash sweep ticker missing permaticker mapping: {cash_sweep_symbol_key}")
            cash_sweep_instrument_id = f"PT{sweep_perm}"
            ensure_series(cash_sweep_instrument_id)
        except Exception as e:
            logger.warning("Disabling cash sweep: unable to resolve %r (%s)", cfg.cash_sweep_ticker, e)
            cash_sweep_enabled = False
            cash_sweep_symbol_key = None
            cash_sweep_instrument_id = None

    def schedule_delist_credit(
        pending: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp, str]]],
        *,
        instrument_id: str,
        shares: int,
        planned_exit: pd.Timestamp,
        lastpricedate: pd.Timestamp | None,
        label_symbol_key: str,
    ) -> None:
        df = ensure_series(instrument_id)
        cutoff = (
            pd.Timestamp(lastpricedate)
            if lastpricedate is not None and not pd.isna(lastpricedate)
            else pd.Timestamp(df["date"].iloc[-1])
        )
        last_trade = _last_close_on_or_before(df, cutoff, close_col=close_col)
        if last_trade is None:
            return
        last_date, raw_px = last_trade
        if last_date >= planned_exit:
            return
        credit_date = next_bench_date_after(last_date) or planned_exit
        if credit_date > end_date:
            credit_date = end_date
        eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = shares * eff_px - cfg.commission_per_trade
        pending.setdefault(credit_date, []).append((instrument_id, shares, raw_px, eff_px, proceeds, last_date, label_symbol_key))
        pending_sell_ids_s.add(instrument_id)

    cash_s = cfg.initial_equity
    holdings_s: dict[str, int] = {}
    labels_s: dict[str, str] = {}
    entry_eff_s: dict[str, float] = {}
    entry_date_s: dict[str, pd.Timestamp] = {}
    pending_s: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp, str]]] = {}
    pending_sell_ids_s: set[str] = set()
    pending_buy_s: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    pending_buy_ids_s: set[str] = set()

    cash_b = cfg.initial_equity
    holdings_b: dict[str, int] = {}
    labels_b: dict[str, str] = {bm_instrument_id: bm_symbol_key}
    pending_b: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp, str]]] = {}

    trades: list[Trade] = []
    selected_rows: list[pd.DataFrame] = []
    position_stopouts: list[dict[str, Any]] = []
    cash_flows: list[dict[str, Any]] = []

    ensure_series(bm_instrument_id)
    price_cache = _make_price_cache(instrument_series, open_col=open_col, low_col=low_col, close_col=close_col)

    strategy_equity = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low_pre_actions = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low_pre_actions = np.full(len(calendar), np.nan, dtype="float64")
    strategy_div_gross = np.zeros(len(calendar), dtype="float64")
    strategy_div_tax = np.zeros(len(calendar), dtype="float64")
    strategy_div_net = np.zeros(len(calendar), dtype="float64")
    bm_div_gross = np.zeros(len(calendar), dtype="float64")
    bm_div_tax = np.zeros(len(calendar), dtype="float64")
    bm_div_net = np.zeros(len(calendar), dtype="float64")
    div_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    withholding_rate = float(cfg.dividend_withholding_rate)

    period_id_by_rebalance_date = {pd.Timestamp(d): int(y) for y, d in rebalance_dates.items()}
    active_period_id: int | None = None
    period_start_equity: float | None = None
    stopped_out = False
    stopout_by_period: dict[int, dict[str, Any]] = {}

    def total_pending_buy_orders() -> int:
        return int(sum(len(v) for v in pending_buy_s.values()))

    def cancel_pending_sell(iid: str) -> None:
        if iid not in pending_sell_ids_s:
            return
        for k in list(pending_s.keys()):
            pending_s[k] = [t for t in pending_s[k] if str(t[0]) != iid]
            if not pending_s[k]:
                pending_s.pop(k, None)
        pending_sell_ids_s.discard(iid)

    def execute_buy_orders_for_day(d: pd.Timestamp) -> None:
        nonlocal cash_s
        orders = pending_buy_s.pop(d, [])
        if not orders:
            return
        orders = sorted(orders, key=lambda o: int(o.get("rank", 2**31 - 1)))
        pending_total = len(orders) + total_pending_buy_orders()
        for order in orders:
            iid = str(order.get("instrument_id", ""))
            if not iid:
                continue
            if holdings_s.get(iid, 0) > 0:
                pending_buy_ids_s.discard(iid)
                pending_total -= 1
                continue
            df_iid = ensure_series(iid)
            px = _price_at(df_iid, d, open_col)
            if px is None:
                px = _price_at(df_iid, d, close_col)
            if px is None:
                nxt = next_bench_date_after(d)
                if nxt is not None and nxt <= calendar[-1]:
                    pending_buy_s.setdefault(nxt, []).append(order)
                else:
                    pending_buy_ids_s.discard(iid)
                    pending_total -= 1
                continue

            if pending_total <= 0:
                pending_buy_ids_s.discard(iid)
                continue

            spendable = cash_s * (1.0 - cfg.cash_buffer_pct)
            base_budget = math.floor(spendable / pending_total)
            eff_px = _apply_slippage(float(px), side="BUY", slippage_bps=cfg.slippage_bps)
            shares = int(math.floor((base_budget - cfg.commission_per_trade) / eff_px))
            if shares < 1:
                pending_buy_ids_s.discard(iid)
                pending_total -= 1
                continue

            cost = shares * eff_px + cfg.commission_per_trade
            if cost > cash_s:
                shares = int(math.floor((cash_s - cfg.commission_per_trade) / eff_px))
                if shares < 1:
                    pending_buy_ids_s.discard(iid)
                    pending_total -= 1
                    continue
                cost = shares * eff_px + cfg.commission_per_trade
                if cost > cash_s:
                    pending_buy_ids_s.discard(iid)
                    pending_total -= 1
                    continue

            label = str(order.get("symbol_key", labels_s.get(iid, "")))
            cash_s -= cost
            holdings_s[iid] = holdings_s.get(iid, 0) + shares
            labels_s[iid] = label
            entry_eff_s[iid] = float(eff_px)
            entry_date_s[iid] = pd.Timestamp(d)
            pending_buy_ids_s.discard(iid)
            pending_total -= 1
            _update_price_cache_symbol(price_cache, iid, df_iid, open_col=open_col, low_col=low_col, close_col=close_col)
            table, ticker = label.split(":", 1) if ":" in label else ("", "")
            trades.append(
                Trade(
                    date=d,
                    price_date=d,
                    portfolio="strategy",
                    side="BUY",
                    instrument_id=iid,
                    symbol_key=label,
                    ticker=ticker,
                    table=table,
                    shares=shares,
                    raw_price=float(px),
                    effective_price=eff_px,
                    commission=cfg.commission_per_trade,
                    slippage_bps=cfg.slippage_bps,
                    notional=shares * eff_px,
                    cash_after=cash_s,
                )
            )

    def maybe_cash_sweep(
        *,
        date: pd.Timestamp,
        portfolio: Literal["strategy", "benchmark"],
    ) -> None:
        nonlocal cash_s, cash_b
        if not cash_sweep_enabled or cash_sweep_instrument_id is None or cash_sweep_symbol_key is None:
            return
        if date >= calendar[-1]:
            return
        # In dividend-withholding mode, we also sweep on rebalance dates so dividend cash doesn't sit idle.
        if withholding_rate <= 0 and date in rebalance_set:
            return
        if portfolio == "strategy":
            if not stopped_out and (pending_buy_s or pending_buy_ids_s):
                return
            cash = float(cash_s)
        else:
            cash = float(cash_b)
        df_sweep = ensure_series(cash_sweep_instrument_id)
        # Dividend-withholding semantics: sweep at same-day close if possible, else effectively defer-forward (we'll try again next day).
        # Otherwise, keep the legacy open_else_close behavior.
        if withholding_rate > 0:
            px = _price_at(df_sweep, date, close_col)
        else:
            px = _price_at(df_sweep, date, open_col)
            if px is None:
                px = _price_at(df_sweep, date, close_col)
        if px is None:
            return
        eff_px = _apply_slippage(float(px), side="BUY", slippage_bps=cfg.slippage_bps)
        shares = _whole_shares_for_spendable_cash(
            cash=cash,
            effective_price=eff_px,
            commission=float(cfg.commission_per_trade),
            cash_buffer_pct=float(cfg.cash_buffer_pct),
        )
        if shares < 1:
            return
        cost = shares * eff_px + float(cfg.commission_per_trade)
        if cost > cash:
            return

        if portfolio == "strategy":
            cash_s = float(cash_s) - cost
            holdings_s[cash_sweep_instrument_id] = holdings_s.get(cash_sweep_instrument_id, 0) + int(shares)
            labels_s[cash_sweep_instrument_id] = cash_sweep_symbol_key
            if cash_sweep_instrument_id not in entry_eff_s:
                entry_eff_s[cash_sweep_instrument_id] = float(eff_px)
                entry_date_s[cash_sweep_instrument_id] = pd.Timestamp(date)
            _update_price_cache_symbol(
                price_cache, cash_sweep_instrument_id, df_sweep, open_col=open_col, low_col=low_col, close_col=close_col
            )
            table, ticker = cash_sweep_symbol_key.split(":", 1)
            trades.append(
                Trade(
                    date=date,
                    price_date=date,
                    portfolio="strategy",
                    side="BUY",
                    instrument_id=cash_sweep_instrument_id,
                    symbol_key=cash_sweep_symbol_key,
                    ticker=ticker,
                    table=table,
                    shares=int(shares),
                    raw_price=float(px),
                    effective_price=float(eff_px),
                    commission=cfg.commission_per_trade,
                    slippage_bps=cfg.slippage_bps,
                    notional=float(shares) * float(eff_px),
                    cash_after=float(cash_s),
                )
            )
        else:
            cash_b = float(cash_b) - cost
            holdings_b[cash_sweep_instrument_id] = holdings_b.get(cash_sweep_instrument_id, 0) + int(shares)
            labels_b[cash_sweep_instrument_id] = cash_sweep_symbol_key
            _update_price_cache_symbol(
                price_cache, cash_sweep_instrument_id, df_sweep, open_col=open_col, low_col=low_col, close_col=close_col
            )
            table, ticker = cash_sweep_symbol_key.split(":", 1)
            trades.append(
                Trade(
                    date=date,
                    price_date=date,
                    portfolio="benchmark",
                    side="BUY",
                    instrument_id=cash_sweep_instrument_id,
                    symbol_key=cash_sweep_symbol_key,
                    ticker=ticker,
                    table=table,
                    shares=int(shares),
                    raw_price=float(px),
                    effective_price=float(eff_px),
                    commission=cfg.commission_per_trade,
                    slippage_bps=cfg.slippage_bps,
                    notional=float(shares) * float(eff_px),
                    cash_after=float(cash_b),
                )
            )

    for i, d in enumerate(calendar):
        if d in period_id_by_rebalance_date:
            active_period_id = int(period_id_by_rebalance_date[d])
            period_start_equity = None
            stopped_out = False
            stopout_by_period.setdefault(
                active_period_id,
                {"period_id": active_period_id, "stopped_out": False, "stopout_date": pd.NaT, "stopout_mae_pct": np.nan},
            )

        for iid, shares, raw_px, eff_px, proceeds, price_date, label in pending_s.pop(d, []):
            if holdings_s.get(iid, 0) >= shares:
                holdings_s[iid] -= shares
                if holdings_s[iid] <= 0:
                    holdings_s.pop(iid, None)
                    labels_s.pop(iid, None)
                    entry_eff_s.pop(iid, None)
                    entry_date_s.pop(iid, None)
                    pending_sell_ids_s.discard(iid)
                cash_s += proceeds
                table, ticker = label.split(":", 1)
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="strategy",
                        side="SELL",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_s,
                    )
                )
        for iid, shares, raw_px, eff_px, proceeds, price_date, label in pending_b.pop(d, []):
            if holdings_b.get(iid, 0) >= shares:
                holdings_b[iid] -= shares
                if holdings_b[iid] <= 0:
                    holdings_b.pop(iid, None)
                    labels_b.pop(iid, None)
                cash_b += proceeds
                table, ticker = label.split(":", 1)
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="benchmark",
                        side="SELL",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_b,
                    )
                )

        # Monthly contributions for annual-lumpsum land on the benchmark month schedule.
        #
        # Ordering:
        # - On rebalance days: sell first (handled below), then contribution, then buy.
        # - On non-rebalance days: contribution happens before any pending buys execute.
        if cfg.monthly_dca_amount > 0 and d in contrib_dates and d not in rebalance_set:
            cash_s += float(cfg.monthly_dca_amount)
            cash_b += float(cfg.monthly_dca_amount)
            cash_flows.append({"date": pd.Timestamp(d), "amount": -float(cfg.monthly_dca_amount)})

        if d in rebalance_set:
            is_exit_rebalance = exit_rebalance_date is not None and d == exit_rebalance_date

            # Sell everything first.
            can_reinvest = True
            for iid, sh in list(holdings_s.items()):
                cancel_pending_sell(iid)
                label = labels_s.get(iid, "")
                fill = _resolve_sell_fill(
                    ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=calendar[-1]
                )
                if fill is None:
                    if sell_policy == "no-forward":
                        can_reinvest = False
                    continue
                eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                if fill.cash_date != d:
                    cash_date = bench_date_on_or_after(fill.cash_date) or calendar[-1]
                    pending_s.setdefault(cash_date, []).append(
                        (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                    )
                    pending_sell_ids_s.add(iid)
                    if sell_policy == "no-forward":
                        can_reinvest = False
                else:
                    cash_s += proceeds
                    holdings_s.pop(iid, None)
                    labels_s.pop(iid, None)
                    entry_eff_s.pop(iid, None)
                    entry_date_s.pop(iid, None)
                    pending_sell_ids_s.discard(iid)
                    table, ticker = label.split(":", 1) if ":" in label else ("", "")
                    trades.append(
                        Trade(
                            date=d,
                            price_date=fill.price_date,
                            portfolio="strategy",
                            side="SELL",
                            instrument_id=iid,
                            symbol_key=label,
                            ticker=ticker,
                            table=table,
                            shares=sh,
                            raw_price=fill.raw_price,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_s,
                        )
                    )

            for iid, sh in list(holdings_b.items()):
                label = labels_b.get(iid, bm_symbol_key)
                fill = _resolve_sell_fill(
                    ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=calendar[-1]
                )
                if fill is None:
                    continue
                eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                if fill.cash_date != d:
                    cash_date = bench_date_on_or_after(fill.cash_date) or calendar[-1]
                    pending_b.setdefault(cash_date, []).append(
                        (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                    )
                else:
                    cash_b += proceeds
                    holdings_b.pop(iid, None)
                    labels_b.pop(iid, None)
                    table, ticker = label.split(":", 1)
                    trades.append(
                        Trade(
                            date=d,
                            price_date=fill.price_date,
                            portfolio="benchmark",
                            side="SELL",
                            instrument_id=iid,
                            symbol_key=label,
                            ticker=ticker,
                            table=table,
                            shares=sh,
                            raw_price=fill.raw_price,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_b,
                        )
                    )

            # Contribution after sells, before buys.
            if cfg.monthly_dca_amount > 0 and d in contrib_dates:
                cash_s += float(cfg.monthly_dca_amount)
                cash_b += float(cfg.monthly_dca_amount)
                cash_flows.append({"date": pd.Timestamp(d), "amount": -float(cfg.monthly_dca_amount)})

            sell_blocked = sell_policy == "no-forward" and not can_reinvest
            if sell_blocked and cfg.allow_overlap_on_sell_block:
                sell_blocked = False

            if not is_exit_rebalance and not sell_blocked:
                candidates = load_candidates_for_select_date(candidates_dir, d)
                ranked = _select_top_n(candidates, cfg)
                if not ranked.empty:
                    ranked = ranked.copy()
                    ranked["selected_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
                    ranked["rebalance_date"] = d
                    selected_rows.append(ranked)

                for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                    iid = str(row.get("instrument_id", ""))
                    if not iid or iid in holdings_s or iid in pending_buy_ids_s:
                        continue
                    label = str(row.get("symbol_key", labels_s.get(iid, "")))
                    raw_buy_date = pd.Timestamp(row["entry_date"]) if "entry_date" in row and pd.notna(row["entry_date"]) else d
                    if raw_buy_date < d:
                        raw_buy_date = d
                    buy_date = _bench_date_on_or_after(bench_dates, raw_buy_date)
                    if buy_date is None or buy_date > calendar[-1]:
                        continue
                    pending_buy_s.setdefault(buy_date, []).append({"instrument_id": iid, "symbol_key": label, "rank": rank})
                    pending_buy_ids_s.add(iid)

                bm_px = _price_at(ensure_series(bm_instrument_id), d, open_col)
                if bm_px is not None and bm_instrument_id not in holdings_b:
                    cash_avail_b = cash_b * (1.0 - cfg.cash_buffer_pct)
                    bm_eff = _apply_slippage(bm_px, side="BUY", slippage_bps=cfg.slippage_bps)
                    bm_sh = int(math.floor((cash_avail_b - cfg.commission_per_trade) / bm_eff))
                    if bm_sh >= 1:
                        bm_cost = bm_sh * bm_eff + cfg.commission_per_trade
                        if bm_cost <= cash_b:
                            cash_b -= bm_cost
                            holdings_b[bm_instrument_id] = holdings_b.get(bm_instrument_id, 0) + bm_sh
                            labels_b[bm_instrument_id] = bm_symbol_key
                            trades.append(
                                Trade(
                                    date=d,
                                    price_date=d,
                                    portfolio="benchmark",
                                    side="BUY",
                                    instrument_id=bm_instrument_id,
                                    symbol_key=bm_symbol_key,
                                    ticker=bm_symbol_key.split(":", 1)[1],
                                    table=bm_symbol_key.split(":", 1)[0],
                                    shares=bm_sh,
                                    raw_price=bm_px,
                                    effective_price=bm_eff,
                                    commission=cfg.commission_per_trade,
                                    slippage_bps=cfg.slippage_bps,
                                    notional=bm_sh * bm_eff,
                                    cash_after=cash_b,
                                )
                            )

        execute_buy_orders_for_day(d)

        # Pre-actions equity_low snapshot: measured before any MAE-triggered liquidations (portfolio or position) and before cash sweep.
        # This captures "how bad it got before protective actions could change exposure".
        _s_val_pre, s_low_pre = _value_portfolio(date=d, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
        strategy_equity_low_pre_actions[i] = float(s_low_pre)
        if not cfg.no_benchmark:
            _b_val_pre, b_low_pre = _value_portfolio(date=d, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
            bm_equity_low_pre_actions[i] = float(b_low_pre)

        # Annual MAE stop: if intraday equity_low breaches threshold vs this period's start equity, liquidate and stay in cash until next rebalance.
        if (
            cfg.mae_stop_pct > 0
            and not stopped_out
            and active_period_id is not None
            and period_start_equity is not None
            and period_start_equity > 0
            and d not in rebalance_set
            and holdings_s
        ):
            mae_pct = (float(s_low_pre) / float(period_start_equity)) - 1.0
            if np.isfinite(mae_pct) and mae_pct <= -float(cfg.mae_stop_pct):
                stopped_out = True
                stopout_by_period[active_period_id] = {
                    "period_id": active_period_id,
                    "stopped_out": True,
                    "stopout_date": pd.Timestamp(d),
                    "stopout_mae_pct": float(mae_pct),
                }
                pending_buy_s.clear()
                pending_buy_ids_s.clear()
                pending_s.clear()
                pending_sell_ids_s.clear()

                for iid, sh in list(holdings_s.items()):
                    label = labels_s.get(iid, "")
                    fill = _resolve_sell_fill(
                        ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=calendar[-1]
                    )
                    if fill is None:
                        continue
                    eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                    proceeds = sh * eff_px - cfg.commission_per_trade
                    if fill.cash_date != d:
                        cash_date = _bench_date_on_or_after(calendar, fill.cash_date) or calendar[-1]
                        pending_s.setdefault(cash_date, []).append(
                            (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                        )
                        pending_sell_ids_s.add(iid)
                    else:
                        cash_s += proceeds
                        holdings_s.pop(iid, None)
                        labels_s.pop(iid, None)
                        entry_eff_s.pop(iid, None)
                        entry_date_s.pop(iid, None)
                        pending_sell_ids_s.discard(iid)
                        table, ticker = label.split(":", 1) if ":" in label else ("", "")
                        trades.append(
                            Trade(
                                date=d,
                                price_date=fill.price_date,
                                portfolio="strategy",
                                side="SELL",
                                instrument_id=iid,
                                symbol_key=label,
                                ticker=ticker,
                                table=table,
                                shares=sh,
                                raw_price=fill.raw_price,
                                effective_price=eff_px,
                                commission=cfg.commission_per_trade,
                                slippage_bps=cfg.slippage_bps,
                                notional=sh * eff_px,
                                cash_after=cash_s,
                            )
                        )

        # Position MAE stop: if a position's low breaches threshold vs its entry effective price, liquidate that position and hold cash until next rebalance.
        if cfg.position_mae_stop_pct > 0 and not stopped_out and d not in rebalance_set and holdings_s:
            to_stop: list[tuple[str, int, float]] = []
            for iid, sh in holdings_s.items():
                if sh <= 0 or iid in pending_sell_ids_s:
                    continue
                entry_eff = float(entry_eff_s.get(iid, np.nan))
                mae_pct = _position_mae_pct_from_cache(
                    price_cache=price_cache,
                    instrument_id=iid,
                    date=d,
                    entry_effective_price=entry_eff,
                )
                if mae_pct is None:
                    continue
                if np.isfinite(mae_pct) and mae_pct <= -float(cfg.position_mae_stop_pct):
                    to_stop.append((iid, int(sh), float(mae_pct)))

            for iid, sh, mae_pct in to_stop:
                label = labels_s.get(iid, "")
                fill = _resolve_sell_fill(
                    ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=calendar[-1]
                )
                if fill is None:
                    continue
                eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                cash_date = bench_date_on_or_after(fill.cash_date) or calendar[-1]
                position_stopouts.append(
                    {
                        "instrument_id": iid,
                        "symbol_key": label,
                        "ticker": label.split(":", 1)[1] if ":" in label else "",
                        "table": label.split(":", 1)[0] if ":" in label else "",
                        "entry_date": entry_date_s.get(iid, pd.NaT),
                        "entry_effective_price": float(entry_eff_s.get(iid, np.nan)),
                        "stop_trigger_date": pd.Timestamp(d),
                        "stop_trigger_mae_pct": float(mae_pct),
                        "shares": int(sh),
                        "fill_price_date": pd.Timestamp(fill.price_date),
                        "fill_cash_date": pd.Timestamp(cash_date),
                        "fill_raw_price": float(fill.raw_price),
                        "fill_effective_price": float(eff_px),
                        "commission_per_trade": float(cfg.commission_per_trade),
                        "slippage_bps": float(cfg.slippage_bps),
                    }
                )
                if cash_date != d:
                    pending_s.setdefault(cash_date, []).append(
                        (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                    )
                    pending_sell_ids_s.add(iid)
                else:
                    cash_s += proceeds
                    holdings_s.pop(iid, None)
                    labels_s.pop(iid, None)
                    entry_eff_s.pop(iid, None)
                    entry_date_s.pop(iid, None)
                    pending_sell_ids_s.discard(iid)
                    table, ticker = label.split(":", 1) if ":" in label else ("", "")
                    trades.append(
                        Trade(
                            date=d,
                            price_date=fill.price_date,
                            portfolio="strategy",
                            side="SELL",
                            instrument_id=iid,
                            symbol_key=label,
                            ticker=ticker,
                            table=table,
                            shares=sh,
                            raw_price=fill.raw_price,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_s,
                        )
                    )

        # Cash sweep: invest any unallocated cash into the sweep ticker (default BIL).
        if withholding_rate > 0:
            gross = 0.0
            tax = 0.0
            for iid, sh in holdings_s.items():
                if sh <= 0:
                    continue
                div_ps = _implied_div_cash_per_share_on_date(div_cache, instrument_id=iid, df=ensure_series(iid), date=d)
                if div_ps <= 0:
                    continue
                g = float(sh) * float(div_ps)
                if not np.isfinite(g) or g <= 0:
                    continue
                gross += g
                tax += g * withholding_rate
            net = gross - tax
            if np.isfinite(net) and net > 0:
                cash_s += float(net)
            strategy_div_gross[i] = gross
            strategy_div_tax[i] = tax
            strategy_div_net[i] = net

            if not cfg.no_benchmark:
                gross_b = 0.0
                tax_b = 0.0
                for iid, sh in holdings_b.items():
                    if sh <= 0:
                        continue
                    div_ps = _implied_div_cash_per_share_on_date(div_cache, instrument_id=iid, df=ensure_series(iid), date=d)
                    if div_ps <= 0:
                        continue
                    g = float(sh) * float(div_ps)
                    if not np.isfinite(g) or g <= 0:
                        continue
                    gross_b += g
                    tax_b += g * withholding_rate
                net_b = gross_b - tax_b
                if np.isfinite(net_b) and net_b > 0:
                    cash_b += float(net_b)
                bm_div_gross[i] = gross_b
                bm_div_tax[i] = tax_b
                bm_div_net[i] = net_b
        maybe_cash_sweep(date=pd.Timestamp(d), portfolio="strategy")
        if not cfg.no_benchmark:
            maybe_cash_sweep(date=pd.Timestamp(d), portfolio="benchmark")

        s_val, s_low = _value_portfolio(date=d, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
        b_val, b_low = _value_portfolio(date=d, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
        strategy_equity[i] = s_val
        strategy_equity_low[i] = s_low
        bm_equity[i] = b_val
        bm_equity_low[i] = b_low

        if d in period_id_by_rebalance_date and (exit_rebalance_date is None or d != exit_rebalance_date):
            period_start_equity = float(s_val)

    # Final liquidation: force-close all remaining positions.
    final_day = calendar[-1]
    for iid, sh in list(holdings_s.items()):
        label = labels_s.get(iid, "")
        fill = _force_liquidation_fill(ensure_series(iid), final_day, close_col=close_col)
        if fill is None:
            continue
        eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = sh * eff_px - cfg.commission_per_trade
        cash_s += proceeds
        holdings_s.pop(iid, None)
        labels_s.pop(iid, None)
        entry_eff_s.pop(iid, None)
        entry_date_s.pop(iid, None)
        pending_sell_ids_s.discard(iid)
        table, ticker = label.split(":", 1) if ":" in label else ("", "")
        trades.append(
            Trade(
                date=final_day,
                price_date=fill.price_date,
                portfolio="strategy",
                side="SELL",
                instrument_id=iid,
                symbol_key=label,
                ticker=ticker,
                table=table,
                shares=sh,
                raw_price=fill.raw_price,
                effective_price=eff_px,
                commission=cfg.commission_per_trade,
                slippage_bps=cfg.slippage_bps,
                notional=sh * eff_px,
                cash_after=cash_s,
            )
        )
    for iid, sh in list(holdings_b.items()):
        label = labels_b.get(iid, bm_symbol_key)
        fill = _force_liquidation_fill(ensure_series(iid), final_day, close_col=close_col)
        if fill is None:
            continue
        eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = sh * eff_px - cfg.commission_per_trade
        cash_b += proceeds
        holdings_b.pop(iid, None)
        labels_b.pop(iid, None)
        table, ticker = label.split(":", 1)
        trades.append(
            Trade(
                date=final_day,
                price_date=fill.price_date,
                portfolio="benchmark",
                side="SELL",
                instrument_id=iid,
                symbol_key=label,
                ticker=ticker,
                table=table,
                shares=sh,
                raw_price=fill.raw_price,
                effective_price=eff_px,
                commission=cfg.commission_per_trade,
                slippage_bps=cfg.slippage_bps,
                notional=sh * eff_px,
                cash_after=cash_b,
            )
        )

    price_cache = _make_price_cache(instrument_series, open_col=open_col, low_col=low_col, close_col=close_col)
    s_val, s_low = _value_portfolio(date=final_day, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
    b_val, b_low = _value_portfolio(date=final_day, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
    strategy_equity[-1] = s_val
    strategy_equity_low[-1] = s_low
    bm_equity[-1] = b_val
    bm_equity_low[-1] = b_low

    equity_df = pd.DataFrame(
        {
            "date": calendar,
            "equity": strategy_equity,
            "equity_low": strategy_equity_low,
            "equity_low_pre_actions": strategy_equity_low_pre_actions,
            "div_gross": strategy_div_gross,
            "div_tax": strategy_div_tax,
            "div_net": strategy_div_net,
        }
    )
    if not cfg.no_benchmark:
        equity_df["bm_equity"] = bm_equity
        equity_df["bm_equity_low"] = bm_equity_low
        equity_df["bm_equity_low_pre_actions"] = bm_equity_low_pre_actions
        equity_df["bm_div_gross"] = bm_div_gross
        equity_df["bm_div_tax"] = bm_div_tax
        equity_df["bm_div_net"] = bm_div_net

    pq.write_table(pa.Table.from_pandas(equity_df, preserve_index=False), out_root / "equity_curve.parquet", compression="zstd")
    if trades:
        pd.DataFrame([dataclasses.asdict(t) for t in trades]).to_csv(out_root / "trades.csv", index=False)
    if selected_rows:
        pq.write_table(
            pa.Table.from_pandas(pd.concat(selected_rows, ignore_index=True), preserve_index=False),
            out_root / "selected.parquet",
            compression="zstd",
        )
    if position_stopouts:
        pd.DataFrame(position_stopouts).to_csv(out_root / "position_stopouts.csv", index=False)
    if cash_flows:
        pd.DataFrame(cash_flows).to_csv(out_root / "cash_flows.csv", index=False)

    periods = _rebalance_periods_stats(equity_df, rebalance_dates, start_year=cfg.start_year, end_year=cfg.end_year)
    if stopout_by_period and not periods.empty and "period_id" in periods.columns:
        stop_df = pd.DataFrame(list(stopout_by_period.values()))
        periods = periods.merge(stop_df, on="period_id", how="left")
    periods.to_csv(out_root / "rebalance_periods.csv", index=False)
    cal = _calendar_yearly_stats(equity_df)
    if not cfg.no_benchmark:
        bh = _benchmark_buy_hold_calendar_year_returns(calendar=calendar, price_cache=price_cache, instrument_id=bm_instrument_id)
        cal = cal.merge(bh, on="year", how="left")
    cal.to_csv(out_root / "calendar_yearly_equity.csv", index=False)
    _write_summary_stats(out_root=out_root, equity_df=equity_df, cash_flows=cash_flows, cfg=cfg)


def _calendar_yearly_stats(equity_df: pd.DataFrame) -> pd.DataFrame:
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)
    out_rows = []
    for y, g in df.groupby("year", sort=True):
        g = g.sort_values("date", kind="mergesort")
        row: dict[str, Any] = {"year": y}
        for prefix in ["", "bm_"]:
            eq_col = f"{prefix}equity"
            if eq_col not in g.columns:
                continue
            start = float(g[eq_col].iloc[0])
            end = float(g[eq_col].iloc[-1])
            ret = (end / start) - 1.0 if start > 0 else np.nan
            max_dd = _max_drawdown_pct(g[eq_col])
            post_low_col = f"{prefix}equity_low"
            pre_low_col = f"{prefix}equity_low_pre_actions"
            mae_post = float((g[post_low_col] / start - 1.0).min()) if (start > 0 and post_low_col in g.columns) else np.nan
            mae_pre = float((g[pre_low_col] / start - 1.0).min()) if (start > 0 and pre_low_col in g.columns) else np.nan
            calmar = (ret / abs(max_dd)) if max_dd < 0 else np.nan
            div_gross_col = f"{prefix}div_gross"
            div_tax_col = f"{prefix}div_tax"
            div_net_col = f"{prefix}div_net"
            row |= {
                f"{prefix}start_equity": start,
                f"{prefix}end_equity": end,
                f"{prefix}return_pct": ret,
                f"{prefix}max_dd_pct": max_dd,
                f"{prefix}mae_pre_actions_pct": mae_pre,
                f"{prefix}mae_post_actions_pct": mae_post,
                f"{prefix}calmar": calmar,
            }
            if div_gross_col in g.columns:
                row[f"{prefix}div_gross_sum"] = float(pd.to_numeric(g[div_gross_col], errors="coerce").fillna(0.0).sum())
            if div_tax_col in g.columns:
                row[f"{prefix}div_tax_sum"] = float(pd.to_numeric(g[div_tax_col], errors="coerce").fillna(0.0).sum())
            if div_net_col in g.columns:
                row[f"{prefix}div_net_sum"] = float(pd.to_numeric(g[div_net_col], errors="coerce").fillna(0.0).sum())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _rebalance_periods_stats(
    equity_df: pd.DataFrame,
    rebalance_dates: dict[int, pd.Timestamp],
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    out: list[dict[str, Any]] = []
    df_end_excl = pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1) if not df.empty else pd.Timestamp.utcnow()

    for y in range(start_year, end_year + 1):
        if y not in rebalance_dates:
            continue
        start = rebalance_dates[y]
        end_excl = rebalance_dates.get(y + 1)
        if end_excl is None:
            end_excl = df_end_excl
        window = df[(df["date"] >= start) & (df["date"] < end_excl)].copy()
        if window.empty:
            continue

        row: dict[str, Any] = {"period_id": y, "start_date": start, "end_date": window["date"].iloc[-1]}
        for prefix in ["", "bm_"]:
            eq_col = f"{prefix}equity"
            if eq_col not in window.columns:
                continue
            start_eq = float(window[eq_col].iloc[0])
            end_eq = float(window[eq_col].iloc[-1])
            ret = (end_eq / start_eq) - 1.0 if start_eq > 0 else np.nan
            max_dd = _max_drawdown_pct(window[eq_col])
            post_low_col = f"{prefix}equity_low"
            pre_low_col = f"{prefix}equity_low_pre_actions"
            mae_post = (
                float((window[post_low_col] / start_eq - 1.0).min())
                if (start_eq > 0 and post_low_col in window.columns)
                else np.nan
            )
            mae_pre = (
                float((window[pre_low_col] / start_eq - 1.0).min())
                if (start_eq > 0 and pre_low_col in window.columns)
                else np.nan
            )
            row |= {
                f"{prefix}start_equity": start_eq,
                f"{prefix}end_equity": end_eq,
                f"{prefix}period_return_pct": ret,
                f"{prefix}max_dd_pct": max_dd,
                f"{prefix}mae_pre_actions_pct": mae_pre,
                f"{prefix}mae_post_actions_pct": mae_post,
            }
        out.append(row)
    return pd.DataFrame(out)


def _run_mode_monthly_dca_symbol_key_legacy(cfg: EntryRotationConfig, out_root: Path, month_index: pd.DataFrame) -> None:
    raise RuntimeError(
        "Legacy symbol_key engine disabled. Use the permaticker-based engine (def _run_mode_monthly_dca) to avoid ticker rename/reuse issues."
    )
    sim_months, start_select_date, end_select_date = _monthly_sim_end_date(
        month_index=month_index,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        hold_period_months=cfg.hold_period_months,
    )

    selection_bench_symbol = _resolve_benchmark_symbol_key(
        cfg.cache_dir / "tickers", cfg.selection_benchmark_ticker, prefer_table="SFP"
    )
    selection_bench_series = _load_symbol_series_for_valuation(cfg, {selection_bench_symbol}).get(selection_bench_symbol)
    if selection_bench_series is None or selection_bench_series.empty:
        raise ValueError(f"No bars for selection benchmark {cfg.selection_benchmark_ticker}")

    bench_dates = pd.DatetimeIndex(selection_bench_series["date"])
    data_end = pd.Timestamp(bench_dates.max())
    start_date = pd.Timestamp(start_select_date)
    end_date = pd.Timestamp(end_select_date)
    if end_date > data_end:
        end_date = data_end
    calendar = pd.DatetimeIndex(bench_dates[(bench_dates >= start_date) & (bench_dates <= end_date)])
    select_dates = set(pd.Timestamp(x) for x in sim_months["select_date"])

    bm_symbol = _resolve_benchmark_symbol_key(cfg.cache_dir / "tickers", cfg.benchmark_ticker, prefer_table="SFP")

    symbol_series: dict[str, pd.DataFrame] = {selection_bench_symbol: selection_bench_series}

    def ensure_series(sym: str) -> pd.DataFrame:
        if sym not in symbol_series:
            symbol_series.update(_load_symbol_series_for_valuation(cfg, {sym}))
        return symbol_series[sym]

    ensure_series(bm_symbol)

    def next_bench_date_after(d: pd.Timestamp) -> pd.Timestamp | None:
        ix = bench_dates.searchsorted(d.to_datetime64(), side="right")
        if ix < len(bench_dates):
            return pd.Timestamp(bench_dates[ix])
        return None

    def schedule_delist_credit(
        pending: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp]]],
        *,
        sym: str,
        shares: int,
        planned_exit: pd.Timestamp,
        lastpricedate: pd.Timestamp | None,
    ) -> None:
        df = ensure_series(sym)
        cutoff = (
            pd.Timestamp(lastpricedate)
            if lastpricedate is not None and not pd.isna(lastpricedate)
            else pd.Timestamp(df["date"].iloc[-1])
        )
        last_trade = _last_close_on_or_before(df, cutoff)
        if last_trade is None:
            return
        last_date, raw_px = last_trade
        if last_date >= planned_exit:
            return
        credit_date = next_bench_date_after(last_date) or planned_exit
        if credit_date > calendar[-1]:
            credit_date = calendar[-1]
        eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = shares * eff_px - cfg.commission_per_trade
        pending.setdefault(credit_date, []).append((sym, shares, raw_px, eff_px, proceeds, last_date))

    # Strategy state
    cash_s = 0.0 if cfg.initial_lump_dca_months > 0 else cfg.initial_lump_sum
    reserve_s = cfg.initial_lump_sum if cfg.initial_lump_dca_months > 0 else 0.0
    reserve_release_s = (reserve_s / cfg.initial_lump_dca_months) if cfg.initial_lump_dca_months > 0 else 0.0
    holdings_s: dict[str, int] = {}
    held_instrument_ids_s: set[str] = set()
    instrument_id_by_symbol_s: dict[str, str] = {}
    lots_s: list[dict[str, Any]] = []
    pending_s: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp]]] = {}

    # Benchmark state
    cash_b = cash_s
    reserve_b = reserve_s
    reserve_release_b = reserve_release_s
    holdings_b: dict[str, int] = {}
    held_instrument_ids_b: set[str] = set()
    instrument_id_by_symbol_b: dict[str, str] = {}
    lots_b: list[dict[str, Any]] = []
    pending_b: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp]]] = {}

    trades: list[Trade] = []
    selected_rows: list[pd.DataFrame] = []
    monthly_actions: list[dict[str, Any]] = []
    cash_flows: list[dict[str, Any]] = []

    if cfg.initial_lump_sum > 0:
        cash_flows.append({"date": start_date, "amount": -cfg.initial_lump_sum})

    sim_months = sim_months.sort_values("select_date", kind="mergesort").reset_index(drop=True)
    month_idx_by_date = {pd.Timestamp(r["select_date"]): int(r["month_idx"]) for _, r in sim_months.iterrows()}
    month_idx_set = set(month_index["month_idx"].astype(int))
    active_month_idx_max = int(sim_months[sim_months["select_date"].dt.year <= cfg.end_year]["month_idx"].max())

    price_cache = _make_price_cache(symbol_series)
    strategy_equity = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low = np.full(len(calendar), np.nan, dtype="float64")

    final_day = calendar[-1]

    for i, d in enumerate(tqdm(calendar, desc="monthly-dca:days")):
        # Apply pending delistings.
        for sym, shares, raw_px, eff_px, proceeds, price_date in pending_s.pop(d, []):
            if holdings_s.get(sym, 0) >= shares:
                holdings_s[sym] -= shares
                if holdings_s[sym] <= 0:
                    holdings_s.pop(sym, None)
                    iid = instrument_id_by_symbol_s.pop(sym, None)
                    if iid is not None:
                        held_instrument_ids_s.discard(iid)
                cash_s += proceeds
                _remove_lots_for_symbol(lots_s, sym)
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="strategy",
                        side="SELL",
                        symbol_key=sym,
                        ticker=sym.split(":", 1)[1],
                        table=sym.split(":", 1)[0],
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_s,
                    )
                )
        for sym, shares, raw_px, eff_px, proceeds, price_date in pending_b.pop(d, []):
            if holdings_b.get(sym, 0) >= shares:
                holdings_b[sym] -= shares
                if holdings_b[sym] <= 0:
                    holdings_b.pop(sym, None)
                    iid = instrument_id_by_symbol_b.pop(sym, None)
                    if iid is not None:
                        held_instrument_ids_b.discard(iid)
                cash_b += proceeds
                _remove_lots_for_symbol(lots_b, sym)
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="benchmark",
                        side="SELL",
                        symbol_key=sym,
                        ticker=sym.split(":", 1)[1],
                        table=sym.split(":", 1)[0],
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_b,
                    )
                )

        if d in select_dates:
            month_idx = month_idx_by_date[d]
            maturity_month_idx = month_idx + cfg.hold_period_months
            maturity_date = (
                pd.Timestamp(month_index.loc[month_index["month_idx"] == maturity_month_idx, "select_date"].iloc[0])
                if maturity_month_idx in month_idx_set
                else None
            )

            # Sell matured lots first.
            sold_s = 0.0
            for lot in list(lots_s):
                if lot.get("maturity_date") != d:
                    continue
                sym = str(lot["symbol_key"])
                sh = int(lot["shares"])
                sell = _sell_price_with_fallback(ensure_series(sym), d)
                if sell is None:
                    lots_s.remove(lot)
                    holdings_s[sym] = holdings_s.get(sym, 0) - sh
                    if holdings_s[sym] <= 0:
                        holdings_s.pop(sym, None)
                        iid = instrument_id_by_symbol_s.pop(sym, None)
                        if iid is not None:
                            held_instrument_ids_s.discard(iid)
                    continue
                sell_date, raw_px = sell
                eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                if sell_date != d:
                    pending_s.setdefault(sell_date, []).append((sym, sh, raw_px, eff_px, proceeds, sell_date))
                else:
                    cash_s += proceeds
                    sold_s += proceeds
                    holdings_s[sym] = holdings_s.get(sym, 0) - sh
                    if holdings_s[sym] <= 0:
                        holdings_s.pop(sym, None)
                        iid = instrument_id_by_symbol_s.pop(sym, None)
                        if iid is not None:
                            held_instrument_ids_s.discard(iid)
                    trades.append(
                        Trade(
                            date=sell_date,
                            price_date=sell_date,
                            portfolio="strategy",
                            side="SELL",
                            symbol_key=sym,
                            ticker=sym.split(":", 1)[1],
                            table=sym.split(":", 1)[0],
                            shares=sh,
                            raw_price=raw_px,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_s,
                        )
                    )
                lots_s.remove(lot)

            sold_b = 0.0
            for lot in list(lots_b):
                if lot.get("maturity_date") != d:
                    continue
                sym = str(lot["symbol_key"])
                sh = int(lot["shares"])
                sell = _sell_price_with_fallback(ensure_series(sym), d)
                if sell is None:
                    lots_b.remove(lot)
                    holdings_b[sym] = holdings_b.get(sym, 0) - sh
                    if holdings_b[sym] <= 0:
                        holdings_b.pop(sym, None)
                        iid = instrument_id_by_symbol_b.pop(sym, None)
                        if iid is not None:
                            held_instrument_ids_b.discard(iid)
                    continue
                sell_date, raw_px = sell
                eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                if sell_date != d:
                    pending_b.setdefault(sell_date, []).append((sym, sh, raw_px, eff_px, proceeds, sell_date))
                else:
                    cash_b += proceeds
                    sold_b += proceeds
                    holdings_b[sym] = holdings_b.get(sym, 0) - sh
                    if holdings_b[sym] <= 0:
                        holdings_b.pop(sym, None)
                        iid = instrument_id_by_symbol_b.pop(sym, None)
                        if iid is not None:
                            held_instrument_ids_b.discard(iid)
                    trades.append(
                        Trade(
                            date=sell_date,
                            price_date=sell_date,
                            portfolio="benchmark",
                            side="SELL",
                            symbol_key=sym,
                            ticker=sym.split(":", 1)[1],
                            table=sym.split(":", 1)[0],
                            shares=sh,
                            raw_price=raw_px,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_b,
                        )
                    )
                lots_b.remove(lot)

            is_active_month = month_idx <= active_month_idx_max
            if is_active_month:
                # External cash flow (both portfolios).
                cash_s += cfg.monthly_dca_amount
                cash_b += cfg.monthly_dca_amount
                cash_flows.append({"date": d, "amount": -cfg.monthly_dca_amount})

                # Reserve release (internal).
                if reserve_s > 0 and reserve_release_s > 0:
                    rel = min(reserve_release_s, reserve_s)
                    reserve_s -= rel
                    cash_s += rel
                if reserve_b > 0 and reserve_release_b > 0:
                    rel = min(reserve_release_b, reserve_b)
                    reserve_b -= rel
                    cash_b += rel

            # Rank candidates and record.
            candidates = load_candidates_for_select_date(cfg.cache_dir / "candidates_monthly", d)
            ranked = _select_top_n(candidates, cfg)
            if not ranked.empty:
                ranked = ranked.copy()
                ranked["selected_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
                ranked["select_date"] = d
                selected_rows.append(ranked)

            buy_sym = None
            buy_shares = 0
            buy_reason = "inactive_month"
            if is_active_month:
                # Strategy buy: pick first not already held in top_n.
                buy_sym, buy_px = _pick_first_unheld(ranked, held_instrument_ids_s)

                buy_reason = ""
                if buy_sym is None or buy_px is None:
                    buy_reason = "all_topn_already_held_or_empty"
                else:
                    spendable = cash_s * (1.0 - cfg.cash_buffer_pct)
                    eff_px = _apply_slippage(buy_px, side="BUY", slippage_bps=cfg.slippage_bps)
                    buy_shares = int(math.floor((spendable - cfg.commission_per_trade) / eff_px))
                    if buy_shares < 1:
                        buy_reason = "insufficient_cash_for_1_share"
                        buy_shares = 0
                    else:
                        cost = buy_shares * eff_px + cfg.commission_per_trade
                        if cost <= cash_s:
                            cash_s -= cost
                            holdings_s[buy_sym] = holdings_s.get(buy_sym, 0) + buy_shares
                            iid = (
                                str(ranked.loc[ranked["symbol_key"] == buy_sym, "instrument_id"].iloc[0])
                                if "instrument_id" in ranked.columns
                                else buy_sym
                            )
                            instrument_id_by_symbol_s[buy_sym] = iid
                            held_instrument_ids_s.add(iid)
                            df_sym = ensure_series(buy_sym)
                            _update_price_cache_symbol(price_cache, buy_sym, df_sym)
                            trades.append(
                                Trade(
                                    date=d,
                                    price_date=d,
                                    portfolio="strategy",
                                    side="BUY",
                                    symbol_key=buy_sym,
                                    ticker=buy_sym.split(":", 1)[1],
                                    table=buy_sym.split(":", 1)[0],
                                    shares=buy_shares,
                                    raw_price=buy_px,
                                    effective_price=eff_px,
                                    commission=cfg.commission_per_trade,
                                    slippage_bps=cfg.slippage_bps,
                                    notional=buy_shares * eff_px,
                                    cash_after=cash_s,
                                )
                            )
                            lots_s.append({"symbol_key": buy_sym, "shares": buy_shares, "maturity_date": maturity_date})
                            if maturity_date is not None:
                                lastpricedate = (
                                    ranked.loc[ranked["symbol_key"] == buy_sym, "lastpricedate"].iloc[0]
                                    if "lastpricedate" in ranked.columns
                                    else pd.NaT
                                )
                                schedule_delist_credit(
                                    pending_s,
                                    sym=buy_sym,
                                    shares=buy_shares,
                                    planned_exit=maturity_date,
                                    lastpricedate=lastpricedate,
                                )

            bm_buy_shares = 0
            bm_buy_reason = "inactive_month"
            if is_active_month:
                # Benchmark buy every month (same funding schedule, 12-month lots).
                bm_buy_px = _price_at(ensure_series(bm_symbol), d, "open_adj")
                bm_buy_reason = ""
                if bm_buy_px is None:
                    bm_buy_reason = "missing_open"
                else:
                    spendable = cash_b * (1.0 - cfg.cash_buffer_pct)
                    eff_px = _apply_slippage(bm_buy_px, side="BUY", slippage_bps=cfg.slippage_bps)
                    bm_buy_shares = int(math.floor((spendable - cfg.commission_per_trade) / eff_px))
                    if bm_buy_shares < 1:
                        bm_buy_reason = "insufficient_cash_for_1_share"
                        bm_buy_shares = 0
                    else:
                        cost = bm_buy_shares * eff_px + cfg.commission_per_trade
                        if cost <= cash_b:
                            cash_b -= cost
                            holdings_b[bm_symbol] = holdings_b.get(bm_symbol, 0) + bm_buy_shares
                            instrument_id_by_symbol_b[bm_symbol] = bm_symbol
                            held_instrument_ids_b.add(bm_symbol)
                            trades.append(
                                Trade(
                                    date=d,
                                    price_date=d,
                                    portfolio="benchmark",
                                    side="BUY",
                                    symbol_key=bm_symbol,
                                    ticker=bm_symbol.split(":", 1)[1],
                                    table=bm_symbol.split(":", 1)[0],
                                    shares=bm_buy_shares,
                                    raw_price=bm_buy_px,
                                    effective_price=eff_px,
                                    commission=cfg.commission_per_trade,
                                    slippage_bps=cfg.slippage_bps,
                                    notional=bm_buy_shares * eff_px,
                                    cash_after=cash_b,
                                )
                            )
                            lots_b.append({"symbol_key": bm_symbol, "shares": bm_buy_shares, "maturity_date": maturity_date})
                            if maturity_date is not None:
                                schedule_delist_credit(
                                    pending_b,
                                    sym=bm_symbol,
                                    shares=bm_buy_shares,
                                    planned_exit=maturity_date,
                                    lastpricedate=pd.NaT,
                                )

            monthly_actions.append(
                {
                    "select_date": d,
                    "month_idx": month_idx,
                    "sold_proceeds": sold_s,
                    "buy_symbol_key": buy_sym,
                    "buy_shares": buy_shares,
                    "buy_skipped_reason": buy_reason,
                    "cash_end": cash_s,
                    "reserve_remaining": reserve_s,
                    "bm_sold_proceeds": sold_b,
                    "bm_buy_shares": bm_buy_shares,
                    "bm_buy_skipped_reason": bm_buy_reason,
                    "bm_cash_end": cash_b,
                    "bm_reserve_remaining": reserve_b,
                }
            )

        # Liquidate at end-of-backtest BEFORE final valuation.
        if d == final_day:
            for sym, sh in list(holdings_s.items()):
                sell = _sell_price_no_forward(ensure_series(sym), d)
                if sell is None:
                    continue
                sell_date, raw_px = sell
                eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                cash_s += proceeds
                holdings_s.pop(sym, None)
                iid = instrument_id_by_symbol_s.pop(sym, None)
                if iid is not None:
                    held_instrument_ids_s.discard(iid)
                trades.append(
                    Trade(
                        date=sell_date,
                        price_date=sell_date,
                        portfolio="strategy",
                        side="SELL",
                        symbol_key=sym,
                        ticker=sym.split(":", 1)[1],
                        table=sym.split(":", 1)[0],
                        shares=sh,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=sh * eff_px,
                        cash_after=cash_s,
                    )
                )
            for sym, sh in list(holdings_b.items()):
                sell = _sell_price_no_forward(ensure_series(sym), d)
                if sell is None:
                    continue
                sell_date, raw_px = sell
                eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                cash_b += proceeds
                holdings_b.pop(sym, None)
                iid = instrument_id_by_symbol_b.pop(sym, None)
                if iid is not None:
                    held_instrument_ids_b.discard(iid)
                trades.append(
                    Trade(
                        date=sell_date,
                        price_date=sell_date,
                        portfolio="benchmark",
                        side="SELL",
                        symbol_key=sym,
                        ticker=sym.split(":", 1)[1],
                        table=sym.split(":", 1)[0],
                        shares=sh,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=sh * eff_px,
                        cash_after=cash_b,
                    )
                )

        s_val, s_low = _value_portfolio(date=d, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
        b_val, b_low = _value_portfolio(date=d, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
        strategy_equity[i] = s_val
        strategy_equity_low[i] = s_low
        bm_equity[i] = b_val
        bm_equity_low[i] = b_low

    equity_df = pd.DataFrame({"date": calendar, "equity": strategy_equity, "equity_low": strategy_equity_low})
    equity_df["equity_low_pre_actions"] = strategy_equity_low_pre_actions
    if not cfg.no_benchmark:
        equity_df["bm_equity"] = bm_equity
        equity_df["bm_equity_low"] = bm_equity_low
        equity_df["bm_equity_low_pre_actions"] = bm_equity_low_pre_actions

    pq.write_table(pa.Table.from_pandas(equity_df, preserve_index=False), out_root / "equity_curve.parquet", compression="zstd")
    if trades:
        pd.DataFrame([dataclasses.asdict(t) for t in trades]).to_csv(out_root / "trades.csv", index=False)
    if selected_rows:
        pq.write_table(
            pa.Table.from_pandas(pd.concat(selected_rows, ignore_index=True), preserve_index=False),
            out_root / "selected.parquet",
            compression="zstd",
        )
    pd.DataFrame(monthly_actions).to_csv(out_root / "monthly_actions.csv", index=False)
    pd.DataFrame(cash_flows).to_csv(out_root / "cash_flows.csv", index=False)
    _calendar_yearly_stats(equity_df).to_csv(out_root / "calendar_yearly_equity.csv", index=False)


def _run_mode_monthly_dca(cfg: EntryRotationConfig, out_root: Path, month_index: pd.DataFrame) -> None:
    """
    Monthly DCA engine keyed by permaticker instrument_id.

    - Each month: sell matured 12m lots, add contribution (active months only), optionally release lump-sum reserve, then buy 1 new instrument (first unheld in top-N).
    - Holdings + "already held" checks are tracked by `instrument_id`, not ticker.
    - Sells follow `cfg.sell_price_policy`, but end-of-backtest liquidation never looks forward.
    """
    sim_months, start_select_date, end_select_date = _monthly_sim_end_date(
        month_index=month_index,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        hold_period_months=cfg.hold_period_months,
    )
    select_date_by_month_idx = {int(r["month_idx"]): pd.Timestamp(r["select_date"]) for _, r in month_index.iterrows()}

    selection_bench_symbol = _resolve_benchmark_symbol_key(
        cfg.cache_dir / "tickers", cfg.selection_benchmark_ticker, prefer_table="SFP"
    )
    selection_bench_series = _load_symbol_series_for_valuation(cfg, {selection_bench_symbol}).get(selection_bench_symbol)
    if selection_bench_series is None or selection_bench_series.empty:
        raise ValueError(f"No bars for selection benchmark {cfg.selection_benchmark_ticker}")
    open_col, low_col, close_col = _price_cols_for_mode(cfg.price_mode)
    if cfg.price_mode == "raw":
        missing = [c for c in (open_col, low_col, close_col) if c not in selection_bench_series.columns]
        if missing:
            raise ValueError(
                f"price_mode=raw requires raw OHLC in the bars cache. Rebuild caches with --rebuild-cache. Missing: {missing}"
            )

    bench_dates = pd.DatetimeIndex(selection_bench_series["date"])
    data_end = pd.Timestamp(bench_dates.max())
    start_date = pd.Timestamp(start_select_date)
    end_date = pd.Timestamp(end_select_date)
    if end_date > data_end:
        end_date = data_end
    calendar = pd.DatetimeIndex(bench_dates[(bench_dates >= start_date) & (bench_dates <= end_date)])
    final_day = calendar[-1]
    select_dates_set = set(pd.Timestamp(x) for x in sim_months["select_date"])

    candidates_dir = cfg.cache_dir / "candidates_monthly"
    sell_policy = cfg.sell_price_policy

    bm_symbol_key = _resolve_benchmark_symbol_key(cfg.cache_dir / "tickers", cfg.benchmark_ticker, prefer_table="SFP")
    sym_to_perm = _symbol_key_to_permaticker_map(cfg.cache_dir / "tickers")
    bm_perm = sym_to_perm.get(bm_symbol_key)
    if bm_perm is None:
        raise ValueError(f"Benchmark ticker missing permaticker mapping: {bm_symbol_key}")
    bm_instrument_id = f"PT{bm_perm}"

    instrument_series: dict[str, pd.DataFrame] = {}

    def ensure_series(instrument_id: str) -> pd.DataFrame:
        if instrument_id not in instrument_series:
            instrument_series.update(_load_series_for_instrument_ids(cfg, {instrument_id}))
        return instrument_series[instrument_id]

    ensure_series(bm_instrument_id)

    def next_bench_date_after(d: pd.Timestamp) -> pd.Timestamp | None:
        ix = bench_dates.searchsorted(d.to_datetime64(), side="right")
        if ix < len(bench_dates):
            return pd.Timestamp(bench_dates[ix])
        return None

    def schedule_delist_credit(
        pending: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp, str]]],
        *,
        instrument_id: str,
        shares: int,
        planned_exit: pd.Timestamp,
        lastpricedate: pd.Timestamp | None,
        label_symbol_key: str,
    ) -> None:
        df = ensure_series(instrument_id)
        cutoff = (
            pd.Timestamp(lastpricedate)
            if lastpricedate is not None and not pd.isna(lastpricedate)
            else pd.Timestamp(df["date"].iloc[-1])
        )
        last_trade = _last_close_on_or_before(df, cutoff)
        if last_trade is None:
            return
        last_date, raw_px = last_trade
        if last_date >= planned_exit:
            return
        credit_date = next_bench_date_after(last_date) or planned_exit
        if credit_date > final_day:
            credit_date = final_day
        eff_px = _apply_slippage(raw_px, side="SELL", slippage_bps=cfg.slippage_bps)
        proceeds = shares * eff_px - cfg.commission_per_trade
        pending.setdefault(credit_date, []).append((instrument_id, shares, raw_px, eff_px, proceeds, last_date, label_symbol_key))

    # Strategy: initial capital + optional lump sum (optionally released monthly).
    cash_s = float(cfg.initial_equity)
    reserve_s = float(cfg.initial_lump_sum) if cfg.initial_lump_dca_months > 0 else 0.0
    reserve_release_s = (reserve_s / cfg.initial_lump_dca_months) if cfg.initial_lump_dca_months > 0 else 0.0
    if cfg.initial_lump_dca_months <= 0:
        cash_s += float(cfg.initial_lump_sum)

    cash_b = float(cfg.initial_equity)
    reserve_b = reserve_s
    reserve_release_b = reserve_release_s
    if cfg.initial_lump_dca_months <= 0:
        cash_b += float(cfg.initial_lump_sum)

    holdings_s: dict[str, int] = {}
    labels_s: dict[str, str] = {}
    entry_eff_s: dict[str, float] = {}
    entry_date_s: dict[str, pd.Timestamp] = {}
    lots_s: list[dict[str, Any]] = []
    pending_s: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp, str]]] = {}
    pending_buy_s: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    pending_sell_ids_s: set[str] = set()

    holdings_b: dict[str, int] = {}
    labels_b: dict[str, str] = {bm_instrument_id: bm_symbol_key}
    lots_b: list[dict[str, Any]] = []
    pending_b: dict[pd.Timestamp, list[tuple[str, int, float, float, float, pd.Timestamp, str]]] = {}

    held_ids_s: set[str] = set()
    pending_buy_ids_s: set[str] = set()
    opened_new_names_by_year_s: dict[int, set[str]] = {}
    reserved_new_names_by_year_s: dict[int, set[str]] = {}

    trades: list[Trade] = []
    selected_rows: list[pd.DataFrame] = []
    monthly_actions: list[dict[str, Any]] = []
    cash_flows: list[dict[str, Any]] = []
    stopouts: list[dict[str, Any]] = []
    position_stopouts: list[dict[str, Any]] = []

    if cfg.initial_lump_sum > 0:
        cash_flows.append({"date": start_date, "amount": -float(cfg.initial_lump_sum)})

    sim_months = sim_months.sort_values("select_date", kind="mergesort").reset_index(drop=True)
    month_idx_by_date = {pd.Timestamp(r["select_date"]): int(r["month_idx"]) for _, r in sim_months.iterrows()}
    active_month_idx_max = int(sim_months[sim_months["select_date"].dt.year <= cfg.end_year]["month_idx"].max())
    sim_select_dates = pd.DatetimeIndex(sim_months["select_date"]).sort_values()

    price_cache = _make_price_cache(instrument_series, open_col=open_col, low_col=low_col, close_col=close_col)
    strategy_equity = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    strategy_equity_low_pre_actions = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low = np.full(len(calendar), np.nan, dtype="float64")
    bm_equity_low_pre_actions = np.full(len(calendar), np.nan, dtype="float64")
    strategy_div_gross = np.zeros(len(calendar), dtype="float64")
    strategy_div_tax = np.zeros(len(calendar), dtype="float64")
    strategy_div_net = np.zeros(len(calendar), dtype="float64")
    bm_div_gross = np.zeros(len(calendar), dtype="float64")
    bm_div_tax = np.zeros(len(calendar), dtype="float64")
    bm_div_net = np.zeros(len(calendar), dtype="float64")
    div_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    withholding_rate = float(cfg.dividend_withholding_rate)

    cash_sweep_enabled = bool(cfg.cash_sweep_enabled) and bool(str(cfg.cash_sweep_ticker or "").strip()) and withholding_rate > 0
    cash_sweep_symbol_key: str | None = None
    cash_sweep_instrument_id: str | None = None
    pending_div_sweep_s = 0.0
    pending_div_sweep_b = 0.0
    if cash_sweep_enabled:
        try:
            cash_sweep_symbol_key = _resolve_benchmark_symbol_key(
                cfg.cache_dir / "tickers", str(cfg.cash_sweep_ticker).strip().upper(), prefer_table="SFP"
            )
            sweep_perm = sym_to_perm.get(cash_sweep_symbol_key)
            if sweep_perm is None:
                raise ValueError(f"Cash sweep ticker missing permaticker mapping: {cash_sweep_symbol_key}")
            cash_sweep_instrument_id = f"PT{sweep_perm}"
            ensure_series(cash_sweep_instrument_id)
            _update_price_cache_symbol(
                price_cache,
                cash_sweep_instrument_id,
                ensure_series(cash_sweep_instrument_id),
                open_col=open_col,
                low_col=low_col,
                close_col=close_col,
            )
        except Exception as e:
            logger.warning("Disabling dividend sweep: unable to resolve %r (%s)", cfg.cash_sweep_ticker, e)
            cash_sweep_enabled = False
            cash_sweep_symbol_key = None
            cash_sweep_instrument_id = None

    def maybe_sweep_dividend_cash(*, date: pd.Timestamp, portfolio: Literal["strategy", "benchmark"]) -> None:
        nonlocal cash_s, cash_b, pending_div_sweep_s, pending_div_sweep_b
        if not cash_sweep_enabled or cash_sweep_instrument_id is None or cash_sweep_symbol_key is None:
            return
        if date >= final_day:
            return
        pending = float(pending_div_sweep_s) if portfolio == "strategy" else float(pending_div_sweep_b)
        if not (np.isfinite(pending) and pending > 0):
            return
        df_sweep = ensure_series(cash_sweep_instrument_id)
        px = _price_at(df_sweep, date, close_col)
        if px is None:
            return
        eff_px = _apply_slippage(float(px), side="BUY", slippage_bps=cfg.slippage_bps)
        shares = _whole_shares_for_spendable_cash(
            cash=pending,
            effective_price=eff_px,
            commission=float(cfg.commission_per_trade),
            cash_buffer_pct=0.0,
        )
        if shares < 1:
            return
        cost = float(shares) * float(eff_px) + float(cfg.commission_per_trade)
        if not (np.isfinite(cost) and cost > 0 and cost <= pending):
            return

        if portfolio == "strategy":
            if cost > float(cash_s):
                return
            cash_s = float(cash_s) - cost
            pending_div_sweep_s = float(pending_div_sweep_s) - cost
            holdings_s[cash_sweep_instrument_id] = holdings_s.get(cash_sweep_instrument_id, 0) + int(shares)
            labels_s[cash_sweep_instrument_id] = cash_sweep_symbol_key
            table, ticker = cash_sweep_symbol_key.split(":", 1)
            trades.append(
                Trade(
                    date=date,
                    price_date=date,
                    portfolio="strategy",
                    side="BUY",
                    instrument_id=cash_sweep_instrument_id,
                    symbol_key=cash_sweep_symbol_key,
                    ticker=ticker,
                    table=table,
                    shares=int(shares),
                    raw_price=float(px),
                    effective_price=float(eff_px),
                    commission=cfg.commission_per_trade,
                    slippage_bps=cfg.slippage_bps,
                    notional=float(shares) * float(eff_px),
                    cash_after=float(cash_s),
                )
            )
        else:
            if cost > float(cash_b):
                return
            cash_b = float(cash_b) - cost
            pending_div_sweep_b = float(pending_div_sweep_b) - cost
            holdings_b[cash_sweep_instrument_id] = holdings_b.get(cash_sweep_instrument_id, 0) + int(shares)
            labels_b[cash_sweep_instrument_id] = cash_sweep_symbol_key
            table, ticker = cash_sweep_symbol_key.split(":", 1)
            trades.append(
                Trade(
                    date=date,
                    price_date=date,
                    portfolio="benchmark",
                    side="BUY",
                    instrument_id=cash_sweep_instrument_id,
                    symbol_key=cash_sweep_symbol_key,
                    ticker=ticker,
                    table=table,
                    shares=int(shares),
                    raw_price=float(px),
                    effective_price=float(eff_px),
                    commission=cfg.commission_per_trade,
                    slippage_bps=cfg.slippage_bps,
                    notional=float(shares) * float(eff_px),
                    cash_after=float(cash_b),
                )
            )

    period_start_equity: float | None = None
    stopout_active = False
    stopout_release_select_date: pd.Timestamp | None = None
    stopout_trigger_month_idx: int | None = None
    stopout_resume_month_idx: int | None = None
    # Post-stopout re-entry ramp: constrain buy budget to release cash in equal monthly tranches.
    reentry_remaining_s = 0.0
    reentry_months_left_s = 0
    buy_budget_s = float("inf")

    def cancel_pending_sell(iid: str) -> None:
        if iid not in pending_sell_ids_s:
            return
        for k in list(pending_s.keys()):
            pending_s[k] = [t for t in pending_s[k] if str(t[0]) != iid]
            if not pending_s[k]:
                pending_s.pop(k, None)
        pending_sell_ids_s.discard(iid)

    for i, d in enumerate(tqdm(calendar, desc="monthly-dca:days")):
        for iid, shares, raw_px, eff_px, proceeds, price_date, label in pending_s.pop(d, []):
            if holdings_s.get(iid, 0) >= shares:
                holdings_s[iid] -= shares
                if holdings_s[iid] <= 0:
                    holdings_s.pop(iid, None)
                    held_ids_s.discard(iid)
                    labels_s.pop(iid, None)
                    entry_eff_s.pop(iid, None)
                    entry_date_s.pop(iid, None)
                    pending_sell_ids_s.discard(iid)
                cash_s += proceeds
                if np.isfinite(buy_budget_s):
                    buy_budget_s += float(proceeds)
                _remove_lots_for_instrument(lots_s, iid)
                table, ticker = label.split(":", 1)
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="strategy",
                        side="SELL",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_s,
                    )
                )
        for iid, shares, raw_px, eff_px, proceeds, price_date, label in pending_b.pop(d, []):
            if holdings_b.get(iid, 0) >= shares:
                holdings_b[iid] -= shares
                if holdings_b[iid] <= 0:
                    holdings_b.pop(iid, None)
                    labels_b.pop(iid, None)
                cash_b += proceeds
                _remove_lots_for_instrument(lots_b, iid)
                table, ticker = label.split(":", 1)
                trades.append(
                    Trade(
                        date=d,
                        price_date=price_date,
                        portfolio="benchmark",
                        side="SELL",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=shares,
                        raw_price=raw_px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_b,
                    )
                )

        if d in select_dates_set:
            month_idx = month_idx_by_date[d]
            maturity_date: pd.Timestamp | None = None
            if cfg.monthly_sell_policy == "maturity":
                maturity_date = select_date_by_month_idx.get(month_idx + cfg.hold_period_months)
            is_active_month = month_idx <= active_month_idx_max

            # If we stopped out earlier, only resume buying after the next scheduled select_date AND only if liquidation has completed.
            if stopout_active:
                if stopout_release_select_date is not None and d >= stopout_release_select_date:
                    if not holdings_s and not pending_sell_ids_s:
                        stopout_active = False
                        # Initialize re-entry ramp from the current cash balance (budget constraint only).
                        if int(cfg.stopout_reentry_dca_months) > 0:
                            reentry_remaining_s, reentry_months_left_s, buy_budget_s = _reentry_dca_init_from_cash(
                                cash_s, int(cfg.stopout_reentry_dca_months)
                            )
                        else:
                            reentry_remaining_s = 0.0
                            reentry_months_left_s = 0
                            buy_budget_s = float("inf")
                        stopout_release_select_date = None
                        stopout_trigger_month_idx = None
                        stopout_resume_month_idx = None
                    else:
                        # Still liquidating; push the release to the next select date.
                        stopout_release_select_date = _next_select_date_after(sim_select_dates, d)

            sell_blocked = False
            sold_s = 0.0
            sold_b = 0.0
            if cfg.monthly_sell_policy == "maturity":
                # Sell matured lots first (strategy).
                for lot in list(lots_s):
                    maturity = lot.get("maturity_date")
                    if maturity is None:
                        continue
                    if pd.Timestamp(maturity) > d:
                        continue
                    iid = str(lot["instrument_id"])
                    sh = int(lot["shares"])
                    label = str(lot.get("symbol_key", labels_s.get(iid, "")))
                    if lot.get("sell_scheduled"):
                        continue
                    fill = _resolve_sell_fill(
                        ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=final_day
                    )
                    if fill is None:
                        if sell_policy == "no-forward":
                            sell_blocked = True
                        continue  # skip-and-carry (will retry on later select dates)
                    eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                    proceeds = sh * eff_px - cfg.commission_per_trade
                    if fill.cash_date != d:
                        cash_date = bench_date_on_or_after(fill.cash_date)
                        if cash_date is None or cash_date > final_day:
                            if sell_policy == "no-forward":
                                sell_blocked = True
                            continue
                        pending_s.setdefault(cash_date, []).append(
                            (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                        )
                        pending_sell_ids_s.add(iid)
                        lot["sell_scheduled"] = True
                        if sell_policy == "no-forward":
                            sell_blocked = True
                    else:
                        cash_s += proceeds
                        sold_s += proceeds
                        holdings_s[iid] = holdings_s.get(iid, 0) - sh
                        if holdings_s[iid] <= 0:
                            holdings_s.pop(iid, None)
                            held_ids_s.discard(iid)
                            labels_s.pop(iid, None)
                            entry_eff_s.pop(iid, None)
                            entry_date_s.pop(iid, None)
                            pending_sell_ids_s.discard(iid)
                        table, ticker = label.split(":", 1)
                        trades.append(
                            Trade(
                                date=d,
                                price_date=fill.price_date,
                                portfolio="strategy",
                                side="SELL",
                                instrument_id=iid,
                                symbol_key=label,
                                ticker=ticker,
                                table=table,
                                shares=sh,
                                raw_price=fill.raw_price,
                                effective_price=eff_px,
                                commission=cfg.commission_per_trade,
                                slippage_bps=cfg.slippage_bps,
                                notional=sh * eff_px,
                                cash_after=cash_s,
                            )
                        )
                        lots_s.remove(lot)

                # Sell matured lots (benchmark).
                for lot in list(lots_b):
                    maturity = lot.get("maturity_date")
                    if maturity is None:
                        continue
                    if pd.Timestamp(maturity) > d:
                        continue
                    iid = str(lot["instrument_id"])
                    sh = int(lot["shares"])
                    label = str(lot.get("symbol_key", labels_b.get(iid, bm_symbol_key)))
                    if lot.get("sell_scheduled"):
                        continue
                    fill = _resolve_sell_fill(
                        ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=final_day
                    )
                    if fill is None:
                        continue  # skip-and-carry (will retry on later select dates)
                    eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                    proceeds = sh * eff_px - cfg.commission_per_trade
                    if fill.cash_date != d:
                        cash_date = bench_date_on_or_after(fill.cash_date)
                        if cash_date is None or cash_date > final_day:
                            continue
                        pending_b.setdefault(cash_date, []).append(
                            (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                        )
                        lot["sell_scheduled"] = True
                    else:
                        cash_b += proceeds
                        sold_b += proceeds
                        holdings_b[iid] = holdings_b.get(iid, 0) - sh
                        if holdings_b[iid] <= 0:
                            holdings_b.pop(iid, None)
                            labels_b.pop(iid, None)
                        table, ticker = label.split(":", 1)
                        trades.append(
                            Trade(
                                date=d,
                                price_date=fill.price_date,
                                portfolio="benchmark",
                                side="SELL",
                                instrument_id=iid,
                                symbol_key=label,
                                ticker=ticker,
                                table=table,
                                shares=sh,
                                raw_price=fill.raw_price,
                                effective_price=eff_px,
                                commission=cfg.commission_per_trade,
                                slippage_bps=cfg.slippage_bps,
                                notional=sh * eff_px,
                                cash_after=cash_b,
                            )
                        )
                        lots_b.remove(lot)

            if is_active_month:
                cash_s += cfg.monthly_dca_amount
                cash_b += cfg.monthly_dca_amount
                cash_flows.append({"date": d, "amount": -cfg.monthly_dca_amount})
                if np.isfinite(buy_budget_s):
                    buy_budget_s += float(cfg.monthly_dca_amount)
                if reserve_s > 0 and reserve_release_s > 0:
                    rel = min(reserve_release_s, reserve_s)
                    reserve_s -= rel
                    cash_s += rel
                    if np.isfinite(buy_budget_s):
                        buy_budget_s += float(rel)
                if reserve_b > 0 and reserve_release_b > 0:
                    rel = min(reserve_release_b, reserve_b)
                    reserve_b -= rel
                    cash_b += rel

            candidates = load_candidates_for_select_date(candidates_dir, d)
            ranked = _select_top_n(candidates, cfg)
            if not ranked.empty:
                ranked = ranked.copy()
                ranked["selected_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
                ranked["select_date"] = d
                selected_rows.append(ranked)

            buy_iid = None
            buy_symbol_key: str | None = None
            buy_exec_date: pd.Timestamp | None = None
            buy_shares = 0
            buy_reason = "inactive_month"
            if stopout_active:
                buy_reason = "portfolio_stopout_active"
                if stopout_release_select_date is not None and d < stopout_release_select_date:
                    buy_reason = "portfolio_stopout_cooldown"
                elif holdings_s or pending_sell_ids_s:
                    buy_reason = "portfolio_stopout_liquidating"
            buy_blocked = sell_policy == "no-forward" and sell_blocked and not cfg.allow_overlap_on_sell_block
            can_buy_after_stop = not stopout_active or (
                stopout_release_select_date is not None
                and d >= stopout_release_select_date
                and not holdings_s
                and not pending_sell_ids_s
            )
            if is_active_month and not buy_blocked and can_buy_after_stop:
                if reentry_months_left_s > 0 and np.isfinite(buy_budget_s):
                    released, reentry_remaining_s, reentry_months_left_s = _reentry_dca_release(
                        reentry_remaining_s, reentry_months_left_s
                    )
                    buy_budget_s += float(released)
                    if reentry_months_left_s <= 0:
                        # Release any rounding dust at the end of the ramp.
                        buy_budget_s = float("inf")
                held_or_pending = held_ids_s | pending_buy_ids_s
                cand = _pick_first_unheld_instrument(ranked, held_or_pending)
                if cand is None:
                    buy_reason = "all_topn_already_held_or_empty"
                else:
                    buy_iid = str(cand["instrument_id"])
                    if buy_iid in pending_buy_ids_s:
                        buy_reason = "already_scheduled"
                    else:
                        label = str(cand["symbol_key"])
                        buy_symbol_key = label
                        lastpricedate = cand["lastpricedate"] if "lastpricedate" in cand else pd.NaT
                        raw_buy_date = (
                            pd.Timestamp(cand["entry_date"]) if "entry_date" in cand and pd.notna(cand["entry_date"]) else d
                        )
                        if raw_buy_date < d:
                            raw_buy_date = d
                        ix = bench_dates.searchsorted(raw_buy_date.to_datetime64(), side="left")
                        if ix >= len(bench_dates):
                            buy_reason = "buy_date_out_of_calendar"
                        else:
                            buy_exec_date = pd.Timestamp(bench_dates[ix])
                            if buy_exec_date >= final_day:
                                buy_reason = "buy_date_out_of_range"
                                buy_exec_date = None
                            else:
                                cap_year = int(pd.Timestamp(d).year)
                                reserved_new_name = False
                                if cfg.monthly_sell_policy == "never":
                                    already_opened = buy_iid in opened_new_names_by_year_s.get(cap_year, set())
                                    already_reserved = buy_iid in reserved_new_names_by_year_s.get(cap_year, set())
                                    if not already_opened and not already_reserved:
                                        opened_count = _count_opened_new_names_for_year(opened_new_names_by_year_s, year=cap_year)
                                        reserved_count = _count_reserved_new_names_for_year(
                                            reserved_new_names_by_year_s, year=cap_year
                                        )
                                        if (opened_count + reserved_count) >= int(cfg.max_new_tickers_per_year):
                                            buy_reason = "yearly_new_names_cap_reached"
                                            buy_exec_date = None
                                        else:
                                            _reserve_new_name_for_year(
                                                reserved_new_names_by_year_s, year=cap_year, instrument_id=buy_iid
                                            )
                                            reserved_new_name = True

                                if buy_exec_date is not None:
                                    pending_buy_s.setdefault(buy_exec_date, []).append(
                                        {
                                            "instrument_id": buy_iid,
                                            "symbol_key": label,
                                            "maturity_date": maturity_date,
                                            "lastpricedate": lastpricedate,
                                            "cap_year": cap_year,
                                            "reserved_new_name": reserved_new_name,
                                        }
                                    )
                                    pending_buy_ids_s.add(buy_iid)
                                    buy_reason = "scheduled"
            elif is_active_month and buy_blocked:
                buy_reason = "sell_blocked_no_forward"

            bm_buy_shares = 0
            bm_buy_reason = "inactive_month"
            if is_active_month:
                bm_buy_px = _price_at(ensure_series(bm_instrument_id), d, open_col)
                bm_buy_reason = ""
                if bm_buy_px is None:
                    bm_buy_reason = "missing_open"
                else:
                    spendable = cash_b * (1.0 - cfg.cash_buffer_pct)
                    eff_px = _apply_slippage(bm_buy_px, side="BUY", slippage_bps=cfg.slippage_bps)
                    bm_buy_shares = int(math.floor((spendable - cfg.commission_per_trade) / eff_px))
                    if bm_buy_shares < 1:
                        bm_buy_reason = "insufficient_cash_for_1_share"
                        bm_buy_shares = 0
                    else:
                        cost = bm_buy_shares * eff_px + cfg.commission_per_trade
                        if cost <= cash_b:
                            cash_b -= cost
                            holdings_b[bm_instrument_id] = holdings_b.get(bm_instrument_id, 0) + bm_buy_shares
                            labels_b[bm_instrument_id] = bm_symbol_key
                            trades.append(
                                Trade(
                                    date=d,
                                    price_date=d,
                                    portfolio="benchmark",
                                    side="BUY",
                                    instrument_id=bm_instrument_id,
                                    symbol_key=bm_symbol_key,
                                    ticker=bm_symbol_key.split(":", 1)[1],
                                    table=bm_symbol_key.split(":", 1)[0],
                                    shares=bm_buy_shares,
                                    raw_price=bm_buy_px,
                                    effective_price=eff_px,
                                    commission=cfg.commission_per_trade,
                                    slippage_bps=cfg.slippage_bps,
                                    notional=bm_buy_shares * eff_px,
                                    cash_after=cash_b,
                                )
                            )
                            bm_maturity_date = maturity_date
                            lots_b.append(
                                {
                                    "instrument_id": bm_instrument_id,
                                    "shares": bm_buy_shares,
                                    "maturity_date": bm_maturity_date,
                                    "symbol_key": bm_symbol_key,
                                }
                            )

            monthly_actions.append(
                {
                    "select_date": d,
                    "month_idx": month_idx,
                    "sold_proceeds": sold_s,
                    "buy_instrument_id": buy_iid,
                    "buy_symbol_key": buy_symbol_key,
                    "buy_exec_date": buy_exec_date,
                    "buy_shares": buy_shares,
                    "buy_skipped_reason": buy_reason,
                    "cash_end": cash_s,
                    "buy_budget_end": buy_budget_s if np.isfinite(buy_budget_s) else cash_s,
                    "reentry_dca_remaining": reentry_remaining_s,
                    "reentry_dca_months_left": reentry_months_left_s,
                    "reserve_remaining": reserve_s,
                    "bm_sold_proceeds": sold_b,
                    "bm_buy_shares": bm_buy_shares,
                    "bm_buy_skipped_reason": bm_buy_reason,
                    "bm_cash_end": cash_b,
                    "bm_reserve_remaining": reserve_b,
                }
            )

        # Execute scheduled buys after sell/contribution logic (sell-first semantics on select dates).
        if d != final_day:
            for order in pending_buy_s.pop(d, []):
                iid = str(order.get("instrument_id", ""))
                if not iid:
                    continue
                cap_year_raw = order.get("cap_year")
                cap_year = int(cap_year_raw) if cap_year_raw is not None and str(cap_year_raw) != "" else None
                reserved_new_name = bool(order.get("reserved_new_name", False))

                def release_reservation() -> None:
                    if reserved_new_name and cap_year is not None:
                        _release_reserved_new_name_for_year(reserved_new_names_by_year_s, year=cap_year, instrument_id=iid)

                if iid in held_ids_s or holdings_s.get(iid, 0) > 0:
                    release_reservation()
                    pending_buy_ids_s.discard(iid)
                    continue
                label = str(order.get("symbol_key", labels_s.get(iid, "")))
                maturity_date = order.get("maturity_date")
                lastpricedate = order.get("lastpricedate", pd.NaT)

                df_iid = ensure_series(iid)
                px = _price_at(df_iid, d, open_col)
                if px is None:
                    px = _price_at(df_iid, d, close_col)
                if px is None:
                    nxt = next_bench_date_after(d)
                    if nxt is not None and nxt <= final_day:
                        pending_buy_s.setdefault(nxt, []).append(order)
                    else:
                        release_reservation()
                        pending_buy_ids_s.discard(iid)
                    continue

                spendable = cash_s * (1.0 - cfg.cash_buffer_pct)
                if np.isfinite(buy_budget_s):
                    spendable = min(spendable, buy_budget_s * (1.0 - cfg.cash_buffer_pct))
                eff_px = _apply_slippage(px, side="BUY", slippage_bps=cfg.slippage_bps)
                shares = int(math.floor((spendable - cfg.commission_per_trade) / eff_px))
                if shares < 1:
                    release_reservation()
                    pending_buy_ids_s.discard(iid)
                    continue
                cost = shares * eff_px + cfg.commission_per_trade
                if cost > cash_s:
                    release_reservation()
                    pending_buy_ids_s.discard(iid)
                    continue
                if np.isfinite(buy_budget_s) and cost > buy_budget_s:
                    # Not enough released budget yet; skip this month's buy.
                    release_reservation()
                    pending_buy_ids_s.discard(iid)
                    continue

                cash_s -= cost
                if np.isfinite(buy_budget_s):
                    buy_budget_s -= cost
                holdings_s[iid] = holdings_s.get(iid, 0) + shares
                labels_s[iid] = label
                entry_eff_s[iid] = float(eff_px)
                entry_date_s[iid] = pd.Timestamp(d)
                held_ids_s.add(iid)
                if cfg.monthly_sell_policy == "never" and cap_year is not None:
                    opened_new_names_by_year_s.setdefault(cap_year, set()).add(iid)
                release_reservation()
                pending_buy_ids_s.discard(iid)
                _update_price_cache_symbol(price_cache, iid, df_iid, open_col=open_col, low_col=low_col, close_col=close_col)
                table, ticker = label.split(":", 1) if ":" in label else ("", "")
                trades.append(
                    Trade(
                        date=d,
                        price_date=d,
                        portfolio="strategy",
                        side="BUY",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=shares,
                        raw_price=px,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=shares * eff_px,
                        cash_after=cash_s,
                    )
                )
                lots_s.append(
                    {
                        "instrument_id": iid,
                        "shares": shares,
                        "maturity_date": maturity_date,
                        "symbol_key": label,
                        "lastpricedate": lastpricedate,
                    }
                )
                planned_exit = (
                    pd.Timestamp(maturity_date)
                    if maturity_date is not None and pd.notna(maturity_date)
                    else pd.Timestamp(final_day)
                )
                lp = pd.Timestamp(lastpricedate) if lastpricedate is not None and pd.notna(lastpricedate) else None
                schedule_delist_credit(
                    pending_s,
                    instrument_id=iid,
                    shares=shares,
                    planned_exit=planned_exit,
                    lastpricedate=lp,
                    label_symbol_key=label,
                )

        # Pre-actions equity_low snapshot: measured before any MAE-triggered liquidations (portfolio or position).
        _s_val_pre, s_low_pre = _value_portfolio(date=d, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
        strategy_equity_low_pre_actions[i] = float(s_low_pre)
        if not cfg.no_benchmark:
            _b_val_pre, b_low_pre = _value_portfolio(date=d, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
            bm_equity_low_pre_actions[i] = float(b_low_pre)

        # Monthly portfolio MAE stop: if intraday equity_low breaches threshold vs the most recent select_date's start equity,
        # liquidate and remain in cash until the next select_date.
        if (
            cfg.mae_stop_pct > 0
            and not stopout_active
            and period_start_equity is not None
            and period_start_equity > 0
            and d not in select_dates_set
            and holdings_s
        ):
            mae_pct = (float(s_low_pre) / float(period_start_equity)) - 1.0
            if np.isfinite(mae_pct) and mae_pct <= -float(cfg.mae_stop_pct):
                stopout_active = True
                # Cooldown is defined in month buckets after the month containing the trigger date (benchmark month index).
                trigger_select_date = pd.Timestamp(sim_select_dates[sim_select_dates <= d].max())
                stopout_trigger_month_idx = int(month_idx_by_date[trigger_select_date])
                stopout_resume_month_idx = _stopout_resume_month_idx(stopout_trigger_month_idx, int(cfg.stopout_cooldown_months))
                stopout_release_select_date = pd.Timestamp(select_date_by_month_idx.get(stopout_resume_month_idx, final_day))
                stopouts.append(
                    {
                        "trigger_date": pd.Timestamp(d),
                        "trigger_select_date": trigger_select_date,
                        "trigger_month_idx": stopout_trigger_month_idx,
                        "cooldown_months": int(cfg.stopout_cooldown_months),
                        "resume_month_idx": stopout_resume_month_idx,
                        "release_select_date": stopout_release_select_date,
                        "period_start_equity": float(period_start_equity),
                        "stopout_mae_pct": float(mae_pct),
                        "mae_stop_pct": float(cfg.mae_stop_pct),
                    }
                )
                pending_buy_s.clear()
                pending_buy_ids_s.clear()
                pending_s.clear()
                pending_sell_ids_s.clear()
                reentry_remaining_s = 0.0
                reentry_months_left_s = 0
                buy_budget_s = float("inf")

                for iid, sh in list(holdings_s.items()):
                    cancel_pending_sell(iid)
                    _remove_lots_for_instrument(lots_s, iid)
                    label = labels_s.get(iid, "")
                    fill = _resolve_sell_fill(
                        ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=final_day
                    )
                    if fill is None:
                        forced = _force_liquidation_fill(ensure_series(iid), d, close_col=close_col)
                        if forced is None:
                            continue
                        fill = SellFill(
                            cash_date=pd.Timestamp(d),
                            price_date=pd.Timestamp(forced.price_date),
                            raw_price=float(forced.raw_price),
                        )
                    eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                    proceeds = sh * eff_px - cfg.commission_per_trade
                    if fill.cash_date != d:
                        cash_date = bench_date_on_or_after(fill.cash_date) or final_day
                        pending_s.setdefault(cash_date, []).append(
                            (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                        )
                        pending_sell_ids_s.add(iid)
                    else:
                        cash_s += proceeds
                        if np.isfinite(buy_budget_s):
                            buy_budget_s += float(proceeds)
                        holdings_s.pop(iid, None)
                        held_ids_s.discard(iid)
                        labels_s.pop(iid, None)
                        entry_eff_s.pop(iid, None)
                        entry_date_s.pop(iid, None)
                        pending_sell_ids_s.discard(iid)
                        table, ticker = label.split(":", 1) if ":" in label else ("", "")
                        trades.append(
                            Trade(
                                date=d,
                                price_date=fill.price_date,
                                portfolio="strategy",
                                side="SELL",
                                instrument_id=iid,
                                symbol_key=label,
                                ticker=ticker,
                                table=table,
                                shares=sh,
                                raw_price=fill.raw_price,
                                effective_price=eff_px,
                                commission=cfg.commission_per_trade,
                                slippage_bps=cfg.slippage_bps,
                                notional=sh * eff_px,
                                cash_after=cash_s,
                            )
                        )

        # Position MAE stop: liquidate positions that breach their own MAE threshold.
        if cfg.position_mae_stop_pct > 0 and not stopout_active and d not in select_dates_set and holdings_s:
            to_stop: list[tuple[str, int, float]] = []
            for iid, sh in holdings_s.items():
                if sh <= 0 or iid in pending_sell_ids_s:
                    continue
                entry_eff = float(entry_eff_s.get(iid, np.nan))
                mae_pct = _position_mae_pct_from_cache(
                    price_cache=price_cache,
                    instrument_id=iid,
                    date=d,
                    entry_effective_price=entry_eff,
                )
                if mae_pct is None:
                    continue
                if np.isfinite(mae_pct) and mae_pct <= -float(cfg.position_mae_stop_pct):
                    to_stop.append((iid, int(sh), float(mae_pct)))

            for iid, sh, mae_pct in to_stop:
                cancel_pending_sell(iid)
                _remove_lots_for_instrument(lots_s, iid)
                label = labels_s.get(iid, "")
                fill = _resolve_sell_fill(
                    ensure_series(iid), d, policy=sell_policy, open_col=open_col, close_col=close_col, final_day=final_day
                )
                if fill is None:
                    continue
                eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                cash_date = bench_date_on_or_after(fill.cash_date) or final_day
                position_stopouts.append(
                    {
                        "instrument_id": iid,
                        "symbol_key": label,
                        "ticker": label.split(":", 1)[1] if ":" in label else "",
                        "table": label.split(":", 1)[0] if ":" in label else "",
                        "entry_date": entry_date_s.get(iid, pd.NaT),
                        "entry_effective_price": float(entry_eff_s.get(iid, np.nan)),
                        "stop_trigger_date": pd.Timestamp(d),
                        "stop_trigger_mae_pct": float(mae_pct),
                        "shares": int(sh),
                        "fill_price_date": pd.Timestamp(fill.price_date),
                        "fill_cash_date": pd.Timestamp(cash_date),
                        "fill_raw_price": float(fill.raw_price),
                        "fill_effective_price": float(eff_px),
                        "commission_per_trade": float(cfg.commission_per_trade),
                        "slippage_bps": float(cfg.slippage_bps),
                    }
                )
                if cash_date != d:
                    pending_s.setdefault(cash_date, []).append(
                        (iid, sh, fill.raw_price, eff_px, proceeds, fill.price_date, label)
                    )
                    pending_sell_ids_s.add(iid)
                else:
                    cash_s += proceeds
                    holdings_s.pop(iid, None)
                    held_ids_s.discard(iid)
                    labels_s.pop(iid, None)
                    entry_eff_s.pop(iid, None)
                    entry_date_s.pop(iid, None)
                    pending_sell_ids_s.discard(iid)
                    table, ticker = label.split(":", 1) if ":" in label else ("", "")
                    trades.append(
                        Trade(
                            date=d,
                            price_date=fill.price_date,
                            portfolio="strategy",
                            side="SELL",
                            instrument_id=iid,
                            symbol_key=label,
                            ticker=ticker,
                            table=table,
                            shares=sh,
                            raw_price=fill.raw_price,
                            effective_price=eff_px,
                            commission=cfg.commission_per_trade,
                            slippage_bps=cfg.slippage_bps,
                            notional=sh * eff_px,
                            cash_after=cash_s,
                        )
                    )

        if withholding_rate > 0:
            gross = 0.0
            tax = 0.0
            for iid, sh in holdings_s.items():
                if sh <= 0:
                    continue
                div_ps = _implied_div_cash_per_share_on_date(div_cache, instrument_id=iid, df=ensure_series(iid), date=d)
                if div_ps <= 0:
                    continue
                g = float(sh) * float(div_ps)
                if not np.isfinite(g) or g <= 0:
                    continue
                gross += g
                tax += g * withholding_rate
            net = gross - tax
            if np.isfinite(net) and net > 0:
                cash_s += float(net)
                pending_div_sweep_s += float(net)
            strategy_div_gross[i] = gross
            strategy_div_tax[i] = tax
            strategy_div_net[i] = net
            maybe_sweep_dividend_cash(date=pd.Timestamp(d), portfolio="strategy")

            if not cfg.no_benchmark:
                gross_b = 0.0
                tax_b = 0.0
                for iid, sh in holdings_b.items():
                    if sh <= 0:
                        continue
                    div_ps = _implied_div_cash_per_share_on_date(div_cache, instrument_id=iid, df=ensure_series(iid), date=d)
                    if div_ps <= 0:
                        continue
                    g = float(sh) * float(div_ps)
                    if not np.isfinite(g) or g <= 0:
                        continue
                    gross_b += g
                    tax_b += g * withholding_rate
                net_b = gross_b - tax_b
                if np.isfinite(net_b) and net_b > 0:
                    cash_b += float(net_b)
                    pending_div_sweep_b += float(net_b)
                bm_div_gross[i] = gross_b
                bm_div_tax[i] = tax_b
                bm_div_net[i] = net_b
                maybe_sweep_dividend_cash(date=pd.Timestamp(d), portfolio="benchmark")

        if d == final_day:
            # Liquidate remaining holdings for both portfolios (force-close).
            for iid, sh in list(holdings_s.items()):
                label = labels_s.get(iid, "")
                fill = _force_liquidation_fill(ensure_series(iid), d, close_col=close_col)
                if fill is None:
                    continue
                eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                cash_s += proceeds
                holdings_s.pop(iid, None)
                held_ids_s.discard(iid)
                labels_s.pop(iid, None)
                entry_eff_s.pop(iid, None)
                entry_date_s.pop(iid, None)
                pending_sell_ids_s.discard(iid)
                table, ticker = label.split(":", 1) if ":" in label else ("", "")
                trades.append(
                    Trade(
                        date=d,
                        price_date=fill.price_date,
                        portfolio="strategy",
                        side="SELL",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=sh,
                        raw_price=fill.raw_price,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=sh * eff_px,
                        cash_after=cash_s,
                    )
                )
            for iid, sh in list(holdings_b.items()):
                label = labels_b.get(iid, bm_symbol_key)
                fill = _force_liquidation_fill(ensure_series(iid), d, close_col=close_col)
                if fill is None:
                    continue
                eff_px = _apply_slippage(fill.raw_price, side="SELL", slippage_bps=cfg.slippage_bps)
                proceeds = sh * eff_px - cfg.commission_per_trade
                cash_b += proceeds
                holdings_b.pop(iid, None)
                labels_b.pop(iid, None)
                table, ticker = label.split(":", 1)
                trades.append(
                    Trade(
                        date=d,
                        price_date=fill.price_date,
                        portfolio="benchmark",
                        side="SELL",
                        instrument_id=iid,
                        symbol_key=label,
                        ticker=ticker,
                        table=table,
                        shares=sh,
                        raw_price=fill.raw_price,
                        effective_price=eff_px,
                        commission=cfg.commission_per_trade,
                        slippage_bps=cfg.slippage_bps,
                        notional=sh * eff_px,
                        cash_after=cash_b,
                    )
                )

        s_val, s_low = _value_portfolio(date=d, cash=cash_s, holdings=holdings_s, price_cache=price_cache)
        b_val, b_low = _value_portfolio(date=d, cash=cash_b, holdings=holdings_b, price_cache=price_cache)
        strategy_equity[i] = s_val
        strategy_equity_low[i] = s_low
        bm_equity[i] = b_val
        bm_equity_low[i] = b_low

        if d in select_dates_set:
            period_start_equity = float(s_val)

    equity_df = pd.DataFrame(
        {
            "date": calendar,
            "equity": strategy_equity,
            "equity_low": strategy_equity_low,
            "equity_low_pre_actions": strategy_equity_low_pre_actions,
            "div_gross": strategy_div_gross,
            "div_tax": strategy_div_tax,
            "div_net": strategy_div_net,
        }
    )
    if not cfg.no_benchmark:
        equity_df["bm_equity"] = bm_equity
        equity_df["bm_equity_low"] = bm_equity_low
        equity_df["bm_equity_low_pre_actions"] = bm_equity_low_pre_actions
        equity_df["bm_div_gross"] = bm_div_gross
        equity_df["bm_div_tax"] = bm_div_tax
        equity_df["bm_div_net"] = bm_div_net

    pq.write_table(pa.Table.from_pandas(equity_df, preserve_index=False), out_root / "equity_curve.parquet", compression="zstd")
    if trades:
        pd.DataFrame([dataclasses.asdict(t) for t in trades]).to_csv(out_root / "trades.csv", index=False)
    if selected_rows:
        pq.write_table(
            pa.Table.from_pandas(pd.concat(selected_rows, ignore_index=True), preserve_index=False),
            out_root / "selected.parquet",
            compression="zstd",
        )
    pd.DataFrame(monthly_actions).to_csv(out_root / "monthly_actions.csv", index=False)
    pd.DataFrame(cash_flows).to_csv(out_root / "cash_flows.csv", index=False)
    if stopouts:
        pd.DataFrame(stopouts).to_csv(out_root / "portfolio_stopouts.csv", index=False)
    if position_stopouts:
        pd.DataFrame(position_stopouts).to_csv(out_root / "position_stopouts.csv", index=False)
    cal = _calendar_yearly_stats(equity_df)
    if not cfg.no_benchmark:
        bh = _benchmark_buy_hold_calendar_year_returns(calendar=calendar, price_cache=price_cache, instrument_id=bm_instrument_id)
        cal = cal.merge(bh, on="year", how="left")
    cal.to_csv(out_root / "calendar_yearly_equity.csv", index=False)
    _write_summary_stats(out_root=out_root, equity_df=equity_df, cash_flows=cash_flows, cfg=cfg)

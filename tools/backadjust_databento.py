"""
Utilities for backadjusting Databento futures data and validating against TradingView exports.

Usage:
    uv run python tools/backadjust_databento.py --help
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import pytz
import structlog
import typer

logger = structlog.get_logger(__name__)

try:
    from databento import DBNStore  # type: ignore
except Exception:  # pragma: no cover
    DBNStore = None  # type: ignore

SessionWindow = Tuple[Optional[datetime], Optional[datetime]]


@dataclass(frozen=True)
class RolloverSpec:
    """Describes when the volume rolls from one contract to the next."""

    effective: datetime  # UTC timestamp of the roll
    from_contract: str
    to_contract: str


def _cme_rollover_dt(year: int, month: int, day: int) -> datetime:
    """Return the rollover datetime localized to 17:00 America/Chicago."""
    cme_tz = pytz.timezone("America/Chicago")
    local_dt = cme_tz.localize(datetime(year, month, day, 17, 0))
    return local_dt.astimezone(pytz.utc)


TRADINGVIEW_ROLLOVERS: Sequence[RolloverSpec] = (
    RolloverSpec(_cme_rollover_dt(2023, 12, 12), "ESZ2023", "ESH2024"),
    RolloverSpec(_cme_rollover_dt(2024, 3, 12), "ESH2024", "ESM2024"),
    RolloverSpec(_cme_rollover_dt(2024, 6, 17), "ESM2024", "ESU2024"),
    RolloverSpec(_cme_rollover_dt(2024, 9, 17), "ESU2024", "ESZ2024"),
    RolloverSpec(_cme_rollover_dt(2024, 12, 17), "ESZ2024", "ESH2025"),
    RolloverSpec(_cme_rollover_dt(2025, 3, 18), "ESH2025", "ESM2025"),
    RolloverSpec(_cme_rollover_dt(2025, 6, 16), "ESM2025", "ESU2025"),
    RolloverSpec(_cme_rollover_dt(2025, 9, 16), "ESU2025", "ESZ2025"),
)

CONTRACT_PATTERN = re.compile(r"(ES[HMUZ]\d{4})", flags=re.IGNORECASE)
DEFAULT_PRICE_COLS = ("open", "high", "low", "close")
CONTRACT_SORT_RE = re.compile(r"ES([HMUZ])(\d{4})", flags=re.IGNORECASE)
MONTH_TO_ORDER = {"H": 3, "M": 6, "U": 9, "Z": 12}


def parse_timestamp_option(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse CLI timestamp options into timezone-aware UTC values."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def canonical_contract(symbol: str, ts_event: pd.Timestamp) -> Optional[str]:
    """Convert Databento symbols into canonical contract codes like ESU2024."""
    if not symbol:
        return None

    clean = symbol.upper()
    if "-" in clean or not clean.startswith("ES"):
        return None

    month_code = clean[2]
    year_part = clean[3:]
    digits = "".join(ch for ch in year_part if ch.isdigit())
    if not digits:
        return None

    year = None
    if len(digits) >= 4:
        year = int(digits[:4])
    elif len(digits) == 2:
        century = (ts_event.year // 100) * 100
        year = century + int(digits)
        if year < ts_event.year - 50:
            year += 100
        elif year > ts_event.year + 50:
            year -= 100
    else:  # single digit year code
        decade = (ts_event.year // 10) * 10
        year = decade + int(digits)
        if year < ts_event.year - 5:
            year += 10
        elif year > ts_event.year + 5:
            year -= 10

    if year is None:
        return None

    return f"ES{month_code}{year}"


def infer_contract_from_path(path: Path) -> str:
    """Infer the contract symbol from the file name."""
    match = CONTRACT_PATTERN.search(path.name.upper())
    if not match:
        raise ValueError(f"Unable to infer ES contract from file name: {path.name}")
    return match.group(1).upper()


def normalize_timestamp_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Detect and normalize the timestamp column to UTC."""
    timestamp_candidates = ("ts_event", "timestamp", "time", "datetime")
    for candidate in timestamp_candidates:
        if candidate in df.columns:
            ts_col = candidate
            break
    else:
        raise ValueError("Databento CSV must contain a timestamp column (ts_event, timestamp, time, or datetime).")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df, ts_col


def load_databento_frames(
    path: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load Databento data (CSV per contract or .dbn.zst archive) keyed by contract.

    Each frame is indexed by UTC timestamp and only contains OHLCV columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Databento path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() == ".csv":
            return _load_frames_from_csv([path], start, end)
        if path.suffix.lower().endswith(".zst"):
            return _load_frames_from_dbn([path], start, end)
        raise ValueError(f"Unsupported databento file type: {path}")

    csv_files = sorted(path.glob("*.csv"))
    dbn_files = sorted(path.glob("*.dbn.zst"))

    if csv_files:
        return _load_frames_from_csv(csv_files, start, end)
    if dbn_files:
        return _load_frames_from_dbn(dbn_files, start, end)

    raise ValueError(f"No CSV or DBN files were found under {path}")


def _slice_frame(frame: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if start is not None:
        frame = frame[frame.index >= start]
    if end is not None:
        frame = frame[frame.index < end]
    return frame


def _load_frames_from_csv(
    files: Sequence[Path],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for csv_file in files:
        contract = infer_contract_from_path(csv_file)

        df = pd.read_csv(csv_file)
        df.columns = [col.strip().lower() for col in df.columns]
        df, ts_col = normalize_timestamp_column(df)

        rename_map = {"vol": "volume", "size": "volume"}
        df = df.rename(columns=rename_map)
        missing_cols = [col for col in DEFAULT_PRICE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{csv_file} is missing price columns: {missing_cols}")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        frame = df[[ts_col, *DEFAULT_PRICE_COLS, "volume"]].copy()
        frame.rename(columns={ts_col: "timestamp"}, inplace=True)
        frame.sort_values("timestamp", inplace=True)
        frame = frame.set_index("timestamp")
        frame = _slice_frame(frame, start, end)
        if frame.empty:
            continue
        frames[contract] = frame
        logger.info("loaded_contract_csv", contract=contract, rows=len(frame), path=str(csv_file))

    if not frames:
        raise ValueError("No contract CSV data matched the requested filters.")

    return frames


def _load_frames_from_dbn(
    files: Sequence[Path],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Dict[str, pd.DataFrame]:
    if DBNStore is None:  # pragma: no cover
        raise ImportError("databento is required for DBN ingestion. Install with `uv sync --extra data`.")
    chunks: Dict[str, list[pd.DataFrame]] = defaultdict(list)

    for dbn_file in files:
        store = DBNStore.from_file(dbn_file)
        df = store.to_df().reset_index()
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        df = df.sort_values("ts_event")
        if start is not None:
            df = df[df["ts_event"] >= start]
        if end is not None:
            df = df[df["ts_event"] < end]
        if df.empty:
            continue

        df["contract"] = [canonical_contract(symbol, ts_event) for symbol, ts_event in zip(df["symbol"], df["ts_event"])]
        df = df[df["contract"].notna()]
        if df.empty:
            continue

        for contract, group in df.groupby("contract", sort=False):
            subset = group.set_index("ts_event")[["open", "high", "low", "close", "volume"]].copy()
            subset.index.name = "timestamp"
            chunks[contract].append(subset)

    frames: Dict[str, pd.DataFrame] = {}
    for contract, parts in chunks.items():
        frame = pd.concat(parts).sort_index()
        frame = _slice_frame(frame, start, end)
        if frame.empty:
            continue
        frames[contract] = frame
        logger.info("loaded_contract_dbn", contract=contract, rows=len(frame))

    if not frames:
        raise ValueError("No contract data could be derived from the provided DBN files.")

    return frames


def build_contract_windows(rollovers: Sequence[RolloverSpec]) -> Dict[str, SessionWindow]:
    """Return time windows for each contract based on the rollover schedule."""
    windows: Dict[str, SessionWindow] = {}
    for rollover in rollovers:
        start, end = windows.get(rollover.from_contract, (None, None))
        windows[rollover.from_contract] = (start, rollover.effective)

        to_start, to_end = windows.get(rollover.to_contract, (None, None))
        if to_start is None or rollover.effective < to_start:
            to_start = rollover.effective
        windows[rollover.to_contract] = (to_start, to_end)
    return windows


def contract_sort_key(contract: str) -> tuple[int, int]:
    match = CONTRACT_SORT_RE.fullmatch(contract.upper())
    if not match:
        return (9999, 99)
    letter, year = match.groups()
    return (int(year), MONTH_TO_ORDER.get(letter, 99))


def extend_rollovers_with_data(
    frames: Dict[str, pd.DataFrame],
    base_rollovers: Sequence[RolloverSpec],
) -> Sequence[RolloverSpec]:
    """Ensure rollovers cover every available contract by inferring gaps from data."""
    known_pairs = {(r.from_contract, r.to_contract) for r in base_rollovers}
    sorted_contracts = sorted(frames.keys(), key=contract_sort_key)
    additional: list[RolloverSpec] = []

    for idx in range(len(sorted_contracts) - 1):
        from_contract = sorted_contracts[idx]
        to_contract = sorted_contracts[idx + 1]
        if (from_contract, to_contract) in known_pairs:
            continue
        from_frame = frames[from_contract]
        effective = from_frame.index.max() - timedelta(days=1)
        start_ts = from_frame.index.min()
        if effective < start_ts:
            effective = start_ts
        effective = effective.tz_convert("UTC") if effective.tzinfo else effective.tz_localize("UTC")
        additional.append(
            RolloverSpec(
                effective=effective.to_pydatetime(),
                from_contract=from_contract,
                to_contract=to_contract,
            )
        )

    all_rollovers = list(base_rollovers) + additional
    all_rollovers.sort(key=lambda r: r.effective)
    return all_rollovers


def _price_before(df: pd.DataFrame, dt: datetime, tolerance: timedelta) -> Optional[float]:
    window = df[(df.index <= dt) & (df.index >= dt - tolerance)]
    if window.empty:
        return None
    last_row = window.iloc[-1]
    return float(last_row["close"])


def _price_after(df: pd.DataFrame, dt: datetime, tolerance: timedelta) -> Optional[float]:
    window = df[(df.index >= dt) & (df.index <= dt + tolerance)]
    if window.empty:
        return None
    first_row = window.iloc[0]
    return float(first_row["close"])


def compute_adjustments(
    frames: Dict[str, pd.DataFrame],
    rollovers: Sequence[RolloverSpec],
    tolerance: timedelta = timedelta(minutes=15),
) -> Dict[str, float]:
    """
    Compute additive adjustments so earlier contracts line up with the latest contract.
    """
    adjustments: Dict[str, float] = {}
    if not rollovers:
        return adjustments

    base_contract = rollovers[-1].to_contract
    adjustments[base_contract] = 0.0

    for rollover in reversed(rollovers):
        if rollover.to_contract not in adjustments:
            continue
        if rollover.to_contract not in frames or rollover.from_contract not in frames:
            logger.warning(
                "missing_contract_for_rollover",
                from_contract=rollover.from_contract,
                to_contract=rollover.to_contract,
            )
            continue

        to_df = frames[rollover.to_contract]
        from_df = frames[rollover.from_contract]
        target_dt = rollover.effective

        to_price = _price_after(to_df, target_dt, tolerance)
        from_price = _price_before(from_df, target_dt, tolerance)

        if to_price is None or from_price is None:
            logger.warning(
                "missing_prices_for_rollover",
                from_contract=rollover.from_contract,
                to_contract=rollover.to_contract,
                effective=str(target_dt),
            )
            adjustments[rollover.from_contract] = adjustments[rollover.to_contract]
            continue

        delta = to_price - from_price
        adjustments[rollover.from_contract] = adjustments[rollover.to_contract] + delta
        logger.info(
            "computed_adjustment",
            from_contract=rollover.from_contract,
            to_contract=rollover.to_contract,
            delta=delta,
            cumulative_adjustment=adjustments[rollover.from_contract],
        )

    return adjustments


def apply_backadjustments(
    frames: Dict[str, pd.DataFrame],
    adjustments: Dict[str, float],
    windows: Dict[str, SessionWindow],
) -> pd.DataFrame:
    """Apply contract windows and adjustments to create a continuous series."""
    adjusted_frames = []
    for contract, frame in frames.items():
        adj = adjustments.get(contract, 0.0)
        start, end = windows.get(contract, (None, None))

        filtered = frame.copy()
        if start:
            filtered = filtered[filtered.index >= start]
        if end:
            filtered = filtered[filtered.index < end]
        if filtered.empty:
            continue

        for col in DEFAULT_PRICE_COLS:
            filtered[col] = filtered[col] + adj
        adjusted_frames.append(filtered)
        logger.info(
            "applied_adjustment",
            contract=contract,
            start=str(start) if start else None,
            end=str(end) if end else None,
            rows=len(filtered),
            adjustment=adj,
        )

    if not adjusted_frames:
        raise ValueError("No contract data remained after applying windows.")

    combined = pd.concat(adjusted_frames)
    combined.sort_index(inplace=True)
    return combined


def parse_timeframe(value: str) -> str:
    """Convert a human-friendly timeframe such as '30m' into a pandas rule."""
    value = value.strip().lower()
    match = re.fullmatch(r"(?P<number>\d+)\s*(?P<unit>[mhd])", value)
    if not match:
        raise ValueError("Timeframe must look like '30m', '1h', or '1d'.")

    number = int(match.group("number"))
    unit = match.group("unit")
    unit_map = {"m": "min", "h": "H", "d": "D"}
    return f"{number}{unit_map[unit]}"


def resample_with_session(
    df: pd.DataFrame,
    rule: str,
    session_start: time = time(17, 0),
    session_timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """Resample while aligning bar boundaries to the CME session open."""
    if df.empty:
        return df

    local = df.tz_convert(session_timezone)
    offset = timedelta(hours=session_start.hour, minutes=session_start.minute)
    local = local.copy()
    local.index = local.index - offset

    agg = local.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg.dropna(subset=["open", "high", "low", "close"], inplace=True)
    agg.index = agg.index + offset
    agg = agg.tz_convert("UTC")
    agg.index.name = "timestamp"
    return agg


def load_tradingview_csv(path: Path, timezone: str = "UTC") -> pd.DataFrame:
    """Load a TradingView export and normalize timestamps + column names."""
    if not path.exists():
        raise FileNotFoundError(f"TradingView CSV does not exist: {path}")

    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    time_col = None
    for candidate in ("time", "timestamp", "date"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError("TradingView CSV must include a `time` column.")

    df = df.rename(columns={"vol": "volume", "value": "volume"})
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    tz = pytz.timezone(timezone)
    if df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize(tz)
    else:
        df[time_col] = df[time_col].dt.tz_convert(tz)
    df[time_col] = df[time_col].dt.tz_convert("UTC")

    missing_cols = [col for col in DEFAULT_PRICE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"TradingView CSV is missing columns: {missing_cols}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df.set_index(time_col)
    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def validate_against_tradingview(adjusted: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    """Join adjusted data with TradingView data to compare price differences."""
    joined = adjusted.join(tv, how="inner", lsuffix="_db", rsuffix="_tv")
    if joined.empty:
        logger.warning("validation_no_overlap")
        return joined

    discrepancy_mask = pd.Series(False, index=joined.index)
    summary = {}
    for col in DEFAULT_PRICE_COLS:
        diff = (joined[f"{col}_db"] - joined[f"{col}_tv"]).abs()
        summary[col] = float(diff.max())
        discrepancy_mask |= diff > 0
    summary["count"] = int(discrepancy_mask.sum())
    logger.info("validation_summary_max_abs_diff", **summary)
    return joined


def infer_timeframe(interval_index: pd.Index) -> Optional[timedelta]:
    """Infer the dominant bar interval from an index of timestamps."""
    if not isinstance(interval_index, pd.DatetimeIndex):
        return None
    diffs = interval_index.to_series().diff().dropna()
    if diffs.empty:
        return None
    counts = diffs.value_counts()
    if counts.empty:
        return None
    return counts.idxmax().to_pytimedelta()


def _round_to_tick(value: float, tick_size: float) -> float:
    if tick_size <= 0:
        return value
    return round(value / tick_size) * tick_size


def fit_adjustments_to_reference(
    frames: Dict[str, pd.DataFrame],
    windows: Dict[str, SessionWindow],
    rule: str,
    reference_df: pd.DataFrame,
    session_start: time,
    session_timezone: str,
    tick_size: float,
) -> Dict[str, float]:
    """Derive contract adjustments directly from a trusted reference series."""
    adjustments: Dict[str, float] = {}
    for contract, frame in frames.items():
        start, end = windows.get(contract, (None, None))
        localized = frame if frame.index.tz is not None else frame.tz_localize("UTC")
        subset = localized
        if start:
            subset = subset[subset.index >= start]
        if end:
            subset = subset[subset.index < end]
        if subset.empty:
            continue

        contract_resampled = resample_with_session(
            subset,
            rule=rule,
            session_start=session_start,
            session_timezone=session_timezone,
        )
        if contract_resampled.empty:
            continue

        ref_slice = reference_df
        if start:
            ref_slice = ref_slice[ref_slice.index >= start]
        if end:
            ref_slice = ref_slice[ref_slice.index < end]
        if ref_slice.empty:
            continue

        joined = contract_resampled.join(ref_slice, how="inner", lsuffix="_raw", rsuffix="_ref")
        if joined.empty:
            continue

        deltas = joined["close_ref"] - joined["close_raw"]
        delta = deltas.median()
        if pd.isna(delta):
            continue

        adjustment = _round_to_tick(float(delta), tick_size)
        adjustments[contract] = adjustment
        logger.info(
            "fit_reference_contract",
            contract=contract,
            adjustment=adjustment,
            samples=len(joined),
        )

    return adjustments


def backadjust_databento(
    databento_path: Path,
    timeframe: str = "30m",
    rollovers: Sequence[RolloverSpec] = TRADINGVIEW_ROLLOVERS,
    session_start: time = time(17, 0),
    session_timezone: str = "America/Chicago",
    reference_df: Optional[pd.DataFrame] = None,
    tick_size: float = 0.25,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Top-level convenience wrapper used by both CLI and tests."""
    frames = load_databento_frames(databento_path, start=start, end=end)
    rollovers = extend_rollovers_with_data(frames, rollovers)
    windows = build_contract_windows(rollovers)
    rule = parse_timeframe(timeframe)
    baseline_adjustments = compute_adjustments(frames, rollovers)
    combined = apply_backadjustments(frames, baseline_adjustments, windows)
    combined = combined.tz_localize("UTC") if combined.index.tz is None else combined

    if reference_df is not None:
        ref_interval = infer_timeframe(reference_df.index)
        try:
            rule_interval = pd.to_timedelta(rule)
        except ValueError:
            rule_interval = None

        if ref_interval and rule_interval and abs(ref_interval - rule_interval) <= timedelta(minutes=1):
            reference_adjustments = fit_adjustments_to_reference(
                frames=frames,
                windows=windows,
                rule=rule,
                reference_df=reference_df,
                session_start=session_start,
                session_timezone=session_timezone,
                tick_size=tick_size,
            )
            if reference_adjustments:
                merged_adjustments = {**baseline_adjustments, **reference_adjustments}
                combined = apply_backadjustments(frames, merged_adjustments, windows)
                combined = combined.tz_localize("UTC") if combined.index.tz is None else combined
        else:
            logger.warning(
                "reference_timeframe_mismatch",
                reference_interval=str(ref_interval),
                requested_rule=rule,
            )

    resampled = resample_with_session(
        combined,
        rule,
        session_start=session_start,
        session_timezone=session_timezone,
    )
    return resampled


def main(
    databento_path: Path = typer.Option(
        ...,
        "--databento-path",
        "--databento-dir",
        exists=True,
        help="Databento directory (per-contract CSVs) or .dbn.zst file.",
    ),
    timeframe: str = typer.Option("30m", help="Target timeframe such as '30m', '1h', or '1d'."),
    output_csv: Optional[Path] = typer.Option(None, help="Optional path to store the backadjusted data."),
    tradingview_csv: Optional[Path] = typer.Option(
        None,
        help="If provided, the script will compare the output against this TradingView CSV.",
    ),
    tradingview_timezone: str = typer.Option(
        "America/Chicago",
        help="Timezone of the TradingView timestamps (defaults to CME local time).",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        help="Optional UTC start timestamp (e.g., '2018-01-01') for Databento data.",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        help="Optional UTC end timestamp (exclusive) for Databento data.",
    ),
) -> None:
    """CLI entry point."""
    reference_df = None
    if tradingview_csv:
        reference_df = load_tradingview_csv(tradingview_csv, timezone=tradingview_timezone)

    start_ts = parse_timestamp_option(start_date)
    end_ts = parse_timestamp_option(end_date)

    adjusted = backadjust_databento(
        databento_path=databento_path,
        timeframe=timeframe,
        reference_df=reference_df,
        start=start_ts,
        end=end_ts,
    )
    typer.echo(f"Generated {len(adjusted)} {timeframe} bars from Databento data.")

    if reference_df is not None:
        joined = validate_against_tradingview(adjusted, reference_df)
        typer.echo(f"Validation rows: {len(joined)}")

    if output_csv:
        adjusted.to_csv(output_csv)
        typer.echo(f"Saved backadjusted data to {output_csv}")


if __name__ == "__main__":
    typer.run(main)

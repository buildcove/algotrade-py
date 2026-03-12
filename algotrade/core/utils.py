"""
Core utilities for the algotrade system.

This module provides utilities organized into logical sections:
- Time and timezone utilities
- Market scheduling functions
- Data processing utilities
- Trading calculations
- System utilities
- Data source integrations
"""

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import holidays
import jwt
import numpy as np
import pandas as pd
import pytz
import structlog
from algotrade.core import types
from algotrade.schedules import (
    CBOT_SCHEDULE,
    CME_SCHEDULE,
    CME_TZ,
    CONTRACT_TO_EXCHANGE,
    Exchange,
    TimezoneDST,
)
from algotrade.settings import settings
from algotrade.tv import get_token as get_tv_auth
from tvDatafeed import TvDatafeed

logger = structlog.get_logger()


# =============================================================================
# TIME AND TIMEZONE UTILITIES
# =============================================================================


def get_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(pytz.utc)


def get_eastern_timezone_label(dt: datetime) -> TimezoneDST:
    """Get the US/Eastern timezone label (EST/EDT) for a given datetime."""
    eastern = pytz.timezone("US/Eastern")

    if dt.tzinfo is None:
        localized = eastern.localize(dt)
    else:
        localized = dt.astimezone(eastern)

    tz_name = localized.tzname()  # returns 'EST' or 'EDT'
    return TimezoneDST(tz_name)


def get_central_timezone_label(dt: datetime) -> TimezoneDST:
    """Get the US/Central timezone label (CST/CDT) for a given datetime."""
    central = pytz.timezone("US/Central")

    if dt.tzinfo is None:
        localized = central.localize(dt)
    else:
        localized = dt.astimezone(central)

    tz_name = localized.tzname()  # returns 'CST' or 'CDT'
    return TimezoneDST(tz_name)


def get_todays_opening_datetime(timezone: str = CME_TZ) -> datetime:
    """Get today's market opening datetime, accounting for holidays."""

    def _get_opening_date(dt):
        if dt.date() in holidays.NYSE():
            dt = dt - timedelta(days=1)
            return _get_opening_date(dt)
        return dt - timedelta(days=1)

    dt = datetime.now(pytz.timezone(timezone))
    dt = _get_opening_date(dt)
    return datetime.combine(dt.date(), time(17, 0))


# =============================================================================
# MARKET SCHEDULING FUNCTIONS
# =============================================================================


def time_until_next_globex_bar(
    stream: types.Candle | types.Stream = None,
    tz: str = "UTC",
) -> float:
    """Calculate seconds until the next CME Globex 4-hour bar."""
    now = stream.timestamp if stream else datetime.now(pytz.timezone(tz))

    # Get EDT/EST label based on current time in Eastern
    tz_label = get_eastern_timezone_label(now)
    bar_times = CME_SCHEDULE.four_hour_times_utc.get(tz_label)

    if not bar_times:
        raise ValueError(f"Timezone '{tz_label}' not found in CME Globex bar times.")

    today = now.date()

    # Convert each bar_time to a datetime in Eastern, then convert to UTC
    bar_datetimes = []
    for bar_time in bar_times:
        bar_dt = datetime.combine(today, bar_time).astimezone(pytz.timezone(tz))
        bar_datetimes.append(bar_dt)

    for bar_dt in bar_datetimes:
        if now < bar_dt:
            return (bar_dt - now).total_seconds()

    # All bars passed — use the first bar of next day
    next_day = today + timedelta(days=1)
    next_bar = datetime.combine(next_day, bar_times[0]).astimezone(pytz.timezone(tz))
    return (next_bar - now).total_seconds()


def next_half_hour(now: Optional[datetime] = None) -> datetime:
    """Get the next 30-minute boundary."""
    if now is None:
        now = get_utc()

    minute = now.minute
    if minute < 30:
        next_run = now.replace(minute=30, second=0, microsecond=0)
    else:
        next_run = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return next_run


def next_half_hour_for_contract(contract: types.SupportedContract, now: Optional[datetime] = None) -> datetime:
    """
    Get the next tradable 30-minute boundary (UTC) for a given contract.

    This mirrors the weekend break logic used by the 4H/daily helpers:
    - CME-group: closed from Fri 22:00 UTC through the CME open hour on Sunday.
    - CBOT: closed from Fri close hour (19/20 UTC) through the CBOT open hour on Sunday.
    """
    now = now or get_utc()
    exchange = CONTRACT_TO_EXCHANGE.get(contract, Exchange.CME)

    def _blocked(dt: datetime) -> bool:
        wd = dt.weekday()
        if wd == 5:
            return True
        if exchange == Exchange.CBOT:
            tz_label = get_central_timezone_label(dt)
            open_hour = CBOT_SCHEDULE.open_times_utc[tz_label].hour
            close_hour = 20 if tz_label == TimezoneDST.CST else 19
            return (wd == 4 and dt.hour >= close_hour) or (wd == 6 and dt.hour < open_hour)

        tz_label = get_eastern_timezone_label(dt)
        open_hour = CME_SCHEDULE.open_times_utc[tz_label].hour
        return (wd == 4 and dt.hour >= 22) or (wd == 6 and dt.hour < open_hour)

    candidate = next_half_hour(now)
    for _ in range(7 * 48):  # 7 days worth of 30m slots
        if not _blocked(candidate):
            return candidate
        candidate += timedelta(minutes=30)
    raise RuntimeError("Unable to find next tradable half-hour within one week")


def next_4hour(now: Optional[datetime] = None, skip_sunday: bool = False) -> datetime:
    """
    Return the next tradable 4-hour Globex bar (UTC).

    Parameters
    ----------
    now : datetime | None
        Current UTC time. If None, uses current UTC time.
    skip_sunday : bool, default False
        If True, ignore all Sunday bars (even those after 22 UTC).

    Notes
    -----
    • Globex is closed from Fri 22 UTC through Sun 22 UTC.
    • Uses CME schedule for bar times based on current US-Eastern session.
    """
    if now is None:
        now = get_utc()

    tz_label = get_eastern_timezone_label(now)
    bar_times = CME_SCHEDULE.four_hour_times_utc[tz_label]

    def _blocked(dt):
        wd = dt.weekday()
        return (
            wd == 5  # Saturday
            or (wd == 4 and dt.hour >= 22)  # Fri ≥ 22 UTC
            or (wd == 6 and dt.hour < 22)  # Sun < 22 UTC
            or (skip_sunday and wd == 6)  # optionally skip Sunday
        )

    # Search forward, day by day, bar by bar
    for day_offset in range(7):  # a full week is plenty
        midnight = (now + timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
        for t in bar_times:
            candidate = midnight.replace(hour=t.hour, minute=t.minute)
            if candidate <= now or _blocked(candidate):
                continue
            return candidate

    # Fallback (should never be reached)
    candidate = now + timedelta(hours=4)
    while _blocked(candidate):
        candidate += timedelta(hours=4)
    return candidate


def next_4hour_cbot(now: Optional[datetime] = None, skip_sunday: bool = False) -> datetime:
    """
    Return the next tradable 4-hour CBOT bar (UTC).

    Parameters
    ----------
    now : datetime | None
        Current UTC time. If None, uses current UTC time.
    skip_sunday : bool, default False
        If True, ignore all Sunday bars.

    Notes
    -----
    • CBOT is closed from Fri 2pm CT (20/19 UTC) through Sun 5pm CT (22/23 UTC).
    • Uses CBOT schedule for bar times based on current US-Central session.
    """
    if now is None:
        now = get_utc()

    tz_label = get_central_timezone_label(now)
    bar_times = CBOT_SCHEDULE.four_hour_times_utc[tz_label]

    def _blocked(dt):
        wd = dt.weekday()
        # CBOT closes Friday 2pm CT (20 UTC during CST, 19 UTC during CDT)
        close_hour = 20 if tz_label == TimezoneDST.CST else 19
        return (
            wd == 5  # Saturday
            or (wd == 4 and dt.hour >= close_hour)  # Fri >= close time
            or (wd == 6 and dt.hour < (23 if tz_label == TimezoneDST.CST else 22))  # Sun before open
            or (skip_sunday and wd == 6)  # optionally skip Sunday
        )

    # Search forward, day by day, bar by bar
    for day_offset in range(7):  # a full week is plenty
        midnight = (now + timedelta(days=day_offset)).replace(hour=0, minute=0, second=0, microsecond=0)
        for t in bar_times:
            candidate = midnight.replace(hour=t.hour, minute=t.minute)
            if candidate <= now or _blocked(candidate):
                continue
            return candidate

    # Fallback (should never be reached)
    candidate = now + timedelta(hours=4)
    while _blocked(candidate):
        candidate += timedelta(hours=4)
    return candidate


def next_daily(now: Optional[datetime] = None) -> datetime:
    """
    Return the next tradable daily Globex bar open (UTC).

    • A daily bar opens every calendar day at 22:00/23:00 UTC.
    • Globex is closed from after Friday-22:00 open until Sunday-22:00.

    The function always returns the first 22/23-UTC bar strictly after `now`.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    label = get_eastern_timezone_label(now)
    OPEN_HOUR = CME_SCHEDULE.open_times_utc[label].hour

    def _blocked(dt: datetime) -> bool:
        """True if the candidate falls in the weekend break."""
        wd = dt.weekday()  # Mon=0 … Sun=6
        return wd == 5 or (wd == 6 and dt.hour < OPEN_HOUR)  # Saturday or Sun before open

    # Scan forward day-by-day
    for day_offset in range(8):  # a full week is ample
        cand_date = (now + timedelta(days=day_offset)).date()
        candidate = datetime(cand_date.year, cand_date.month, cand_date.day, OPEN_HOUR, 0, 0, 0, tzinfo=timezone.utc)

        if candidate <= now or _blocked(candidate):
            continue
        return candidate

    # Fallback (should never be reached)
    candidate = now.replace(hour=OPEN_HOUR, minute=0, second=0, microsecond=0)
    while candidate <= now or _blocked(candidate):
        candidate += timedelta(days=1)
    return candidate


def next_daily_cbot(now: Optional[datetime] = None) -> datetime:
    """
    Return the next tradable daily CBOT bar open (UTC).

    • A daily bar opens every calendar day at the CBOT opening time.
    • CBOT is closed from Friday 2pm CT until Sunday 5pm CT.
    • Uses Central timezone for determining the opening hour.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    label = get_central_timezone_label(now)
    OPEN_HOUR = CBOT_SCHEDULE.open_times_utc[label].hour

    def _blocked(dt: datetime) -> bool:
        """True if the candidate falls in the weekend break."""
        wd = dt.weekday()  # Mon=0 … Sun=6
        # CBOT closes Friday 2pm CT (20 UTC during CST, 19 UTC during CDT)
        close_hour = 20 if label == TimezoneDST.CST else 19
        return (
            wd == 5  # Saturday
            or (wd == 4 and dt.hour >= close_hour)  # Fri >= close time
            or (wd == 6 and dt.hour < OPEN_HOUR)  # Sun before open
        )

    # Scan forward day-by-day
    for day_offset in range(8):  # a full week is ample
        cand_date = (now + timedelta(days=day_offset)).date()
        candidate = datetime(cand_date.year, cand_date.month, cand_date.day, OPEN_HOUR, 0, 0, 0, tzinfo=timezone.utc)

        if candidate <= now or _blocked(candidate):
            continue
        return candidate

    # Fallback (should never be reached)
    candidate = now.replace(hour=OPEN_HOUR, minute=0, second=0, microsecond=0)
    while candidate <= now or _blocked(candidate):
        candidate += timedelta(days=1)
    return candidate


def next_friday_20utc(now: Optional[datetime] = None) -> datetime:
    """Return the next occurrence of Friday 20:00 UTC."""
    now = now or datetime.now(timezone.utc)

    TARGET_WD = 4  # Monday=0 … Sunday=6 → Friday=4
    days_ahead = (TARGET_WD - now.weekday()) % 7

    # Candidate Friday 20:00 UTC of this week
    candidate = (now + timedelta(days=days_ahead)).replace(hour=20, minute=0, second=0, microsecond=0)

    # If we're already past that moment, jump ahead one week
    if candidate <= now:
        candidate += timedelta(days=7)

    return candidate


def next_4hour_for_contract(
    contract: types.SupportedContract, now: Optional[datetime] = None, skip_sunday: bool = False
) -> datetime:
    """
    Return the next tradable 4-hour bar (UTC) for the given contract.
    Uses the appropriate scheduling function based on the contract's exchange.
    """
    exchange_schedule = CONTRACT_TO_EXCHANGE.get(contract, Exchange.CME)

    if exchange_schedule == Exchange.CBOT:
        return next_4hour_cbot(now, skip_sunday)
    else:  # Default to CME for all others
        return next_4hour(now, skip_sunday)


def next_daily_for_contract(contract: types.SupportedContract, now: Optional[datetime] = None) -> datetime:
    """
    Return the next tradable daily bar (UTC) for the given contract.
    Uses the appropriate scheduling function based on the contract's exchange.
    """
    exchange_schedule = CONTRACT_TO_EXCHANGE.get(contract, Exchange.CME)

    if exchange_schedule == Exchange.CBOT:
        return next_daily_cbot(now)
    else:  # Default to CME for all others
        return next_daily(now)


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================


def adjust_daily_dates(df: pd.DataFrame, start_time: time = time(17, 0), end_time: time = time(15, 59)) -> pd.DataFrame:
    """
    Adjust daily trading dates to account for market opening times and holidays.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'dt' column containing datetime data
    start_time : time, default time(17, 0)
        Market opening time
    end_time : time, default time(15, 59)
        Market closing time

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted date columns (a_dt, a_open, a_high, a_low, a_close, a_dir)
    """
    # Find the index of the first occurrence of start_time
    start_index = df[df["dt"].dt.time == start_time].index[0]
    df = df[start_index:].reset_index(drop=True)

    # Build set of non-trading days (holidays)
    year = int(datetime.now().year)
    non_trading_days = set()
    for year in range(2007, year + 1):
        for dt, name in sorted(holidays.NYSE(years=year).items()):
            non_trading_days.add(dt)

    # Adjust dates accounting for holidays
    is_holiday = df["dt"].dt.date.isin(non_trading_days)
    is_day_open_time = df["dt"].dt.time == start_time
    df["a_dt"] = pd.NaT
    df["a_dt"] = np.where(~is_holiday & is_day_open_time, df["dt"] + pd.Timedelta(days=1), df["a_dt"])
    df["a_dt"] = df["a_dt"].ffill()
    df["a_dt"] = np.where(df["a_dt"].dt.date.isin(non_trading_days), df["a_dt"] + pd.Timedelta(days=1), df["a_dt"])

    # Create adjusted OHLC columns
    df["a_open"] = np.nan
    df["a_high"] = np.nan
    df["a_low"] = np.nan
    df["a_close"] = np.nan

    # Calculate adjusted OHLC values grouped by adjusted date
    dt_group = df.groupby("a_dt")
    df["a_open"] = dt_group["open"].transform("first")
    df["a_high"] = dt_group["high"].transform("max")
    df["a_low"] = dt_group["low"].transform("min")
    df["a_close"] = dt_group["close"].transform("last")

    # Forward fill NaN values
    df["a_open"] = df["a_open"].ffill()
    df["a_high"] = df["a_high"].ffill()
    df["a_low"] = df["a_low"].ffill()
    df["a_close"] = df["a_close"].ffill()

    # Determine direction
    df["a_dir"] = np.where(df["a_close"] > df["a_open"], "LONG", "SHORT")

    # Handle incomplete last bar
    last_row = df.tail(1)
    if last_row["a_dt"].dt.time.values[0] != end_time:
        df.loc[last_row.index, "a_close"] = np.nan

    return df


class FileSource:
    """
    File-based data source for reading and processing trading data.

    Attributes
    ----------
    path : str
        Path to the data file
    df : pd.DataFrame
        Processed DataFrame with adjusted daily dates
    """

    def __init__(self, path: str, delimiter: str = ";"):
        self.path = path
        self.read(delimiter)
        self.df = adjust_daily_dates(self.df)

    def read(self, delimiter: str) -> None:
        """Read and preprocess data from CSV file."""
        self.df = pd.read_csv(self.path, delimiter=delimiter)
        self.df.columns = ["date", "time", "open", "high", "low", "close", "vol"]

        # Combine date and time columns
        self.df["dt"] = pd.to_datetime(self.df["date"] + " " + self.df["time"], dayfirst=True)
        self.df.drop(columns=["date", "time", "vol"], inplace=True)
        self.df = self.df.sort_values("dt")

        # Keep only recent data (approximately 3 months)
        day = 1440
        year = day * 365
        quarter = int(year / 4)
        self.df = self.df.tail(quarter)
        self.df.reset_index(drop=True, inplace=True)

    def export(self, path: str, dt=None) -> None:
        """Export DataFrame to CSV."""
        self.df.to_csv(path, index=False)

    def export_adjusted_daily_dates(self, path: str) -> None:
        """Export adjusted daily data to CSV."""
        df_export = self.df[["a_dt", "a_open", "a_high", "a_low", "a_close", "a_dir"]]
        df_export = df_export.drop_duplicates(subset=["a_dt"])
        df_export.to_csv(path, index=False)

    def df_run(self) -> None:
        """Placeholder for DataFrame-based trading logic."""
        # Placeholder for trading signal logic
        # This method can be extended to implement specific trading strategies
        pass


def simulate_stream(price: float, timestamp: datetime, spread: float = 1 / 64) -> types.Stream:
    """Create a simulated market data stream."""
    bid = price
    ask = price
    return types.Stream(bid, ask, timestamp)


# =============================================================================
# TRADING CALCULATIONS
# =============================================================================


def compute_rr(entry: float, stop_loss: float, exit: float) -> float:
    """
    Calculate risk-reward ratio for a trade.

    Parameters
    ----------
    entry : float
        Entry price
    stop_loss : float
        Stop loss price
    exit : float
        Target exit price

    Returns
    -------
    float
        Risk-reward ratio (reward/risk)
    """
    reward = exit - entry
    risk = entry - stop_loss
    if risk == 0:
        return float("inf") if reward > 0 else float("-inf")  # avoid division by zero
    return reward / risk


def round_to_tick(price: float, tick_size: float) -> float:
    """
    Round price to the nearest tick size.

    Parameters
    ----------
    price : float
        Price to round
    tick_size : float
        Minimum price increment

    Returns
    -------
    float
        Price rounded to nearest tick
    """
    rounded_price = round(round(price / tick_size) * tick_size, 10)
    return rounded_price


# =============================================================================
# SYSTEM UTILITIES
# =============================================================================


def is_jwt_expired_no_sig(token: str) -> bool:
    """
    Check if a JWT token is expired without verifying signature.

    Parameters
    ----------
    token : str
        JWT token to check

    Returns
    -------
    bool
        True if token is expired or invalid
    """
    try:
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,  # no secret / public key needed
                "verify_exp": False,  # don't raise ExpiredSignatureError
            },
        )
    except jwt.DecodeError:
        # Invalid token == expired
        return True

    exp = payload.get("exp")
    if exp is None:
        raise ValueError("Token has no exp claim")

    now_ts = get_utc().timestamp()
    return now_ts >= exp


def upsert_env(file_path: str, key: str, value: str) -> None:
    """
    Add or update KEY=VALUE in a .env file.

    - Keeps the order and any comments/blank lines already present.
    - Overwrites the line if KEY already exists (case-sensitive).

    Parameters
    ----------
    file_path : str
        Path to .env file
    key : str
        Environment variable key
    value : str
        Environment variable value
    """
    path = Path(file_path)
    lines, found = [], False

    if path.exists():  # Read current lines
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")  # Replace existing
                    found = True
                else:
                    lines.append(line)

    if not found:  # Append if it wasn't there
        lines.append(f"{key}={value}\n")

    with path.open("w", encoding="utf-8") as f:  # Rewrite the file
        f.writelines(lines)


# =============================================================================
# DATA SOURCE INTEGRATIONS
# =============================================================================


def get_tv() -> TvDatafeed:
    """
    Get authenticated TradingView data feed instance.

    Returns
    -------
    TvDatafeed
        Authenticated TradingView data feed instance
    """
    tv_instance = TvDatafeed()
    if is_jwt_expired_no_sig(tv_instance.token):
        tv_instance.token = get_tv_auth(settings.tradingview_sessionid)

    logger.debug(
        "TradingView token ready",
        token_present=bool(tv_instance.token),
        token_expired=is_jwt_expired_no_sig(tv_instance.token) if tv_instance.token else None,
    )
    return tv_instance

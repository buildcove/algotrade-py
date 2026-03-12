from datetime import datetime, timezone

import pytest
import pytz
from algotrade.core import types, utils
from algotrade.schedules import CME_SCHEDULE


def test_time_until_next_globex_bar():
    # Test with a stream object
    """
        cme_globex_utc_4h = {
        "EST": [
            time(3, 0),
            time(7, 0),
            time(11, 0),
            time(15, 0),
            time(19, 0),
            time(23, 0),
        ],  # EST
        "EDT": [
            time(2, 0),
            time(6, 0),
            time(10, 0),
            time(14, 0),
            time(18, 0),
            time(22, 0),
        ],  # EDT
    }"""

    # EST
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2024, 11, 4, 3, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds

    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2024, 11, 4, 7, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds

    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2024, 11, 4, 11, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds

    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2024, 11, 4, 15, 0, 0).astimezone(pytz.timezone("UTC")),
    )

    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2024, 11, 4, 19, 0, 0).astimezone(pytz.timezone("UTC")),
    )

    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2024, 11, 4, 23, 0, 0).astimezone(pytz.timezone("UTC")),
    )

    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds

    # EDT
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2025, 3, 10, 2, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2025, 3, 10, 6, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2025, 3, 10, 10, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2025, 3, 10, 14, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2025, 3, 10, 18, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds
    stream = types.Stream(
        bid=100.0,
        ask=101.0,
        timestamp=types.datetime(2025, 3, 10, 22, 0, 0).astimezone(pytz.timezone("UTC")),
    )
    result = utils.time_until_next_globex_bar(stream=stream)
    assert result == 14400.0  # 4 hours in seconds


def test_next_4hour():
    now = types.datetime(2024, 11, 4, 2, 0, 0).astimezone(pytz.timezone("UTC"))
    tz = utils.get_eastern_timezone_label(now)
    bars = CME_SCHEDULE.four_hour_times_utc[tz]
    times = [datetime.combine(now.date(), t).astimezone(pytz.timezone("UTC")) for t in bars]

    for row in times:
        result = utils.next_4hour(row)
        assert result >= row


def test_next_4hour_cme_friday_close_open_sunday():
    now = types.datetime(2025, 4, 25, 18, 0, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_4hour(now)
    assert str(result) == "2025-04-27 22:00:00+00:00"


def test_next_4hour_cme_open_sunday():
    now = types.datetime(2025, 4, 26, 2, 0, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_4hour(now)
    assert str(result) == "2025-04-27 22:00:00+00:00"


def test_next_4hour_cme_open_monday():
    now = types.datetime(2025, 4, 26, 2, 0, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_4hour(now, skip_sunday=True)
    assert str(result) == "2025-04-28 02:00:00+00:00"

    now = types.datetime(2025, 5, 5, 0, 0, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_4hour(now, skip_sunday=True)
    assert str(result) == "2025-05-05 02:00:00+00:00"

    now = types.datetime(2025, 5, 25, 14, 49, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_4hour(now, skip_sunday=True)
    assert str(result) == "2025-05-26 02:00:00+00:00"


def test_price_round_to_tick():
    # ES
    assert utils.round_to_tick(100.11, 0.25) == 100.0
    assert utils.round_to_tick(100.30, 0.25) == 100.25
    assert utils.round_to_tick(100.495, 0.25) == 100.5

    # GC
    assert utils.round_to_tick(100.11, 0.1) == 100.1
    assert utils.round_to_tick(100.30, 0.1) == 100.3
    assert utils.round_to_tick(100.49, 0.1) == 100.5

    # CL
    assert utils.round_to_tick(100.11, 0.01) == 100.11
    assert utils.round_to_tick(100.30, 0.01) == 100.3
    assert utils.round_to_tick(100.592, 0.01) == 100.59

    # PL
    assert utils.round_to_tick(100.11, 0.10) == 100.1
    assert utils.round_to_tick(100.30, 0.10) == 100.3
    assert utils.round_to_tick(100.49, 0.10) == 100.5

    # HG
    assert utils.round_to_tick(100.11, 0.0005) == 100.11
    assert utils.round_to_tick(100.30, 0.0005) == 100.3
    assert utils.round_to_tick(100.5922948, 0.0005) == 100.5925


def test_next_friday_20utc():
    now = types.datetime(2024, 11, 4, 2, 0, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_friday_20utc(now)
    assert str(result) == "2024-11-08 20:00:00+00:00"

    now = types.datetime(2024, 11, 8, 19, 59, 59).astimezone(pytz.timezone("UTC"))
    result = utils.next_friday_20utc(now)
    assert str(result) == "2024-11-08 20:00:00+00:00"

    now = types.datetime(2024, 11, 8, 20, 0, 0).astimezone(pytz.timezone("UTC"))
    result = utils.next_friday_20utc(now)
    assert str(result) == "2024-11-15 20:00:00+00:00"


@pytest.mark.parametrize(
    "now, expected",
    [
        # ── Mid-week behaviour ─────────────────────────────────────────
        # Wed 10:00  → same-day 22:00
        (
            datetime(2025, 6, 18, 10, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 18, 22, 0, tzinfo=timezone.utc),
        ),
        # Wed 23:00  → Thu 22:00
        (
            datetime(2025, 6, 18, 23, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 19, 22, 0, tzinfo=timezone.utc),
        ),
        # ── Friday evening boundary ───────────────────────────────────
        # Fri 21:59 → Fri 22:00 (still tradable)
        (
            datetime(2025, 6, 20, 21, 59, tzinfo=timezone.utc),
            datetime(2025, 6, 20, 22, 0, tzinfo=timezone.utc),
        ),
        # Fri 22:00 (market just closed) → Sun 22:00
        (
            datetime(2025, 6, 20, 22, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 22, 22, 0, tzinfo=timezone.utc),
        ),
        # ── Weekend break ─────────────────────────────────────────────
        # Sat 12:00 → Sun 22:00
        (
            datetime(2025, 6, 21, 12, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 22, 22, 0, tzinfo=timezone.utc),
        ),
        # Sun 21:00 → Sun 22:00 (one hour before reopen)
        (
            datetime(2025, 6, 22, 21, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 22, 22, 0, tzinfo=timezone.utc),
        ),
        # Sun 22:00 (reopen) → Mon 22:00
        (
            datetime(2025, 6, 22, 22, 0, tzinfo=timezone.utc),
            datetime(2025, 6, 23, 22, 0, tzinfo=timezone.utc),
        ),
    ],
)
def test_daily_returns_next_tradable_bar(now, expected):
    """daily() should return the first valid 22-UTC bar strictly after *now*."""
    assert utils.next_daily(now=now) == expected

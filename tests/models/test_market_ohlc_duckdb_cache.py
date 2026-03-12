from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from algotrade.core.market_ohlc_duckdb import MarketOhlcDuckdbCache

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # type: ignore


pytestmark = pytest.mark.skipif(duckdb is None, reason="duckdb is required for DuckDB OHLC cache tests")


def test_duckdb_cache_seeds_and_upserts(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "data" / "market_ohlc.duckdb"
    cache = MarketOhlcDuckdbCache(duckdb_path=duckdb_path)
    dataset_key = "ES:CME:m30:fut1:ext1"

    seed_csv = tmp_path / "seed.csv"
    seed_df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 00:30:00+00:00",
                "2024-01-01 01:00:00+00:00",
            ],
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    seed_df.to_csv(seed_csv, index=False)

    cache.ensure_seeded(dataset_key=dataset_key, seed_csv=seed_csv)
    latest = cache.latest_timestamp(dataset_key=dataset_key)
    assert latest is not None
    assert pd.Timestamp(latest, tz="UTC") == pd.Timestamp("2024-01-01 01:00:00+00:00")

    tv_like_df = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 01:00:00+00:00",  # duplicate
                "2024-01-01 01:30:00+00:00",  # new
            ],
            "open": [3.0, 4.0],
            "high": [3.5, 4.5],
            "low": [2.5, 3.5],
            "close": [3.2, 4.2],
            "volume": [30.0, 40.0],
        }
    )
    inserted = cache.insert_frame(dataset_key=dataset_key, df=tv_like_df, require_full_schema=False)
    assert inserted == 1

    loaded = cache.load_last_n(dataset_key=dataset_key, n=10)
    assert len(loaded) == 4
    assert loaded["timestamp"].iloc[-1] == pd.Timestamp("2024-01-01 01:30:00+00:00")

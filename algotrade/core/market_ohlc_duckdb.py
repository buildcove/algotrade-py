from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import duckdb  # type: ignore
except Exception as exc:  # pragma: no cover
    duckdb = None  # type: ignore
    _DUCKDB_IMPORT_ERROR = exc
else:  # pragma: no cover
    _DUCKDB_IMPORT_ERROR = None


REQUIRED_OHLC_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketDatasetKey:
    symbol: str
    exchange: str
    timeframe: str
    fut_contract: int = 1
    extended_session: bool = True

    def as_str(self) -> str:
        ext = 1 if self.extended_session else 0
        return f"{self.symbol}:{self.exchange}:{self.timeframe}:fut{self.fut_contract}:ext{ext}"


class MarketOhlcDuckdbCache:
    """
    DuckDB-backed OHLC cache.

    - Stores OHLC rows keyed by a stable dataset key + timestamp.
    - Requires an explicit seed CSV when a dataset key has no stored data.
    - Updates by inserting only rows strictly newer than the latest stored timestamp.
    """

    def __init__(self, duckdb_path: Path) -> None:
        self.duckdb_path = duckdb_path

    def _connect(self):
        if duckdb is None:  # pragma: no cover
            raise ImportError("duckdb is required for MarketOhlcDuckdbCache") from _DUCKDB_IMPORT_ERROR
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.duckdb_path))

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlc (
                    dataset_key TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE
                );
                """
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ohlc_dataset_ts_idx ON ohlc(dataset_key, timestamp);")

    def latest_timestamp(self, dataset_key: str) -> Optional[datetime]:
        self.ensure_schema()
        with self._connect() as conn:
            value = conn.execute(
                "SELECT max(timestamp) FROM ohlc WHERE dataset_key = ?;",
                [dataset_key],
            ).fetchone()
            if not value or value[0] is None:
                return None
            ts = pd.Timestamp(value[0])
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.tz_localize(None).to_pydatetime()

    def load_last_n(self, dataset_key: str, n: int) -> pd.DataFrame:
        if n <= 0:
            raise ValueError("n must be positive")
        self.ensure_schema()
        with self._connect() as conn:
            df = conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlc
                WHERE dataset_key = ?
                ORDER BY timestamp DESC
                LIMIT ?;
                """,
                [dataset_key, n],
            ).df()
        if df.empty:
            return df
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def ensure_seeded(self, dataset_key: str, seed_csv: Optional[Path]) -> None:
        """
        Ensure the dataset key exists in the DB. If empty, require seed_csv.
        """
        latest = self.latest_timestamp(dataset_key)
        if latest is not None:
            return
        if seed_csv is None:
            raise ValueError(f"No DuckDB OHLC data for {dataset_key} in {self.duckdb_path}. Provide a seed CSV to initialize.")
        self.seed_from_csv(dataset_key=dataset_key, csv_path=seed_csv)

    def seed_from_csv(self, dataset_key: str, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Seed CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        self.insert_frame(dataset_key=dataset_key, df=df, require_full_schema=True)

    def insert_frame(self, dataset_key: str, df: pd.DataFrame, require_full_schema: bool) -> int:
        self.ensure_schema()
        missing = [c for c in REQUIRED_OHLC_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required OHLC columns: {missing}")

        frame = df[list(REQUIRED_OHLC_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
        frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        if require_full_schema and frame.empty:
            raise ValueError("Seed CSV produced an empty OHLC frame after normalization.")

        frame.insert(0, "dataset_key", dataset_key)

        with self._connect() as conn:
            conn.register("incoming_ohlc", frame)
            before = conn.execute(
                "SELECT count(*) FROM ohlc WHERE dataset_key = ?;",
                [dataset_key],
            ).fetchone()[0]
            conn.execute(
                """
                INSERT INTO ohlc (dataset_key, timestamp, open, high, low, close, volume)
                SELECT dataset_key, timestamp, open, high, low, close, volume
                FROM incoming_ohlc
                ON CONFLICT DO NOTHING;
                """
            )
            after = conn.execute(
                "SELECT count(*) FROM ohlc WHERE dataset_key = ?;",
                [dataset_key],
            ).fetchone()[0]
        return int(after - before)

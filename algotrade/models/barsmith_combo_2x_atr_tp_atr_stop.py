"""
Barsmith combo strategy model.

This model reuses the core structure of `tribar_v2.TribarExecutor` but drives
entries from a pinned Barsmith-engineered combo (long-only):

    is_tribar_green && is_kf_breakout_potential && kf_innovation_abs>5.02 && macd_hist>=-1.0 && trend_strength<=0.25

Trade semantics match Barsmith's Rust `2x_atr_tp_atr_stop` target:
  - Entry at signal bar close, tick-quantized (Nearest).
  - Long stop: entry - 1x ATR, tick-quantized (Floor).
  - Long take-profit: entry + 2x ATR, tick-quantized (Ceil).
  - Multi-bar hold until TP/SL, with gap-aware fills (bar open beyond TP/SL fills at open).

This model pulls OHLC from TradingView and computes Barsmith indicators using
the parity implementation in `tools/check_barsmith_indicator_parity.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

import pandas as pd
import retry
from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, engine, repository, types, utils
from algotrade.core.safety import require_live_trading_enabled
from algotrade.schedules import (
    CBOT_SCHEDULE,
    CME_SCHEDULE,
    CONTRACT_TO_EXCHANGE,
    IB_TZ,
    TIMEFRAME_TO_TV_INTERVAL,
    Exchange,
)
from algotrade.settings import settings
from algotrade.tv import get_token as get_tv_auth
from structlog import get_logger
from tvDatafeed import TvDatafeed

logger = get_logger()


class Variation(str, Enum):
    BARSMITH_COMBO_LONG = "BARSMITH_COMBO_LONG"


SMALL_DIVISOR: float = 1e-9
TP_ATR_MULTIPLE: float = 2.0
FLOAT_EPSILON: float = float.fromhex("0x1.0p-52")
# Minimum TradingView bars so the latest row has all combo-required indicators
# (dominant warmup is `200sma`). Because this model uses iloc[-2] as the
# signal bar (iloc[-1] is treated as the “current/open” bar), we need 200 bars
# of warmup *plus* the current bar => 201.
BARSMITH_COMBO_REQUIRED_TV_BARS: int = 201
# After the first warm fetch, only refresh a smaller window from TradingView.
# The rolling cache maintains enough history for indicator warmups.
BARSMITH_TV_REFRESH_BARS_WARM: int = 50
BARSMITH_RAW_CACHE_EXTRA_BARS: int = 50


class TickRoundMode(str, Enum):
    NEAREST = "nearest"
    FLOOR = "floor"
    CEIL = "ceil"


def quantize_price_to_tick(price: float, tick_size: float, mode: TickRoundMode) -> float:
    if not math.isfinite(price) or tick_size <= 0:
        return price
    ticks = price / tick_size
    rounded = {
        TickRoundMode.NEAREST: round(ticks),
        TickRoundMode.FLOOR: math.floor(ticks),
        TickRoundMode.CEIL: math.ceil(ticks),
    }[mode]
    return float(rounded) * tick_size


class TradeSide(str, Enum):
    LONG = "LONG"


class ExecutionMode(str, Enum):
    SIM = "sim"
    LIVE = "live"


@dataclass
class Candle(types.Candle):
    """
    Candle enriched with the subset of Barsmith features required for the
    pinned combo and risk management.
    """

    is_tribar_green: bool = False
    is_kf_breakout_potential: bool = False
    kf_innovation_abs: float = 0.0
    macd_hist: float = 0.0
    trend_strength: float = 0.0
    atr: float = 0.0
    kf_adx: float = 0.0
    kf_atr: float = 0.0


@dataclass
class ActiveTrade:
    """
    Internal state for a single open trade under Barsmith 2x_atr_tp_atr_stop-style targets.
    """

    side: TradeSide
    entry_dt: datetime
    entry: float
    stop: float
    tp: float
    risk: float
    atr: float


@dataclass
class Extra:
    variation: Optional[Variation] = None
    drawdown: float = 0.0
    drawdown_price: float = 0.0
    drawdown_dt: Optional[datetime] = None
    realized_rr: float = 0.0


@dataclass
class Param:
    """
    Strategy parameters for the Barsmith combo model.

    The structure mirrors `tribar_v2.Param` where it is useful for risk and
    filtering, but removes tribar-specific knobs.
    """

    contract: types.ContractSpec
    supported_contract: types.SupportedContract
    tz: str = IB_TZ
    timeframe: types.Timeframe = types.Timeframe.m30

    # Risk and sizing
    capital: float = 10000.0
    account_risk_pct: float = 0.01  # 1% risk per trade
    min_quantity: int = 1
    max_quantity: int = 1
    minimum_risk_reward: float = 1.2
    target_atr_multiple: float = TP_ATR_MULTIPLE

    # Optional inequality filters on close / target if needed later
    gt: Optional[float] = None
    gte: Optional[float] = None
    lt: Optional[float] = None
    lte: Optional[float] = None

    gt_target: Optional[float] = None
    gte_target: Optional[float] = None
    lt_target: Optional[float] = None
    lte_target: Optional[float] = None


@dataclass
class Var:
    """
    Runtime mutable state for the strategy.
    """

    open_trade: bool = False
    active_trade: Optional[ActiveTrade] = None
    setups: List[dict] = field(default_factory=list)


class PrerequisiteFailed(Exception):
    pass


class DataFetchError(Exception):
    pass


@dataclass
class LiveBracketSnapshot:
    signal_bar_ts_utc: str
    submitted_at_utc: str
    entry: float
    stop_loss: float
    take_profit: float
    quantity: int
    ib_order_ids: list[int] = field(default_factory=list)


@dataclass
class CircuitBreakerState:
    trade_date_utc: str
    trades_submitted_today: int = 0
    consecutive_failures: int = 0
    disabled_until_utc: Optional[str] = None


class JournalWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, **payload}
        line = json.dumps(record, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class BarsmithComboExecutor(engine.BaseEngine):
    """
    Execute pinned Barsmith combos using TradingView OHLC plus the parity
    indicator pipeline (no CSV/DuckDB dependency).
    """

    def __init__(
        self,
        broker: brokers.BacktestingBroker | brokers.IBKRBroker,
        param: Param,
        var: Var,
        extra: Extra,
        tv_required_bars: int = BARSMITH_COMBO_REQUIRED_TV_BARS,
        mode: ExecutionMode = ExecutionMode.SIM,
        tv_lag_seconds: float = 60.0,
        max_poll_retries: int = 5,
        poll_retry_sleep_seconds: float = 0.5,
        max_trades_per_day: int = 1,
        max_consecutive_failures: int = 5,
        failure_cooldown_seconds: float = 3600.0,
        require_market_open: bool = True,
        entry_timeout_seconds: float = 7200.0,
        repair_orders: bool = True,
        enforce_exit_prices: bool = False,
        state_path: Optional[str] = None,
        journal_path: Optional[str] = None,
    ) -> None:
        super().__init__(broker)
        self.broker = broker

        self.default_extra: Extra = extra
        self.extra: Extra = deepcopy(self.default_extra)

        self.param_default: Param = param
        self.param: Param = deepcopy(self.param_default)

        self.var_default: Var = var
        self.var: Var = deepcopy(self.var_default)

        self.fatal = False
        self.precheck_failed = False

        # TradingView handle (same pattern as tribar_v2).
        self.tv_instance: TvDatafeed | None = None
        if tv_required_bars <= 0:
            raise ValueError("tv_required_bars must be positive")
        self.tv_required_bars = tv_required_bars

        if not math.isfinite(tv_lag_seconds) or tv_lag_seconds < 0:
            raise ValueError("tv_lag_seconds must be >= 0")
        if max_poll_retries < 0:
            raise ValueError("max_poll_retries must be >= 0")
        if not math.isfinite(poll_retry_sleep_seconds) or poll_retry_sleep_seconds <= 0:
            raise ValueError("poll_retry_sleep_seconds must be > 0")
        if max_trades_per_day < 0:
            raise ValueError("max_trades_per_day must be >= 0")
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be > 0")
        if not math.isfinite(failure_cooldown_seconds) or failure_cooldown_seconds < 0:
            raise ValueError("failure_cooldown_seconds must be >= 0")
        if not math.isfinite(entry_timeout_seconds) or entry_timeout_seconds < 0:
            raise ValueError("entry_timeout_seconds must be >= 0")

        self.tv_lag_seconds = float(tv_lag_seconds)
        self.max_poll_retries = int(max_poll_retries)
        self.poll_retry_sleep_seconds = float(poll_retry_sleep_seconds)

        self.mode = mode
        self.last_seen_raw_bar_ts_utc: Optional[pd.Timestamp] = None
        self.last_processed_signal_bar_ts_utc: Optional[pd.Timestamp] = None
        self.last_submitted_signal_bar_ts_utc: Optional[pd.Timestamp] = None
        self.last_bracket_snapshot: Optional[LiveBracketSnapshot] = None

        self.max_trades_per_day = int(max_trades_per_day)
        self.max_consecutive_failures = int(max_consecutive_failures)
        self.failure_cooldown_seconds = float(failure_cooldown_seconds)
        self.require_market_open = bool(require_market_open)
        self.entry_timeout_seconds = float(entry_timeout_seconds)
        self.repair_orders = bool(repair_orders)
        self.enforce_exit_prices = bool(enforce_exit_prices)

        today_utc = utils.get_utc().date().isoformat()
        self.circuit_breaker = CircuitBreakerState(trade_date_utc=today_utc)

        self._raw_cache: Optional[pd.DataFrame] = None
        self._raw_cache_max_bars = max(self.tv_required_bars, BARSMITH_COMBO_REQUIRED_TV_BARS) + BARSMITH_RAW_CACHE_EXTRA_BARS

        self.state_path = Path(state_path) if state_path else self._default_state_path()
        self.journal = JournalWriter(Path(journal_path)) if journal_path else JournalWriter(self._default_journal_path())
        self._load_state()

    # ───────────────────────── TradingView helpers ────────────────────────────

    def get_tv(self) -> TvDatafeed:
        if self.tv_instance is None:
            self.tv_instance = TvDatafeed()

        if utils.is_jwt_expired_no_sig(self.tv_instance.token):
            self.tv_instance.token = get_tv_auth(settings.tradingview_sessionid)

        logger.debug(
            "TradingView token ready",
            token_present=bool(self.tv_instance.token),
            token_expired=utils.is_jwt_expired_no_sig(self.tv_instance.token) if self.tv_instance.token else None,
        )
        return self.tv_instance

    @staticmethod
    def _to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        tz = getattr(idx, "tz", None)
        if tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")

    @staticmethod
    def _to_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _default_state_path(self) -> Path:
        base = Path("tmp") / "state"
        base.mkdir(parents=True, exist_ok=True)
        symbol = str(getattr(self.param.contract, "symbol", "UNKNOWN")).lower()
        timeframe = str(getattr(self.param.timeframe, "value", self.param.timeframe)).lower()
        return base / f"barsmith_combo_2x_atr_tp_atr_stop_{symbol}_{timeframe}.json"

    def _default_journal_path(self) -> Path:
        base = Path("tmp") / "journal"
        base.mkdir(parents=True, exist_ok=True)
        symbol = str(getattr(self.param.contract, "symbol", "UNKNOWN")).lower()
        timeframe = str(getattr(self.param.timeframe, "value", self.param.timeframe)).lower()
        return base / f"barsmith_combo_2x_atr_tp_atr_stop_{symbol}_{timeframe}.jsonl"

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
        except Exception as exc:
            logger.warning("Failed to load state; starting fresh", path=str(self.state_path), error=str(exc))
            return

        def _ts(key: str) -> Optional[pd.Timestamp]:
            raw = data.get(key)
            if not raw:
                return None
            return pd.to_datetime(raw, utc=True, errors="coerce")

        self.last_seen_raw_bar_ts_utc = _ts("last_seen_raw_bar_ts_utc")
        self.last_processed_signal_bar_ts_utc = _ts("last_processed_signal_bar_ts_utc")
        self.last_submitted_signal_bar_ts_utc = _ts("last_submitted_signal_bar_ts_utc")

        snap = data.get("last_bracket_snapshot")
        if isinstance(snap, dict):
            try:
                self.last_bracket_snapshot = LiveBracketSnapshot(
                    signal_bar_ts_utc=str(snap["signal_bar_ts_utc"]),
                    submitted_at_utc=str(snap["submitted_at_utc"]),
                    entry=float(snap["entry"]),
                    stop_loss=float(snap["stop_loss"]),
                    take_profit=float(snap["take_profit"]),
                    quantity=int(snap["quantity"]),
                    ib_order_ids=[int(x) for x in snap.get("ib_order_ids", [])],
                )
            except Exception:
                self.last_bracket_snapshot = None

        cb = data.get("circuit_breaker")
        if isinstance(cb, dict):
            try:
                self.circuit_breaker = CircuitBreakerState(
                    trade_date_utc=str(cb.get("trade_date_utc") or utils.get_utc().date().isoformat()),
                    trades_submitted_today=int(cb.get("trades_submitted_today", 0)),
                    consecutive_failures=int(cb.get("consecutive_failures", 0)),
                    disabled_until_utc=cb.get("disabled_until_utc"),
                )
            except Exception:
                pass

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload: Optional[dict[str, Any]] = None
        if self.last_bracket_snapshot is not None:
            snapshot_payload = {
                "signal_bar_ts_utc": self.last_bracket_snapshot.signal_bar_ts_utc,
                "submitted_at_utc": self.last_bracket_snapshot.submitted_at_utc,
                "entry": self.last_bracket_snapshot.entry,
                "stop_loss": self.last_bracket_snapshot.stop_loss,
                "take_profit": self.last_bracket_snapshot.take_profit,
                "quantity": self.last_bracket_snapshot.quantity,
                "ib_order_ids": self.last_bracket_snapshot.ib_order_ids,
            }

        cb = self.circuit_breaker
        payload = {
            "last_seen_raw_bar_ts_utc": (
                self.last_seen_raw_bar_ts_utc.isoformat() if self.last_seen_raw_bar_ts_utc is not None else None
            ),
            "last_processed_signal_bar_ts_utc": (
                self.last_processed_signal_bar_ts_utc.isoformat() if self.last_processed_signal_bar_ts_utc is not None else None
            ),
            "last_submitted_signal_bar_ts_utc": (
                self.last_submitted_signal_bar_ts_utc.isoformat() if self.last_submitted_signal_bar_ts_utc is not None else None
            ),
            "last_bracket_snapshot": snapshot_payload,
            "circuit_breaker": {
                "trade_date_utc": cb.trade_date_utc,
                "trades_submitted_today": cb.trades_submitted_today,
                "consecutive_failures": cb.consecutive_failures,
                "disabled_until_utc": cb.disabled_until_utc,
            },
        }
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self.state_path)

    def _expected_latest_bar_start_ts_utc(self, now_utc: datetime) -> pd.Timestamp:
        """
        Expected timestamp (UTC) for the latest bar *start* time.

        We treat TradingView timestamps as bar-start timestamps; the signal bar is
        the previous bar (iloc[-2]) once the latest bar (iloc[-1]) exists.
        """
        now_utc_ts = pd.Timestamp(now_utc).tz_convert("UTC")
        contract_enum = self.param.supported_contract

        if self.param.timeframe == types.Timeframe.m30:
            nxt = utils.next_half_hour_for_contract(contract_enum, now_utc)
            expected = nxt - timedelta(minutes=30)
        elif self.param.timeframe == types.Timeframe.h4:
            nxt = utils.next_4hour_for_contract(contract_enum, now_utc, skip_sunday=True)
            expected = nxt - timedelta(hours=4)
        elif self.param.timeframe == types.Timeframe.daily:
            nxt = utils.next_daily_for_contract(contract_enum, now_utc)
            expected = nxt - timedelta(days=1)
        else:
            raise ValueError(f"Unsupported timeframe: {self.param.timeframe}")

        expected_ts = pd.Timestamp(expected)
        if expected_ts.tz is None:
            expected_ts = expected_ts.tz_localize("UTC")
        else:
            expected_ts = expected_ts.tz_convert("UTC")
        if expected_ts > now_utc_ts:
            expected_ts = now_utc_ts.floor("min")
        return expected_ts

    def _log_recent_raw_bar_timestamps(self, df_raw: pd.DataFrame) -> None:
        if "timestamp" not in df_raw.columns or df_raw.empty:
            return
        tail = df_raw[["timestamp"]].dropna().tail(3)
        stamps = [pd.Timestamp(x).tz_convert("UTC").isoformat() for x in tail["timestamp"].tolist()]
        logger.debug("Recent raw bars", last3=stamps)

    def _validate_raw_bar_timestamps(self, df_raw: pd.DataFrame) -> None:
        """
        Best-effort validation/logging for TradingView timestamps.

        This is intentionally non-fatal (logs warnings) because provider timestamp
        behavior can vary slightly; trading gating is enforced elsewhere.
        """
        if "timestamp" not in df_raw.columns:
            return
        ts = pd.to_datetime(df_raw["timestamp"], utc=True, errors="coerce").dropna()
        if len(ts) < 2:
            return

        last = pd.Timestamp(ts.iloc[-1]).tz_convert("UTC")
        prev = pd.Timestamp(ts.iloc[-2]).tz_convert("UTC")
        if prev >= last:
            logger.warning("Non-monotonic timestamps", prev=str(prev), last=str(last))

        if self.param.timeframe == types.Timeframe.m30:
            if last.second != 0 or last.microsecond != 0 or last.minute not in (0, 30):
                logger.warning("Unexpected m30 timestamp alignment", last=str(last))

    def _reconcile_live_state(self) -> None:
        if not isinstance(self.broker, brokers.IBKRBroker):
            return
        try:
            self.broker.ensure_connected()
            contract = self.param.contract.ib
            positions = self.broker.get_positions(contract=contract)
            open_orders = self.broker.ib.openOrders()
            logger.debug(
                "IB reconcile",
                positions=len(positions),
                open_orders=len(open_orders),
            )
        except Exception as exc:
            logger.warning("IB reconcile failed", error=str(exc))

    def _journal(self, event: str, **payload: Any) -> None:
        try:
            self.journal.write(
                event,
                {
                    "ts_utc": utils.get_utc().isoformat(),
                    "contract": self.param.supported_contract.value,
                    "timeframe": self.param.timeframe.value,
                    "mode": self.mode.value,
                    **payload,
                },
            )
        except Exception as exc:
            logger.warning("Journal write failed", error=str(exc))

    def _reset_daily_if_needed(self, now_utc: datetime) -> None:
        today = now_utc.date().isoformat()
        if self.circuit_breaker.trade_date_utc != today:
            self.circuit_breaker.trade_date_utc = today
            self.circuit_breaker.trades_submitted_today = 0
            self._save_state()

    def _is_disabled(self, now_utc: datetime) -> bool:
        disabled_until = self.circuit_breaker.disabled_until_utc
        if disabled_until is None:
            return False
        until = pd.Timestamp(disabled_until).tz_convert("UTC").to_pydatetime()
        if now_utc >= until:
            self.circuit_breaker.disabled_until_utc = None
            self._save_state()
            return False
        return True

    def _note_failure(self, now_utc: datetime, error: str) -> None:
        self.circuit_breaker.consecutive_failures += 1
        if self.circuit_breaker.consecutive_failures >= self.max_consecutive_failures:
            disabled_until = now_utc + timedelta(seconds=self.failure_cooldown_seconds)
            self.circuit_breaker.disabled_until_utc = disabled_until.isoformat()
            logger.error(
                "Circuit breaker engaged",
                consecutive_failures=self.circuit_breaker.consecutive_failures,
                disabled_until=str(disabled_until),
                error=error,
            )
            self._journal(
                "circuit_breaker_engaged",
                consecutive_failures=self.circuit_breaker.consecutive_failures,
                disabled_until=str(disabled_until),
                error=error,
            )
        self._save_state()

    def _note_success(self) -> None:
        if self.circuit_breaker.consecutive_failures != 0:
            self.circuit_breaker.consecutive_failures = 0
            self._save_state()

    def _is_market_open_utc(self, now_utc: datetime) -> bool:
        wd = now_utc.weekday()  # Mon=0 ... Sun=6
        if wd == 5:
            return False
        exchange = CONTRACT_TO_EXCHANGE.get(self.param.supported_contract, Exchange.CME)
        if exchange == Exchange.CBOT:
            tz_label = utils.get_central_timezone_label(now_utc)
            open_hour = CBOT_SCHEDULE.open_times_utc[tz_label].hour
            close_hour = 20 if tz_label.value == "CST" else 19
            if wd == 4 and now_utc.hour >= close_hour:
                return False
            if wd == 6 and now_utc.hour < open_hour:
                return False
            return True

        tz_label = utils.get_eastern_timezone_label(now_utc)
        open_hour = CME_SCHEDULE.open_times_utc[tz_label].hour
        if wd == 4 and now_utc.hour >= 22:
            return False
        if wd == 6 and now_utc.hour < open_hour:
            return False
        return True

    def _has_protective_orders(self, orders: list[Any], exit_action: str) -> bool:
        has_tp = False
        has_sl = False
        for o in orders:
            order_type = getattr(o, "orderType", None)
            action = getattr(o, "action", None)
            if action != exit_action:
                continue
            if order_type == "LMT":
                has_tp = True
            elif order_type == "STP":
                has_sl = True
        return has_tp and has_sl

    def _maybe_recover_live_protection(self, positions: list[Any]) -> None:
        if not isinstance(self.broker, brokers.IBKRBroker):
            return
        if not positions:
            return

        position = positions[0]
        qty = int(getattr(position, "position", 0))
        if qty == 0:
            return

        contract = getattr(position, "contract", self.param.contract.ib)
        open_orders = self.broker.get_orders(contract=contract)
        exit_action = "SELL" if qty > 0 else "BUY"
        if self._has_protective_orders(open_orders, exit_action=exit_action):
            return

        snap = self.last_bracket_snapshot
        if snap is None or not math.isfinite(snap.stop_loss) or not math.isfinite(snap.take_profit):
            logger.warning("Missing bracket snapshot; cannot recover protection", qty=qty)
            self._journal("recover_protection_missing_snapshot", qty=qty)
            return

        oca_group = f"barsmith_recover_{self.param.contract.ib.symbol}_{int(time.time())}"
        logger.warning("Recovering missing protection orders", qty=qty, oca_group=oca_group)
        self._journal(
            "recover_protection_attempt",
            qty=qty,
            stop_loss=snap.stop_loss,
            take_profit=snap.take_profit,
            oca_group=oca_group,
        )
        trades = self.broker.place_oca_exit_orders(
            contract=contract,
            quantity=abs(qty),
            take_profit_price=float(snap.take_profit),
            stop_loss_price=float(snap.stop_loss),
            action=types.Action.SHORT if qty > 0 else types.Action.LONG,
            oca_group=oca_group,
        )
        order_ids: list[int] = []
        for t in trades:
            oid = getattr(getattr(t, "order", None), "orderId", None)
            if oid is not None:
                order_ids.append(int(oid))
        self._journal("recover_protection_submitted", qty=qty, order_ids=order_ids)

    def _cancel_open_orders(self, contract: Any, orders: list[Any], reason: str) -> None:
        if not isinstance(self.broker, brokers.IBKRBroker):
            return
        if not orders:
            return
        try:
            logger.warning("Canceling open orders", reason=reason, open_orders=len(orders))
            self._journal("cancel_open_orders", reason=reason, open_orders=len(orders))
            self.broker.cancel_orders(orders)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Failed to cancel open orders", reason=reason, error=str(exc))
            self._journal("cancel_open_orders_failed", reason=reason, error=str(exc))

    def _reconcile_live_orders(self, run_id: str) -> None:
        """
        Reconcile IBKR state vs persisted snapshot and attempt safe repairs.

        This runs independently of TradingView availability so that:
        - if a position exists, we can ensure TP/SL protection remains attached
        - if an entry has been resting too long, we can cancel it
        """
        if not isinstance(self.broker, brokers.IBKRBroker):
            return

        self.broker.ensure_connected()
        contract = self.param.contract.ib

        positions = self.broker.get_positions(contract=contract)
        open_orders = self.broker.get_orders(contract=contract)
        position_qty = 0
        if positions:
            try:
                position_qty = int(getattr(positions[0], "position", 0))
            except Exception:
                position_qty = 0

        if position_qty != 0:
            if self.repair_orders:
                self._maybe_recover_live_protection(positions=positions)
            self._journal(
                "reconcile_position_open",
                run_id=run_id,
                position_qty=position_qty,
                open_orders=len(open_orders),
            )
            return

        # No position: consider canceling stale open orders from a prior submission.
        if open_orders:
            snap = self.last_bracket_snapshot
            if snap is None:
                self._journal("reconcile_open_orders_no_snapshot", run_id=run_id, open_orders=len(open_orders))
                return

            try:
                submitted_at = pd.Timestamp(snap.submitted_at_utc).tz_convert("UTC").to_pydatetime()
            except Exception:
                self._journal("reconcile_open_orders_bad_snapshot_time", run_id=run_id, open_orders=len(open_orders))
                return

            age_seconds = (utils.get_utc() - submitted_at).total_seconds()
            self._journal(
                "reconcile_open_orders",
                run_id=run_id,
                open_orders=len(open_orders),
                age_seconds=round(age_seconds, 3),
                entry_timeout_seconds=self.entry_timeout_seconds,
            )

            if self.entry_timeout_seconds > 0 and age_seconds >= self.entry_timeout_seconds:
                self._cancel_open_orders(contract=contract, orders=open_orders, reason="entry_timeout")
            return

        # No position, no open orders: clear any stale snapshot to reduce confusion in logs.
        if self.last_bracket_snapshot is not None:
            self._journal("reconcile_clear_snapshot_no_position", run_id=run_id)
            self.last_bracket_snapshot = None
            self._save_state()

    def _tv_hist_to_raw_df(self, fetched: pd.DataFrame, idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
        fetched = fetched.copy()
        fetched.index = idx_utc
        df_raw = fetched.reset_index().rename(columns={"datetime": "timestamp"})
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True, errors="coerce")
        return df_raw[["timestamp", "open", "high", "low", "close"]]

    def _merge_raw_cache(self, df_raw_new: pd.DataFrame) -> pd.DataFrame:
        if self._raw_cache is None or self._raw_cache.empty:
            merged = df_raw_new
        else:
            merged = pd.concat([self._raw_cache, df_raw_new], ignore_index=True)

        merged = merged.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        if len(merged) > self._raw_cache_max_bars:
            merged = merged.tail(self._raw_cache_max_bars)

        self._raw_cache = merged
        return merged

    def pull_historical_data(self) -> None:
        run_id = uuid4().hex
        now_utc = utils.get_utc()
        self._reset_daily_if_needed(now_utc)

        if self.mode == ExecutionMode.LIVE:
            if self.require_market_open and not self._is_market_open_utc(now_utc):
                logger.info("Market closed; skipping scheduled run")
                self._journal("market_closed_skip", run_id=run_id)
                return
            if self._is_disabled(now_utc):
                logger.warning(
                    "Trading disabled by circuit breaker; skipping run", disabled_until=self.circuit_breaker.disabled_until_utc
                )
                self._journal("disabled_skip", run_id=run_id, disabled_until=self.circuit_breaker.disabled_until_utc)
                return

        try:
            self._pull_historical_data_once(run_id=run_id)
            self._note_success()
        except Exception as exc:
            logger.exception("Barsmith combo run failed", run_id=run_id, error=str(exc))
            self._journal("run_failed", run_id=run_id, error=str(exc), error_type=type(exc).__name__)
            self._note_failure(now_utc=utils.get_utc(), error=str(exc))
            raise

    @retry.retry(exceptions=DataFetchError, tries=10, max_delay=10, delay=2, backoff=5)
    def _pull_historical_data_once(self, run_id: str) -> None:
        """
        Single attempt for fetching OHLC + computing indicators + processing the latest signal.

        Decorated with a bounded retry on `DataFetchError` only (TradingView lag / missing bar).
        """
        tv = self.get_tv()
        interval = TIMEFRAME_TO_TV_INTERVAL[self.param.timeframe]
        start = time.time()
        fetch_bars = (
            self.tv_required_bars if self._raw_cache is None else min(self.tv_required_bars, BARSMITH_TV_REFRESH_BARS_WARM)
        )

        if self.mode == ExecutionMode.LIVE:
            self._reconcile_live_state()

        expected_latest_bar_start_utc = self._expected_latest_bar_start_ts_utc(utils.get_utc())

        fetched = None
        idx_utc: Optional[pd.DatetimeIndex] = None
        last_raw_ts_utc: Optional[pd.Timestamp] = None

        for attempt in range(self.max_poll_retries + 1):
            now_utc = utils.get_utc()
            expected_latest_bar_start_utc = max(expected_latest_bar_start_utc, self._expected_latest_bar_start_ts_utc(now_utc))

            fetched = tv.get_hist(
                symbol=self.param.contract.symbol,
                exchange=self.param.contract.exchange,
                interval=interval,
                n_bars=fetch_bars,
                fut_contract=1,
                extended_session=True,
            )
            if fetched is None or fetched.empty:
                raise DataFetchError("No data fetched from TradingView")

            idx_utc = self._to_utc_index(pd.DatetimeIndex(fetched.index))
            last_raw_ts_utc = pd.Timestamp(idx_utc[-1])

            if last_raw_ts_utc >= expected_latest_bar_start_utc:
                break

            lag_seconds = (pd.Timestamp(now_utc).tz_convert("UTC") - expected_latest_bar_start_utc).total_seconds()
            if lag_seconds <= self.tv_lag_seconds and attempt < self.max_poll_retries:
                logger.debug(
                    "TradingView bar lag; retrying",
                    run_id=run_id,
                    expected=str(expected_latest_bar_start_utc),
                    got=str(last_raw_ts_utc),
                    lag_seconds=round(lag_seconds, 3),
                    attempt=attempt + 1,
                    max_attempts=self.max_poll_retries,
                )
                time.sleep(self.poll_retry_sleep_seconds)
                continue

            raise DataFetchError(
                f"Latest bar not yet available (expected {expected_latest_bar_start_utc}, got {last_raw_ts_utc})."
            )

        assert fetched is not None and idx_utc is not None and last_raw_ts_utc is not None

        if self.last_seen_raw_bar_ts_utc is not None and last_raw_ts_utc <= self.last_seen_raw_bar_ts_utc:
            logger.debug("No new TradingView bar to process (raw)", run_id=run_id, last=str(last_raw_ts_utc))
            self._journal("no_new_raw_bar", run_id=run_id, last_raw_ts=str(last_raw_ts_utc))
            return

        df_raw_new = self._tv_hist_to_raw_df(fetched, idx_utc)
        df_raw = self._merge_raw_cache(df_raw_new)
        self._log_recent_raw_bar_timestamps(df_raw)
        self._validate_raw_bar_timestamps(df_raw)

        ind_start = time.time()
        candle, signal_bar_ts_utc = self._signal_candle_from_raw_ohlc(
            df_raw=df_raw,
            expected_latest_bar_start_utc=expected_latest_bar_start_utc,
        )
        ind_seconds = time.time() - ind_start

        if self.last_processed_signal_bar_ts_utc is not None and signal_bar_ts_utc <= self.last_processed_signal_bar_ts_utc:
            self.last_seen_raw_bar_ts_utc = last_raw_ts_utc
            self._save_state()
            logger.debug("No new signal bar to process", run_id=run_id, signal=str(signal_bar_ts_utc))
            self._journal("no_new_signal_bar", run_id=run_id, signal_bar_ts=str(signal_bar_ts_utc))
            return

        self._journal("signal_bar_ready", run_id=run_id, signal_bar_ts=str(signal_bar_ts_utc), last_raw_ts=str(last_raw_ts_utc))
        self.next(candle, historical=False)
        self.last_seen_raw_bar_ts_utc = last_raw_ts_utc
        self.last_processed_signal_bar_ts_utc = signal_bar_ts_utc
        self._save_state()
        logger.debug(
            "TV fetch+process time",
            run_id=run_id,
            seconds=time.time() - start,
            tv_bars=fetch_bars,
            cache_bars=len(df_raw),
            indicator_seconds=ind_seconds,
            timeframe=interval,
            expected_latest_bar_start=str(expected_latest_bar_start_utc),
            last_raw_ts=str(last_raw_ts_utc),
            signal_bar_ts=str(signal_bar_ts_utc),
        )

    # ────────────────────── Indicator pipeline for OHLC ──────────────────────

    def _signal_candle_from_raw_ohlc(
        self, df_raw: pd.DataFrame, expected_latest_bar_start_utc: pd.Timestamp
    ) -> tuple[Candle, pd.Timestamp]:
        """
        Compute Barsmith-style indicators via the parity helper and return the
        signal candle (iloc[-2]) only after the latest bar (iloc[-1]) exists.

        Returns (signal_candle, signal_bar_ts_utc) where signal_bar_ts_utc is the
        UTC bar-start timestamp for the signal bar.
        """
        from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
            build_python_indicator_frame_minimal,
        )

        ind_df = build_python_indicator_frame_minimal(df_raw)
        ind_df = ind_df.sort_values("timestamp")

        if "is_kf_breakout_potential" not in ind_df.columns and {"kf_adx", "kf_atr"}.issubset(ind_df.columns):
            ind_df["kf_adx_increasing"] = ind_df["kf_adx"].diff() > 0.0
            ind_df["kf_atr_expanding"] = ind_df["kf_atr"].diff() > 0.0
            ind_df["is_kf_breakout_potential"] = ind_df["kf_adx_increasing"] & ind_df["kf_atr_expanding"]

        required_cols = [
            "open",
            "high",
            "low",
            "close",
            "timestamp",
            "is_tribar_green",
            "is_kf_breakout_potential",
            "kf_innovation_abs",
            "macd_hist",
            "trend_strength",
            "atr",
            "kf_adx",
            "kf_atr",
        ]
        missing = [c for c in required_cols if c not in ind_df.columns]
        if missing:
            raise KeyError(f"Missing required columns from indicator frame: {missing}")

        ready = ind_df.dropna(subset=required_cols).copy()
        if len(ready) < 2:
            raise DataFetchError(
                "Not enough TradingView history to compute Barsmith indicators for the pinned combos. "
                f"Need at least {BARSMITH_COMBO_REQUIRED_TV_BARS} bars; fetched {len(ind_df)}."
            )

        expected_latest_naive = pd.Timestamp(expected_latest_bar_start_utc).tz_convert(None)
        latest_ready_naive = pd.Timestamp(ready["timestamp"].iloc[-1])
        if latest_ready_naive < expected_latest_naive:
            raise DataFetchError(
                f"Latest bar not yet in indicator frame (expected {expected_latest_naive}, got {latest_ready_naive})."
            )

        signal_row = ready.iloc[-2]
        signal_ts_utc = pd.to_datetime(signal_row["timestamp"], utc=True, errors="coerce")
        if pd.isna(signal_ts_utc):
            raise DataFetchError("Signal bar timestamp is invalid")

        ts_local = pd.to_datetime(signal_row["timestamp"], utc=True).tz_convert(self.param.tz)
        return Candle(
            open=signal_row["open"],
            high=signal_row["high"],
            low=signal_row["low"],
            close=signal_row["close"],
            timestamp=ts_local,
            is_tribar_green=bool(signal_row["is_tribar_green"]),
            is_kf_breakout_potential=bool(signal_row["is_kf_breakout_potential"]),
            kf_innovation_abs=float(signal_row["kf_innovation_abs"]),
            macd_hist=float(signal_row["macd_hist"]),
            trend_strength=float(signal_row["trend_strength"]),
            atr=float(signal_row["atr"]),
            kf_adx=float(signal_row["kf_adx"]),
            kf_atr=float(signal_row["kf_atr"]),
        ), pd.Timestamp(signal_ts_utc)

    def _compute_setup_from_signal_bar(self, candle: Candle) -> Optional[dict]:
        """
        Compute a single setup (entry/stop/tp/rr) from a signal bar.

        This does not manage open trades; it's meant for dry-run scanning of prior setups.
        """
        if not self._barsmith_combo_signal_long(candle):
            return None

        body = candle.close - candle.open
        if abs(body) <= FLOAT_EPSILON:
            return None

        tick_size = self.param.contract.tick_size
        entry = quantize_price_to_tick(float(candle.close), tick_size, TickRoundMode.NEAREST)
        atr = float(candle.atr)
        if not math.isfinite(entry) or not math.isfinite(atr) or atr <= 0:
            return None

        stop_raw = entry - atr
        tp_raw = entry + float(self.param.target_atr_multiple) * atr
        stop = quantize_price_to_tick(stop_raw, tick_size, TickRoundMode.FLOOR)
        tp = quantize_price_to_tick(tp_raw, tick_size, TickRoundMode.CEIL)
        if not math.isfinite(stop) or not math.isfinite(tp) or stop >= entry:
            return None

        rr = utils.compute_rr(entry=entry, stop_loss=stop, exit=tp)
        if not math.isfinite(rr) or rr < self.param.minimum_risk_reward:
            return None

        return {
            "signal_dt": candle.timestamp,
            "entry": entry,
            "stop": stop,
            "tp": tp,
            "rr": rr,
            "atr": atr,
            "target_atr_multiple": float(self.param.target_atr_multiple),
        }

    def scan_setups_from_raw_ohlc(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Scan prior bars for qualifying setups.

        The newest bar is treated as the in-progress bar, so scanning only evaluates
        completed bars (equivalent to evaluating `iloc[-2]` in live polling).
        """
        from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
            build_python_indicator_frame_minimal,
        )

        ind_df = build_python_indicator_frame_minimal(df_raw).sort_values("timestamp")
        required_cols = [
            "open",
            "high",
            "low",
            "close",
            "timestamp",
            "is_tribar_green",
            "is_kf_breakout_potential",
            "kf_innovation_abs",
            "macd_hist",
            "trend_strength",
            "atr",
            "kf_adx",
            "kf_atr",
        ]
        ready = ind_df.dropna(subset=required_cols)
        if len(ready) < 2:
            return pd.DataFrame(columns=["signal_dt", "entry", "stop", "tp", "rr", "atr", "target_atr_multiple"])

        setups: list[dict] = []
        for _, row in ready.iloc[:-1].iterrows():
            ts_local = pd.to_datetime(row["timestamp"], utc=True).tz_convert(self.param.tz)
            candle = Candle(
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                timestamp=ts_local,
                is_tribar_green=bool(row["is_tribar_green"]),
                is_kf_breakout_potential=bool(row["is_kf_breakout_potential"]),
                kf_innovation_abs=float(row["kf_innovation_abs"]),
                macd_hist=float(row["macd_hist"]),
                trend_strength=float(row["trend_strength"]),
                atr=float(row["atr"]),
                kf_adx=float(row["kf_adx"]),
                kf_atr=float(row["kf_atr"]),
            )
            setup = self._compute_setup_from_signal_bar(candle)
            if setup is not None:
                setups.append(setup)

        return pd.DataFrame(setups)

    def simulate_realized_trades_from_raw_ohlc(self, df_raw: pd.DataFrame, include_open_trade: bool = False) -> pd.DataFrame:
        """
        Simulate the strategy over historical bars and return realized trade outcomes.

        - Uses the same indicator pipeline as live parity.
        - Treats the newest bar as in-progress and does not process it (matches live `iloc[-2]` signal gating).
        - Returns rows from `self.var.setups` (closed trades) with realized `rr`.
        """
        from algotrade.barsmith_parity.check_barsmith_indicator_parity import (
            build_python_indicator_frame_minimal,
        )

        # Reset runtime trade state for a clean simulation.
        self.var = deepcopy(self.var_default)
        self.extra = deepcopy(self.default_extra)

        ind_df = build_python_indicator_frame_minimal(df_raw).sort_values("timestamp")
        required_cols = [
            "open",
            "high",
            "low",
            "close",
            "timestamp",
            "is_tribar_green",
            "is_kf_breakout_potential",
            "kf_innovation_abs",
            "macd_hist",
            "trend_strength",
            "atr",
            "kf_adx",
            "kf_atr",
        ]
        ready = ind_df.dropna(subset=required_cols)
        if len(ready) < 2:
            return pd.DataFrame()

        # Exclude the newest bar (treat as in-progress).
        for _, row in ready.iloc[:-1].iterrows():
            ts_local = pd.to_datetime(row["timestamp"], utc=True).tz_convert(self.param.tz)
            candle = Candle(
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                timestamp=ts_local,
                is_tribar_green=bool(row["is_tribar_green"]),
                is_kf_breakout_potential=bool(row["is_kf_breakout_potential"]),
                kf_innovation_abs=float(row["kf_innovation_abs"]),
                macd_hist=float(row["macd_hist"]),
                trend_strength=float(row["trend_strength"]),
                atr=float(row["atr"]),
                kf_adx=float(row["kf_adx"]),
                kf_atr=float(row["kf_atr"]),
            )
            self.next(candle, historical=True)

        trades = list(self.var.setups)
        if include_open_trade and self.var.active_trade is not None:
            t = self.var.active_trade
            trades.append(
                {
                    "variation": self.extra.variation.value if self.extra.variation else None,
                    "side": t.side.value,
                    "entry_dt": t.entry_dt,
                    "exit_dt": None,
                    "entry": t.entry,
                    "exit": None,
                    "stop": t.stop,
                    "tp": t.tp,
                    "risk": t.risk,
                    "atr": t.atr,
                    "pnl_points": None,
                    "pnl_ticks": None,
                    "rr": None,
                    "hit_tp": None,
                    "gap_fill": None,
                    "open_trade": True,
                }
            )

        df = pd.DataFrame(trades)
        if not df.empty and "entry_dt" in df.columns:
            df = df.sort_values("entry_dt")
        return df

    # ───────────────────────── Risk / sizing helpers ──────────────────────────

    def get_quantity(self, stoploss_ticks: int) -> int:
        if stoploss_ticks <= 0:
            raise ValueError("Stop loss ticks must be positive.")

        risk_per_contract = stoploss_ticks * self.param.contract.tick_value
        if risk_per_contract <= 0:
            raise ValueError("Stop loss must translate to positive monetary risk.")

        risk_amount = self.param.capital * self.param.account_risk_pct
        raw_quantity = risk_amount / risk_per_contract
        quantity = int(round(raw_quantity))
        quantity = max(self.param.min_quantity, min(quantity, self.param.max_quantity))
        return quantity

    def _build_long_bracket(self, signal: Candle) -> Optional[types.BracketOrder]:
        """
        Build a long-side bracket (entry, stop, target) from a single signal bar.
        """
        body = signal.close - signal.open
        if abs(body) <= FLOAT_EPSILON or body < 0.0:
            return None

        tick_size = self.param.contract.tick_size

        # Entry is at signal bar close (IB requires tick-aligned limit price).
        entry_price = quantize_price_to_tick(float(signal.close), tick_size, TickRoundMode.NEAREST)
        stop_raw = entry_price - float(signal.atr)
        stop_price = quantize_price_to_tick(stop_raw, tick_size, TickRoundMode.FLOOR)
        if not math.isfinite(stop_price) or stop_price >= entry_price:
            return None

        stoploss_ticks = int((entry_price - stop_price) / tick_size)
        quantity = self.get_quantity(stoploss_ticks)

        order = types.LimitOrder(
            symbol=self.param.contract.ib,
            quantity=quantity,
            price=entry_price,
            action=types.Action.LONG,
            timestamp=None,
        )

        stop_loss = types.StopOrder(
            symbol=self.param.contract.ib,
            price=stop_price,
            action=types.Action.SHORT,
            quantity=quantity,
            timestamp=None,
        )

        take_profit_raw = entry_price + float(signal.atr) * float(self.param.target_atr_multiple)
        take_profit_price = quantize_price_to_tick(take_profit_raw, tick_size, TickRoundMode.CEIL)

        take_profit = types.LimitOrder(
            symbol=self.param.contract.ib,
            quantity=quantity,
            price=take_profit_price,
            action=types.Action.SHORT,
            timestamp=None,
        )

        bracket = types.BracketOrder(
            symbol=self.param.contract.ib,
            order=order,
            stop_loss=stop_loss,
            take_profit=take_profit,
            extra=deepcopy(self.extra),
            params=deepcopy(self.param),
            var=deepcopy(self.var),
            timestamp=signal.timestamp,
        )

        rr = utils.compute_rr(
            entry=bracket.order.price,
            stop_loss=bracket.stop_loss.price,
            exit=bracket.take_profit.price,
        )

        logger.debug(
            "Barsmith combo bracket",
            rr=rr,
            entry=bracket.order.price,
            stop_loss=bracket.stop_loss.price,
            target=bracket.take_profit.price,
            qty=bracket.order.quantity,
            atr=signal.atr,
            atr_product=signal.atr * self.param.target_atr_multiple,
        )

        if rr < self.param.minimum_risk_reward:
            logger.debug("Skipping: RR less than minimum", rr=rr, min_rr=self.param.minimum_risk_reward)
            return None

        self.var.setups.append(
            {
                "date": signal.timestamp,
                "entry": bracket.order.price,
                "target": bracket.take_profit.price,
                "stop_loss": bracket.stop_loss.price,
                "rr": rr,
                "atr": signal.atr,
            }
        )

        return bracket

    def _build_short_bracket(self, signal: Candle) -> Optional[types.BracketOrder]:
        """
        This model is long-only (combo uses is_tribar_green).
        """
        return None

    # ───────────────────────── Core signal evaluation ─────────────────────────

    @staticmethod
    def _barsmith_combo_signal_long(candle: Candle) -> bool:
        """
        Pinned Barsmith combo (long, evaluated at bar close):

          is_tribar_green && is_kf_breakout_potential && kf_innovation_abs>5.02 && macd_hist>=-1.0 && trend_strength<=0.25
        """
        return (
            candle.is_tribar_green
            and candle.is_kf_breakout_potential
            and candle.kf_innovation_abs > 5.02
            and candle.macd_hist >= -1.0
            and candle.trend_strength <= 0.25
        )

    @staticmethod
    def _barsmith_combo_signal_short(candle: Candle) -> bool:
        """
        This model is long-only (combo uses is_tribar_green).
        """
        return False

    def _maybe_close_active_trade(self, bar: Candle) -> None:
        trade = self.var.active_trade
        if trade is None:
            return

        stop = trade.stop
        tp = trade.tp
        entry = trade.entry
        risk = trade.risk
        side = trade.side

        if side != TradeSide.LONG:
            return

        # Gap-aware fills: if the bar opens beyond stop/TP, fill at open.
        if math.isfinite(bar.open):
            if bar.open <= stop:
                exit_price = bar.open
                rr = (exit_price - entry) / risk
                hit_tp = False
                self._record_trade(
                    trade=trade,
                    exit_dt=bar.timestamp,
                    exit_price=exit_price,
                    rr=rr,
                    hit_tp=hit_tp,
                    gap_fill=True,
                )
                return
            if bar.open >= tp:
                exit_price = bar.open
                rr = (exit_price - entry) / risk
                hit_tp = True
                self._record_trade(
                    trade=trade,
                    exit_dt=bar.timestamp,
                    exit_price=exit_price,
                    rr=rr,
                    hit_tp=hit_tp,
                    gap_fill=True,
                )
                return

        # Conservative ordering: SL dominates if both touched in-bar.
        if bar.low <= stop:
            exit_price = stop
            rr = -1.0
            hit_tp = False
            self._record_trade(
                trade=trade,
                exit_dt=bar.timestamp,
                exit_price=exit_price,
                rr=rr,
                hit_tp=hit_tp,
                gap_fill=False,
            )
            return

        if bar.high >= tp:
            exit_price = tp
            rr = (exit_price - entry) / risk
            hit_tp = True
            self._record_trade(
                trade=trade,
                exit_dt=bar.timestamp,
                exit_price=exit_price,
                rr=rr,
                hit_tp=hit_tp,
                gap_fill=False,
            )

    def _record_trade(
        self,
        trade: ActiveTrade,
        exit_dt: datetime,
        exit_price: float,
        rr: float,
        hit_tp: bool,
        gap_fill: bool,
    ) -> None:
        pnl_points = exit_price - trade.entry
        pnl_ticks = pnl_points / self.param.contract.tick_size
        self.var.setups.append(
            {
                "variation": self.extra.variation.value if self.extra.variation else None,
                "side": trade.side.value,
                "entry_dt": trade.entry_dt,
                "exit_dt": exit_dt,
                "entry": trade.entry,
                "exit": exit_price,
                "stop": trade.stop,
                "tp": trade.tp,
                "risk": trade.risk,
                "atr": trade.atr,
                "pnl_points": pnl_points,
                "pnl_ticks": pnl_ticks,
                "rr": rr,
                "hit_tp": hit_tp,
                "gap_fill": gap_fill,
            }
        )
        logger.debug(
            "Barsmith 2x_atr_tp_atr_stop trade",
            variation=self.extra.variation.value if self.extra.variation else None,
            entry_dt=trade.entry_dt,
            exit_dt=exit_dt,
            entry=trade.entry,
            exit=exit_price,
            stop=trade.stop,
            tp=trade.tp,
            rr=rr,
            hit_tp=hit_tp,
            gap_fill=gap_fill,
        )
        self.var.active_trade = None
        self.var.open_trade = False

    def next(self, stream: Candle, historical: bool = False) -> None:
        if self.precheck_failed:
            logger.error("Precheck failed. Exiting.")
            return

        if not isinstance(stream, Candle):
            raise TypeError("Stream must be of type Candle.")

        if self.mode == ExecutionMode.LIVE:
            signal_ts_utc = self._to_utc_timestamp(stream.timestamp)
            if self.last_submitted_signal_bar_ts_utc is not None and signal_ts_utc <= self.last_submitted_signal_bar_ts_utc:
                logger.debug("Skip duplicate live signal", signal=str(signal_ts_utc))
                return

            if not self._barsmith_combo_signal_long(stream):
                self._journal("live_no_signal", signal=str(signal_ts_utc))
                return

            if not isinstance(self.broker, brokers.IBKRBroker):
                raise RuntimeError("LIVE mode requires IBKRBroker")

            now_utc = utils.get_utc()
            self._reset_daily_if_needed(now_utc)
            if self._is_disabled(now_utc):
                logger.warning(
                    "Trading disabled by circuit breaker; skipping live signal",
                    disabled_until=self.circuit_breaker.disabled_until_utc,
                )
                self._journal(
                    "disabled_skip_signal", signal=str(signal_ts_utc), disabled_until=self.circuit_breaker.disabled_until_utc
                )
                return
            if self.require_market_open and not self._is_market_open_utc(now_utc):
                logger.info("Market closed; skipping live signal")
                self._journal("market_closed_skip_signal", signal=str(signal_ts_utc))
                return
            if self.max_trades_per_day > 0 and self.circuit_breaker.trades_submitted_today >= self.max_trades_per_day:
                logger.warning(
                    "Daily trade limit reached; skipping signal",
                    limit=self.max_trades_per_day,
                    trades_submitted_today=self.circuit_breaker.trades_submitted_today,
                )
                self._journal(
                    "daily_limit_skip_signal",
                    signal=str(signal_ts_utc),
                    limit=self.max_trades_per_day,
                    trades_submitted_today=self.circuit_breaker.trades_submitted_today,
                )
                return

            self.broker.ensure_connected()
            contract = self.param.contract.ib
            positions = self.broker.get_positions(contract=contract)
            if any(getattr(p, "position", 0) != 0 for p in positions):
                self._maybe_recover_live_protection(positions=positions)
                logger.info("Position already open; skipping new entry", positions=len(positions))
                self._journal("position_open_skip_entry", signal=str(signal_ts_utc), positions=len(positions))
                return

            open_orders = self.broker.get_orders(contract=contract)
            if open_orders:
                logger.info("Open orders exist; skipping new entry", open_orders=len(open_orders))
                self._journal("open_orders_skip_entry", signal=str(signal_ts_utc), open_orders=len(open_orders))
                return

            bracket = self._build_long_bracket(stream)
            if bracket is None:
                self._journal("bracket_none", signal=str(signal_ts_utc))
                return

            logger.info(
                "Submitting bracket (live)",
                signal=str(signal_ts_utc),
                entry=bracket.order.price,
                stop=bracket.stop_loss.price if bracket.stop_loss else None,
                tp=bracket.take_profit.price if bracket.take_profit else None,
                qty=bracket.order.quantity,
            )
            try:
                trades = self.broker.order(bracket)
            except Exception as exc:
                self._journal("live_order_submit_failed", signal=str(signal_ts_utc), error=str(exc))
                raise

            order_ids: list[int] = []
            try:
                if isinstance(trades, list):
                    for t in trades:
                        oid = getattr(getattr(t, "order", None), "orderId", None)
                        if oid is not None:
                            order_ids.append(int(oid))
            except Exception:
                order_ids = []

            self.last_submitted_signal_bar_ts_utc = signal_ts_utc
            self.circuit_breaker.trades_submitted_today += 1
            self.last_bracket_snapshot = LiveBracketSnapshot(
                signal_bar_ts_utc=str(signal_ts_utc),
                submitted_at_utc=utils.get_utc().isoformat(),
                entry=float(bracket.order.price),
                stop_loss=float(bracket.stop_loss.price) if bracket.stop_loss else float("nan"),
                take_profit=float(bracket.take_profit.price) if bracket.take_profit else float("nan"),
                quantity=int(bracket.order.quantity),
                ib_order_ids=order_ids,
            )
            self._save_state()
            self._journal(
                "live_bracket_submitted",
                signal=str(signal_ts_utc),
                entry=bracket.order.price,
                stop=bracket.stop_loss.price if bracket.stop_loss else None,
                tp=bracket.take_profit.price if bracket.take_profit else None,
                qty=bracket.order.quantity,
                order_ids=order_ids,
                trades_submitted_today=self.circuit_breaker.trades_submitted_today,
            )
            return

        # Resolve an existing open trade first (no stacking).
        if self.var.active_trade is not None:
            self._maybe_close_active_trade(stream)
            return

        body = stream.close - stream.open
        if abs(body) <= FLOAT_EPSILON:
            return

        if not self._barsmith_combo_signal_long(stream):
            return
        self.extra.variation = Variation.BARSMITH_COMBO_LONG

        tick_size = self.param.contract.tick_size
        entry = quantize_price_to_tick(float(stream.close), tick_size, TickRoundMode.NEAREST)
        atr = float(stream.atr)
        if not math.isfinite(entry) or not math.isfinite(atr) or atr <= 0:
            return

        stop_raw = entry - atr
        tp_raw = entry + float(self.param.target_atr_multiple) * atr
        stop = quantize_price_to_tick(stop_raw, tick_size, TickRoundMode.FLOOR)
        tp = quantize_price_to_tick(tp_raw, tick_size, TickRoundMode.CEIL)
        if not math.isfinite(stop) or not math.isfinite(tp) or stop >= entry:
            return
        risk = entry - stop

        if risk <= SMALL_DIVISOR:
            return

        self.var.active_trade = ActiveTrade(
            side=TradeSide.LONG,
            entry_dt=stream.timestamp,
            entry=entry,
            stop=stop,
            tp=tp,
            risk=risk,
            atr=atr,
        )
        self.var.open_trade = True
        logger.debug(
            "Opened 2x_atr_tp_atr_stop trade",
            side=TradeSide.LONG.value,
            entry_dt=stream.timestamp,
            entry=entry,
            stop=stop,
            tp=tp,
            risk=risk,
            atr=atr,
        )

    # ────────────────────── TradingView-only model ────────────────────────────


def dry_run_barsmith_combo(
    contract: types.SupportedContract,
    timeframe: types.Timeframe = types.Timeframe.m30,
    capital: float = 10000.0,
    account_risk_pct: float = 0.01,
    use_broker: bool = False,
    ib_client_id: int = 1,
    tv_required_bars: int = BARSMITH_COMBO_REQUIRED_TV_BARS,
) -> BarsmithComboExecutor:
    """
    Convenience helper to run the Barsmith combo against TradingView OHLC using
    Python parity indicators (no CSV/DuckDB dependencies).
    """
    contract_spec = get_contract_spec(contract)
    repo = repository.Repository()

    broker: brokers.BaseBroker
    broker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)
    if use_broker:
        import ib_async

        ib_async.util.startLoop()
        broker = brokers.IBKRBroker(repo, port=4002, account=settings.ib_account, ib_client_id=ib_client_id)
        broker.connect()

    param = Param(
        contract=contract_spec,
        supported_contract=contract,
        tz=IB_TZ,
        timeframe=timeframe,
        capital=capital,
        account_risk_pct=account_risk_pct,
    )

    var = Var()
    extra = Extra()
    strategy = BarsmithComboExecutor(
        broker=broker,
        param=param,
        var=var,
        extra=extra,
        tv_required_bars=tv_required_bars,
    )
    strategy.pull_historical_data()

    if isinstance(broker, brokers.IBKRBroker):
        broker.disconnect()

    return strategy


def _parse_enum_value(enum_cls: type[Enum], value: str) -> Enum:
    normalized = value.strip()
    if not normalized:
        raise ValueError("empty value")
    upper = normalized.upper()
    for member in enum_cls:  # type: ignore[assignment]
        if getattr(member, "name", "").upper() == upper:
            return member
        if str(getattr(member, "value", "")).upper() == upper:
            return member
    choices = ", ".join(sorted({m.name for m in enum_cls}))  # type: ignore[arg-type]
    raise ValueError(f"unknown value '{value}' (choices: {choices})")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="barsmith_combo_2x_atr_tp_atr_stop",
        description="Run the pinned Barsmith combo model (2x_atr_tp_atr_stop).",
    )
    parser.add_argument("--contract", default="MES", help="SupportedContract name (e.g. MES)")
    parser.add_argument("--timeframe", default="m30", help="Timeframe (m30, h4, daily)")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--account-risk-pct", type=float, default=0.01)
    parser.add_argument("--tv-required-bars", type=int, default=BARSMITH_COMBO_REQUIRED_TV_BARS)
    parser.add_argument("--tv-lag-seconds", type=float, default=60.0, help="Max tolerated TV bar delay near boundaries")
    parser.add_argument("--max-poll-retries", type=int, default=5, help="Retries when the latest bar hasn't appeared yet")
    parser.add_argument("--poll-retry-sleep-seconds", type=float, default=0.5, help="Sleep between retries for latest bar")
    parser.add_argument("--max-trades-per-day", type=int, default=1, help="Max live entries per UTC day (0 disables limit)")
    parser.add_argument("--max-consecutive-failures", type=int, default=5, help="Failures before circuit breaker trips")
    parser.add_argument(
        "--failure-cooldown-seconds", type=float, default=3600.0, help="Disable trading for this long after breaker trips"
    )
    parser.add_argument(
        "--require-market-open", action=argparse.BooleanOptionalAction, default=True, help="Skip runs during weekend breaks"
    )
    parser.add_argument(
        "--entry-timeout-seconds",
        type=float,
        default=7200.0,
        help="Cancel resting entry orders after this many seconds (0 disables)",
    )
    parser.add_argument(
        "--repair-orders", action=argparse.BooleanOptionalAction, default=True, help="Attempt safe order repairs in live mode"
    )
    parser.add_argument(
        "--enforce-exit-prices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, enforce snapshot TP/SL prices when repairing",
    )
    parser.add_argument("--state-path", default=None, help="Optional JSON state file path (defaults to tmp/state/...)")
    parser.add_argument("--journal-path", default=None, help="Optional JSONL journal path (defaults to tmp/journal/...)")

    parser.add_argument("--use-broker", action="store_true", help="Place real orders via IBKR (paper port 4002)")
    parser.add_argument("--ib-client-id", type=int, default=1)
    parser.add_argument("--mode", choices=[m.value for m in ExecutionMode], default=ExecutionMode.SIM.value)

    parser.add_argument("--poll", action="store_true", help="Poll TradingView repeatedly instead of running once")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--nbars", type=int, default=0, help="Fetch N bars and scan prior setups (0 disables scan mode)")
    parser.add_argument("--scan-output", default=None, help="Optional CSV path for scan output (defaults to tmp/...)")

    args = parser.parse_args(argv)

    try:
        contract = _parse_enum_value(types.SupportedContract, args.contract)  # type: ignore[assignment]
        timeframe = _parse_enum_value(types.Timeframe, args.timeframe)  # type: ignore[assignment]
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if not math.isfinite(args.capital) or args.capital <= 0:
        parser.error("--capital must be positive")
    if not math.isfinite(args.account_risk_pct) or args.account_risk_pct <= 0:
        parser.error("--account-risk-pct must be positive")
    if args.tv_required_bars <= 0:
        parser.error("--tv-required-bars must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not math.isfinite(args.tv_lag_seconds) or args.tv_lag_seconds < 0:
        parser.error("--tv-lag-seconds must be >= 0")
    if args.max_poll_retries < 0:
        parser.error("--max-poll-retries must be >= 0")
    if not math.isfinite(args.poll_retry_sleep_seconds) or args.poll_retry_sleep_seconds <= 0:
        parser.error("--poll-retry-sleep-seconds must be > 0")
    if args.max_trades_per_day < 0:
        parser.error("--max-trades-per-day must be >= 0")
    if args.max_consecutive_failures <= 0:
        parser.error("--max-consecutive-failures must be > 0")
    if not math.isfinite(args.failure_cooldown_seconds) or args.failure_cooldown_seconds < 0:
        parser.error("--failure-cooldown-seconds must be >= 0")
    if not math.isfinite(args.entry_timeout_seconds) or args.entry_timeout_seconds < 0:
        parser.error("--entry-timeout-seconds must be >= 0")

    if args.use_broker:
        import ib_async

        ib_async.util.startLoop()
    if ExecutionMode(args.mode) == ExecutionMode.LIVE and not args.use_broker:
        parser.error("--mode live requires --use-broker (IBKR connection)")
    if ExecutionMode(args.mode) == ExecutionMode.LIVE:
        require_live_trading_enabled(
            is_live=settings.is_live,
            live_confirm=settings.live_confirm,
            operation="barsmith_combo_2x_atr_tp_atr_stop.main",
        )

    contract_spec = get_contract_spec(contract)  # type: ignore[arg-type]
    repo = repository.Repository()
    broker: brokers.BaseBroker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)
    if args.use_broker:
        broker = brokers.IBKRBroker(repo, port=4002, account=settings.ib_account, ib_client_id=args.ib_client_id)
        broker.connect()

    param = Param(
        contract=contract_spec,
        supported_contract=contract,  # type: ignore[arg-type]
        tz=IB_TZ,
        timeframe=timeframe,  # type: ignore[arg-type]
        capital=float(args.capital),
        account_risk_pct=float(args.account_risk_pct),
    )

    var = Var()
    extra = Extra()
    strategy = BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=var,
        extra=extra,
        tv_required_bars=int(args.tv_required_bars),
        mode=ExecutionMode(args.mode),
        tv_lag_seconds=float(args.tv_lag_seconds),
        max_poll_retries=int(args.max_poll_retries),
        poll_retry_sleep_seconds=float(args.poll_retry_sleep_seconds),
        max_trades_per_day=int(args.max_trades_per_day),
        max_consecutive_failures=int(args.max_consecutive_failures),
        failure_cooldown_seconds=float(args.failure_cooldown_seconds),
        require_market_open=bool(args.require_market_open),
        entry_timeout_seconds=float(args.entry_timeout_seconds),
        repair_orders=bool(args.repair_orders),
        enforce_exit_prices=bool(args.enforce_exit_prices),
        state_path=args.state_path,
        journal_path=args.journal_path,
    )

    try:
        if args.nbars and args.nbars > 0:
            tv = strategy.get_tv()
            interval = TIMEFRAME_TO_TV_INTERVAL[strategy.param.timeframe]
            fetched = tv.get_hist(
                symbol=strategy.param.contract.symbol,
                exchange=strategy.param.contract.exchange,
                interval=interval,
                n_bars=int(args.nbars),
                fut_contract=1,
                extended_session=True,
            )
            if fetched is None or fetched.empty:
                raise DataFetchError("No data fetched from TradingView")
            idx_utc = strategy._to_utc_index(pd.DatetimeIndex(fetched.index))
            df_raw = strategy._tv_hist_to_raw_df(fetched, idx_utc)
            df_setups = strategy.simulate_realized_trades_from_raw_ohlc(df_raw)

            if args.scan_output:
                out_path = Path(args.scan_output)
            else:
                out_dir = Path("tmp") / "barsmith_combo"
                out_dir.mkdir(parents=True, exist_ok=True)
                symbol = str(getattr(contract, "name", contract)).lower()
                tf = str(getattr(timeframe, "value", timeframe)).lower()
                out_path = out_dir / f"setups_2x_atr_tp_atr_stop_{symbol}_{tf}.csv"
            df_setups.to_csv(out_path, index=False)
            logger.info("Wrote realized trades scan", path=str(out_path), rows=len(df_setups))
            return 0

        if not args.poll:
            strategy.pull_historical_data()
            return 0

        logger.info(
            "Starting polling loop",
            contract=str(getattr(contract, "name", contract)),
            timeframe=str(getattr(timeframe, "value", timeframe)),
            poll_seconds=args.poll_seconds,
            tv_required_bars=args.tv_required_bars,
            use_broker=args.use_broker,
        )
        while True:
            strategy.pull_historical_data()
            time.sleep(float(args.poll_seconds))
    except KeyboardInterrupt:
        logger.info("Interrupted, exiting")
        return 0
    finally:
        if isinstance(broker, brokers.IBKRBroker):
            broker.disconnect()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

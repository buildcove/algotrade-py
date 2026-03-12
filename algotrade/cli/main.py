"""
Main CLI module for the algotrade system.

This module provides command-line interface for running trading strategies,
scheduling automated trading, and managing broker operations.
"""

import asyncio
import time
from datetime import timedelta
from enum import Enum
from typing import List, Optional

import ib_async
import pytz
import structlog
import typer
from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, repository, types, utils
from algotrade.core.safety import require_live_trading_enabled
from algotrade.models import barsmith_combo_2x_atr_tp_atr_stop as barsmith_combo
from algotrade.models import tribar_v2 as tribar
from algotrade.scheduler import scheduler
from algotrade.settings import settings
from apscheduler.triggers.date import DateTrigger

logger = structlog.get_logger()
app = typer.Typer()


class ScheduleType(str, Enum):
    """Type of trading environment for scheduling."""

    live = "live"
    paper = "paper"


class MonitorType(str, Enum):
    """Type of monitoring job to schedule."""

    end_of_week = "end_of_week"
    stoploss_mover = "stoploss_mover"


@app.command()
def api(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """Run the FastAPI server via uvicorn."""
    import uvicorn

    uvicorn.run("algotrade.api.app:app", host=host, port=port, reload=reload)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_trading_param(
    contract: types.SupportedContract,
    timeframe: types.Timeframe,
    allowed_setups: List[tribar.Variation] = None,
    not_allowed_setups: List[tribar.Variation] = None,
    capital: float = 10000.0,
    account_risk: tribar.AccountRisk = tribar.AccountRisk.ONE_PERCENT,
    max_quantity: int = 1,
    min_quantity: int = 1,
    risk_profile: tribar.RiskProfile = tribar.RiskProfile.ROUND,
    gt: Optional[float] = None,
    gte: Optional[float] = None,
    lt: Optional[float] = None,
    lte: Optional[float] = None,
    gt_target: Optional[float] = None,
    gte_target: Optional[float] = None,
    lt_target: Optional[float] = None,
    lte_target: Optional[float] = None,
    long_gt: Optional[float] = None,
    long_gte: Optional[float] = None,
    short_gt: Optional[float] = None,
    short_gte: Optional[float] = None,
    target_atr_mul: Optional[float] = None,
    with_crossing_sma: bool = False,
    follow_200sma: bool = False,
    target_bar_dt: Optional[types.datetime] = None,
    last_bar_as_signal: bool = False,
) -> tribar.Param:
    """Create a standardized Param object for trading strategies."""
    contract_spec = get_contract_spec(contract)

    param = tribar.Param(
        contract=contract_spec,
        tz="UTC",
        allowed_setups=allowed_setups or [],
        not_allowed_setups=not_allowed_setups or [],
        capital=capital,
        account_risk=account_risk,
        max_quantity=max_quantity,
        min_quantity=min_quantity,
        risk_profile=risk_profile,
        timeframe=timeframe,
        gt=gt,
        gte=gte,
        lt=lt,
        lte=lte,
        gt_target=gt_target,
        gte_target=gte_target,
        lt_target=lt_target,
        lte_target=lte_target,
        long_gt=long_gt,
        long_gte=long_gte,
        short_gt=short_gt,
        short_gte=short_gte,
        target_bar_dt=target_bar_dt,
        last_bar_as_signal=last_bar_as_signal,
        with_crossing_sma=with_crossing_sma,
        follow_200sma=follow_200sma,
    )

    if target_atr_mul is not None:
        param.target_atr_multiple = target_atr_mul

    return param


def create_broker(type_: ScheduleType, ib_client_id: int) -> brokers.IBKRBroker:
    """Create and return an IBKR broker instance."""
    if type_ == ScheduleType.live:
        require_live_trading_enabled(
            is_live=settings.is_live,
            live_confirm=settings.live_confirm,
            operation="algotrade.cli.main.create_broker",
        )

    ib_async.util.startLoop()
    repo = repository.Repository()
    port = 4001 if type_ == ScheduleType.live else 4002
    broker = brokers.IBKRBroker(repo, port=port, account=settings.ib_account, ib_client_id=ib_client_id)
    broker.connect()
    return broker


def run_trading_task(
    type_: str,
    param: tribar.Param,
    ib_client_id: int,
    precheck_only: bool = False,
    check_tv: bool = False,
) -> None:
    """Execute a trading task with the given parameters."""
    task = tribar.Live(
        type_=type_,
        param=param,
        ib_client_id=ib_client_id,
    )

    task.run_precheck()

    if check_tv:
        task.check_tv_live()

    if not precheck_only:
        task.run()

    task.broker.disconnect()


def calculate_next_run_time(
    timeframe: types.Timeframe,
    contract: types.SupportedContract,
    now: types.datetime,
) -> types.datetime:
    """Calculate the next run time based on timeframe and contract."""
    if timeframe == types.Timeframe.m30:
        return utils.next_half_hour_for_contract(contract, now)
    elif timeframe == types.Timeframe.h4:
        return utils.next_4hour_for_contract(contract, now, skip_sunday=True)
    elif timeframe == types.Timeframe.daily:
        return utils.next_daily_for_contract(contract, now)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def compute_barsmith_next_run(
    timeframe: types.Timeframe,
    contract: types.SupportedContract,
    now: types.datetime,
    close_delay_seconds: float,
) -> tuple[types.datetime, types.datetime]:
    """
    Return (next_bar_boundary, run_at) where run_at is next_bar_boundary + close_delay_seconds.

    This is used to avoid polling TradingView; instead we fetch once per expected bar.
    """
    if close_delay_seconds < 0:
        raise ValueError("close_delay_seconds must be >= 0")
    next_bar = calculate_next_run_time(timeframe=timeframe, contract=contract, now=now)
    run_at = next_bar + timedelta(seconds=close_delay_seconds)
    return next_bar, run_at


# =============================================================================
# TRADING COMMANDS
# =============================================================================


@app.command()
def backtest_run():
    """Run backtesting analysis."""
    try:
        tribar.backtest()
        logger.info("Backtest completed successfully")
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        raise typer.Exit(1)


@app.command()
def dry_run(
    use_broker: bool = False,
    contract: types.SupportedContract = types.SupportedContract.ZM,
    timeframe: types.Timeframe = types.Timeframe.daily,
    allowed_setups: List[tribar.Variation] = [],
    not_allowed_setups: List[tribar.Variation] = [],
    capital: float = 10000.0,
    account_risk: tribar.AccountRisk = tribar.AccountRisk.ONE_PERCENT,
    max_quantity: int = 1,
    min_quantity: int = 1,
    risk_profile: tribar.RiskProfile = tribar.RiskProfile.ROUND,
    gt: Optional[float] = None,
    gte: Optional[float] = None,
    lt: Optional[float] = None,
    lte: Optional[float] = None,
    gt_target: Optional[float] = None,
    gte_target: Optional[float] = None,
    lt_target: Optional[float] = None,
    lte_target: Optional[float] = None,
    long_gt: Optional[float] = None,
    long_gte: Optional[float] = None,
    short_gt: Optional[float] = None,
    short_gte: Optional[float] = None,
    target_atr_mul: Optional[float] = None,
    with_crossing_sma: bool = False,
    follow_200sma: bool = False,
    last_bar_as_signal: bool = False,
    ib_client_id: int = 1,
):
    """Run dry run without broker connection."""
    tribar.dry_run(
        use_broker=use_broker,
        contract=contract,
        timeframe=timeframe,
        allowed_setups=allowed_setups,
        not_allowed_setups=not_allowed_setups,
        capital=capital,
        account_risk=account_risk,
        max_quantity=max_quantity,
        min_quantity=min_quantity,
        risk_profile=risk_profile,
        gt=gt,
        gte=gte,
        lt=lt,
        lte=lte,
        gt_target=gt_target,
        gte_target=gte_target,
        lt_target=lt_target,
        lte_target=lte_target,
        long_gt=long_gt,
        long_gte=long_gte,
        short_gt=short_gt,
        short_gte=short_gte,
        target_atr_mul=target_atr_mul,
        with_crossing_sma=with_crossing_sma,
        # follow_200sma=follow_200sma,
        last_bar_as_signal=last_bar_as_signal,
        ib_client_id=ib_client_id,
    )


@app.command()
def live_run(
    timeframe: types.Timeframe,
    contract: types.SupportedContract = types.SupportedContract.MGC,
):
    """Run live trading for specified timeframe."""
    try:
        v_start = time.time()

        # Set last_bar_as_signal to True for daily timeframe
        last_bar_as_signal = timeframe == types.Timeframe.daily

        param = create_trading_param(
            contract=contract,
            timeframe=timeframe,
            last_bar_as_signal=last_bar_as_signal,
        )

        logger.debug("Initializing Live", time=time.time() - v_start)

        run_trading_task(
            type_="live",
            param=param,
            ib_client_id=1,
        )

        logger.debug("Total execution", time=time.time() - v_start)
        logger.info("Live run completed successfully")

    except Exception as e:
        logger.error("Live run failed", error=str(e))
        raise typer.Exit(1)


@app.command()
def paper_run(
    timeframe: types.Timeframe,
    contract: types.SupportedContract = types.SupportedContract.MGC,
):
    """Run paper trading for specified timeframe."""
    try:
        v_start = time.time()

        # Set last_bar_as_signal to True for daily timeframe
        last_bar_as_signal = timeframe == types.Timeframe.daily

        param = create_trading_param(
            contract=contract,
            timeframe=timeframe,
            last_bar_as_signal=last_bar_as_signal,
        )

        logger.debug("Initializing Paper", time=time.time() - v_start)

        run_trading_task(
            type_="paper",
            param=param,
            ib_client_id=1,
        )

        logger.debug("Total execution", time=time.time() - v_start)
        logger.info("Paper run completed successfully")

    except Exception as e:
        logger.error("Paper run failed", error=str(e))
        raise typer.Exit(1)


@app.command()
def barsmith_schedule(
    type_: ScheduleType,
    timeframe: types.Timeframe,
    contract: types.SupportedContract,
    mode: barsmith_combo.ExecutionMode = barsmith_combo.ExecutionMode.SIM,
    close_delay_seconds: float = 3.0,
    n_times: int = 0,
    tv_required_bars: int = barsmith_combo.BARSMITH_COMBO_REQUIRED_TV_BARS,
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
    lag_retry_seconds: float = 10.0,
    lag_retry_max: int = 6,
    state_path: Optional[str] = None,
    journal_path: Optional[str] = None,
    ib_client_id: int = 1,
):
    """
    Schedule Barsmith combo runs at the expected next bar boundary (no polling).

    - Sleeps until the next expected bar time (based on timeframe/contract), then runs once.
    - Adds a small close delay (seconds) to allow TradingView to publish the latest bar.
    - If the latest bar is still missing, retries a bounded number of times, then waits for the next boundary.
    """
    if close_delay_seconds < 0:
        raise typer.BadParameter("close_delay_seconds must be >= 0")
    if n_times < 0:
        raise typer.BadParameter("n_times must be >= 0 (0 means run forever)")
    if tv_required_bars <= 0:
        raise typer.BadParameter("tv_required_bars must be > 0")
    if tv_lag_seconds < 0:
        raise typer.BadParameter("tv_lag_seconds must be >= 0")
    if max_poll_retries < 0:
        raise typer.BadParameter("max_poll_retries must be >= 0")
    if poll_retry_sleep_seconds <= 0:
        raise typer.BadParameter("poll_retry_sleep_seconds must be > 0")
    if lag_retry_seconds <= 0:
        raise typer.BadParameter("lag_retry_seconds must be > 0")
    if lag_retry_max < 0:
        raise typer.BadParameter("lag_retry_max must be >= 0")
    if max_trades_per_day < 0:
        raise typer.BadParameter("max_trades_per_day must be >= 0")
    if max_consecutive_failures <= 0:
        raise typer.BadParameter("max_consecutive_failures must be > 0")
    if failure_cooldown_seconds < 0:
        raise typer.BadParameter("failure_cooldown_seconds must be >= 0")
    if entry_timeout_seconds < 0:
        raise typer.BadParameter("entry_timeout_seconds must be >= 0")

    repo = repository.Repository()
    contract_spec = get_contract_spec(contract)

    broker: brokers.BaseBroker
    if mode == barsmith_combo.ExecutionMode.LIVE:
        port = 4001 if type_ == ScheduleType.live else 4002
        ib_async.util.startLoop()
        broker = brokers.IBKRBroker(repo, port=port, account=settings.ib_account, ib_client_id=ib_client_id)
    else:
        broker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)

    param = barsmith_combo.Param(
        contract=contract_spec,
        supported_contract=contract,
        tz="UTC",
        timeframe=timeframe,
    )

    strategy = barsmith_combo.BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=barsmith_combo.Var(),
        extra=barsmith_combo.Extra(),
        tv_required_bars=tv_required_bars,
        mode=mode,
        tv_lag_seconds=tv_lag_seconds,
        max_poll_retries=max_poll_retries,
        poll_retry_sleep_seconds=poll_retry_sleep_seconds,
        max_trades_per_day=max_trades_per_day,
        max_consecutive_failures=max_consecutive_failures,
        failure_cooldown_seconds=failure_cooldown_seconds,
        require_market_open=require_market_open,
        entry_timeout_seconds=entry_timeout_seconds,
        repair_orders=repair_orders,
        enforce_exit_prices=enforce_exit_prices,
        state_path=state_path,
        journal_path=journal_path,
    )

    runs_remaining: Optional[int] = None if n_times == 0 else n_times
    try:
        while runs_remaining is None or runs_remaining > 0:
            now = utils.get_utc()
            next_bar, run_at = compute_barsmith_next_run(
                timeframe=timeframe, contract=contract, now=now, close_delay_seconds=close_delay_seconds
            )
            sleep_seconds = max(0.0, (run_at - now).total_seconds())
            tz = pytz.timezone(settings.user_tz)
            logger.info(
                "Barsmith scheduled run",
                contract=contract.value,
                timeframe=timeframe.value,
                mode=mode.value,
                next_bar=next_bar.astimezone(tz).strftime("%b %d, %Y, %-I:%M:%S %p %z"),
                run_at=run_at.astimezone(tz).strftime("%b %d, %Y, %-I:%M:%S %p %z"),
                sleep_seconds=round(sleep_seconds, 3),
            )
            time.sleep(sleep_seconds)

            # Run once at the expected time, with bounded retry if TV is still late.
            succeeded = False
            last_error: Optional[Exception] = None
            for attempt in range(lag_retry_max + 1):
                try:
                    strategy.pull_historical_data()
                    succeeded = True
                    break
                except barsmith_combo.DataFetchError as exc:
                    last_error = exc
                    if attempt >= lag_retry_max:
                        break
                    logger.warning(
                        "Barsmith run missing latest bar; retrying",
                        attempt=attempt + 1,
                        max_attempts=lag_retry_max,
                        sleep_seconds=lag_retry_seconds,
                        error=str(exc),
                    )
                    time.sleep(lag_retry_seconds)

            if not succeeded:
                logger.error("Barsmith scheduled run failed; waiting for next boundary", error=str(last_error))

            if runs_remaining is not None:
                runs_remaining -= 1
    finally:
        if isinstance(broker, brokers.IBKRBroker):
            broker.disconnect()


# =============================================================================
# SCHEDULING COMMANDS
# =============================================================================


@app.command()
def schedule(
    type_: ScheduleType,
    timeframe: types.Timeframe,
    contract: types.SupportedContract,
    n_times: int,
    ib_client_id: int,
    allowed_setups: List[tribar.Variation] = [],
    not_allowed_setups: List[tribar.Variation] = [],
    capital: float = 10000.0,
    account_risk: tribar.AccountRisk = tribar.AccountRisk.ONE_PERCENT,
    max_quantity: int = 1,
    min_quantity: int = 1,
    risk_profile: tribar.RiskProfile = tribar.RiskProfile.ROUND,
    gt: Optional[float] = None,
    gte: Optional[float] = None,
    lt: Optional[float] = None,
    lte: Optional[float] = None,
    gt_target: Optional[float] = None,
    gte_target: Optional[float] = None,
    lt_target: Optional[float] = None,
    lte_target: Optional[float] = None,
    long_gt: Optional[float] = None,
    long_gte: Optional[float] = None,
    short_gt: Optional[float] = None,
    short_gte: Optional[float] = None,
    target_atr_mul: Optional[float] = None,
    with_crossing_sma: bool = False,
    follow_200sma: bool = False,
    execute_now: bool = False,
    last_bar_as_signal: bool = False,
):
    """Schedule automated trading jobs."""
    job_done = asyncio.Event()

    def precheck():
        """Run precheck validation."""
        ib_async.util.startLoop()

        param = create_trading_param(
            contract=contract,
            timeframe=timeframe,
            allowed_setups=allowed_setups,
            not_allowed_setups=not_allowed_setups,
            capital=capital,
            account_risk=account_risk,
            max_quantity=max_quantity,
            min_quantity=min_quantity,
            risk_profile=risk_profile,
            gt=gt,
            gte=gte,
            lt=lt,
            lte=lte,
            gt_target=gt_target,
            gte_target=gte_target,
            lt_target=lt_target,
            lte_target=lte_target,
            long_gt=long_gt,
            long_gte=long_gte,
            short_gt=short_gt,
            short_gte=short_gte,
            target_atr_mul=target_atr_mul,
            with_crossing_sma=with_crossing_sma,
            follow_200sma=follow_200sma,
        )

        run_trading_task(
            type_=type_.value,
            param=param,
            ib_client_id=ib_client_id,
            precheck_only=True,
            check_tv=True,
        )

    def job(target_candle_dt: types.datetime, last_bar_as_signal: bool):
        """Execute trading job."""
        start = time.time()
        logger.debug("Running job")

        param = create_trading_param(
            contract=contract,
            timeframe=timeframe,
            allowed_setups=allowed_setups,
            not_allowed_setups=not_allowed_setups,
            capital=capital,
            account_risk=account_risk,
            max_quantity=max_quantity,
            min_quantity=min_quantity,
            risk_profile=risk_profile,
            gt=gt,
            gte=gte,
            lt=lt,
            lte=lte,
            gt_target=gt_target,
            gte_target=gte_target,
            lt_target=lt_target,
            lte_target=lte_target,
            long_gt=long_gt,
            long_gte=long_gte,
            short_gt=short_gt,
            short_gte=short_gte,
            target_atr_mul=target_atr_mul,
            with_crossing_sma=with_crossing_sma,
            target_bar_dt=target_candle_dt,
            last_bar_as_signal=last_bar_as_signal,
        )

        run_trading_task(
            type_=type_.value,
            param=param,
            ib_client_id=ib_client_id,
        )

        logger.debug("Job completed", time=time.time() - start)

    async def loop():
        """Main scheduling loop."""
        now = utils.get_utc()
        scheduler.add_job(precheck)

        for _ in range(n_times):
            next_run = calculate_next_run_time(timeframe, contract, now)
            now = next_run

            kwargs = {"target_candle_dt": next_run, "last_bar_as_signal": False}
            if timeframe == types.Timeframe.daily:
                next_run -= timedelta(minutes=30)
                if last_bar_as_signal is False:
                    logger.warning(
                        "last_bar_as signal is set to False, however it's set to daily timeframe. Forcing last_bar_as_signal=True"
                    )

                kwargs["last_bar_as_signal"] = True

            trigger = DateTrigger(run_date=next_run)
            scheduler.add_job(job, trigger, kwargs=kwargs)

            tz = pytz.timezone(settings.user_tz)
            time_readable = next_run.astimezone(tz).strftime("%b %d, %Y, %-I:%M %p %z")
            logger.info("Added job", next_run=time_readable)

        scheduler.start()
        logger.info("Scheduler started")
        await job_done.wait()

    if execute_now:
        logger.info("Executing instantly")
        job(target_candle_dt=None, last_bar_as_signal=last_bar_as_signal)
        return

    try:
        asyncio.run(loop())
    except Exception as e:
        logger.error("Scheduling failed", error=str(e))
        raise typer.Exit(1)


@app.command()
def dry_run_async(use_broker: bool = False):
    """Run dry run asynchronously with scheduler."""
    job_done = asyncio.Event()

    def job():
        """Execute dry run job."""
        ib_async.util.startLoop()
        start = time.time()
        tribar.dry_run(use_broker=use_broker)
        logger.debug("Job completed", time=time.time() - start)

    async def loop():
        """Main async loop."""
        now = utils.get_utc() + timedelta(seconds=5)
        trigger = DateTrigger(run_date=now)
        scheduler.add_job(job, trigger)

        scheduler.start()
        logger.info("Scheduler started")
        await job_done.wait()

    try:
        asyncio.run(loop())
    except Exception as e:
        logger.error("Async dry run failed", error=str(e))
        raise typer.Exit(1)


# =============================================================================
# MONITORING COMMANDS
# =============================================================================


@app.command()
def schedule_monitor(type_: ScheduleType, ib_client_id: int, monitor_type: MonitorType):
    """Schedule monitoring jobs (end of week, stoploss mover)."""
    job_done = asyncio.Event()

    # Test broker connection
    try:
        broker = create_broker(type_, ib_client_id)
        logger.debug(
            "Successfully connected to IBKR",
            port=4001 if type_ == ScheduleType.live else 4002,
            account=settings.ib_account,
            ib_client_id=ib_client_id,
            type_=type_.value,
        )
        broker.disconnect()
    except Exception as e:
        logger.error("Failed to connect to broker", error=str(e))
        raise typer.Exit(1)

    def job_end_of_week():
        """Close all positions at end of week."""
        ib_async.util.startLoop()
        try:
            broker = create_broker(type_, ib_client_id)
            broker.close_all_positions()
            broker.disconnect()
            job_done.set()
        except Exception as e:
            logger.error("End of week job failed", error=str(e))

    def job_rr_monitor():
        """Monitor and move stop losses."""
        ib_async.util.startLoop()
        try:
            broker = create_broker(type_, ib_client_id)
            tribar.move_stoploss_to_05_rr(broker)
            broker.disconnect()
        except Exception as e:
            logger.error("RR monitor job failed", error=str(e))

    async def loop():
        """Main monitoring loop."""
        if monitor_type == MonitorType.end_of_week:
            next_friday = utils.next_friday_20utc()
            trigger = DateTrigger(run_date=next_friday)
            scheduler.add_job(job_end_of_week, trigger)

            tz = pytz.timezone(settings.user_tz)
            time_readable = next_friday.astimezone(tz).strftime("%b %d, %Y, %-I:%M %p %z")
            logger.info("Added end_of_week job", next_run=time_readable)

        elif monitor_type == MonitorType.stoploss_mover:
            next_run = utils.next_4hour(skip_sunday=True)
            logger.info("Starting stoploss mover", next_run=next_run)

            while next_run.weekday() != 6:
                # Avoid clashing with every 4 hour - add 5 minute offset
                _next_run = next_run + timedelta(minutes=5)
                trigger = DateTrigger(run_date=_next_run)
                scheduler.add_job(job_rr_monitor, trigger)

                tz = pytz.timezone(settings.user_tz)
                time_readable = _next_run.astimezone(tz).strftime("%b %d, %Y, %-I:%M %p %z")
                logger.info("Added RR monitor job", next_run=time_readable)

                next_run = utils.next_4hour(next_run, skip_sunday=False)

        scheduler.start()
        logger.info("Monitor scheduler started")
        await job_done.wait()

    try:
        asyncio.run(loop())
    except Exception as e:
        logger.error("Monitor scheduling failed", error=str(e))
        raise typer.Exit(1)


# =============================================================================
# BROKER MANAGEMENT COMMANDS
# =============================================================================


@app.command()
def stoploss_mover_dry_run(type_: ScheduleType, ib_client_id: int = 12):
    """Test stoploss mover functionality."""
    try:
        broker = create_broker(type_, ib_client_id)
        tribar.move_stoploss_to_05_rr(broker)
        broker.disconnect()
        logger.info("Stoploss mover dry run completed successfully")
    except Exception as e:
        logger.error("Stoploss mover dry run failed", error=str(e))
        raise typer.Exit(1)


@app.command()
def flatten_positions(type_: ScheduleType, ib_client_id: int):
    """Close all open positions."""
    try:
        broker = create_broker(type_, ib_client_id)
        broker.close_all_positions()
        broker.disconnect()
        logger.info("All positions have been closed")
    except Exception as e:
        logger.error("Failed to close positions", error=str(e))
        raise typer.Exit(1)


@app.command()
def cancel_all_orders(type_: ScheduleType, ib_client_id: int):
    """Cancel all open orders."""
    try:
        broker = create_broker(type_, ib_client_id)

        all_orders = broker.ib.openOrders()
        if not all_orders:
            logger.info("No orders to cancel")
        else:
            broker.cancel_orders(all_orders)
            logger.info(f"Cancelled {len(all_orders)} orders")

        broker.disconnect()
    except Exception as e:
        logger.error("Failed to cancel orders", error=str(e))
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

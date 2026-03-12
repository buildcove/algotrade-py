from __future__ import annotations

import time
from datetime import datetime

import ib_async
from algotrade.contracts import get_contract_spec
from algotrade.core import types, utils
from algotrade.models import tribar_v2 as tribar
from algotrade.scheduler import scheduler
from apscheduler.triggers.date import DateTrigger
from fastapi import APIRouter, HTTPException
from structlog import get_logger

from . import schema

logger = get_logger()

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


def _next_run_at(timeframe: types.Timeframe, contract: types.SupportedContract, now: datetime) -> datetime:
    if timeframe == types.Timeframe.m30:
        return utils.next_half_hour_for_contract(contract=contract, now=now)
    if timeframe == types.Timeframe.h4:
        return utils.next_4hour_for_contract(contract=contract, now=now, skip_sunday=True)
    if timeframe == types.Timeframe.daily:
        return utils.next_daily_for_contract(contract=contract, now=now)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


@router.get("/list")
def list_jobs():
    jobs = scheduler.get_jobs()
    return {"jobs": [{"id": job.id, "next_run_time": job.next_run_time} for job in jobs]}


@router.post("/add")
def add_job(data: schema.AddJobRequest):
    if data.n_times <= 0:
        raise HTTPException(status_code=400, detail="n_times must be >= 1")

    now = utils.get_utc()
    first_run_at = data.run_at_utc or _next_run_at(timeframe=data.timeframe, contract=data.contract, now=now)
    if data.execute_now:
        first_run_at = now

    contract_spec = get_contract_spec(data.contract)

    def _job(target_candle_dt: datetime | None) -> None:
        start = time.time()
        ib_async.util.startLoop()

        param = tribar.Param(
            contract=contract_spec,
            tz="UTC",
            timeframe=data.timeframe,
            allowed_setups=data.allowed_setups or [],
            not_allowed_setups=data.not_allowed_setups or [],
            target_bar_dt=target_candle_dt,
            last_bar_as_signal=(data.timeframe == types.Timeframe.daily),
        )

        task = tribar.Live(type_=data.type_, param=param, ib_client_id=data.ib_client_id)
        task.run_precheck()
        task.run()
        task.broker.disconnect()
        logger.info(
            "scheduler_job_done", elapsed=time.time() - start, contract=data.contract.value, timeframe=data.timeframe.value
        )

    job_ids: list[str] = []
    run_at = first_run_at
    for _ in range(data.n_times):
        trigger = DateTrigger(run_date=run_at)
        job = scheduler.add_job(_job, trigger=trigger, kwargs={"target_candle_dt": run_at}, replace_existing=False)
        job_ids.append(job.id)
        run_at = _next_run_at(timeframe=data.timeframe, contract=data.contract, now=run_at)

    return {"job_ids": job_ids, "first_run_at_utc": first_run_at}

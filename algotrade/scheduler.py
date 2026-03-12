from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = AsyncIOScheduler()
scheduler_blocking = BlockingScheduler()

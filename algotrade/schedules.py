from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Dict, List

import tvDatafeed

from .core import types


class Exchange(Enum):
    CME = "CME"
    CBOT = "CBOT"


class TimezoneDST(Enum):
    EST = "EST"
    EDT = "EDT"
    CST = "CST"
    CDT = "CDT"


@dataclass(frozen=True)
class ExchangeSchedule:
    timezone: str
    open_times_utc: Dict[TimezoneDST, time]
    four_hour_times_utc: Dict[TimezoneDST, List[time]]


CME_TZ = "US/Eastern"
CBOT_TZ = "US/Central"
IB_TZ = "UTC"

CME_SCHEDULE = ExchangeSchedule(
    timezone=CME_TZ,
    open_times_utc={
        TimezoneDST.EST: time(23, 0),
        TimezoneDST.EDT: time(22, 0),
    },
    four_hour_times_utc={
        TimezoneDST.EST: [
            time(3, 0),
            time(7, 0),
            time(11, 0),
            time(15, 0),
            time(19, 0),
            time(23, 0),
        ],
        TimezoneDST.EDT: [
            time(2, 0),
            time(6, 0),
            time(10, 0),
            time(14, 0),
            time(18, 0),
            time(22, 0),
        ],
    },
)

CBOT_SCHEDULE = ExchangeSchedule(
    timezone=CBOT_TZ,
    open_times_utc={
        TimezoneDST.CST: time(1, 0),
        TimezoneDST.CDT: time(0, 0),
    },
    four_hour_times_utc={
        TimezoneDST.CST: [
            time(1, 0),
            time(5, 0),
            time(9, 0),
            time(13, 0),
            time(17, 0),
            time(21, 0),
        ],
        TimezoneDST.CDT: [
            time(0, 0),
            time(4, 0),
            time(8, 0),
            time(12, 0),
            time(16, 0),
            time(20, 0),
        ],
    },
)

EXCHANGE_SCHEDULES = {
    Exchange.CME: CME_SCHEDULE,
    Exchange.CBOT: CBOT_SCHEDULE,
}

TIMEFRAME_TO_TV_INTERVAL = {
    types.Timeframe.m30: tvDatafeed.Interval.in_30_minute,
    types.Timeframe.h4: tvDatafeed.Interval.in_4_hour,
    types.Timeframe.daily: tvDatafeed.Interval.in_daily,
}

CONTRACT_TO_EXCHANGE = {
    types.SupportedContract.MGC: Exchange.CME,
    types.SupportedContract.MHG: Exchange.CME,
    types.SupportedContract.PL: Exchange.CME,
    types.SupportedContract.MES: Exchange.CME,
    types.SupportedContract.MCL: Exchange.CME,
    types.SupportedContract.CL: Exchange.CME,
    types.SupportedContract.MNG: Exchange.CME,
    types.SupportedContract.NG: Exchange.CME,
    types.SupportedContract.M6E: Exchange.CME,
    types.SupportedContract.JPY6J: Exchange.CME,
    types.SupportedContract.GBP6B: Exchange.CME,
    types.SupportedContract.ZM: Exchange.CBOT,
}


def get_schedule_for_contract(contract: types.SupportedContract) -> ExchangeSchedule:
    exchange = CONTRACT_TO_EXCHANGE[contract]
    return EXCHANGE_SCHEDULES[exchange]

"""
Backward compatibility module for constants.
This module re-exports items from the refactored modules for backward compatibility.
"""

from .contracts import CONTRACT_SPECIFICATIONS as contract_spec
from .schedules import CBOT_SCHEDULE
from .schedules import CBOT_TZ as cbot_tz
from .schedules import CME_SCHEDULE
from .schedules import CME_TZ as cme_tz
from .schedules import CONTRACT_TO_EXCHANGE
from .schedules import IB_TZ as ib_tz
from .schedules import TIMEFRAME_TO_TV_INTERVAL as timeframe_to_tv_interval
from .schedules import TimezoneDST
from .treasury import DECIMAL_TO_TREASURY_PART as decimal_to_treasury_part
from .treasury import TREASURY_PART_TO_DECIMAL as treasury_part_to_decimal

contract_to_exchange_schedule = {contract: exchange.value for contract, exchange in CONTRACT_TO_EXCHANGE.items()}

cme_globex_open_to_UTC = {
    "EST": CME_SCHEDULE.open_times_utc[TimezoneDST.EST],
    "EDT": CME_SCHEDULE.open_times_utc[TimezoneDST.EDT],
}

cme_globex_utc_4h = {
    "EST": CME_SCHEDULE.four_hour_times_utc[TimezoneDST.EST],
    "EDT": CME_SCHEDULE.four_hour_times_utc[TimezoneDST.EDT],
}

cbot_open_to_UTC = {
    "CST": CBOT_SCHEDULE.open_times_utc[TimezoneDST.CST],
    "CDT": CBOT_SCHEDULE.open_times_utc[TimezoneDST.CDT],
}

cbot_utc_4h = {
    "CST": CBOT_SCHEDULE.four_hour_times_utc[TimezoneDST.CST],
    "CDT": CBOT_SCHEDULE.four_hour_times_utc[TimezoneDST.CDT],
}

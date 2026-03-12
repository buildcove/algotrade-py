from datetime import datetime
from typing import List, Literal, Optional

from algotrade.core import types
from algotrade.models import tribar_v2 as tribar
from pydantic import BaseModel, Field


class AddJobRequest(BaseModel):
    type_: Literal["live", "paper"]
    timeframe: types.Timeframe
    contract: types.SupportedContract
    allowed_setups: Optional[List[tribar.Variation]] = Field(default_factory=list)
    not_allowed_setups: Optional[List[tribar.Variation]] = Field(default_factory=list)
    n_times: int = Field(default=1, ge=1)
    ib_client_id: int = Field(default=1, ge=1)
    execute_now: bool = Field(default=False)
    run_at_utc: Optional[datetime] = None

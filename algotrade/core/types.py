from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import ib_async


class SupportedContract(Enum):
    MGC = "mgc"
    MHG = "mhg"
    PL = "pl"

    MES = "mes"
    MCL = "mcl"
    CL = "cl"
    MNG = "mng"
    NG = "ng"
    M6E = "m6e"
    JPY6J = "6j"
    GBP6B = "6b"
    ZM = "zm"


@dataclass
class ContractSpec:
    ib: ib_async.Future
    symbol: str
    exchange: str
    tick_size: float
    tick_value: float
    tick_spread: int


@dataclass
class Stream:
    bid: float
    ask: float
    timestamp: datetime


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime


class Timeframe(Enum):
    h4 = "h4"
    m30 = "m30"
    daily = "daily"


@dataclass
class Action:
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(kw_only=True)
class BaseOrder:
    symbol: object | str
    quantity: int
    action: Action
    timestamp: datetime | None = None
    # extra: object | None = None
    # params: object | None = None
    # var: object | None = None


@dataclass
class BracketOrder:
    symbol: object | str
    order: BaseOrder
    take_profit: BaseOrder | None
    stop_loss: BaseOrder | None
    timestamp: datetime | None = None
    extra: object | None = None
    params: object | None = None
    var: object | None = None


@dataclass(kw_only=True)
class LimitOrder(BaseOrder):
    symbol: object | str
    quantity: int
    action: Action
    price: float
    timestamp: datetime | None = None


@dataclass(kw_only=True)
class MarketOrder(BaseOrder):
    pass


@dataclass(kw_only=True)
class StopOrder(BaseOrder):
    price: float


@dataclass
class Position:
    symbol: object | str
    quantity: int
    action: Action
    entry_price: float
    entry_timestamp: datetime


@dataclass
class ClosedPosition(Position):
    exit_price: float
    exit_timestamp: datetime
    gain_ticks: int
    # extra: object | None  # repeat since I want it to be at the end when printing.
    # params: object | None  # repeat since I want it to be at the end when printing.


@dataclass
class Trade:
    symbol: object | str
    entry: Position
    exit: ClosedPosition
    target: LimitOrder
    stop_loss: StopOrder

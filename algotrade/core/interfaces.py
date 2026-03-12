from __future__ import annotations

from typing import Protocol, runtime_checkable

from . import types


@runtime_checkable
class Broker(Protocol):
    def order(self, order: types.BaseOrder): ...

    def get_orders(self, contract): ...  # noqa: ANN001 - depends on broker implementation

    def cancel_orders(self, orders): ...  # noqa: ANN001 - depends on broker implementation

    def get_positions(self, contract=None): ...  # noqa: ANN001


@runtime_checkable
class Clock(Protocol):
    def now_utc(self): ...  # noqa: ANN001 - datetime

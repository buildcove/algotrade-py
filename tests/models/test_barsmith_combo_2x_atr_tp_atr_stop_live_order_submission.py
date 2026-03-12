from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("ib_async")

from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, repository, types
from algotrade.models.barsmith_combo_2x_atr_tp_atr_stop import (
    BarsmithComboExecutor,
    Candle,
    ExecutionMode,
    Extra,
    LiveBracketSnapshot,
    Param,
    Var,
)


@dataclass
class _DummyIb:
    def openOrders(self):  # noqa: N802 - match ib_async naming
        return []


@dataclass
class _DummyPosition:
    contract: object
    position: int


class FakeIBKRBroker(brokers.IBKRBroker):
    """
    Test stub: behaves like an IBKRBroker for isinstance checks, but does not connect anywhere.
    """

    def __init__(self, repo: repository.Repository):
        brokers.BaseBroker.__init__(self, repo)
        self.ib = _DummyIb()
        self.bracket_calls: list[types.BracketOrder] = []
        self.open_orders: list[object] = []
        self.positions: list[object] = []
        self.recover_calls: list[dict] = []
        self.cancel_calls: list[list[object]] = []

    def ensure_connected(self) -> None:  # noqa: D401 - test stub
        return

    def get_positions(self, contract=None):  # noqa: ANN001
        return list(self.positions)

    def get_orders(self, contract=None):  # noqa: ANN001
        return list(self.open_orders)

    def place_oca_exit_orders(  # noqa: D401, ANN001
        self,
        contract,
        quantity,
        take_profit_price,
        stop_loss_price,
        action,
        oca_group,
    ):
        self.recover_calls.append(
            {
                "contract": contract,
                "quantity": quantity,
                "take_profit_price": take_profit_price,
                "stop_loss_price": stop_loss_price,
                "action": action,
                "oca_group": oca_group,
            }
        )
        return []

    def cancel_orders(self, orders):  # noqa: ANN001
        self.cancel_calls.append(list(orders))

    def _limit_order(self, order: types.LimitOrder):  # pragma: no cover
        raise NotImplementedError

    def _market_order(self, order: types.MarketOrder):  # pragma: no cover
        raise NotImplementedError

    def _stop_order(self, order: types.StopOrder):  # pragma: no cover
        raise NotImplementedError

    def _bracket_order(self, order: types.BracketOrder):
        self.bracket_calls.append(order)


def _signal_candle() -> Candle:
    return Candle(
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        timestamp=datetime(2025, 1, 3, 15, 0, 0),
        atr=1.0,
        is_tribar_green=True,
        is_kf_breakout_potential=True,
        kf_innovation_abs=6.0,
        macd_hist=-1.0,
        trend_strength=0.25,
    )


def test_live_mode_submits_bracket_once(tmp_path: Path) -> None:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = FakeIBKRBroker(repo)
    param = Param(contract=contract_spec, supported_contract=types.SupportedContract.MES)

    state_path = tmp_path / "state.json"
    exe = BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=Var(),
        extra=Extra(),
        mode=ExecutionMode.LIVE,
        require_market_open=False,
        state_path=str(state_path),
    )

    candle = _signal_candle()
    exe.next(candle, historical=True)
    assert len(broker.bracket_calls) == 1

    exe.next(candle, historical=True)
    assert len(broker.bracket_calls) == 1

    assert state_path.exists()


def test_live_mode_skips_when_open_orders_exist(tmp_path: Path) -> None:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = FakeIBKRBroker(repo)
    broker.open_orders = [object()]
    param = Param(contract=contract_spec, supported_contract=types.SupportedContract.MES)

    exe = BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=Var(),
        extra=Extra(),
        mode=ExecutionMode.LIVE,
        require_market_open=False,
        state_path=str(tmp_path / "state.json"),
    )

    exe.next(_signal_candle(), historical=True)
    assert len(broker.bracket_calls) == 0


def test_live_mode_enforces_max_trades_per_day(tmp_path: Path) -> None:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = FakeIBKRBroker(repo)
    param = Param(contract=contract_spec, supported_contract=types.SupportedContract.MES)

    exe = BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=Var(),
        extra=Extra(),
        mode=ExecutionMode.LIVE,
        require_market_open=False,
        max_trades_per_day=1,
        state_path=str(tmp_path / "state.json"),
    )

    first = _signal_candle()
    second = Candle(**{**first.__dict__, "timestamp": datetime(2025, 1, 3, 15, 30, 0)})  # type: ignore[arg-type]
    exe.next(first, historical=True)
    exe.next(second, historical=True)
    assert len(broker.bracket_calls) == 1


def test_live_mode_recovers_missing_protection_when_position_open(tmp_path: Path) -> None:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = FakeIBKRBroker(repo)
    param = Param(contract=contract_spec, supported_contract=types.SupportedContract.MES)

    exe = BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=Var(),
        extra=Extra(),
        mode=ExecutionMode.LIVE,
        require_market_open=False,
        state_path=str(tmp_path / "state.json"),
    )

    exe.last_bracket_snapshot = LiveBracketSnapshot(
        signal_bar_ts_utc="2025-01-03T15:00:00+00:00",
        submitted_at_utc="2025-01-03T15:00:01+00:00",
        entry=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        quantity=1,
        ib_order_ids=[],
    )
    broker.positions = [_DummyPosition(contract=contract_spec.ib, position=1)]

    exe.next(_signal_candle(), historical=True)
    assert len(broker.recover_calls) == 1


def test_live_reconcile_cancels_stale_open_orders(tmp_path: Path) -> None:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = FakeIBKRBroker(repo)
    broker.open_orders = [object(), object()]
    param = Param(contract=contract_spec, supported_contract=types.SupportedContract.MES)

    exe = BarsmithComboExecutor(
        broker=broker,  # type: ignore[arg-type]
        param=param,
        var=Var(),
        extra=Extra(),
        mode=ExecutionMode.LIVE,
        require_market_open=False,
        entry_timeout_seconds=1.0,
        state_path=str(tmp_path / "state.json"),
    )
    exe.last_bracket_snapshot = LiveBracketSnapshot(
        signal_bar_ts_utc="2025-01-03T15:00:00+00:00",
        submitted_at_utc="2025-01-03T15:00:00+00:00",
        entry=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        quantity=1,
        ib_order_ids=[],
    )

    exe._reconcile_live_orders(run_id="test")  # noqa: SLF001
    assert len(broker.cancel_calls) == 1

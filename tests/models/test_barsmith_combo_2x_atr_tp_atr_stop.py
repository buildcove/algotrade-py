from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("ib_async")

from algotrade.contracts import get_contract_spec
from algotrade.core import brokers, repository, types

try:
    from algotrade.models.barsmith_combo_2x_atr_tp_atr_stop import (
        BarsmithComboExecutor,
        Candle,
        Extra,
        Param,
        Var,
    )
except Exception as exc:
    BarsmithComboExecutor = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


pytestmark = pytest.mark.skipif(BarsmithComboExecutor is None, reason=f"barsmith_combo import failed: {_IMPORT_ERROR}")


def _make_executor() -> BarsmithComboExecutor:
    contract_spec = get_contract_spec(types.SupportedContract.MES)
    repo = repository.Repository()
    broker = brokers.BacktestingBroker(repo, tick_size=contract_spec.tick_size)
    param = Param(contract=contract_spec, supported_contract=types.SupportedContract.MES)
    return BarsmithComboExecutor(broker=broker, param=param, var=Var(), extra=Extra())


def _signal_bar(**overrides: float | bool) -> Candle:
    base = dict(
        open=99.0,
        high=101.0,
        low=98.0,
        close=100.0,
        timestamp=datetime(2024, 1, 1, 0, 0, 0),
        atr=1.0,
        is_tribar_green=True,
        is_kf_breakout_potential=True,
        kf_innovation_abs=6.0,
        macd_hist=-1.0,
        trend_strength=0.25,
    )
    base.update(overrides)
    return Candle(**base)  # type: ignore[arg-type]


def test_signal_opens_trade_when_combo_true() -> None:
    exe = _make_executor()

    exe.next(_signal_bar(), historical=True)
    assert exe.var.active_trade is not None
    assert exe.var.active_trade.entry == 100.0
    assert exe.var.active_trade.stop == 99.0
    assert exe.var.active_trade.tp == 102.0


def test_signal_does_not_open_trade_when_combo_false() -> None:
    exe = _make_executor()

    exe.next(_signal_bar(is_tribar_green=False), historical=True)
    assert exe.var.active_trade is None

    exe.next(_signal_bar(kf_innovation_abs=5.0), historical=True)
    assert exe.var.active_trade is None

    exe.next(_signal_bar(macd_hist=-1.25), historical=True)
    assert exe.var.active_trade is None

    exe.next(_signal_bar(trend_strength=0.26), historical=True)
    assert exe.var.active_trade is None


def test_two_x_atr_tp_atr_stop_gap_stop_fill() -> None:
    exe = _make_executor()

    exe.next(_signal_bar(), historical=True)
    assert exe.var.active_trade is not None

    bar = Candle(
        open=98.0,
        high=99.0,
        low=97.0,
        close=98.5,
        timestamp=datetime(2024, 1, 1, 0, 30, 0),
        atr=1.0,
    )
    exe.next(bar, historical=True)
    assert exe.var.active_trade is None
    trade = exe.var.setups[0]
    assert trade["gap_fill"] is True
    assert trade["hit_tp"] is False
    assert trade["rr"] == pytest.approx(-2.0)


def test_two_x_atr_tp_atr_stop_intrabar_tp_hit() -> None:
    exe = _make_executor()

    exe.next(_signal_bar(), historical=True)
    assert exe.var.active_trade is not None

    bar = Candle(
        open=100.5,
        high=102.0,
        low=100.0,
        close=101.0,
        timestamp=datetime(2024, 1, 1, 0, 30, 0),
        atr=1.0,
    )
    exe.next(bar, historical=True)
    assert exe.var.active_trade is None
    trade = exe.var.setups[0]
    assert trade["gap_fill"] is False
    assert trade["hit_tp"] is True
    assert trade["rr"] == pytest.approx(2.0)

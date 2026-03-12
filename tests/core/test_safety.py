import pytest
from algotrade.core.safety import (
    LIVE_CONFIRM_PHRASE,
    LiveModeNotAllowed,
    require_live_trading_enabled,
)


def test_require_live_trading_enabled_rejects_when_is_live_false() -> None:
    with pytest.raises(LiveModeNotAllowed, match="Live trading is disabled"):
        require_live_trading_enabled(is_live=False, live_confirm=LIVE_CONFIRM_PHRASE, operation="unit-test")


def test_require_live_trading_enabled_rejects_when_confirm_missing() -> None:
    with pytest.raises(LiveModeNotAllowed, match="confirmation missing"):
        require_live_trading_enabled(is_live=True, live_confirm=None, operation="unit-test")


def test_require_live_trading_enabled_accepts_when_enabled_and_confirmed() -> None:
    require_live_trading_enabled(is_live=True, live_confirm=LIVE_CONFIRM_PHRASE, operation="unit-test")

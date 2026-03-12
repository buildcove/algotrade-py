from __future__ import annotations

from enum import Enum

LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_LIVE_TRADING"


class RunMode(str, Enum):
    DRY = "dry"
    PAPER = "paper"
    LIVE = "live"


class LiveModeNotAllowed(RuntimeError):
    pass


def require_live_trading_enabled(*, is_live: bool, live_confirm: str | None, operation: str) -> None:
    """
    Raise if the caller attempts a live-trading action without explicit opt-in.

    This intentionally duplicates the guardrail in every entrypoint that might
    place orders: CLI, API scheduler jobs, and any direct library calls.
    """
    if not is_live:
        raise LiveModeNotAllowed(
            f"Live trading is disabled (operation={operation}). "
            "Set `is_live=true` in your environment if you intend to place live orders."
        )

    if live_confirm != LIVE_CONFIRM_PHRASE:
        raise LiveModeNotAllowed(
            f"Live trading confirmation missing (operation={operation}). "
            f"Set `live_confirm={LIVE_CONFIRM_PHRASE}` to acknowledge risk."
        )

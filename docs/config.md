# Configuration

Configuration is loaded from environment variables (typically via `.env`). Start from `.env.example`.

## Core

- `user_tz` (default: `Asia/Manila`): display timezone for CLI logs.

## TradingView

- `tradingview_sessionid`: cookie token used by `tvdatafeed` in this repo.
- `tradingview_username`, `tradingview_password`: optional alternative login fields (depends on provider support).

## IBKR

- `ib_account`: account identifier used for routing/position queries.

## Live trading safety gate

Live-trading actions require both:
- `is_live=true`
- `live_confirm=I_UNDERSTAND_LIVE_TRADING`

This repo intentionally defaults to safe behavior to avoid accidental live trading.

## API auth (optional)

If both values are set, the API requires HTTP Basic auth on protected routes:
- `basic_username`
- `basic_password`


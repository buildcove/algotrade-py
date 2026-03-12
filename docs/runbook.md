# Runbook

This runbook is intentionally conservative. Treat it as a safety checklist.

## Modes

- `dry`: no broker connection (recommended for development).
- `paper`: connects to IBKR paper (TWS/Gateway on paper port).
- `live`: connects to IBKR live (requires explicit opt-in).

## Live safety checklist

Before any live run:
- Ensure you understand trading risk and broker failure modes.
- Set the safety gate:
  - `is_live=true`
  - `live_confirm=I_UNDERSTAND_LIVE_TRADING`
- Verify your IBKR session (TWS/Gateway) is running and stable.
- Start with a small account / small size and validate behavior in paper first.

## API server

- Run: `uv run algotrade api --reload --port 8000`
- Health:
  - `GET /healthz` should return `{"status":"ok"}`
  - `GET /readyz` returns scheduler status

## Scheduler API

Scheduler endpoints live under `/scheduler/*` and run strategy jobs on a timer.
If `basic_username`/`basic_password` are set, scheduler routes require HTTP Basic auth.

## Common issues

- TradingView lag: the latest bar may not be available exactly at the expected boundary.
- IBKR disconnects: network/TWS restarts can cause mid-run failures; use paper mode first.


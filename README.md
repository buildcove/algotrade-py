# algotrade (personal project)

Personal automated futures trading system (CLI + FastAPI) with broker abstractions, strategy runners, and backtesting utilities.

WARNING: This repository is for research use and is not production-hardened for unattended trading. Use at your own risk.

## Quickstart (local)

Prereqs: Python 3.12 + `uv`.

- Install: `uv sync --extra dev --extra test`
- CLI help: `uv run algotrade --help` (legacy: `uv run python main.py --help`)
- Run API: `uv run algotrade api --reload --port 8000` (legacy: `uv run python api.py`)
- Tests: `uv run pytest -q`

## Configuration

Copy `.env.example` to `.env` and fill in values as needed.

- TradingView: set `tradingview_sessionid` (and/or username/password if supported by your setup)
- IBKR: set `ib_account`
- API Basic auth: set `basic_username` and `basic_password` (optional; when unset the API auth is disabled)
- Live trading safety gate: set `is_live=true` and `live_confirm=I_UNDERSTAND_LIVE_TRADING`

## Notes

- Trading involves substantial risk. This project is for research/personal use.
- Do not commit real credentials; use `.env` locally (see `.env.example`).

## Docs

- `docs/quickstart.md`
- `docs/config.md`
- `docs/runbook.md`
- `docs/architecture.md`
- `docs/security.md`

## Open source

- License: MIT (see `LICENSE`)
- Security policy: `SECURITY.md`
- Contributing: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`

# Quickstart (local)

Prereqs:
- Python 3.12
- `uv` (recommended)

## Install

- `uv sync --extra dev --extra test`

## Run tests (offline)

- `uv run pytest -q`

## CLI

- Help: `uv run algotrade --help`
- Run API server (FastAPI): `uv run algotrade api --reload --port 8000`

## Notes

- This is a personal research project. It is not production-hardened for unattended trading.
- Live trading is guarded behind explicit opt-in; see `docs/runbook.md`.


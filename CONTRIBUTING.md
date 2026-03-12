# Contributing

Thanks for your interest in contributing.

## Development setup

Prerequisites:

- Python 3.12
- `uv` (recommended)

Install dependencies:

```bash
uv sync --extra dev --extra test
```

## Code quality

```bash
uv run black .
uv run isort .
uv run flake8
```

## Tests

```bash
uv run pytest -q
```

Notes:

- Tests must run without broker connectivity and without TradingView credentials.
- If you add integration tests that require network access, mark them and skip by default in CI.

## Security

Do not include credentials, tokens, cookies, or account identifiers in commits, logs, or test fixtures.


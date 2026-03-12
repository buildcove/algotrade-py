# Security

## Secrets

- Do not commit credentials (TradingView cookies, broker account identifiers, passwords).
- Use `.env` locally and keep `.env.example` with placeholders only.

## Reporting vulnerabilities

See `SECURITY.md`.

## Dependency scanning

This repo includes an optional GitHub Actions workflow to scan dependencies via `pip-audit`.
Handle advisories by:
- upgrading the vulnerable dependency, or
- documenting and suppressing the advisory with justification (only when unavoidable).


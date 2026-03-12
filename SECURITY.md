# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability, please report it privately.

- Preferred: open a private GitHub Security Advisory for this repository.
- Alternative: open an issue with minimal details and ask maintainers to move the discussion to a private channel.

Please include:
- A clear description of the issue and its impact.
- Steps to reproduce (or a proof-of-concept).
- Any suggested mitigation.

## Supported versions

Only the `main` branch is currently supported.

## Dependency vulnerabilities

CI can run dependency audits using `pip-audit` (optional).

Local run:

```bash
uv run python -m pip install --upgrade pip-audit
uv run pip-audit
```


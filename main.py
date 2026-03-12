"""
Legacy CLI entrypoint.

Prefer the installable CLI: `algotrade --help`.
"""

from algotrade.cli.main import app, compute_barsmith_next_run  # noqa: F401

if __name__ == "__main__":
    app()

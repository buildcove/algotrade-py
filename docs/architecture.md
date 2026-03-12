# Architecture

This repository is split into:

- `algotrade/`: runtime package (library + CLI + API)
- `tools/`: ad-hoc scripts used for parity checks and evaluations (not packaged)
- `experiments/`: exploratory research code (not packaged)

## Runtime package (`algotrade/`)

- `algotrade/core/`: types, brokers, repository layer, indicators, utilities
- `algotrade/models/`: strategy executors (some are research-grade; see README warnings)
- `algotrade/api/`: FastAPI app + scheduler routes
- `algotrade/cli/`: Typer CLI entrypoint (`algotrade`)

## Safety-by-default

Live trading is disabled unless explicitly enabled via environment flags. Broker
connections are explicit (`connect()`), not implicit in constructors.


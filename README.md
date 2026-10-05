# ivcrypto

Implied volatility surfaces for crypto options, built from real Deribit order book snapshots:
implied volatilities, per expiry SVI fits, static arbitrage checks and a Heston calibration, with
an honest account of where each model fails.

> Work in progress. This README becomes a short research note (motivation, methodology, data
> issues, results, limitations, reproduction) as the milestones land.

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Project scaffold, CI | done |
| M1 | Deribit data layer, Parquet snapshots, committed sample | planned |
| M2 | Cleaning: conversion to USD, forwards, log moneyness, filters | planned |
| M3 | Black 76 and implied volatility, validation against Deribit | planned |
| M4 | SVI per expiry | planned |
| M5 | Static arbitrage checks (SSVI as a stretch goal) | planned |
| M6 | Heston pricing, validation and calibration | planned |
| M7 | CLI, plots, research note | planned |

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```bash
uv sync                    # create the environment from uv.lock
uv run pytest              # offline test suite
uv run ruff check .        # lint
uv run ruff format .       # format
```

All tests run offline. A fixture blocks every network connection, and tests that talk to the live
API are marked `network` and only run on request (`uv run pytest -m network`).

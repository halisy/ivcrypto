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
| M1 | Deribit data layer, Parquet snapshots, committed samples | done |
| M2 | Cleaning: conversion to USD, forwards, log moneyness, filters | planned |
| M3 | Black 76 and implied volatility, validation against Deribit | planned |
| M4 | SVI per expiry | planned |
| M5 | Static arbitrage checks (SSVI as a stretch goal) | planned |
| M6 | Heston pricing, validation and calibration | planned |
| M7 | CLI, plots, research note | planned |

## Data

Snapshots come from Deribit's public API v2 (no account needed). One snapshot is the full option
chain of one currency from a single bulk call, plus the futures, the index and, optionally, one
ticker per option:

```bash
uv run ivcrypto fetch --currency BTC              # writes data/raw/BTC/<YYYYMMDDTHHMMSSZ>/
uv run ivcrypto fetch --currency ETH --tickers    # also Deribit's own bid/ask IVs and greeks
```

Each snapshot directory holds one Parquet file per API table, with field names and values exactly
as returned, and a `manifest.json` with the server timestamps of every call. Two real snapshots
are committed so that the tests and the demo run offline:

| Sample | Taken (UTC) | Options | Expiries | Size |
|---|---|---|---|---|
| `data/sample/BTC/20261005T222705Z` | 2026-10-05 22:27:05 | 946 | 12 (9.5 hours to 353 days) | 196 KB |
| `data/sample/ETH/20261005T223117Z` | 2026-10-05 22:31:17 | 790 | 12 (9.5 hours to 353 days) | 172 KB |

[docs/deribit_api.md](docs/deribit_api.md) lists every field the pipeline relies on, the units
and null conventions, and the checks that established Deribit's pricing convention.

## Design decisions

Each decision with a meaningful alternative is recorded here with its tradeoff.

* **One bulk call per snapshot, not one ticker per option.** `get_book_summary_by_currency`
  returns the whole chain computed within 20 ms. A ticker sweep takes about 4 minutes, and quotes
  taken minutes apart create spurious arbitrage. Tickers are kept only to validate our IVs
  against Deribit's.
* **Raw storage.** Snapshots store the API rows unchanged; all interpretation happens in later
  stages, so a change of method never requires refetching data.
* **Rate limits.** The client throttles itself to half of Deribit's documented limits (the per IP
  limit for public requests is unpublished) and retries only what retrying can fix: rate limit
  errors, server errors and network failures.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```bash
uv sync                    # create the environment from uv.lock
uv run pytest              # offline test suite
uv run pytest -m network   # live checks that Deribit's schema has not changed
uv run ruff check .        # lint
uv run ruff format .       # format
```

All tests run offline. A fixture blocks every network connection, and the live tests run only on
request.

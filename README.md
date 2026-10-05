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
| M2 | Cleaning: conversion to USD, forwards, log moneyness, filters | done |
| M3 | Black 76 and implied volatility, validation against Deribit | done |
| M4 | SVI per expiry | done |
| M5 | Static arbitrage checks (SSVI as a stretch goal) | checks done, SSVI next |
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
* **Coin prices to USD with the expiry's forward, not spot.** A Deribit call pays
  (S_T - K)^+ / S_T coins, so its coin price is Black76(F) / F and the USD forward premium is
  the coin price times F. This is Deribit's own convention (it reproduces their bid and ask IVs
  exactly); converting with the spot index instead mixes the futures basis into every premium
  and produced errors of up to 80 vol points.
* **Forward from put call parity, not the future.** In coin terms parity reads
  C - P = 1 - K/F, so every strike quoted on both sides implies F with no model, spot or USD
  rate. We take the eight strikes nearest the money, drop outliers beyond five robust standard
  deviations, and average with weights from each pair's bid ask spreads; an expiry falls back to
  Deribit's underlying price (the future) when fewer than two pairs exist or they scatter by
  more than 25 bps. On the BTC sample the pairs scatter by 0.5 to 2.8 bps, so each forward has
  a standard error of at most 1.05 bps. The future is observable and liquid, but the options
  imply a forward 2.5 to 3.2 bps above it for the 1 to 6 month expiries (3 to 5 standard
  errors, about one futures bid ask spread) and within 1.6 bps elsewhere. It matters at the
  money: with the future's forward, call and put IVs at the same strike differ by up to
  0.16 vol points (median per expiry), with the parity forward by at most 0.11. The zero coin
  rate behind the formula is tested, not assumed: regressing C - P on K gives a coin discount
  factor between 0.9995 and 1.0006 for every BTC expiry.

  An earlier version took the median of the six pairs with the tightest spreads. Validation
  showed that this choice can be lopsided (all six on one side of the money) and that a median
  of six pairs with 10 to 30 bps of spread noise is itself uncertain by several bps, which had
  made the gap to the futures look larger (up to 4.8 bps) than it is.
* **Time to expiry** is ACT/365 calendar time (the market never closes) to 08:00 UTC, measured
  from when Deribit computed the quotes (`creation_timestamp`), which can precede the API
  response by up to a second.
* **Filters.** Every threshold is configurable (`config/default.toml`) and every quote keeps the
  name of the first filter that removed it. On the BTC sample:

  | Filter | Default | Removed | Remaining |
  |---|---|---|---|
  | expiry closer than | 2 days | 120 | 826 |
  | no bid (null or 0) | | 17 | 809 |
  | no ask, crossed | | 0 | 809 |
  | relative spread above | 50% | 23 | 786 |
  | in the money | OTM only | 413 | 373 |

  The two day minimum drops the 0.4 and 1.4 day expiries, where the slices are thin (7 two
  sided OTM quotes for the shortest) and the 30 minute settlement average distorts prices. The
  shortest expiry kept is 2.4 days, so the short end where Heston is expected to struggle stays
  in the sample.
* **Implied volatility by Brent's method on the OTM time value.** Under undiscounted Black 76
  a call and a put at one strike differ by exactly F - K, so every price maps to the time
  value of the OTM option at its strike, which is inverted in normalized units
  (s = sigma sqrt(T), price divided by F). Inverting ITM prices directly is badly conditioned:
  their time value is a small difference of large numbers. Volatility is bracketed in
  [0.01%, 1000%]; prices at or below intrinsic value or at or above the upper bound return NaN
  rather than a forced solution. Bid, mid and ask IVs are inverted separately (the mid IV is
  not the average of the bid and ask IVs). A vectorized Newton solver or Jaeckel's "Let's be
  rational" would be faster, but 2,000 inversions take 0.2 s, and Brent's bracketing cannot
  diverge.

### Validation against Deribit (BTC sample)

| Comparison | n | Median | 5th to 95th percentile | Agreement |
|---|---|---|---|---|
| Mark price inverted with Deribit's forward vs Deribit's mark IV | 458 | -0.001 | -0.023 to +0.022 | max 0.009 beyond 90 days |
| Ticker best bid vs Deribit's bid IV, OTM | 422 | -0.001 | -0.005 to +0.005 | 98% within rounding |
| Ticker best ask vs Deribit's ask IV, OTM | 473 | 0.000 | -0.005 to +0.010 | 91% within rounding |
| Our mid IV (our forward) vs Deribit's mark IV, fitted quotes | 373 | +0.009 | -0.26 to +0.29 | mark inside our bid ask IV band for 99.5% |

Units are vol points. Deribit reports IVs rounded to 0.01 vol points, so differences up to
0.006 are agreement. The first three rows test the solver and the conventions (forward, zero
rates, ACT/365 from `creation_timestamp`); the last row compares methods: Deribit's mark is its
own smooth surface, and our mids sit around it within the market's spread. We and Deribit also
agree on when no IV exists: 161 BTC ITM bids sit at or below intrinsic value, where Deribit
reports 0 and we return NaN. The residual disagreements have identified causes:

* Marks of the shortest expiries: within one expiry Deribit reports `underlying_price` values a
  few dollars apart (rows computed milliseconds apart), and a two dollar forward error moves a
  near the money 2 day IV by a few hundredths of a vol point.
* Tickers: a few percent of OTM quotes differ by more than rounding because the fields of one
  ticker are not computed at the same instant (for one call the reported bid is 0.0100 BTC but
  its `bid_iv` corresponds to 0.0106, the mark). ITM quotes are worse conditioned (small vega),
  so a few dollars of such asynchrony moves their IV by up to 5 vol points; they are not used
  for fitting.
* Six marks of at most 1.5e-6 BTC cannot be inverted meaningfully: rounding to 8 decimals
  alone moves their IV by more than 0.01 vol points. They are reported separately, not hidden.

### SVI per expiry

* **Parameterization and constraints.** Raw SVI in total variance, w(k) = a + b(rho(k - m) +
  sqrt((k - m)^2 + sigma^2)), fitted to the OTM mid IVs of each expiry separately. The optimizer
  works in the two asymptotic wing slopes s_L = b(1 - rho), s_R = b(1 + rho), the minimum total
  variance v, m and sigma, so every constraint is a plain bound: slopes in [1e-6, 2] give b > 0,
  |rho| < 1 and Roger Lee's wing bound b(1 + |rho|) <= 2 (a necessary condition for no
  arbitrage, so it excludes nothing legitimate); v >= 0 is the nonnegative minimum variance;
  sigma >= 1e-4. A first version bounded b by 2 / (1 + |rho|) directly; a deliberately hard
  test (a symmetric smile with wings steeper than the bound) showed the solver stalling on the
  kink of |rho| at rho = 0, which the slope parameterization removes.
* **Starting points.** For fixed (m, sigma), SVI is linear in the other three parameters (De
  Marco and Martini's quasi explicit reduction), so a 21 by 20 grid over (m, sigma) costs one
  small linear least squares per point. The eight best grid points and one heuristic start seed
  a bounded least squares fit (trust region reflective), and the best result is polished with an
  active set method that lands exactly on any binding bound. On both samples all nine starts end
  within 0.01% of the same cost on every expiry, so the fits are not initialization accidents.
* **Weights.** Residuals are divided by each quote's bid ask width in total variance by default,
  which targets the quality measure we care about (fitted vols inside the market's band) and
  trusts quotes in proportion to how tightly they are made. Over all fitted quotes:

  | Weighting | BTC RMSE | BTC inside band | ETH RMSE | ETH inside band | Worst slice inside band |
  |---|---|---|---|---|---|
  | inverse spread (default) | 0.23 | 98.1% | 0.27 | 99.0% | 94% |
  | uniform in total variance | 0.20 | 96.8% | 0.23 | 97.1% | 87% |
  | vega | 0.38 | 95.4% | 0.43 | 96.1% | 82% |

  RMSE is in vol points. Uniform weights give the smallest average error, inverse spread weights
  put the most fitted vols inside the market; vega weights neglect the low vega wings, where
  their errors reach 3.7 vol points.

Per expiry on the BTC sample (inverse spread weights):

| Expiry | Days | Quotes | ATM vol | rho | RMSE (vol pts) | Max error | Inside band |
|---|---|---|---|---|---|---|---|
| 8OCT26 | 2.4 | 20 | 29.1% | -0.23 | 0.27 | 0.55 | 100% |
| 9OCT26 | 3.4 | 19 | 31.2% | -0.31 | 0.34 | 1.03 | 95% |
| 16OCT26 | 10.4 | 19 | 32.2% | -0.39 | 0.23 | 0.51 | 100% |
| 23OCT26 | 17.4 | 23 | 32.5% | -0.34 | 0.15 | 0.30 | 100% |
| 30OCT26 | 24.4 | 49 | 33.5% | -0.09 | 0.37 | 1.73 | 94% |
| 27NOV26 | 52.4 | 48 | 35.7% | -0.37 | 0.17 | 0.62 | 98% |
| 25DEC26 | 80.4 | 48 | 36.4% | -0.29 | 0.28 | 1.02 | 96% |
| 26MAR27 | 171.4 | 52 | 37.3% | -0.25 | 0.16 | 0.60 | 100% |
| 25JUN27 | 262.4 | 54 | 38.4% | -0.25 | 0.11 | 0.45 | 100% |
| 24SEP27 | 353.4 | 41 | 38.9% | -0.58 | 0.07 | 0.34 | 100% |

The term structure is upward sloping (29% at 2 days to 39% at a year) with a put skew at every
maturity. One fit leans on a constraint: on the one year expiry (BTC and ETH alike) the left
wing sits exactly on Lee's bound, i.e. the quoted puts alone would extrapolate to a steeper
asymptotic wing than any arbitrage free smile allows. Inside the quoted range that fit is the
best of all (0.07 vol points RMSE); beyond it, the far left wing is set by the bound, not by
data.

### Static arbitrage

The fitted smiles are tested on a dense grid (k from -1.5 to 1.5, strikes from 22% to 448% of
the forward): Durrleman's g(k) >= 0 for butterflies, total variance nondecreasing in maturity at
fixed k for calendars. The raw coin quotes are tested separately, with no forward and no model:
calls falling and puts rising with strike, both convex. Each market test runs on mids
(consistency of the data) and on executable prices, buying at the ask and selling at the bid
over every pair or triple of strikes (could anyone trade it?). A data level calendar test
compares mid total variance at matched k across consecutive expiries.

| Check | Prices | BTC tests | BTC violations | ETH tests | ETH violations |
|---|---|---|---|---|---|
| butterfly (SVI, per slice) | model | 10 slices | 1, outside the quotes | 10 slices | 0 |
| calendar (SVI) | model | 9 pairs | 3, outside the quotes | 9 pairs | 0 |
| call and put monotonicity | mid | 869 | 0 | 727 | 1 |
| call and put monotonicity | executable | 18,779 | 0 | 12,524 | 0 |
| call and put convexity | mid | 845 | 84 | 703 | 71 |
| call and put convexity | executable | 285,785 | 0 | 145,187 | 0 |
| calendar at matched k | mid, interpolated | 325 | 0 | 261 | 0 |

* **The data contain no tradable static arbitrage.** About one in ten neighbouring triples of
  mids is not convex, concentrated in deep ITM strikes where spreads are widest (the largest,
  BTC 24SEP27 puts at 170k, 180k and 190k, misses by 0.005 BTC), but none survives the bid ask
  spread.
* **The SVI fits are arbitrage free wherever there are quotes**: g stays positive inside every
  slice's quoted range and total variance rises between every pair of consecutive expiries over
  their common quoted range. Outside the quotes, independent per expiry fits can and do break:
  on BTC, 23OCT26's left wing below k = -0.53 has negative density, and its total variance
  crosses 30OCT26's in both far wings (k below -0.38 and above 1.08), as does 27NOV26 against
  25DEC26 far in the left wing. Per slice SVI says nothing about consistency across maturities,
  which motivates the surface fit below.

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

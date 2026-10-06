# ivcrypto

Implied volatility surfaces for Bitcoin and Ether options, built from real Deribit order book
snapshots. The package computes implied volatilities that reproduce Deribit's own, fits SVI to
every expiry, tests the quotes and the fits for static arbitrage, fits an arbitrage free SSVI
surface and calibrates Heston and Bates (Heston with jumps), then measures where each model
fails and by how much.

This README is a short research note on one BTC and one ETH snapshot taken on the evening of
5 October 2026. Every number in it comes from the generated reports in
[`reports/sample`](reports/sample), which the commands under [Reproduce](#reproduce) rebuild
from the committed data.

## Findings

On the BTC snapshot (22:27 UTC; 373 out of the money quotes on ten expiries from 2.4 days to
one year):

1. **Our implied volatilities reproduce Deribit's own** within its 0.01 vol point rounding for
   98% of OTM bids and 91% of OTM asks. That takes two conventions the API does not state:
   coin premiums convert to USD at each expiry's forward, because the options are inverse, and
   the forward comes from put call parity, which sits up to 3.2 bps above the future.
2. **SVI fitted per expiry matches the smiles within the spread**: 0.23 vol points RMSE, with
   98% of fitted vols inside the bid ask band (ETH: 0.27 and 99%). All nine starting points of
   every fit agree.
3. **The quotes contain no tradable static arbitrage**, and the SVI fits are arbitrage free
   wherever there are quotes. Beyond the quotes the independent fits break (4 violations). An
   SSVI surface is arbitrage free everywhere, verified at 130 maturities, but misses by 2.1 vol
   points RMSE, with only 34% of fitted vols inside the band.
4. **Heston fails, in a specific way.** With one parameter set it misses by 2.0 vol points RMSE
   (20% inside the band), nine times SVI's error, and needs a vol of vol near 5 that breaks the
   Feller condition (ratio 0.23). The market's ATM smile curvature keeps growing as maturity
   shrinks, roughly like 1/T from a year down to 2.4 days. Heston's levels off below 10 days,
   as it must in any diffusive stochastic volatility model, and from 2 to 10 days the market's
   smiles even tilt the opposite way to Heston's. Fitted to one expiry at a time, Heston still
   misses by 1.3 vol points and pushes vol of vol and mean reversion to their bounds.
5. **Jumps (Bates) fix the middle of the surface, not the short end, and not as crashes.** Bates
   more than halves Heston's error on BTC (0.87 against 2.01 vol points RMSE, 43% of fitted vols
   inside the band) and cuts it by a quarter on ETH (2.27 against 3.06). Its smile curvature
   matches the market's within about 20% from 10 to 80 days, where Heston's is up to 2.3 times
   too high, but the 2 and 3 day smiles stay the worst fitted: at 2.4 days Bates has a quarter
   of the market's curvature on BTC and nearly twice it on ETH. The jumps the calibration asks
   for are frequent, small and upward on average (87 to 100 a year of +1% to +2.5%, the BTC
   intensity at its cap), not rare crashes: when jumps must be down on average, their mean goes
   to exactly zero on both coins. The 2 to 3 day smiles need something else, such as a second,
   fast variance factor or rough volatility.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/sample/BTC/20261005T222705Z/figures/smiles_dark.png">
  <img alt="BTC smiles for ten expiries: market bid ask bands and mids, SVI per expiry, and Heston and Bates with one parameter set each" src="reports/sample/BTC/20261005T222705Z/figures/smiles_light.png">
</picture>

*BTC smiles: market bid to ask (bars) and mids (dots), SVI fitted per expiry, and Heston and
Bates, each with one parameter set for all expiries. Heston misses the short dated curvature and
the far put wings. Bates fixes the wings and the maturities beyond 10 days but not the 2 and 3
day smiles, where its nearly fixed jump size puts a kink next to the money.*

## Motivation

Deribit's BTC and ETH options are a demanding test bench for volatility models. Expiries run
from hours to over a year, so the short end, where diffusive stochastic volatility models are
known to struggle, is densely quoted: the BTC snapshot has 12 expiries between 9.5 hours and
353 days. Smiles are steep (from 29% at the money to over 90% in the far put wings), the market
trades around the clock, and the contracts are inverse, quoted and settled in coin. That last
point makes the data layer a source of errors in its own right whenever conventions are assumed
rather than checked. The API is public, so every result can be reproduced.

The note asks four questions:

1. Can we compute implied volatilities that match the exchange's own?
2. How well does SVI fit each expiry, judged against the bid ask spread rather than the mid
   alone?
3. Are the quotes, and the surfaces fitted to them, free of static arbitrage?
4. Where, and by how much, does Heston fail, does adding jumps (Bates) fix it, and what does
   the pattern of failure point to?

## Data

Snapshots come from Deribit's public API v2 (no account needed). One snapshot is the full
option chain of one currency from a single bulk call, plus the futures, the index and,
optionally, one ticker per option:

```bash
uv run ivcrypto fetch --currency BTC              # writes data/raw/BTC/<YYYYMMDDTHHMMSSZ>/
uv run ivcrypto fetch --currency ETH --tickers    # also Deribit's own bid/ask IVs and greeks
```

Each snapshot directory holds one Parquet file per API table, with field names and values
exactly as returned, and a `manifest.json` with the server timestamps of every call. Two real
snapshots are committed so that the tests and every result below run offline:

| Sample | Taken (UTC) | Options | Expiries | Size |
|---|---|---|---|---|
| `data/sample/BTC/20261005T222705Z` | 2026-10-05 22:27:05 | 946 | 12 (9.5 hours to 353 days) | 196 KB |
| `data/sample/ETH/20261005T223117Z` | 2026-10-05 22:31:17 | 790 | 12 (9.5 hours to 353 days) | 172 KB |

[docs/deribit_api.md](docs/deribit_api.md) lists every field the pipeline relies on, its units
and null conventions, and the checks that established Deribit's pricing convention.

* **One bulk call per snapshot, not one ticker per option.** `get_book_summary_by_currency`
  returns the whole chain computed within 20 ms. A ticker sweep takes about 4 minutes, and
  quotes taken minutes apart create spurious arbitrage. Tickers are kept only to validate our
  IVs against Deribit's.
* **Raw storage.** Snapshots store the API rows unchanged; all interpretation happens in later
  stages, so a change of method never requires refetching data.
* **Rate limits.** The client throttles itself to half of Deribit's documented limits (the per
  IP limit for public requests is unpublished) and retries only what retrying can fix: rate
  limit errors, server errors and network failures.

## Method

`ivcrypto build` runs the analysis of one snapshot in this order: forwards and cleaning,
implied volatilities, validation against Deribit, SVI per expiry, SSVI, static arbitrage checks,
Heston, Bates, comparison. Where a step had a meaningful alternative, the choice and its
tradeoff are stated with it.

### Prices, forwards and time

* **Coin prices to USD with the expiry's forward, not spot.** A Deribit call pays
  `(S_T - K)^+ / S_T` coins, so its coin price is `Black76(F) / F` and its USD forward premium
  is the coin price times F. This is Deribit's own convention: it reproduces their bid and ask
  IVs exactly. Converting with the spot index instead mixes the futures basis into every
  premium.
* **Forward from put call parity, not the future.** In coin terms parity reads
  `C - P = 1 - K/F`, so every strike quoted on both sides implies F with no model, spot or USD
  rate. The estimator takes the eight strikes nearest the money, drops pairs beyond five robust
  standard deviations and averages the rest with weights from each pair's bid ask spreads. An
  expiry falls back to Deribit's underlying price (the future) when fewer than two pairs exist
  or they scatter by more than 25 bps. On the BTC sample the pairs scatter by 0.5 to 2.8 bps,
  so each forward has a standard error of at most 1.05 bps. The future is observable and
  liquid, but the options imply a forward 2.5 to 3.2 bps above it for the four 1 to 6 month
  expiries (about 3 to 5 standard errors, comparable to the futures' own bid ask spreads of 1.1
  to 5.1 bps) and within 1.6 bps of it elsewhere. On those four expiries the choice is visible
  near the money: call and put IVs at the same strike differ by 0.09 to 0.16 vol points (median
  per expiry) with the future as the forward, and by 0.01 to 0.11 with the parity forward.
* **Zero coin interest rate, tested rather than assumed.** Regressing `C - P` on K gives a coin
  discount factor between 0.9995 and 1.0006 for every BTC expiry.
* **Time to expiry** is ACT/365 calendar time (the market never closes) to the 08:00 UTC
  expiry, measured from when Deribit computed the quotes (`creation_timestamp`). ACT/365 is
  what reproduces Deribit's IVs; ACT/365.25 shifts the reproduced marks by +0.014 vol points
  (median). A business time clock that gives weekends less variance would be a refinement; it
  is not used.

### Filters

Every threshold is configurable (`config/default.toml`), and every quote keeps the name of the
first filter that removed it. On the BTC sample:

| Filter | Default | Removed | Remaining |
|---|---|---|---|
| expiry closer than | 2 days | 120 | 826 |
| no bid (null or 0) | | 17 | 809 |
| no ask, crossed | | 0 | 809 |
| relative spread above | 50% | 23 | 786 |
| in the money | OTM only | 413 | 373 |

The two day minimum drops the 0.4 and 1.4 day expiries, whose slices are thin (7 two sided OTM
quotes for the shortest) and whose payoff, on a 30 minute average of the index, differs
measurably from a European option on the 08:00 forward. The shortest expiry kept is 2.4 days,
so the short end where Heston is expected to struggle stays in the sample. In the money options
are dropped because their information is already in the OTM option at the same strike (through
parity) and their IVs are badly conditioned.

### Implied volatility

Brent's method on the OTM time value. Under undiscounted Black 76 a call and a put at one
strike differ by exactly `F - K`, so every price maps to the time value of the OTM option at
its strike, which is inverted in normalized units (`s = sigma sqrt(T)`, price divided by F).
Volatility is bracketed in [0.01%, 1000%]; prices at or below intrinsic value or at or above
the upper bound return NaN rather than a forced solution. Bid, mid and ask IVs are inverted
separately (the mid IV is not the average of the bid and ask IVs). A vectorized Newton solver
or Jäckel's "Let's be rational" would be faster, but 2,000 inversions take 0.2 s, and Brent's
bracketing cannot diverge.

### SVI per expiry

* **Parameterization and constraints.** Raw SVI in total variance,
  `w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))`, fitted to the OTM mid IVs of each
  expiry separately. The optimizer works in the two asymptotic wing slopes `b (1 - rho)` and
  `b (1 + rho)`, the minimum total variance, m and sigma, so every constraint is a plain bound:
  slopes in [1e-6, 2] give b > 0, |rho| < 1 and Roger Lee's wing bound `b (1 + |rho|) <= 2` (a
  necessary condition for no arbitrage, so it excludes nothing legitimate); a nonnegative
  minimum variance; sigma >= 1e-4. A first version bounded b by `2 / (1 + |rho|)` directly. A
  deliberately hard test, a symmetric smile with wings steeper than the bound, showed the
  solver stalling on the kink of |rho| at rho = 0, which the slope parameterization removes.
* **Starting points.** For fixed (m, sigma), SVI is linear in the other three parameters (De
  Marco and Martini's quasi explicit reduction), so a 21 by 20 grid over (m, sigma) costs one
  small linear least squares per point. The eight best grid points and one heuristic start
  seed a bounded least squares fit (trust region reflective), and the best result is polished
  with an active set method that lands exactly on any binding bound. On both samples all nine
  starts end within 0.01% of the same cost on every expiry, so the fits are not initialization
  accidents.
* **Weights.** Residuals are divided by each quote's bid ask width in total variance. This
  targets the quality measure that matters (fitted vols inside the market's band) and trusts
  quotes in proportion to how tightly they are made. Over all fitted quotes:

  | Weighting | BTC RMSE | BTC inside band | BTC worst slice | ETH RMSE | ETH inside band | ETH worst slice |
  |---|---|---|---|---|---|---|
  | inverse spread (default) | 0.23 | 98.1% | 94% | 0.27 | 99.0% | 94% |
  | uniform in total variance | 0.20 | 96.8% | 88% | 0.23 | 97.1% | 87% |
  | vega | 0.38 | 95.4% | 84% | 0.43 | 96.1% | 82% |

  RMSE is in vol points; the worst slice is the expiry with the smallest share inside the band.
  Uniform weights give the smallest average error and inverse spread weights put the most
  fitted vols inside the market; vega weights neglect the low vega wings, where their errors
  reach 3.7 vol points.

### Static arbitrage checks

The fitted smiles are tested on a dense grid (k from -1.5 to 1.5, strikes from 22% to 448% of
the forward): Durrleman's condition g(k) >= 0 for butterflies, and total variance nondecreasing
in maturity at fixed k for calendars. The raw coin quotes are tested separately, with no forward
and no model: calls falling and puts rising with strike, both convex. Each market test runs on
mids (is the data consistent?) and on executable prices, buying at the ask and selling at the
bid over every pair or triple of strikes (could anyone trade it?). A data level calendar test
compares mid total variance at matched k across consecutive expiries. Parameter constraints do
not replace these checks: a test reproduces Vogt's example from Gatheral and Jacquier (2014),
which satisfies every SVI constraint and still has negative density, and the checker flags it.

### SSVI surface

SSVI (Gatheral and Jacquier 2014) describes the whole surface with the ATM total variance theta
of each expiry and three global parameters,
`w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))`, with the power law
`phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))`. Their sufficient conditions are
imposed as bounds: `eta (1 + |rho|) <= 2` and `0 < gamma <= 1/2` (no butterfly arbitrage), and
theta nondecreasing in maturity (no calendar arbitrage). Every SSVI slice is a raw SVI slice, so
the checker above verifies the result independently rather than trusting the theorem, including
at maturities between expiries, where theta is interpolated linearly in time.

### Heston

**Pricer.** European prices come from Lewis's single integral on the forward,
`C = F - sqrt(FK)/pi * integral of Re[exp(iu ln(F/K)) phi(u - i/2)] / (u^2 + 1/4) du`, with the
"little trap" characteristic function of Albrecher et al. (2007), which keeps the complex
logarithm on its principal branch at long maturities. Lewis rather than Carr Madan FFT: no
damping parameter to tune and no strike grid to interpolate onto Deribit's irregular strikes.
Lewis rather than COS: COS is faster, but its accuracy rests on a truncation interval from
cumulants that is delicate for two day maturities at high volatility, and speed is not the
bottleneck (about 1 ms per maturity). The characteristic function depends only on maturity, so
it is evaluated once per maturity and every strike costs one matrix product. The integral uses
composite Gauss Legendre quadrature up to a limit read off the decay of the characteristic
function (up to about 250 for two day options with a vol of vol of 3).

Three numerical details came out of testing, not theory:

* Panels must be narrow near the origin: the factor `1/(u^2 + 1/4)` has poles at u = ±i/2, and
  panels 2 wide there left errors of 1e-10 (relative to F) instead of 1e-15.
* The textbook little trap formula divides `beta - d` by xi^2, which cancels catastrophically as
  xi → 0; the exact rewrite `beta - d = -xi^2 (iu + u^2) / (beta + d)` removes it.
* numpy's complex `log1p` is computed as `log(1 + z)` and loses all relative accuracy for tiny z
  (4e-5 relative error at |z| = 2e-12 with numpy 2.4), so a short series is used there. Without
  it, the Black 76 limit stalled at an error of 7e-6 of F.

**Validation.**

| Test | Result |
|---|---|
| Lewis (2000) reference calls, K = 80 to 120 | maximum error 1.4e-12 |
| Fang and Oosterlee (2008) test case (Feller violated) | 5.785155434, against their published 5.785155450; a 30 digit evaluation gives 5.785155434376, so their reference is good to 2e-8 and ours to 3e-13 |
| Fast quadrature against adaptive quadrature, 2 days to 3 years, vol of vol 3, strikes to 3 standard deviations | largest difference 2e-13 of F |
| Black 76 limit: xi → 0 with v0 = theta | error falls linearly in xi: 7.4e-4, 7.4e-6, 7.4e-8 of F at xi = 1e-2, 1e-4, 1e-6 |
| Monte Carlo, Andersen's QE scheme with martingale correction, 100,000 paths | every price within 4 standard errors (largest z = 1.85 over 36 prices), including Feller violating parameters; the simulated forward is a martingale |
| Calibration on quotes priced by Heston itself | parameters recovered to 1e-4 |
| Short maturity limits | ATM skew converges to `rho xi / (4 sqrt(v0))` and the curvature levels off, as theory says |

**Calibration.** Least squares (trust region reflective) on vega weighted price errors,
`(model - mid) / vega`, which are IV errors to first order without a root search inside the
optimizer. Every expiry gets equal total weight, so that the 50 strike expiries do not drown the
20 strike ones. Bounds: v0 and theta in [1e-4, 4], kappa in [1e-3, 50], xi in [1e-2, 10], rho in
[-0.99, 0.99]. The best four of eight starting points are refined, and on both samples all four
end at the same cost (within 1e-8 relative). Reported errors are exact IV errors after inverting
the model prices. The Feller condition is reported, not imposed: imposing it would forbid the
high vol of vol that the smiles demand. As a diagnostic, Heston is also fitted to each expiry
alone, which shows whether a failure is about the term structure or about the shape of single
smiles.

### Bates

**Model.** Bates (1996) adds a jump to Heston's forward: `lam` jumps a year, with log sizes
N(`mu_j`, `sigma_j^2`), independent of the diffusion and compensated so that the forward stays
a martingale. The characteristic function is Heston's times
`exp(lam T (exp(i u mu_j - u^2 sigma_j^2 / 2) - 1 - i u m))`, with `m = exp(mu_j + sigma_j^2/2) - 1`
the mean relative jump, so Bates slots into the Lewis pricer unchanged; the quadrature only adds
panels for the extra phase the jumps bring. Lognormal jumps rather than Kou's double
exponential jumps (separate up and down tails, at one more parameter) or jumps in the variance
as well (SVJJ, Duffie, Pan and Singleton 2000): Bates is the standard first step and the model
the Heston diagnosis points to, and it keeps every other part of the pipeline unchanged.

**Validation.**

| Test | Result |
|---|---|
| No jumps (`lam` = 0) | identical to Heston, bit for bit |
| xi → 0 with v0 = theta, against Merton's jump diffusion priced as a Poisson mixture of Black 76 prices | within 1e-13 of F at 2 days, 0.1 and 1 year (test tolerance 1e-12) |
| Fast quadrature against adaptive quadrature, 2 days to 1 year, including 100 jumps a year of fixed size | largest difference 1.2e-12 of F |
| Monte Carlo: QE for the variance, exact compound Poisson jumps at maturity, 100,000 paths | every price within 4 standard errors (largest z = 2.03 over 21 prices); the simulated forward is a martingale |
| Characteristic function at u = 0 and u = -i | 1 (the forward is a martingale) |
| Calibration on quotes priced by Bates itself | IV errors under 0.01 vol points, intensity recovered within 5% |
| Calibration on quotes priced by Heston | no jumps found (`lam` at 0), Heston parameters recovered to 1e-4 |

**Calibration.** Exactly Heston's objective, quotes and weights, so that the two models differ
only in the jumps. Bates nests Heston, and the calibrated Heston parameters with no jumps are
always among the refined starts, so Bates can only improve on Heston. Bounds: Heston's for the
variance; `lam` in [0, 100] a year, `mu_j` in [-1, 1], `sigma_j` in [0, 1]. The cap on `lam` is
the one judgment call: at two jumps a week, jumps stop being rare events and start to act as a
second source of diffusion. Starting points: the calibrated Heston parameters and the eight
Heston starts, each crossed with twelve jump settings (2, 10 or 40 jumps a year; mean log jump
-0.1 or +0.05; jump volatility 0.04 or 0.12); the four best by initial cost are refined, plus
the no jump start if it is not among them. Three diagnostic refits each change one jump bound:
jumps down on average (`mu_j` at most 0), rare jumps (at most 10 a year), and many jumps (up to
1000 a year). They show which kind of jump the data ask for.

### Comparing models

Every model is scored on the same quotes: RMSE of model minus mid IV, largest error, and the
share of model IVs inside the bid ask band, overall and by maturity and moneyness. The ATM term
structure compares skew and curvature over one ATM standard deviation either side of the money,
`s = sqrt(w(0))`, at the same three strikes for the market (its SVI fit) and for each model:

    skew = (sigma(s) - sigma(-s)) / (2 s),    curvature = (sigma(s) + sigma(-s) - 2 sigma(0)) / s^2.

A first version used the derivatives at k = 0. For smooth smiles the two agree (the market's
power law exponents move by at most 0.01), but the calibrated Bates smile at 2.4 days has a
kink next to the money, from its nearly fixed jump size, and a negative second derivative at
k = 0 although it rises on both sides over the quoted strikes. With s proportional to sqrt(T), the
measure keeps the short maturity limits of the derivatives (finite for Heston, growing like
T^(-1/2) with jumps).

## Data issues encountered

* **Inverse settlement.** The obvious conversion, coin price times the spot index, produced IV
  errors of up to 80 vol points. Deribit's options are quoted and settled in coin, and the
  right conversion uses each expiry's forward.
* **The future is not the forward.** The options imply forwards 2.5 to 3.2 bps above the futures
  for the 1 to 6 month expiries; using the future leaves a systematic call put IV gap near the
  money.
* **A forward estimator that looked precise and was not.** An earlier version took the median of
  the six pairs with the tightest spreads. Validation showed that this set can be lopsided (all
  six on one side of the money) and that a median of six pairs with 10 to 30 bps of spread noise
  is itself uncertain by several bps, which had made the gap to the futures look larger (up to
  4.8 bps) than it is. The weighted mean of the eight nearest pairs, with a reported standard
  error, replaced it.
* **Timestamps.** The bulk summary is a cache: rows carry a `creation_timestamp` up to 0.9 s
  before the response time, and rows of one call are computed up to 20 ms apart. Times to
  expiry are measured from `creation_timestamp`.
* **Forwards within an expiry.** Deribit reports `underlying_price` values a few dollars apart
  within one expiry (rows computed milliseconds apart). A two dollar forward error moves a near
  the money 2 day IV by a few hundredths of a vol point, which is most of the residual
  disagreement on the shortest marks.
* **Ticker fields are not synchronous.** A few percent of OTM ticker quotes differ from
  Deribit's IVs by more than rounding because the fields of one ticker are not computed at the
  same instant: for one call the reported bid is 0.0100 BTC but its `bid_iv` corresponds to
  0.0106, the mark. ITM quotes are worse conditioned (small vega), so a few dollars of such
  asynchrony moves their IV by up to 5 vol points; they are not used for fitting.
* **Bids below intrinsic value.** 161 BTC ITM bids (146 on ETH) sit at or below intrinsic value.
  Deribit reports an IV of 0 for them; we return NaN, and both agree that no IV exists.
* **Marks too small to invert.** Six BTC marks of at most 1.5e-6 BTC cannot be inverted
  meaningfully: rounding to 8 decimals alone moves their IV by more than 0.01 vol points. They
  are reported separately, not hidden.
* **Mids are not always convex.** 84 of 845 neighbouring BTC strike triples have non convex mid
  prices, concentrated deep in the money where spreads are widest (the largest, the 24SEP27
  puts at 170k, 180k and 190k, misses by 0.005 BTC). None survives the bid ask spread.
* **The shortest expiries.** The 0.4 and 1.4 day slices are thin and settle on a 30 minute
  average of the index; they are filtered out (see Filters).

## Results

### Our implied volatilities agree with Deribit's

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/sample/BTC/20261005T222705Z/figures/iv_validation_dark.png">
  <img alt="Histograms of differences between our IVs and Deribit's: mark reproduction and our mid against Deribit's mark" src="reports/sample/BTC/20261005T222705Z/figures/iv_validation_light.png">
</picture>

| Comparison (BTC) | n | Median | 5th to 95th percentile | Agreement |
|---|---|---|---|---|
| Mark price inverted with Deribit's forward vs Deribit's mark IV | 458 | -0.001 | -0.023 to +0.022 | max 0.009 beyond 90 days |
| Ticker best bid vs Deribit's bid IV, OTM | 422 | -0.001 | -0.005 to +0.005 | 98% within rounding |
| Ticker best ask vs Deribit's ask IV, OTM | 473 | 0.000 | -0.004 to +0.010 | 91% within rounding |
| Our mid IV (our forward) vs Deribit's mark IV, fitted quotes | 373 | +0.009 | -0.26 to +0.29 | mark inside our bid ask IV band for 99.5% |

Units are vol points. Deribit reports IVs rounded to 0.01 vol points, so differences up to
0.006 are agreement. The first three rows test the solver and the conventions (forward, zero
rates, ACT/365 from `creation_timestamp`); on ETH, 97% of OTM bids and 95% of OTM asks agree
within rounding. The last row compares methods: Deribit's mark is its own smooth surface, and
our mids sit around it within the market's spread (ETH: 99.3% inside the band). The residual
disagreements all have identified causes, listed under data issues.

### SVI fits the smiles within the spread

Per expiry RMSE ranges from 0.07 to 0.37 vol points, and 94% to 100% of fitted vols lie inside
the bid ask band (per expiry table in the Heston section). The term structure slopes upward,
from 29% ATM vol at 2.4 days to 39% at a year. Every smile is steeper on the put side (SVI rho
from -0.09 to -0.58), although below 10 days its minimum sits just below the forward, so the
slope at the money is positive there. One fit leans on a constraint: on the one year expiry,
for BTC and ETH alike, the left wing sits exactly on Lee's bound, so the quoted puts alone
would extrapolate to a steeper asymptotic wing than any arbitrage free smile allows. Inside the
quoted range that fit is the best of all (0.07 vol points RMSE); beyond it, the far left wing
is set by the bound, not by data.

### No tradable arbitrage in the quotes; SVI fits are clean where quoted

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
  mids is not convex, but none survives the bid ask spread.
* **The SVI fits are arbitrage free wherever there are quotes**: g stays positive inside every
  slice's quoted range, and total variance rises between every pair of consecutive expiries
  over their common quoted range. Outside the quotes, independent per expiry fits can and do
  break: on BTC, 23OCT26's left wing has negative density between k = -0.95 and -0.53, and its
  total variance crosses 30OCT26's in both far wings (k below -0.38 and above 1.08), as does
  27NOV26 against 25DEC26 far in the left wing (k below -1.07). Per slice SVI says nothing about
  consistency across maturities, which motivates the surface fit.

### SSVI: arbitrage free everywhere, at nine times the error

* **Verified**: zero butterfly or calendar violations on both samples over k from -1.5 to 1.5,
  at the 10 expiries and at 120 maturities in between (smallest g on BTC: 0.26). Breaking
  `eta (1 + |rho|) <= 2` in a test does produce butterflies that the checker finds.
* **Costly**: over all BTC quotes the RMSE rises from 0.23 vol points (SVI per expiry) to 2.1,
  and only 34% of fitted vols lie inside the bid ask band instead of 98% (ETH: 2.6 vol points,
  37%). The same holds under every weighting (RMSE 2.0 to 2.7 on BTC, 2.4 to 3.3 on ETH). Two
  rigidities explain it: one rho for all maturities, where the per expiry fits want anything
  from -0.09 to -0.58, and the curvature exponent gamma, which sits on its bound of 1/2 under
  every weighting: the short dated smiles curve more than this arbitrage free family allows.

Per expiry SVI is the better description of the quoted smiles and is arbitrage free where quotes
exist; SSVI is the safe interpolator and extrapolator. Extended SSVI with a maturity dependent
rho (Hendriks and Martini 2019), or SVI slices fitted with a penalty on arbitrage and started
from the SSVI fit, as Gatheral and Jacquier suggest, would combine the two.

### Heston: where and how much it fails

On the BTC sample the single parameter set is v0 = 0.105 (32% vol), theta = 0.185 (43%),
kappa = 15.3, xi = 4.96, rho = -0.11. The Feller ratio `2 kappa theta / xi^2` is 0.23: the
calibration needs a vol of vol near 5, with mean reversion fast enough (half life 16 days) to
stop that vol of vol from flattening the long end.

Every model scored on the same 373 quotes (RMSE in vol points; share of model IVs inside the
bid ask band):

| Quotes | n | SVI | SSVI | Heston | Heston per expiry | Bates | SVI inside band | Heston inside band | Bates inside band |
|---|---|---|---|---|---|---|---|---|---|
| all | 373 | 0.23 | 2.10 | 2.01 | 1.26 | 0.87 | 98% | 20% | 43% |
| under 7 days | 39 | 0.30 | 2.24 | 2.74 | 1.23 | 1.39 | 97% | 18% | 56% |
| 7 to 30 days | 91 | 0.30 | 2.24 | 1.86 | 1.13 | 0.77 | 97% | 21% | 53% |
| 30 to 90 days | 96 | 0.23 | 3.01 | 2.65 | 1.89 | 0.99 | 97% | 24% | 41% |
| over 90 days | 147 | 0.13 | 0.88 | 1.23 | 0.70 | 0.62 | 100% | 18% | 36% |
| k below -0.2 (far puts) | 82 | 0.33 | 3.92 | 3.21 | 2.20 | 1.00 | 98% | 30% | 57% |
| k -0.2 to -0.05 | 69 | 0.26 | 1.08 | 1.72 | 1.16 | 1.04 | 96% | 22% | 23% |
| k -0.05 to 0.05 (at the money) | 85 | 0.16 | 1.54 | 1.85 | 0.82 | 0.89 | 100% | 8% | 36% |
| k 0.05 to 0.2 | 72 | 0.14 | 0.85 | 1.03 | 0.67 | 0.40 | 100% | 18% | 58% |
| k above 0.2 (far calls) | 65 | 0.18 | 0.84 | 1.14 | 0.44 | 0.83 | 97% | 23% | 40% |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/sample/BTC/20261005T222705Z/figures/residual_heatmaps_dark.png">
  <img alt="Heatmaps of mean model minus mid IV by expiry and moneyness bucket, for SVI, Heston and Bates" src="reports/sample/BTC/20261005T222705Z/figures/residual_heatmaps_light.png">
</picture>

*Mean model minus mid IV (vol points) by expiry and moneyness. SVI's errors stay under half a vol
point in every cell. Heston's have structure: its 2 and 3 day smiles are too flat (wings 1.7 to
4.3 vol points low), its 10 to 80 day smiles too curved (ATM 1.3 to 2.2 vol points low), and
most of its far puts are too cheap. Bates removes most of that structure; what remains sits at
2 and 3 days.*

| Expiry | Days | Quotes | ATM vol | SVI | SVI inside band | SSVI | Heston | Heston per expiry | Bates |
|---|---|---|---|---|---|---|---|---|---|
| 8OCT26 | 2.4 | 20 | 29.1% | 0.27 | 100% | 2.34 | 2.44 | 0.86 | 1.30 |
| 9OCT26 | 3.4 | 19 | 31.2% | 0.34 | 95% | 2.12 | 3.03 | 1.53 | 1.47 |
| 16OCT26 | 10.4 | 19 | 32.2% | 0.23 | 100% | 1.79 | 1.90 | 0.67 | 0.90 |
| 23OCT26 | 17.4 | 23 | 32.5% | 0.15 | 100% | 1.15 | 1.39 | 0.74 | 0.56 |
| 30OCT26 | 24.4 | 49 | 33.5% | 0.37 | 94% | 2.74 | 2.03 | 1.39 | 0.80 |
| 27NOV26 | 52.4 | 48 | 35.7% | 0.17 | 98% | 0.92 | 1.40 | 0.81 | 0.75 |
| 25DEC26 | 80.4 | 48 | 36.4% | 0.28 | 96% | 4.16 | 3.47 | 2.55 | 1.19 |
| 26MAR27 | 171.4 | 52 | 37.3% | 0.16 | 100% | 1.16 | 1.41 | 0.89 | 0.44 |
| 25JUN27 | 262.4 | 54 | 38.4% | 0.11 | 100% | 0.82 | 1.33 | 0.74 | 0.58 |
| 24SEP27 | 353.4 | 41 | 38.9% | 0.07 | 100% | 0.44 | 0.76 | 0.17 | 0.83 |

*RMSE in vol points per BTC expiry.*

**The short end.** In Heston, as in any diffusive stochastic volatility model, the ATM skew and
curvature of the smile converge to finite limits as T → 0; the skew at k = 0 converges to
`rho xi / (4 sqrt(v0))`, -0.41 for these parameters. The market's curvature shows no such limit.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/sample/BTC/20261005T222705Z/figures/atm_term_structure_dark.png">
  <img alt="ATM implied volatility and smile curvature against days to expiry, for the market, Heston and Bates" src="reports/sample/BTC/20261005T222705Z/figures/atm_term_structure_light.png">
</picture>

* The market's smile curvature (over one ATM standard deviation, see Comparing models) grows
  like T^(-1.07) from a year down to 2.4 days, and faster, like T^(-1.26), over the three
  shortest expiries: it rises 64% from 3.4 to 2.4 days (47.5 to 78.0). Heston's rises 11% over
  the same step (50.8 to 56.4). Its curvature is nearly flat below 10 days (exponent 0.49) and
  falls steeply beyond 50 days (exponent 1.56, against the market's 1.06), as smiles flatten
  beyond the mean reversion time 1/kappa of 24 days. A single power law fitted to all ten
  Heston points gives 1.10, close to the market's 1.07, which is why such a summary misleads:
  it averages two wrong regimes.
* The single parameter set compromises between those regimes: too little curvature at 2.4 days,
  too much from 3.4 to 171 days (1.7 to 2.3 times the market's between 10 and 80 days, 28.1
  against 12.0 at 10 days), and too little again beyond 262 days, which spreads its errors over
  all maturities.
* At the very short end even the tilt of the smile is wrong. The market's 2 to 10 day smiles
  slope upward at the money (their minimum sits just below the forward; the skew over one
  standard deviation is +0.58 at 2.4 days), while Heston's skew is negative at every maturity
  (-0.34 at 2.4 days).

Curvature that keeps growing as maturity shrinks is what jumps produce. In a jump diffusion the
ATM skew still converges, but the curvature grows without bound, like T^(-1/2) as T → 0; rough
volatility models (Bayer, Friz and Gatheral 2016) also steepen short dated smiles without bound.
[`tests/test_asymptotics.py`](tests/test_asymptotics.py) checks both limits numerically, against
Heston and against Merton's jump diffusion.

**The deep put wing.** The largest errors are far OTM puts. The worst quote overall is the
25DEC26 put at strike 30k (k = -1.06): market 90.9% against 75.1% for the calibrated Heston, 15.8
vol points short, and still 79.2% (11.7 vol points short) when Heston is fitted to that expiry
alone. Heston's tails are too thin for this wing; Bates, below, gets to 85.3%.

**Refitting each expiry separately does not rescue it.** Each smile alone still misses by 0.2 to
2.5 vol points, and the parameters it needs are implausible and unstable: vol of vol at the cap
of 10 on three expiries, mean reversion at the cap of 50 on four, and v0 ranging from 0.015 to
0.53, quadrupling between neighbouring expiries (0.115 at 24 days, 0.481 at 52).

| Expiry | Days | v0 | kappa | theta | xi | rho | Feller ratio | RMSE |
|---|---|---|---|---|---|---|---|---|
| 8OCT26 | 2.4 | 0.015 | 7.1 | 3.89 | 10.0 | +0.10 | 0.55 | 0.86 |
| 9OCT26 | 3.4 | 0.021 | 11.7 | 1.91 | 10.0 | +0.05 | 0.45 | 1.53 |
| 16OCT26 | 10.4 | 0.017 | 50.0 | 0.24 | 6.1 | 0.00 | 0.65 | 0.67 |
| 23OCT26 | 17.4 | 0.055 | 50.0 | 0.17 | 6.9 | -0.07 | 0.36 | 0.74 |
| 30OCT26 | 24.4 | 0.115 | 50.0 | 0.15 | 9.3 | -0.13 | 0.17 | 1.39 |
| 27NOV26 | 52.4 | 0.481 | 50.0 | 0.10 | 9.0 | -0.14 | 0.13 | 0.81 |
| 25DEC26 | 80.4 | 0.534 | 36.1 | 0.11 | 10.0 | -0.18 | 0.08 | 2.55 |
| 26MAR27 | 171.4 | 0.236 | 24.5 | 0.17 | 8.6 | -0.15 | 0.11 | 0.89 |
| 25JUN27 | 262.4 | 0.271 | 16.2 | 0.17 | 6.6 | -0.13 | 0.13 | 0.74 |
| 24SEP27 | 353.4 | 0.128 | 9.7 | 0.19 | 4.1 | -0.10 | 0.22 | 0.17 |

The bounds are not what holds the fits back. Widening them in `heston/calibrate.py` (xi up to
40, kappa up to 500) sends the 2 and 3 day fits to xi = 19 and 26 and kappa = 420 and 500, for
RMSE improvements of only 0.07 and 0.17 vol points. The failure is structural, not a matter of
bounds or optimizer.

### Bates: jumps fix the middle, not the short end

On BTC the calibrated jumps are 100 a year, the cap, each of a fixed +2.5% (`sigma_j` at its
floor of 0), with the forward drifting down between jumps by as much as they add on average.
The variance part: v0 = 0.046 (21% vol), theta = 0.133 (36%), kappa = 16.3, xi = 9.2,
rho = -0.22, a Feller ratio of 0.05. Jumps contribute 0.060 of variance a year against 0.133
from the diffusion. The optimum is not sharp: of the four refined starts, three end at a second
optimum 10% worse in cost, with jump sizes spread by 2.4% around +0.7%.

* **What it fixes.** RMSE falls from 2.01 to 0.87 vol points, and the share inside the band
  rises from 20% to 43% (table above). The far puts improve most, 3.21 to 1.00 (the 30k put now
  misses by 5.6 vol points instead of 15.8), and every maturity bucket at least roughly halves
  its error. The smile curvature matches the market's within 12% from 10 to 80 days, where
  Heston's is 1.7 to 2.3 times too high (figure above).
* **What it does not.** The 2 and 3 day expiries remain the worst fitted (1.30 and 1.47 vol
  points). At 2.4 days Bates has a quarter of the market's curvature (18.0 against 78.0), and
  its curvature falls rather than rises from 3.4 to 2.4 days. Its skew there is positive, like
  the market's, but twice as large (+1.14 against +0.58). With a fixed jump size of +2.5%,
  close to the 2.4% standard deviation of a 2.4 day return, the short dated return distribution
  is a mixture of a few shifted lumps, and the smile kinks next to the money (first figure).

Which jumps do the data ask for? Each diagnostic refit changes one jump bound (RMSE in vol
points; short end: under 7 days; far puts: k below -0.2):

| Variant | | jumps a year | mean log jump | log jump vol | xi | RMSE | short end | far puts | inside band |
|---|---|---|---|---|---|---|---|---|---|
| default bounds | BTC | 100 (cap) | +0.024 | 0 (floor) | 9.2 | 0.87 | 1.39 | 1.00 | 43% |
| jumps down on average | BTC | 100 (cap) | 0 (bound) | 0.027 | 9.99 | 1.03 | 2.23 | 0.95 | 37% |
| at most 10 jumps a year | BTC | 10 (cap) | +0.054 | 0 (floor) | 5.6 | 1.36 | 2.05 | 1.90 | 34% |
| up to 1000 jumps a year | BTC | 237 | +0.006 | 0.016 | 10 (cap) | 0.77 | 1.31 | 0.76 | 46% |
| default bounds | ETH | 87 | +0.011 | 0.030 | 10 (cap) | 2.27 | 5.25 | 3.38 | 39% |
| jumps down on average | ETH | 2.1 | 0 (bound) | 0.148 | 5.2 | 2.16 | 3.17 | 3.34 | 29% |
| at most 10 jumps a year | ETH | 10 (cap) | +0.062 | 0.021 | 8.3 | 2.49 | 5.48 | 3.87 | 31% |
| up to 1000 jumps a year | ETH | 87 | +0.011 | 0.030 | 10 (cap) | 2.27 | 5.25 | 3.38 | 39% |

* **Not crashes.** Forced to be down on average, the mean jump goes to exactly zero, its bound,
  on both coins, and limited to 10 a year the jumps come out upward (mean log jumps of +0.054
  and +0.062). The short dated smiles bottom below the forward, which upward jumps reproduce and
  Heston's negative rho cannot.
* **The cap on BTC binds, but the conclusion does not depend on it.** Up to 1000 jumps a year,
  BTC takes 237 small ones (0.6% on average) and puts xi on its own cap; the short end improves
  only from 1.39 to 1.31.
* **On ETH, RMSE and the objective disagree.** Symmetric rare jumps (2 a year of 15% volatility)
  have a lower RMSE (2.16) but a higher objective than the default fit, because RMSE counts
  every quote equally while the objective gives every expiry equal weight; at the short end
  both still miss by more than 3 vol points.

The reading: in Bates the jumps act here as a fast, positively skewed noise rather than as crash
insurance, and that is enough to fix the maturities beyond 10 days, but not to bend the 2 and 3
day smiles the way the market does. Candidates for that are a second, fast mean reverting
variance factor (as in Christoffersen, Heston and Jacobs 2009), rough volatility, or jumps with
separate up and down tails.

### ETH tells the same story, more strongly

| | BTC | ETH |
|---|---|---|
| Options in snapshot, fitted OTM quotes | 946, 373 | 790, 307 |
| ATM vol, 2.4 days to one year | 29.1% to 38.9% | 33.8% to 54.6% |
| SVI per expiry: RMSE, inside band | 0.23, 98% | 0.27, 99% |
| SSVI: RMSE, inside band | 2.10, 34% | 2.62, 37% |
| Heston: RMSE, inside band | 2.01, 20% | 3.06, 21% |
| Heston per expiry: RMSE, range over expiries | 1.26, 0.17 to 2.55 | 1.91, 0.22 to 5.87 |
| Bates: RMSE, inside band | 0.87, 43% | 2.27, 39% |
| Bates: RMSE at 2.4 and 3.4 days | 1.30, 1.47 | 2.68, 6.93 |
| Heston v0, kappa, theta, xi, rho | 0.105, 15.3, 0.185, 4.96, -0.11 | 0.137, 27.8, 0.340, 8.11, -0.10 |
| Heston Feller ratio | 0.23 | 0.29 |
| Bates jumps a year, mean jump, xi | 100 (cap), +2.5%, 9.2 | 87, +1.2%, 10 (cap) |
| Skew at 2.4 days: market, Heston, Bates | +0.58, -0.34, +1.14 | +0.62, -0.38, +0.82 |
| Curvature at 2.4 days over 3.4 days: market, Heston, Bates | 1.64, 1.11, 0.70 | 1.51, 1.26, 1.85 |
| Bates curvature over the market's: 2.4 days, 10 to 80 days | 0.23, 0.88 to 1.10 | 1.80, 1.08 to 1.21 |
| Curvature exponent up to 10 days: market, Heston | 1.26, 0.49 | 1.22, 0.86 |
| Curvature exponent beyond 50 days: market, Heston | 1.06, 1.56 | 1.07, 1.73 |
| Per expiry Heston fits with xi at its cap of 10 | 3 of 10 | 7 of 10 |
| Mid convexity misses (executable) | 84 of 845 (0) | 71 of 703 (0) |
| SVI arbitrage outside the quotes | 4 | 0 |

ETH's calibrated Heston has more vol of vol (8.1) and faster mean reversion (27.8), so its
curvature overshoots the market's from 2.4 to 80 days (up to 1.8 times between 10 and 24 days)
and undershoots beyond 171 days. The shape is wrong in the same way as on BTC: too flat at the
short end (exponent 0.86 against 1.22), too steep at the long end (1.73 against 1.07), with the
wrong sign of skew up to 10 days. Bates fixes 17 to 80 days (0.6 to 0.8 vol points RMSE) but
overshoots the 2.4 day curvature by 80%. The 3.4 day ETH smile, which rises from 37% at the money
to 93% at k = -0.21, defeats both: 7.8 vol points RMSE for Heston with one parameter set, 5.9
fitted alone, 3.7 with xi allowed up to 40 (the fit goes there, with kappa = 470), and 6.9 for
Bates. The per expiry Heston fits put xi at its cap on seven of ten expiries, theta at a bound on
three, and v0 anywhere from 0.04 to 2.4.

### The surface

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/sample/BTC/20261005T222705Z/figures/surface_dark.png">
  <img alt="BTC SVI implied volatility surface against ATM standard deviations and days to expiry" src="reports/sample/BTC/20261005T222705Z/figures/surface_light.png">
</picture>

*The BTC SVI surface, each expiry drawn over ±2 of its own ATM standard deviations
(`k / sqrt(w(0))`), so every slice stays inside its quoted region; between expiries the vols are
interpolated linearly in log maturity, for display only. An interactive version (plotly) is in
the same folder.*

## Limitations

* **One evening.** Each currency is one snapshot, taken minutes apart on 5 October 2026. Nothing
  here says how stable the parameters are over time, how the models hedge, or whether the
  short dated features (such as the positive ATM skew below 10 days) are typical. A daily series
  of snapshots, which `ivcrypto fetch` can collect, is the obvious extension.
* **Mids as targets.** Fits target mid IVs and are judged by the share inside the bid ask band.
  The mid is not a traded price, and the filters keep quotes with spreads up to 50% of the mid.
* **Near simultaneous, not simultaneous, quotes.** The bulk summary is computed within about
  20 ms, and forwards within one expiry differ by a few dollars.
* **Settlement and clock.** Options settle on a 30 minute average of the index before 08:00 UTC
  and are treated as European on the forward at 08:00, which is why expiries under 2 days are
  excluded. Time is calendar time; weekends and scheduled events are not modelled.
* **Rates.** The coin rate is taken as zero (verified within ±6 bps of discount), and USD rates
  enter only through the forward.
* **OTM only.** In the money quotes inform the forward through parity but are not fitted.
* **In sample.** Fit quality is measured on the quotes fitted. There is no test on held out
  strikes or on the next day's quotes.
* **Calibration choices.** Heston and Bates are calibrated on first order IV errors with equal
  weight per expiry, from multiple local starts rather than a global search, within bounds whose
  effect is measured but not removed. Another weighting would move the single parameter
  compromise between maturities, and on ETH RMSE and the objective already rank two Bates fits
  differently.
* **One jump model, weakly identified.** Bates with a single lognormal jump size is the only
  jump model tested. Its best fits sit on the intensity cap (BTC) or the vol of vol cap (ETH),
  and BTC has a second optimum within 10% of the best, so the fit and its failures say more
  than the parameter values. A second variance factor, rough volatility and asymmetric jumps
  are untested. SVI per expiry is not arbitrage free outside its quotes, SSVI fits poorly, and
  extended SSVI is not implemented.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer. From the committed samples:

```bash
uv sync                                    # environment from uv.lock
uv run pytest                              # offline test suite, about a minute and a half

uv run ivcrypto build data/sample/BTC/20261005T222705Z --out results
uv run ivcrypto report results/BTC/20261005T222705Z --out reports/sample/BTC/20261005T222705Z
uv run ivcrypto build data/sample/ETH/20261005T223117Z --out results
uv run ivcrypto report results/ETH/20261005T223117Z --out reports/sample/ETH/20261005T223117Z
```

`build` takes 2 to 4 minutes per currency, most of it in two diagnostics, Heston fitted to
every expiry and the Bates variants (`--no-heston-per-expiry` and `--no-bates-variants` skip
them; the rest takes under a minute). It writes every table as Parquet or CSV under
`results/<CURRENCY>/<snapshot>/`, with a `manifest.json` recording the configuration, the git
commit and the timings. `report` turns a results directory into `report.md` with light and dark
figures; the committed reports were produced this way. Nothing is random outside the Monte Carlo
tests: rebuilding gives the same tables byte for byte with the library versions pinned in
`uv.lock`, so regenerated reports differ from the committed ones only in the git commit they
record (another CPU or BLAS may change the last digits).

On fresh data, and with your own settings (any subset of the keys in `config/default.toml`):

```bash
uv run ivcrypto fetch --currency BTC --tickers
uv run ivcrypto build data/raw/BTC/<snapshot> --out results --config my.toml
uv run ivcrypto report results/BTC/<snapshot> --out reports/BTC/<snapshot>
```

## Repository layout

```
src/ivcrypto/
  data/           Deribit client (rate limits, retries), snapshot fetch, Parquet store
  instruments.py  instrument names, 08:00 UTC expiries, ACT/365
  forwards.py     put call parity forwards, coin rate regression
  cleaning.py     USD conversion, log moneyness, filters
  black76.py      Black 76 prices, vega, no arbitrage bounds
  implied_vol.py  Brent implied volatility on the OTM time value
  validation.py   comparison with Deribit's IVs
  svi/            raw SVI, per expiry fits, SSVI
  arbitrage.py    butterfly, calendar and raw quote checks
  heston/         characteristic function, Lewis pricer, QE Monte Carlo, calibration
  bates/          Heston plus jumps: characteristic function, Monte Carlo, calibration, variants
  compare.py      model scores and ATM term structures
  pipeline.py     the whole analysis of a snapshot, results on disk
  plots.py        figures, light and dark
  report.py       Markdown report
  cli.py          ivcrypto fetch, build and report
config/default.toml   every threshold and setting, documented
data/sample/          committed BTC and ETH snapshots
reports/sample/       reports generated from them
docs/deribit_api.md   API fields, units and conventions
tests/                offline test suite with recorded API responses
```

## Development

```bash
uv run pytest              # offline test suite
uv run pytest -m network   # live checks that Deribit's schema has not changed
uv run ruff check .        # lint
uv run ruff format .       # format
```

All tests run offline: a fixture blocks every network connection, and the live tests run only
on request. CI runs ruff and the test suite on Python 3.11, 3.12 and 3.13. The project was built
in milestones, each with its tests:

| Milestone | Scope |
|---|---|
| M0 | Project scaffold, CI |
| M1 | Deribit data layer, Parquet snapshots, committed samples |
| M2 | Cleaning: conversion to USD, forwards, log moneyness, filters |
| M3 | Black 76 and implied volatility, validation against Deribit |
| M4 | SVI per expiry |
| M5 | Static arbitrage checks, plus SSVI |
| M6 | Heston pricing, validation and calibration, comparison with SVI |
| M7 | CLI, figures, reports and this note |
| M8 | Bates: pricer, validation, calibration, diagnostic refits, comparison |

## References

* Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap.
  Wilmott Magazine, January 2007.
* Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility
  model. Journal of Computational Finance 11(3).
* Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in
  Deutsche Mark options. Review of Financial Studies 9(1).
* Bayer, C., Friz, P. and Gatheral, J. (2016). Pricing under rough volatility. Quantitative
  Finance 16(6).
* Black, F. (1976). The pricing of commodity contracts. Journal of Financial Economics 3.
* Carr, P. and Madan, D. (1999). Option valuation using the fast Fourier transform. Journal of
  Computational Finance 2(4).
* Christoffersen, P., Heston, S. and Jacobs, K. (2009). The shape and term structure of the
  index option smirk: why multifactor stochastic volatility models work so well. Management
  Science 55(12).
* De Marco, S. and Martini, C. (2009). Quasi-explicit calibration of Gatheral's SVI model.
  Zeliade Systems white paper.
* Duffie, D., Pan, J. and Singleton, K. (2000). Transform analysis and asset pricing for affine
  jump-diffusions. Econometrica 68(6).
* Durrleman, V. (2010). From implied to spot volatilities. Finance and Stochastics 14(2).
* Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on
  Fourier-cosine series expansions. SIAM Journal on Scientific Computing 31(2).
* Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with
  application to the valuation of volatility derivatives. Global Derivatives and Risk
  Management, Madrid.
* Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative
  Finance 14(1).
* Hendriks, S. and Martini, C. (2019). The extended SSVI volatility surface. Journal of
  Computational Finance 22(5).
* Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with
  applications to bond and currency options. Review of Financial Studies 6(2).
* Jäckel, P. (2015). Let's be rational. Wilmott 2015(75).
* Kou, S. G. (2002). A jump-diffusion model for option pricing. Management Science 48(8).
* Lee, R. W. (2004). The moment formula for implied volatility at extreme strikes. Mathematical
  Finance 14(3).
* Lewis, A. L. (2000). Option Valuation under Stochastic Volatility. Finance Press.
* Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other exponential
  Lévy processes. Working paper, SSRN.
* Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal
  of Financial Economics 3.

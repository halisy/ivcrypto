# Deribit API notes

Everything on this page was observed on the live production API (v2) on 2026-10-05, in the raw
responses and in the committed samples (`data/sample/BTC/20261005T222705Z`,
`data/sample/ETH/20261005T223117Z`). The data layer in `src/ivcrypto/data/` is built around these
observations, and `fetch_snapshot` raises if a field it relies on disappears. To recheck the
live schema, run `uv run pytest -m network`.

## Endpoints

All public methods are plain HTTPS GETs of `https://www.deribit.com/api/v2/{method}?{params}`,
with no authentication.

| Table | Method and parameters | Rows (BTC, ETH) | Used for |
|---|---|---|---|
| `option_instruments` | `public/get_instruments?currency=BTC&kind=option&expired=false` | 946, 790 | strikes, expiry times, tick sizes |
| `option_book` | `public/get_book_summary_by_currency?currency=BTC&kind=option` | 946, 790 | bid, ask, mark, mark IV, underlying price, open interest |
| `future_book` | `public/get_book_summary_by_currency?currency=BTC&kind=future` | 13, 13 | forward diagnostics |
| `index` | `public/get_index_price?index_name=btc_usd` | 1, 1 | spot index |
| `option_tickers` | `public/ticker?instrument_name=...`, once per option | 946, 790 | Deribit's own bid and ask IVs, greeks, quote sizes |

Booleans must be sent in lowercase (`expired=false`). The index name is read from the
instruments' `price_index` field (`btc_usd`, `eth_usd`) rather than built from the currency.

## Envelope, errors and rate limits

* A success is `{"jsonrpc", "result", "usIn", "usOut", "usDiff", "testnet"}`. `usIn` and `usOut`
  are the server receive and send times in microseconds since the epoch.
* An error is sent with HTTP 400 and `{"error": {"code", "message", "data"}}`. Observed:
  `-32602 Invalid params` (with `data.reason` and `data.param`) and `-32601 Method not found`.
* Responses carry no rate limit headers (the API sits behind Cloudflare).
* [Documented limits](https://docs.deribit.com/articles/rate-limits): non matching engine requests
  default to 20 per second with bursts of 100, `public/get_instruments` to 1 per second with
  bursts of 50, and unauthenticated requests are limited per IP at an unpublished level. An
  exhausted limit returns `too_many_requests`, code 10028. The client throttles itself to half
  the documented rates and retries 10028, HTTP 429, 5xx, timeouts and dropped connections with
  exponential backoff and jitter. Other errors fail immediately.
* In practice a full ticker sweep ran at about 4 requests per second, limited by round trip
  latency, so the throttle never engaged: 946 BTC tickers took about 4 minutes.

## Fields the pipeline relies on

### `option_book`

| Field | Meaning and unit | Observations |
|---|---|---|
| `bid_price`, `ask_price` | best quotes, in BTC per option | a missing bid is `null`, never 0 (53 of 946 BTC options); asks were always present |
| `mid_price` | Deribit's mid, BTC | `null` whenever the bid is missing |
| `mark_price` | Deribit's mark, BTC | 0.0 for 9 far OTM BTC options expiring within a day, which still report a `mark_iv` |
| `mark_iv` | mark implied volatility, **in percent** | 40.71 means 40.71% |
| `underlying_price` | forward used by Deribit for that expiry, USD | equals the matching future's mark price within a few dollars; takes 2 or 3 slightly different values within one expiry because rows are computed milliseconds apart |
| `underlying_index` | the future the expiry is written on, e.g. `BTC-27NOV26` | a listed future for every expiry in our samples (no synthetic underlyings) |
| `open_interest` | contracts; `contract_size` is 1 coin | |
| `interest_rate` | | always 0.0 |
| `creation_timestamp` | when Deribit computed the row, ms since the epoch | all rows of one response lie within 20 ms, and 0.07 to 0.9 s before the response: the summary is a cache, so time to expiry is measured from this field |

### `option_instruments`

* Names look like `BTC-6OCT26-75000-C`: currency, expiry as day (no leading zero), month and two
  digit year, integer strike, `C` or `P`. The strike token always equals the `strike` field.
* `expiration_timestamp` (ms) is always 08:00 UTC. `settlement_period` is `day`, `week` or
  `month` (quarterly expiries are labelled `month`).
* `instrument_type` is `reversed` (inverse, coin settled) and `settlement_currency` is the coin;
  the fetch rejects anything else.
* `tick_size` is 0.0001 with `tick_size_steps` `[{"tick_size": 0.0005, "above_price": 0.005}]`:
  prices above 0.005 BTC move in steps of 0.0005.

### `future_book` and `index`

Futures are quoted in USD and include `BTC-PERPETUAL`, the only row with `current_funding` and
`funding_8h`. The index response is `{"index_price", "estimated_delivery_price"}`.

### `option_tickers`

* A missing bid is `best_bid_price: 0.0` here, not `null` as in the book summary. Cleaning must
  treat both as "no bid".
* `bid_iv`, `ask_iv` and `mark_iv` are in percent. `bid_iv` is 0.0 for 212 of 946 BTC options:
  51 have no bid, and the other 161 are ITM options bid at or below intrinsic value, where no
  implied volatility exists.
* `greeks.vega` is in USD per vol point: it matches Black 76 vega on `underlying_price` divided
  by 100 to within 0.05% (median ratio 1.0000).
* `timestamp` is the instrument's last update, which can predate the request.

## Pricing convention

Deribit prices these inverse options with undiscounted Black 76 on `underlying_price`, zero rates
and an ACT/365 year:

    price in BTC = Black76(F = underlying_price, K, T, sigma) / F

so the USD forward premium is the BTC price times the forward of that expiry. Evidence:

* Inverting the ticker's `best_bid_price` and `best_ask_price` this way reproduces Deribit's
  `bid_iv` and `ask_iv` to the two decimals reported.
* Inverting `mark_price` for OTM options reproduces `mark_iv` with a median error of +0.0025 vol
  points (5th to 95th percentile about ±0.06) using ACT/365 from `creation_timestamp`. ACT/365.25
  is measurably worse (median +0.017).
* Converting with the spot index instead of the forward produced errors of up to 80 vol points.

The formal validation lives in M3 (implied volatility).

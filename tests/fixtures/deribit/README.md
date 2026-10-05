# Deribit response fixtures

Real responses from Deribit's public API v2 (production), captured with curl on 2026-10-05
between 22:06 and 22:17 UTC.

* List results are trimmed to seven BTC options and three futures. The options were chosen to
  cover the cases the code must handle: an at the money call and put expiring within a day, a
  call and put pair expiring in December, a far out of the money call, and options with no bid
  (`bid_price` null) and a mark price of 0.
* Every remaining row and every envelope field (`usIn`, `usOut`, `usDiff`, `testnet`) is
  unmodified.
* `error_*.json` are the real error bodies Deribit sends with HTTP 400.

Payloads that cannot be captured politely, such as the rate limit error (code 10028), are built
inside the tests that need them.

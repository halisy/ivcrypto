"""Assemble one raw market snapshot for a currency from Deribit's public API.

A snapshot is the set of raw API rows the rest of the pipeline needs, with field
names and values exactly as Deribit returned them:

==================  ===========================================  ===============================
table               endpoint                                     used for
==================  ===========================================  ===============================
option_instruments  public/get_instruments (kind=option)         strikes, expiry times, ticks
option_book         public/get_book_summary_by_currency          bid, ask, mark, mark IV,
                    (kind=option)                                underlying price, open interest
future_book         public/get_book_summary_by_currency          forward diagnostics
                    (kind=future)
index               public/get_index_price                       spot index level
option_tickers      public/ticker, once per option (opt in)      Deribit's own bid and ask IVs,
                                                                 greeks, quote sizes
==================  ===========================================  ===============================

All option quotes come from one bulk call, so the chain is priced at nearly one
instant (in the responses we inspected, every row was computed within 10 ms).
Fetching one ticker per option takes minutes at a polite request rate, so the
tickers are only used for validation, never to build the surface.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ivcrypto.data.client import DeribitAPIError, RpcResult

logger = logging.getLogger(__name__)

OPTION_INSTRUMENTS = "option_instruments"
OPTION_BOOK = "option_book"
FUTURE_BOOK = "future_book"
INDEX = "index"
OPTION_TICKERS = "option_tickers"

INVERSE_INSTRUMENT_TYPE = "reversed"
"""Deribit's ``instrument_type`` for coin settled (inverse) options."""

REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    OPTION_INSTRUMENTS: frozenset(
        {
            "instrument_name",
            "expiration_timestamp",
            "strike",
            "option_type",
            "tick_size",
            "tick_size_steps",
            "contract_size",
            "instrument_type",
            "settlement_currency",
            "price_index",
        }
    ),
    OPTION_BOOK: frozenset(
        {
            "instrument_name",
            "bid_price",
            "ask_price",
            "mid_price",
            "mark_price",
            "mark_iv",
            "underlying_price",
            "underlying_index",
            "open_interest",
            "interest_rate",
            "creation_timestamp",
        }
    ),
    FUTURE_BOOK: frozenset(
        {"instrument_name", "bid_price", "ask_price", "mark_price", "creation_timestamp"}
    ),
    INDEX: frozenset({"index_price"}),
    OPTION_TICKERS: frozenset(
        {
            "instrument_name",
            "timestamp",
            "best_bid_price",
            "best_ask_price",
            "best_bid_amount",
            "best_ask_amount",
            "bid_iv",
            "ask_iv",
            "mark_iv",
            "mark_price",
            "underlying_price",
            "greeks",
        }
    ),
}
"""Fields the pipeline relies on, per table. Values may be null; the keys must exist."""


class SnapshotError(RuntimeError):
    """The API responses cannot be turned into a usable snapshot."""


class RpcCaller(Protocol):
    """What ``fetch_snapshot`` needs from a client (``DeribitClient`` satisfies it)."""

    base_url: str

    def call(self, method: str, **params: Any) -> RpcResult: ...


@dataclass(frozen=True)
class CallRecord:
    """Provenance of one table: the call that produced it and the server timestamps."""

    table: str
    method: str
    params: Mapping[str, Any]
    rows: int
    us_in: int | None
    us_out: int | None
    testnet: bool


@dataclass
class RawSnapshot:
    """Raw rows per table, the call records, and the snapshot reference time.

    ``timestamp`` is the server send time of the ``option_book`` call: the moment
    the option quotes were taken, and the reference for time to expiry.
    """

    currency: str
    timestamp: datetime
    source: str
    tables: dict[str, list[dict[str, Any]]]
    calls: list[CallRecord]
    notes: dict[str, Any] = field(default_factory=dict)


def fetch_snapshot(client: RpcCaller, currency: str, *, tickers: bool = False) -> RawSnapshot:
    """Fetch the option chain of ``currency`` (``BTC`` or ``ETH``) and its context."""
    currency = currency.upper()
    tables: dict[str, list[dict[str, Any]]] = {}
    calls: list[CallRecord] = []
    notes: dict[str, Any] = {}

    def record(table: str, result: RpcResult) -> list[dict[str, Any]]:
        rows = _as_rows(table, result.result)
        check_required_fields(table, rows)
        tables[table] = rows
        calls.append(
            CallRecord(
                table=table,
                method=result.method,
                params=dict(result.params),
                rows=len(rows),
                us_in=result.us_in,
                us_out=result.us_out,
                testnet=result.testnet,
            )
        )
        return rows

    instruments = record(
        OPTION_INSTRUMENTS,
        client.call("public/get_instruments", currency=currency, kind="option", expired=False),
    )
    check_coin_settled(instruments, currency)
    index_name = single_price_index(instruments)

    # Quotes, futures and index back to back, so they describe the same moment.
    book_result = client.call(
        "public/get_book_summary_by_currency", currency=currency, kind="option"
    )
    book = record(OPTION_BOOK, book_result)
    record(
        FUTURE_BOOK,
        client.call("public/get_book_summary_by_currency", currency=currency, kind="future"),
    )
    record(INDEX, client.call("public/get_index_price", index_name=index_name))
    timestamp = server_time(book_result.us_out)

    listed = {row["instrument_name"] for row in instruments}
    quoted = {row["instrument_name"] for row in book}
    if listed != quoted:
        notes["instruments_without_quotes"] = sorted(listed - quoted)
        notes["quotes_without_instrument"] = sorted(quoted - listed)
        logger.warning(
            "%d listed options have no book summary and %d quoted options are not listed "
            "(listings changed between calls?)",
            len(listed - quoted),
            len(quoted - listed),
        )

    if tickers:
        _fetch_tickers(client, sorted(quoted), tables, calls, notes)

    logger.info(
        "%s snapshot at %s: %s",
        currency,
        timestamp.isoformat(),
        ", ".join(f"{call.table}={call.rows}" for call in calls),
    )
    return RawSnapshot(
        currency=currency,
        timestamp=timestamp,
        source=client.base_url,
        tables=tables,
        calls=calls,
        notes=notes,
    )


def _fetch_tickers(
    client: RpcCaller,
    names: Sequence[str],
    tables: dict[str, list[dict[str, Any]]],
    calls: list[CallRecord],
    notes: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    results: list[RpcResult] = []
    skipped: list[str] = []
    for count, name in enumerate(names, start=1):
        try:
            result = client.call("public/ticker", instrument_name=name)
        except DeribitAPIError as err:  # e.g. the option expired while we were fetching
            logger.warning("ticker for %s skipped: %s", name, err)
            skipped.append(name)
            continue
        rows.append(result.result)
        results.append(result)
        if count % 100 == 0 or count == len(names):
            logger.info("tickers: %d of %d", count, len(names))
    notes["tickers_skipped"] = skipped
    if not results:
        logger.warning("no ticker could be fetched")
        return
    check_required_fields(OPTION_TICKERS, rows)
    tables[OPTION_TICKERS] = rows
    # The window spanned by all ticker calls, whatever order the server times came in.
    times_in = [r.us_in for r in results if r.us_in is not None]
    times_out = [r.us_out for r in results if r.us_out is not None]
    calls.append(
        CallRecord(
            table=OPTION_TICKERS,
            method="public/ticker",
            params={"instrument_name": f"each of the {len(names)} options in {OPTION_BOOK}"},
            rows=len(rows),
            us_in=min(times_in, default=None),
            us_out=max(times_out, default=None),
            testnet=any(r.testnet for r in results),
        )
    )


def check_required_fields(table: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail loudly if Deribit stopped sending a field the pipeline relies on."""
    if not rows:
        raise SnapshotError(f"{table}: the API returned no rows")
    missing = sorted(name for name in REQUIRED_FIELDS[table] if any(name not in r for r in rows))
    if missing:
        raise SnapshotError(
            f"{table}: fields missing from the API response: {missing}. "
            "Deribit may have changed its schema; see docs/deribit_api.md."
        )


def check_coin_settled(instruments: Sequence[Mapping[str, Any]], currency: str) -> None:
    """Only inverse options (premium and settlement in the coin) are supported."""
    other = [
        row["instrument_name"]
        for row in instruments
        if row["instrument_type"] != INVERSE_INSTRUMENT_TYPE
        or row["settlement_currency"] != currency
    ]
    if other:
        raise SnapshotError(
            f"{len(other)} {currency} options are not coin settled inverse options "
            f"(for example {other[:3]}); only those are supported"
        )


def single_price_index(instruments: Sequence[Mapping[str, Any]]) -> str:
    """The index the options are written on, as reported by Deribit (e.g. ``btc_usd``)."""
    names = {row["price_index"] for row in instruments}
    if len(names) != 1:
        raise SnapshotError(f"expected a single price index, got {sorted(names)}")
    return names.pop()


def server_time(us: int | None) -> datetime:
    """Convert Deribit's microsecond server timestamp to an exact UTC datetime."""
    if us is None:
        logger.warning("response carried no server time; using the local clock")
        return datetime.now(UTC)
    seconds, micros = divmod(us, 1_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=micros)


def _as_rows(table: str, result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list) and all(isinstance(row, dict) for row in result):
        return result
    raise SnapshotError(f"{table}: unexpected result type {type(result).__name__}")

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from ivcrypto.data.fetch import (
    FUTURE_BOOK,
    INDEX,
    OPTION_BOOK,
    OPTION_INSTRUMENTS,
    OPTION_TICKERS,
    REQUIRED_FIELDS,
    SnapshotError,
    fetch_snapshot,
    server_time,
)

N_OPTIONS = 7  # options kept in the trimmed fixtures


def test_snapshot_has_raw_tables_and_provenance(fixture_caller):
    caller = fixture_caller()
    raw = fetch_snapshot(caller, "BTC")

    assert raw.currency == "BTC"
    assert raw.source == "https://www.deribit.com/api/v2"
    assert [call.table for call in raw.calls] == [
        OPTION_INSTRUMENTS,
        OPTION_BOOK,
        FUTURE_BOOK,
        INDEX,
    ]
    assert {name: len(rows) for name, rows in raw.tables.items()} == {
        OPTION_INSTRUMENTS: N_OPTIONS,
        OPTION_BOOK: N_OPTIONS,
        FUTURE_BOOK: 3,
        INDEX: 1,
    }
    book_call = raw.calls[1]
    assert raw.timestamp == server_time(book_call.us_out)
    assert all(call.testnet is False for call in raw.calls)
    assert raw.notes == {}


def test_calls_use_documented_parameters(fixture_caller):
    caller = fixture_caller()
    fetch_snapshot(caller, "btc")
    assert caller.calls == [
        ("public/get_instruments", {"currency": "BTC", "kind": "option", "expired": False}),
        ("public/get_book_summary_by_currency", {"currency": "BTC", "kind": "option"}),
        ("public/get_book_summary_by_currency", {"currency": "BTC", "kind": "future"}),
        # The index name comes from the instruments' price_index field, not a guess.
        ("public/get_index_price", {"index_name": "btc_usd"}),
    ]


def test_rows_are_kept_exactly_as_returned(fixture_caller):
    raw = fetch_snapshot(fixture_caller(), "BTC")
    book = {row["instrument_name"]: row for row in raw.tables[OPTION_BOOK]}
    no_bid = book["BTC-6OCT26-95000-C"]
    assert no_bid["bid_price"] is None
    assert no_bid["mid_price"] is None
    assert no_bid["mark_price"] == 0.0
    assert no_bid["mark_iv"] == 50.72
    assert book["BTC-25DEC26-90000-C"]["underlying_index"] == "BTC-25DEC26"


def test_tickers_are_fetched_for_every_quoted_option(fixture_caller):
    raw = fetch_snapshot(fixture_caller(), "BTC", tickers=True)
    tickers = raw.tables[OPTION_TICKERS]
    assert len(tickers) == N_OPTIONS
    assert {t["instrument_name"] for t in tickers} == {
        r["instrument_name"] for r in raw.tables[OPTION_BOOK]
    }
    record = raw.calls[-1]
    assert record.table == OPTION_TICKERS
    assert record.rows == N_OPTIONS
    assert record.us_in is not None
    assert record.us_out is not None
    assert record.us_in <= record.us_out
    assert raw.notes["tickers_skipped"] == []


def test_ticker_errors_are_skipped_and_recorded(fixture_caller):
    def edit(method, payload):
        if (
            method == "public/ticker"
            and payload["result"]["instrument_name"] == "BTC-6OCT26-75000-P"
        ):
            # The real error body Deribit sent for an instrument name it did not accept.
            return {"error": {"code": -32602, "message": "Invalid params"}}
        return payload

    raw = fetch_snapshot(fixture_caller(edit), "BTC", tickers=True)
    assert raw.notes["tickers_skipped"] == ["BTC-6OCT26-75000-P"]
    assert len(raw.tables[OPTION_TICKERS]) == N_OPTIONS - 1


def test_missing_field_fails_loudly(fixture_caller):
    def edit(method, payload):
        if method == "public/get_book_summary_by_currency" and "mark_iv" in payload["result"][0]:
            del payload["result"][0]["mark_iv"]
        return payload

    with pytest.raises(SnapshotError, match="mark_iv"):
        fetch_snapshot(fixture_caller(edit), "BTC")


def test_empty_result_fails(fixture_caller):
    def edit(method, payload):
        if method == "public/get_book_summary_by_currency":
            payload["result"] = []
        return payload

    with pytest.raises(SnapshotError, match="no rows"):
        fetch_snapshot(fixture_caller(edit), "BTC")


def test_linear_options_are_rejected(fixture_caller):
    def edit(method, payload):
        if method == "public/get_instruments":
            payload["result"][0]["instrument_type"] = "linear"
        return payload

    with pytest.raises(SnapshotError, match="coin settled"):
        fetch_snapshot(fixture_caller(edit), "BTC")


def test_listing_changes_between_calls_are_noted(fixture_caller, caplog):
    def edit(method, payload):
        if method == "public/get_instruments":
            payload["result"] = [
                r for r in payload["result"] if r["instrument_name"] != "BTC-6OCT26-86000-P"
            ]
        return payload

    with caplog.at_level(logging.WARNING):
        raw = fetch_snapshot(fixture_caller(edit), "BTC")
    assert raw.notes["quotes_without_instrument"] == ["BTC-6OCT26-86000-P"]
    assert raw.notes["instruments_without_quotes"] == []
    assert "not listed" in caplog.text


def test_required_fields_were_all_observed_in_the_fixtures(fixture_caller):
    # Guards against listing a field here that Deribit never actually sends.
    raw = fetch_snapshot(fixture_caller(), "BTC", tickers=True)
    for table, fields in REQUIRED_FIELDS.items():
        for row in raw.tables[table]:
            assert fields <= row.keys(), table


def test_server_time_is_exact_to_the_microsecond():
    us = 1_791_237_971_008_103
    expected = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=us)
    assert server_time(us) == expected
    assert server_time(us).microsecond == 8103

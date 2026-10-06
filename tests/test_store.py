from __future__ import annotations

import json
import math
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ivcrypto.data import store
from ivcrypto.data.fetch import (
    FUTURE_BOOK,
    OPTION_BOOK,
    OPTION_INSTRUMENTS,
    OPTION_TICKERS,
    fetch_snapshot,
)
from ivcrypto.data.store import (
    MANIFEST,
    find_snapshots,
    load_snapshot,
    rows_to_table,
    snapshot_dirname,
    write_snapshot,
)


@pytest.fixture
def raw(fixture_caller):
    return fetch_snapshot(fixture_caller(), "BTC", tickers=True)


def test_round_trip_preserves_rows(raw, tmp_path):
    path = write_snapshot(raw, tmp_path)
    loaded = load_snapshot(path)

    assert loaded.currency == "BTC"
    assert loaded.timestamp == raw.timestamp
    assert set(loaded.tables) == set(raw.tables)
    for name, rows in raw.tables.items():
        frame = loaded[name]
        assert len(frame) == len(rows)
        assert list(frame.columns) == list(dict.fromkeys(k for row in rows for k in row))

    book = loaded[OPTION_BOOK].set_index("instrument_name")
    original = {row["instrument_name"]: row for row in raw.tables[OPTION_BOOK]}
    for name, row in original.items():
        for column in ("ask_price", "mark_price", "mark_iv", "underlying_price", "open_interest"):
            assert book.loc[name, column] == row[column]
        if row["bid_price"] is None:
            assert math.isnan(book.loc[name, "bid_price"])
        else:
            assert book.loc[name, "bid_price"] == row["bid_price"]


def test_nested_and_sparse_fields_survive(raw, tmp_path):
    loaded = load_snapshot(write_snapshot(raw, tmp_path))
    steps = loaded[OPTION_INSTRUMENTS]["tick_size_steps"].iloc[0]
    assert list(steps) == [{"tick_size": 0.0005, "above_price": 0.005}]
    greeks = loaded[OPTION_TICKERS]["greeks"].iloc[0]
    assert set(greeks) == {"delta", "gamma", "vega", "theta", "rho"}
    # Only the perpetual reports funding; the column exists and is null elsewhere.
    futures = loaded[FUTURE_BOOK].set_index("instrument_name")
    assert not math.isnan(futures.loc["BTC-PERPETUAL", "current_funding"])
    assert math.isnan(futures.loc["BTC-25DEC26", "current_funding"])


def test_parquet_files_carry_timestamp_and_provenance(raw, tmp_path):
    path = write_snapshot(raw, tmp_path)
    metadata = pq.read_schema(path / f"{OPTION_BOOK}.parquet").metadata
    assert datetime.fromisoformat(metadata[b"ivcrypto.snapshot_utc"].decode()) == raw.timestamp
    assert metadata[b"ivcrypto.method"] == b"public/get_book_summary_by_currency"
    assert json.loads(metadata[b"ivcrypto.params"]) == {"currency": "BTC", "kind": "option"}
    assert metadata[b"ivcrypto.currency"] == b"BTC"


def test_manifest_records_every_call(raw, tmp_path):
    path = write_snapshot(raw, tmp_path)
    manifest = json.loads((path / MANIFEST).read_text())
    assert manifest["schema_version"] == store.SCHEMA_VERSION
    assert manifest["currency"] == "BTC"
    assert manifest["snapshot_utc"].endswith("Z")
    assert set(manifest["tables"]) == set(raw.tables)
    entry = manifest["tables"][OPTION_BOOK]
    assert entry["rows"] == len(raw.tables[OPTION_BOOK])
    assert entry["server_us_out"] == raw.calls[1].us_out
    assert entry["server_utc_out"] == manifest["snapshot_utc"]
    assert manifest["notes"] == {"tickers_skipped": []}


def test_directory_is_named_after_the_utc_timestamp(raw, tmp_path):
    path = write_snapshot(raw, tmp_path)
    assert path == tmp_path / "BTC" / snapshot_dirname(raw.timestamp)
    assert snapshot_dirname(raw.timestamp) == raw.timestamp.strftime("%Y%m%dT%H%M%SZ")


def test_existing_snapshot_is_never_overwritten(raw, tmp_path):
    write_snapshot(raw, tmp_path)
    with pytest.raises(FileExistsError):
        write_snapshot(raw, tmp_path)
    assert [p.name for p in (tmp_path / "BTC").iterdir()] == [snapshot_dirname(raw.timestamp)]


def test_failed_write_leaves_nothing_behind(raw, tmp_path, monkeypatch):
    def broken_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store.pq, "write_table", broken_write)
    with pytest.raises(OSError, match="disk full"):
        write_snapshot(raw, tmp_path)
    assert list((tmp_path / "BTC").iterdir()) == []


def test_rows_to_table_takes_the_union_of_keys():
    table = rows_to_table([{"a": 1}, {"a": 2.5, "b": "x"}])
    assert table.column_names == ["a", "b"]
    assert table.schema.field("a").type == pa.float64()
    assert table.column("b").to_pylist() == [None, "x"]


def test_rows_to_table_keeps_unmixable_values_as_json_text():
    table = rows_to_table([{"a": 1}, {"a": "one"}, {"a": None}])
    assert table.column("a").to_pylist() == ["1", '"one"', None]


def test_find_snapshots_lists_oldest_first(raw, tmp_path):
    first = write_snapshot(raw, tmp_path)
    raw.timestamp = raw.timestamp.replace(year=raw.timestamp.year + 1)
    second = write_snapshot(raw, tmp_path)
    assert find_snapshots(tmp_path) == [first, second]
    assert find_snapshots(tmp_path, "btc") == [first, second]
    assert find_snapshots(tmp_path, "ETH") == []


def test_unknown_schema_version_is_rejected(raw, tmp_path):
    path = write_snapshot(raw, tmp_path)
    manifest = json.loads((path / MANIFEST).read_text())
    manifest["schema_version"] = 99
    (path / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema version 99"):
        load_snapshot(path)


def test_missing_table_has_a_helpful_error(raw, tmp_path):
    loaded = load_snapshot(write_snapshot(raw, tmp_path))
    with pytest.raises(KeyError, match="no table 'nope'"):
        loaded["nope"]

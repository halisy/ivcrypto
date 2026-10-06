"""Properties of the committed real snapshots that later milestones rely on."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from ivcrypto.data.fetch import (
    FUTURE_BOOK,
    INDEX,
    INVERSE_INSTRUMENT_TYPE,
    OPTION_BOOK,
    OPTION_INSTRUMENTS,
    OPTION_TICKERS,
    REQUIRED_FIELDS,
)


@pytest.fixture(params=["BTC", "ETH"])
def sample(request, btc_sample, eth_sample):
    return {"BTC": btc_sample, "ETH": eth_sample}[request.param]


def test_all_tables_and_required_fields_are_present(sample):
    assert set(sample.tables) == {
        OPTION_INSTRUMENTS,
        OPTION_BOOK,
        FUTURE_BOOK,
        INDEX,
        OPTION_TICKERS,
    }
    for table, fields in REQUIRED_FIELDS.items():
        assert fields <= set(sample[table].columns), table
        assert len(sample[table]) == sample.manifest["tables"][table]["rows"]


def test_parquet_metadata_matches_manifest(sample):
    for table, entry in sample.manifest["tables"].items():
        metadata = pq.read_schema(sample.path / entry["file"]).metadata
        assert metadata[b"ivcrypto.snapshot_utc"].decode() == sample.manifest["snapshot_utc"]
        assert metadata[b"ivcrypto.table"].decode() == table


def test_instruments_and_quotes_cover_the_same_options(sample):
    listed = set(sample[OPTION_INSTRUMENTS]["instrument_name"])
    quoted = set(sample[OPTION_BOOK]["instrument_name"])
    tickers = set(sample[OPTION_TICKERS]["instrument_name"])
    assert listed == quoted == tickers
    assert sample.manifest["notes"] == {"tickers_skipped": []}


def test_all_options_are_coin_settled_inverse_options(sample):
    instruments = sample[OPTION_INSTRUMENTS]
    assert (instruments["instrument_type"] == INVERSE_INSTRUMENT_TYPE).all()
    assert (instruments["settlement_currency"] == sample.currency).all()


def test_quotes_were_computed_together_just_before_the_snapshot(sample):
    created = sample[OPTION_BOOK]["creation_timestamp"]
    snapshot_ms = sample.timestamp.timestamp() * 1000
    assert created.max() - created.min() < 1_000
    assert created.max() <= snapshot_ms


def test_every_option_expiry_has_a_listed_future(sample):
    # True for these snapshots (no synthetic underlyings); M2 reports it per expiry.
    futures = set(sample[FUTURE_BOOK]["instrument_name"])
    assert set(sample[OPTION_BOOK]["underlying_index"]) <= futures


def test_missing_bids_are_null_in_the_book_summary(sample):
    book = sample[OPTION_BOOK]
    assert book["bid_price"].isna().any()
    assert (book["bid_price"].dropna() > 0).all()
    assert book["ask_price"].notna().all()

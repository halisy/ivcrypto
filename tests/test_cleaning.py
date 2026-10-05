"""Cleaning on the committed real snapshots."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ivcrypto.cleaning import FILTERS, append_filter, clean_snapshot
from ivcrypto.config import CleaningConfig
from ivcrypto.instruments import SECONDS_PER_YEAR


@pytest.fixture(scope="module")
def clean_btc(btc_sample):
    return clean_snapshot(btc_sample)


def with_table(snapshot, name, table):
    return replace(snapshot, tables={**snapshot.tables, name: table})


def test_filter_report_adds_up(clean_btc, btc_sample):
    report = clean_btc.filters.set_index("step")
    assert report.loc["options in snapshot", "remaining"] == len(btc_sample["option_book"])
    remaining = clean_btc.filters["remaining"].to_numpy()
    removed = clean_btc.filters["removed"].to_numpy()
    np.testing.assert_array_equal(remaining[1:], remaining[:-1] - removed[1:])
    counts = clean_btc.quotes["removed_by"].value_counts()
    for step, _ in FILTERS:
        assert counts.get(step, 0) == report.loc[step, "removed"], step
    assert len(clean_btc.kept) == remaining[-1]


def test_filters_match_an_independent_recount(clean_btc, btc_sample):
    book = btc_sample["option_book"]
    meta = btc_sample["option_instruments"].set_index("instrument_name")
    valuation_ms = book["creation_timestamp"].max()
    days = (meta.loc[book["instrument_name"], "expiration_timestamp"].to_numpy() - valuation_ms) / (
        86_400_000
    )
    near = days < CleaningConfig().min_days_to_expiry
    assert (clean_btc.quotes["removed_by"] == "expiry_too_close").sum() == near.sum()
    no_bid = (book["bid_price"].isna() | (book["bid_price"] <= 0)).to_numpy() & ~near
    assert (clean_btc.quotes["removed_by"] == "no_bid").sum() == no_bid.sum()


def test_kept_quotes_pass_every_filter(clean_btc):
    kept = clean_btc.kept
    config = CleaningConfig()
    assert (kept["days"] >= config.min_days_to_expiry).all()
    assert (kept["bid_btc"] > 0).all()
    assert (kept["ask_btc"] > kept["bid_btc"]).all()
    assert (kept["rel_spread"] <= config.max_relative_spread).all()
    assert kept["otm"].all()
    assert (kept.loc[kept["option_type"] == "put", "k"] < 0).all()
    assert (kept.loc[kept["option_type"] == "call", "k"] >= 0).all()


def test_coin_prices_are_converted_with_the_expiry_forward(clean_btc):
    quotes = clean_btc.quotes
    assert (quotes.groupby("expiry_code")["forward"].nunique() == 1).all()
    for side in ("bid", "ask", "mid", "mark"):
        np.testing.assert_allclose(
            quotes[f"{side}_usd"], quotes[f"{side}_btc"] * quotes["forward"], equal_nan=True
        )
    np.testing.assert_allclose(quotes["k"], np.log(quotes["strike"] / quotes["forward"]))


def test_deribit_mark_iv_is_converted_from_percent(clean_btc, btc_sample):
    raw = btc_sample["option_book"].set_index("instrument_name")["mark_iv"]
    quotes = clean_btc.quotes.set_index("instrument_name")
    np.testing.assert_allclose(quotes["deribit_mark_iv"], raw.loc[quotes.index] / 100)
    assert quotes["deribit_mark_iv"].between(0.05, 5.0).all()


def test_time_to_expiry_is_act365_from_the_valuation_time(clean_btc, btc_sample):
    quotes = clean_btc.quotes
    expected_valuation = pd.Timestamp(
        btc_sample["option_book"]["creation_timestamp"].max(), unit="ms"
    )
    assert pd.Timestamp(clean_btc.valuation_time).tz_convert(None) == expected_valuation
    seconds = (quotes["expiry"] - pd.Timestamp(clean_btc.valuation_time)).dt.total_seconds()
    np.testing.assert_allclose(quotes["T"], seconds / SECONDS_PER_YEAR, rtol=1e-12)
    np.testing.assert_allclose(quotes["days"], quotes["T"] * 365)
    assert (quotes["expiry"].dt.hour == 8).all()


def test_zero_bids_count_as_missing_like_nulls(btc_sample):
    book = btc_sample["option_book"].copy()
    clean = clean_snapshot(btc_sample)
    target = clean.kept["instrument_name"].iloc[0]
    book.loc[book["instrument_name"] == target, "bid_price"] = 0.0  # the ticker convention
    modified = clean_snapshot(with_table(btc_sample, "option_book", book))
    removed_by = modified.quotes.set_index("instrument_name").loc[target, "removed_by"]
    assert removed_by == "no_bid"


def test_thresholds_are_configurable(btc_sample):
    loose = clean_snapshot(
        btc_sample, CleaningConfig(min_days_to_expiry=0.0, max_relative_spread=5)
    )
    strict = clean_snapshot(
        btc_sample, CleaningConfig(min_days_to_expiry=30.0, max_relative_spread=0.1)
    )
    counts = {
        name: frame["removed_by"].value_counts()
        for name, frame in (("loose", loose.quotes), ("strict", strict.quotes))
    }
    assert counts["loose"].get("expiry_too_close", 0) == 0
    assert counts["strict"]["expiry_too_close"] > 120
    assert counts["loose"].get("wide_spread", 0) < counts["strict"]["wide_spread"]
    assert loose.kept["days"].min() < 1.0
    assert strict.kept["days"].min() >= 30.0


def test_underlying_method_uses_deribits_forward(btc_sample):
    clean = clean_snapshot(btc_sample, CleaningConfig(forward_method="underlying"))
    forwards = clean.forwards.set_index("expiry_code")
    assert (forwards["forward_source"] == "underlying").all()
    np.testing.assert_allclose(forwards["forward"], forwards["underlying_price"])


def test_parity_forwards_on_the_sample(clean_btc):
    forwards = clean_btc.forwards
    assert (forwards["forward_source"] == "parity").all()
    # Documented properties of this snapshot: the option implied forward sits within a few
    # basis points of the futures, and C - P implies a coin discount factor of 1.
    assert forwards["parity_vs_underlying_bps"].abs().max() < 10
    assert forwards["regression_discount"].between(0.999, 1.001).all()
    assert forwards["parity_dispersion_bps"].max() < 5


def test_append_filter_extends_the_log(clean_btc):
    kept = clean_btc.kept
    mask = clean_btc.quotes["instrument_name"] == kept["instrument_name"].iloc[0]
    extended = append_filter(clean_btc, "test_step", "a later stage", mask)
    assert extended.filters["step"].iloc[-1] == "test_step"
    assert extended.filters["removed"].iloc[-1] == 1
    assert len(extended.kept) == len(kept) - 1
    assert len(clean_btc.kept) == len(kept)  # the original is untouched


def test_metadata_disagreement_is_an_error(btc_sample):
    meta = btc_sample["option_instruments"].copy()
    meta.loc[0, "strike"] = meta.loc[0, "strike"] + 1.0
    with pytest.raises(ValueError, match="disagree with the metadata on strike"):
        clean_snapshot(with_table(btc_sample, "option_instruments", meta))


def test_eth_sample_cleans(eth_sample):
    clean = clean_snapshot(eth_sample)
    assert clean.currency == "ETH"
    assert len(clean.kept) > 200
    assert (clean.forwards["forward_source"] == "parity").all()

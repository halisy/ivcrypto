from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from ivcrypto.instruments import (
    MS_PER_YEAR,
    parse_option_name,
    parse_option_names,
    to_epoch_ms,
    year_fraction,
)


@pytest.mark.parametrize(
    ("name", "currency", "expiry", "strike", "option_type"),
    [
        ("BTC-6OCT26-75000-C", "BTC", datetime(2026, 10, 6, 8, tzinfo=UTC), 75000.0, "call"),
        ("BTC-27NOV26-100000-P", "BTC", datetime(2026, 11, 27, 8, tzinfo=UTC), 100000.0, "put"),
        ("ETH-24SEP27-2500-C", "ETH", datetime(2027, 9, 24, 8, tzinfo=UTC), 2500.0, "call"),
        ("ETH-1JAN27-1-P", "ETH", datetime(2027, 1, 1, 8, tzinfo=UTC), 1.0, "put"),
    ],
)
def test_parse_valid_names(name, currency, expiry, strike, option_type):
    parsed = parse_option_name(name)
    assert parsed.currency == currency
    assert parsed.expiry == expiry
    assert parsed.strike == strike
    assert parsed.option_type == option_type
    assert parsed.expiry_code == name.split("-")[1]


@pytest.mark.parametrize(
    "name",
    [
        "BTC-PERPETUAL",
        "BTC-27NOV26",
        "BTC_USDC-27NOV26-100000-C",  # linear USDC options are out of scope
        "BTC-27NOV26-1d5-C",
        "btc-27NOV26-100000-C",
        "BTC-27XYZ26-100000-C",
        "BTC-31FEB26-100000-C",
        "BTC-27NOV26-100000-X",
    ],
)
def test_parse_rejects_anything_else(name):
    with pytest.raises(ValueError, match=r"(not a coin settled|unknown month|invalid expiry)"):
        parse_option_name(name)


def test_vectorized_parse_matches_scalar():
    names = ["BTC-6OCT26-75000-C", "BTC-25DEC26-90000-P"]
    frame = parse_option_names(names)
    assert list(frame["instrument_name"]) == names
    assert list(frame["option_type"]) == ["call", "put"]
    assert frame["expiry"].iloc[1] == pd.Timestamp("2026-12-25 08:00", tz="UTC")
    assert frame["strike"].dtype == np.float64


def test_year_fraction_is_act_365():
    start = to_epoch_ms(datetime(2026, 1, 1, 8, tzinfo=UTC))
    assert year_fraction(start, start + 365 * 86_400_000) == pytest.approx(1.0)
    assert year_fraction(start, start + 366 * 86_400_000) == pytest.approx(366 / 365)
    ends = np.array([start + 86_400_000, start + 2 * 86_400_000])
    np.testing.assert_allclose(year_fraction(start, ends), [1 / 365, 2 / 365])
    assert MS_PER_YEAR == 365 * 86_400_000


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError, match="timezone aware"):
        to_epoch_ms(datetime(2026, 1, 1))


@pytest.mark.parametrize("currency", ["BTC", "ETH"])
def test_every_sample_name_agrees_with_its_metadata(currency, btc_sample, eth_sample):
    sample = {"BTC": btc_sample, "ETH": eth_sample}[currency]
    meta = sample["option_instruments"]
    parsed = parse_option_names(meta["instrument_name"])
    expiry_ms = (parsed["expiry"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    assert (expiry_ms.to_numpy() == meta["expiration_timestamp"].to_numpy()).all()
    assert (parsed["strike"].to_numpy() == meta["strike"].to_numpy()).all()
    assert (parsed["option_type"].to_numpy() == meta["option_type"].to_numpy()).all()
    assert (parsed["expiry"].dt.hour == 8).all()

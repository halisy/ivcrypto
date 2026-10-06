"""Forward estimation on synthetic inverse option quotes (synthetic data only in tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ivcrypto.config import CleaningConfig
from ivcrypto.forwards import (
    estimate_forwards,
    parity_forward,
    parity_pairs,
    weighted_median,
)

F_TRUE = 100_000.0
T = 0.25


def coin_prices(strike, option_type, discount=1.0, forward=F_TRUE):
    """Inverse option prices in coins: discount * Black76(F, K, T, sigma(k)) / F."""
    strike = np.asarray(strike, dtype=float)
    k = np.log(strike / forward)
    sigma = 0.55 + 0.4 * k**2 - 0.1 * k  # a smile, so the test is not a flat vol world
    sd = sigma * np.sqrt(T)
    d1 = (-k + 0.5 * sd**2) / sd
    d2 = d1 - sd
    call = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    usd = np.where(np.asarray(option_type) == "call", call, call - (forward - strike))
    return discount * usd / forward


def synthetic_quotes(strikes, *, discount=1.0, noise=0.0, seed=0, underlying=F_TRUE * 1.0003):
    """Calls and puts with spreads that widen with price, like Deribit's."""
    rng = np.random.default_rng(seed)
    rows = []
    for option_type in ("call", "put"):
        fair = coin_prices(strikes, [option_type] * len(strikes), discount)
        half = np.maximum(0.0001, 0.01 * fair)
        mid = fair + noise * half * rng.uniform(-1, 1, size=fair.shape)
        for strike, m, h in zip(strikes, mid, half, strict=True):
            rows.append(
                {"strike": strike, "option_type": option_type, "bid_btc": m - h, "ask_btc": m + h}
            )
    quotes = pd.DataFrame(rows)
    quotes["expiry_code"] = "TEST"
    quotes["expiry"] = pd.Timestamp("2027-01-01 08:00", tz="UTC")
    quotes["T"] = T
    quotes["days"] = T * 365
    quotes["underlying_index"] = "BTC-TEST"
    quotes["underlying_price"] = underlying
    return quotes


STRIKES = np.arange(70_000.0, 132_000.0, 2_000.0)


def test_weighted_median():
    assert weighted_median([3.0, 1.0, 2.0], [1, 1, 1]) == 2.0
    assert weighted_median([1.0, 2.0, 3.0], [10, 1, 1]) == 1.0
    assert weighted_median([1.0, 2.0, 3.0], [1, 1, 10]) == 3.0
    with pytest.raises(ValueError, match="weighted_median"):
        weighted_median([], [])


def test_every_pair_recovers_the_forward_from_exact_mids():
    pairs = parity_pairs(synthetic_quotes(STRIKES))
    assert len(pairs) == len(STRIKES)
    np.testing.assert_allclose(pairs["forward"], F_TRUE, rtol=1e-12)


def test_exact_mids_give_the_forward_with_zero_scatter():
    estimate = parity_forward(parity_pairs(synthetic_quotes(STRIKES)), 8, reference=F_TRUE)
    assert estimate.forward == pytest.approx(F_TRUE, rel=1e-12)
    assert estimate.pairs_used == 8
    assert estimate.pairs_trimmed == 0
    assert estimate.scatter_bps == pytest.approx(0.0, abs=1e-6)


def test_pairs_nearest_the_money_are_balanced():
    pairs = parity_pairs(synthetic_quotes(STRIKES))
    distance = np.abs(np.log(pairs["strike"] / F_TRUE))
    chosen = pairs.iloc[np.argsort(distance, kind="stable")[:8]]["strike"]
    assert (chosen < F_TRUE).sum() >= 3
    assert (chosen > F_TRUE).sum() >= 3


def test_noisy_quotes_stay_within_their_precision():
    pairs = parity_pairs(synthetic_quotes(STRIKES, noise=1.0, seed=3))
    estimate = parity_forward(pairs, 8, reference=F_TRUE)
    distance = np.abs(np.log(pairs["strike"] / F_TRUE))
    near = pairs.iloc[np.argsort(distance, kind="stable")[:8]]
    assert abs(estimate.forward - F_TRUE) <= near["forward_err"].max()
    assert 0 < estimate.se_bps < estimate.scatter_bps < 25


def test_a_stale_quote_is_trimmed():
    quotes = synthetic_quotes(STRIKES)
    stale = (quotes["strike"] == 98_000.0) & (quotes["option_type"] == "put")
    quotes.loc[stale, ["bid_btc", "ask_btc"]] += 0.01  # a put left 1% of a coin too high
    estimate = parity_forward(parity_pairs(quotes), 8, reference=F_TRUE)
    assert estimate.pairs_trimmed == 1
    assert estimate.forward == pytest.approx(F_TRUE, rel=1e-12)


def test_regression_recovers_a_nonzero_coin_rate():
    discount = 0.98
    pairs = parity_pairs(synthetic_quotes(STRIKES, discount=discount))
    estimate = parity_forward(pairs, 8, reference=F_TRUE)
    assert estimate.regression_discount == pytest.approx(discount, rel=1e-9)
    assert estimate.regression_forward == pytest.approx(F_TRUE, rel=1e-9)
    # Assuming a zero rate biases each pair by about (1 - D)(1 - K/F) / (K/F), which
    # vanishes at the money: the estimate from strikes near the money stays close.
    assert abs(estimate.forward / F_TRUE - 1) < 1e-3


def test_one_sided_quotes_are_not_paired():
    quotes = synthetic_quotes(STRIKES)
    quotes.loc[(quotes["strike"] == 100_000.0) & (quotes["option_type"] == "put"), "bid_btc"] = (
        np.nan
    )
    quotes.loc[(quotes["strike"] == 102_000.0) & (quotes["option_type"] == "call"), "ask_btc"] = 0.0
    pairs = parity_pairs(quotes)
    assert 100_000.0 not in set(pairs["strike"])
    assert 102_000.0 not in set(pairs["strike"])
    assert len(pairs) == len(STRIKES) - 2


def test_parity_forward_is_used_when_pairs_agree():
    forwards = estimate_forwards(synthetic_quotes(STRIKES), None, CleaningConfig())
    row = forwards.iloc[0]
    assert row["forward_source"] == "parity"
    assert row["forward"] == pytest.approx(F_TRUE, rel=1e-12)
    assert row["fallback_reason"] is None
    assert row["parity_vs_underlying_bps"] == pytest.approx(1e4 * (1 / 1.0003 - 1), rel=1e-6)


def test_fallback_when_too_few_pairs():
    quotes = synthetic_quotes(np.array([100_000.0]))
    row = estimate_forwards(quotes, None, CleaningConfig()).iloc[0]
    assert row["forward_source"] == "underlying"
    assert row["forward"] == pytest.approx(F_TRUE * 1.0003)
    assert "1 usable" in row["fallback_reason"]


def test_fallback_when_pairs_disagree():
    quotes = synthetic_quotes(STRIKES)
    rng = np.random.default_rng(7)
    shift = rng.normal(0, 0.01, size=len(quotes))  # stale quotes, about 1% of the coin
    quotes["bid_btc"] += shift
    quotes["ask_btc"] += shift
    row = estimate_forwards(quotes, None, CleaningConfig()).iloc[0]
    assert row["forward_source"] == "underlying"
    assert "scatter" in row["fallback_reason"]


def test_underlying_method_keeps_parity_as_a_diagnostic():
    config = CleaningConfig(forward_method="underlying")
    row = estimate_forwards(synthetic_quotes(STRIKES), None, config).iloc[0]
    assert row["forward_source"] == "underlying"
    assert row["forward"] == pytest.approx(F_TRUE * 1.0003)
    assert row["parity_forward"] == pytest.approx(F_TRUE, rel=1e-12)
    assert row["fallback_reason"] is None


def test_futures_quotes_are_attached():
    futures = pd.DataFrame(
        {
            "instrument_name": ["BTC-TEST"],
            "bid_price": [100_020.0],
            "ask_price": [100_040.0],
            "mark_price": [100_030.0],
        }
    )
    row = estimate_forwards(synthetic_quotes(STRIKES), futures, CleaningConfig()).iloc[0]
    assert (row["future_bid"], row["future_ask"], row["future_mark"]) == (
        100_020.0,
        100_040.0,
        100_030.0,
    )

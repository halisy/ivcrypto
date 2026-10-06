from __future__ import annotations

import math

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.cleaning import clean_snapshot
from ivcrypto.implied_vol import (
    DEFAULT_LOWER,
    DEFAULT_UPPER,
    add_implied_vols,
    implied_vol,
    implied_vols,
)

F = 85_000.0
LOG_MONEYNESS = np.linspace(-1.5, 1.5, 13)
MATURITIES = [2 / 365, 30 / 365, 1.0, 3.0]
VOLS = [0.05, 0.3, 0.8, 2.5]


def cases():
    for k in LOG_MONEYNESS:
        for T in MATURITIES:
            for sigma in VOLS:
                for is_call in (True, False):
                    yield F * math.exp(k), T, sigma, is_call


@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_price_to_iv_to_price(is_call):
    checked = 0
    for K, T, sigma, call in cases():
        if call is not is_call:
            continue
        price = float(black76.price(F, K, T, sigma, call))
        time_value = price - float(black76.intrinsic_value(F, K, call))
        if time_value < 1e-10 * F:  # indistinguishable from intrinsic in double precision
            continue
        iv = implied_vol(price, F, K, T, call)
        assert math.isfinite(iv), (K, T, sigma)
        assert float(black76.price(F, K, T, iv, call)) == pytest.approx(price, rel=1e-11, abs=1e-9)
        if float(black76.vega(F, K, T, sigma)) > 1e-4 * F:
            assert iv == pytest.approx(sigma, rel=1e-9)
        checked += 1
    assert checked > 100


def test_call_and_put_at_one_strike_give_one_vol():
    for K, T, sigma, _ in cases():
        call = float(black76.price(F, K, T, sigma, True))
        put = float(black76.price(F, K, T, sigma, False))
        if min(call, put) - float(black76.intrinsic_value(F, K, call > put)) < 1e-6 * F:
            continue
        assert implied_vol(call, F, K, T, True) == pytest.approx(
            implied_vol(put, F, K, T, False), rel=1e-8
        )


def test_vectorized_matches_scalar():
    K = np.array([60_000.0, 85_000.0, 120_000.0])
    T = np.array([0.02, 0.5, 2.0])
    prices = black76.price(F, K, T, 0.6, [True, False, True])
    vector = implied_vols(prices, F, K, T, [True, False, True])
    scalar = [
        implied_vol(p, F, k, t, c)
        for p, k, t, c in zip(prices, K, T, [True, False, True], strict=True)
    ]
    np.testing.assert_allclose(vector, scalar, rtol=0, atol=0)
    np.testing.assert_allclose(vector, 0.6, rtol=1e-9)


@pytest.mark.parametrize(
    ("price", "K", "is_call"),
    [
        (4_999.0, 80_000.0, True),  # below intrinsic (5000)
        (5_000.0, 80_000.0, True),  # exactly intrinsic: zero time value
        (F, 80_000.0, True),  # a call worth the whole forward
        (100_000.0, 100_000.0, False),  # a put worth its whole strike
        (-1.0, 90_000.0, True),
        (0.0, 90_000.0, True),
        (math.nan, 90_000.0, True),
    ],
)
def test_arbitrage_violations_return_nan(price, K, is_call):
    assert math.isnan(implied_vol(price, F, K, 0.5, is_call))


@pytest.mark.parametrize(("F_", "K", "T"), [(0.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 0.0)])
def test_degenerate_inputs_return_nan(F_, K, T):
    assert math.isnan(implied_vol(0.1, F_, K, T, True))


def test_vols_outside_the_bracket_return_nan():
    T = 0.5
    too_high = float(black76.price(F, F, T, 2 * DEFAULT_UPPER, True))
    too_low = float(black76.price(F, F, T, 0.5 * DEFAULT_LOWER, True))
    assert too_low > 0  # at the money the price stays representable at tiny vols
    assert math.isnan(implied_vol(too_high, F, F, T, True))
    assert math.isnan(implied_vol(too_low, F, F, T, True))


def test_discounted_prices_invert_with_their_discount():
    discount = math.exp(-0.05 * 2.0)
    price = float(black76.price(F, 90_000.0, 2.0, 0.45, False, discount))
    assert implied_vol(price, F, 90_000.0, 2.0, False, discount=discount) == pytest.approx(0.45)


def test_sample_ivs_are_ordered_and_complete(btc_sample):
    clean = add_implied_vols(clean_snapshot(btc_sample))
    kept = clean.kept
    assert kept[["iv_bid", "iv_mid", "iv_ask", "vega_mid"]].notna().all().all()
    assert (kept["iv_bid"] < kept["iv_mid"]).all()
    assert (kept["iv_mid"] < kept["iv_ask"]).all()
    assert (kept["vega_mid"] > 0).all()
    assert kept["iv_mid"].between(0.1, 3.0).all()
    assert clean.filters["step"].iloc[-1] == "no_implied_vol"
    # ITM bids below intrinsic exist in the raw quotes and correctly get no IV.
    itm_no_bid_iv = clean.quotes["iv_bid"].isna() & clean.quotes["bid_btc"].notna()
    assert itm_no_bid_iv.sum() > 0
    assert (~clean.quotes.loc[itm_no_bid_iv, "otm"]).all()

"""Static arbitrage checks: constructed surfaces (synthetic, built here) and the real sample."""

from __future__ import annotations

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.arbitrage import (
    Slice,
    check_butterfly,
    check_calendar,
    check_strike_shape,
    check_surface,
    durrleman_g,
    slices_from_svi,
)
from ivcrypto.cleaning import clean_snapshot
from ivcrypto.implied_vol import add_implied_vols
from ivcrypto.svi import SVIParams, fit_svi

# Axel Vogt's raw SVI slice (Gatheral and Jacquier 2014): admissible parameters, yet the
# implied density is negative for k roughly between 0.64 and 1.26.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
CLEAN = SVIParams(a=0.05, b=0.2, rho=0.05, m=0.0, sigma=0.4)


def test_vogt_slice_is_admissible_but_has_butterfly_arbitrage():
    assert VOGT.violations() == []
    result = check_butterfly([Slice("VOGT", 1.0, VOGT, -1.0, 1.5)])
    assert len(result.violations) == 1
    row = result.violations.iloc[0]
    assert row["k_low"] == pytest.approx(0.643, abs=2e-3)
    assert row["k_high"] == pytest.approx(1.257, abs=2e-3)
    assert row["magnitude"] == pytest.approx(0.0329, abs=5e-4)
    assert row["in_quoted_range"]


def test_negative_g_means_negative_density():
    # Independent check of the formula: Breeden Litzenberger on Black prices of the slice.
    k = np.linspace(-1.0, 1.5, 20001)
    strikes = np.exp(k)
    calls = black76.price(1.0, strikes, 1.0, np.sqrt(VOGT.total_variance(k)), True)
    density = np.gradient(np.gradient(calls, strikes), strikes)
    g = durrleman_g(VOGT, k)
    inner = slice(100, -100)  # finite differences are unreliable at the edges
    g, density = g[inner], density[inner]
    assert np.all(density[g < -1e-3] < 0)
    assert np.all(density[g > 1e-3] > 0)
    assert (g < -1e-3).sum() > 1000  # the negative region is well sampled


def test_violations_outside_the_quoted_range_are_labelled():
    result = check_butterfly([Slice("VOGT", 1.0, VOGT, -0.5, 0.5)])
    assert not result.violations["in_quoted_range"].any()


def test_clean_slice_passes():
    result = check_butterfly([Slice("CLEAN", 1.0, CLEAN, -1.0, 1.0)])
    assert result.violations.empty
    assert durrleman_g(CLEAN, np.linspace(-1.5, 1.5, 3001)).min() > 0


def test_crossing_slices_are_calendar_arbitrage():
    short = Slice("SHORT", 0.5, SVIParams(a=0.02, b=0.25, rho=-0.6, m=0.0, sigma=0.2), -1, 1)
    long = Slice("LONG", 1.0, SVIParams(a=0.04, b=0.10, rho=0.0, m=0.0, sigma=0.3), -1, 1)
    result = check_calendar([long, short])  # order does not matter
    assert not result.violations.empty
    assert set(result.violations["expiry"]) == {"SHORT/LONG"}
    k = np.linspace(-1.5, 1.5, 3001)
    crossing = k[long.params.total_variance(k) < short.params.total_variance(k)]
    first = result.violations.iloc[0]
    assert first["k_low"] == pytest.approx(crossing.min(), abs=1e-3)


def test_nested_slices_pass_the_calendar_check():
    base = SVIParams(a=0.02, b=0.2, rho=-0.4, m=0.0, sigma=0.2)
    later = SVIParams(a=0.05, b=0.2, rho=-0.4, m=0.0, sigma=0.2)  # w shifted up everywhere
    result = check_calendar([Slice("A", 0.5, base, -1, 1), Slice("B", 1.0, later, -1, 1)])
    assert result.violations.empty
    assert result.tests["tested"].iloc[0] == 1


def coin_quotes(option_type, half_spread=0.0005):
    strikes = np.arange(60_000.0, 121_000.0, 5_000.0)
    vols = 0.5 + 0.3 * np.log(strikes / 90_000.0) ** 2
    mid = black76.price(90_000.0, strikes, 0.25, vols, option_type == "call") / 90_000.0
    return strikes, mid - half_spread, mid + half_spread


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_arbitrage_free_quotes_pass(option_type):
    strikes, bid, ask = coin_quotes(option_type)
    result = check_strike_shape(strikes, bid, ask, option_type, "TEST")
    assert result.violations.empty
    tested = result.tests.set_index("prices")["tested"]
    assert tested.sum() > 0


def test_a_dent_inside_the_spread_is_a_mid_violation_only():
    half = 0.002
    strikes, bid, ask = coin_quotes("call", half_spread=half)
    i = 6
    mid = 0.5 * (bid + ask)
    lam = (strikes[i + 1] - strikes[i]) / (strikes[i + 1] - strikes[i - 1])
    margin = lam * mid[i - 1] + (1 - lam) * mid[i + 1] - mid[i]  # convexity slack at i
    # Raise the body past the slack, but by less than a full spread: the mids break
    # convexity while bid and ask still admit no trade.
    bid[i] += margin + half
    ask[i] += margin + half
    result = check_strike_shape(strikes, bid, ask, "call", "TEST")
    v = result.violations
    assert set(v["prices"]) == {"mid"}
    assert set(v["check"]) == {"call convexity"}


def test_a_large_dent_is_tradable():
    strikes, bid, ask = coin_quotes("call")
    i = 6
    bid[i] += 0.02
    ask[i] += 0.02
    v = check_strike_shape(strikes, bid, ask, "call", "TEST").violations
    executable = v[v["prices"] == "executable"]
    assert not executable.empty
    assert f"{strikes[i]:g}" in executable.iloc[0]["location"]
    assert executable["magnitude"].max() > 0.0


def test_a_call_bid_above_a_lower_strike_ask_is_tradable_monotonicity():
    strikes, bid, ask = coin_quotes("call")
    bid[5] = ask[4] + 0.001  # sell the higher strike above the price of the lower one
    ask[5] = bid[5] + 0.001
    v = check_strike_shape(strikes, bid, ask, "call", "TEST").violations
    rows = v[(v["check"] == "call monotonicity") & (v["prices"] == "executable")]
    assert len(rows) >= 1
    assert rows.iloc[0]["magnitude"] == pytest.approx(0.001)


def test_put_monotonicity_runs_the_other_way():
    strikes, bid, ask = coin_quotes("put")
    bid[3], ask[3] = ask[4] + 0.001, ask[4] + 0.002  # a lower strike put priced above
    v = check_strike_shape(strikes, bid, ask, "put", "TEST").violations
    assert not v[(v["check"] == "put monotonicity") & (v["prices"] == "executable")].empty


@pytest.fixture(scope="module")
def btc_report(btc_sample):
    clean = add_implied_vols(clean_snapshot(btc_sample))
    return check_surface(slices_from_svi(fit_svi(clean)), clean)


def test_sample_report_counts_add_up(btc_report):
    summary = btc_report.summary()
    assert (summary["violations"] <= summary["tested"]).all()
    assert (summary["violations_in_quoted_range"] <= summary["violations"]).all()
    assert summary["violations"].sum() == len(btc_report.violations)
    assert set(summary["prices"]) == {"model", "mid", "executable", "mid, interpolated"}


def test_btc_sample_has_no_tradable_arbitrage(btc_report):
    # Documented properties of the committed snapshot: mid prices break convexity here and
    # there, but nothing is tradable at bid and ask, and the SVI fits are free of static
    # arbitrage wherever quotes exist.
    v = btc_report.violations
    assert (v["prices"] == "executable").sum() == 0
    assert not v[v["prices"] == "model"]["in_quoted_range"].any()
    assert (btc_report.durrleman["min_g_quoted"] > 0).all()
    assert (btc_report.calendar_margins["min_increase_quoted"] > 0).all()

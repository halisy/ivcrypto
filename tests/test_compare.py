"""SVI against Heston on the committed BTC snapshot: the documented findings, as tests."""

from __future__ import annotations

import numpy as np
import pytest

from ivcrypto.cleaning import clean_snapshot
from ivcrypto.compare import atm_term_structure, compare_models, model_ivs, power_law
from ivcrypto.heston.calibrate import calibrate_heston, calibrate_per_expiry, calibration_quotes
from ivcrypto.implied_vol import add_implied_vols
from ivcrypto.svi import fit_svi


@pytest.fixture(scope="module")
def models(btc_sample):
    clean = add_implied_vols(clean_snapshot(btc_sample))
    svi = fit_svi(clean)
    quotes = calibration_quotes(clean)
    heston = calibrate_heston(quotes)
    shortest_and_longest = [quotes[0], quotes[-1]]
    per_expiry = calibrate_per_expiry(shortest_and_longest)
    return clean, svi, heston, per_expiry


def test_every_model_is_scored_on_the_same_quotes(models):
    clean, svi, heston, per_expiry = models
    ivs = model_ivs(svi, heston, per_expiry)
    assert len(ivs) == len(clean.kept)
    assert ivs["iv_heston"].notna().all()
    covered = ivs["expiry_code"].isin(per_expiry)
    assert ivs.loc[covered, "iv_heston_expiry"].notna().all()
    assert ivs.loc[~covered, "iv_heston_expiry"].isna().all()


def test_heston_fits_far_worse_than_svi(models):
    _, svi, heston, per_expiry = models
    overall = compare_models(model_ivs(svi, heston, per_expiry), by=None).iloc[0]
    assert overall["SVI rmse"] < 0.3
    assert overall["Heston rmse"] > 5 * overall["SVI rmse"]
    assert overall["Heston inside_band"] < 0.5 < overall["SVI inside_band"]
    by_expiry = compare_models(model_ivs(svi, heston, per_expiry)).set_index("group")
    for code in per_expiry:  # one smile at a time fits better, but still not like SVI
        row = by_expiry.loc[code]
        assert row["Heston per expiry rmse"] <= row["Heston rmse"] + 1e-9
        assert row["Heston per expiry rmse"] > row["SVI rmse"]


def test_calibrated_heston_violates_feller(models):
    _, _, heston, _ = models
    assert not heston.params.feller_satisfied
    assert heston.params.xi > 2.0  # a very high vol of vol is needed to bend the smiles


def test_market_curvature_explodes_while_heston_flattens(models):
    _, svi, heston, _ = models
    ts = atm_term_structure(svi, heston.params)
    _, alpha_market = power_law(ts["T"], ts["market_atm_curvature"])
    assert 0.9 < alpha_market < 1.25  # close to the 1/T scaling that jumps produce
    first, second = ts.iloc[0], ts.iloc[1]
    market_ratio = first["market_atm_curvature"] / second["market_atm_curvature"]
    heston_ratio = first["heston_atm_curvature"] / second["heston_atm_curvature"]
    assert market_ratio > 1.4
    assert heston_ratio < 1.15
    # Short end: the market smile tilts up at the money, Heston's always tilts down.
    assert first["market_atm_skew"] > 0 > first["heston_atm_skew"]
    assert (ts["heston_atm_skew"] < 0).all()


def test_power_law_recovers_its_exponent():
    T = np.array([0.01, 0.1, 1.0])
    c, alpha = power_law(T, 2.0 * T**-0.5)
    assert c == pytest.approx(2.0)
    assert alpha == pytest.approx(0.5)

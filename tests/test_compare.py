"""SVI against Heston and Bates on the committed BTC snapshot: the documented findings, as
tests."""

from __future__ import annotations

import numpy as np
import pytest

from ivcrypto.bates import calibrate_bates
from ivcrypto.bates.calibrate import UPPER
from ivcrypto.cleaning import clean_snapshot
from ivcrypto.compare import atm_term_structure, compare_models, model_ivs, power_law
from ivcrypto.config import BatesConfig
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


@pytest.fixture(scope="module")
def bates(models):
    clean, _, heston, _ = models
    # Two starts besides Heston: enough to reach the optimum of the default build here.
    return calibrate_bates(calibration_quotes(clean), heston.params, config=BatesConfig(2))


def test_every_model_is_scored_on_the_same_quotes(models, bates):
    clean, svi, heston, per_expiry = models
    ivs = model_ivs(svi, heston, per_expiry, bates=bates)
    assert len(ivs) == len(clean.kept)
    assert ivs["iv_heston"].notna().all()
    assert ivs["iv_bates"].notna().all()
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
    ts = atm_term_structure(svi, {"heston": heston.params})
    _, alpha_market = power_law(ts["T"], ts["market_atm_curvature"])
    assert 0.9 < alpha_market < 1.25  # about 1/T from 2.4 days to a year
    first, second = ts.iloc[0], ts.iloc[1]
    market_ratio = first["market_atm_curvature"] / second["market_atm_curvature"]
    heston_ratio = first["heston_atm_curvature"] / second["heston_atm_curvature"]
    assert market_ratio > 1.4
    assert heston_ratio < 1.2
    # Short end: the market smile tilts up at the money, Heston's always tilts down.
    assert first["market_atm_skew"] > 0 > first["heston_atm_skew"]
    assert (ts["heston_atm_skew"] < 0).all()


def test_power_law_recovers_its_exponent():
    T = np.array([0.01, 0.1, 1.0])
    c, alpha = power_law(T, 2.0 * T**-0.5)
    assert c == pytest.approx(2.0)
    assert alpha == pytest.approx(0.5)


def test_bates_halves_hestons_error_but_not_with_crash_jumps(models, bates):
    _, svi, heston, _ = models
    overall = compare_models(model_ivs(svi, heston, bates=bates), by=None).iloc[0]
    assert bates.cost <= heston.cost
    assert overall["Bates rmse"] < 0.5 * overall["Heston rmse"]
    assert overall["Bates rmse"] > 3 * overall["SVI rmse"]  # still far from SVI
    # What the fit asks for: jumps at the intensity cap, small and upward on average.
    params = bates.params
    assert params.lam == pytest.approx(UPPER[5])
    assert "lam at upper bound" in bates.at_bounds
    assert 0 < params.jump_mean < 0.05
    assert params.xi > heston.params.xi  # the variance process works even harder


def test_bates_fixes_the_middle_of_the_curvature_term_structure_not_the_short_end(models, bates):
    _, svi, heston, _ = models
    ts = atm_term_structure(svi, {"heston": heston.params, "bates": bates.params})
    ratio = ts["bates_atm_curvature"] / ts["market_atm_curvature"]
    middle = ts["days"].between(10, 90)
    assert np.all(np.abs(ratio[middle] - 1) < 0.15)
    assert ratio.iloc[0] < 0.5  # at 2.4 days Bates' smile is far flatter than the market's
    assert ts["bates_atm_skew"].iloc[0] > 0  # but tilted up, like the market's

"""Raw SVI and its fit: synthetic slices (built here) and the committed real snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.cleaning import clean_snapshot
from ivcrypto.config import SVIConfig
from ivcrypto.implied_vol import add_implied_vols
from ivcrypto.svi import SVIParams, fit_slice, fit_svi
from ivcrypto.svi.fit import SliceQuotes, from_params, to_params
from ivcrypto.svi.raw import LEE_MAX_SLOPE

CASES = {
    "short dated crypto": (
        SVIParams(a=0.0002, b=0.03, rho=-0.2, m=0.0, sigma=0.05),
        10 / 365,
        np.linspace(-0.25, 0.2, 21),
    ),
    "equity like skew": (
        SVIParams(a=0.02, b=0.12, rho=-0.7, m=0.05, sigma=0.15),
        0.5,
        np.linspace(-0.6, 0.4, 25),
    ),
    "long dated smile": (
        SVIParams(a=0.05, b=0.2, rho=0.05, m=0.0, sigma=0.4),
        1.0,
        np.linspace(-0.8, 0.8, 31),
    ),
}


def synthetic_slice(iv_mid, T, k, half_band=0.005):
    iv_mid = np.asarray(iv_mid, dtype=float)
    return SliceQuotes(
        expiry_code="TEST",
        T=T,
        instrument_name=np.array([f"opt{i}" for i in range(k.size)], dtype=object),
        k=k,
        iv_bid=iv_mid - half_band,
        iv_mid=iv_mid,
        iv_ask=iv_mid + half_band,
        vega=black76.vega(1.0, np.exp(k), T, iv_mid),
    )


def admissible(params):
    return params.violations() == []


@pytest.mark.parametrize("name", list(CASES))
def test_cases_are_admissible(name):
    assert admissible(CASES[name][0])


def test_derivatives_match_finite_differences():
    params = CASES["equity like skew"][0]
    k = np.linspace(-1.0, 1.0, 41)
    h = 1e-5
    w, dw, d2w = params.derivatives(k)
    np.testing.assert_allclose(w, params.total_variance(k))
    up, down = params.total_variance(k + h), params.total_variance(k - h)
    np.testing.assert_allclose(dw, (up - down) / (2 * h), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(d2w, (up - 2 * w + down) / h**2, rtol=1e-4, atol=1e-6)


def test_minimum_and_wings():
    params = CASES["equity like skew"][0]
    k = np.linspace(-3, 3, 600_001)
    w = params.total_variance(k)
    assert params.min_variance == pytest.approx(w.min(), abs=1e-10)
    assert params.argmin == pytest.approx(k[np.argmin(w)], abs=1e-4)
    left, right = params.wing_slopes
    far = 1e6
    assert (params.total_variance(-far) / far) == pytest.approx(left, rel=1e-5)
    assert (params.total_variance(far) / far) == pytest.approx(right, rel=1e-5)


@pytest.mark.parametrize(
    ("params", "broken"),
    [
        (SVIParams(a=0.01, b=-0.1, rho=0.0, m=0.0, sigma=0.1), "b >= 0"),
        (SVIParams(a=0.01, b=0.1, rho=1.0, m=0.0, sigma=0.1), "|rho| < 1"),
        (SVIParams(a=0.01, b=0.1, rho=0.0, m=0.0, sigma=0.0), "sigma > 0"),
        (SVIParams(a=-0.1, b=0.1, rho=0.0, m=0.0, sigma=0.1), "minimum variance >= 0"),
        (SVIParams(a=0.01, b=1.5, rho=0.5, m=0.0, sigma=0.1), "Lee wing slope <= 2"),
    ],
)
def test_each_violation_is_detected(params, broken):
    assert broken in params.violations()


def test_reparameterization_round_trips():
    for params, _, _ in CASES.values():
        recovered = to_params(from_params(params))
        np.testing.assert_allclose(recovered.as_array(), params.as_array(), atol=1e-14)


@pytest.mark.parametrize("weighting", ["spread", "vega", "uniform"])
@pytest.mark.parametrize("name", list(CASES))
def test_noiseless_slices_are_recovered(name, weighting):
    params, T, k = CASES[name]
    fit = fit_slice(synthetic_slice(params.implied_vol(k, T), T, k), SVIConfig(weighting=weighting))
    np.testing.assert_allclose(fit.params.total_variance(k), params.total_variance(k), atol=1e-9)
    np.testing.assert_allclose(fit.params.as_array(), params.as_array(), rtol=1e-4, atol=1e-6)
    assert fit.in_band_share == 1.0
    assert fit.rmse_vol < 1e-5


@pytest.mark.parametrize("name", list(CASES))
def test_noisy_slices_stay_admissible_and_mostly_in_band(name):
    params, T, k = CASES[name]
    rng = np.random.default_rng(11)
    half = 0.005
    noisy = params.implied_vol(k, T) + rng.uniform(-0.6, 0.6, k.size) * half
    fit = fit_slice(synthetic_slice(noisy, T, k, half))
    assert admissible(fit.params)
    assert fit.in_band_share >= 0.9
    assert fit.rmse_vol < 100 * half


def check_bounded(fit):
    p = fit.params
    assert admissible(p), p.violations()
    assert p.b >= 0
    assert abs(p.rho) < 1
    assert p.sigma > 0
    assert p.min_variance >= -1e-14
    assert p.b * (1 + abs(p.rho)) <= LEE_MAX_SLOPE + 1e-12


def test_constraints_hold_when_the_data_want_steeper_wings():
    k, T = np.linspace(-0.8, 0.8, 33), 1.0
    w = 2.6 * np.abs(k) + 0.001  # wings steeper than Lee's bound allows
    fit = fit_slice(synthetic_slice(np.sqrt(w / T), T, k))
    check_bounded(fit)
    assert "left wing at Lee bound" in fit.active_bounds
    assert "right wing at Lee bound" in fit.active_bounds


def test_constraints_hold_for_a_v_shape_with_a_vanishing_minimum():
    k, T = np.linspace(-0.3, 0.3, 25), 0.1
    w = 0.5 * np.abs(k - 0.01) + 1e-7
    check_bounded(fit_slice(synthetic_slice(np.sqrt(w / T), T, k, half_band=0.01)))


def test_constraints_hold_for_a_frown():
    k, T = np.linspace(-0.4, 0.4, 21), 0.5
    w = 0.04 - 0.05 * k**2  # concave: no admissible SVI matches it exactly
    check_bounded(fit_slice(synthetic_slice(np.sqrt(w / T), T, k)))


def test_every_start_is_reported():
    params, T, k = CASES["equity like skew"]
    config = SVIConfig(n_starts=5)
    fit = fit_slice(synthetic_slice(params.implied_vol(k, T), T, k), config)
    assert len(fit.start_costs) == config.n_starts + 1  # grid starts plus the heuristic one
    assert fit.cost == min(fit.start_costs)
    assert 1 <= fit.converged_starts <= len(fit.start_costs)


def test_weighting_changes_the_fit():
    params, T, k = CASES["long dated smile"]
    rng = np.random.default_rng(5)
    iv = params.implied_vol(k, T) + rng.normal(0, 0.004, k.size)
    data = synthetic_slice(iv, T, k)
    widths = np.linspace(0.002, 0.03, k.size)  # tight quotes on one side, wide on the other
    data = SliceQuotes(**{**data.__dict__, "iv_bid": iv - widths, "iv_ask": iv + widths})
    spread = fit_slice(data, SVIConfig(weighting="spread")).params.as_array()
    vega = fit_slice(data, SVIConfig(weighting="vega")).params.as_array()
    assert not np.allclose(spread, vega, rtol=1e-3)


@pytest.fixture(scope="module")
def btc_clean(btc_sample):
    return add_implied_vols(clean_snapshot(btc_sample))


def test_every_btc_expiry_fits_inside_the_market(btc_clean):
    surface = fit_svi(btc_clean)
    summary = surface.summary()
    assert surface.skipped == {}
    assert list(summary["expiry_code"]) == list(
        btc_clean.kept.groupby("expiry_code")["T"].first().sort_values().index
    )
    for fit in surface.fits.values():
        assert admissible(fit.params)
        assert fit.in_band_share >= 0.9
        assert fit.rmse_vol < 0.5
        assert fit.converged_starts >= 5
    assert summary["days"].is_monotonic_increasing
    residuals = surface.residuals()
    assert len(residuals) == len(btc_clean.kept)


def test_slices_with_too_few_quotes_are_skipped(btc_clean):
    surface = fit_svi(btc_clean, SVIConfig(min_quotes=50))
    assert surface.skipped
    assert all("fewer than min_quotes" in reason for reason in surface.skipped.values())
    assert all(fit.n_quotes >= 50 for fit in surface.fits.values())

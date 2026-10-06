"""SSVI: conditions, independent arbitrage checks, recovery on a synthetic surface, real data."""

from __future__ import annotations

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.arbitrage import (
    K_GRID,
    Slice,
    check_butterfly,
    check_calendar,
    check_surface,
    slices_from_ssvi,
)
from ivcrypto.cleaning import clean_snapshot
from ivcrypto.implied_vol import add_implied_vols
from ivcrypto.svi import fit_svi
from ivcrypto.svi.fit import SliceQuotes
from ivcrypto.svi.ssvi import SSVIParams, fit_ssvi, fit_ssvi_slices

GOOD = SSVIParams(rho=-0.3, eta=1.2, gamma=0.4)  # eta (1 + |rho|) = 1.56 <= 2
THETAS = np.geomspace(1e-4, 2.0, 40)


def test_each_slice_is_a_raw_svi_slice():
    k = np.linspace(-2, 2, 401)
    for theta in (1e-4, 0.01, 0.3, 2.0):
        raw = GOOD.slice_params(theta)
        np.testing.assert_allclose(raw.total_variance(k), GOOD.total_variance(k, theta), rtol=1e-12)
        assert raw.total_variance(0.0) == pytest.approx(theta)  # theta is ATM total variance
        assert raw.violations() == []


def test_conditions_give_no_arbitrage_on_the_independent_checker():
    assert GOOD.violations() == []
    slices = [
        Slice(f"t{i}", float(i + 1), GOOD.slice_params(t), -1, 1) for i, t in enumerate(THETAS)
    ]
    assert check_butterfly(slices, K_GRID).violations.empty
    assert check_calendar(slices, K_GRID).violations.empty


def test_breaking_the_butterfly_condition_creates_arbitrage():
    bad = SSVIParams(rho=0.5, eta=4.0, gamma=0.5)
    assert "eta (1 + |rho|) <= 2" in bad.violations()
    slices = [
        Slice(f"t{i}", float(i + 1), bad.slice_params(t), -1, 1) for i, t in enumerate(THETAS)
    ]
    assert not check_butterfly(slices, np.linspace(-3, 3, 6001)).violations.empty


@pytest.mark.parametrize(
    ("params", "broken"),
    [
        (SSVIParams(rho=1.0, eta=0.5, gamma=0.3), "|rho| < 1"),
        (SSVIParams(rho=0.0, eta=0.0, gamma=0.3), "eta > 0"),
        (SSVIParams(rho=0.5, eta=1.5, gamma=0.3), "eta (1 + |rho|) <= 2"),
        (SSVIParams(rho=0.0, eta=1.0, gamma=0.7), "0 < gamma <= 1/2"),
    ],
)
def test_each_condition_is_reported(params, broken):
    assert broken in params.violations()


TRUE = SSVIParams(rho=-0.25, eta=1.1, gamma=0.35)
MATURITIES = np.array([0.02, 0.1, 0.3, 0.8])
THETA_TRUE = np.array([0.003, 0.012, 0.04, 0.11])


def synthetic_slices(params=TRUE):
    slices = []
    for T, theta in zip(MATURITIES, THETA_TRUE, strict=True):
        width = 4 * np.sqrt(theta)
        k = np.linspace(-width, 0.8 * width, 25)
        iv = np.sqrt(params.total_variance(k, theta) / T)
        slices.append(
            SliceQuotes(
                expiry_code=f"T{T}",
                T=float(T),
                instrument_name=np.array([f"q{i}" for i in range(k.size)], dtype=object),
                k=k,
                iv_bid=iv - 0.005,
                iv_mid=iv,
                iv_ask=iv + 0.005,
                vega=black76.vega(1.0, np.exp(k), T, iv),
            )
        )
    return slices


def test_synthetic_surface_is_recovered():
    slices = synthetic_slices()
    fit = fit_ssvi_slices(slices, THETA_TRUE * 1.2, rho0=0.0)  # deliberately off start
    assert fit.params.rho == pytest.approx(TRUE.rho, abs=1e-6)
    assert fit.params.eta == pytest.approx(TRUE.eta, rel=1e-6)
    assert fit.params.gamma == pytest.approx(TRUE.gamma, rel=1e-6)
    np.testing.assert_allclose(fit.theta, THETA_TRUE, rtol=1e-6)
    assert fit.residuals["in_band"].all()


def test_theta_interpolation_is_monotone_and_exact_at_expiries():
    fit = fit_ssvi_slices(synthetic_slices(), THETA_TRUE)
    np.testing.assert_allclose(fit.theta_at(MATURITIES), fit.theta, rtol=1e-12)
    grid = np.linspace(0.0, 1.5, 301)
    assert np.all(np.diff(fit.theta_at(grid)) >= 0)
    assert fit.theta_at(0.0) == 0.0
    assert fit.theta_at(1.6) == pytest.approx(fit.theta[-1] * 1.6 / MATURITIES[-1])


def test_unordered_slices_are_rejected():
    with pytest.raises(ValueError, match="increasing maturity"):
        fit_ssvi_slices(synthetic_slices()[::-1], THETA_TRUE[::-1])


@pytest.fixture(scope="module")
def btc_fits(btc_sample):
    clean = add_implied_vols(clean_snapshot(btc_sample))
    svi = fit_svi(clean)
    return svi, fit_ssvi(clean, svi)


def test_btc_ssvi_is_arbitrage_free_between_expiries_too(btc_fits):
    _, ssvi = btc_fits
    assert ssvi.params.violations() == []
    assert np.all(np.diff(ssvi.theta) >= 0)
    report = check_surface(slices_from_ssvi(ssvi, np.linspace(1 / 365, 1.2, 60)))
    assert report.violations.empty
    assert report.tests.set_index("check").loc["butterfly", "tested"] == len(ssvi.T) + 60


def test_btc_ssvi_pays_for_it_in_fit_quality(btc_fits):
    # Documented result: one rho for all maturities and bounded short dated curvature fit
    # this snapshot much worse than independent SVI slices.
    svi, ssvi = btc_fits
    svi_rmse = np.sqrt(np.mean(svi.residuals()["error_vol"] ** 2))
    ssvi_rmse = np.sqrt(np.mean(ssvi.residuals["error_vol"] ** 2))
    assert ssvi_rmse > 3 * svi_rmse
    assert ssvi.residuals["in_band"].mean() < svi.residuals()["in_band"].mean()
    assert ssvi.params.gamma == pytest.approx(0.5)  # the curvature exponent hits its bound

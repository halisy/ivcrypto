"""Bates pricer, Monte Carlo and calibration (synthetic data only inside these tests)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.bates import BatesParams, calibrate_bates, mc_price
from ivcrypto.bates.calibrate import LOWER, UPPER, VARIANTS, variant_bounds
from ivcrypto.heston import HestonParams, calibrate_heston, price, price_quad
from ivcrypto.heston.calibrate import binding_bounds

F = 86_000.0
CRYPTO_LIKE = BatesParams(
    v0=0.12, kappa=3.0, theta=0.2, xi=2.5, rho=-0.2, lam=10.0, mu_j=-0.1, sigma_j=0.1
)
CALIBRATED_LIKE = BatesParams(  # close to the BTC calibration: many small upward jumps
    v0=0.046, kappa=16.3, theta=0.133, xi=9.2, rho=-0.22, lam=100.0, mu_j=0.025, sigma_j=0.0
)


def merton_prices(sigma, lam, mu, delta, K, T, is_call):
    """Merton's jump diffusion as a Poisson mixture of Black 76 prices."""
    mean_jump = math.exp(mu + 0.5 * delta**2) - 1.0
    total = np.zeros_like(K)
    for n in range(100):
        weight = math.exp(-lam * T) * (lam * T) ** n / math.factorial(n)
        forward = F * math.exp(-lam * mean_jump * T + n * (mu + 0.5 * delta**2))
        vol = math.sqrt(sigma**2 + n * delta**2 / T)
        total = total + weight * black76.price(forward, K, T, vol, is_call)
    return total


@pytest.mark.parametrize("T", [2 / 365, 0.25, 1.0])
def test_without_jumps_bates_is_heston(T):
    heston = HestonParams(v0=0.12, kappa=3.0, theta=0.2, xi=2.5, rho=-0.2)
    bates = BatesParams(*heston.as_array(), lam=0.0, mu_j=-0.1, sigma_j=0.1)
    strikes = F * np.exp(np.linspace(-1.0, 1.0, 9) * math.sqrt(0.2 * T) * 3)
    calls = strikes >= F
    np.testing.assert_array_equal(
        price(F, strikes, T, bates, calls), price(F, strikes, T, heston, calls)
    )


@pytest.mark.parametrize("T", [2 / 365, 0.1, 1.0])
def test_converges_to_merton_as_vol_of_vol_vanishes(T):
    sigma, lam, mu, delta = 0.4, 5.0, -0.08, 0.12
    bates = BatesParams(sigma**2, 1.0, sigma**2, 1e-8, 0.0, lam, mu, delta)
    strikes = F * np.exp(np.linspace(-1.0, 1.0, 9) * math.sqrt(sigma**2 * T) * 4)
    calls = strikes >= F
    expected = merton_prices(sigma, lam, mu, delta, strikes, T, calls)
    np.testing.assert_allclose(price(F, strikes, T, bates, calls), expected, rtol=0, atol=1e-12 * F)


@pytest.mark.parametrize("params", [CRYPTO_LIKE, CALIBRATED_LIKE])
def test_characteristic_function_keeps_the_forward_a_martingale(params):
    for T in (2 / 365, 1.0):
        assert params.cf(np.array([0.0, -1j]), T) == pytest.approx([1.0, 1.0], abs=1e-12)


@pytest.mark.parametrize("params", [CRYPTO_LIKE, CALIBRATED_LIKE])
@pytest.mark.parametrize("T", [2 / 365, 10 / 365, 0.25, 1.0])
def test_fast_pricer_matches_adaptive_quadrature(params, T):
    strikes = F * np.exp(np.linspace(-3.0, 3.0, 9) * np.sqrt(0.3 * T))
    calls = strikes >= F
    fast = price(F, strikes, T, params, calls)
    slow = [price_quad(F, k, T, params, c) for k, c in zip(strikes, calls, strict=True)]
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-11 * F)


def test_put_call_parity_on_the_forward():
    strikes = F * np.exp(np.linspace(-0.5, 0.5, 7))
    calls = price(F, strikes, 0.1, CRYPTO_LIKE, True)
    puts = price(F, strikes, 0.1, CRYPTO_LIKE, False)
    np.testing.assert_allclose(calls - puts, F - strikes, rtol=0, atol=1e-9 * F)


@pytest.mark.parametrize(
    ("params", "T"), [(CRYPTO_LIKE, 10 / 365), (CRYPTO_LIKE, 0.5), (CALIBRATED_LIKE, 3 / 365)]
)
def test_monte_carlo_agrees_with_the_fourier_pricer(params, T):
    strikes = F * np.exp(np.linspace(-1.5, 1.5, 7) * np.sqrt(0.3 * T))
    calls = strikes >= F
    mc = mc_price(F, strikes, T, params, calls, n_paths=100_000, seed=11)
    z = (mc.price - price(F, strikes, T, params, calls)) / mc.std_error
    assert np.all(np.abs(z) < 4.0), z
    assert abs(mc.forward_mean - F) < 4.0 * mc.forward_std_error


def test_calibration_recovers_bates_prices(synthetic_quotes):
    truth = BatesParams(
        v0=0.3, kappa=2.5, theta=0.4, xi=1.2, rho=-0.35, lam=5.0, mu_j=-0.08, sigma_j=0.1
    )
    quotes = synthetic_quotes(truth, [0.02, 0.1, 0.4, 1.0])
    heston = calibrate_heston(quotes)
    bates = calibrate_bates(quotes, heston.params)
    assert bates.cost < 1e-3 * heston.cost
    assert bates.residuals["in_band"].all()
    assert np.abs(bates.residuals["error_vol"]).max() < 0.01  # vol points
    assert bates.params.lam == pytest.approx(truth.lam, rel=0.05)


def test_bates_finds_no_jumps_in_heston_data(synthetic_quotes):
    truth = HestonParams(v0=0.3, kappa=2.5, theta=0.4, xi=1.2, rho=-0.35)
    quotes = synthetic_quotes(truth, [0.05, 0.5])
    heston = calibrate_heston(quotes)
    bates = calibrate_bates(quotes, heston.params)
    assert bates.cost <= heston.cost + 1e-20  # nested, so never worse (both are rounding here)
    assert "lam at lower bound" in bates.at_bounds
    np.testing.assert_allclose(bates.params.heston.as_array(), truth.as_array(), rtol=1e-4)


def test_binding_bounds_names_parameters_on_their_bounds():
    x = np.array([UPPER[0], 1.0, LOWER[2], 2.0, 0.0, UPPER[5], 0.0, LOWER[7]])
    names = ("v0", "kappa", "theta", "xi", "rho", "lam", "mu_j", "sigma_j")
    assert binding_bounds(x, LOWER, UPPER, names) == (
        "v0 at upper bound",
        "theta at lower bound",
        "lam at upper bound",
        "sigma_j at lower bound",
    )


def test_variants_change_one_jump_bound_each():
    for changes in VARIANTS.values():
        lower, upper = variant_bounds(changes)
        changed = np.flatnonzero((lower != LOWER) | (upper != UPPER))
        assert changed.size == 1
        assert lower[changed[0]] < upper[changed[0]]

"""Heston pricer, Monte Carlo and calibration (synthetic data only inside these tests)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.heston import HestonParams, characteristic_function, mc_price, price, price_quad
from ivcrypto.heston.calibrate import ExpiryQuotes, calibrate_heston
from ivcrypto.heston.charfunc import complex_log1p
from ivcrypto.implied_vol import implied_vols

# Lewis (2000), "Option Valuation under Stochastic Volatility": S = 100, r = 1%, q = 2%,
# T = 1, v0 = 0.04, kappa = 4, theta = 0.25, xi = 1, rho = -0.5. A 30 digit evaluation of the
# integral reproduces every digit below.
LEWIS = HestonParams(v0=0.04, kappa=4.0, theta=0.25, xi=1.0, rho=-0.5)
LEWIS_CALLS = {
    80.0: 26.774758743998854,
    90.0: 20.933349000596710,
    100.0: 16.070154917028834,
    110.0: 12.132211516709830,
    120.0: 9.024913483457836,
}
# Fang and Oosterlee (2008), the COS paper: S = K = 100, T = 1, zero rates. The Feller
# condition fails (2 kappa theta / xi^2 = 0.38). They publish 5.785155450; a 30 digit
# evaluation gives 5.785155434376, so the published value is good to about 2e-8.
FANG_OOSTERLEE = HestonParams(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
FANG_OOSTERLEE_PUBLISHED = 5.785155450
FANG_OOSTERLEE_30_DIGITS = 5.78515543437619

CRYPTO_LIKE = HestonParams(v0=0.25, kappa=3.0, theta=0.35, xi=3.0, rho=-0.3)


def test_lewis_reference_values():
    r, q, T = 0.01, 0.02, 1.0
    forward, discount = 100.0 * math.exp((r - q) * T), math.exp(-r * T)
    strikes = np.array(sorted(LEWIS_CALLS))
    calls = discount * price(forward, strikes, T, LEWIS, True)
    np.testing.assert_allclose(calls, [LEWIS_CALLS[k] for k in strikes], rtol=0, atol=1e-10)


def test_fang_oosterlee_reference_value():
    value = float(price(100.0, np.array([100.0]), 1.0, FANG_OOSTERLEE, True)[0])
    assert value == pytest.approx(FANG_OOSTERLEE_30_DIGITS, abs=1e-10)
    assert value == pytest.approx(FANG_OOSTERLEE_PUBLISHED, abs=2e-8)
    assert not FANG_OOSTERLEE.feller_satisfied
    assert FANG_OOSTERLEE.feller_ratio == pytest.approx(2 * 1.5768 * 0.0398 / 0.5751**2)


@pytest.mark.parametrize("T", [2 / 365, 10 / 365, 0.25, 1.0, 3.0])
def test_fast_pricer_matches_adaptive_quadrature(T):
    F = 86_000.0
    strikes = F * np.exp(np.linspace(-3.0, 3.0, 13) * np.sqrt(0.3 * T))
    calls = strikes >= F
    fast = price(F, strikes, T, CRYPTO_LIKE, calls)
    slow = [price_quad(F, k, T, CRYPTO_LIKE, c) for k, c in zip(strikes, calls, strict=True)]
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-10 * F)


def test_put_call_parity_on_the_forward():
    F, T = 86_000.0, 0.3
    strikes = F * np.exp(np.linspace(-1, 1, 9))
    calls = price(F, strikes, T, CRYPTO_LIKE, True)
    puts = price(F, strikes, T, CRYPTO_LIKE, False)
    np.testing.assert_allclose(calls - puts, F - strikes, rtol=0, atol=1e-9 * F)


def test_converges_to_black76_as_vol_of_vol_vanishes():
    # With xi -> 0 and v0 = theta the variance stays at theta: Black 76 with sigma^2 = theta.
    theta, F = 0.36, 86_000.0
    errors = {}
    for xi in (1e-2, 1e-4, 1e-6):
        params = HestonParams(v0=theta, kappa=1.0, theta=theta, xi=xi, rho=-0.4)
        worst = 0.0
        for T in (2 / 365, 0.25, 1.0, 3.0):
            strikes = F * np.exp(np.linspace(-2, 2, 11) * np.sqrt(theta * T))
            calls = strikes >= F
            heston = price(F, strikes, T, params, calls)
            black = black76.price(F, strikes, T, math.sqrt(theta), calls)
            worst = max(worst, float(np.max(np.abs(heston - black))) / F)
        errors[xi] = worst
    assert errors[1e-6] < 1e-7
    # The gap closes linearly in xi, down to the smallest value tested.
    assert errors[1e-2] / errors[1e-4] == pytest.approx(100, rel=0.1)
    assert errors[1e-4] / errors[1e-6] == pytest.approx(100, rel=0.1)


def test_little_trap_is_continuous_at_long_maturities():
    # Heston's original form crosses the branch cut of the complex log here; the little trap
    # form must stay continuous in u and agree with adaptive quadrature.
    params = HestonParams(v0=0.05, kappa=0.3, theta=0.1, xi=1.5, rho=-0.8)
    T = 30.0
    u = np.linspace(0.0, 30.0, 30_001)
    phi = characteristic_function(u - 0.5j, T, params)
    assert np.max(np.abs(np.diff(phi))) < 1e-2
    strikes = np.array([50.0, 100.0, 200.0])
    np.testing.assert_allclose(
        price(100.0, strikes, T, params, True),
        [price_quad(100.0, k, T, params, True) for k in strikes],
        rtol=0,
        atol=1e-9 * 100,
    )


def test_complex_log1p_is_accurate_for_tiny_arguments():
    z = np.array([1e-12 * (1 + 2j), 3e-8 - 1e-9j, 0.5 + 0.5j])
    exact = np.array([z[0] - z[0] ** 2 / 2, z[1] - z[1] ** 2 / 2 + z[1] ** 3 / 3, np.log(1 + z[2])])
    np.testing.assert_allclose(complex_log1p(z), exact, rtol=1e-15)


@pytest.mark.parametrize(
    ("params", "F", "T"),
    [
        (LEWIS, 100.0, 1.0),
        (FANG_OOSTERLEE, 100.0, 1.0),
        (HestonParams(v0=0.12, kappa=3.0, theta=0.2, xi=2.5, rho=-0.2), 86_000.0, 10 / 365),
    ],
)
def test_monte_carlo_agrees_with_the_fourier_pricer(params, F, T):
    strikes = F * np.exp(np.linspace(-1.5, 1.5, 7) * np.sqrt(params.theta * T))
    calls = strikes >= F
    mc = mc_price(F, strikes, T, params, calls, n_paths=100_000, steps_per_year=100, seed=7)
    exact = price(F, strikes, T, params, calls)
    z = (mc.price - exact) / mc.std_error
    assert np.all(np.abs(z) < 4.0), z
    assert abs(mc.forward_mean - F) < 4.0 * mc.forward_std_error  # the forward is a martingale


def synthetic_quotes(params, maturities, F=86_000.0):
    """Quotes priced by Heston itself, with a 1 vol point band, for a recovery test."""
    quotes = []
    for T in maturities:
        k = np.linspace(-2.5, 2.0, 15) * np.sqrt(params.theta * T)
        strikes = F * np.exp(k)
        calls = k >= 0
        mid = price(F, strikes, T, params, calls)
        iv = implied_vols(mid, F, strikes, T, calls)
        quotes.append(
            ExpiryQuotes(
                expiry_code=f"T{T:.3f}",
                T=T,
                F=F,
                instrument_name=np.array([f"q{i}" for i in range(k.size)], dtype=object),
                K=strikes,
                k=k,
                is_call=calls,
                mid=mid,
                iv_bid=iv - 0.005,
                iv_mid=iv,
                iv_ask=iv + 0.005,
                vega=black76.vega(F, strikes, T, iv),
            )
        )
    return quotes


def test_calibration_recovers_known_parameters():
    truth = HestonParams(v0=0.3, kappa=2.5, theta=0.4, xi=1.2, rho=-0.35)
    calibration = calibrate_heston(synthetic_quotes(truth, [0.05, 0.25, 0.75, 1.5]))
    np.testing.assert_allclose(
        calibration.params.as_array(), truth.as_array(), rtol=1e-4, atol=1e-6
    )
    assert calibration.residuals["in_band"].all()
    assert calibration.summary()["rmse_vol"].max() < 1e-3

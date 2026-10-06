from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from ivcrypto import black76

F_GRID = 100.0
K_GRID = np.array([40.0, 70.0, 90.0, 100.0, 110.0, 150.0, 250.0])
T_GRID = np.array([2 / 365, 0.25, 1.0, 3.0])
SIGMA_GRID = np.array([0.05, 0.4, 1.2, 3.0])


def grid():
    K, T, sigma = np.meshgrid(K_GRID, T_GRID, SIGMA_GRID, indexing="ij")
    return K.ravel(), T.ravel(), sigma.ravel()


@pytest.mark.parametrize("discount", [1.0, 0.97])
def test_put_call_parity(discount):
    K, T, sigma = grid()
    call = black76.price(F_GRID, K, T, sigma, True, discount)
    put = black76.price(F_GRID, K, T, sigma, False, discount)
    np.testing.assert_allclose(call - put, discount * (F_GRID - K), atol=1e-11 * F_GRID)


def test_textbook_futures_option():
    # Hull, Options, Futures, and Other Derivatives, Black's model example: futures and
    # strike 20, rate 9%, four months, volatility 25%; call and put are both $1.12.
    discount = math.exp(-0.09 * 4 / 12)
    for is_call in (True, False):
        value = float(black76.price(20.0, 20.0, 4 / 12, 0.25, is_call, discount))
        assert value == pytest.approx(1.12, abs=0.005)


def test_at_the_money_closed_form():
    s = 0.2
    expected = 100.0 * (2 * norm.cdf(s / 2) - 1)
    assert float(black76.price(100.0, 100.0, 1.0, s, True)) == pytest.approx(expected, rel=1e-14)


def test_prices_lie_within_no_arbitrage_bounds_and_rise_with_vol():
    K, T, _ = grid()
    previous = None
    for sigma in (0.01, 0.1, 0.5, 2.0, 8.0):
        for is_call in (True, False):
            value = black76.price(F_GRID, K, T, sigma, is_call)
            assert np.all(value >= black76.intrinsic_value(F_GRID, K, is_call) - 1e-12)
            assert np.all(value <= black76.upper_bound(F_GRID, K, is_call) + 1e-12)
        call = black76.price(F_GRID, K, T, sigma, True)
        if previous is not None:
            assert np.all(call >= previous - 1e-12)
        previous = call


def test_zero_vol_or_zero_time_is_intrinsic():
    for is_call in (True, False):
        intrinsic = black76.intrinsic_value(F_GRID, K_GRID, is_call)
        np.testing.assert_allclose(black76.price(F_GRID, K_GRID, 1.0, 0.0, is_call), intrinsic)
        np.testing.assert_allclose(black76.price(F_GRID, K_GRID, 0.0, 0.5, is_call), intrinsic)


def test_vega_matches_a_finite_difference():
    # Vega is the same for a call and a put, so difference the OTM option: an ITM price is
    # mostly intrinsic value and its finite difference would lose digits to cancellation.
    K, T, sigma = grid()
    otm_call = F_GRID <= K
    h = 1e-6
    numeric = (
        black76.price(F_GRID, K, T, sigma + h, otm_call)
        - black76.price(F_GRID, K, T, sigma - h, otm_call)
    ) / (2 * h)
    np.testing.assert_allclose(black76.vega(F_GRID, K, T, sigma), numeric, rtol=1e-6, atol=1e-9)


def test_prices_scale_with_the_forward():
    K, T, sigma = grid()
    for is_call in (True, False):
        base = black76.price(F_GRID, K, T, sigma, is_call)
        scaled = black76.price(3.0 * F_GRID, 3.0 * K, T, sigma, is_call)
        np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-12, atol=1e-12)


def test_broadcasting():
    K = np.array([[90.0], [100.0], [110.0]])
    sigma = np.array([[0.2, 0.4, 0.6, 0.8]])
    assert black76.price(100.0, K, 0.5, sigma, True).shape == (3, 4)
    assert black76.vega(100.0, K, 0.5, sigma).shape == (3, 4)

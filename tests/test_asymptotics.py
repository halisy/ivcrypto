"""Short maturity limits of the ATM smile: Heston's level off, a jump diffusion's do not.

The README reads the market's ATM term structure through two facts, checked here numerically
rather than assumed:

* In Heston the ATM skew and curvature converge to finite limits as T -> 0, the skew to
  rho xi / (4 sqrt(v0)).
* In Merton's jump diffusion the ATM skew also converges, to lam (E[e^Y] - 1) / sigma, but the
  curvature grows like T^(-1/2). To first order in lam T the jumps add
  lam T (c N(d1) + E[(e^Y - e^k)^+]) to the call price, with c = 1 - E[e^Y]; dividing by the
  vega and differentiating twice at k = 0 gives
  curvature * sqrt(T) -> lam sqrt(2 pi) (c / 2 + E[(e^Y - 1)^+]) / sigma^2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from ivcrypto import black76
from ivcrypto.compare import model_atm
from ivcrypto.heston.charfunc import HestonParams
from ivcrypto.implied_vol import implied_vols

DAY = 1.0 / 365.0
HESTON = HestonParams(v0=0.105, kappa=15.3, theta=0.185, xi=4.96, rho=-0.11)  # about BTC's
SIGMA, LAM, MU, DELTA = 0.30, 3.0, -0.10, 0.15  # 30% diffusion, 3 jumps a year of N(-10%, 15%)
MEAN_JUMP = math.exp(MU + 0.5 * DELTA**2)  # E[e^Y]


def merton_call(k: np.ndarray, T: float) -> np.ndarray:
    """Undiscounted call on a unit forward, as a Poisson mixture of Black 76 prices."""
    price = np.zeros_like(k)
    for n in range(40):
        weight = math.exp(-LAM * T) * (LAM * T) ** n / math.factorial(n)
        forward = math.exp(-LAM * (MEAN_JUMP - 1.0) * T + n * (MU + 0.5 * DELTA**2))
        vol = math.sqrt(SIGMA**2 + n * DELTA**2 / T)
        price += weight * black76.price(forward, np.exp(k), T, vol, True)
    return price


def merton_atm(T: float) -> tuple[float, float]:
    """ATM skew and curvature of the Merton smile by central differences."""
    h = 0.05 * SIGMA * math.sqrt(T)
    k = np.array([-h, 0.0, h])
    vols = implied_vols(merton_call(k, T), 1.0, np.exp(k), T, True)
    return (vols[2] - vols[0]) / (2 * h), (vols[2] - 2 * vols[1] + vols[0]) / h**2


def test_heston_atm_skew_and_curvature_converge_as_maturity_shrinks():
    ts = model_atm(HESTON, np.array([1 / 16, 1 / 4, 1]) * DAY)
    limit = HESTON.rho * HESTON.xi / (4 * math.sqrt(HESTON.v0))
    assert ts["atm_skew"].iloc[0] == pytest.approx(limit, rel=0.01)
    curvature = ts["atm_curvature"].to_numpy()
    assert (curvature > 0).all()
    # Sixteen times shorter, and the curvature has not grown (T^-1/2 would multiply it by 4);
    # the steps shrink, so it converges.
    assert curvature[0] < 1.1 * curvature[2]
    assert abs(curvature[0] - curvature[1]) < abs(curvature[1] - curvature[2])


def test_jump_diffusion_atm_curvature_grows_like_inverse_square_root_of_maturity():
    c = 1.0 - MEAN_JUMP
    jump_payoff = MEAN_JUMP * norm.cdf((MU + DELTA**2) / DELTA) - norm.cdf(MU / DELTA)
    predicted = LAM * math.sqrt(2 * math.pi) * (0.5 * c + jump_payoff) / SIGMA**2
    skew, curvature = merton_atm(0.001 * DAY)
    assert skew == pytest.approx(-LAM * c / SIGMA, rel=0.005)
    assert curvature * math.sqrt(0.001 * DAY) == pytest.approx(predicted, rel=0.02)
    _, longer = merton_atm(0.016 * DAY)
    assert longer * math.sqrt(0.016 * DAY) == pytest.approx(predicted, rel=0.05)
    assert curvature / longer == pytest.approx(4.0, rel=0.05)  # 16 times shorter, 4 times higher

"""SSVI: one surface across all expiries, free of static arbitrage by construction.

Gatheral and Jacquier (2014) parameterize the whole surface by the ATM total variance
theta_t of each expiry and three global parameters:

    w(k, theta) = theta / 2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),
    phi(theta)  = eta / (theta^gamma (1 + theta)^(1 - gamma))       (power law)

Sufficient conditions (their Section 4), all imposed here as simple bounds:

* no butterfly arbitrage: eta (1 + |rho|) <= 2 and 0 < gamma <= 1/2, which imply their
  conditions theta phi (1 + |rho|) < 4 and theta phi^2 (1 + |rho|) <= 4 for every theta;
* no calendar arbitrage: theta_t nondecreasing in t (for gamma <= 1, theta phi(theta) is
  nondecreasing and within their upper bound).

As with SVI, the optimizer works with "wing" variables e_L = eta (1 - rho) and
e_R = eta (1 + rho) in [1e-4, 2], so eta (1 + |rho|) = max(e_L, e_R) <= 2 is a box with no
kink at rho = 0, and theta is built from a first level plus nonnegative increments.

Every SSVI slice is a raw SVI slice (a = theta (1 - rho^2) / 2, b = theta phi / 2, rho,
m = -rho / phi, sigma = sqrt(1 - rho^2) / phi), so the arbitrage checker can verify the
surface independently, also between expiries: interpolating theta linearly in maturity keeps
it nondecreasing, which gives arbitrage free interpolation. The price of this guarantee is
flexibility: three global parameters and one level per expiry instead of five free
parameters per expiry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares

from ivcrypto.cleaning import CleanQuotes
from ivcrypto.config import SVIConfig
from ivcrypto.svi.fit import SliceQuotes, SVISurface, residual_scale, slice_quotes
from ivcrypto.svi.raw import SVIParams

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

ETA_WING_MIN = 1e-4
ETA_WING_MAX = 2.0
GAMMA_MIN = 0.01
GAMMA_MAX = 0.5
THETA_MIN = 1e-8


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float

    def phi(self, theta: npt.ArrayLike) -> FloatArray:
        theta = np.asarray(theta, dtype=float)
        return self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def total_variance(self, k: npt.ArrayLike, theta: float) -> FloatArray:
        k = np.asarray(k, dtype=float)
        phi = float(self.phi(theta))
        root = np.sqrt((phi * k + self.rho) ** 2 + 1.0 - self.rho**2)
        return 0.5 * theta * (1.0 + self.rho * phi * k + root)

    def slice_params(self, theta: float) -> SVIParams:
        """The raw SVI parameters of the slice with ATM total variance ``theta``."""
        phi = float(self.phi(theta))
        return SVIParams(
            a=0.5 * theta * (1.0 - self.rho**2),
            b=0.5 * theta * phi,
            rho=self.rho,
            m=-self.rho / phi,
            sigma=float(np.sqrt(1.0 - self.rho**2)) / phi,
        )

    def violations(self, tol: float = 1e-12) -> list[str]:
        """Which of the sufficient no arbitrage conditions fail (empty when all hold)."""
        broken = []
        if not abs(self.rho) < 1.0:
            broken.append("|rho| < 1")
        if not self.eta > 0:
            broken.append("eta > 0")
        if self.eta * (1.0 + abs(self.rho)) > 2.0 + tol:
            broken.append("eta (1 + |rho|) <= 2")
        if not 0.0 < self.gamma <= 0.5 + tol:
            broken.append("0 < gamma <= 1/2")
        return broken


@dataclass(frozen=True)
class SSVIFit:
    params: SSVIParams
    expiries: tuple[str, ...]
    T: FloatArray
    theta: FloatArray
    weighting: str
    cost: float
    start_costs: tuple[float, ...]
    residuals: pd.DataFrame = field(repr=False)

    def theta_at(self, T: npt.ArrayLike) -> FloatArray:
        """ATM total variance at any maturity: linear in T between expiries, and linear to
        zero before the first. Beyond the last expiry the ATM variance rate is held."""
        T = np.asarray(T, dtype=float)
        knots_T = np.concatenate([[0.0], self.T])
        knots_theta = np.concatenate([[0.0], self.theta])
        inside = np.interp(T, knots_T, knots_theta)
        beyond = self.theta[-1] * T / self.T[-1]
        return np.where(self.T[-1] < T, beyond, inside)

    def slice_params(self, T: float) -> SVIParams:
        return self.params.slice_params(float(self.theta_at(T)))

    def summary(self) -> pd.DataFrame:
        rows = []
        for expiry, part in self.residuals.groupby("expiry_code", sort=False):
            error = part["iv_ssvi"] - part["iv_mid"]
            rows.append(
                {
                    "expiry_code": expiry,
                    "days": float(part["T"].iloc[0]) * 365.0,
                    "theta": float(self.theta[self.expiries.index(expiry)]),
                    "n": len(part),
                    "rmse_vol": float(100.0 * np.sqrt(np.mean(error**2))),
                    "max_error_vol": float(100.0 * np.max(np.abs(error))),
                    "in_band_share": float(part["in_band"].mean()),
                }
            )
        return pd.DataFrame(rows)


def fit_ssvi(clean: CleanQuotes, svi: SVISurface, config: SVIConfig | None = None) -> SSVIFit:
    """Fit SSVI to the same quotes as the per expiry SVI fits, started from them."""
    expiries = tuple(svi.fits)
    atm = [float(svi.fits[code].params.total_variance(0.0)) for code in expiries]
    rho0 = float(np.mean([fit.params.rho for fit in svi.fits.values()]))
    return fit_ssvi_slices([slice_quotes(clean, code) for code in expiries], atm, rho0, config)


def fit_ssvi_slices(
    slices: list[SliceQuotes],
    atm_total_variance: npt.ArrayLike,
    rho0: float = 0.0,
    config: SVIConfig | None = None,
) -> SSVIFit:
    """Fit SSVI to expiry slices in maturity order, starting theta from ``atm_total_variance``."""
    config = config if config is not None else SVIConfig()
    expiries = tuple(data.expiry_code for data in slices)
    scales = [residual_scale(data, config.weighting) for data in slices]
    T = np.array([data.T for data in slices])
    if np.any(np.diff(T) <= 0):
        raise ValueError("slices must be in increasing maturity order")
    w_max = max(float(data.w_mid.max()) for data in slices)

    atm = np.maximum.accumulate(np.asarray(atm_total_variance, dtype=float))
    theta0 = np.concatenate([[atm[0]], np.diff(atm)])
    rho0 = float(np.clip(rho0, -0.9, 0.9))

    n = len(expiries)
    lower = np.concatenate([[ETA_WING_MIN, ETA_WING_MIN, GAMMA_MIN, THETA_MIN], np.zeros(n - 1)])
    upper = np.concatenate(
        [[ETA_WING_MAX, ETA_WING_MAX, GAMMA_MAX, 2 * w_max], np.full(n - 1, w_max)]
    )

    def unpack(x: FloatArray) -> tuple[SSVIParams, FloatArray]:
        e_left, e_right, gamma = x[0], x[1], x[2]
        params = SSVIParams(
            rho=float((e_right - e_left) / (e_right + e_left)),
            eta=float(0.5 * (e_left + e_right)),
            gamma=float(gamma),
        )
        return params, np.cumsum(x[3:])

    def residuals(x: FloatArray) -> FloatArray:
        params, theta = unpack(x)
        return np.concatenate(
            [
                scale * (params.total_variance(data.k, th) - data.w_mid)
                for data, scale, th in zip(slices, scales, theta, strict=True)
            ]
        )

    starts = []
    for gamma in (0.1, 0.25, 0.4, 0.5):
        for eta_fraction in (0.3, 0.6, 0.9):
            eta = eta_fraction * 2.0 / (1.0 + abs(rho0))
            wings = [eta * (1.0 - rho0), eta * (1.0 + rho0)]
            starts.append(np.concatenate([wings, [gamma], theta0]))
    results = []
    for start in starts:
        x0 = np.clip(start, lower + 1e-12, upper - 1e-12)
        results.append(
            least_squares(
                residuals, x0, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=3000
            )
        )
    costs = np.array([2.0 * r.cost for r in results])
    best = results[int(np.argmin(costs))]
    polished = least_squares(
        residuals, best.x, bounds=(lower, upper), method="dogbox", x_scale="jac", max_nfev=3000
    )
    final = polished if 2.0 * polished.cost <= costs.min() * (1.0 + 1e-9) else best
    params, theta = unpack(final.x)
    return SSVIFit(
        params=params,
        expiries=expiries,
        T=T,
        theta=theta,
        weighting=config.weighting,
        cost=2.0 * float(final.cost),
        start_costs=tuple(float(c) for c in costs),
        residuals=_residual_table(params, slices, theta),
    )


def _residual_table(
    params: SSVIParams, slices: list[SliceQuotes], theta: FloatArray
) -> pd.DataFrame:
    parts = []
    for data, th in zip(slices, theta, strict=True):
        iv = np.sqrt(params.total_variance(data.k, th) / data.T)
        parts.append(
            pd.DataFrame(
                {
                    "expiry_code": data.expiry_code,
                    "T": data.T,
                    "instrument_name": data.instrument_name,
                    "k": data.k,
                    "iv_bid": data.iv_bid,
                    "iv_mid": data.iv_mid,
                    "iv_ask": data.iv_ask,
                    "iv_ssvi": iv,
                    "error_vol": 100.0 * (iv - data.iv_mid),
                    "in_band": (iv >= data.iv_bid - 1e-12) & (iv <= data.iv_ask + 1e-12),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)

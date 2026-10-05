"""Heston characteristic function in the numerically stable "little trap" form.

Under the forward measure the forward is a martingale:

    dF = sqrt(v) F dW1,    dv = kappa (theta - v) dt + xi sqrt(v) dW2,    d<W1, W2> = rho dt.

The characteristic function of X_T = ln(F_T / F_0) is exp(C(u, T) + D(u, T) v0) with
(Albrecher, Mayer, Schoutens and Tistaert 2007, "The little Heston trap")

    beta = kappa - rho xi i u,   d = sqrt(beta^2 + xi^2 (i u + u^2)),   g = (beta - d) / (beta + d)
    C = kappa theta / xi^2 [ (beta - d) T - 2 ln((1 - g e^{-dT}) / (1 - g)) ]
    D = (beta - d) / xi^2 * (1 - e^{-dT}) / (1 - g e^{-dT}).

Heston's original form uses g' = 1 / g and e^{+dT}; its complex logarithm then crosses the
branch cut for long maturities and the price jumps. In the form above |g e^{-dT}| < 1 and the
logarithm stays on its principal branch (d is taken with nonnegative real part).

The implementation also avoids the cancellation in beta - d when xi is small (the Black 76
limit), using the identity beta - d = -xi^2 (i u + u^2) / (beta + d), and evaluates the
logarithm as log1p(g (1 - e^{-dT}) / (1 - g)) with an accurate complex log1p. Both are
exact rewrites of the formulas above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / xi^2: at least 1 means the variance never reaches zero."""
        return 2.0 * self.kappa * self.theta / self.xi**2

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio >= 1.0

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])


def complex_log1p(z: npt.ArrayLike) -> ComplexArray:
    """log(1 + z) accurate for tiny complex z.

    numpy's complex ``log1p`` evaluates log(1 + z) directly and loses the relative accuracy
    for small |z| (4e-5 relative error at |z| = 2e-12 with numpy 2.4), which wrecks the
    xi -> 0 limit. Below |z| = 1e-3 a Taylor series has truncation error under 2e-16.
    """
    z = np.asarray(z, dtype=complex)
    series = z * (1.0 - z * (0.5 - z * (1.0 / 3.0 - z * (0.25 - z * 0.2))))
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.log(1.0 + z)
    return np.where(np.abs(z) < 1e-3, series, direct)


def characteristic_function(u: npt.ArrayLike, T: float, p: HestonParams) -> ComplexArray:
    """E[exp(i u ln(F_T / F_0))] for real or complex ``u``."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    a = iu + u * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + p.xi * p.xi * a)
    beta_plus_d = beta + d
    beta_minus_d_over_xi2 = -a / beta_plus_d  # (beta - d) / xi^2 without cancellation
    g = p.xi * p.xi * beta_minus_d_over_xi2 / beta_plus_d  # (beta - d) / (beta + d)
    exp_dt = np.exp(-d * T)
    log_term = complex_log1p(g * (1.0 - exp_dt) / (1.0 - g))  # ln((1 - g e^{-dT}) / (1 - g))
    C = p.kappa * p.theta * (beta_minus_d_over_xi2 * T - 2.0 * log_term / p.xi**2)
    D = beta_minus_d_over_xi2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return np.exp(C + D * p.v0)

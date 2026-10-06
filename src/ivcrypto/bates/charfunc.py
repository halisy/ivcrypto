"""Bates (1996): Heston stochastic variance plus lognormal jumps in the forward.

Under the forward measure

    dF / F = sqrt(v) dW1 + (e^J - 1) dN - lam m dt,    m = E[e^J] - 1 = exp(mu_j + sigma_j^2/2) - 1,

with the Heston variance of ``heston/charfunc.py``, N a Poisson process with intensity lam
(jumps per year) independent of everything else, and log jump sizes J ~ N(mu_j, sigma_j^2).
The compensator lam m dt keeps the forward a martingale. Since the jumps are independent of
the diffusion, the characteristic function of ln(F_T / F_0) factorizes:

    phi(u) = phi_Heston(u) exp(lam T (exp(i u mu_j - u^2 sigma_j^2 / 2) - 1 - i u m)).

With lam = 0 this is Heston; with xi -> 0 and v0 = theta it is Merton's (1976) jump
diffusion, whose prices are a Poisson mixture of Black 76 prices. Tests check both limits.

Why lognormal jumps: they keep the characteristic function closed form, so Bates slots into
the Lewis pricer and the calibration of Heston unchanged, and they are the standard model.
Kou's double exponential jumps would allow different up and down tails at one more
parameter; Duffie, Pan and Singleton's SVJJ adds jumps in variance. Both are natural
extensions if a single lognormal jump size proves too rigid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ivcrypto.heston.charfunc import ComplexArray, HestonParams
from ivcrypto.heston.charfunc import characteristic_function as heston_characteristic_function


@dataclass(frozen=True)
class BatesParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    lam: float
    """Jump intensity, jumps per year."""
    mu_j: float
    """Mean log jump size."""
    sigma_j: float
    """Standard deviation of the log jump size."""

    @property
    def heston(self) -> HestonParams:
        """The stochastic variance part."""
        return HestonParams(self.v0, self.kappa, self.theta, self.xi, self.rho)

    @property
    def jump_mean(self) -> float:
        """m = E[e^J] - 1, the mean relative jump of the forward."""
        return float(np.expm1(self.mu_j + 0.5 * self.sigma_j**2))

    @property
    def jump_variance(self) -> float:
        """lam E[J^2]: the annual quadratic variation contributed by jumps."""
        return self.lam * (self.mu_j**2 + self.sigma_j**2)

    @property
    def feller_ratio(self) -> float:
        return self.heston.feller_ratio

    @property
    def feller_satisfied(self) -> bool:
        return self.heston.feller_satisfied

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.v0, self.kappa, self.theta, self.xi, self.rho, self.lam, self.mu_j, self.sigma_j]
        )

    def cf(self, u: npt.ArrayLike, T: float) -> ComplexArray:
        """Characteristic function of ln(F_T / F_0), see :func:`characteristic_function`."""
        return characteristic_function(u, T, self)

    def oscillation_rate(self, T: float) -> float:
        """Bound on the phase rate the jumps add along the pricing contour: the compensator
        drift lam T m, plus at most lam T (|mu_j| + sigma_j) from the jump term itself."""
        return self.lam * T * (abs(self.jump_mean) + abs(self.mu_j) + self.sigma_j)


def jump_characteristic_function(u: npt.ArrayLike, T: float, p: BatesParams) -> ComplexArray:
    """The compensated compound Poisson factor of the Bates characteristic function."""
    u = np.asarray(u, dtype=complex)
    jumps = np.exp(1j * u * p.mu_j - 0.5 * u * u * p.sigma_j**2) - 1.0
    return np.exp(p.lam * T * (jumps - 1j * u * p.jump_mean))


def characteristic_function(u: npt.ArrayLike, T: float, p: BatesParams) -> ComplexArray:
    """E[exp(i u ln(F_T / F_0))] under Bates, for real or complex ``u``."""
    return heston_characteristic_function(u, T, p.heston) * jump_characteristic_function(u, T, p)

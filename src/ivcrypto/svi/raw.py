"""Raw SVI (Gatheral 2004) for one expiry, in total implied variance.

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)),    k = ln(K/F),  w = iv^2 T

A slice is admissible when b >= 0, |rho| < 1, sigma > 0 and its minimum total variance
a + b sigma sqrt(1 - rho^2) is nonnegative. We also impose Roger Lee's moment bound on the
wings: the slopes b (1 - rho) and b (1 + rho) of w for large |k| cannot exceed 2, i.e.
b (1 + |rho|) <= 2. Every arbitrage free slice satisfies it (it follows from finite
expectations), so the bound excludes nothing legitimate while keeping extrapolated wings
sane.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

LEE_MAX_SLOPE = 2.0


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: npt.ArrayLike) -> FloatArray:
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma))

    def derivatives(self, k: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Total variance and its first and second derivatives in k."""
        x = np.asarray(k, dtype=float) - self.m
        root = np.sqrt(x * x + self.sigma * self.sigma)
        w = self.a + self.b * (self.rho * x + root)
        dw = self.b * (self.rho + x / root)
        d2w = self.b * self.sigma * self.sigma / root**3
        return w, dw, d2w

    def implied_vol(self, k: npt.ArrayLike, T: float) -> FloatArray:
        """Black implied volatility; NaN where the total variance is negative."""
        w = self.total_variance(k)
        with np.errstate(invalid="ignore"):
            return np.sqrt(w / T)

    @property
    def min_variance(self) -> float:
        """Minimum of w over k, reached at :attr:`argmin`."""
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho * self.rho)

    @property
    def argmin(self) -> float:
        return self.m - self.rho * self.sigma / np.sqrt(1.0 - self.rho * self.rho)

    @property
    def wing_slopes(self) -> tuple[float, float]:
        """Asymptotic slopes of w: (left, as k -> -inf, in absolute value; right)."""
        return self.b * (1.0 - self.rho), self.b * (1.0 + self.rho)

    def violations(self, tol: float = 1e-12) -> list[str]:
        """Names of the admissibility conditions this slice breaks (empty when admissible)."""
        broken = []
        if self.b < -tol:
            broken.append("b >= 0")
        if not abs(self.rho) < 1.0:
            broken.append("|rho| < 1")
        if not self.sigma > 0:
            broken.append("sigma > 0")
        elif abs(self.rho) < 1.0 and self.min_variance < -tol:
            broken.append("minimum variance >= 0")
        if self.b * (1.0 + abs(self.rho)) > LEE_MAX_SLOPE + tol:
            broken.append("Lee wing slope <= 2")
        return broken

    def as_array(self) -> FloatArray:
        return np.array(astuple(self), dtype=float)

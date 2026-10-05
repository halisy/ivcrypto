"""European option prices under Heston, by Lewis's single integral formula.

For a forward F, strike K and k = ln(F / K), with phi the characteristic function of
ln(F_T / F_0) (``charfunc.py``), Lewis (2001) gives the undiscounted call price

    C = F - sqrt(F K) / pi * I(k),   I(k) = int_0^inf Re[e^{i u k} phi(u - i/2)] / (u^2 + 1/4) du,

and by put call parity the put is P = K - sqrt(F K) / pi * I(k).

Why Lewis: one real integral per strike on a fixed contour (no damping parameter to tune,
unlike Carr and Madan's FFT), no strike grid to interpolate (Deribit's strikes are
irregular), and the characteristic function depends only on maturity, so each maturity
evaluates it once and every strike costs one matrix product. COS is faster still, but its
accuracy hinges on a truncation range from cumulants that is delicate for two day
maturities at high volatility, and speed is not our bottleneck.

Quadrature: composite Gauss Legendre on [0, U]. The integrand decays like
|phi(u - i/2)| / u^2, slowly for short maturities with a high vol of vol, so U is found per
maturity by scanning that envelope until it falls below a tolerance. ``price_quad`` is a
slow adaptive reference used only to validate the fast pricer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from ivcrypto.heston.charfunc import HestonParams, characteristic_function

FloatArray = npt.NDArray[np.float64]

ENVELOPE_TOL = 1e-13
"""Integration stops where |phi(u - i/2)| / (u^2 + 1/4) stays below this."""
U_MAX = 20_000.0
NODES_PER_PANEL = 16
NEAR_ORIGIN = 10.0
"""Panels are 0.5 wide on [0, NEAR_ORIGIN] (see ``_nodes``)."""
_GL_X, _GL_W = np.polynomial.legendre.leggauss(NODES_PER_PANEL)


def integration_limit(T: float, params: HestonParams, tol: float = ENVELOPE_TOL) -> float:
    """Smallest U on a geometric grid beyond which the integrand envelope stays below ``tol``."""
    grid = np.geomspace(0.5, U_MAX, 400)
    envelope = np.abs(characteristic_function(grid - 0.5j, T, params)) / (grid * grid + 0.25)
    above = np.flatnonzero(~(envelope < tol))  # NaN counts as not yet converged
    if above.size == 0:
        return float(grid[0])
    return float(grid[min(above[-1] + 1, grid.size - 1)])


def _nodes(upper: float, max_abs_k: float) -> tuple[FloatArray, FloatArray]:
    """Composite Gauss Legendre nodes and weights on [0, upper].

    Near the origin the factor 1 / (u^2 + 1/4) has poles at u = +-i/2, only 0.5 away from
    the real axis; Gauss Legendre converges slowly on panels much wider than that distance
    (2 wide panels leave errors near 1e-10), so the first panels are 0.5 wide. Further out,
    panels only need to resolve the oscillation e^{iuk} (wavelength 2 pi / |k|).
    """
    near = min(upper, NEAR_ORIGIN)
    far_width = min(2.0, 4.0 / max(max_abs_k, 1e-3))
    near_edges = np.linspace(0.0, near, max(1, int(np.ceil(near / 0.5))) + 1)
    far_panels = int(np.ceil((upper - near) / far_width)) if upper > near else 0
    far_edges = np.linspace(near, upper, far_panels + 1)[1:]
    edges = np.concatenate([near_edges, far_edges])
    half = 0.5 * np.diff(edges)
    centre = 0.5 * (edges[:-1] + edges[1:])
    nodes = (centre[:, None] + half[:, None] * _GL_X[None, :]).ravel()
    weights = (half[:, None] * _GL_W[None, :]).ravel()
    return nodes, weights


def lewis_integral(k: npt.ArrayLike, T: float, params: HestonParams) -> FloatArray:
    """I(k) for an array of log moneyness k = ln(F / K) at one maturity."""
    k = np.atleast_1d(np.asarray(k, dtype=float))
    upper = integration_limit(T, params)
    u, w = _nodes(upper, float(np.max(np.abs(k))) if k.size else 0.0)
    phi = characteristic_function(u - 0.5j, T, params)
    scaled = w / (u * u + 0.25)
    phase = np.outer(k, u)
    return np.cos(phase) @ (scaled * phi.real) - np.sin(phase) @ (scaled * phi.imag)


def price(
    F: float,
    K: npt.ArrayLike,
    T: float,
    params: HestonParams,
    is_call: npt.ArrayLike,
) -> FloatArray:
    """Undiscounted Heston prices of calls and puts on the forward F, at one maturity."""
    K = np.asarray(K, dtype=float)
    shape = np.broadcast(K, np.asarray(is_call)).shape
    K_flat = np.broadcast_to(K, shape).ravel()
    calls = np.broadcast_to(np.asarray(is_call, dtype=bool), shape).ravel()
    integral = lewis_integral(np.log(F / K_flat), T, params)
    common = np.sqrt(F * K_flat) / np.pi * integral
    values = np.where(calls, F - common, K_flat - common)
    return values.reshape(shape)


def price_quad(F: float, K: float, T: float, params: HestonParams, is_call: bool) -> float:
    """Adaptive quadrature reference for one option (slow; used for validation)."""
    k = np.log(F / K)

    def integrand(u: float) -> float:
        phi = characteristic_function(u - 0.5j, T, params)
        return float((np.exp(1j * u * k) * phi).real / (u * u + 0.25))

    integral, _ = quad(integrand, 0.0, np.inf, limit=5000, epsabs=1e-14, epsrel=1e-13)
    common = np.sqrt(F * K) / np.pi * integral
    return float(F - common if is_call else K - common)

"""Monte Carlo under Heston, used only to validate the Fourier pricer.

Variance: Andersen's (2008) quadratic exponential (QE) scheme with switching level
psi_c = 1.5. It matches the first two conditional moments of v and handles the mass near
zero that Euler schemes mishandle when the Feller condition fails (as it does for crypto).

Log forward: Andersen's discretization with central weights (gamma1 = gamma2 = 1/2),

    X' = X + K0* + K1 v + K2 v' + sqrt(K3 v + K4 v') Z,

with his martingale correction for K0*, so that E[F_{t+dt} | F_t, v_t] = F_t exactly and the
simulated forward is a martingale, as it must be. A control variate on F_T, whose mean F_0
is known exactly, reduces the variance of the price estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ivcrypto.heston.charfunc import HestonParams

FloatArray = npt.NDArray[np.float64]

PSI_CRITICAL = 1.5


def simulate_log_forward(
    params: HestonParams,
    T: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Simulated ln(F_T / F_0) for ``n_paths`` paths of ``n_steps`` QE steps each."""
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    dt = T / n_steps
    decay = np.exp(-kappa * dt)
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = 0.5 * dt * (1.0 - rho * rho)
    k4 = k3
    a_coef = k2 + 0.5 * k4

    v = np.full(n_paths, params.v0)
    x = np.zeros(n_paths)
    for _ in range(n_steps):
        m = theta + (v - theta) * decay
        s2 = v * xi**2 * decay * (1.0 - decay) / kappa + theta * xi**2 * (1.0 - decay) ** 2 / (
            2.0 * kappa
        )
        psi = s2 / (m * m)
        quadratic = psi <= PSI_CRITICAL
        v_next = np.empty(n_paths)
        log_m = np.empty(n_paths)  # ln E[exp(A v') | v], for the martingale correction

        inv = 2.0 / psi[quadratic]
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
        a = m[quadratic] / (1.0 + b2)
        if np.any(a_coef * a >= 0.5):
            raise ValueError("time step too large for the QE martingale correction")
        z = rng.standard_normal(a.size)
        v_next[quadratic] = a * (np.sqrt(b2) + z) ** 2
        log_m[quadratic] = a_coef * b2 * a / (1.0 - 2.0 * a_coef * a) - 0.5 * np.log(
            1.0 - 2.0 * a_coef * a
        )

        exponential = ~quadratic
        p = (psi[exponential] - 1.0) / (psi[exponential] + 1.0)
        beta = (1.0 - p) / m[exponential]
        if np.any(a_coef >= beta):
            raise ValueError("time step too large for the QE martingale correction")
        u = rng.uniform(size=p.size)
        v_next[exponential] = np.where(
            u <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta
        )
        log_m[exponential] = np.log(p + beta * (1.0 - p) / (beta - a_coef))

        k0 = -log_m - (k1 + 0.5 * k3) * v
        x += (
            k0 + k1 * v + k2 * v_next + np.sqrt(k3 * v + k4 * v_next) * rng.standard_normal(n_paths)
        )
        v = v_next
    return x


@dataclass(frozen=True)
class MCResult:
    price: FloatArray
    std_error: FloatArray
    forward_mean: float
    """Sample mean of F_T, which should equal F_0 up to sampling error."""
    forward_std_error: float


def mc_price(
    F: float,
    K: npt.ArrayLike,
    T: float,
    params: HestonParams,
    is_call: npt.ArrayLike,
    *,
    n_paths: int = 100_000,
    steps_per_year: int = 100,
    seed: int = 0,
    control_variate: bool = True,
) -> MCResult:
    """Undiscounted Heston prices by Monte Carlo, with standard errors."""
    rng = np.random.default_rng(seed)
    n_steps = max(4, int(np.ceil(steps_per_year * T)))
    log_forward = simulate_log_forward(params, T, n_paths, n_steps, rng)
    return price_paths(F, K, F * np.exp(log_forward), is_call, control_variate=control_variate)


def price_paths(
    F: float,
    K: npt.ArrayLike,
    forward_T: FloatArray,
    is_call: npt.ArrayLike,
    *,
    control_variate: bool = True,
) -> MCResult:
    """Prices, standard errors and the forward check from simulated terminal forwards."""
    n_paths = forward_T.size
    K = np.atleast_1d(np.asarray(K, dtype=float))
    calls = np.broadcast_to(np.asarray(is_call, dtype=bool), K.shape)
    payoff = np.where(
        calls[:, None],
        np.maximum(forward_T[None, :] - K[:, None], 0.0),
        np.maximum(K[:, None] - forward_T[None, :], 0.0),
    )
    if control_variate:
        centred = forward_T - F  # known mean zero
        beta = (payoff - payoff.mean(axis=1, keepdims=True)) @ centred / (centred @ centred)
        payoff = payoff - beta[:, None] * centred[None, :]
    return MCResult(
        price=payoff.mean(axis=1),
        std_error=payoff.std(axis=1, ddof=1) / np.sqrt(n_paths),
        forward_mean=float(forward_T.mean()),
        forward_std_error=float(forward_T.std(ddof=1) / np.sqrt(n_paths)),
    )

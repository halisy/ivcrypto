"""Monte Carlo under Bates, used only to validate the Fourier pricer.

The jumps are independent of the variance and the diffusion, so they need no time stepping:
the Heston part is simulated with the QE scheme of ``heston/mc.py`` and the terminal log
forward receives N_T ~ Poisson(lam T) jumps, whose sum given N_T is exactly
N(N_T mu_j, N_T sigma_j^2), minus the compensator lam m T. The simulated forward is a
martingale by construction.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ivcrypto.bates.charfunc import BatesParams
from ivcrypto.heston.mc import MCResult, price_paths, simulate_log_forward


def mc_price(
    F: float,
    K: npt.ArrayLike,
    T: float,
    params: BatesParams,
    is_call: npt.ArrayLike,
    *,
    n_paths: int = 100_000,
    steps_per_year: int = 100,
    seed: int = 0,
    control_variate: bool = True,
) -> MCResult:
    """Undiscounted Bates prices by Monte Carlo, with standard errors."""
    rng = np.random.default_rng(seed)
    n_steps = max(4, int(np.ceil(steps_per_year * T)))
    log_forward = simulate_log_forward(params.heston, T, n_paths, n_steps, rng)
    n_jumps = rng.poisson(params.lam * T, n_paths)
    log_forward += (
        n_jumps * params.mu_j
        + np.sqrt(n_jumps) * params.sigma_j * rng.standard_normal(n_paths)
        - params.lam * params.jump_mean * T
    )
    return price_paths(F, K, F * np.exp(log_forward), is_call, control_variate=control_variate)

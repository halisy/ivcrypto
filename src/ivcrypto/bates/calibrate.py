"""Bates calibration, on exactly the objective Heston is calibrated on.

The residuals, weights and quotes are those of ``heston/calibrate.py`` (the ``[heston]``
settings for weighting and expiry weights apply to both models), so the two calibrations
differ only in the model. Bates nests Heston (lam = 0), and the calibrated Heston parameters
with no jumps are always among the refined starting points, so the Bates fit can only match
or improve on Heston's cost.

Bounds: the Heston bounds for the variance parameters; lam in [0, 100] jumps per year,
mu_j in [-1, 1] and sigma_j in [0, 1] for the log jump size. The cap on lam is the one
judgment call: at 100 jumps a year (two a week) jumps stop being rare events and start to act
as a second source of diffusion; ``VARIANTS`` refits with it raised, and with crash only or
rare jumps, to show what the data ask for. Starting points: the calibrated Heston parameters
and the Heston starting grid, each crossed with a few jump intensities, means and sizes; the
best by initial cost are refined.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from ivcrypto.bates.charfunc import BatesParams
from ivcrypto.config import BatesConfig, HestonConfig
from ivcrypto.heston.calibrate import LOWER as HESTON_LOWER
from ivcrypto.heston.calibrate import UPPER as HESTON_UPPER
from ivcrypto.heston.calibrate import Calibration, ExpiryQuotes, FloatArray, fit_model
from ivcrypto.heston.calibrate import starting_points as heston_starting_points
from ivcrypto.heston.charfunc import HestonParams

LOWER = np.concatenate([HESTON_LOWER, [0.0, -1.0, 0.0]])
UPPER = np.concatenate([HESTON_UPPER, [100.0, 1.0, 1.0]])
PARAMETER_NAMES = ("v0", "kappa", "theta", "xi", "rho", "lam", "mu_j", "sigma_j")
JUMP_GRID = (
    (2.0, 10.0, 40.0),  # jumps per year
    (-0.1, 0.05),  # mean log jump
    (0.04, 0.12),  # log jump volatility
)


def calibrate_bates(
    quotes: list[ExpiryQuotes],
    heston: HestonParams,
    objective: HestonConfig | None = None,
    config: BatesConfig | None = None,
    bounds: tuple[FloatArray, FloatArray] = (LOWER, UPPER),
) -> Calibration:
    """One Bates parameter set for all the given expiries, started from a Heston fit."""
    objective = objective if objective is not None else HestonConfig()
    config = config if config is not None else BatesConfig()
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    starts = [np.clip(x, lower, upper) for x in starting_points(quotes, heston)]
    return fit_model(
        quotes,
        objective,
        BatesParams,
        starts,
        (lower, upper),
        n_starts=config.n_starts,
        max_evaluations=config.max_evaluations,
        always_refine=(0,),
        column="iv_bates",
        names=PARAMETER_NAMES,
    )


def starting_points(quotes: list[ExpiryQuotes], heston: HestonParams) -> list[FloatArray]:
    """Heston without jumps first, then variance starts crossed with the jump grid."""
    starts = [np.concatenate([heston.as_array(), [0.0, 0.0, 0.1]])]
    variance_starts = [heston.as_array(), *heston_starting_points(quotes)]
    for variance, jump in product(variance_starts, product(*JUMP_GRID)):
        starts.append(np.concatenate([variance, jump]))
    return starts


VARIANTS: dict[str, dict[str, tuple[float, float]]] = {
    "jumps down on average": {"mu_j": (-1.0, 0.0)},
    "at most 10 jumps a year": {"lam": (0.0, 10.0)},
    "up to 1000 jumps a year": {"lam": (0.0, 1000.0)},
}
"""Diagnostic refits with one jump bound changed: do the data want crashes, rare jumps, or
more jumps than the default cap allows?"""


def variant_bounds(changes: dict[str, tuple[float, float]]) -> tuple[FloatArray, FloatArray]:
    lower, upper = LOWER.copy(), UPPER.copy()
    for name, (lo, hi) in changes.items():
        index = PARAMETER_NAMES.index(name)
        lower[index], upper[index] = lo, hi
    return lower, upper


def calibrate_variants(
    quotes: list[ExpiryQuotes],
    heston: HestonParams,
    objective: HestonConfig | None = None,
    config: BatesConfig | None = None,
) -> dict[str, Calibration]:
    """Bates refitted under each of :data:`VARIANTS`."""
    return {
        name: calibrate_bates(quotes, heston, objective, config, variant_bounds(changes))
        for name, changes in VARIANTS.items()
    }

"""Fitting raw SVI to each expiry slice: constraints, weights, starting points, quality.

Residuals are total variance errors, w_model(k_i) - w_mid_i, scaled by the weighting:

* ``spread`` (default): divided by the quote's bid ask width in total variance,
  w_ask - w_bid. A residual of 0.5 puts the fit at the edge of the band (for a central mid),
  so the objective targets the "inside the band" quality measure, and each quote is trusted
  in proportion to how tightly it is made.
* ``vega``: multiplied by vega / (2 iv T), which turns a total variance error into a price
  error to first order (dprice = vega div and dw = 2 iv T div). Wings, which have little
  vega, get little weight.
* ``uniform``: plain total variance errors.

Constraints become simple bounds through a smooth change of variables, so a bounded least
squares solver enforces them exactly. The optimizer works with the asymptotic wing slopes
s_L = b (1 - rho) and s_R = b (1 + rho) and the minimum total variance v:

    b = (s_L + s_R) / 2,   rho = (s_R - s_L) / (s_R + s_L),   a = v - sigma sqrt(s_L s_R)

    1e-6 <= s_L, s_R <= 2    gives b > 0, |rho| < 1 and Lee's wing bound b (1 + |rho|) <= 2
    v >= 0                   nonnegative minimum variance
    sigma >= 1e-4,           m within the quoted range plus a margin

(b sigma sqrt(1 - rho^2) equals sigma sqrt(s_L s_R).) An earlier version bounded b by
2 / (1 + |rho|) directly; its kink at rho = 0 stalled the solver on symmetric smiles whose
wings sit at the bound.

Starting points: SVI fits are sensitive to initialization, so starts come from De Marco and
Martini's quasi explicit reduction (2009). For fixed (m, sigma) and y = (k - m) / sigma, the
model w = a + d y + c sqrt(y^2 + 1) is linear in (a, d, c), with b = c / sigma and
rho = d / c. A grid of (m, sigma) values therefore costs one small weighted linear least
squares each. The best grid points, projected onto the admissible set, plus a heuristic
start seed the full nonlinear fit. The best result wins, and the agreement between starts
is reported so that a fragile fit is visible.
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
from ivcrypto.svi.raw import LEE_MAX_SLOPE, SVIParams

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

SLOPE_MIN = 1e-6
"""Smallest asymptotic wing slope; keeps |rho| strictly below 1."""
SIGMA_MIN = 1e-4
SIGMA_MAX = 5.0
GRID_SIGMA = (1e-3, 2.0)
"""Range of the geometric sigma grid used for starting points."""
AGREEMENT_RTOL = 1e-4
"""Starts that end within this relative distance of the final cost count as agreeing."""


@dataclass(frozen=True)
class SliceQuotes:
    """The fitted quotes of one expiry, sorted by log moneyness."""

    expiry_code: str
    T: float
    instrument_name: npt.NDArray[np.object_]
    k: FloatArray
    iv_bid: FloatArray
    iv_mid: FloatArray
    iv_ask: FloatArray
    vega: FloatArray

    @property
    def w_mid(self) -> FloatArray:
        return self.iv_mid**2 * self.T

    @property
    def w_bid(self) -> FloatArray:
        return self.iv_bid**2 * self.T

    @property
    def w_ask(self) -> FloatArray:
        return self.iv_ask**2 * self.T


def slice_quotes(clean: CleanQuotes, expiry_code: str) -> SliceQuotes:
    rows = clean.kept[clean.kept["expiry_code"] == expiry_code].sort_values("k")
    return SliceQuotes(
        expiry_code=expiry_code,
        T=float(rows["T"].iloc[0]),
        instrument_name=rows["instrument_name"].to_numpy(dtype=object),
        k=rows["k"].to_numpy(dtype=float),
        iv_bid=rows["iv_bid"].to_numpy(dtype=float),
        iv_mid=rows["iv_mid"].to_numpy(dtype=float),
        iv_ask=rows["iv_ask"].to_numpy(dtype=float),
        vega=rows["vega_mid"].to_numpy(dtype=float),
    )


def residual_scale(data: SliceQuotes, weighting: str) -> FloatArray:
    """Per quote multiplier applied to total variance residuals (see module docstring)."""
    if weighting == "spread":
        width = data.w_ask - data.w_bid
        return 1.0 / np.maximum(width, 1e-3 * float(np.median(width)))
    if weighting == "vega":
        scale = data.vega / (2.0 * data.iv_mid * data.T)
        return scale / float(np.mean(scale))
    if weighting == "uniform":
        return np.ones_like(data.k)
    raise ValueError(f"unknown weighting {weighting!r}")


def to_params(theta: npt.ArrayLike) -> SVIParams:
    """Optimizer variables (v, s_L, s_R, m, sigma) to raw SVI parameters."""
    v, s_left, s_right, m, sigma = (float(x) for x in np.asarray(theta, dtype=float))
    b = 0.5 * (s_left + s_right)
    rho = (s_right - s_left) / (s_right + s_left)
    a = v - sigma * np.sqrt(s_left * s_right)
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def from_params(params: SVIParams) -> FloatArray:
    s_left, s_right = params.wing_slopes
    return np.array([params.min_variance, s_left, s_right, params.m, params.sigma])


@dataclass(frozen=True)
class SVIFit:
    expiry_code: str
    T: float
    params: SVIParams
    weighting: str
    cost: float
    start_costs: tuple[float, ...]
    converged_starts: int
    rmse_vol: float
    """Root mean square of fitted minus mid IV, in vol points."""
    max_error_vol: float
    in_band_share: float
    """Share of quotes whose fitted IV lies inside the bid ask IV band."""
    active_bounds: tuple[str, ...]
    """Constraints binding at the solution, e.g. the Lee wing bound: where they bind, the
    shape there comes from the constraint rather than from the data."""
    residuals: pd.DataFrame = field(repr=False)

    @property
    def n_quotes(self) -> int:
        return len(self.residuals)


@dataclass(frozen=True)
class SVISurface:
    """Per expiry SVI fits in maturity order, and the expiries that could not be fitted."""

    fits: dict[str, SVIFit]
    skipped: dict[str, str]
    weighting: str

    def summary(self) -> pd.DataFrame:
        rows = []
        for fit in self.fits.values():
            p = fit.params
            left, right = p.wing_slopes
            rows.append(
                {
                    "expiry_code": fit.expiry_code,
                    "days": fit.T * 365.0,
                    "T": fit.T,
                    "n": fit.n_quotes,
                    "a": p.a,
                    "b": p.b,
                    "rho": p.rho,
                    "m": p.m,
                    "sigma": p.sigma,
                    "min_variance": p.min_variance,
                    "atm_vol": float(p.implied_vol(0.0, fit.T)),
                    "left_slope": left,
                    "right_slope": right,
                    "rmse_vol": fit.rmse_vol,
                    "max_error_vol": fit.max_error_vol,
                    "in_band_share": fit.in_band_share,
                    "cost": fit.cost,
                    "starts": len(fit.start_costs),
                    "converged_starts": fit.converged_starts,
                    "active_bounds": ", ".join(fit.active_bounds),
                }
            )
        return pd.DataFrame(rows)

    def residuals(self) -> pd.DataFrame:
        return pd.concat([fit.residuals for fit in self.fits.values()], ignore_index=True)


def fit_svi(clean: CleanQuotes, config: SVIConfig | None = None) -> SVISurface:
    """Fit every expiry of ``clean`` (which needs IV columns) independently."""
    config = config if config is not None else SVIConfig()
    kept = clean.kept
    order = kept.groupby("expiry_code")["T"].first().sort_values().index
    fits: dict[str, SVIFit] = {}
    skipped: dict[str, str] = {}
    for expiry_code in order:
        data = slice_quotes(clean, expiry_code)
        if data.k.size < config.min_quotes:
            skipped[expiry_code] = f"{data.k.size} quotes, fewer than min_quotes"
            logger.warning("SVI %s skipped: %s", expiry_code, skipped[expiry_code])
            continue
        fits[expiry_code] = fit_slice(data, config)
    return SVISurface(fits=fits, skipped=skipped, weighting=config.weighting)


def fit_slice(data: SliceQuotes, config: SVIConfig | None = None) -> SVIFit:
    config = config if config is not None else SVIConfig()
    scale = residual_scale(data, config.weighting)
    k, w = data.k, data.w_mid
    span = max(float(k.max() - k.min()), 0.05)
    lower = np.array([0.0, SLOPE_MIN, SLOPE_MIN, float(k.min()) - span, SIGMA_MIN])
    upper = np.array(
        [2.0 * float(w.max()), LEE_MAX_SLOPE, LEE_MAX_SLOPE, float(k.max()) + span, SIGMA_MAX]
    )

    def residuals(theta: FloatArray) -> FloatArray:
        return scale * (to_params(theta).total_variance(k) - w)

    starts = [*grid_starts(data, scale, config), heuristic_start(data)]
    results = []
    for start in starts:
        theta0 = np.clip(start, lower + 1e-12, upper - 1e-12)
        result = least_squares(
            residuals,
            theta0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        results.append(result)
    costs = np.array([2.0 * r.cost for r in results])  # sum of squared scaled residuals
    best = results[int(np.argmin(costs))]
    # The trust region reflective method approaches bounds only asymptotically; an active
    # set polish lands exactly on any bound that binds and reports which ones do.
    polished = least_squares(
        residuals,
        best.x,
        bounds=(lower, upper),
        method="dogbox",
        x_scale="jac",
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=5000,
    )
    final = polished if 2.0 * polished.cost <= costs.min() * (1.0 + 1e-9) else best
    cost = 2.0 * float(final.cost)
    agreeing = int(np.sum(costs <= cost * (1.0 + AGREEMENT_RTOL) + 1e-15))
    return _assess(
        data,
        to_params(final.x),
        config.weighting,
        cost,
        costs,
        agreeing,
        active_bounds(final.x, lower, upper),
    )


BOUND_NAMES = (
    ("minimum variance at 0", None),
    ("left wing flat", "left wing at Lee bound"),
    ("right wing flat", "right wing at Lee bound"),
    ("m at lower edge", "m at upper edge"),
    ("sigma at lower edge", "sigma at upper edge"),
)


def active_bounds(
    theta: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, tol: float = 1e-9
) -> tuple[str, ...]:
    """Names of the bounds the optimizer variables sit on, within ``tol`` of their range."""
    active = []
    for x, lo, hi, (low_name, high_name) in zip(theta, lower, upper, BOUND_NAMES, strict=True):
        margin = tol * max(hi - lo, 1.0)
        if low_name and x - lo <= margin:
            active.append(low_name)
        if high_name and hi - x <= margin:
            active.append(high_name)
    return tuple(active)


def grid_starts(data: SliceQuotes, scale: FloatArray, config: SVIConfig) -> list[FloatArray]:
    """The ``config.n_starts`` best points of the quasi explicit (m, sigma) grid."""
    k, w = data.k, data.w_mid
    candidates = []
    for m in np.linspace(k.min(), k.max(), config.grid_m):
        for sigma in np.geomspace(*GRID_SIGMA, config.grid_sigma):
            y = (k - m) / sigma
            design = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
            coef, *_ = np.linalg.lstsq(design * scale[:, None], w * scale, rcond=None)
            params = _project(*coef, m=m, sigma=sigma)
            cost = float(np.sum((scale * (params.total_variance(k) - w)) ** 2))
            candidates.append((cost, from_params(params)))
    candidates.sort(key=lambda item: item[0])
    return [theta for _, theta in candidates[: config.n_starts]]


def heuristic_start(data: SliceQuotes) -> FloatArray:
    """A start read off the data: vertex at the lowest variance, slopes from the wings."""
    k, w = data.k, data.w_mid
    i = int(np.argmin(w))
    m, w_min = float(k[i]), float(w[i])
    left = (w[0] - w_min) / (m - k[0]) if k[0] < m else 0.0
    right = (w[-1] - w_min) / (k[-1] - m) if k[-1] > m else 0.0
    b = max(0.5 * (left + right), 1e-4)
    rho = (right - left) / (right + left) if right + left > 0 else 0.0
    sigma = max(0.1 * float(k.max() - k.min()), 0.01)
    return from_params(_project(w_min - b * sigma, rho * b * sigma, b * sigma, m=m, sigma=sigma))


def _project(a: float, d: float, c: float, *, m: float, sigma: float) -> SVIParams:
    """Quasi explicit coefficients to the nearest admissible raw SVI parameters."""
    b = max(c / sigma, 0.0)
    rho = float(np.clip(d / c, -1.0, 1.0)) if c > 0.0 else 0.0
    s_left = float(np.clip(b * (1.0 - rho), SLOPE_MIN, 0.99 * LEE_MAX_SLOPE))
    s_right = float(np.clip(b * (1.0 + rho), SLOPE_MIN, 0.99 * LEE_MAX_SLOPE))
    v = max(a + sigma * np.sqrt(s_left * s_right), 0.0)
    return to_params([v, s_left, s_right, m, sigma])


def _assess(
    data: SliceQuotes,
    params: SVIParams,
    weighting: str,
    cost: float,
    costs: FloatArray,
    converged: int,
    active: tuple[str, ...],
) -> SVIFit:
    iv_fit = params.implied_vol(data.k, data.T)
    error = iv_fit - data.iv_mid
    in_band = (iv_fit >= data.iv_bid - 1e-12) & (iv_fit <= data.iv_ask + 1e-12)
    residuals = pd.DataFrame(
        {
            "expiry_code": data.expiry_code,
            "T": data.T,
            "instrument_name": data.instrument_name,
            "k": data.k,
            "iv_bid": data.iv_bid,
            "iv_mid": data.iv_mid,
            "iv_ask": data.iv_ask,
            "iv_svi": iv_fit,
            "error_vol": 100.0 * error,
            "in_band": in_band,
        }
    )
    return SVIFit(
        expiry_code=data.expiry_code,
        T=data.T,
        params=params,
        weighting=weighting,
        cost=cost,
        start_costs=tuple(float(c) for c in costs),
        converged_starts=converged,
        rmse_vol=float(100.0 * np.sqrt(np.mean(error**2))),
        max_error_vol=float(100.0 * np.max(np.abs(error))),
        in_band_share=float(np.mean(in_band)),
        active_bounds=active,
        residuals=residuals,
    )

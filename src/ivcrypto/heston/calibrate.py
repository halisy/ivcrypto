"""Heston calibration to the cleaned surface, and the least squares machinery Bates reuses.

Objective: vega weighted price errors, (model - mid) / vega, which equal implied
volatility errors to first order without a root search per option per iteration (with
``weighting = "spread"`` they are further divided by the quote's bid ask IV width, as for
SVI). Each expiry gets the same total weight, so the many strikes of the busiest expiries do
not dominate. Vegas are floored at 1% of the largest vega of their expiry so that a far wing
quote cannot take over the fit. After calibration the model prices are inverted exactly
(Brent, ``implied_vol.py``) and every quality figure is in true IV terms.

Bounds: v0 and theta in [1e-4, 4] (volatilities up to 200%), kappa in [1e-3, 50], xi in
[1e-2, 10], rho in [-0.99, 0.99]. The Feller condition 2 kappa theta >= xi^2 is reported,
not imposed. Starting points: a fixed grid around the data (short dated ATM variance for v0,
long dated for theta) crossed with a few values of kappa, xi and rho; the most promising
ones by initial cost are refined with trust region reflective least squares.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares

from ivcrypto.cleaning import CleanQuotes
from ivcrypto.config import HestonConfig
from ivcrypto.heston.charfunc import HestonParams
from ivcrypto.heston.pricer import FourierModel, price
from ivcrypto.implied_vol import implied_vols

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

LOWER = np.array([1e-4, 1e-3, 1e-4, 1e-2, -0.99])
UPPER = np.array([4.0, 50.0, 4.0, 10.0, 0.99])
PARAMETER_NAMES = ("v0", "kappa", "theta", "xi", "rho")
VEGA_FLOOR = 0.01


@dataclass(frozen=True)
class ExpiryQuotes:
    expiry_code: str
    T: float
    F: float
    instrument_name: npt.NDArray[np.object_]
    K: FloatArray
    k: FloatArray
    is_call: npt.NDArray[np.bool_]
    mid: FloatArray
    iv_bid: FloatArray
    iv_mid: FloatArray
    iv_ask: FloatArray
    vega: FloatArray


def calibration_quotes(clean: CleanQuotes, expiries: list[str] | None = None) -> list[ExpiryQuotes]:
    """The fitted (kept) quotes grouped by expiry, in maturity order."""
    kept = clean.kept.dropna(subset=["iv_mid", "vega_mid"])
    order = kept.groupby("expiry_code")["T"].first().sort_values().index
    groups = []
    for code in order:
        if expiries is not None and code not in expiries:
            continue
        rows = kept[kept["expiry_code"] == code].sort_values("k")
        groups.append(
            ExpiryQuotes(
                expiry_code=code,
                T=float(rows["T"].iloc[0]),
                F=float(rows["forward"].iloc[0]),
                instrument_name=rows["instrument_name"].to_numpy(dtype=object),
                K=rows["strike"].to_numpy(dtype=float),
                k=rows["k"].to_numpy(dtype=float),
                is_call=(rows["option_type"] == "call").to_numpy(),
                mid=rows["mid_usd"].to_numpy(dtype=float),
                iv_bid=rows["iv_bid"].to_numpy(dtype=float),
                iv_mid=rows["iv_mid"].to_numpy(dtype=float),
                iv_ask=rows["iv_ask"].to_numpy(dtype=float),
                vega=rows["vega_mid"].to_numpy(dtype=float),
            )
        )
    return groups


def _scales(quotes: list[ExpiryQuotes], config: HestonConfig) -> list[FloatArray]:
    """Per quote residual weights: 1 / vega (floored), optionally / spread and / sqrt(n)."""
    scales = []
    for q in quotes:
        vega = np.maximum(q.vega, VEGA_FLOOR * q.vega.max())
        scale = 1.0 / vega
        if config.weighting == "spread":
            scale = scale / (q.iv_ask - q.iv_bid)
        if config.equal_expiry_weights:
            scale = scale / np.sqrt(q.K.size)
        scales.append(scale)
    return scales


def model_prices(params: FourierModel, quotes: list[ExpiryQuotes]) -> list[FloatArray]:
    return [price(q.F, q.K, q.T, params, q.is_call) for q in quotes]


@dataclass(frozen=True)
class Calibration:
    """A calibrated model (Heston or Bates) and its exact IV residuals."""

    params: Any
    """``HestonParams`` or ``BatesParams``."""
    cost: float
    start_costs: tuple[float, ...]
    """Final cost of every refined start, a check that the optimum is not an accident."""
    start_params: tuple[tuple[float, ...], ...]
    """Final parameters of every refined start, in the order of ``start_costs``."""
    expiries: tuple[str, ...]
    residuals: pd.DataFrame = field(repr=False)
    column: str = "iv_heston"
    """Name of the model IV column in ``residuals``."""
    at_bounds: tuple[str, ...] = ()
    """Parameters of the best fit that sit on a bound, e.g. ``"xi at upper bound"``."""

    def summary(self) -> pd.DataFrame:
        return _quality(self.residuals, self.column)


def binding_bounds(
    x: npt.ArrayLike,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
    names: Sequence[str],
    rtol: float = 1e-5,
) -> tuple[str, ...]:
    """Names of the parameters within ``rtol`` of the bound width from a bound."""
    x, lower, upper = (np.asarray(a, dtype=float) for a in (x, lower, upper))
    tol = rtol * (upper - lower)
    found = []
    for name, value, lo, hi, eps in zip(names, x, lower, upper, tol, strict=True):
        if value - lo <= eps:
            found.append(f"{name} at lower bound")
        elif hi - value <= eps:
            found.append(f"{name} at upper bound")
    return tuple(found)


def fit_model(
    quotes: list[ExpiryQuotes],
    objective: HestonConfig,
    make_params: Callable[..., Any],
    starts: Sequence[FloatArray],
    bounds: tuple[FloatArray, FloatArray],
    *,
    n_starts: int,
    max_evaluations: int,
    always_refine: Sequence[int] = (),
    column: str = "iv_heston",
    names: Sequence[str] = PARAMETER_NAMES,
) -> Calibration:
    """Least squares calibration of any Fourier priced model to vega weighted price errors.

    ``objective`` sets the residual weights (``weighting`` and ``equal_expiry_weights``).
    The ``n_starts`` starting points with the lowest initial cost are refined, plus those
    listed in ``always_refine`` (indices into ``starts``).
    """
    if not quotes:
        raise ValueError("no quotes to calibrate to")
    lower, upper = bounds
    scales = _scales(quotes, objective)
    mids = [q.mid for q in quotes]

    def residuals(x: FloatArray) -> FloatArray:
        model = model_prices(make_params(*x), quotes)
        return np.concatenate(
            [s * (m - mid) for s, m, mid in zip(scales, model, mids, strict=True)]
        )

    initial = [float(np.sum(residuals(x) ** 2)) for x in starts]
    order = list(np.argsort(initial)[:n_starts])
    order += [i for i in always_refine if i not in order]
    results = [
        least_squares(
            residuals,
            starts[index],
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max_evaluations,
        )
        for index in order
    ]
    costs = np.array([2.0 * r.cost for r in results])
    best = results[int(np.argmin(costs))]
    params = make_params(*best.x)
    logger.info(
        "%s on %d expiries: %s, cost %.4g (starts: %s)",
        type(params).__name__.removesuffix("Params"),
        len(quotes),
        ", ".join(f"{n}={v:.4g}" for n, v in zip(names, best.x, strict=True)),
        costs.min(),
        np.array2string(np.sort(costs), precision=4),
    )
    return Calibration(
        params=params,
        cost=float(costs.min()),
        start_costs=tuple(float(c) for c in costs),
        start_params=tuple(tuple(float(v) for v in r.x) for r in results),
        expiries=tuple(q.expiry_code for q in quotes),
        residuals=_residual_table(params, quotes, column),
        column=column,
        at_bounds=binding_bounds(best.x, lower, upper, names),
    )


def calibrate_heston(quotes: list[ExpiryQuotes], config: HestonConfig | None = None) -> Calibration:
    """One Heston parameter set for all the given expiries."""
    config = config if config is not None else HestonConfig()
    return fit_model(
        quotes,
        config,
        HestonParams,
        starting_points(quotes),
        (LOWER, UPPER),
        n_starts=config.n_starts,
        max_evaluations=config.max_evaluations,
    )


def calibrate_per_expiry(
    quotes: list[ExpiryQuotes], config: HestonConfig | None = None
) -> dict[str, Calibration]:
    """A separate Heston fit to each expiry: a diagnostic of what one smile alone asks for."""
    return {q.expiry_code: calibrate_heston([q], config) for q in quotes}


def starting_points(quotes: list[ExpiryQuotes]) -> list[FloatArray]:
    """A fixed grid around the data: short dated ATM variance for v0, long dated for theta."""
    if not quotes:
        raise ValueError("no quotes to calibrate to")
    atm = [float(np.interp(0.0, q.k, q.iv_mid)) ** 2 for q in quotes]
    v0, theta = atm[0], atm[-1]
    starts = []
    for kappa, xi, rho in product((1.0, 5.0), (0.5, 2.0), (-0.5, 0.0)):
        x = np.array([v0, kappa, theta, xi, rho])
        starts.append(np.clip(x, LOWER + 1e-9, UPPER - 1e-9))
    return starts


def _residual_table(
    params: FourierModel, quotes: list[ExpiryQuotes], column: str = "iv_heston"
) -> pd.DataFrame:
    parts = []
    for q, model in zip(quotes, model_prices(params, quotes), strict=True):
        iv = implied_vols(model, q.F, q.K, q.T, q.is_call)
        parts.append(
            pd.DataFrame(
                {
                    "expiry_code": q.expiry_code,
                    "T": q.T,
                    "instrument_name": q.instrument_name,
                    "k": q.k,
                    "iv_bid": q.iv_bid,
                    "iv_mid": q.iv_mid,
                    "iv_ask": q.iv_ask,
                    column: iv,
                    "error_vol": 100.0 * (iv - q.iv_mid),
                    "in_band": (iv >= q.iv_bid - 1e-12) & (iv <= q.iv_ask + 1e-12),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _quality(residuals: pd.DataFrame, column: str = "iv_heston") -> pd.DataFrame:
    rows = []
    for code, part in residuals.groupby("expiry_code", sort=False):
        error = part["error_vol"]
        rows.append(
            {
                "expiry_code": code,
                "days": float(part["T"].iloc[0]) * 365.0,
                "n": len(part),
                "rmse_vol": float(np.sqrt(np.mean(error**2))),
                "max_error_vol": float(np.max(np.abs(error))),
                "in_band_share": float(part["in_band"].mean()),
                "no_iv": int(part[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)

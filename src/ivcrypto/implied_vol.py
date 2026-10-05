"""Implied volatility by Brent's method, NaN outside the no arbitrage bounds.

Under undiscounted Black 76 a call and a put on the same strike differ by exactly F - K,
so any price maps to the time value of the out of the money option at that strike,
``price - intrinsic``. The solver inverts that OTM time value: inverting an ITM price
directly is badly conditioned, because the time value is a small difference of large
numbers. The root is searched in s = sigma sqrt(T), the dimensionless total standard
deviation, with sigma bracketed in ``[lower, upper]``.

A price gets NaN instead of a forced solution when it violates the no arbitrage bounds
(at or below intrinsic value, or at or above F for a call and K for a put), or when its
implied volatility lies outside the bracket.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import brentq

from ivcrypto import black76
from ivcrypto.cleaning import CleanQuotes, append_filter

FloatArray = npt.NDArray[np.float64]

DEFAULT_LOWER = 1e-4
"""Lowest volatility searched (0.01%)."""
DEFAULT_UPPER = 10.0
"""Highest volatility searched (1000%)."""

_SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT2)


def _otm_time_value(k: float, s: float, otm_is_call: bool) -> float:
    """Normalized (divided by F) price of the OTM option at log moneyness k."""
    d1 = -k / s + 0.5 * s
    d2 = d1 - s
    if otm_is_call:
        return _ncdf(d1) - math.exp(k) * _ncdf(d2)
    return math.exp(k) * _ncdf(-d2) - _ncdf(-d1)


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    is_call: bool,
    *,
    discount: float = 1.0,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> float:
    """Black 76 implied volatility of one option price, or NaN (see module docstring)."""
    values = (price, F, K, T, discount)
    if not all(math.isfinite(v) for v in values) or min(F, K, T, discount) <= 0:
        return math.nan
    forward_price = price / discount
    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    bound = F if is_call else K
    if not intrinsic < forward_price < bound:
        return math.nan
    target = (forward_price - intrinsic) / F
    k = math.log(K / F)
    otm_is_call = K >= F
    sqrt_t = math.sqrt(T)

    def objective(s: float) -> float:
        return _otm_time_value(k, s, otm_is_call) - target

    s_low, s_high = lower * sqrt_t, upper * sqrt_t
    f_low, f_high = objective(s_low), objective(s_high)
    if f_low > 0.0 or f_high < 0.0:
        return math.nan
    if f_low == 0.0:
        return lower
    if f_high == 0.0:
        return upper
    s = brentq(objective, s_low, s_high, xtol=1e-15, rtol=1e-13, maxiter=500)
    return s / sqrt_t


def implied_vols(
    price: npt.ArrayLike,
    F: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    is_call: npt.ArrayLike,
    *,
    discount: npt.ArrayLike = 1.0,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
) -> FloatArray:
    """Element wise :func:`implied_vol` with numpy broadcasting."""
    arrays = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(F, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(T, dtype=float),
        np.asarray(is_call, dtype=bool),
        np.asarray(discount, dtype=float),
    )
    flat = [a.ravel() for a in arrays]
    out = np.fromiter(
        (
            implied_vol(p, f, k, t, bool(c), discount=d, lower=lower, upper=upper)
            for p, f, k, t, c, d in zip(*flat, strict=True)
        ),
        dtype=float,
        count=flat[0].size,
    )
    return out.reshape(arrays[0].shape)


IV_COLUMNS = ["iv_bid", "iv_mid", "iv_ask", "iv_mark", "iv_spread", "vega_mid"]
"""Columns added by :func:`add_implied_vols` (vols as decimals, vega in USD per unit vol)."""


def add_implied_vols(clean: CleanQuotes) -> CleanQuotes:
    """Bid, mid, ask and mark IVs for every quote, with our forward and time to expiry.

    Quotes still kept whose bid, mid or ask has no implied volatility are removed by the
    filter ``no_implied_vol``, recorded in the filter log like the cleaning filters.
    """
    quotes = clean.quotes.copy()
    is_call = (quotes["option_type"] == "call").to_numpy()
    F = quotes["forward"].to_numpy()
    K = quotes["strike"].to_numpy()
    T = quotes["T"].to_numpy()
    for side in ("bid", "mid", "ask", "mark"):
        quotes[f"iv_{side}"] = implied_vols(quotes[f"{side}_usd"].to_numpy(), F, K, T, is_call)
    quotes["iv_spread"] = quotes["iv_ask"] - quotes["iv_bid"]
    quotes["vega_mid"] = black76.vega(F, K, T, quotes["iv_mid"].to_numpy())
    with_ivs = CleanQuotes(
        currency=clean.currency,
        valuation_time=clean.valuation_time,
        quotes=quotes,
        forwards=clean.forwards,
        filters=clean.filters,
        notes=clean.notes,
    )
    missing = pd.Series(
        quotes[["iv_bid", "iv_mid", "iv_ask"]].isna().any(axis=1), index=quotes.index
    )
    return append_filter(
        with_ivs,
        "no_implied_vol",
        "bid, mid or ask outside the no arbitrage bounds (no implied volatility)",
        missing,
    )

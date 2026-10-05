"""Black 76: European options on a forward.

Prices are undiscounted (forward premiums) unless a discount factor is passed. That is the
form the pipeline uses: ``cleaning.py`` produces forward USD premiums, and Deribit prices
its options with zero rates on the expiry's forward (docs/deribit_api.md).

Every function broadcasts over numpy arrays. ``is_call`` is a boolean or boolean array.
Writing k = ln(K/F) and s = sigma * sqrt(T) (total standard deviation), the undiscounted
prices divided by F depend on (k, s) only:

    call / F = N(d1) - e^k N(d2),    put / F = e^k N(-d2) - N(-d1),
    d1 = -k/s + s/2,                 d2 = d1 - s.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def normalized_price(k: npt.ArrayLike, s: npt.ArrayLike, is_call: npt.ArrayLike) -> FloatArray:
    """Undiscounted price divided by F as a function of k = ln(K/F) and s = sigma sqrt(T).

    At s = 0 the price is the intrinsic value.
    """
    k, s, is_call = np.broadcast_arrays(
        np.asarray(k, dtype=float), np.asarray(s, dtype=float), np.asarray(is_call, dtype=bool)
    )
    strike_over_forward = np.exp(k)
    positive = s > 0
    safe_s = np.where(positive, s, 1.0)
    d1 = -k / safe_s + 0.5 * safe_s
    d2 = d1 - safe_s
    call = ndtr(d1) - strike_over_forward * ndtr(d2)
    put = strike_over_forward * ndtr(-d2) - ndtr(-d1)
    value = np.where(is_call, call, put)
    intrinsic = np.where(
        is_call,
        np.maximum(1.0 - strike_over_forward, 0.0),
        np.maximum(strike_over_forward - 1.0, 0.0),
    )
    return np.where(positive, value, intrinsic)


def price(
    F: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    sigma: npt.ArrayLike,
    is_call: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
) -> FloatArray:
    """Black 76 price: ``discount * F * normalized_price(ln(K/F), sigma sqrt(T))``."""
    F = np.asarray(F, dtype=float)
    k = np.log(np.asarray(K, dtype=float) / F)
    s = np.asarray(sigma, dtype=float) * np.sqrt(np.asarray(T, dtype=float))
    return np.asarray(discount, dtype=float) * F * normalized_price(k, s, is_call)


def vega(
    F: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    sigma: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
) -> FloatArray:
    """Derivative of the price with respect to sigma (per unit of volatility, not per point)."""
    F = np.asarray(F, dtype=float)
    sqrt_t = np.sqrt(np.asarray(T, dtype=float))
    s = np.asarray(sigma, dtype=float) * sqrt_t
    k = np.log(np.asarray(K, dtype=float) / F)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / s + 0.5 * s
    density = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    return np.asarray(discount, dtype=float) * F * density * sqrt_t


def intrinsic_value(F: npt.ArrayLike, K: npt.ArrayLike, is_call: npt.ArrayLike) -> FloatArray:
    """Undiscounted intrinsic value, the lower no arbitrage bound."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    return np.where(np.asarray(is_call, dtype=bool), np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))


def upper_bound(F: npt.ArrayLike, K: npt.ArrayLike, is_call: npt.ArrayLike) -> FloatArray:
    """Undiscounted upper no arbitrage bound: F for a call, K for a put."""
    return np.where(
        np.asarray(is_call, dtype=bool), np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    )

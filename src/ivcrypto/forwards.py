"""Forward price per expiry, from put call parity on the coin denominated quotes.

Deribit's options are inverse: a call pays (S_T - K)^+ / S_T coins. With a zero coin
interest rate (Deribit's convention, confirmed in docs/deribit_api.md), the coin prices of
a call and a put with the same strike K satisfy

    C - P = 1 - K / F,

so every strike quoted on both sides implies a forward F = K / (1 - (C - P)) without a
model, a spot price or a USD interest rate. A forward that is consistent with the option
quotes themselves matters: it puts k = ln(K/F) = 0 where calls and puts agree, so the OTM
put and call wings meet without a jump.

Each pair's precision follows from its quotes: an error e in C - P moves the forward by
F^2 / K * e, and e is bounded by the sum of the two half spreads. The estimate is the
precision weighted median of the most precise pairs, which a single stale quote cannot
move. An expiry falls back to Deribit's ``underlying_price`` (the expiry's future) when
too few pairs exist or they disagree.

A weighted regression of C - P on K over all pairs, C - P = D (1 - K/F), is reported as a
diagnostic: D is the coin discount factor implied by the quotes, a direct test of the
zero rate assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ivcrypto.config import CleaningConfig

FloatArray = npt.NDArray[np.float64]


def weighted_median(values: npt.ArrayLike, weights: npt.ArrayLike) -> float:
    """Smallest value at which the cumulative weight reaches half the total."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0 or v.shape != w.shape or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weighted_median needs matching non empty values and positive weights")
    order = np.argsort(v)
    cumulative = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def parity_pairs(quotes: pd.DataFrame) -> pd.DataFrame:
    """Strikes of one expiry where the call and the put both have a valid bid and ask.

    ``quotes`` needs ``strike``, ``option_type``, ``bid_btc`` and ``ask_btc`` (coin prices,
    NaN when missing). Returns one row per strike with the implied forward and its error.
    """
    valid = quotes[
        quotes["bid_btc"].notna()
        & quotes["ask_btc"].notna()
        & (quotes["bid_btc"] < quotes["ask_btc"])
    ]
    sides = {}
    for option_type in ("call", "put"):
        side = valid.loc[valid["option_type"] == option_type, ["strike", "bid_btc", "ask_btc"]]
        if side["strike"].duplicated().any():
            raise ValueError(f"duplicate {option_type} strikes within one expiry")
        sides[option_type] = side.set_index("strike")
    pairs = sides["call"].join(sides["put"], how="inner", lsuffix="_call", rsuffix="_put")
    strike = pairs.index.to_numpy(dtype=float)
    mid_call = 0.5 * (pairs["bid_btc_call"] + pairs["ask_btc_call"]).to_numpy()
    mid_put = 0.5 * (pairs["bid_btc_put"] + pairs["ask_btc_put"]).to_numpy()
    diff = mid_call - mid_put
    diff_err = (
        0.5
        * (
            (pairs["ask_btc_call"] - pairs["bid_btc_call"])
            + (pairs["ask_btc_put"] - pairs["bid_btc_put"])
        ).to_numpy()
    )
    usable = diff < 1.0  # always true for real quotes; guards the division below
    forward = strike[usable] / (1.0 - diff[usable])
    return pd.DataFrame(
        {
            "strike": strike[usable],
            "diff": diff[usable],
            "diff_err": diff_err[usable],
            "forward": forward,
            "forward_err": forward**2 / strike[usable] * diff_err[usable],
        }
    )


@dataclass(frozen=True)
class ParityEstimate:
    forward: float
    pairs_available: int
    pairs_used: int
    dispersion_bps: float
    regression_forward: float
    regression_discount: float


def parity_forward(pairs: pd.DataFrame, n_pairs: int) -> ParityEstimate:
    """Precision weighted median forward of the ``n_pairs`` most precise pairs."""
    nan = float("nan")
    if pairs.empty:
        return ParityEstimate(nan, 0, 0, nan, nan, nan)
    best = pairs.nsmallest(n_pairs, "forward_err")
    weights = 1.0 / np.maximum(best["forward_err"].to_numpy(), 1e-12) ** 2
    forward = weighted_median(best["forward"], weights)
    # Unweighted on purpose: one dominant pair must not make the pairs look unanimous.
    deviation = np.abs(best["forward"].to_numpy() - forward)
    dispersion_bps = 1e4 * float(np.median(deviation)) / forward
    regression_forward, discount = _parity_regression(pairs)
    return ParityEstimate(
        forward=forward,
        pairs_available=len(pairs),
        pairs_used=len(best),
        dispersion_bps=dispersion_bps,
        regression_forward=regression_forward,
        regression_discount=discount,
    )


def _parity_regression(pairs: pd.DataFrame) -> tuple[float, float]:
    """Weighted least squares of C - P = D - (D / F) K; returns (F, D)."""
    nan = float("nan")
    if pairs["strike"].nunique() < 3:
        return nan, nan
    sqrt_w = 1.0 / np.maximum(pairs["diff_err"].to_numpy(), 1e-12)
    design = np.column_stack([np.ones(len(pairs)), pairs["strike"].to_numpy()])
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], pairs["diff"].to_numpy() * sqrt_w)
    intercept, slope = float(coef[0]), float(coef[1])
    if slope >= 0:
        return nan, nan
    return -intercept / slope, intercept


FORWARD_COLUMNS = [
    "expiry_code",
    "expiry",
    "T",
    "days",
    "underlying_index",
    "forward",
    "forward_source",
    "fallback_reason",
    "parity_forward",
    "parity_pairs_available",
    "parity_pairs_used",
    "parity_dispersion_bps",
    "parity_vs_underlying_bps",
    "regression_forward",
    "regression_discount",
    "underlying_price",
    "future_bid",
    "future_ask",
    "future_mark",
]


def estimate_forwards(
    quotes: pd.DataFrame, futures: pd.DataFrame | None, config: CleaningConfig
) -> pd.DataFrame:
    """One row per expiry: the forward used downstream, its source and diagnostics.

    ``quotes`` needs ``expiry_code``, ``expiry``, ``T``, ``days``, ``underlying_index``,
    ``underlying_price``, ``strike``, ``option_type``, ``bid_btc`` and ``ask_btc``.
    ``futures`` is the raw ``future_book`` table (optional, for comparison only).
    """
    future_quotes = (
        futures.set_index("instrument_name")[["bid_price", "ask_price", "mark_price"]]
        if futures is not None
        else pd.DataFrame(columns=["bid_price", "ask_price", "mark_price"])
    )
    rows = []
    for expiry_code, group in quotes.groupby("expiry_code", sort=False):
        underlying = float(group["underlying_price"].median())
        estimate = parity_forward(parity_pairs(group), config.parity_pairs)
        reason = None
        if config.forward_method == "parity":
            if estimate.pairs_used < config.parity_min_pairs:
                reason = f"only {estimate.pairs_used} usable call/put pairs"
            elif estimate.dispersion_bps > config.parity_max_dispersion_bps:
                reason = f"pairs disagree by {estimate.dispersion_bps:.1f} bps"
        use_parity = config.forward_method == "parity" and reason is None
        underlying_index = group["underlying_index"].iloc[0]
        future = (
            future_quotes.loc[underlying_index] if underlying_index in future_quotes.index else None
        )
        rows.append(
            {
                "expiry_code": expiry_code,
                "expiry": group["expiry"].iloc[0],
                "T": float(group["T"].iloc[0]),
                "days": float(group["days"].iloc[0]),
                "underlying_index": underlying_index,
                "forward": estimate.forward if use_parity else underlying,
                "forward_source": "parity" if use_parity else "underlying",
                "fallback_reason": reason,
                "parity_forward": estimate.forward,
                "parity_pairs_available": estimate.pairs_available,
                "parity_pairs_used": estimate.pairs_used,
                "parity_dispersion_bps": estimate.dispersion_bps,
                "parity_vs_underlying_bps": 1e4 * (estimate.forward / underlying - 1.0),
                "regression_forward": estimate.regression_forward,
                "regression_discount": estimate.regression_discount,
                "underlying_price": underlying,
                "future_bid": float(future["bid_price"]) if future is not None else np.nan,
                "future_ask": float(future["ask_price"]) if future is not None else np.nan,
                "future_mark": float(future["mark_price"]) if future is not None else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=FORWARD_COLUMNS).sort_values("T", ignore_index=True)

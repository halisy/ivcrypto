"""Forward price per expiry, from put call parity on the coin denominated quotes.

Deribit's options are inverse: a call pays (S_T - K)^+ / S_T coins. With a zero coin
interest rate (Deribit's convention, confirmed in docs/deribit_api.md), the coin prices of
a call and a put with the same strike K satisfy

    C - P = 1 - K / F,

so every strike quoted on both sides implies a forward F = K / (1 - (C - P)) without a
model, a spot price or a USD interest rate. A forward consistent with the option quotes
themselves puts k = ln(K/F) = 0 where calls and puts agree, so the OTM put and call wings
meet without a jump.

Estimator, per expiry:

1. Take the ``n`` strikes nearest the money (nearest to Deribit's forward) quoted two sided
   on both the call and the put. Choosing by distance rather than by precision keeps the
   set balanced on both sides of the money.
2. Weight each pair by its precision from the quotes: an error e in C - P moves the forward
   by F^2 / K * e, and e is at most the sum of the two half spreads.
3. Drop pairs further than five robust standard deviations from the weighted median (a
   stale quote), then take the precision weighted mean.
4. Report the scatter of the pairs around it and the standard error of the mean. On real
   data the mids scatter far less than the half spreads suggest (about 1 to 3 bps against
   10 to 30), so the standard error comes from the observed scatter.

An expiry falls back to Deribit's ``underlying_price`` (the expiry's future) when too few
pairs exist or they scatter too much. A weighted regression of C - P on K over all pairs,
C - P = D (1 - K/F), is reported as a diagnostic: D is the coin discount factor implied by
the quotes, a direct test of the zero rate assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ivcrypto.config import CleaningConfig

FloatArray = npt.NDArray[np.float64]

TRIM_ROBUST_SIGMAS = 5.0
"""Pairs further than this many robust standard deviations from the median are dropped."""
_MAD_TO_SIGMA = 1.4826


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
    pairs_trimmed: int
    scatter_bps: float
    se_bps: float
    regression_forward: float
    regression_discount: float


def parity_forward(pairs: pd.DataFrame, n_pairs: int, reference: float) -> ParityEstimate:
    """Precision weighted mean forward of the ``n_pairs`` pairs nearest ``reference``."""
    nan = float("nan")
    if pairs.empty:
        return ParityEstimate(nan, 0, 0, 0, nan, nan, nan, nan)
    distance = np.abs(np.log(pairs["strike"].to_numpy() / reference))
    near = pairs.iloc[np.argsort(distance, kind="stable")[:n_pairs]]
    forwards = near["forward"].to_numpy()
    weights = 1.0 / np.maximum(near["forward_err"].to_numpy(), 1e-12) ** 2

    median = weighted_median(forwards, weights)
    robust_sigma = _MAD_TO_SIGMA * float(np.median(np.abs(forwards - median)))
    tolerance = max(TRIM_ROBUST_SIGMAS * robust_sigma, 1e-4 * median)  # never below 1 bp
    keep = np.abs(forwards - median) <= tolerance
    forwards, weights = forwards[keep], weights[keep]

    forward = float(np.sum(weights * forwards) / np.sum(weights))
    n = forwards.size
    if n > 1:
        variance = float(np.sum(weights * (forwards - forward) ** 2) / np.sum(weights))
        variance *= n / (n - 1)
        effective_n = float(np.sum(weights) ** 2 / np.sum(weights**2))
        scatter_bps = 1e4 * np.sqrt(variance) / forward
        se_bps = scatter_bps / np.sqrt(effective_n)
    else:
        scatter_bps = se_bps = nan
    regression_forward, discount = _parity_regression(pairs)
    return ParityEstimate(
        forward=forward,
        pairs_available=len(pairs),
        pairs_used=n,
        pairs_trimmed=int((~keep).sum()),
        scatter_bps=scatter_bps,
        se_bps=se_bps,
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
    "parity_pairs_trimmed",
    "parity_scatter_bps",
    "parity_se_bps",
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
        estimate = parity_forward(parity_pairs(group), config.parity_pairs, underlying)
        reason = None
        if config.forward_method == "parity":
            if estimate.pairs_used < config.parity_min_pairs:
                reason = f"only {estimate.pairs_used} usable call/put pairs"
            elif estimate.scatter_bps > config.parity_max_scatter_bps:
                reason = f"pairs scatter by {estimate.scatter_bps:.1f} bps"
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
                "parity_pairs_trimmed": estimate.pairs_trimmed,
                "parity_scatter_bps": estimate.scatter_bps,
                "parity_se_bps": estimate.se_bps,
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

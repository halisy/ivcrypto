"""Static arbitrage checks on the fitted smiles and on the raw quotes.

Model checks, on each fitted slice over a dense grid of log moneyness k:

* **Butterfly**: Durrleman's condition g(k) >= 0, with w = w(k) the total variance and

      g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1 / w + 1 / 4) + w'' / 2.

  The density implied by a smile is a positive multiple of g, so g < 0 means a butterfly
  spread with a negative price. Parameter constraints alone do not prevent it: Vogt's
  example in Gatheral and Jacquier (2014) satisfies them all.
* **Calendar**: at fixed k = ln(K / F_T), total variance must not decrease with maturity
  (Gatheral and Jacquier 2014, Lemma 2.1). Checking consecutive expiries suffices.

Every model violation is labelled as inside the quoted range (where quotes pin the fit
down) or in the extrapolated wings.

Market checks, on the raw coin quotes, with no forward and no model:

* Call prices must not increase with strike and put prices must not decrease, and both must
  be convex in strike. A Deribit call pays (S - K)^+ / S coins, decreasing and convex in K
  state by state, so coin prices obey the same shape conditions as USD prices.
* Each condition is tested on mid prices (is the data internally consistent?) and on
  executable prices (could the arbitrage be traded, buying at the ask and selling at the
  bid?). Executable tests use every pair or triple of strikes, not only neighbours.
* Calendar in the data: at the log moneyness of each fitted quote, the next expiry's mid
  total variance (interpolated linearly in k between its quotes) must not be lower. This
  needs our IVs and forwards, so it is labelled as interpolated.

Separating the two tells arbitrage in the data apart from arbitrage introduced by a fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pandas as pd

from ivcrypto.cleaning import CleanQuotes
from ivcrypto.svi.fit import SVISurface
from ivcrypto.svi.raw import SVIParams
from ivcrypto.svi.ssvi import SSVIFit

FloatArray = npt.NDArray[np.float64]

K_GRID = np.linspace(-1.5, 1.5, 3001)
"""Dense log moneyness grid for the model checks (strikes from 22% to 448% of F)."""
TOLERANCE = 1e-10
"""Numerical slack: a shortfall must exceed this to count as a violation."""

VIOLATION_COLUMNS = [
    "check",
    "prices",
    "expiry",
    "location",
    "k_low",
    "k_high",
    "magnitude",
    "in_quoted_range",
]
"""``prices`` is ``model`` (fitted smile), ``mid``, ``executable`` (bid and ask) or
``mid, interpolated``. ``magnitude`` is the size of the shortfall: g for butterflies,
total variance for calendars, coins for market strike tests."""


@dataclass(frozen=True)
class Slice:
    """A fitted smile to check, with the log moneyness range its quotes cover."""

    expiry_code: str
    T: float
    params: SVIParams
    k_min: float
    k_max: float


def slices_from_svi(surface: SVISurface) -> list[Slice]:
    return [
        Slice(
            expiry_code=fit.expiry_code,
            T=fit.T,
            params=fit.params,
            k_min=float(fit.residuals["k"].min()),
            k_max=float(fit.residuals["k"].max()),
        )
        for fit in surface.fits.values()
    ]


def slices_from_ssvi(fit: SSVIFit, extra_maturities: npt.ArrayLike = ()) -> list[Slice]:
    """SSVI slices at the fitted expiries, plus any maturities in between (no quotes there,
    so their violations would count as outside the quoted range)."""
    ranges = fit.residuals.groupby("expiry_code")["k"].agg(["min", "max"])
    slices = [
        Slice(code, float(T), fit.slice_params(float(T)), *ranges.loc[code].to_numpy(dtype=float))
        for code, T in zip(fit.expiries, fit.T, strict=True)
    ]
    for T in np.asarray(extra_maturities, dtype=float):
        name = f"{T * 365.0:.2f} days"
        slices.append(Slice(name, float(T), fit.slice_params(float(T)), np.inf, -np.inf))
    return sorted(slices, key=lambda s: s.T)


def durrleman_g(params: SVIParams, k: npt.ArrayLike) -> FloatArray:
    w, dw, d2w = params.derivatives(k)
    k = np.asarray(k, dtype=float)
    return (1.0 - k * dw / (2.0 * w)) ** 2 - 0.25 * dw * dw * (1.0 / w + 0.25) + 0.5 * d2w


@dataclass(frozen=True)
class CheckResult:
    violations: pd.DataFrame
    tests: pd.DataFrame
    """One row per check and price type: how many tests ran and how many failed."""


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Index ranges [start, end] of the runs of True in ``mask``."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=VIOLATION_COLUMNS)


def _tests(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["check", "prices", "tested", "violations"])


def check_butterfly(slices: Sequence[Slice], k_grid: FloatArray = K_GRID) -> CheckResult:
    rows = []
    for s in slices:
        g = durrleman_g(s.params, k_grid)
        for start, end in _runs(g < -TOLERANCE):
            lo, hi = float(k_grid[start]), float(k_grid[end])
            worst = start + int(np.argmin(g[start : end + 1]))
            rows.append(
                {
                    "check": "butterfly",
                    "prices": "model",
                    "expiry": s.expiry_code,
                    "location": f"k in [{lo:.3f}, {hi:.3f}]; worst g = {g[worst]:.2e} "
                    f"at k = {k_grid[worst]:.3f}",
                    "k_low": lo,
                    "k_high": hi,
                    "magnitude": float(-g[worst]),
                    "in_quoted_range": bool(hi >= s.k_min and lo <= s.k_max),
                }
            )
    return CheckResult(_frame(rows), _tests([("butterfly", "model", len(slices), len(rows))]))


def check_calendar(slices: Sequence[Slice], k_grid: FloatArray = K_GRID) -> CheckResult:
    ordered = sorted(slices, key=lambda s: s.T)
    rows = []
    for short, long in pairwise(ordered):
        gap = long.params.total_variance(k_grid) - short.params.total_variance(k_grid)
        lo_quoted, hi_quoted = max(short.k_min, long.k_min), min(short.k_max, long.k_max)
        for start, end in _runs(gap < -TOLERANCE):
            lo, hi = float(k_grid[start]), float(k_grid[end])
            worst = start + int(np.argmin(gap[start : end + 1]))
            rows.append(
                {
                    "check": "calendar",
                    "prices": "model",
                    "expiry": f"{short.expiry_code}/{long.expiry_code}",
                    "location": f"k in [{lo:.3f}, {hi:.3f}]; total variance falls by "
                    f"{-gap[worst]:.2e} at k = {k_grid[worst]:.3f}",
                    "k_low": lo,
                    "k_high": hi,
                    "magnitude": float(-gap[worst]),
                    "in_quoted_range": bool(hi >= lo_quoted and lo <= hi_quoted),
                }
            )
    pairs = max(len(ordered) - 1, 0)
    return CheckResult(_frame(rows), _tests([("calendar", "model", pairs, len(rows))]))


def check_strike_shape(
    strikes: npt.ArrayLike,
    bid: npt.ArrayLike,
    ask: npt.ArrayLike,
    option_type: str,
    expiry: str,
) -> CheckResult:
    """Monotonicity and convexity in strike for one expiry and option type (coin prices)."""
    order = np.argsort(np.asarray(strikes, dtype=float))
    K = np.asarray(strikes, dtype=float)[order]
    bid = np.asarray(bid, dtype=float)[order]
    ask = np.asarray(ask, dtype=float)[order]
    mid = 0.5 * (bid + ask)
    n = K.size
    rows: list[dict[str, object]] = []

    def add(check: str, prices: str, idx: Sequence[int], shortfall: float) -> None:
        rows.append(
            {
                "check": f"{option_type} {check}",
                "prices": prices,
                "expiry": expiry,
                "location": "strikes " + ", ".join(f"{K[i]:g}" for i in idx),
                "k_low": float(K[idx[0]]),
                "k_high": float(K[idx[-1]]),
                "magnitude": float(shortfall),
                "in_quoted_range": True,
            }
        )

    # Calls must fall with strike, puts must rise. For any i < j, the executable
    # arbitrage buys the dearer side at its ask and sells the other at its bid.
    falling = option_type == "call"
    for i in range(n - 1):
        excess = mid[i + 1] - mid[i] if falling else mid[i] - mid[i + 1]
        if excess > TOLERANCE:
            add("monotonicity", "mid", (i, i + 1), excess)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        profit = bid[j] - ask[i] if falling else bid[i] - ask[j]
        if profit > TOLERANCE:
            add("monotonicity", "executable", (i, j), profit)

    # Convexity: a butterfly (long the wings, short the body) must cost nothing or more.
    for i in range(n - 2):
        lam = (K[i + 2] - K[i + 1]) / (K[i + 2] - K[i])
        value = lam * mid[i] + (1 - lam) * mid[i + 2] - mid[i + 1]
        if value < -TOLERANCE:
            add("convexity", "mid", (i, i + 1, i + 2), -value)
    triples = [(a, b, c) for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]
    if triples:
        lo, body, hi = np.array(triples).T
        lam = (K[hi] - K[body]) / (K[hi] - K[lo])
        profit = bid[body] - (lam * ask[lo] + (1 - lam) * ask[hi])
        for idx in np.flatnonzero(profit > TOLERANCE):
            add("convexity", "executable", triples[idx], profit[idx])

    violations = _frame(rows)
    counts = violations.groupby(["check", "prices"]).size() if rows else pd.Series(dtype=int)
    tested = {
        ("monotonicity", "mid"): max(n - 1, 0),
        ("monotonicity", "executable"): len(pairs),
        ("convexity", "mid"): max(n - 2, 0),
        ("convexity", "executable"): len(triples),
    }
    tests = _tests(
        [
            (
                f"{option_type} {check}",
                prices,
                count,
                int(counts.get((f"{option_type} {check}", prices), 0)),
            )
            for (check, prices), count in tested.items()
        ]
    )
    return CheckResult(violations, tests)


def check_market_quotes(clean: CleanQuotes) -> CheckResult:
    """Strike shape tests on every two sided quote of every expiry, calls and puts apart."""
    q = clean.quotes
    two_sided = q[q["bid_btc"].notna() & q["ask_btc"].notna() & (q["bid_btc"] < q["ask_btc"])]
    results = [
        check_strike_shape(group["strike"], group["bid_btc"], group["ask_btc"], option_type, expiry)
        for (expiry, option_type), group in two_sided.groupby(
            ["expiry_code", "option_type"], sort=False
        )
    ]
    return _combine(results)


def check_market_calendar(clean: CleanQuotes) -> CheckResult:
    """Total variance at matched k across consecutive expiries, from the fitted quotes."""
    kept = clean.kept.dropna(subset=["iv_mid"])
    expiries = kept.groupby("expiry_code")["T"].first().sort_values()
    rows = []
    tested = 0
    for short_code, long_code in pairwise(expiries.index):
        short = kept[kept["expiry_code"] == short_code].sort_values("k")
        long = kept[kept["expiry_code"] == long_code].sort_values("k")
        inside = short[short["k"].between(long["k"].min(), long["k"].max())]
        tested += len(inside)
        if inside.empty:
            continue
        w_long = np.interp(inside["k"], long["k"], long["iv_mid"] ** 2 * long["T"])
        w_short = (inside["iv_mid"] ** 2 * inside["T"]).to_numpy()
        for k, wl, ws in zip(inside["k"], w_long, w_short, strict=True):
            if wl < ws - TOLERANCE:
                rows.append(
                    {
                        "check": "calendar",
                        "prices": "mid, interpolated",
                        "expiry": f"{short_code}/{long_code}",
                        "location": f"k = {k:.3f}",
                        "k_low": float(k),
                        "k_high": float(k),
                        "magnitude": float(ws - wl),
                        "in_quoted_range": True,
                    }
                )
    return CheckResult(_frame(rows), _tests([("calendar", "mid, interpolated", tested, len(rows))]))


@dataclass(frozen=True)
class ArbitrageReport:
    violations: pd.DataFrame
    tests: pd.DataFrame
    """Per check: tests run and violations, model and market apart."""
    durrleman: pd.DataFrame
    """Per slice: minimum of g inside the quoted range and over the whole grid."""
    calendar_margins: pd.DataFrame
    """Per consecutive pair: smallest increase in total variance inside the jointly quoted
    range and over the whole grid (negative means calendar arbitrage)."""

    def summary(self) -> pd.DataFrame:
        """Tests and violations per check, with violations inside the quoted range."""
        v = self.violations
        inside = (
            v[v["in_quoted_range"]].groupby(["check", "prices"]).size()
            if not v.empty
            else pd.Series(dtype=int)
        )
        out = self.tests.groupby(["check", "prices"], sort=False)[["tested", "violations"]].sum()
        out["violations_in_quoted_range"] = [int(inside.get(key, 0)) for key in out.index]
        return out.reset_index()


def check_surface(
    slices: Sequence[Slice], clean: CleanQuotes | None = None, k_grid: FloatArray = K_GRID
) -> ArbitrageReport:
    """All model tests on ``slices``, plus the market tests when quotes are given."""
    results = [check_butterfly(slices, k_grid), check_calendar(slices, k_grid)]
    if clean is not None:
        results += [check_market_quotes(clean), check_market_calendar(clean)]
    combined = _combine(results)
    return ArbitrageReport(
        violations=combined.violations,
        tests=combined.tests,
        durrleman=_durrleman_table(slices, k_grid),
        calendar_margins=_calendar_table(slices, k_grid),
    )


def _combine(results: Sequence[CheckResult]) -> CheckResult:
    violations = [r.violations for r in results if not r.violations.empty]
    tests = pd.concat([r.tests for r in results], ignore_index=True) if results else _tests([])
    if not tests.empty:
        tests = tests.groupby(["check", "prices"], sort=False, as_index=False)[
            ["tested", "violations"]
        ].sum()
    frame = pd.concat(violations, ignore_index=True) if violations else _frame([])
    return CheckResult(frame, tests)


def _durrleman_table(slices: Sequence[Slice], k_grid: FloatArray) -> pd.DataFrame:
    rows = []
    for s in slices:
        g = durrleman_g(s.params, k_grid)
        quoted = (k_grid >= s.k_min) & (k_grid <= s.k_max)
        rows.append(
            {
                "expiry_code": s.expiry_code,
                "T": s.T,
                "min_g_quoted": float(g[quoted].min()) if quoted.any() else np.nan,
                "min_g_grid": float(g.min()),
                "k_at_min_g": float(k_grid[np.argmin(g)]),
            }
        )
    return pd.DataFrame(rows)


def _calendar_table(slices: Sequence[Slice], k_grid: FloatArray) -> pd.DataFrame:
    rows = []
    for short, long in pairwise(sorted(slices, key=lambda s: s.T)):
        gap = long.params.total_variance(k_grid) - short.params.total_variance(k_grid)
        lo, hi = max(short.k_min, long.k_min), min(short.k_max, long.k_max)
        quoted = (k_grid >= lo) & (k_grid <= hi)
        rows.append(
            {
                "pair": f"{short.expiry_code}/{long.expiry_code}",
                "min_increase_quoted": float(gap[quoted].min()) if quoted.any() else np.nan,
                "min_increase_grid": float(gap.min()),
                "k_at_min": float(k_grid[np.argmin(gap)]),
            }
        )
    return pd.DataFrame(rows)

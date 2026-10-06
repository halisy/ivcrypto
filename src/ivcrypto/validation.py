"""How our implied volatilities compare with Deribit's own numbers.

Each comparison answers a different question:

1. **Mark reproduction** (convention and solver): invert Deribit's mark price with
   Deribit's own forward (``underlying_price``) and the row's own timestamp, and compare
   with Deribit's mark IV. Beyond rounding, a difference would reveal a different pricing
   convention, day count or a solver error. A mark whose price is too small for its
   8 decimal rounding to pin the volatility down to 0.01 vol points is flagged as not
   resolvable and reported separately.
2. **Ticker reproduction** (the same, for quotes): the best bid and ask of each ticker
   against Deribit's ``bid_iv`` and ``ask_iv``, split into OTM and ITM quotes, including
   whether both sides agree on when no implied volatility exists. ITM quotes are badly
   conditioned: their vega is small, so a few dollars of asynchrony between a quote and
   the reported forward moves their IV by whole vol points.
3. **Mid versus mark** (methodology): our mid IVs, with our forward, against Deribit's mark
   IV, and how often the mark lies inside our bid ask IV band.
4. **Call put gap** (forward choice): the difference between call and put mid IVs at the
   same strike within one standard deviation of the money, with our parity forward and
   with Deribit's forward. With the right forward the two agree up to quote noise.

All differences are in vol points (1 vol point = 0.01 in decimal volatility). Deribit
reports IVs with two decimals in percent, so agreement within 0.006 vol points is
agreement up to their rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ivcrypto import black76
from ivcrypto.cleaning import CleanQuotes
from ivcrypto.data.fetch import OPTION_INSTRUMENTS, OPTION_TICKERS
from ivcrypto.data.store import Snapshot
from ivcrypto.implied_vol import implied_vols
from ivcrypto.instruments import MS_PER_DAY, MS_PER_YEAR

MARK_PRICE_RESOLUTION = 1e-8
"""Deribit reports mark prices with at most 8 decimals (coins)."""
RESOLVABLE_VOL_POINTS = 0.01
"""A mark is resolvable if its price rounding moves the IV by less than this."""
ROUNDING_VOL_POINTS = 0.006
"""Deribit rounds IVs to 0.01 vol points: differences up to half of that, plus floating
point noise, mean agreement."""
CALL_PUT_MAX_ABS_K = 0.1
"""Upper limit of the call put gap window; it is also capped at one ATM standard deviation."""

DAY_BINS = [0.0, 7.0, 30.0, 90.0, np.inf]
DAY_LABELS = ["under 7 days", "7 to 30 days", "30 to 90 days", "over 90 days"]
K_BINS = [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf]
K_LABELS = [
    "k below -0.2",
    "k -0.2 to -0.05",
    "k -0.05 to 0.05",
    "k 0.05 to 0.2",
    "k above 0.2",
]


@dataclass(frozen=True)
class IVValidation:
    mark: pd.DataFrame
    tickers: pd.DataFrame | None
    mid_vs_mark: pd.DataFrame
    call_put: pd.DataFrame
    summary: pd.DataFrame


def validate_ivs(clean: CleanQuotes, snapshot: Snapshot) -> IVValidation:
    """Run every comparison; ``clean`` must already have IV columns (``add_implied_vols``)."""
    mark = mark_check(clean)
    tickers = ticker_check(snapshot) if OPTION_TICKERS in snapshot.tables else None
    mids = mid_vs_mark(clean)
    gaps = call_put_gap(clean)
    resolvable = mark[mark["resolvable"]]
    name = "mark IV reproduction, resolvable marks"
    parts = [
        summarize(resolvable, "diff", name, tolerance=ROUNDING_VOL_POINTS),
        summarize(resolvable, "diff", name, "days_bucket", tolerance=ROUNDING_VOL_POINTS),
        summarize(resolvable, "diff", name, "k_bucket", tolerance=ROUNDING_VOL_POINTS),
        summarize(mark[~mark["resolvable"]], "diff", "mark IV reproduction, unresolvable marks"),
    ]
    if tickers is not None:
        both = tickers[tickers["status"] == "both"]
        for moneyness, part in (("OTM", both[both["otm"]]), ("ITM", both[~both["otm"]])):
            parts.append(
                summarize(
                    part,
                    "diff",
                    f"ticker IV reproduction, {moneyness} quotes",
                    "side",
                    tolerance=ROUNDING_VOL_POINTS,
                )
            )
    parts += [
        summarize(mids, "diff", "our mid IV minus Deribit mark IV"),
        summarize(mids, "diff", "our mid IV minus Deribit mark IV", "days_bucket"),
        summarize(mids, "diff", "our mid IV minus Deribit mark IV", "k_bucket"),
        summarize(gaps, "gap_parity", "call minus put mid IV, parity forward", "expiry_code"),
        summarize(gaps, "gap_deribit", "call minus put mid IV, Deribit forward", "expiry_code"),
    ]
    return IVValidation(
        mark=mark,
        tickers=tickers,
        mid_vs_mark=mids,
        call_put=gaps,
        summary=pd.concat(parts, ignore_index=True),
    )


def mark_check(clean: CleanQuotes) -> pd.DataFrame:
    """Invert every positive OTM mark with Deribit's forward and compare with its mark IV."""
    q = clean.quotes
    expiry_ms = (q["expiry"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    T = ((expiry_ms - q["creation_timestamp"]) / MS_PER_YEAR).to_numpy()
    F = q["underlying_price"].to_numpy()
    K = q["strike"].to_numpy()
    is_call = (q["option_type"] == "call").to_numpy()
    otm = np.where(is_call, K >= F, K < F)
    selected = otm & (q["mark_btc"].to_numpy() > 0)
    F, K, T, is_call = F[selected], K[selected], T[selected], is_call[selected]
    rows = q.loc[selected]
    deribit = rows["deribit_mark_iv"].to_numpy()
    ours = implied_vols(rows["mark_btc"].to_numpy() * F, F, K, T, is_call)
    vega_coin = black76.vega(F, K, T, deribit) / F
    with np.errstate(divide="ignore"):
        rounding = 100 * 0.5 * MARK_PRICE_RESOLUTION / vega_coin
    frame = pd.DataFrame(
        {
            "instrument_name": rows["instrument_name"].to_numpy(),
            "expiry_code": rows["expiry_code"].to_numpy(),
            "days": rows["days"].to_numpy(),
            "k": np.log(K / F),
            "mark_btc": rows["mark_btc"].to_numpy(),
            "deribit_mark_iv": deribit,
            "iv_from_mark": ours,
            "diff": 100 * (ours - deribit),
            "rounding_vol_points": rounding,
            "resolvable": rounding < RESOLVABLE_VOL_POINTS,
        }
    )
    return _with_buckets(frame)


def ticker_check(snapshot: Snapshot) -> pd.DataFrame:
    """Our IVs from each ticker's best bid and ask against Deribit's bid_iv and ask_iv."""
    meta = snapshot[OPTION_INSTRUMENTS][
        ["instrument_name", "strike", "option_type", "expiration_timestamp"]
    ]
    t = snapshot[OPTION_TICKERS].merge(meta, on="instrument_name", how="inner")
    T = ((t["expiration_timestamp"] - t["timestamp"]) / MS_PER_YEAR).to_numpy()
    F = t["underlying_price"].to_numpy()
    K = t["strike"].to_numpy()
    is_call = (t["option_type"] == "call").to_numpy()
    otm = np.where(is_call, K >= F, K < F)
    parts = []
    for side in ("bid", "ask"):
        price = t[f"best_{side}_price"].to_numpy()
        deribit = t[f"{side}_iv"].to_numpy() / 100
        ours = np.where(price > 0, implied_vols(price * F, F, K, T, is_call), np.nan)
        has_ours = np.isfinite(ours)
        has_deribit = deribit > 0
        status = np.select(
            [price <= 0, has_ours & has_deribit, ~has_ours & ~has_deribit, has_ours],
            ["no quote", "both", "neither", "only ours"],
            default="only deribit",
        )
        parts.append(
            pd.DataFrame(
                {
                    "instrument_name": t["instrument_name"].to_numpy(),
                    "side": side,
                    "days": (t["expiration_timestamp"] - t["timestamp"]).to_numpy() / MS_PER_DAY,
                    "k": np.log(K / F),
                    "otm": otm,
                    "price_btc": price,
                    "deribit_iv": deribit,
                    "our_iv": ours,
                    "diff": np.where(status == "both", 100 * (ours - deribit), np.nan),
                    "status": status,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def mid_vs_mark(clean: CleanQuotes) -> pd.DataFrame:
    """Our mid IVs on the fitted quotes against Deribit's mark IV."""
    kept = clean.kept
    frame = pd.DataFrame(
        {
            "instrument_name": kept["instrument_name"].to_numpy(),
            "expiry_code": kept["expiry_code"].to_numpy(),
            "days": kept["days"].to_numpy(),
            "k": kept["k"].to_numpy(),
            "iv_bid": kept["iv_bid"].to_numpy(),
            "iv_mid": kept["iv_mid"].to_numpy(),
            "iv_ask": kept["iv_ask"].to_numpy(),
            "deribit_mark_iv": kept["deribit_mark_iv"].to_numpy(),
        }
    )
    frame["diff"] = 100 * (frame["iv_mid"] - frame["deribit_mark_iv"])
    frame["mark_in_band"] = frame["deribit_mark_iv"].between(frame["iv_bid"], frame["iv_ask"])
    return _with_buckets(frame)


def call_put_gap(clean: CleanQuotes) -> pd.DataFrame:
    """Call minus put mid IV at the same strike, with our forward and with Deribit's.

    Uses strikes within one ATM standard deviation (capped at |k| = 0.1): further out, one
    of the two options is deep in the money and its mid gives an ill conditioned IV.
    """
    q = clean.quotes
    window = _atm_total_vol(clean).clip(upper=CALL_PUT_MAX_ABS_K)
    in_window = q["k"].abs() <= q["expiry_code"].map(window).fillna(0.0)
    two_sided = q["bid_btc"].notna() & q["ask_btc"].notna() & (q["bid_btc"] < q["ask_btc"])
    q = q[two_sided & in_window]
    columns = ["expiry_code", "days", "strike", "k", "gap_parity", "gap_deribit"]
    if q.empty:
        return pd.DataFrame(columns=columns)
    is_call = (q["option_type"] == "call").to_numpy()
    ivs = {}
    for label, forward in (("parity", q["forward"]), ("deribit", q["underlying_price"])):
        F = forward.to_numpy()
        ivs[label] = implied_vols(
            q["mid_btc"].to_numpy() * F, F, q["strike"].to_numpy(), q["T"].to_numpy(), is_call
        )
    frame = q[["expiry_code", "days", "strike", "k", "option_type"]].assign(
        iv_parity=ivs["parity"], iv_deribit=ivs["deribit"]
    )
    wide = frame.pivot_table(
        index=["expiry_code", "days", "strike", "k"],
        columns="option_type",
        values=["iv_parity", "iv_deribit"],
    ).dropna()
    if wide.empty or "call" not in wide["iv_parity"] or "put" not in wide["iv_parity"]:
        return pd.DataFrame(columns=columns)
    out = wide.index.to_frame(index=False)
    for label in ("parity", "deribit"):
        gap = wide[(f"iv_{label}", "call")] - wide[(f"iv_{label}", "put")]
        out[f"gap_{label}"] = 100 * gap.to_numpy()
    return out.sort_values(["days", "strike"], ignore_index=True)


def summarize(
    frame: pd.DataFrame,
    column: str,
    comparison: str,
    by: str | None = None,
    *,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Distribution of ``frame[column]`` overall or per group, one row per group.

    With ``tolerance``, also the share of values within it in absolute value.
    """
    if by is None:
        groups = [("all", frame[column])]
    else:
        grouped = frame.groupby(by, sort=True, observed=True)[column]
        groups = [(str(name), values) for name, values in grouped]
    rows = []
    for name, values in groups:
        row = {"comparison": comparison, "group": name, **distribution(values)}
        finite = values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        within = (
            float(np.mean(np.abs(finite) <= tolerance)) if tolerance and finite.size else np.nan
        )
        row["share_within_rounding"] = within
        rows.append(row)
    return pd.DataFrame(rows)


def distribution(values: pd.Series) -> dict[str, float]:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, **dict.fromkeys(["mean", "median", "std", "p5", "p95", "max_abs"], np.nan)}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "max_abs": float(np.abs(finite).max()),
    }


def _atm_total_vol(clean: CleanQuotes) -> pd.Series:
    """Per expiry, sigma sqrt(T) from the mid IVs of the four kept quotes nearest the money."""
    kept = clean.kept.dropna(subset=["iv_mid"])
    nearest = kept.assign(abs_k=kept["k"].abs()).sort_values("abs_k").groupby("expiry_code").head(4)
    total_vol = nearest["iv_mid"] * np.sqrt(nearest["T"])
    return total_vol.groupby(nearest["expiry_code"]).median()


def _with_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    frame["days_bucket"] = pd.cut(frame["days"], DAY_BINS, labels=DAY_LABELS, right=False)
    frame["k_bucket"] = pd.cut(frame["k"], K_BINS, labels=K_LABELS)
    return frame

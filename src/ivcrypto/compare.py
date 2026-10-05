"""Model comparison: SVI, SSVI and Heston on the same quotes, and the ATM skew term structure.

Every model is scored on the same fitted quotes with the same measures: RMSE of fitted minus
mid IV (vol points), largest error, and the share of fitted IVs inside the bid ask band. The
breakdown by maturity and moneyness shows where a model fails, not only how much.

The ATM term structure is the sharpest test of Heston for short dated crypto options. In
Heston the ATM skew dsigma/dk and curvature d2sigma/dk2 tend to constants as T -> 0 (the skew
to rho xi / (4 sqrt(v0))), whereas jumps make them explode as T shrinks. The market values
are read off the SVI slices,

    sigma'(0) = w' / (2 sqrt(w T)),    sigma''(0) = w'' / (2 sqrt(w T)) - w'^2 / (4 w^1.5 sqrt(T)),

the Heston values by central differences of its implied vols, and a power law
c T^(-alpha) is fitted to each. On short dated crypto smiles whose bottom sits slightly off
the money, the skew changes sign across maturities, so the curvature is the cleaner measure.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ivcrypto.heston.calibrate import HestonCalibration
from ivcrypto.heston.charfunc import HestonParams
from ivcrypto.heston.pricer import price
from ivcrypto.implied_vol import implied_vols
from ivcrypto.svi.fit import SVISurface
from ivcrypto.svi.ssvi import SSVIFit
from ivcrypto.validation import DAY_BINS, DAY_LABELS, K_BINS, K_LABELS


def _scores(frame: pd.DataFrame, model_column: str) -> dict[str, float]:
    error = 100.0 * (frame[model_column] - frame["iv_mid"])
    inside = (frame[model_column] >= frame["iv_bid"] - 1e-12) & (
        frame[model_column] <= frame["iv_ask"] + 1e-12
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
        "inside_band": float(np.mean(inside)),
    }


def model_ivs(
    svi: SVISurface,
    heston: HestonCalibration,
    heston_per_expiry: Mapping[str, HestonCalibration] | None = None,
    ssvi: SSVIFit | None = None,
) -> pd.DataFrame:
    """One row per fitted quote with the mid, bid, ask and every model's IV."""
    frame = svi.residuals()[
        ["expiry_code", "T", "instrument_name", "k", "iv_bid", "iv_mid", "iv_ask", "iv_svi"]
    ]
    frame = frame.merge(
        heston.residuals[["instrument_name", "iv_heston"]], on="instrument_name", how="left"
    )
    if heston_per_expiry:
        per = pd.concat(
            [cal.residuals[["instrument_name", "iv_heston"]] for cal in heston_per_expiry.values()]
        ).rename(columns={"iv_heston": "iv_heston_expiry"})
        frame = frame.merge(per, on="instrument_name", how="left")
    if ssvi is not None:
        frame = frame.merge(
            ssvi.residuals[["instrument_name", "iv_ssvi"]], on="instrument_name", how="left"
        )
    frame["days"] = frame["T"] * 365.0
    frame["days_bucket"] = pd.cut(frame["days"], DAY_BINS, labels=DAY_LABELS, right=False)
    frame["k_bucket"] = pd.cut(frame["k"], K_BINS, labels=K_LABELS)
    return frame


MODEL_COLUMNS = {
    "SVI": "iv_svi",
    "SSVI": "iv_ssvi",
    "Heston": "iv_heston",
    "Heston per expiry": "iv_heston_expiry",
}


def compare_models(ivs: pd.DataFrame, by: str | None = "expiry_code") -> pd.DataFrame:
    """Scores of every available model, overall (``by=None``) or per group."""
    models = {name: col for name, col in MODEL_COLUMNS.items() if col in ivs}
    groups = [("all", ivs)] if by is None else list(ivs.groupby(by, sort=False, observed=True))
    rows = []
    for name, part in groups:
        row: dict[str, object] = {
            "group": str(name),
            "days": float(part["days"].mean()),
            "n": len(part),
        }
        for model, column in models.items():
            for measure, value in _scores(part.dropna(subset=[column]), column).items():
                row[f"{model} {measure}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def svi_atm(svi: SVISurface) -> pd.DataFrame:
    """ATM vol, skew dsigma/dk and curvature d2sigma/dk2 at k = 0 of each SVI slice."""
    rows = []
    for fit in svi.fits.values():
        w, dw, d2w = (float(x) for x in fit.params.derivatives(0.0))
        root = np.sqrt(w * fit.T)
        rows.append(
            {
                "expiry_code": fit.expiry_code,
                "T": fit.T,
                "atm_vol": float(np.sqrt(w / fit.T)),
                "atm_skew": dw / (2.0 * root),
                "atm_curvature": d2w / (2.0 * root) - dw * dw / (4.0 * w * root),
            }
        )
    return pd.DataFrame(rows)


def heston_atm(params: HestonParams, T: np.ndarray, forward: float = 1.0) -> pd.DataFrame:
    """ATM vol, skew and curvature of a Heston model by central differences of its IVs."""
    rows = []
    for t in np.asarray(T, dtype=float):
        h = min(0.01, 0.25 * np.sqrt(params.theta * t))
        k = np.array([-h, 0.0, h])
        strikes = forward * np.exp(k)
        calls = k >= 0
        ivs = implied_vols(price(forward, strikes, t, params, calls), forward, strikes, t, calls)
        rows.append(
            {
                "T": t,
                "atm_vol": float(ivs[1]),
                "atm_skew": float((ivs[2] - ivs[0]) / (2 * h)),
                "atm_curvature": float((ivs[2] - 2 * ivs[1] + ivs[0]) / (h * h)),
            }
        )
    return pd.DataFrame(rows)


def power_law(T: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Fit |values| = c T^(-alpha) by least squares in logs; returns (c, alpha)."""
    T = np.asarray(T, dtype=float)
    y = np.abs(np.asarray(values, dtype=float))
    ok = (T > 0) & (y > 0)
    slope, intercept = np.polyfit(np.log(T[ok]), np.log(y[ok]), 1)
    return float(np.exp(intercept)), float(-slope)


def atm_term_structure(svi: SVISurface, heston: HestonParams) -> pd.DataFrame:
    """Market (SVI) and Heston ATM vol and skew at the fitted expiries."""
    market = svi_atm(svi)
    model = heston_atm(heston, market["T"].to_numpy())
    return pd.DataFrame(
        {
            "expiry_code": market["expiry_code"],
            "T": market["T"],
            "days": market["T"] * 365.0,
            "market_atm_vol": market["atm_vol"],
            "heston_atm_vol": model["atm_vol"],
            "market_atm_skew": market["atm_skew"],
            "heston_atm_skew": model["atm_skew"],
            "market_atm_curvature": market["atm_curvature"],
            "heston_atm_curvature": model["atm_curvature"],
        }
    )

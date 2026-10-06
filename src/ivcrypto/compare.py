"""Model comparison: SVI, SSVI, Heston and Bates on the same quotes, and ATM term structures.

Every model is scored on the same fitted quotes with the same measures: RMSE of fitted minus
mid IV (vol points), largest error, and the share of fitted IVs inside the bid ask band. The
breakdown by maturity and moneyness shows where a model fails, not only how much.

The ATM term structure is the sharpest test for short dated crypto options. In Heston the
skew dsigma/dk and curvature d2sigma/dk2 at the money tend to finite limits as T -> 0 (the
skew to rho xi / (4 sqrt(v0))), whereas with jumps the curvature grows without bound, like
T^(-1/2) in a jump diffusion (``tests/test_asymptotics.py`` checks both with
:func:`model_atm`).

To compare market and models, skew and curvature are measured over one ATM standard deviation,
s = sqrt(w(0)), at the same three strikes for every model:

    skew = (sigma(s) - sigma(-s)) / (2 s),    curvature = (sigma(s) + sigma(-s) - 2 sigma(0)) / s^2,

a risk reversal and a butterfly scaled to derivatives. The market's vols come from its SVI fit,
the models' from their own prices. Derivatives exactly at k = 0 are fragile: a smile whose
bottom sits just off the money, or a jump model whose nearly fixed jump sizes make its short
dated smile wavy, can have a negative second derivative at k = 0 although it is convex over
the quoted strikes. For smooth smiles the two measures agree (on the BTC sample the market's
power law exponents differ by under 0.01), and with s proportional to sqrt(T) both keep the
short maturity limits above. A power law c T^(-alpha) is fitted to each curvature term
structure.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ivcrypto.bates.charfunc import BatesParams
from ivcrypto.heston.calibrate import Calibration
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
    heston: Calibration,
    heston_per_expiry: Mapping[str, Calibration] | None = None,
    ssvi: SSVIFit | None = None,
    bates: Calibration | None = None,
) -> pd.DataFrame:
    """One row per fitted quote with the mid, bid, ask and every model's IV."""
    frame = svi.residuals()[
        ["expiry_code", "T", "instrument_name", "k", "iv_bid", "iv_mid", "iv_ask", "iv_svi"]
    ]
    for calibration in (heston, bates):
        if calibration is not None:
            columns = ["instrument_name", calibration.column]
            frame = frame.merge(calibration.residuals[columns], on="instrument_name", how="left")
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
    "Bates": "iv_bates",
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


def _model_vols(params: HestonParams | BatesParams, k: np.ndarray, T: float) -> np.ndarray:
    strikes = np.exp(k)
    calls = k >= 0
    return implied_vols(price(1.0, strikes, T, params, calls), 1.0, strikes, T, calls)


def _shape(name: str, vols: np.ndarray, s: float) -> dict[str, float]:
    low, atm, high = (float(v) for v in vols)
    return {
        f"{name}_atm_vol": atm,
        f"{name}_atm_skew": (high - low) / (2.0 * s),
        f"{name}_atm_curvature": (high + low - 2.0 * atm) / (s * s),
    }


def model_atm(
    params: HestonParams | BatesParams, T: np.ndarray, forward: float = 1.0
) -> pd.DataFrame:
    """Local ATM vol, skew and curvature of a model by central differences of its IVs."""
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


def atm_term_structure(
    svi: SVISurface, models: Mapping[str, HestonParams | BatesParams]
) -> pd.DataFrame:
    """ATM vol, skew and curvature of the market (its SVI fit) and of each model, measured at
    the money and one ATM standard deviation either side (see the module docstring)."""
    rows = []
    for fit in svi.fits.values():
        s = float(np.sqrt(fit.params.total_variance(0.0)))
        k = np.array([-s, 0.0, s])
        row: dict[str, object] = {
            "expiry_code": fit.expiry_code,
            "T": fit.T,
            "days": fit.T * 365.0,
            "atm_std": s,
            **_shape("market", fit.params.implied_vol(k, fit.T), s),
        }
        for name, params in models.items():
            row.update(_shape(name, _model_vols(params, k, fit.T), s))
        rows.append(row)
    return pd.DataFrame(rows)

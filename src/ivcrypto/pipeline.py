"""The whole analysis of one snapshot, and its results on disk.

``analyze`` runs every stage (cleaning, implied volatilities, validation against Deribit,
SVI, SSVI, static arbitrage, Heston, Bates and the model comparison). ``write_results`` stores
everything under ``<out>/<CURRENCY>/<snapshot id>/`` as Parquet, CSV and JSON, together with
the effective configuration and the git commit, so every figure and number in a report
traces back to its inputs. ``load_results`` reads a results directory back for reporting.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ivcrypto import __version__
from ivcrypto.arbitrage import (
    ArbitrageReport,
    check_surface,
    slices_from_ssvi,
    slices_from_svi,
)
from ivcrypto.bates.calibrate import calibrate_bates, calibrate_variants
from ivcrypto.cleaning import CleanQuotes, clean_snapshot
from ivcrypto.compare import atm_term_structure, compare_models, model_ivs
from ivcrypto.config import Config, config_to_dict
from ivcrypto.data.store import Snapshot, load_snapshot
from ivcrypto.heston.calibrate import (
    Calibration,
    calibrate_heston,
    calibrate_per_expiry,
    calibration_quotes,
)
from ivcrypto.implied_vol import add_implied_vols
from ivcrypto.svi.fit import SVISurface, fit_svi
from ivcrypto.svi.ssvi import SSVIFit, fit_ssvi
from ivcrypto.validation import IVValidation, validate_ivs

logger = logging.getLogger(__name__)

SSVI_CHECK_MATURITIES = 120
"""Maturities between the first day and 1.2 years at which the SSVI surface is checked."""


@dataclass
class Analysis:
    snapshot: Snapshot
    config: Config
    clean: CleanQuotes
    validation: IVValidation
    svi: SVISurface
    ssvi: SSVIFit
    arbitrage: ArbitrageReport
    ssvi_arbitrage: ArbitrageReport
    heston: Calibration
    heston_per_expiry: dict[str, Calibration]
    bates: Calibration
    bates_variants: dict[str, Calibration]
    model_ivs: pd.DataFrame
    atm: pd.DataFrame
    timings: dict[str, float]


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    timings[stage] = round(time.perf_counter() - start, 3)
    logger.info("%s done in %.1f s", stage, timings[stage])


def analyze(
    snapshot: Snapshot,
    config: Config,
    *,
    per_expiry_heston: bool = True,
    bates_variants: bool = True,
) -> Analysis:
    timings: dict[str, float] = {}
    with _timed(timings, "cleaning and implied volatilities"):
        clean = add_implied_vols(clean_snapshot(snapshot, config.cleaning))
    with _timed(timings, "validation against Deribit"):
        validation = validate_ivs(clean, snapshot)
    with _timed(timings, "SVI"):
        svi = fit_svi(clean, config.svi)
    with _timed(timings, "SSVI"):
        ssvi = fit_ssvi(clean, svi, config.svi)
    with _timed(timings, "static arbitrage"):
        arbitrage = check_surface(slices_from_svi(svi), clean)
        maturities = pd.Series(range(1, SSVI_CHECK_MATURITIES + 1)) / SSVI_CHECK_MATURITIES * 1.2
        ssvi_arbitrage = check_surface(slices_from_ssvi(ssvi, maturities.to_numpy()))
    quotes = calibration_quotes(clean)
    with _timed(timings, "Heston"):
        heston = calibrate_heston(quotes, config.heston)
    per_expiry: dict[str, Calibration] = {}
    if per_expiry_heston:
        with _timed(timings, "Heston per expiry"):
            per_expiry = calibrate_per_expiry(quotes, config.heston)
    with _timed(timings, "Bates"):
        bates = calibrate_bates(quotes, heston.params, config.heston, config.bates)
    variants: dict[str, Calibration] = {}
    if bates_variants:
        with _timed(timings, "Bates variants"):
            variants = calibrate_variants(quotes, heston.params, config.heston, config.bates)
    with _timed(timings, "comparison"):
        ivs = model_ivs(svi, heston, per_expiry or None, ssvi, bates)
        atm = atm_term_structure(svi, {"heston": heston.params, "bates": bates.params})
    return Analysis(
        snapshot=snapshot,
        config=config,
        clean=clean,
        validation=validation,
        svi=svi,
        ssvi=ssvi,
        arbitrage=arbitrage,
        ssvi_arbitrage=ssvi_arbitrage,
        heston=heston,
        heston_per_expiry=per_expiry,
        bates=bates,
        bates_variants=variants,
        model_ivs=ivs,
        atm=atm,
        timings=timings,
    )


def results_dir(root: Path | str, snapshot: Snapshot) -> Path:
    return Path(root) / snapshot.currency / snapshot.path.name


def write_results(analysis: Analysis, root: Path | str) -> Path:
    """Write every result table under ``root/<CURRENCY>/<snapshot id>/``; returns the path."""
    out = results_dir(root, analysis.snapshot)
    out.mkdir(parents=True, exist_ok=True)
    a = analysis

    def parquet(name: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(out / f"{name}.parquet", index=False)

    def csv(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(out / f"{name}.csv", index=False)

    parquet("quotes", a.clean.quotes)
    csv("forwards", a.clean.forwards)
    csv("filters", a.clean.filters)
    csv("validation_summary", a.validation.summary)
    parquet("validation_marks", a.validation.mark)
    parquet("validation_mid_vs_mark", a.validation.mid_vs_mark)
    csv("validation_call_put", a.validation.call_put)
    if a.validation.tickers is not None:
        parquet("validation_tickers", a.validation.tickers)
    csv("svi", a.svi.summary())
    parquet("svi_residuals", a.svi.residuals())
    csv("ssvi", a.ssvi.summary())
    parquet("ssvi_residuals", a.ssvi.residuals)
    csv("arbitrage_tests", a.arbitrage.summary())
    csv("arbitrage_violations", a.arbitrage.violations)
    csv("arbitrage_durrleman", a.arbitrage.durrleman)
    csv("arbitrage_calendar", a.arbitrage.calendar_margins)
    csv("ssvi_arbitrage_tests", a.ssvi_arbitrage.summary())
    csv("heston", a.heston.summary())
    parquet("heston_residuals", a.heston.residuals)
    if a.heston_per_expiry:
        csv("heston_per_expiry", _per_expiry_table(a.heston_per_expiry))
    csv("bates", a.bates.summary())
    parquet("bates_residuals", a.bates.residuals)
    if a.bates_variants:
        variants = {"default bounds": a.bates, **a.bates_variants}
        csv("bates_variants", _variants_table(variants))
    parquet("model_ivs", a.model_ivs)
    csv("comparison_overall", compare_models(a.model_ivs, by=None))
    csv("comparison_by_expiry", compare_models(a.model_ivs, by="expiry_code"))
    csv("comparison_by_maturity", compare_models(a.model_ivs, by="days_bucket"))
    csv("comparison_by_moneyness", compare_models(a.model_ivs, by="k_bucket"))
    csv("atm_term_structure", a.atm)
    manifest = {
        "ivcrypto_version": __version__,
        "git_commit": _git_commit(),
        "snapshot": str(a.snapshot.path),
        "snapshot_utc": a.snapshot.manifest["snapshot_utc"],
        "currency": a.snapshot.currency,
        "valuation_time": a.clean.valuation_time.isoformat(),
        "config": config_to_dict(a.config),
        "timings_seconds": a.timings,
        "heston": {
            **asdict(a.heston.params),
            "feller_ratio": a.heston.params.feller_ratio,
            "cost": a.heston.cost,
            "start_costs": list(a.heston.start_costs),
            "start_params": [list(x) for x in a.heston.start_params],
            "at_bounds": list(a.heston.at_bounds),
        },
        "bates": {
            **asdict(a.bates.params),
            "feller_ratio": a.bates.params.feller_ratio,
            "jump_mean": a.bates.params.jump_mean,
            "jump_variance": a.bates.params.jump_variance,
            "cost": a.bates.cost,
            "start_costs": list(a.bates.start_costs),
            "start_params": [list(x) for x in a.bates.start_params],
            "at_bounds": list(a.bates.at_bounds),
        },
        "ssvi": {
            **asdict(a.ssvi.params),
            "expiries": list(a.ssvi.expiries),
            "T": a.ssvi.T.tolist(),
            "theta": a.ssvi.theta.tolist(),
            "cost": a.ssvi.cost,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    logger.info("results written to %s", out)
    return out


def _per_expiry_table(per_expiry: dict[str, Calibration]) -> pd.DataFrame:
    rows = []
    for code, cal in per_expiry.items():
        quality = cal.summary().iloc[0].to_dict()
        rows.append(
            {
                "expiry_code": code,
                **asdict(cal.params),
                "feller_ratio": cal.params.feller_ratio,
                **{k: v for k, v in quality.items() if k != "expiry_code"},
                "at_bounds": "; ".join(cal.at_bounds),
            }
        )
    return pd.DataFrame(rows)


SHORT_END_DAYS = 7.0
FAR_PUT_K = -0.2


def _variants_table(fits: dict[str, Calibration]) -> pd.DataFrame:
    """Bates fits under different jump bounds: parameters, fit quality and binding bounds."""
    rows = []
    for name, cal in fits.items():
        r = cal.residuals
        error = r["error_vol"]
        short = r["T"] * 365.0 < SHORT_END_DAYS
        far_put = r["k"] < FAR_PUT_K
        rows.append(
            {
                "variant": name,
                **asdict(cal.params),
                "feller_ratio": cal.params.feller_ratio,
                "jump_mean": cal.params.jump_mean,
                "cost": cal.cost,
                "rmse_vol": float(np.sqrt(np.mean(error**2))),
                "rmse_short_end": float(np.sqrt(np.mean(error[short] ** 2))),
                "rmse_far_puts": float(np.sqrt(np.mean(error[far_put] ** 2))),
                "in_band_share": float(r["in_band"].mean()),
                "at_bounds": "; ".join(cal.at_bounds),
            }
        )
    return pd.DataFrame(rows)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class Results:
    """A results directory read back: tables by name and the manifest."""

    path: Path
    manifest: dict[str, Any]
    tables: dict[str, pd.DataFrame]

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def get(self, name: str) -> pd.DataFrame | None:
        return self.tables.get(name)


def load_results(path: Path | str) -> Results:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    tables: dict[str, pd.DataFrame] = {}
    for file in sorted(path.iterdir()):
        if file.suffix == ".parquet":
            tables[file.stem] = pd.read_parquet(file)
        elif file.suffix == ".csv":
            tables[file.stem] = pd.read_csv(file)
    return Results(path=path, manifest=manifest, tables=tables)


def build(
    snapshot_path: Path | str,
    config: Config,
    out: Path | str,
    *,
    per_expiry_heston: bool = True,
    bates_variants: bool = True,
) -> Path:
    """Load a snapshot, analyze it and write the results; returns the results directory."""
    analysis = analyze(
        load_snapshot(snapshot_path),
        config,
        per_expiry_heston=per_expiry_heston,
        bates_variants=bates_variants,
    )
    return write_results(analysis, out)

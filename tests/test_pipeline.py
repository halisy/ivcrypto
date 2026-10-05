"""build and report end to end on the committed BTC snapshot (offline), plus helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ivcrypto import cli
from ivcrypto.pipeline import load_results
from ivcrypto.plots import residual_table, surface_grid
from ivcrypto.report import markdown_table

EXPECTED_TABLES = {
    "quotes",
    "forwards",
    "filters",
    "validation_summary",
    "validation_marks",
    "validation_tickers",
    "svi",
    "svi_residuals",
    "ssvi",
    "arbitrage_tests",
    "arbitrage_violations",
    "ssvi_arbitrage_tests",
    "heston",
    "heston_residuals",
    "model_ivs",
    "comparison_by_expiry",
    "comparison_by_maturity",
    "comparison_by_moneyness",
    "atm_term_structure",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory, btc_sample):
    out = tmp_path_factory.mktemp("results")
    code = cli.main(["build", str(btc_sample.path), "--out", str(out), "--no-heston-per-expiry"])
    assert code == 0
    return load_results(out / "BTC" / btc_sample.path.name)


def test_build_writes_every_table_and_a_manifest(built, btc_sample):
    assert set(built.tables) >= EXPECTED_TABLES
    assert "heston_per_expiry" not in built.tables  # skipped on request
    manifest = built.manifest
    assert manifest["snapshot_utc"] == btc_sample.manifest["snapshot_utc"]
    assert manifest["currency"] == "BTC"
    assert set(manifest["heston"]) >= {"v0", "kappa", "theta", "xi", "rho", "feller_ratio"}
    assert len(manifest["ssvi"]["theta"]) == len(built["svi"])
    assert manifest["config"]["cleaning"]["forward_method"] == "parity"
    assert all(seconds >= 0 for seconds in manifest["timings_seconds"].values())


def test_results_round_trip_the_analysis(built, btc_sample):
    quotes = built["quotes"]
    assert len(quotes) == len(btc_sample["option_book"])
    kept = quotes[quotes["removed_by"].isna()]
    assert len(built["model_ivs"]) == len(kept)
    assert built["svi"]["in_band_share"].min() >= 0.9
    assert (built["ssvi_arbitrage_tests"]["violations"] == 0).all()


def test_report_writes_markdown_and_figures(built, tmp_path, capsys):
    out = tmp_path / "report"
    code = cli.main(["report", str(built.path), "--out", str(out), "--themes", "light"])
    assert code == 0
    text = (out / "report.md").read_text()
    for heading in (
        "## Cleaning",
        "## Implied volatility against Deribit",
        "## Smiles",
        "## Static arbitrage",
        "## Heston",
        "## Surface",
    ):
        assert heading in text
    assert "prefers-color-scheme" not in text  # light only was requested
    for name in ("smiles", "atm_term_structure", "surface", "residual_heatmaps", "iv_validation"):
        figure = out / "figures" / f"{name}_light.png"
        assert figure.stat().st_size > 10_000
        assert not (out / "figures" / f"{name}_dark.png").exists()
    assert str(out / "report.md") in capsys.readouterr().out


def test_residual_table_averages_by_expiry_and_bucket():
    ivs = pd.DataFrame(
        {
            "expiry_code": ["A", "A", "A", "B"],
            "T": [0.1, 0.1, 0.1, 0.5],
            "k": [-0.3, -0.25, 0.0, 0.3],
            "iv_mid": [0.5, 0.5, 0.4, 0.45],
            "iv_svi": [0.51, 0.53, 0.4, 0.44],
        }
    )
    table = residual_table(ivs, "iv_svi")
    assert list(table.index) == ["A", "B"]
    assert table.loc["A", "k below -0.2"] == pytest.approx(2.0)  # mean of +1 and +3
    assert table.loc["A", "k -0.05 to 0.05"] == pytest.approx(0.0)
    assert table.loc["B", "k above 0.2"] == pytest.approx(-1.0)
    assert np.isnan(table.loc["B", "k below -0.2"])


def test_surface_grid_matches_the_slices_at_expiries(built):
    x = np.linspace(-1.0, 1.0, 5)
    log_days, vols = surface_grid(built, x, n_days=7)
    assert vols.shape == (7, 5)
    svi = built["svi"].sort_values("T")
    assert log_days[0] == pytest.approx(np.log10(svi["days"].iloc[0]))
    assert vols[0, 2] == pytest.approx(svi["atm_vol"].iloc[0], rel=1e-9)  # x = 0 is ATM


def test_markdown_table_formats_and_blanks_missing_values():
    frame = pd.DataFrame({"name": ["a", "b"], "value": [1.23456, float("nan")]})
    table = markdown_table(frame, [("name", "Name", ""), ("value", "Value", ".2f")])
    assert table.splitlines() == ["| Name | Value |", "|---|---|", "| a | 1.23 |", "| b |  |"]


def test_manifest_is_valid_json(built):
    json.loads((built.path / "manifest.json").read_text())

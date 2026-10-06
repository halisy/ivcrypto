"""Our IVs against Deribit's own numbers on the committed real snapshot."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ivcrypto.cleaning import clean_snapshot
from ivcrypto.implied_vol import add_implied_vols
from ivcrypto.validation import ROUNDING_VOL_POINTS, validate_ivs


@pytest.fixture(scope="module")
def btc_validation(btc_sample):
    return validate_ivs(add_implied_vols(clean_snapshot(btc_sample)), btc_sample)


def summary_row(validation, comparison, group="all"):
    summary = validation.summary
    rows = summary[(summary["comparison"] == comparison) & (summary["group"] == group)]
    assert len(rows) == 1, (comparison, group)
    return rows.iloc[0]


def test_marks_are_reproduced_from_deribits_own_forward(btc_validation):
    mark = btc_validation.mark
    assert mark["resolvable"].mean() > 0.95
    row = summary_row(btc_validation, "mark IV reproduction, resolvable marks")
    assert abs(row["median"]) < 0.005
    assert -0.05 < row["p5"] < row["p95"] < 0.05
    long_dated = summary_row(
        btc_validation, "mark IV reproduction, resolvable marks", "over 90 days"
    )
    assert long_dated["max_abs"] < 0.02


def test_ticker_quotes_are_reproduced_up_to_rounding(btc_validation):
    for side in ("bid", "ask"):
        row = summary_row(btc_validation, "ticker IV reproduction, OTM quotes", side)
        assert row["share_within_rounding"] > 0.9
        assert abs(row["median"]) <= ROUNDING_VOL_POINTS


def test_we_and_deribit_agree_on_when_no_iv_exists(btc_validation):
    status = btc_validation.tickers["status"]
    assert not status.isin(["only ours", "only deribit"]).any()
    assert (status == "neither").sum() > 100  # ITM bids at or below intrinsic value


def test_deribits_mark_lies_inside_our_bid_ask_band(btc_validation):
    assert btc_validation.mid_vs_mark["mark_in_band"].mean() > 0.98
    row = summary_row(btc_validation, "our mid IV minus Deribit mark IV")
    assert abs(row["median"]) < 0.05


def test_parity_forward_shrinks_the_call_put_gap(btc_validation):
    gaps = btc_validation.call_put[btc_validation.call_put["days"] >= 7]
    assert len(gaps) > 50
    assert np.median(np.abs(gaps["gap_parity"])) < np.median(np.abs(gaps["gap_deribit"]))
    per_expiry = gaps.groupby("expiry_code")["gap_parity"].median()
    assert per_expiry.abs().max() < 0.2


def test_validation_runs_without_tickers(btc_sample):
    tables = {k: v for k, v in btc_sample.tables.items() if k != "option_tickers"}
    snapshot = replace(btc_sample, tables=tables)
    validation = validate_ivs(add_implied_vols(clean_snapshot(snapshot)), snapshot)
    assert validation.tickers is None
    assert not validation.summary["comparison"].str.startswith("ticker").any()

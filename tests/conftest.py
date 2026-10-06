"""Shared test configuration.

Every test runs offline: an autouse fixture makes any socket connection fail
loudly, so a test can never silently depend on the live Deribit API. Tests
that are meant to hit the network carry ``@pytest.mark.network`` and are
deselected by default (see ``addopts`` in pyproject.toml).
"""

from __future__ import annotations

import copy
import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ivcrypto import black76
from ivcrypto.data.client import PRODUCTION_URL, DeribitAPIError, RpcResult
from ivcrypto.data.store import Snapshot, find_snapshots, load_snapshot
from ivcrypto.heston.calibrate import ExpiryQuotes
from ivcrypto.heston.pricer import FourierModel, price
from ivcrypto.implied_vol import implied_vols

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DERIBIT_FIXTURES = FIXTURES / "deribit"
SAMPLE_ROOT = REPO_ROOT / "data" / "sample"


class NetworkAccessError(RuntimeError):
    """Raised when a test opens a network connection without the network marker."""


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("network") is not None:
        return

    def guard(*args: object, **kwargs: object) -> None:
        raise NetworkAccessError("tests must run offline, but a network connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def _only_sample(currency: str) -> Snapshot:
    paths = find_snapshots(SAMPLE_ROOT, currency)
    assert len(paths) == 1, f"expected one committed {currency} sample, found {paths}"
    return load_snapshot(paths[0])


@pytest.fixture(scope="session")
def btc_sample() -> Snapshot:
    """The committed real BTC snapshot (read only; copy tables before modifying)."""
    return _only_sample("BTC")


@pytest.fixture(scope="session")
def eth_sample() -> Snapshot:
    """The committed real ETH snapshot (read only; copy tables before modifying)."""
    return _only_sample("ETH")


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((DERIBIT_FIXTURES / name).read_text(encoding="utf-8"))


def _fixture_name(method: str, params: dict[str, Any]) -> str:
    if method == "public/get_instruments":
        assert params == {"currency": "BTC", "kind": "option", "expired": False}
        return "get_instruments_btc_option.json"
    if method == "public/get_book_summary_by_currency":
        assert params["currency"] == "BTC"
        return f"get_book_summary_btc_{params['kind']}.json"
    if method == "public/get_index_price":
        return f"get_index_price_{params['index_name']}.json"
    if method == "public/ticker":
        return f"ticker_{params['instrument_name']}.json"
    raise AssertionError(f"unexpected call {method} {params}")


Edit = Callable[[str, dict[str, Any]], dict[str, Any]]


class FixtureCaller:
    """Stands in for ``DeribitClient``: answers calls with the captured real responses.

    ``edit(method, payload)`` may return a modified copy of a payload, so a test
    can simulate a schema change or an error without touching the fixture files.
    """

    base_url = PRODUCTION_URL

    def __init__(self, edit: Edit | None = None) -> None:
        self.edit = edit
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, **params: Any) -> RpcResult:
        self.calls.append((method, params))
        payload = copy.deepcopy(load_fixture(_fixture_name(method, params)))
        if self.edit is not None:
            payload = self.edit(method, payload)
        if "error" in payload:
            error = payload["error"]
            raise DeribitAPIError(method, error["code"], error["message"], error.get("data"), 400)
        return RpcResult(
            method=method,
            params=params,
            result=payload["result"],
            us_in=payload["usIn"],
            us_out=payload["usOut"],
            testnet=payload["testnet"],
        )


@pytest.fixture
def fixture_caller() -> Callable[..., FixtureCaller]:
    return FixtureCaller


def _synthetic_quotes(
    params: FourierModel, maturities: list[float], F: float = 86_000.0
) -> list[ExpiryQuotes]:
    """Quotes priced by the model itself, with a 1 vol point band, for recovery tests."""
    quotes = []
    for T in maturities:
        k = np.linspace(-2.5, 2.0, 15) * np.sqrt(params.theta * T)
        strikes = F * np.exp(k)
        calls = k >= 0
        mid = price(F, strikes, T, params, calls)
        iv = implied_vols(mid, F, strikes, T, calls)
        quotes.append(
            ExpiryQuotes(
                expiry_code=f"T{T:.3f}",
                T=T,
                F=F,
                instrument_name=np.array([f"q{i}" for i in range(k.size)], dtype=object),
                K=strikes,
                k=k,
                is_call=calls,
                mid=mid,
                iv_bid=iv - 0.005,
                iv_mid=iv,
                iv_ask=iv + 0.005,
                vega=black76.vega(F, strikes, T, iv),
            )
        )
    return quotes


@pytest.fixture
def synthetic_quotes() -> Callable[..., list[ExpiryQuotes]]:
    """``synthetic_quotes(params, maturities)``: model priced quotes (synthetic, tests only)."""
    return _synthetic_quotes

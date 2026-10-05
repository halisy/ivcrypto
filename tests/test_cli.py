from __future__ import annotations

import argparse

import pytest

from ivcrypto import cli
from ivcrypto.data.store import find_snapshots, load_snapshot


def test_fetch_command_writes_a_snapshot(fixture_caller, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "make_client", lambda args: fixture_caller())
    assert cli.main(["fetch", "--currency", "btc", "--out", str(tmp_path), "--tickers"]) == 0
    [path] = find_snapshots(tmp_path, "BTC")
    snapshot = load_snapshot(path)
    assert len(snapshot["option_tickers"]) == len(snapshot["option_book"])
    assert str(path) in capsys.readouterr().out


def test_rate_flag_sets_the_default_limit():
    args = argparse.Namespace(rate=4.0, testnet=False)
    client = cli.make_client(args)
    assert client.base_url == "https://www.deribit.com/api/v2"
    assert client._buckets["default"].limit.rate == 4.0
    assert client._buckets["default"].limit.burst == 8
    assert client._buckets["public/get_instruments"].limit.rate == 0.5


def test_testnet_flag_switches_host():
    client = cli.make_client(argparse.Namespace(rate=10.0, testnet=True))
    assert client.base_url == "https://test.deribit.com/api/v2"


def test_fetch_requires_valid_rate():
    with pytest.raises(ValueError, match="invalid rate limit"):
        cli.make_client(argparse.Namespace(rate=0.0, testnet=False))

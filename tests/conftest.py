"""Shared test configuration.

Every test runs offline: an autouse fixture makes any socket connection fail
loudly, so a test can never silently depend on the live Deribit API. Tests
that are meant to hit the network carry ``@pytest.mark.network`` and are
deselected by default (see ``addopts`` in pyproject.toml).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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

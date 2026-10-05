from __future__ import annotations

import socket
import urllib.request

import pytest

import ivcrypto
from ivcrypto.cli import main


def test_version_is_exposed():
    assert isinstance(ivcrypto.__version__, str)
    assert ivcrypto.__version__ != ""


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: ivcrypto" in capsys.readouterr().out


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert ivcrypto.__version__ in capsys.readouterr().out


def test_network_is_blocked_in_tests():
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("www.deribit.com", 443), timeout=1)
    with pytest.raises(RuntimeError, match="offline"):
        urllib.request.urlopen("https://www.deribit.com/api/v2/public/get_time", timeout=1)

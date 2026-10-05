"""Implied volatility surfaces for Deribit crypto options."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ivcrypto")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

__all__ = ["__version__"]

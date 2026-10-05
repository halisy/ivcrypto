"""Typed configuration: every threshold and modeling switch in one place.

Defaults live in the dataclasses below, and ``config/default.toml`` documents every key
with its default (a test keeps the two in sync). A TOML file can override any subset of
keys. Unknown sections or keys are rejected, so a typo cannot silently fall back to a
default.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

FORWARD_METHODS = ("parity", "underlying")


@dataclass(frozen=True)
class CleaningConfig:
    """Turning raw quotes into the OTM quotes used for fitting (see ``cleaning.py``)."""

    min_days_to_expiry: float = 2.0
    """Expiries closer than this many calendar days are dropped."""
    max_relative_spread: float = 0.5
    """Quotes with (ask - bid) / mid above this are dropped."""
    forward_method: str = "parity"
    """``parity``: forward implied by put call parity on the option quotes, falling back to
    Deribit's underlying price when too few clean pairs exist. ``underlying``: Deribit's
    ``underlying_price`` (the expiry's future)."""
    parity_pairs: int = 6
    """Number of call/put pairs (the most precise ones) combined into the parity forward."""
    parity_min_pairs: int = 2
    """Fewer usable pairs than this triggers the fallback."""
    parity_max_dispersion_bps: float = 25.0
    """Fallback when the pairs disagree by more than this (median absolute deviation of
    the pairs used, in basis points of the forward)."""

    def __post_init__(self) -> None:
        if self.forward_method not in FORWARD_METHODS:
            raise ValueError(f"forward_method must be one of {FORWARD_METHODS}")
        _require(self.min_days_to_expiry >= 0, "min_days_to_expiry must be >= 0")
        _require(self.max_relative_spread > 0, "max_relative_spread must be > 0")
        _require(self.parity_pairs >= 1, "parity_pairs must be >= 1")
        _require(self.parity_min_pairs >= 1, "parity_min_pairs must be >= 1")
        _require(self.parity_max_dispersion_bps > 0, "parity_max_dispersion_bps must be > 0")


@dataclass(frozen=True)
class Config:
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)


SECTIONS: Mapping[str, type] = {"cleaning": CleaningConfig}


def load_config(path: Path | str | None = None) -> Config:
    """Defaults, overridden by the TOML file at ``path`` when one is given."""
    if path is None:
        return Config()
    with Path(path).open("rb") as handle:
        return config_from_dict(tomllib.load(handle))


def config_from_dict(data: Mapping[str, Any]) -> Config:
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown config sections {unknown}; known: {sorted(SECTIONS)}")
    config = Config()
    for name, values in data.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"config section [{name}] must be a table")
        section = getattr(config, name)
        config = replace(config, **{name: _override(name, section, values)})
    return config


def config_to_dict(config: Config) -> dict[str, Any]:
    """Plain dict of the effective configuration (saved next to every build)."""
    return asdict(config)


def _override(name: str, section: Any, values: Mapping[str, Any]) -> Any:
    defaults = {f.name: getattr(section, f.name) for f in fields(section)}
    unknown = sorted(set(values) - set(defaults))
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {unknown}; known: {sorted(defaults)}")
    changes = {key: _coerce(f"{name}.{key}", defaults[key], value) for key, value in values.items()}
    return replace(section, **changes)


def _coerce(key: str, default: Any, value: Any) -> Any:
    if isinstance(default, bool) or isinstance(value, bool):
        if not (isinstance(default, bool) and isinstance(value, bool)):
            raise TypeError(f"{key}: expected {type(default).__name__}, got {value!r}")
        return value
    if isinstance(default, float) and isinstance(value, int):
        return float(value)
    if not isinstance(value, type(default)):
        raise TypeError(f"{key}: expected {type(default).__name__}, got {value!r}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

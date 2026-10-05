"""Deribit option names, expiry times and year fractions.

Names look like ``BTC-27NOV26-90000-C``: currency, expiry date (day without a leading
zero, three letter month, two digit year), integer strike, and ``C`` or ``P``. Every
option expires at 08:00 UTC on its expiry date (docs/deribit_api.md). Time is measured
in ACT/365 calendar years, the convention that reproduces Deribit's own implied
volatilities; crypto trades around the clock, so there are no business days to skip.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

EXPIRY_HOUR_UTC = 8
SECONDS_PER_YEAR = 365.0 * 86_400.0
MS_PER_YEAR = 1000.0 * SECONDS_PER_YEAR
MS_PER_DAY = 86_400_000.0

_MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_MONTHS = {name: number for number, name in enumerate(_MONTH_NAMES, start=1)}
_NAME = re.compile(
    r"^(?P<currency>[A-Z]+)-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})"
    r"-(?P<strike>\d+)-(?P<type>[CP])$"
)


@dataclass(frozen=True)
class OptionName:
    currency: str
    expiry: datetime
    strike: float
    option_type: str

    @property
    def expiry_code(self) -> str:
        """The date part of the name, e.g. ``27NOV26`` (independent of the locale)."""
        month = _MONTH_NAMES[self.expiry.month - 1]
        return f"{self.expiry.day}{month}{self.expiry.year % 100:02d}"


def parse_option_name(name: str) -> OptionName:
    """Parse a coin settled option name; raises ``ValueError`` on anything else."""
    match = _NAME.match(name)
    if match is None:
        raise ValueError(f"not a coin settled Deribit option name: {name!r}")
    month = _MONTHS.get(match["month"])
    if month is None:
        raise ValueError(f"unknown month {match['month']!r} in {name!r}")
    try:
        expiry = datetime(
            2000 + int(match["year"]), month, int(match["day"]), EXPIRY_HOUR_UTC, tzinfo=UTC
        )
    except ValueError as err:
        raise ValueError(f"invalid expiry date in {name!r}: {err}") from None
    option_type = "call" if match["type"] == "C" else "put"
    return OptionName(match["currency"], expiry, float(match["strike"]), option_type)


def parse_option_names(names: Iterable[str]) -> pd.DataFrame:
    """One row per name: currency, expiry (UTC), expiry_code, strike, option_type."""
    parsed = [(name, parse_option_name(name)) for name in names]
    return pd.DataFrame(
        {
            "instrument_name": [name for name, _ in parsed],
            "currency": [p.currency for _, p in parsed],
            "expiry": pd.to_datetime([p.expiry for _, p in parsed], utc=True),
            "expiry_code": [p.expiry_code for _, p in parsed],
            "strike": np.array([p.strike for _, p in parsed], dtype=float),
            "option_type": [p.option_type for _, p in parsed],
        }
    )


def to_epoch_ms(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone aware (UTC)")
    return round(timestamp.timestamp() * 1000)


def year_fraction(
    start_ms: npt.ArrayLike, end_ms: npt.ArrayLike
) -> npt.NDArray[np.float64] | float:
    """ACT/365 year fraction between epoch milliseconds."""
    result = (np.asarray(end_ms, dtype=float) - np.asarray(start_ms, dtype=float)) / MS_PER_YEAR
    return float(result) if result.ndim == 0 else result

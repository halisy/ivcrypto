"""From a raw snapshot to clean option quotes in forward USD terms.

1. Join the book summary with the instrument metadata, and check that every name parses
   to the expiry, strike and type in the metadata.
2. Measure time to expiry (ACT/365) from the valuation time: the moment Deribit computed
   the book summary (latest ``creation_timestamp``), not the time of the response.
3. Normalize quotes: a missing bid is null in the book summary but 0.0 in tickers, and
   both mean "no bid".
4. Estimate one forward per expiry (``forwards.py``) and convert coin prices into
   forward (undiscounted) USD premiums, ``price_usd = price_coin * F``. This is Deribit's
   own convention, Black 76 on F with zero rates (docs/deribit_api.md). Converting with
   the spot index instead would mix the futures basis into every premium.
5. Compute log moneyness ``k = ln(K/F)`` and apply the filters in order. Every quote stays
   in the table; ``removed_by`` names the first filter that removed it (null if kept).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ivcrypto.config import CleaningConfig
from ivcrypto.data.fetch import FUTURE_BOOK, OPTION_BOOK, OPTION_INSTRUMENTS
from ivcrypto.data.store import Snapshot
from ivcrypto.forwards import estimate_forwards
from ivcrypto.instruments import MS_PER_DAY, MS_PER_YEAR, parse_option_names

logger = logging.getLogger(__name__)

CLEAN_COLUMNS = [
    "instrument_name",
    "currency",
    "expiry_code",
    "expiry",
    "underlying_index",
    "strike",
    "option_type",
    "T",
    "days",
    "bid_btc",
    "ask_btc",
    "mid_btc",
    "mark_btc",
    "rel_spread",
    "deribit_mark_iv",
    "underlying_price",
    "forward",
    "k",
    "otm",
    "bid_usd",
    "ask_usd",
    "mid_usd",
    "mark_usd",
    "open_interest",
    "tick_size",
    "creation_timestamp",
    "removed_by",
]
"""Columns of ``CleanQuotes.quotes``. Prices ending in ``_btc`` are coin prices as quoted;
``_usd`` are forward USD premiums. ``deribit_mark_iv`` is Deribit's mark IV as a decimal
(the API reports percent)."""

FILTERS: Sequence[tuple[str, str]] = (
    ("expiry_too_close", "expiry closer than min_days_to_expiry"),
    ("no_bid", "no bid (null or zero)"),
    ("no_ask", "no ask (null or zero)"),
    ("crossed", "bid at or above ask"),
    ("wide_spread", "(ask - bid) / mid above max_relative_spread"),
    ("in_the_money", "in the money (only OTM options are fitted)"),
)


@dataclass(frozen=True)
class CleanQuotes:
    """Every option of the snapshot with derived columns, plus the forwards and filter log."""

    currency: str
    valuation_time: datetime
    quotes: pd.DataFrame
    forwards: pd.DataFrame
    filters: pd.DataFrame
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def kept(self) -> pd.DataFrame:
        """The quotes that survived every filter (OTM, two sided, liquid enough)."""
        return self.quotes[self.quotes["removed_by"].isna()]


def clean_snapshot(snapshot: Snapshot, config: CleaningConfig | None = None) -> CleanQuotes:
    config = config if config is not None else CleaningConfig()
    book = snapshot[OPTION_BOOK]
    instruments = snapshot[OPTION_INSTRUMENTS]
    quotes = _join(book, instruments)
    valuation_ms = int(quotes["creation_timestamp"].max())
    quotes["T"] = (quotes["expiration_timestamp"] - valuation_ms) / MS_PER_YEAR
    quotes["days"] = (quotes["expiration_timestamp"] - valuation_ms) / MS_PER_DAY
    _normalize_quotes(quotes)

    forwards = estimate_forwards(quotes, snapshot.tables.get(FUTURE_BOOK), config)
    quotes = quotes.merge(forwards[["expiry_code", "forward"]], on="expiry_code", how="left")
    quotes["k"] = np.log(quotes["strike"] / quotes["forward"])
    quotes["otm"] = np.where(
        quotes["option_type"] == "call",
        quotes["strike"] >= quotes["forward"],
        quotes["strike"] < quotes["forward"],
    )
    for side in ("bid", "ask", "mid", "mark"):
        quotes[f"{side}_usd"] = quotes[f"{side}_btc"] * quotes["forward"]

    quotes["removed_by"] = pd.Series(pd.NA, index=quotes.index, dtype="string")
    masks = {
        "expiry_too_close": quotes["days"] < config.min_days_to_expiry,
        "no_bid": quotes["bid_btc"].isna(),
        "no_ask": quotes["ask_btc"].isna(),
        "crossed": quotes["bid_btc"] >= quotes["ask_btc"],
        "wide_spread": quotes["rel_spread"] > config.max_relative_spread,
        "in_the_money": ~quotes["otm"],
    }
    report = [
        {"step": "options in snapshot", "description": "", "removed": 0, "remaining": len(book)}
    ]
    unmatched = len(book) - len(quotes)
    report.append(
        {
            "step": "unmatched",
            "description": "book rows without instrument metadata",
            "removed": unmatched,
            "remaining": len(quotes),
        }
    )
    for name, description in FILTERS:
        report.append(apply_filter(quotes, name, description, masks[name]))

    valuation_time = datetime.fromtimestamp(valuation_ms / 1000, tz=UTC)
    logger.info(
        "%s quotes valued at %s: %d of %d options kept for fitting",
        snapshot.currency,
        valuation_time.isoformat(),
        int(quotes["removed_by"].isna().sum()),
        len(book),
    )
    return CleanQuotes(
        currency=snapshot.currency,
        valuation_time=valuation_time,
        quotes=quotes[CLEAN_COLUMNS].sort_values(["T", "strike", "option_type"], ignore_index=True),
        forwards=forwards,
        filters=pd.DataFrame(report),
    )


def apply_filter(
    quotes: pd.DataFrame, name: str, description: str, mask: pd.Series
) -> dict[str, object]:
    """Mark quotes in ``mask`` that are still kept as removed by ``name``; log the count."""
    newly = mask.fillna(False).astype(bool) & quotes["removed_by"].isna()
    quotes.loc[newly, "removed_by"] = name
    removed = int(newly.sum())
    remaining = int(quotes["removed_by"].isna().sum())
    logger.info("filter %-16s removed %4d quotes, %4d remain", name, removed, remaining)
    return {"step": name, "description": description, "removed": removed, "remaining": remaining}


def append_filter(clean: CleanQuotes, name: str, description: str, mask: pd.Series) -> CleanQuotes:
    """A later stage (e.g. implied volatility) removing more quotes, with the same log."""
    quotes = clean.quotes.copy()
    row = apply_filter(quotes, name, description, mask)
    filters = pd.concat([clean.filters, pd.DataFrame([row])], ignore_index=True)
    return replace(clean, quotes=quotes, filters=filters)


def _join(book: pd.DataFrame, instruments: pd.DataFrame) -> pd.DataFrame:
    meta = instruments[
        ["instrument_name", "expiration_timestamp", "strike", "option_type", "tick_size"]
    ].rename(columns={"strike": "strike_meta", "option_type": "option_type_meta"})
    quotes = book.merge(meta, on="instrument_name", how="inner")
    parsed = parse_option_names(quotes["instrument_name"])
    quotes = quotes.merge(parsed, on="instrument_name", how="left")
    # Timedelta arithmetic is independent of the datetime resolution pandas picked.
    epoch = pd.Timestamp(0, tz="UTC")
    expiry_ms = (quotes["expiry"] - epoch) // pd.Timedelta(milliseconds=1)
    checks = {
        "expiry": expiry_ms == quotes["expiration_timestamp"],
        "strike": quotes["strike"] == quotes["strike_meta"],
        "option_type": quotes["option_type"] == quotes["option_type_meta"],
    }
    for what, ok in checks.items():
        if not ok.all():
            bad = quotes.loc[~ok, "instrument_name"].tolist()[:3]
            raise ValueError(f"instrument names disagree with the metadata on {what}: {bad}")
    return quotes.drop(columns=["strike_meta", "option_type_meta"])


def _normalize_quotes(quotes: pd.DataFrame) -> None:
    """Coin prices with NaN for missing quotes, Deribit's mark IV as a decimal."""
    quotes["bid_btc"] = quotes["bid_price"].where(quotes["bid_price"] > 0)
    quotes["ask_btc"] = quotes["ask_price"].where(quotes["ask_price"] > 0)
    quotes["mid_btc"] = 0.5 * (quotes["bid_btc"] + quotes["ask_btc"])
    quotes["mark_btc"] = quotes["mark_price"]
    quotes["deribit_mark_iv"] = quotes["mark_iv"] / 100.0
    quotes["rel_spread"] = (quotes["ask_btc"] - quotes["bid_btc"]) / quotes["mid_btc"]

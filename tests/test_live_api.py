"""Live checks against the production API. Deselected by default.

Run with ``uv run pytest -m network`` to confirm that Deribit still sends every
field the pipeline relies on (``fetch_snapshot`` raises if one disappears).
"""

from __future__ import annotations

import pytest

from ivcrypto.data import DeribitClient, fetch_snapshot
from ivcrypto.data.fetch import OPTION_BOOK


@pytest.mark.network
@pytest.mark.parametrize("currency", ["BTC", "ETH"])
def test_live_snapshot_matches_expected_schema(currency):
    raw = fetch_snapshot(DeribitClient(), currency)
    assert len(raw.tables[OPTION_BOOK]) > 100
    assert all(call.testnet is False for call in raw.calls)

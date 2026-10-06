"""Parquet storage for raw snapshots.

Layout: ``<root>/<CURRENCY>/<YYYYMMDDTHHMMSSZ>/`` with one Parquet file per table,
holding the rows exactly as the API returned them (field names unchanged, nulls
kept), and a ``manifest.json`` recording which call produced each table and the
server timestamps. Every Parquet file also carries the snapshot timestamp and
its provenance in the schema metadata, so a file copied out of its directory
still says when and how it was taken.

Snapshots are written to a temporary directory first and renamed into place,
so an interrupted fetch never leaves a half written snapshot behind.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ivcrypto import __version__
from ivcrypto.data.fetch import CallRecord, RawSnapshot, server_time

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DIRNAME_FORMAT = "%Y%m%dT%H%M%SZ"
MANIFEST = "manifest.json"
METADATA_PREFIX = "ivcrypto."


@dataclass(frozen=True)
class Snapshot:
    """A stored snapshot loaded back as pandas DataFrames, one per raw table."""

    path: Path
    currency: str
    timestamp: datetime
    manifest: Mapping[str, Any]
    tables: Mapping[str, pd.DataFrame]

    def __getitem__(self, table: str) -> pd.DataFrame:
        try:
            return self.tables[table]
        except KeyError:
            raise KeyError(
                f"snapshot {self.path} has no table {table!r}; tables: {sorted(self.tables)}"
            ) from None


def rows_to_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """One column per key, in order of first appearance; absent keys become nulls.

    Types are inferred by Arrow across all rows (ints mixed with floats become
    doubles, nested objects become structs). A column whose values Arrow cannot
    unify is stored as the exact JSON text of each value instead.
    """
    keys = list(dict.fromkeys(key for row in rows for key in row))
    columns: dict[str, pa.Array] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        try:
            columns[key] = pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            logger.warning("column %r has mixed types; storing its values as JSON text", key)
            columns[key] = pa.array(
                [None if value is None else json.dumps(value) for value in values],
                type=pa.string(),
            )
    return pa.table(columns)


def snapshot_dirname(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).strftime(DIRNAME_FORMAT)


def write_snapshot(raw: RawSnapshot, root: Path | str) -> Path:
    """Write ``raw`` under ``root`` and return the snapshot directory."""
    if sorted(raw.tables) != sorted(call.table for call in raw.calls):
        raise ValueError("every table needs exactly one call record")
    target = Path(root) / raw.currency / snapshot_dirname(raw.timestamp)
    if target.exists():
        raise FileExistsError(f"snapshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # A plain mkdir (not tempfile.mkdtemp, which forces mode 0700) respects the umask.
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
    staging.mkdir()
    try:
        files: dict[str, str] = {}
        for call in raw.calls:
            table = rows_to_table(raw.tables[call.table])
            metadata = {
                "snapshot_utc": _iso(raw.timestamp),
                "currency": raw.currency,
                "table": call.table,
                "method": call.method,
                "params": json.dumps(dict(call.params), sort_keys=True),
                "source": raw.source,
            }
            existing = table.schema.metadata or {}
            table = table.replace_schema_metadata(
                {
                    **existing,
                    **{f"{METADATA_PREFIX}{k}".encode(): v.encode() for k, v in metadata.items()},
                }
            )
            files[call.table] = f"{call.table}.parquet"
            pq.write_table(table, staging / files[call.table], compression="zstd")
        manifest = _manifest(raw, files)
        (staging / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("wrote snapshot %s", target)
    return target


def load_snapshot(path: Path | str) -> Snapshot:
    """Load a snapshot directory written by :func:`write_snapshot`."""
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported snapshot schema version {version}")
    tables = {
        name: pq.read_table(path / info["file"]).to_pandas()
        for name, info in manifest["tables"].items()
    }
    return Snapshot(
        path=path,
        currency=manifest["currency"],
        timestamp=datetime.fromisoformat(manifest["snapshot_utc"]),
        manifest=manifest,
        tables=tables,
    )


def find_snapshots(root: Path | str, currency: str | None = None) -> list[Path]:
    """Snapshot directories under ``root`` (optionally one currency), oldest first."""
    pattern = f"{currency.upper() if currency else '*'}/*/{MANIFEST}"
    found = [manifest.parent for manifest in Path(root).glob(pattern)]
    return sorted(found, key=lambda p: (p.name, p.parent.name))


def _manifest(raw: RawSnapshot, files: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ivcrypto_version": __version__,
        "currency": raw.currency,
        "snapshot_utc": _iso(raw.timestamp),
        "snapshot_reference": "server send time (usOut) of the option_book call",
        "source": raw.source,
        "tables": {call.table: _call_entry(call, files[call.table]) for call in raw.calls},
        "notes": raw.notes,
    }


def _call_entry(call: CallRecord, file: str) -> dict[str, Any]:
    return {
        "file": file,
        "method": call.method,
        "params": dict(call.params),
        "rows": call.rows,
        "server_us_in": call.us_in,
        "server_us_out": call.us_out,
        "server_utc_in": _iso(server_time(call.us_in)) if call.us_in is not None else None,
        "server_utc_out": _iso(server_time(call.us_out)) if call.us_out is not None else None,
        "testnet": call.testnet,
    }


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

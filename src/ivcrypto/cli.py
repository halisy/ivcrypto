"""Command line interface: ``ivcrypto <command> [options]``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from ivcrypto import __version__
from ivcrypto.data.client import (
    DEFAULT_RATE_LIMITS,
    PRODUCTION_URL,
    TESTNET_URL,
    DeribitClient,
    RateLimit,
)
from ivcrypto.data.fetch import fetch_snapshot
from ivcrypto.data.store import write_snapshot

logger = logging.getLogger("ivcrypto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ivcrypto",
        description="Implied volatility surfaces for Deribit crypto options.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log at DEBUG level instead of INFO"
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    _add_fetch(commands)
    _add_build(commands)
    _add_report(commands)
    return parser


def _add_fetch(commands: argparse._SubParsersAction) -> None:
    fetch = commands.add_parser(
        "fetch",
        help="download one raw option chain snapshot from Deribit",
        description="Download one raw snapshot (option chain, futures, index) and store it as "
        "Parquet under OUT/CURRENCY/YYYYMMDDTHHMMSSZ/.",
    )
    fetch.add_argument(
        "--currency",
        default="BTC",
        help="currency with coin settled options on Deribit, BTC or ETH (default: BTC)",
    )
    fetch.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="root directory for snapshots (default: data/raw)",
    )
    fetch.add_argument(
        "--tickers",
        action="store_true",
        help="also fetch every option's ticker (Deribit's bid and ask IVs, greeks, quote "
        "sizes); one request per option, so this takes a few minutes",
    )
    fetch.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE_LIMITS["default"].rate,
        help="maximum sustained requests per second (default: %(default)s, half of "
        "Deribit's documented limit)",
    )
    fetch.add_argument(
        "--testnet", action="store_true", help="use test.deribit.com instead of production"
    )
    fetch.set_defaults(handler=_cmd_fetch)


def _add_build(commands: argparse._SubParsersAction) -> None:
    build = commands.add_parser(
        "build",
        help="run the full analysis of one snapshot and write the results",
        description="Clean the quotes, compute IVs, validate them against Deribit, fit SVI and "
        "SSVI, check static arbitrage, calibrate Heston and Bates and compare the models. "
        "Results go to OUT/CURRENCY/SNAPSHOT_ID/.",
    )
    build.add_argument("snapshot", type=Path, help="snapshot directory, e.g. data/sample/BTC/...")
    build.add_argument(
        "--config", type=Path, help="TOML file overriding config/default.toml settings"
    )
    build.add_argument(
        "--out", type=Path, default=Path("results"), help="results root (default: results)"
    )
    build.add_argument(
        "--no-heston-per-expiry",
        action="store_true",
        help="skip the per expiry Heston refits (a diagnostic)",
    )
    build.add_argument(
        "--no-bates-variants",
        action="store_true",
        help="skip the Bates refits under other jump bounds (the slowest diagnostic)",
    )
    build.set_defaults(handler=_cmd_build)


def _add_report(commands: argparse._SubParsersAction) -> None:
    report = commands.add_parser(
        "report",
        help="draw the figures and write a Markdown report from build results",
        description="Read a results directory written by 'build' and write report.md plus "
        "figures (light and dark) to OUT, by default reports/CURRENCY/SNAPSHOT_ID/.",
    )
    report.add_argument("results", type=Path, help="results directory written by 'build'")
    report.add_argument("--out", type=Path, help="report directory (default: see above)")
    report.add_argument(
        "--themes",
        nargs="+",
        choices=["light", "dark"],
        default=["light", "dark"],
        help="figure themes to draw (default: both)",
    )
    report.set_defaults(handler=_cmd_report)


def make_client(args: argparse.Namespace) -> DeribitClient:
    limits = dict(DEFAULT_RATE_LIMITS)
    limits["default"] = RateLimit(rate=args.rate, burst=max(1, round(2 * args.rate)))
    return DeribitClient(TESTNET_URL if args.testnet else PRODUCTION_URL, rate_limits=limits)


def _cmd_fetch(args: argparse.Namespace) -> int:
    raw = fetch_snapshot(make_client(args), args.currency, tickers=args.tickers)
    path = write_snapshot(raw, args.out)
    counts = ", ".join(f"{call.table}: {call.rows}" for call in raw.calls)
    print(f"{path}\n  taken {raw.timestamp.isoformat()}\n  rows  {counts}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from ivcrypto.config import load_config
    from ivcrypto.pipeline import build

    path = build(
        args.snapshot,
        load_config(args.config),
        args.out,
        per_expiry_heston=not args.no_heston_per_expiry,
        bates_variants=not args.no_bates_variants,
    )
    print(path)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from ivcrypto.pipeline import load_results
    from ivcrypto.report import write_report

    results = load_results(args.results)
    out = args.out or Path("reports") / args.results.parent.name / args.results.name
    print(write_report(results, out, args.themes))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command is None:
        parser.print_help()
        return 0
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

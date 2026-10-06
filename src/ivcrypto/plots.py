"""Figures for the report, drawn from a results directory (``pipeline.load_results``).

Styling comes from a few tokens per theme: one chart surface, ink for text, a hairline
solid grid, and three series colors (blue, orange, aqua, the first three slots of a
reference categorical palette, validated for color vision deficiency in both themes). The
market is drawn in ink and models in color, so a model never looks like data. Every figure
is drawn in a light and a dark theme; no chart uses two y axes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from ivcrypto.bates.charfunc import BatesParams
from ivcrypto.compare import power_law
from ivcrypto.heston.charfunc import HestonParams
from ivcrypto.heston.pricer import price as model_price
from ivcrypto.implied_vol import implied_vols
from ivcrypto.pipeline import Results
from ivcrypto.svi.raw import SVIParams
from ivcrypto.validation import K_BINS, K_LABELS

DPI = 200


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    series: tuple[str, str, str]
    diverging: tuple[str, str, str]
    sequential: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    diverging=("#2a78d6", "#f0efec", "#e34948"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    diverging=("#3987e5", "#383835", "#e66767"),
    # On a dark surface small values recede toward the surface, so the ramp runs dark to light.
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)
THEMES = {"light": LIGHT, "dark": DARK}

SVI_LABEL = "SVI, fitted per expiry"
HESTON_LABEL = "Heston, one parameter set"
BATES_LABEL = "Bates, one parameter set"
MARKET_LABEL = "Market: bid to ask, and mid"


@contextmanager
def themed(theme: Theme) -> Iterator[None]:
    rc = {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.baseline,
        "axes.linewidth": 0.8,
        "axes.labelcolor": theme.ink_secondary,
        "axes.titlecolor": theme.ink,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": theme.grid,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "text.color": theme.ink,
        "xtick.color": theme.baseline,
        "ytick.color": theme.baseline,
        "xtick.labelcolor": theme.ink_secondary,
        "ytick.labelcolor": theme.ink_secondary,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": theme.ink_secondary,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
    }
    with plt.rc_context(rc):
        yield


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out


def _model_smile(params: HestonParams | BatesParams, k: np.ndarray, T: float) -> np.ndarray:
    strikes = np.exp(k)
    calls = k >= 0
    return implied_vols(model_price(1.0, strikes, T, params, calls), 1.0, strikes, T, calls)


def _svi_params(row: pd.Series) -> SVIParams:
    return SVIParams(a=row["a"], b=row["b"], rho=row["rho"], m=row["m"], sigma=row["sigma"])


def _heston_params(values: dict | pd.Series) -> HestonParams:
    return HestonParams(**{f.name: float(values[f.name]) for f in fields(HestonParams)})


def _bates_params(values: dict | pd.Series) -> BatesParams:
    return BatesParams(**{f.name: float(values[f.name]) for f in fields(BatesParams)})


def smiles(results: Results, theme: Theme, out: Path) -> Path:
    """One panel per fitted expiry: the market band and mid, SVI, Heston and Bates."""
    quotes = results["quotes"]
    kept = quotes[quotes["removed_by"].isna()]
    svi = results["svi"].set_index("expiry_code")
    heston = _heston_params(results.manifest["heston"])
    bates = _bates_params(results.manifest["bates"])
    codes = list(svi.index)
    cols = 4
    rows = int(np.ceil(len(codes) / cols))
    blue, orange, aqua = theme.series
    with themed(theme):
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2.35 * rows + 0.5), squeeze=False)
        for ax, code in zip(axes.flat, codes, strict=False):
            q = kept[kept["expiry_code"] == code].sort_values("k")
            T = float(svi.loc[code, "T"])
            span = q["k"].max() - q["k"].min()
            grid = np.linspace(q["k"].min() - 0.05 * span, q["k"].max() + 0.05 * span, 200)
            ax.vlines(q["k"], 100 * q["iv_bid"], 100 * q["iv_ask"], color=theme.muted, lw=1.1)
            ax.plot(
                q["k"],
                100 * q["iv_mid"],
                "o",
                ms=3.2,
                color=theme.ink_secondary,
                mec=theme.surface,
                mew=0.6,
            )
            ax.plot(grid, 100 * _svi_params(svi.loc[code]).implied_vol(grid, T), color=blue)
            ax.plot(grid, 100 * _model_smile(heston, grid, T), color=orange)
            ax.plot(grid, 100 * _model_smile(bates, grid, T), color=aqua)
            ax.set_title(f"{code}, {T * 365:.1f} days", loc="left")
        for ax in axes.flat[len(codes) :]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("implied vol (%)")
        for col in range(cols):
            visible = [ax for ax in axes[:, col] if ax.get_visible()]
            if visible:
                visible[-1].set_xlabel("log moneyness k = ln(K/F)")
        handles = [
            plt.Line2D(
                [],
                [],
                color=theme.muted,
                lw=1.1,
                marker="o",
                ms=3.2,
                mfc=theme.ink_secondary,
                mec=theme.surface,
                label=MARKET_LABEL,
            ),
            plt.Line2D([], [], color=blue, label=SVI_LABEL),
            plt.Line2D([], [], color=orange, label=HESTON_LABEL),
            plt.Line2D([], [], color=aqua, label=BATES_LABEL),
        ]
        fig.legend(
            handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0)
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, out)


def atm_term_structure(results: Results, theme: Theme, out: Path) -> Path:
    """ATM vol, and smile curvature over one ATM standard deviation on log axes: market
    against Heston and Bates (``compare.atm_term_structure``)."""
    ts = results["atm_term_structure"]
    _, orange, aqua = theme.series
    lines = [
        ("market", "Market (SVI fit)", theme.ink_secondary, 1.0),
        ("heston", HESTON_LABEL, orange, 1.4),
        ("bates", BATES_LABEL, aqua, 1.4),
    ]
    with themed(theme):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, column, title in (
            (left, "atm_vol", "ATM implied volatility (%)"),
            (right, "atm_curvature", "Smile curvature over one ATM std dev (log scale)"),
        ):
            scale = 100.0 if column == "atm_vol" else 1.0
            for name, label, color, width in lines:
                if f"{name}_{column}" not in ts:
                    continue
                ax.plot(
                    ts["days"],
                    scale * ts[f"{name}_{column}"],
                    "-o",
                    color=color,
                    lw=width,
                    ms=4,
                    mec=theme.surface,
                    mew=0.8,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xlabel("days to expiry (log scale)")
            ax.set_title(title, loc="left")
            ticks = [d for d in (2, 5, 10, 30, 90, 365) if ts["days"].min() * 0.8 <= d]
            ax.set_xticks(ticks, [str(d) for d in ticks])
        right.set_yscale("log")
        c, alpha = power_law(ts["T"], ts["market_atm_curvature"])
        fit_days = np.geomspace(ts["days"].min(), ts["days"].max(), 50)
        right.plot(fit_days, c * (fit_days / 365.0) ** -alpha, color=theme.muted, lw=0.9)
        right.annotate(
            f"market fit: T^-{alpha:.2f}",
            xy=(fit_days[-12], c * (fit_days[-12] / 365.0) ** -alpha),
            xytext=(-8, -16),
            textcoords="offset points",
            ha="right",
            color=theme.ink_secondary,
            fontsize=8,
        )
        left.legend(loc="lower right")
        fig.tight_layout()
        return _save(fig, out)


def surface_grid(results: Results, x: np.ndarray, n_days: int = 60) -> tuple[np.ndarray, ...]:
    """SVI implied vol on a grid of ATM standard deviations x = k / sqrt(w(0)) and maturity.

    Each expiry is evaluated in units of its own ATM total volatility, which keeps every
    slice inside its quoted region; between expiries the vol is interpolated linearly in
    log maturity, for display only.
    """
    svi = results["svi"].sort_values("T")
    log_days = np.log10(svi["days"].to_numpy())
    rows = []
    for _, row in svi.iterrows():
        params = _svi_params(row)
        scale = np.sqrt(float(params.total_variance(0.0)))
        rows.append(params.implied_vol(x * scale, float(row["T"])))
    slices = np.array(rows)
    grid_log_days = np.linspace(log_days.min(), log_days.max(), n_days)
    vols = np.array([np.interp(grid_log_days, log_days, slices[:, j]) for j in range(x.size)]).T
    return grid_log_days, vols


def surface_3d(results: Results, theme: Theme, out: Path) -> Path:
    x = np.linspace(-2.0, 2.0, 41)
    log_days, vols = surface_grid(results, x)
    X, Y = np.meshgrid(x, log_days)
    cmap = LinearSegmentedColormap.from_list("ramp", theme.sequential)
    with themed(theme):
        fig = plt.figure(figsize=(8, 5.6))
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(
            X,
            Y,
            100 * vols,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
            alpha=0.95,
        )
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(theme.surface)
            pane.set_edgecolor(theme.grid)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = theme.grid  # matplotlib has no public setter
            axis._axinfo["grid"]["linewidth"] = 0.5
        ticks = [d for d in (2, 5, 10, 30, 90, 365) if 10 ** log_days[0] * 0.8 <= d]
        ax.set_yticks(np.log10(ticks), [str(t) for t in ticks])
        ax.set_xlabel("ATM standard deviations k / sqrt(w(0))", labelpad=6)
        ax.set_ylabel("days to expiry", labelpad=6)
        ax.set_zlabel("implied vol (%)", labelpad=6)
        ax.view_init(elev=24, azim=-58)
        ax.set_title("SVI implied volatility surface", loc="left")
        fig.tight_layout()
        return _save(fig, out)


def surface_html(results: Results, theme: Theme, out: Path) -> Path | None:
    """Interactive 3D surface with plotly, when the optional dependency is installed."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    x = np.linspace(-2.0, 2.0, 41)
    log_days, vols = surface_grid(results, x)
    colorscale = [[i / (len(theme.sequential) - 1), c] for i, c in enumerate(theme.sequential)]
    fig = go.Figure(
        go.Surface(
            x=x,
            y=10**log_days,
            z=100 * vols,
            colorscale=colorscale,
            colorbar={"title": "IV (%)"},
            hovertemplate="%{x:.2f} ATM std devs<br>%{y:.1f} days<br>IV %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="SVI implied volatility surface",
        paper_bgcolor=theme.surface,
        font={"color": theme.ink, "family": "system-ui, -apple-system, Segoe UI, sans-serif"},
        scene={
            "xaxis_title": "ATM standard deviations",
            "yaxis": {"title": "days to expiry", "type": "log"},
            "zaxis_title": "implied vol (%)",
            "bgcolor": theme.surface,
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", div_id="ivcrypto_surface")  # stable output
    return out


def residual_table(model_ivs: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean model minus mid IV (vol points) by expiry and moneyness bucket."""
    frame = model_ivs.assign(
        error=100.0 * (model_ivs[column] - model_ivs["iv_mid"]),
        bucket=pd.cut(model_ivs["k"], K_BINS, labels=K_LABELS),
    )
    order = frame.groupby("expiry_code")["T"].first().sort_values().index
    table = frame.pivot_table(
        index="expiry_code", columns="bucket", values="error", aggfunc="mean", observed=False
    )
    return table.reindex(index=order, columns=K_LABELS)


BUCKET_TICKS = ["below\n-0.2", "-0.2 to\n-0.05", "-0.05 to\n0.05", "0.05 to\n0.2", "above\n0.2"]


def residual_heatmaps(results: Results, theme: Theme, out: Path) -> Path:
    ivs = results["model_ivs"]
    tables = {
        SVI_LABEL: residual_table(ivs, "iv_svi"),
        HESTON_LABEL: residual_table(ivs, "iv_heston"),
        BATES_LABEL: residual_table(ivs, "iv_bates"),
    }
    limit = max(float(np.nanmax(np.abs(t.to_numpy()))) for t in tables.values())
    cmap = LinearSegmentedColormap.from_list("diverging", theme.diverging).with_extremes(
        bad=theme.surface
    )
    days = ivs.groupby("expiry_code")["days"].first()
    with themed(theme):
        fig, axes = plt.subplots(1, len(tables), figsize=(12, 4.4), sharey=True)
        for ax, (title, table) in zip(axes, tables.items(), strict=True):
            image = ax.imshow(
                np.ma.masked_invalid(table.to_numpy()),
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )
            ax.set_xticks(range(len(K_LABELS)), BUCKET_TICKS)
            ax.set_yticks(range(len(table.index)), [f"{c} ({days[c]:.0f}d)" for c in table.index])
            ax.tick_params(length=0)
            ax.set_xlabel("log moneyness k")
            ax.set_title(title, loc="left")
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for (i, j), value in np.ndenumerate(table.to_numpy()):
                if not np.isfinite(value):
                    ax.text(
                        j, i, "no quotes", ha="center", va="center", fontsize=6.5, color=theme.muted
                    )
                elif abs(value) >= 1.0:  # label only the large misses
                    rgb = np.array(cmap((value + limit) / (2 * limit))[:3])
                    light_fill = rgb @ np.array([0.299, 0.587, 0.114]) > 0.55
                    ax.text(
                        j,
                        i,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#0b0b0b" if light_fill else "#ffffff",
                    )
        bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
        bar.set_label("mean model minus mid IV (vol points)")
        bar.outline.set_visible(False)
        return _save(fig, out)


def iv_validation(results: Results, theme: Theme, out: Path) -> Path:
    marks = results["validation_marks"]
    marks = marks[marks["resolvable"]]
    mids = results["validation_mid_vs_mark"]
    blue = theme.series[0]
    with themed(theme):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.2))
        for ax, values, title in (
            (left, marks["diff"], "Deribit mark IV reproduced from its own inputs"),
            (right, mids["diff"], "Our mid IV minus Deribit mark IV"),
        ):
            ax.hist(values, bins=60, color=blue, edgecolor=theme.surface, linewidth=0.4)
            ax.axvline(0.0, color=theme.baseline, lw=0.9)
            ax.set_xlabel("difference (vol points)")
            ax.set_ylabel("quotes")
            ax.set_title(title, loc="left")
        fig.tight_layout()
        return _save(fig, out)


FIGURES = {
    "smiles": smiles,
    "atm_term_structure": atm_term_structure,
    "surface": surface_3d,
    "residual_heatmaps": residual_heatmaps,
    "iv_validation": iv_validation,
}


def draw_all(
    results: Results, out_dir: Path, themes: tuple[str, ...] = ("light", "dark")
) -> list[Path]:
    """Every figure in every requested theme, named ``<figure>_<theme>.png``."""
    written = []
    for theme_name in themes:
        theme = THEMES[theme_name]
        for name, draw in FIGURES.items():
            written.append(draw(results, theme, out_dir / f"{name}_{theme.name}.png"))
        html = surface_html(results, theme, out_dir / f"surface_{theme.name}.html")
        if html is not None:
            written.append(html)
    return written

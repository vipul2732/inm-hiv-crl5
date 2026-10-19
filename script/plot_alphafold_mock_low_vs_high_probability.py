#!/usr/bin/env python3
"""
Plot AlphaFold Multimer / AF3 summary metrics for mock edges stratified by
INM posterior edge probability (low vs high), using paired CSV exports in
Alphafold_multimer/.

Overlaid **distributions** (KDE) for ipTM, pTM, and **fraction disordered**
(AF3 summary) by INM edge class.

By default, each arm is stratified by ipTM within the low/high posterior CSVs:
**bottom N** ipTM rows from the negative CSV and **top N** from the positive CSV
(default ``N=10``; use ``--iptm-extremes-n 0`` to use all rows).

Examples:
  python3 script/plot_alphafold_mock_low_vs_high_probability.py
  python3 script/plot_alphafold_mock_low_vs_high_probability.py --out myplot.png
  python3 script/plot_alphafold_mock_low_vs_high_probability.py --conditions wt vif --out mywtvif.png
  # All three: rows are always HIV WT, HIV ΔVIF, mock (order independent of CLI):
  python3 script/plot_alphafold_mock_low_vs_high_probability.py --conditions wt vif mock
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde, mannwhitneyu
except ImportError:
    gaussian_kde = None  # type: ignore
    mannwhitneyu = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]

# Readable defaults for publication-style figures
FIG_W, FIG_H = 16.0, 6.0
AXIS_LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 10
LINE_W = 2.8
VLINE_W = 1.5
SAVE_DPI = 180

DEFAULT_NEGATIVE = ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_mock_negative.csv"
DEFAULT_POSITIVE = ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_mock_positive.csv"
DEFAULT_OUT = ROOT / "Alphafold_multimer" / "alphafold_mock_low_vs_high_probability.png"

# Row / legend order for multi-condition figures (manuscript convention).
CANONICAL_CONDITION_ORDER: tuple[str, ...] = ("wt", "vif", "mock")

CONDITION_SPECS: dict[str, dict[str, object]] = {
    # "positve" matches the existing filename typo in the repo.
    "mock": {
        "neg": ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_mock_negative.csv",
        "pos": ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_mock_positive.csv",
        "display": "mock infection",
    },
    "wt": {
        "neg": ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_wt_negative.csv",
        "pos": ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_wt_positive.csv",
        "display": "HIV WT infection",
    },
    "vif": {
        "neg": ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_vif_negative.csv",
        "pos": ROOT / "Alphafold_multimer" / "af3_summary_metrics_with_probability_vif_positive.csv",
        "display": "HIV ΔVIF infection",
    },
}

DEFAULT_OUT_BY_SINGLE_COND: dict[str, Path] = {
    "mock": DEFAULT_OUT,
    "wt": ROOT / "Alphafold_multimer" / "alphafold_wt_low_vs_high_probability.png",
    "vif": ROOT / "Alphafold_multimer" / "alphafold_vif_low_vs_high_probability.png",
}


def sort_conditions_canonical(conditions: list[str]) -> list[str]:
    """Return conditions in HIV WT → HIV ΔVIF → mock order (subset of CLI list)."""
    want = set(conditions)
    return [c for c in CANONICAL_CONDITION_ORDER if c in want]


def default_out_for_conditions(conditions: list[str]) -> Path:
    # Deterministic ordering regardless of CLI argument order.
    conds_sorted = sort_conditions_canonical(conditions)
    if len(conds_sorted) == 1:
        return DEFAULT_OUT_BY_SINGLE_COND[conds_sorted[0]]
    suffix = "_".join(conds_sorted)
    return ROOT / "Alphafold_multimer" / f"alphafold_low_vs_high_probability_{suffix}.png"


def _subset_by_iptm_extremes(df: pd.DataFrame, *, n: int, largest: bool) -> pd.DataFrame:
    """Keep bottom n ipTM (largest=False) or top n ipTM (largest=True). NaN ipTM dropped."""
    if "iptm" not in df.columns:
        raise SystemExit("CSV missing column 'iptm'")
    work = df.copy()
    work["_iptm"] = pd.to_numeric(work["iptm"], errors="coerce")
    work = work.dropna(subset=["_iptm"])
    if work.empty:
        return work.drop(columns=["_iptm"])
    k = min(int(n), len(work))
    sub = work.nlargest(k, "_iptm") if largest else work.nsmallest(k, "_iptm")
    return sub.drop(columns=["_iptm"])


def load_combined(
    neg_path: Path,
    pos_path: Path,
    *,
    iptm_extremes_n: int | None = None,
) -> pd.DataFrame:
    df_n = pd.read_csv(neg_path)
    df_p = pd.read_csv(pos_path)
    if iptm_extremes_n is not None and iptm_extremes_n > 0:
        df_n = _subset_by_iptm_extremes(df_n, n=iptm_extremes_n, largest=False)
        df_p = _subset_by_iptm_extremes(df_p, n=iptm_extremes_n, largest=True)
    df_n = df_n.assign(
        model_class="Low posterior P(edge)",
        infection_call="Negative (low probability)",
    )
    df_p = df_p.assign(
        model_class="High posterior P(edge)",
        infection_call="Positive (high probability)",
    )
    return pd.concat([df_n, df_p], ignore_index=True)


def kde_density(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Univariate KDE; safe for small n and degenerate samples."""
    if gaussian_kde is None:
        raise RuntimeError("scipy is required for KDE plots")
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 2:
        return np.zeros_like(x_grid)
    if np.all(s == s[0]):
        center = float(s[0])
        sigma = 0.02
        dens = np.exp(-0.5 * ((x_grid - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return dens / dens.max() if dens.max() > 0 else dens
    kde = gaussian_kde(s, bw_method="scott")
    return kde.evaluate(x_grid)


def plot_distribution_panel(
    ax: plt.Axes,
    low: np.ndarray,
    high: np.ndarray,
    metric_name: str,
    x_grid: np.ndarray,
    colors: tuple[str, str],
    *,
    legend_labels: tuple[str, str] | None = None,
    mw_alternative: str | None = "less",
) -> None:
    """Draw one KDE panel; legend and stats stay inside the axes patch."""
    k_low = kde_density(low, x_grid)
    k_high = kde_density(high, x_grid)

    c0, c1 = colors
    ax.fill_between(x_grid, 0, k_low, color=c0, alpha=0.25, linewidth=0, clip_on=True)
    ax.fill_between(x_grid, 0, k_high, color=c1, alpha=0.25, linewidth=0, clip_on=True)
    (line_low,) = ax.plot(x_grid, k_low, color=c0, lw=LINE_W, clip_on=True)
    (line_high,) = ax.plot(x_grid, k_high, color=c1, lw=LINE_W, clip_on=True)

    ymax = float(max(k_low.max(), k_high.max(), 1e-9))
    ax.set_ylim(0.0, ymax * 1.22)

    # Mean markers (within x-range; y clipped to axes)
    ax.axvline(float(np.mean(low)), color=c0, ls="--", lw=VLINE_W, alpha=0.85, clip_on=True)
    ax.axvline(float(np.mean(high)), color=c1, ls="--", lw=VLINE_W, alpha=0.85, clip_on=True)

    ax.set_xlabel(metric_name, fontsize=AXIS_LABEL_FS, fontweight="bold")
    ax.set_ylabel("Density (KDE)", fontsize=AXIS_LABEL_FS, fontweight="bold")
    # Tight x-range from data so Gaussian KDE tails do not suggest mass below/above observed values.
    all_v = np.concatenate(
        [np.asarray(low, dtype=float).ravel(), np.asarray(high, dtype=float).ravel()]
    )
    all_v = all_v[np.isfinite(all_v)]
    lo, hi = float(all_v.min()), float(all_v.max())
    pad = 0.025
    xmin_raw = max(0.0, lo - pad)
    xmax_raw = min(1.0, hi + pad)
    bin_w = 0.1
    xmin = np.floor(xmin_raw / bin_w) * bin_w
    xmax = min(1.0, np.ceil(xmax_raw / bin_w) * bin_w)
    if xmax <= xmin + 1e-9:
        xmin, xmax = 0.0, 1.0
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    m_low, m_high = float(np.mean(low)), float(np.mean(high))
    p_val: float | None = None
    if (
        mannwhitneyu is not None
        and mw_alternative is not None
        and len(low) >= 2
        and len(high) >= 2
    ):
        try:
            _, p_val = mannwhitneyu(low, high, alternative=mw_alternative)
        except ValueError:
            pass

    if legend_labels is not None:
        leg_labels = [
            f"{legend_labels[0]} (mean = {m_low:.3f})",
            f"{legend_labels[1]} (mean = {m_high:.3f})",
        ]
        handles: list = [line_low, line_high]
        if p_val is not None:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="none",
                    color="none",
                    linewidth=0,
                )
            )
            leg_labels.append(f"p = {p_val:.3g}")
        leg = ax.legend(
            handles,
            leg_labels,
            loc="upper right",
            bbox_to_anchor=(0.97, 1.0),
            ncol=1,
            fontsize=LEGEND_FS,
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            framealpha=0.95,
            borderpad=0.35,
            labelspacing=0.45,
            handlelength=1.8,
            handletextpad=0.45,
        )
        # Allow a small upward nudge without clipping the frame
        leg.set_clip_on(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--conditions",
        nargs="+",
        choices=["mock", "wt", "vif"],
        default=None,
        help="Which infection conditions to plot (one row per condition). "
        "If omitted, falls back to --negative-csv/--positive-csv (mock defaults).",
    )
    ap.add_argument("--negative-csv", type=Path, default=None)
    ap.add_argument("--positive-csv", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    ap.add_argument(
        "--iptm-extremes-n",
        type=int,
        default=10,
        help="Per CSV: keep only the bottom N ipTM rows (negative) and top N (positive). "
        "Use 0 to disable and use all rows.",
    )
    args = ap.parse_args()

    c_low, c_high = "#4c72b0", "#dd8452"

    x_grid = np.linspace(0.0, 1.0, 256)

    if args.conditions is None:
        # Backward compatible behavior: use mock defaults or override via csv args.
        neg_path = (args.negative_csv or DEFAULT_NEGATIVE).resolve()
        pos_path = (args.positive_csv or DEFAULT_POSITIVE).resolve()
        conds = ["mock"]
        out = Path(args.out) if args.out is not None else DEFAULT_OUT
        display_names = {"mock": str(CONDITION_SPECS["mock"]["display"])}
    else:
        conds = sort_conditions_canonical(list(args.conditions))
        out = args.out if args.out is not None else default_out_for_conditions(conds)
        display_names = {c: str(CONDITION_SPECS[c]["display"]) for c in conds}

    if len(conds) == 0:
        raise SystemExit("No conditions specified")

    iptm_n = args.iptm_extremes_n if args.iptm_extremes_n > 0 else None

    nrows = len(conds)
    fig, axes = plt.subplots(nrows, 3, figsize=(FIG_W, FIG_H * nrows))
    if nrows == 1:
        axes = np.asarray([axes])  # shape: (1, 3)

    dis_col = "fraction_disordered"
    for row_idx, cond in enumerate(conds):
        if args.conditions is None:
            neg_path = (args.negative_csv or DEFAULT_NEGATIVE).resolve()
            pos_path = (args.positive_csv or DEFAULT_POSITIVE).resolve()
        else:
            neg_path = Path(CONDITION_SPECS[cond]["neg"]).resolve()  # type: ignore[assignment]
            pos_path = Path(CONDITION_SPECS[cond]["pos"]).resolve()  # type: ignore[assignment]

        if not neg_path.is_file():
            raise SystemExit(f"Missing: {neg_path}")
        if not pos_path.is_file():
            raise SystemExit(f"Missing: {pos_path}")

        df = load_combined(neg_path, pos_path, iptm_extremes_n=iptm_n)
        if dis_col not in df.columns:
            raise SystemExit(
                f"CSV missing column {dis_col!r} (AlphaFold disorder fraction) in {cond!r}"
            )

        low_mask = df["model_class"].str.startswith("Low")
        high_mask = df["model_class"].str.startswith("High")
        n_low, n_high = int(low_mask.sum()), int(high_mask.sum())
        print(f"{cond}: rows low={n_low} high={n_high}")

        low_iptm = df.loc[low_mask, "iptm"].astype(float).to_numpy()
        high_iptm = df.loc[high_mask, "iptm"].astype(float).to_numpy()
        low_ptm = df.loc[low_mask, "ptm"].astype(float).to_numpy()
        high_ptm = df.loc[high_mask, "ptm"].astype(float).to_numpy()
        low_dis = df.loc[low_mask, dis_col].astype(float).to_numpy()
        high_dis = df.loc[high_mask, dis_col].astype(float).to_numpy()

        legend_labels = (
            f"{display_names[cond]} low prob. pairs",
            f"{display_names[cond]} high prob. pairs",
        )

        plot_distribution_panel(
            axes[row_idx, 0],
            low_iptm,
            high_iptm,
            "ipTM",
            x_grid,
            (c_low, c_high),
            legend_labels=legend_labels,
        )
        plot_distribution_panel(
            axes[row_idx, 1],
            low_ptm,
            high_ptm,
            "pTM",
            x_grid,
            (c_low, c_high),
            legend_labels=legend_labels,
        )
        plot_distribution_panel(
            axes[row_idx, 2],
            low_dis,
            high_dis,
            "Fraction disordered (AF3)",
            x_grid,
            (c_low, c_high),
            legend_labels=legend_labels,
            mw_alternative="greater",
        )

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Side-by-side **distribution styles** for the same mock low- vs high-probability AF3 data
(ipTM, pTM, fraction disordered): KDE, normalized histogram, ECDF, violin.

Use this to pick which visualization separates the classes most clearly.

Examples:
  python3 script/plot_alphafold_mock_distribution_style_comparison.py
  python3 script/plot_alphafold_mock_distribution_style_comparison.py --out Alphafold_multimer/compare.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde, mannwhitneyu
except ImportError:
    gaussian_kde = None  # type: ignore
    mannwhitneyu = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NEGATIVE = ROOT / "Alphafold_multimer" / "af3_summary_metrics_mock_negative.csv"
DEFAULT_POSITIVE = ROOT / "Alphafold_multimer" / "af3_summary_metrics_mock_positve.csv"
DEFAULT_OUT = (
    ROOT / "Alphafold_multimer" / "alphafold_mock_distribution_styles_comparison.png"
)


def load_combined(neg_path: Path, pos_path: Path) -> pd.DataFrame:
    df_n = pd.read_csv(neg_path)
    df_p = pd.read_csv(pos_path)
    df_n = df_n.assign(model_class="Low")
    df_p = df_p.assign(model_class="High")
    return pd.concat([df_n, df_p], ignore_index=True)


def kde_density(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if gaussian_kde is None:
        raise RuntimeError("scipy is required")
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 2:
        return np.zeros_like(x_grid)
    if np.all(s == s[0]):
        center = float(s[0])
        sigma = 0.02
        dens = np.exp(-0.5 * ((x_grid - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return dens / dens.max() if dens.max() > 0 else dens
    return gaussian_kde(s, bw_method="scott").evaluate(x_grid)


def xlim_from_data(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
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
        return 0.0, 1.0
    return xmin, xmax


def mw_p(low: np.ndarray, high: np.ndarray, alternative: str) -> float | None:
    if mannwhitneyu is None or len(low) < 2 or len(high) < 2:
        return None
    try:
        _, p = mannwhitneyu(low, high, alternative=alternative)
        return float(p)
    except ValueError:
        return None


def annotate_stats(ax, low: np.ndarray, high: np.ndarray, alt: str, fontsize: int = 8) -> None:
    m0, m1 = float(np.mean(low)), float(np.mean(high))
    p = mw_p(low, high, alt)
    lines = [f"mean: {m0:.3f} vs {m1:.3f}"]
    if p is not None:
        lines.append(f"p = {p:.3g}")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.9),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--negative-csv", type=Path, default=DEFAULT_NEGATIVE)
    ap.add_argument("--positive-csv", type=Path, default=DEFAULT_POSITIVE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    neg, pos = args.negative_csv.resolve(), args.positive_csv.resolve()
    if not neg.is_file() or not pos.is_file():
        raise SystemExit(f"Missing CSV: {neg} or {pos}")

    df = load_combined(neg, pos)
    dis_col = "fraction_disordered"
    if dis_col not in df.columns:
        raise SystemExit(f"Missing {dis_col!r}")

    low_m = df["model_class"].eq("Low")
    high_m = df["model_class"].eq("High")

    panels: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        ("ipTM", df.loc[low_m, "iptm"].to_numpy(float), df.loc[high_m, "iptm"].to_numpy(float), "less"),
        ("pTM", df.loc[low_m, "ptm"].to_numpy(float), df.loc[high_m, "ptm"].to_numpy(float), "less"),
        (
            "Fraction disordered",
            df.loc[low_m, dis_col].to_numpy(float),
            df.loc[high_m, dis_col].to_numpy(float),
            "greater",
        ),
    ]

    c0, c1 = "#4c72b0", "#dd8452"
    styles = ["KDE", "Histogram (density)", "ECDF", "Violin"]
    nrows, ncols = len(styles), len(panels)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.15 * nrows), squeeze=False)
    fig.suptitle(
        "Distribution style comparison — mock low vs high INM edge probability (AF3)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    x_grid_full = np.linspace(0.0, 1.0, 256)

    for j, (name, low, high, alt) in enumerate(panels):
        xmin, xmax = xlim_from_data(low, high)

        # --- Row 0: KDE ---
        ax = axes[0, j]
        xg = x_grid_full[(x_grid_full >= xmin) & (x_grid_full <= xmax)]
        if xg.size < 2:
            xg = np.linspace(xmin, xmax, 128)
        k0 = kde_density(low, xg)
        k1 = kde_density(high, xg)
        ax.fill_between(xg, 0, k0, color=c0, alpha=0.25, linewidth=0)
        ax.fill_between(xg, 0, k1, color=c1, alpha=0.25, linewidth=0)
        ax.plot(xg, k0, color=c0, lw=2.0, label="Low prob.")
        ax.plot(xg, k1, color=c1, lw=2.0, label="High prob.")
        ax.set_ylabel("Density")
        if j == ncols - 1:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name, fontweight="bold", fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.tick_params(axis="x", labelbottom=False)
        annotate_stats(ax, low, high, alt)

        # --- Row 1: Histogram ---
        ax = axes[1, j]
        bins = np.linspace(xmin, xmax, max(12, int((xmax - xmin) / 0.06) + 1))
        ax.hist(
            low,
            bins=bins,
            density=True,
            alpha=0.45,
            color=c0,
            label="Low prob.",
            histtype="stepfilled",
            edgecolor="0.3",
            linewidth=0.6,
        )
        ax.hist(
            high,
            bins=bins,
            density=True,
            alpha=0.45,
            color=c1,
            label="High prob.",
            histtype="stepfilled",
            edgecolor="0.3",
            linewidth=0.6,
        )
        ax.set_ylabel("Density")
        ax.set_xlim(xmin, xmax)
        annotate_stats(ax, low, high, alt)
        ax.tick_params(axis="x", labelbottom=False)

        # --- Row 2: ECDF ---
        ax = axes[2, j]

        def ecdf_plot(a: np.ndarray, color: str, lab: str) -> None:
            v = np.sort(np.asarray(a, float)[np.isfinite(a)])
            if v.size == 0:
                return
            y = np.arange(1, v.size + 1) / v.size
            ax.step(v, y, where="post", color=color, lw=2.0, label=lab)

        ecdf_plot(low, c0, "Low prob.")
        ecdf_plot(high, c1, "High prob.")
        ax.set_ylabel("ECDF")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        annotate_stats(ax, low, high, alt)
        ax.set_xlabel(name, fontsize=10, fontweight="bold")

        # --- Row 3: Violin ---
        ax = axes[3, j]
        lo_f = low[np.isfinite(low)]
        hi_f = high[np.isfinite(high)]
        if lo_f.size >= 2 and hi_f.size >= 2:
            parts = ax.violinplot(
                [lo_f, hi_f],
                positions=[1, 2],
                widths=0.55,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )
            for i, b in enumerate(parts["bodies"]):
                b.set_facecolor([c0, c1][i])
                b.set_alpha(0.55)
                b.set_edgecolor("0.25")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Low prob.", "High prob."], fontsize=9)
        ylab = "Frac. disordered" if name.startswith("Fraction") else name
        ax.set_ylabel(ylab, fontsize=10, fontweight="bold")
        ax.set_xlim(0.4, 2.6)
        y_lo, y_hi = float(np.nanmin(np.r_[lo_f, hi_f])), float(np.nanmax(np.r_[lo_f, hi_f]))
        ax.set_ylim(max(0, y_lo - 0.05), min(1.05, y_hi + 0.08))
        ax.grid(axis="y", alpha=0.25)
        annotate_stats(ax, low, high, alt)

    for i, st in enumerate(styles):
        axes[i, 0].text(
            -0.28,
            0.5,
            st,
            transform=axes[i, 0].transAxes,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )

    for j in range(ncols):
        axes[-1, j].set_xlabel("Low vs high prob. pairs", fontsize=9, fontweight="bold")

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=[0.06, 0, 1, 0.97])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Histogram of main-cluster mean edge frequencies for each base model (mock / wt / vif).

Reads post_sampling_analysis/base_models/*/main_cluster_network/main_cluster_edges.tsv
(column `frequency`: mean binary edge in the τ* largest cluster).

Default: Y = frequency, X = number of pairs per bin (horizontal histogram).
Writes full-range [0, 1] and a zoomed [0.95, 1] figure unless disabled with flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent
REPO = _SCRIPT.parent


def _load_frequency_rows(base_models_root: Path) -> List[Tuple[str, Path, np.ndarray]]:
    rows: List[Tuple[str, Path, np.ndarray]] = []
    for folder, label in [
        ("cullin_mock_config_multi_seeds", "Mock"),
        ("cullin_wt_config_multi_seeds", "WT"),
        ("cullin_vif_config_multi_seeds", "VIF"),
    ]:
        tsv = base_models_root / folder / "main_cluster_network" / "main_cluster_edges.tsv"
        if not tsv.is_file():
            raise FileNotFoundError(tsv)
        df = pd.read_csv(tsv, sep="\t")
        if "frequency" not in df.columns:
            raise ValueError(f"{tsv}: missing 'frequency' column")
        f = df["frequency"].astype(float).clip(0.0, 1.0).to_numpy()
        rows.append((label, tsv, f))
    return rows


def _plot_panels(
    rows: List[Tuple[str, Path, np.ndarray]],
    *,
    freq_min: float,
    freq_max: float,
    n_bins: int,
    out_path: Path,
    dpi: int,
    suptitle: str,
    y_tick_step: float = 0.1,
) -> None:
    n_bins = max(5, int(n_bins))
    bin_edges = np.linspace(freq_min, freq_max, n_bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2), sharey=True)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    yticks = np.arange(freq_min, freq_max + 1e-9, y_tick_step)
    if len(yticks) > 12:
        yticks = np.linspace(freq_min, freq_max, 11)

    for ax, (label, _tsv, f) in zip(axes, rows):
        ax.hist(
            f,
            bins=bin_edges,
            orientation="horizontal",
            color="steelblue",
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
        )
        ax.set_xlabel("Number of pairs", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        med = float(np.median(f))
        n_in_window = int(np.sum((f >= freq_min) & (f <= freq_max)))
        ax.set_title(
            f"{label}\n"
            f"n = {len(f):,} pairs · n in [{freq_min:g}, {freq_max:g}] = {n_in_window:,} · "
            f"median = {med:.4f} · mean = {np.mean(f):.4f}",
            fontsize=10,
        )
        ax.set_ylim(freq_min, freq_max)
        ax.set_yticks(yticks)
        ax.grid(axis="x", alpha=0.25)
        if freq_min <= med <= freq_max:
            ax.axhline(med, color="crimson", ls="--", lw=1.2, alpha=0.85, label="Median (all pairs)")
            ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-models-root",
        type=Path,
        default=REPO / "post_sampling_analysis" / "base_models",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO
        / "post_sampling_analysis"
        / "base_models"
        / "main_cluster_edge_frequency_distribution.png",
    )
    ap.add_argument("--bins", type=int, default=50, help="Bins for full [0,1] plot.")
    ap.add_argument(
        "--zoom-bins",
        type=int,
        default=50,
        help="Bins for zoomed plot (spread across [zoom-min, zoom-max]).",
    )
    ap.add_argument("--zoom-min", type=float, default=0.95, help="Lower frequency for zoomed plot.")
    ap.add_argument("--zoom-max", type=float, default=1.0, help="Upper frequency for zoomed plot.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-full", action="store_true", help="Skip the [0, 1] figure.")
    ap.add_argument("--no-zoom", action="store_true", help="Skip the zoomed figure.")
    args = ap.parse_args()

    rows = _load_frequency_rows(args.base_models_root)

    if not args.no_full:
        _plot_panels(
            rows,
            freq_min=0.0,
            freq_max=1.0,
            n_bins=args.bins,
            out_path=args.out,
            dpi=args.dpi,
            suptitle=(
                "Distribution of main-cluster edge frequencies (all model pairs)\n"
                "Y: mean frequency in tau* largest cluster; X: pair count per bin"
            ),
            y_tick_step=0.1,
        )
        print(f"Saved {args.out.resolve()}")

    if not args.no_zoom:
        lo = float(args.zoom_min)
        hi = float(args.zoom_max)
        if not (0.0 <= lo < hi <= 1.0):
            raise SystemExit("--zoom-min and --zoom-max must satisfy 0 <= zoom-min < zoom-max <= 1")
        zoom_out = args.out.parent / f"{args.out.stem}_zoom_{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}{args.out.suffix}"
        _plot_panels(
            rows,
            freq_min=lo,
            freq_max=hi,
            n_bins=args.zoom_bins,
            out_path=zoom_out,
            dpi=args.dpi,
            suptitle=(
                f"Main-cluster edge frequencies — zoom [{lo:g}, {hi:g}]\n"
                "Pairs outside a bin do not add to that bin; counts reflect the high-frequency tail"
            ),
            y_tick_step=0.01 if (hi - lo) <= 0.08 else (0.02 if (hi - lo) <= 0.15 else 0.05),
        )
        print(f"Saved {zoom_out.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot AUROC summary TSV produced by plot_roc_base_vs_ablations.py.

Input:
  validation_ablations/roc_base_vs_ablations_summary.tsv

Output:
  validation_ablations/roc_base_vs_ablations_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "validation_ablations" / "roc_base_vs_ablations_summary.tsv",
        help="Input summary TSV",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation_ablations" / "roc_base_vs_ablations_summary.png",
        help="Output PNG path",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    if df.empty:
        raise SystemExit(f"No rows in {args.input}")

    # normalize condition order (WT, ΔVIF, mock)
    cond_order = ["wt", "vif", "mock"]
    df["Condition"] = df["Condition"].astype(str).str.lower()
    df = df[df["Condition"].isin(cond_order)].copy()

    # model order (legend / ablation names from plot_roc_base_vs_ablations.py)
    model_order = [
        "Full model",
        "No AP-MS profile similarity score",
        "No SAINT-derived score",
        "No priors",
    ]
    df["Model"] = df["Model"].astype(str)
    # keep only known models, but allow others if present
    keep_models = [m for m in model_order if m in set(df["Model"])]
    other_models = [m for m in df["Model"].unique() if m not in keep_models]
    model_order_final = keep_models + sorted(other_models)

    # pivot to matrix (conds x models)
    piv = df.pivot_table(index="Condition", columns="Model", values="AUROC", aggfunc="mean")
    piv = piv.reindex(cond_order)
    piv = piv.reindex(columns=model_order_final)

    x = np.arange(len(piv.index))
    n_models = len(piv.columns)
    width = min(0.18, 0.8 / max(1, n_models))
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(piv.columns):
        vals = piv[m].to_numpy()
        ax.bar(x + offsets[i], vals, width=width, label=m)
        # annotate
        for xi, yi in zip(x + offsets[i], vals):
            if np.isfinite(yi):
                ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in piv.index])
    ax.set_ylabel("AUROC (vs databases/PDB)")
    ax.set_title("AUROC comparison: Full model vs ablations")
    ax.axhline(0.5, ls="--", lw=1.5, color="gray", alpha=0.7, label="Random (0.5)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(4, n_models), fontsize=9)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()


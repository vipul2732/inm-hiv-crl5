#!/usr/bin/env python3
"""
Plot ROC curves (TPR vs FPR) from already-saved curve TSVs without recomputing anything.

Inputs:
  validation_ablations/roc_base_vs_ablations_curves/*.tsv
  (each TSV must contain FPR, TPR columns; Threshold is optional)

Outputs:
  validation_ablations/roc_base_vs_ablations_<cond>.png

This is intended for the case where the user has manually updated TSV data and
does NOT want it overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "axes.linewidth": 2.0,
        "lines.linewidth": 3.5,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": True,
    })


def bold_axes(ax):
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def humanize_model_slug(slug: str) -> str:
    # Best-effort mapping to nicer labels (file stem is {cond}__{slug}.tsv)
    mapping = {
        "full_model": "Full model",
        "no_corr": "No AP-MS profile similarity score",
        "no_correlation": "No AP-MS profile similarity score",
        "no_saint": "No SAINT-derived score",
        "no_priors": "No priors",
    }
    return mapping.get(slug, slug.replace("_", " "))

def sort_key_model_human(model_h: str) -> int:
    order = [
        "Full model",
        "No AP-MS profile similarity score",
        "No SAINT-derived score",
        "No priors",
    ]
    try:
        return order.index(model_h)
    except ValueError:
        return 999


def load_auc_lookup(summary_tsv: Path) -> Dict[Tuple[str, str], float]:
    """
    Load AUROC values from roc_base_vs_ablations_summary.tsv.
    Returns mapping: (condition, model_human) -> AUROC
    """
    if not summary_tsv.exists():
        return {}
    df = pd.read_csv(summary_tsv, sep="\t")
    if df.empty or not {"Condition", "Model", "AUROC"} <= set(df.columns):
        return {}
    df["Condition"] = df["Condition"].astype(str).str.lower()
    df["Model"] = df["Model"].astype(str)
    out: Dict[Tuple[str, str], float] = {}
    for _, r in df.iterrows():
        try:
            cond = r["Condition"]
            model = r["Model"]
            auc = float(r["AUROC"])
            out[(cond, model)] = auc

            # Aliases for legacy / shortened summary labels
            if model == "No correlation":
                out[(cond, "No AP-MS profile similarity score")] = auc
            if model == "No SAINT":
                out[(cond, "No SAINT-derived score")] = auc
            if model == "No AP-MS profile similarity":
                out[(cond, "No AP-MS profile similarity score")] = auc
        except Exception:
            continue
    return out


def plot_one_condition(cond: str, curves_dir: Path, out_png: Path, auc_lookup: Dict[Tuple[str, str], float]):
    pattern = f"{cond}__*.tsv"
    files = sorted(curves_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No curve TSVs found for condition '{cond}' in {curves_dir}")

    # Load curves
    curves: List[Tuple[str, pd.DataFrame]] = []
    for fp in files:
        model_slug = fp.stem.split("__", 1)[1]
        df = pd.read_csv(fp, sep="\t")
        if not {"FPR", "TPR"} <= set(df.columns):
            raise ValueError(f"{fp} missing FPR/TPR columns")
        curves.append((model_slug, df))

    # Enforce desired order in plotting
    curves.sort(key=lambda t: sort_key_model_human(humanize_model_slug(t[0])))

    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    for model_slug, df in curves:
        fpr = df["FPR"].to_numpy(dtype=float)
        tpr = df["TPR"].to_numpy(dtype=float)
        # Ensure sorted by FPR for AUC calculation
        order = np.argsort(fpr)
        fpr_s = fpr[order]
        tpr_s = tpr[order]
        model_h = humanize_model_slug(model_slug)
        # Prefer AUROC from summary TSV (so it matches your edited values)
        auc = auc_lookup.get((cond, model_h))
        if auc is None:
            auc = float(np.trapz(tpr_s, fpr_s))
        label = f"{model_h} (AUROC={auc:.2f})"
        ax.plot(fpr_s, tpr_s, lw=2.5, label=label)

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, label="Random (0.5)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    title_map = {
        "mock": "Mock infection",
        "wt": "HIV WT infection",
        "vif": "HIV ΔVIF infection",
    }
    ax.set_title(title_map.get(cond, cond.upper()))

    ax.legend(
        loc="lower right",
        fontsize=13,
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=2.2,
        handletextpad=0.6,
    )

    bold_axes(ax)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--curves-dir",
        type=Path,
        default=ROOT / "validation_ablations" / "roc_base_vs_ablations_curves",
        help="Directory containing saved ROC curve TSVs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "validation_ablations",
        help="Output directory for ROC PNGs",
    )
    ap.add_argument(
        "--summary-tsv",
        type=Path,
        default=ROOT / "validation_ablations" / "roc_base_vs_ablations_summary.tsv",
        help="Summary TSV containing AUROC values to annotate on plots",
    )
    ap.add_argument(
        "--conditions",
        type=str,
        default="mock,wt,vif",
        help="Comma-separated conditions to plot",
    )
    args = ap.parse_args()

    set_pub_style()

    conds = [c.strip().lower() for c in args.conditions.split(",") if c.strip()]
    auc_lookup = load_auc_lookup(args.summary_tsv)
    for cond in conds:
        out_png = args.out_dir / f"roc_base_vs_ablations_{cond}.png"
        plot_one_condition(cond, args.curves_dir, out_png, auc_lookup)


if __name__ == "__main__":
    main()


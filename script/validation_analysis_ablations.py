#!/usr/bin/env python3
"""
Generate 4-panel PU validation figures for ablation models
(same format as validation_analysis.py for base models).

Combines all three ablations (no_corr, no_saint, no_priors) into a single figure
for each condition (mock, wt, vif).

Panels:
1) Enrichment (top vs bottom quantiles)
2) Cumulative recovery vs rank
3) Evidence rate in top-N% predictions
4) Precision@K (PU; lower bound)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Dataset name mapping for display
DATASET_DISPLAY_NAMES = {
    'WT': 'HIV WT type infection',
    'MOCK': 'mock infection',
    'VIF': 'HIV Δ VIF infection'
}


def normalize_protein_name(name: str) -> str:
    """Normalize protein name for comparison."""
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def load_ablation_data(ablation_pairs_file: Path, base_comparison_file: Path) -> pd.DataFrame:
    """
    Load ablation pairs and match with base model database comparison to get KnownPositive labels.
    
    Args:
        ablation_pairs_file: TSV with Protein1, Protein2, Probability from ablation
        base_comparison_file: TSV with base model pairs and In_Any_Database column
    
    Returns:
        DataFrame with Probability and KnownPositive columns (same format as base model)
    """
    # Load ablation pairs
    df_ablation = pd.read_csv(ablation_pairs_file, sep="\t")
    df_ablation["Probability"] = pd.to_numeric(df_ablation["Probability"], errors="coerce")
    df_ablation = df_ablation.dropna(subset=["Probability"]).copy()
    df_ablation["Probability"] = df_ablation["Probability"].clip(0, 1)
    
    # Normalize protein names
    df_ablation["Protein1_norm"] = df_ablation["Protein1"].apply(normalize_protein_name)
    df_ablation["Protein2_norm"] = df_ablation["Protein2"].apply(normalize_protein_name)
    
    # Create sorted pair key for matching
    df_ablation["PairKey"] = df_ablation.apply(
        lambda row: tuple(sorted([row["Protein1_norm"], row["Protein2_norm"]])), axis=1
    )
    
    # Load base comparison
    df_base = pd.read_csv(base_comparison_file, sep="\t")
    df_base["Protein1_norm"] = df_base["Protein1"].apply(normalize_protein_name)
    df_base["Protein2_norm"] = df_base["Protein2"].apply(normalize_protein_name)
    df_base["PairKey"] = df_base.apply(
        lambda row: tuple(sorted([row["Protein1_norm"], row["Protein2_norm"]])), axis=1
    )
    
    # Create lookup dict for KnownPositive
    base_lookup = dict(zip(df_base["PairKey"], df_base["In_Any_Database"]))
    
    # Match and add KnownPositive
    df_ablation["KnownPositive"] = df_ablation["PairKey"].map(
        lambda key: int(base_lookup.get(key, False)) if key in base_lookup else 0
    )
    
    # Keep only columns needed (same as base model format)
    df_result = df_ablation[["Probability", "KnownPositive"]].copy()
    
    logger.info(f"Loaded {len(df_result)} pairs, {df_result['KnownPositive'].sum()} known positives")
    
    return df_result


def load_base_model_data(base_comparison_file: Path) -> pd.DataFrame:
    """
    Load base model data from database_comparison.tsv file.
    
    Returns:
        DataFrame with Probability and KnownPositive columns
    """
    df = pd.read_csv(base_comparison_file, sep="\t")
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)
    
    # PU definition (same as validation_analysis.py)
    df["KnownPositive"] = (
        df["In_Any_Database"].fillna(False).astype(bool).astype(int)
    )
    
    # Keep only columns needed
    df_result = df[["Probability", "KnownPositive"]].copy()
    
    logger.info(f"Loaded base model: {len(df_result)} pairs, {df_result['KnownPositive'].sum()} known positives")
    
    return df_result


# ----------------------------
# Global publication style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.6,
        "lines.linewidth": 3.0,
        "lines.markersize": 8,
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


# ----------------------------
# Panel 1 — Enrichment
# ----------------------------
def plot_enrichment(ax, datasets):
    ax.set_title("Enrichment of known interactions")
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("Enrichment ratio")

    quantiles = [5, 10, 20, 30, 50]
    
    for name, df in datasets.items():
        d = df.sort_values("Probability", ascending=False)
        N = len(d)

        xs, ys = [], []
        for q in quantiles:
            n = max(1, int(N * q / 100))
            top = d.iloc[:n]
            bot = d.iloc[-n:]

            top_r = top["KnownPositive"].mean()
            bot_r = bot["KnownPositive"].mean()

            # Calculate enrichment: if bottom has no positives, use a more reasonable upper bound
            # Instead of top_r * N (which can be huge), use top_r / (1/N) = top_r * N, but cap it
            # Or better: use the overall baseline rate as denominator
            baseline_rate = d["KnownPositive"].mean()
            if bot_r > 0:
                y = top_r / bot_r
            else:
                # When bottom has zero positives, enrichment is effectively infinite
                # Use enrichment relative to overall baseline as a more reasonable upper bound
                y = top_r / baseline_rate if baseline_rate > 0 else top_r * N
                # Cap at a reasonable maximum to avoid misleading values
                # If it's still very high, it means the model is doing very well
                if y > 100:
                    y = min(y, 100)  # Cap at 100x for visualization
            xs.append(q)
            ys.append(y)

        line = ax.plot(xs, ys, marker="o", label=name)
        c = line[0].get_color()

        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.1f}×",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=12,
                fontweight="bold",
                color=c,
            )

    ax.axhline(1.0, ls="--", lw=2, color="gray", alpha=0.6)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Panel 2 — Cumulative recovery
# ----------------------------
def plot_cumulative(ax, datasets):
    ax.set_title("Cumulative recovery")
    ax.set_xlabel("Top ranked pairs")
    ax.set_ylabel("Recovered known interactions")

    for name, df in datasets.items():
        d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        N = len(d)
        P = d["KnownPositive"].sum()

        x = np.arange(1, N + 1)
        y = d["KnownPositive"].cumsum()

        line = ax.plot(x, y, label=name)
        c = line[0].get_color()
        ax.plot(x, x * (P / N), ls="--", lw=2, alpha=0.5, color=c)

    ax.legend(loc="lower right")
    bold_axes(ax)


# ----------------------------
# Panel 3 — Top-N% evidence rate
# ----------------------------
def plot_top_quantiles(ax, datasets):
    ax.set_title("Evidence rate in top predictions")
    ax.set_xlabel("Top N% of ranked predictions")
    ax.set_ylabel("Evidence rate (%)")

    quantiles = [5, 10, 20, 30, 50, 100]
    
    for name, df in datasets.items():
        d = df.sort_values("Probability", ascending=False)
        N = len(d)

        xs, ys, counts = [], [], []
        for q in quantiles:
            n = max(1, int(N * q / 100))
            top = d.iloc[:n]
            xs.append(q)
            ys.append(top["KnownPositive"].mean() * 100)
            counts.append(top["KnownPositive"].sum())

        line = ax.plot(xs, ys, marker="s", label=name)
        c = line[0].get_color()

        # Annotate 10% and 100% for each ablation
        for q, y, cnt in zip(xs, ys, counts):
            if q in (10, 100):
                ax.annotate(
                    f"{cnt}/{int(N*q/100)}",
                    (q, y),
                    textcoords="offset points",
                    xytext=(0, 20 if q == 10 else -20),
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                    color=c,
                )

        # Baseline
        overall = d["KnownPositive"].mean() * 100
        ax.axhline(overall, ls=":", lw=2, alpha=0.4, color=c)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Panel 4 — Precision@K
# ----------------------------
def plot_precision(ax, datasets):
    ax.set_title("Precision@K (PU)")
    ax.set_xlabel("Top K predictions")
    ax.set_ylabel("Precision (%)")

    topks = [10, 25, 50, 100, 200, 500, 1000]
    
    for name, df in datasets.items():
        d = df.sort_values("Probability", ascending=False)
        ks, ps, cs = [], [], []

        for k in topks:
            if k > len(d):
                continue
            sub = d.iloc[:k]
            cpos = sub["KnownPositive"].sum()
            ks.append(k)
            ps.append(cpos / k * 100)
            cs.append(cpos)

        line = ax.plot(ks, ps, marker="D", label=name)
        c = line[0].get_color()

        for k, p, cpos in zip(ks, ps, cs):
            if k in (25, 100, max(ks)):
                ax.annotate(
                    f"{cpos}/{k}",
                    (k, p),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                    color=c,
                )

        overall = d["KnownPositive"].mean() * 100
        ax.axhline(overall, ls=":", lw=2, alpha=0.4, color=c)

    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Main
# ----------------------------
def main():
    set_pub_style()

    ap = argparse.ArgumentParser(
        description="Generate 4-panel validation figures for ablation models (all three ablations combined)"
    )
    ap.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["mock", "wt", "vif"],
        help="Condition name (mock, wt, or vif)"
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing ablation results with *_all_pairs.tsv files"
    )
    ap.add_argument(
        "--base-comparison-dir",
        type=Path,
        required=True,
        help="Directory containing base model database_comparison.tsv files"
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG file path"
    )
    args = ap.parse_args()

    # Load all three ablations for this condition
    condition = args.condition
    ablation_names = [
        f"cullin_{condition}_no_corr",
        f"cullin_{condition}_no_saint",
        f"cullin_{condition}_no_priors"
    ]
    
    base_comparison_file = args.base_comparison_dir / f"cullin_{condition}_database_comparison.tsv"
    
    if not base_comparison_file.exists():
        raise SystemExit(f"Base comparison file not found: {base_comparison_file}")
    
    datasets = {}
    
    # Load all three ablations (no base model here)
    for ablation_name in ablation_names:
        ablation_pairs_file = args.results_dir / ablation_name / f"{ablation_name}_all_pairs.tsv"
        if not ablation_pairs_file.exists():
            logger.warning(f"Skipping {ablation_name}: {ablation_pairs_file} not found")
            continue
        
        df = load_ablation_data(ablation_pairs_file, base_comparison_file)
        if len(df) > 0:
            # Use short name for display
            short_name = ablation_name.replace(f"cullin_{condition}_", "").replace("_", " ").title()
            datasets[short_name] = df
    
    if len(datasets) == 0:
        raise SystemExit("No data loaded.")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    plot_enrichment(axes[0, 0], datasets)
    plot_cumulative(axes[0, 1], datasets)
    plot_top_quantiles(axes[1, 0], datasets)
    plot_precision(axes[1, 1], datasets)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

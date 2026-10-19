#!/usr/bin/env python3
"""
Robustness check: Validation analysis with excluded databases.

This script allows you to test how robust the validation metrics are
by removing specific databases from the scoring.

Usage:
    # Exclude BioGRID
    python validation_analysis_robustness.py \
        --input-dir validation_pdb_and_databases \
        --output validation_robustness_no_biogrid.png \
        --exclude BioGRID
    
    # Exclude CORUM
    python validation_analysis_robustness.py \
        --input-dir validation_pdb_and_databases \
        --output validation_robustness_no_corum.png \
        --exclude CORUM
    
    # Exclude PDB
    python validation_analysis_robustness.py \
        --input-dir validation_pdb_and_databases \
        --output validation_robustness_no_pdb.png \
        --exclude PDB
    
    # Exclude multiple databases
    python validation_analysis_robustness.py \
        --input-dir validation_pdb_and_databases \
        --output validation_robustness_no_biogrid_corum.png \
        --exclude BioGRID CORUM
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
# Data loading with database exclusion
# ----------------------------
def load_validation_data(
    input_dir: Path, 
    exclude_databases: Set[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load validation data, excluding specified databases.
    
    Args:
        input_dir: Directory containing database comparison TSV files
        exclude_databases: Set of database names to exclude (e.g., {'BioGRID', 'CORUM', 'PDB'})
    
    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    if exclude_databases is None:
        exclude_databases = set()
    
    # Map database names to column names
    db_column_map = {
        'PDB': 'Has_Shared_PDB',
        'BioGRID': 'In_BioGRID',
        'CORUM': 'In_CORUM',
    }
    
    datasets = {}
    for ds in ["wt", "mock", "vif"]:
        fp = input_dir / f"cullin_{ds}_database_comparison.tsv"
        if not fp.exists():
            logger.warning(f"Missing: {fp}")
            continue

        df = pd.read_csv(fp, sep="\t")
        df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
        df = df.dropna(subset=["Probability"]).copy()
        df["Probability"] = df["Probability"].clip(0, 1)

        # Build KnownPositive by excluding specified databases
        # Start with all False
        df["KnownPositive"] = False
        
        # Add databases that are NOT excluded
        if 'PDB' not in exclude_databases and 'Has_Shared_PDB' in df.columns:
            df["KnownPositive"] = df["KnownPositive"] | df["Has_Shared_PDB"].fillna(False)
        
        if 'BioGRID' not in exclude_databases and 'In_BioGRID' in df.columns:
            df["KnownPositive"] = df["KnownPositive"] | df["In_BioGRID"].fillna(False)
        
        if 'CORUM' not in exclude_databases and 'In_CORUM' in df.columns:
            df["KnownPositive"] = df["KnownPositive"] | df["In_CORUM"].fillna(False)
        
        # Convert to int for calculations
        df["KnownPositive"] = df["KnownPositive"].astype(int)
        
        # Create exclusion label for title
        exclude_label = f" (excluding {', '.join(sorted(exclude_databases))})" if exclude_databases else ""
        
        datasets[ds.upper()] = df
        n_pos = df['KnownPositive'].sum()
        logger.info(f"{ds.upper()}{exclude_label}: N={len(df)} | Pos={n_pos} ({100*n_pos/len(df):.1f}%)")

    return datasets, exclude_label


# ----------------------------
# Panel 1 — Enrichment
# ----------------------------
def plot_enrichment(ax, datasets, exclude_label=""):
    title = "Enrichment of known interactions"
    if exclude_label:
        title += exclude_label
    ax.set_title(title)
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

            y = top_r / bot_r if bot_r > 0 else top_r * N
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
                fontsize=16,
                fontweight="bold",
                color=c,
            )

    ax.axhline(1.0, ls="--", lw=2, color="gray", alpha=0.6)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Panel 2 — Cumulative recovery
# ----------------------------
def plot_cumulative(ax, datasets, exclude_label=""):
    title = "Cumulative recovery"
    if exclude_label:
        title += exclude_label
    ax.set_title(title)
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
def plot_top_quantiles(ax, datasets, exclude_label=""):
    title = "Evidence rate in top predictions"
    if exclude_label:
        title += exclude_label
    ax.set_title(title)
    ax.set_xlabel("Top N% of ranked predictions")
    ax.set_ylabel("Evidence rate (%)")

    quantiles = [5, 10, 20, 30, 50, 100]
    offsets_y = {"WT": 25, "MOCK": 0, "VIF": -25}
    offsets_x = {"WT": -8, "MOCK": 0, "VIF": 8}

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

        for q, y, cnt in zip(xs, ys, counts):
            if q in (10, 100):
                offset_y = offsets_y[name] if q == 10 else -offsets_y[name]
                offset_x = offsets_x[name]
                ax.annotate(
                    f"{cnt}/{int(N*q/100)}",
                    (q, y),
                    textcoords="offset points",
                    xytext=(offset_x, offset_y),
                    ha="center",
                    fontsize=16,
                    fontweight="bold",
                    color=c,
                )

        overall = d["KnownPositive"].mean() * 100
        ax.axhline(overall, ls=":", lw=2, alpha=0.4, color=c)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Panel 4 — Precision@K
# ----------------------------
def plot_precision(ax, datasets, exclude_label=""):
    title = "Precision@K (PU)"
    if exclude_label:
        title += exclude_label
    ax.set_title(title)
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
                    fontsize=16,
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
        description="Validation analysis with database exclusion for robustness testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Exclude BioGRID
  python validation_analysis_robustness.py \\
    --input-dir validation_pdb_and_databases \\
    --output validation_no_biogrid.png \\
    --exclude BioGRID

  # Exclude CORUM
  python validation_analysis_robustness.py \\
    --input-dir validation_pdb_and_databases \\
    --output validation_no_corum.png \\
    --exclude CORUM

  # Exclude PDB
  python validation_analysis_robustness.py \\
    --input-dir validation_pdb_and_databases \\
    --output validation_no_pdb.png \\
    --exclude PDB

  # Exclude multiple databases
  python validation_analysis_robustness.py \\
    --input-dir validation_pdb_and_databases \\
    --output validation_no_biogrid_corum.png \\
    --exclude BioGRID CORUM
        """
    )
    
    ap.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing database comparison TSV files")
    ap.add_argument("--output", type=Path, required=True,
                   help="Output figure path")
    ap.add_argument("--exclude", nargs="+", default=[],
                   choices=['PDB', 'BioGRID', 'CORUM'],
                   help="Databases to exclude from validation (can specify multiple)")
    
    args = ap.parse_args()
    
    exclude_databases = set(args.exclude)
    if exclude_databases:
        logger.info(f"Excluding databases: {', '.join(sorted(exclude_databases))}")
    else:
        logger.info("No databases excluded (using all databases)")

    datasets, exclude_label = load_validation_data(args.input_dir, exclude_databases)
    if not datasets:
        raise SystemExit("No datasets loaded.")

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    plot_enrichment(axes[0, 0], datasets, exclude_label)
    plot_cumulative(axes[0, 1], datasets, exclude_label)
    plot_top_quantiles(axes[1, 0], datasets, exclude_label)
    plot_precision(axes[1, 1], datasets, exclude_label)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved: {args.output}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for name, df in datasets.items():
        n_total = len(df)
        n_pos = df['KnownPositive'].sum()
        pct = 100 * n_pos / n_total if n_total > 0 else 0
        print(f"{name}: {n_pos}/{n_total} positives ({pct:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate comprehensive statistics file from validation analysis.
Useful for writing papers and reporting results.

Usage:
  python script/generate_validation_stats.py \
    --input-dir validation_pdb_and_databases \
    --output validation_pdb_and_databases/validation_statistics.txt

Cluster-1 edge frequency (same ranking as validation_analysis 4-panel edge-freq figure):
  python script/generate_validation_stats.py \
    --input-dir validation_pdb_and_databases \
    --output validation_pdb_and_databases/validation_statistics_cluster1_edgefreq.txt \
    --results-dir results \
    --edge-frequency-mock updated_analysis/.../mean_edges_cluster_1.npy \
    --edge-frequency-wt updated_analysis/.../mean_edges_cluster_1.npy \
    --edge-frequency-vif updated_analysis/.../mean_edges_cluster_1.npy
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validation_analysis import load_validation_data, quantile_enrichment_ratio  # noqa: E402


def compute_statistics(
    datasets: Dict[str, pd.DataFrame],
    score_col: str = "Probability",
    ranking_score_label: Optional[str] = None,
    known_positive_label: Optional[str] = None,
) -> str:
    """Compute and format all statistics."""
    if ranking_score_label is not None:
        score_label = ranking_score_label
    else:
        score_label = (
            "edge frequency (cluster-1, fraction of binary samples with edge on)"
            if score_col == "Edge_frequency"
            else "probability (posterior mean)"
        )
    if known_positive_label is None:
        known_positive_label = "In_Any_Database (BioGRID, CORUM, or PDB)"
    lines = []
    lines.append("=" * 80)
    lines.append("VALIDATION STATISTICS - PU Analysis")
    lines.append(f"KnownPositive = {known_positive_label}")
    lines.append(f"Ranking score: {score_label}")
    lines.append("Enrichment: top vs bottom quantile with Haldane pseudocount (k+0.5)/(n+1) per side")
    lines.append("=" * 80)
    lines.append("")

    for name, df in datasets.items():
        lines.append(f"\n{'='*80}")
        lines.append(f"DATASET: {name}")
        lines.append(f"{'='*80}\n")

        d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        N = len(d)
        P = int(d["KnownPositive"].sum())
        overall_rate = d["KnownPositive"].mean() * 100

        # Overall statistics
        lines.append("OVERALL STATISTICS:")
        lines.append(f"  Total pairs: {N:,}")
        lines.append(f"  Known positives: {P:,} ({overall_rate:.2f}%)")
        lines.append(f"  Mean score: {d[score_col].mean():.4f}")
        lines.append(f"  Median score: {d[score_col].median():.4f}")
        lines.append(f"  Std score: {d[score_col].std():.4f}")
        lines.append("")

        # Top N% quantiles
        lines.append("VALIDATION RATE BY TOP N% QUANTILES:")
        quantiles = [5, 10, 20, 30, 50, 100]
        for q in quantiles:
            n = max(1, int(N * q / 100))
            top_n = d.iloc[:n]
            val_rate = top_n["KnownPositive"].mean() * 100
            val_count = int(top_n["KnownPositive"].sum())
            lines.append(f"  Top {q:3d}% ({n:5,} pairs): {val_count:4d}/{n:5,} = {val_rate:5.2f}%")
        lines.append("")

        # Enrichment ratios (top vs bottom)
        lines.append("ENRICHMENT RATIOS (Top N% vs Bottom N%):")
        enrichment_quantiles = [5, 10, 20, 30, 50]
        for q in enrichment_quantiles:
            n = max(1, int(N * q / 100))
            top_n = d.iloc[:n]
            bottom_n = d.iloc[-n:]
            top_rate = top_n["KnownPositive"].mean()
            bottom_rate = bottom_n["KnownPositive"].mean()
            enrichment = quantile_enrichment_ratio(top_n, bottom_n)
            lines.append(f"  {q:2d}% quantile: {enrichment:.2f}x (top: {top_rate*100:.2f}%, bottom: {bottom_rate*100:.2f}%)")
        lines.append("")

        # Precision@K
        lines.append("PRECISION@K (Lower bound in PU setting):")
        topks = [10, 25, 50, 100, 200, 300, 500, 1000]
        for k in topks:
            if k > N:
                continue
            sub = d.iloc[:k]
            cpos = int(sub["KnownPositive"].sum())
            prec = (cpos / k) * 100.0
            lines.append(f"  K={k:4d}: {cpos:3d}/{k:4d} = {prec:5.2f}%")
        lines.append("")

        # Cumulative recovery at key ranks
        lines.append("CUMULATIVE RECOVERY AT KEY RANKS:")
        recovery_ranks = [10, 25, 50, 100, 200, 300, 500, 1000, 2000]
        cumsum = d["KnownPositive"].cumsum()
        for rank in recovery_ranks:
            if rank > N:
                continue
            recovered = int(cumsum.iloc[rank - 1])
            pct_recovered = (recovered / P) * 100.0 if P > 0 else 0.0
            lines.append(f"  Rank {rank:4d}: {recovered:3d}/{P:3d} known positives ({pct_recovered:5.2f}% of total)")
        lines.append("")

        # Score distribution (same bins for probability or edge frequency in [0,1])
        lines.append("SCORE DISTRIBUTION ([0,1] bins):")
        prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(prob_bins) - 1):
            low, high = prob_bins[i], prob_bins[i + 1]
            mask = (d[score_col] >= low) & (d[score_col] < high)
            if i == len(prob_bins) - 2:  # Last bin includes upper bound
                mask = (d[score_col] >= low) & (d[score_col] <= high)
            n_bin = mask.sum()
            if n_bin > 0:
                val_rate_bin = d.loc[mask, "KnownPositive"].mean() * 100
                lines.append(f"  [{low:.1f}-{high:.1f}]: {n_bin:5,} pairs, {val_rate_bin:5.2f}% validated")
        lines.append("")

    # Cross-dataset comparison
    lines.append(f"\n{'='*80}")
    lines.append("CROSS-DATASET COMPARISON")
    lines.append(f"{'='*80}\n")

    comparison_data = []
    for name, df in datasets.items():
        d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        comparison_data.append({
            "Dataset": name,
            "Total_Pairs": len(d),
            "Known_Positives": int(d["KnownPositive"].sum()),
            "Overall_Rate_%": d["KnownPositive"].mean() * 100,
            "Top10%_Rate_%": d.iloc[:max(1, int(len(d)*0.1))]["KnownPositive"].mean() * 100,
            "Top50%_Rate_%": d.iloc[:max(1, int(len(d)*0.5))]["KnownPositive"].mean() * 100,
            "Enrichment_5%": compute_enrichment(d, 5),
            "Precision@100": (d.iloc[:100]["KnownPositive"].sum() / 100) * 100,
        })

    comp_df = pd.DataFrame(comparison_data)
    lines.append(comp_df.to_string(index=False))
    lines.append("")

    return "\n".join(lines)


def compute_enrichment(df, quantile):
    """Compute enrichment ratio for a given quantile."""
    N = len(df)
    n = max(1, int(N * quantile / 100))
    top_n = df.iloc[:n]
    bottom_n = df.iloc[-n:]
    return quantile_enrichment_ratio(top_n, bottom_n)


def main():
    ap = argparse.ArgumentParser(description="Generate validation statistics file")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory with validation TSV files")
    ap.add_argument("--output", type=Path, required=True, help="Output statistics file")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--edge-frequency-mock", type=Path, default=None)
    ap.add_argument("--edge-frequency-wt", type=Path, default=None)
    ap.add_argument("--edge-frequency-vif", type=Path, default=None)
    ap.add_argument(
        "--ranking-score-label",
        type=str,
        default=None,
        help='Override printed "Ranking score:" line when using edge-frequency npys.',
    )
    ap.add_argument(
        "--comparison-suffix",
        type=str,
        default=None,
        help="Load cullin_{ds}_database_comparison_<suffix>.tsv files.",
    )
    ap.add_argument(
        "--known-positive-column",
        type=str,
        default="In_Any_Database",
        help="TSV column used as KnownPositive label.",
    )
    ap.add_argument(
        "--known-positive-label",
        type=str,
        default=None,
        help="Human-readable KnownPositive definition for the stats header.",
    )
    args = ap.parse_args()

    ef = [args.edge_frequency_mock, args.edge_frequency_wt, args.edge_frequency_vif]
    if any(ef) and not all(ef):
        raise SystemExit("Provide all three --edge-frequency-{mock,wt,vif} or none.")
    edge_npys = None
    if all(ef):
        edge_npys = {
            "mock": args.edge_frequency_mock.resolve(),
            "wt": args.edge_frequency_wt.resolve(),
            "vif": args.edge_frequency_vif.resolve(),
        }
        if args.results_dir is None:
            raise SystemExit("--results-dir required with edge-frequency npy files.")

    datasets, score_col = load_validation_data(
        args.input_dir,
        results_dir=args.results_dir,
        edge_frequency_npys=edge_npys,
        comparison_suffix=args.comparison_suffix,
        known_positive_column=args.known_positive_column,
    )
    if not datasets:
        raise SystemExit("No datasets loaded.")

    stats_text = compute_statistics(
        datasets,
        score_col=score_col,
        ranking_score_label=args.ranking_score_label,
        known_positive_label=args.known_positive_label,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(stats_text)

    print(f"Statistics saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

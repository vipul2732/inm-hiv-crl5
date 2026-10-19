#!/usr/bin/env python3
"""
Generate a comprehensive validation statistics text file for ablation runs,
mirroring the base model's `validation_pdb_and_databases/validation_statistics.txt`.

Key idea:
  - Load ablation predicted pairs + probabilities from:
      script/results/cullin_<cond>_<ablation>/cullin_<cond>_<ablation>_all_pairs.tsv
  - Define KnownPositive using the *base* comparison TSV (same positive set):
      validation_pdb_and_databases/cullin_<cond>_database_comparison.tsv

Usage:
  python script/generate_ablation_validation_stats.py \
    --results-dir script/results \
    --base-comparison-dir validation_pdb_and_databases \
    --output validation_ablations/validation_statistics_ablations.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def normalize_protein_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def compute_enrichment(df: pd.DataFrame, quantile: int) -> float:
    """Compute enrichment ratio for a given quantile (top vs bottom)."""
    N = len(df)
    n = max(1, int(N * quantile / 100))
    top_n = df.iloc[:n]
    bottom_n = df.iloc[-n:]
    top_rate = top_n["KnownPositive"].mean()
    bottom_rate = bottom_n["KnownPositive"].mean()
    if bottom_rate > 0:
        return float(top_rate / bottom_rate)
    return float(top_rate * N)


def compute_statistics(datasets: Dict[str, pd.DataFrame]) -> str:
    """Compute and format all statistics (same structure as base stats file)."""
    lines = []
    lines.append("=" * 80)
    lines.append("VALIDATION STATISTICS - PU Analysis (Ablations)")
    lines.append("KnownPositive = In_Any_Database (BioGRID, CORUM, or PDB) from base comparison TSVs")
    lines.append("=" * 80)
    lines.append("")

    for name, df in datasets.items():
        lines.append(f"\n{'='*80}")
        lines.append(f"DATASET: {name}")
        lines.append(f"{'='*80}\n")

        d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        N = len(d)
        P = int(d["KnownPositive"].sum())
        overall_rate = d["KnownPositive"].mean() * 100 if N else 0.0

        # Overall statistics
        lines.append("OVERALL STATISTICS:")
        lines.append(f"  Total pairs: {N:,}")
        lines.append(f"  Known positives: {P:,} ({overall_rate:.2f}%)")
        lines.append(f"  Mean probability: {d['Probability'].mean():.4f}" if N else "  Mean probability: nan")
        lines.append(f"  Median probability: {d['Probability'].median():.4f}" if N else "  Median probability: nan")
        lines.append(f"  Std probability: {d['Probability'].std():.4f}" if N else "  Std probability: nan")
        lines.append("")

        # Top N% quantiles
        lines.append("VALIDATION RATE BY TOP N% QUANTILES:")
        quantiles = [5, 10, 20, 30, 50, 100]
        for q in quantiles:
            n = max(1, int(N * q / 100)) if N else 0
            if n == 0:
                continue
            top_n = d.iloc[:n]
            val_rate = top_n["KnownPositive"].mean() * 100
            val_count = int(top_n["KnownPositive"].sum())
            lines.append(f"  Top {q:3d}% ({n:5,} pairs): {val_count:4d}/{n:5,} = {val_rate:5.2f}%")
        lines.append("")

        # Enrichment ratios (top vs bottom)
        lines.append("ENRICHMENT RATIOS (Top N% vs Bottom N%):")
        enrichment_quantiles = [5, 10, 20, 30, 50]
        for q in enrichment_quantiles:
            n = max(1, int(N * q / 100)) if N else 0
            if n == 0:
                continue
            top_n = d.iloc[:n]
            bottom_n = d.iloc[-n:]
            top_rate = top_n["KnownPositive"].mean()
            bottom_rate = bottom_n["KnownPositive"].mean()
            if bottom_rate > 0:
                enrichment = top_rate / bottom_rate
            else:
                enrichment = top_rate * N  # lower bound approximation (same as base)
            lines.append(
                f"  {q:2d}% quantile: {enrichment:.2f}x (top: {top_rate*100:.2f}%, bottom: {bottom_rate*100:.2f}%)"
            )
        lines.append("")

        # Precision@K
        lines.append("PRECISION@K (Lower bound in PU setting):")
        topks = [10, 25, 50, 100, 200, 300, 500, 1000]
        for k in topks:
            if k > N or k <= 0:
                continue
            sub = d.iloc[:k]
            cpos = int(sub["KnownPositive"].sum())
            prec = (cpos / k) * 100.0
            lines.append(f"  K={k:4d}: {cpos:3d}/{k:4d} = {prec:5.2f}%")
        lines.append("")

        # Cumulative recovery at key ranks
        lines.append("CUMULATIVE RECOVERY AT KEY RANKS:")
        recovery_ranks = [10, 25, 50, 100, 200, 300, 500, 1000, 2000]
        cumsum = d["KnownPositive"].cumsum() if N else pd.Series([], dtype=int)
        for rank in recovery_ranks:
            if rank > N or rank <= 0:
                continue
            recovered = int(cumsum.iloc[rank - 1])
            pct_recovered = (recovered / P) * 100.0 if P > 0 else 0.0
            lines.append(f"  Rank {rank:4d}: {recovered:3d}/{P:3d} known positives ({pct_recovered:5.2f}% of total)")
        lines.append("")

        # Probability distribution statistics
        lines.append("PROBABILITY DISTRIBUTION:")
        prob_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(prob_bins) - 1):
            low, high = prob_bins[i], prob_bins[i + 1]
            if N == 0:
                continue
            mask = (d["Probability"] >= low) & (d["Probability"] < high)
            if i == len(prob_bins) - 2:
                mask = (d["Probability"] >= low) & (d["Probability"] <= high)
            n_bin = int(mask.sum())
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
        d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        if len(d) == 0:
            continue
        comparison_data.append({
            "Dataset": name,
            "Total_Pairs": len(d),
            "Known_Positives": int(d["KnownPositive"].sum()),
            "Overall_Rate_%": d["KnownPositive"].mean() * 100,
            "Top10%_Rate_%": d.iloc[:max(1, int(len(d)*0.1))]["KnownPositive"].mean() * 100,
            "Top50%_Rate_%": d.iloc[:max(1, int(len(d)*0.5))]["KnownPositive"].mean() * 100,
            "Enrichment_5%": compute_enrichment(d, 5),
            "Precision@100": (d.iloc[:100]["KnownPositive"].sum() / 100) * 100 if len(d) >= 100 else np.nan,
        })

    comp_df = pd.DataFrame(comparison_data)
    if len(comp_df) > 0:
        lines.append(comp_df.to_string(index=False))
    else:
        lines.append("(no datasets loaded)")
    lines.append("")

    return "\n".join(lines)


def load_one_ablation_dataset(
    ablation_pairs_tsv: Path,
    base_comparison_tsv: Path,
    dataset_label: str,
) -> pd.DataFrame:
    df_ab = pd.read_csv(ablation_pairs_tsv, sep="\t")
    if "Probability" not in df_ab.columns or "Protein1" not in df_ab.columns or "Protein2" not in df_ab.columns:
        raise ValueError(f"{ablation_pairs_tsv} missing required columns")

    df_ab["Probability"] = pd.to_numeric(df_ab["Probability"], errors="coerce")
    df_ab = df_ab.dropna(subset=["Probability"]).copy()
    df_ab["Probability"] = df_ab["Probability"].clip(0, 1)

    df_ab["Protein1_norm"] = df_ab["Protein1"].apply(normalize_protein_name)
    df_ab["Protein2_norm"] = df_ab["Protein2"].apply(normalize_protein_name)
    df_ab["PairKey"] = df_ab.apply(lambda r: tuple(sorted((r["Protein1_norm"], r["Protein2_norm"]))), axis=1)

    df_base = pd.read_csv(base_comparison_tsv, sep="\t")
    if "In_Any_Database" not in df_base.columns or "Protein1" not in df_base.columns or "Protein2" not in df_base.columns:
        raise ValueError(f"{base_comparison_tsv} missing required columns")
    df_base["Protein1_norm"] = df_base["Protein1"].apply(normalize_protein_name)
    df_base["Protein2_norm"] = df_base["Protein2"].apply(normalize_protein_name)
    df_base["PairKey"] = df_base.apply(lambda r: tuple(sorted((r["Protein1_norm"], r["Protein2_norm"]))), axis=1)
    lookup = dict(zip(df_base["PairKey"], df_base["In_Any_Database"]))

    df_ab["KnownPositive"] = df_ab["PairKey"].map(lambda k: int(bool(lookup.get(k, False))))

    out = df_ab[["Probability", "KnownPositive"]].copy()
    out.attrs["label"] = dataset_label
    return out


def load_ablation_datasets(results_dir: Path, base_comparison_dir: Path) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    conditions = ["wt", "mock", "vif"]
    ablations = ["no_corr", "no_saint", "no_priors"]

    for cond in conditions:
        base_tsv = base_comparison_dir / f"cullin_{cond}_database_comparison.tsv"
        if not base_tsv.exists():
            continue
        for ab in ablations:
            ds = f"cullin_{cond}_{ab}"
            ab_tsv = results_dir / ds / f"{ds}_all_pairs.tsv"
            if not ab_tsv.exists():
                continue
            label = f"{cond.upper()}_{ab}"
            datasets[label] = load_one_ablation_dataset(ab_tsv, base_tsv, label)

    return datasets


def main():
    ap = argparse.ArgumentParser(description="Generate ablation validation statistics file")
    ap.add_argument("--results-dir", type=Path, required=True, help="Directory with ablation results (script/results)")
    ap.add_argument("--base-comparison-dir", type=Path, required=True, help="Directory with base comparison TSVs")
    ap.add_argument("--output", type=Path, required=True, help="Output statistics text file")
    args = ap.parse_args()

    datasets = load_ablation_datasets(args.results_dir, args.base_comparison_dir)
    if not datasets:
        raise SystemExit("No ablation datasets loaded (missing *_all_pairs.tsv files?).")

    stats_text = compute_statistics(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stats_text)
    print(f"Statistics saved to: {args.output}")


if __name__ == "__main__":
    main()


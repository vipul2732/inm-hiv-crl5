#!/usr/bin/env python3
"""
Compare ablations to the base model by direct Top-K overlap.

For each condition (mock/wt/vif) and each ablation (no_corr/no_saint/no_priors),
compute:
  - overlap_fraction = |TopK_base ∩ TopK_ablation| / K
This answers: "of the base model's top K pairs, what fraction are also in the
ablation's top K?"

Also report symmetric Jaccard overlap as a secondary measure:
  - jaccard = |A ∩ B| / |A ∪ B|

Inputs:
  - Base all-pairs TSV: results/cullin_<cond>_config_multi_seeds/all_pairs.tsv
  - Ablation all-pairs TSV: script/results/cullin_<cond>_<ablation>/cullin_<cond>_<ablation>_all_pairs.tsv

Outputs:
  - validation_ablations/topk_overlap_with_base_<cond>.png
  - validation_ablations/topk_overlap_with_base_<cond>.tsv

Additionally, we compute a normalized "accuracy vs base" score:
  random_overlap ≈ K / N_pairs
  accuracy = (overlap_fraction - random_overlap) / (1 - random_overlap)
so that random ≈ 0 and perfect match = 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def normalize_protein_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def load_topk_pairs(tsv: Path, k: int) -> Set[Tuple[str, str]]:
    df = pd.read_csv(tsv, sep="\t")
    # accept either naming
    if "Probability" not in df.columns:
        raise ValueError(f"{tsv} missing Probability column")
    p1_col = "Protein1" if "Protein1" in df.columns else None
    p2_col = "Protein2" if "Protein2" in df.columns else None
    if not p1_col or not p2_col:
        raise ValueError(f"{tsv} missing Protein1/Protein2 columns")

    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)

    df["P1"] = df[p1_col].apply(normalize_protein_name)
    df["P2"] = df[p2_col].apply(normalize_protein_name)
    # canonical ordering so (A,B)==(B,A)
    df["A"] = df.apply(lambda r: min(r["P1"], r["P2"]), axis=1)
    df["B"] = df.apply(lambda r: max(r["P1"], r["P2"]), axis=1)

    d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    k = min(k, len(d))
    top = d.iloc[:k]
    return set(zip(top["A"], top["B"]))


def jaccard(a: Set[Tuple[str, str]], b: Set[Tuple[str, str]]) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def plot_condition(condition: str, ks: List[int], out_dir: Path):
    base_tsv = ROOT / "results" / f"cullin_{condition}_config_multi_seeds" / "all_pairs.tsv"
    if not base_tsv.exists():
        raise FileNotFoundError(f"Base all_pairs TSV not found: {base_tsv}")

    ablations = ["no_corr", "no_saint", "no_priors"]
    ab_tsvs = {
        "No correlation": ROOT / "script" / "results" / f"cullin_{condition}_no_corr" / f"cullin_{condition}_no_corr_all_pairs.tsv",
        "No SAINT": ROOT / "script" / "results" / f"cullin_{condition}_no_saint" / f"cullin_{condition}_no_saint_all_pairs.tsv",
        "No priors": ROOT / "script" / "results" / f"cullin_{condition}_no_priors" / f"cullin_{condition}_no_priors_all_pairs.tsv",
    }

    # compute table
    rows = []
    # N pairs for random baseline (same for base and ablations after alignment)
    # Use base TSV length after cleaning
    df_base_all = pd.read_csv(base_tsv, sep="\t")
    df_base_all["Probability"] = pd.to_numeric(df_base_all["Probability"], errors="coerce")
    df_base_all = df_base_all.dropna(subset=["Probability"]).copy()
    N_pairs = int(len(df_base_all))
    for k in ks:
        top_base = load_topk_pairs(base_tsv, k)
        for label, ab_tsv in ab_tsvs.items():
            if not ab_tsv.exists():
                continue
            top_ab = load_topk_pairs(ab_tsv, k)
            inter = len(top_base & top_ab)
            overlap_frac = inter / k if k else np.nan
            random_overlap = (k / N_pairs) if (N_pairs > 0 and k > 0) else np.nan
            # normalized accuracy: random->0, perfect->1
            if np.isfinite(overlap_frac) and np.isfinite(random_overlap) and (1 - random_overlap) > 0:
                acc_norm = (overlap_frac - random_overlap) / (1 - random_overlap)
                # clamp numerical noise
                acc_norm = float(max(0.0, min(1.0, acc_norm)))
            else:
                acc_norm = np.nan
            rows.append({
                "Condition": condition,
                "Ablation": label,
                "K": k,
                "Overlap_count": inter,
                "Overlap_fraction_of_base_topk": overlap_frac,
                "Random_expected_overlap_fraction": random_overlap,
                "Accuracy_vs_base_normalized": acc_norm,
                "Jaccard": jaccard(top_base, top_ab),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_out = out_dir / f"topk_overlap_with_base_{condition}.tsv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(tsv_out, sep="\t", index=False)

    # Plot normalized accuracy vs K (ROC-like: base=1, random≈0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in df_out["Ablation"].unique():
        sub = df_out[df_out["Ablation"] == label].sort_values("K")
        ax.plot(sub["K"], sub["Accuracy_vs_base_normalized"], marker="o", label=label)

    ax.axhline(1.0, color="black", lw=1.5, alpha=0.5, label="Base (1.0)")
    ax.axhline(0.0, color="gray", ls="--", lw=1.5, alpha=0.6, label="Random (0.0)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("K")
    ax.set_ylabel("Accuracy vs base (normalized Top-K match)")
    ax.set_title(f"Top-K accuracy vs base (cullin_{condition})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    png_out = out_dir / f"topk_overlap_with_base_{condition}.png"
    fig.savefig(png_out, dpi=300)
    plt.close(fig)

    print(f"Saved {png_out}")
    print(f"Saved {tsv_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="10,25,50,100,200,300", help="Comma-separated K values")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation_ablations", help="Output directory")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    for cond in ["mock", "wt", "vif"]:
        plot_condition(cond, ks, args.out_dir)


if __name__ == "__main__":
    main()


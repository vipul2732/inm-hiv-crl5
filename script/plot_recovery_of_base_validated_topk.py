#!/usr/bin/env python3
"""
Base-anchored validation: "Only give credit for database-supported pairs that are
also in the base model's Top-Kref list".

For each condition (mock/wt/vif):
  1) Take base Top-Kref pairs (from results/.../all_pairs.tsv).
  2) Intersect with base KnownPositive labels (from validation_pdb_and_databases/..._database_comparison.tsv).
     This produces the reference set S = {pairs in base Top-Kref that are In_Any_Database}.
  3) For each ablation, compute how many pairs from S are recovered in the ablation Top-K,
     and plot recall = |S ∩ TopK_ablation| / |S|.

This matches the request:
  - Do NOT count "other" database-supported pairs as positives if they are not in base Top-Kref.

Outputs:
  - validation_ablations/recovery_base_top{Kref}_validated_<cond>.png
  - validation_ablations/recovery_base_top{Kref}_validated_<cond>.tsv
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


def pairkey(p1: str, p2: str) -> Tuple[str, str]:
    a = normalize_protein_name(p1)
    b = normalize_protein_name(p2)
    return (a, b) if a <= b else (b, a)


def load_topk_pairs(tsv: Path, k: int) -> List[Tuple[Tuple[str, str], float]]:
    """Return list of (PairKey, prob) for top-k by probability."""
    df = pd.read_csv(tsv, sep="\t")
    if not {"Protein1", "Protein2", "Probability"} <= set(df.columns):
        raise ValueError(f"{tsv} must have Protein1, Protein2, Probability")
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)
    df["PairKey"] = df.apply(lambda r: pairkey(r["Protein1"], r["Protein2"]), axis=1)
    d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    k = min(k, len(d))
    top = d.iloc[:k]
    return list(zip(top["PairKey"].tolist(), top["Probability"].tolist()))


def load_knownpositive_lookup(base_comparison_tsv: Path) -> Dict[Tuple[str, str], int]:
    df = pd.read_csv(base_comparison_tsv, sep="\t")
    if not {"Protein1", "Protein2", "In_Any_Database"} <= set(df.columns):
        raise ValueError(f"{base_comparison_tsv} must have Protein1, Protein2, In_Any_Database")
    df["PairKey"] = df.apply(lambda r: pairkey(r["Protein1"], r["Protein2"]), axis=1)
    return {pk: int(bool(v)) for pk, v in zip(df["PairKey"], df["In_Any_Database"].fillna(False))}


def compute_reference_set(base_all_pairs: Path, base_comparison: Path, kref: int) -> Set[Tuple[str, str]]:
    kp = load_knownpositive_lookup(base_comparison)
    top_base = load_topk_pairs(base_all_pairs, kref)
    s = {pk for pk, _ in top_base if kp.get(pk, 0) == 1}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kref", type=int, default=1000, help="Base Top-K reference cutoff (default 1000)")
    ap.add_argument("--ks", type=str, default="10,25,50,100,200,300,500,1000", help="Comma-separated K values for ablation retrieval")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation_ablations", help="Output directory")
    args = ap.parse_args()

    kref = int(args.kref)
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in ["mock", "wt", "vif"]:
        base_all = ROOT / "results" / f"cullin_{cond}_config_multi_seeds" / "all_pairs.tsv"
        base_comp = ROOT / "validation_pdb_and_databases" / f"cullin_{cond}_database_comparison.tsv"
        if not base_all.exists() or not base_comp.exists():
            continue

        ref_set = compute_reference_set(base_all, base_comp, kref)
        ref_size = len(ref_set)
        if ref_size == 0:
            print(f"[warn] cullin_{cond}: reference set empty for kref={kref}")
            continue

        ab_tsvs = {
            "No correlation": ROOT / "script" / "results" / f"cullin_{cond}_no_corr" / f"cullin_{cond}_no_corr_all_pairs.tsv",
            "No SAINT": ROOT / "script" / "results" / f"cullin_{cond}_no_saint" / f"cullin_{cond}_no_saint_all_pairs.tsv",
            "No priors": ROOT / "script" / "results" / f"cullin_{cond}_no_priors" / f"cullin_{cond}_no_priors_all_pairs.tsv",
        }

        rows = []
        for label, ab_tsv in ab_tsvs.items():
            if not ab_tsv.exists():
                continue
            for k in ks:
                top_ab = load_topk_pairs(ab_tsv, k)
                top_ab_set = {pk for pk, _ in top_ab}
                hit = len(ref_set & top_ab_set)
                rows.append({
                    "Condition": cond,
                    "Ablation": label,
                    "Kref_base": kref,
                    "Ref_validated_pairs": ref_size,
                    "K_ablation": k,
                    "Recovered_ref_pairs": hit,
                    "Recall_of_ref_validated_pairs": hit / ref_size if ref_size else np.nan,
                })

        df_out = pd.DataFrame(rows)
        tsv_out = out_dir / f"recovery_base_top{kref}_validated_{cond}.tsv"
        df_out.to_csv(tsv_out, sep="\t", index=False)

        # plot
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for ab in df_out["Ablation"].unique():
            sub = df_out[df_out["Ablation"] == ab].sort_values("K_ablation")
            ax.plot(sub["K_ablation"], sub["Recall_of_ref_validated_pairs"], marker="o", label=ab)

        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Ablation Top-K")
        ax.set_ylabel(f"Recall of base Top-{kref} validated pairs")
        ax.set_title(f"Recovery of base Top-{kref} validated pairs (cullin_{cond})\n|S|={ref_size} validated pairs")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        png_out = out_dir / f"recovery_base_top{kref}_validated_{cond}.png"
        fig.savefig(png_out, dpi=300)
        plt.close(fig)
        print(f"Saved {png_out}")
        print(f"Saved {tsv_out}")


if __name__ == "__main__":
    main()


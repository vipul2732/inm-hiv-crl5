#!/usr/bin/env python3
"""
Base-anchored ROC on a restricted universe (e.g., base Top-1000 pairs).

Requested behavior:
  - Consider only the base model's Top-Kref pairs as the evaluation universe U.
  - Define positives ONLY within U, using base database labels:
      y=1 if pair is In_Any_Database in base comparison TSV AND pair ∈ U
      y=0 otherwise (within U)
  - Compute ROC curves for:
      * Full model scores on U
      * Each ablation's scores on the SAME U
    Crucially, ablations do NOT get credit for any other database-supported pairs
    outside U (because they are not part of the universe here).

Inputs:
  - Base all-pairs TSV (for selecting top-Kref by probability):
      results/cullin_<cond>_config_multi_seeds/all_pairs.tsv
  - Base comparison TSV (for labels in U):
      validation_pdb_and_databases/cullin_<cond>_database_comparison.tsv
  - Ablation all-pairs TSVs (for scores):
      script/results/cullin_<cond>_<ablation>/cullin_<cond>_<ablation>_all_pairs.tsv

Outputs:
  - validation_ablations/roc_baseanchored_top{Kref}_<cond>.png
  - validation_ablations/roc_baseanchored_top{Kref}_summary.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


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


def load_base_topk_universe(base_all_pairs_tsv: Path, kref: int) -> pd.DataFrame:
    df = pd.read_csv(base_all_pairs_tsv, sep="\t")
    if not {"Protein1", "Protein2", "Probability"} <= set(df.columns):
        raise ValueError(f"{base_all_pairs_tsv} must have Protein1, Protein2, Probability")
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)
    df["PairKey"] = df.apply(lambda r: pairkey(r["Protein1"], r["Protein2"]), axis=1)
    d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    kref = min(int(kref), len(d))
    u = d.iloc[:kref][["PairKey", "Probability"]].copy()
    u = u.rename(columns={"Probability": "Score_Full"})
    return u


def load_base_labels(base_comparison_tsv: Path) -> Dict[Tuple[str, str], int]:
    df = pd.read_csv(base_comparison_tsv, sep="\t")
    if not {"Protein1", "Protein2", "In_Any_Database"} <= set(df.columns):
        raise ValueError(f"{base_comparison_tsv} must have Protein1, Protein2, In_Any_Database")
    df["PairKey"] = df.apply(lambda r: pairkey(r["Protein1"], r["Protein2"]), axis=1)
    return {pk: int(bool(v)) for pk, v in zip(df["PairKey"], df["In_Any_Database"].fillna(False))}


def load_scores(tsv: Path, score_colname: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    if not {"Protein1", "Protein2", "Probability"} <= set(df.columns):
        raise ValueError(f"{tsv} must have Protein1, Protein2, Probability")
    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)
    df["PairKey"] = df.apply(lambda r: pairkey(r["Protein1"], r["Protein2"]), axis=1)
    return df[["PairKey", "Probability"]].rename(columns={"Probability": score_colname})


def compute_roc(y_true: np.ndarray, y_score: np.ndarray):
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kref", type=int, default=1000, help="Base Top-K universe size (default 1000)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation_ablations")
    args = ap.parse_args()

    kref = int(args.kref)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cond in ["mock", "wt", "vif"]:
        base_all = ROOT / "results" / f"cullin_{cond}_config_multi_seeds" / "all_pairs.tsv"
        base_comp = ROOT / "validation_pdb_and_databases" / f"cullin_{cond}_database_comparison.tsv"
        if not base_all.exists() or not base_comp.exists():
            continue

        # Universe U = base Top-Kref by base probability
        u = load_base_topk_universe(base_all, kref)  # PairKey, Score_Full
        # Labels y only for pairs in U, from base comparison
        label_lookup = load_base_labels(base_comp)
        u["KnownPositive"] = u["PairKey"].map(lambda pk: int(label_lookup.get(pk, 0)))

        y = u["KnownPositive"].to_numpy(dtype=int)
        s_full = u["Score_Full"].to_numpy(dtype=float)
        fpr_f, tpr_f, auc_f = compute_roc(y, s_full)

        curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {
            "Full model": (fpr_f, tpr_f, auc_f)
        }

        # Ablation scores aligned to the same U
        ab_map = {
            "No correlation": ROOT / "script" / "results" / f"cullin_{cond}_no_corr" / f"cullin_{cond}_no_corr_all_pairs.tsv",
            "No SAINT": ROOT / "script" / "results" / f"cullin_{cond}_no_saint" / f"cullin_{cond}_no_saint_all_pairs.tsv",
            "No priors": ROOT / "script" / "results" / f"cullin_{cond}_no_priors" / f"cullin_{cond}_no_priors_all_pairs.tsv",
        }

        for label, ab_tsv in ab_map.items():
            if not ab_tsv.exists():
                continue
            df_ab = load_scores(ab_tsv, score_colname="Score")
            m = u[["PairKey", "KnownPositive"]].merge(df_ab, on="PairKey", how="left")
            # If missing, fill with 0.0 (worst) within the universe
            m["Score"] = m["Score"].fillna(0.0)
            fpr, tpr, auc = compute_roc(m["KnownPositive"].to_numpy(int), m["Score"].to_numpy(float))
            curves[label] = (fpr, tpr, auc)

        # Plot ROC
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for label, (fpr, tpr, auc) in curves.items():
            ax.plot(fpr, tpr, lw=2.5, label=f"{label} (AUROC={auc:.3f})")
            summary_rows.append({
                "Condition": cond,
                "Universe": f"base_top{kref}",
                "Model": label,
                "AUROC": auc,
                "N_pairs": int(len(u)),
                "N_known_positives": int(u["KnownPositive"].sum()),
            })

        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.5, label="Random (AUROC=0.5)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC within base Top-{kref} universe (cullin_{cond})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_png = out_dir / f"roc_baseanchored_top{kref}_{cond}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved {out_png}")

    out_tsv = out_dir / f"roc_baseanchored_top{kref}_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved {out_tsv}")


if __name__ == "__main__":
    main()


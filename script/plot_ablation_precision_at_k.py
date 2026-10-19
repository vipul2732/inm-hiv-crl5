#!/usr/bin/env python3
"""
Plot Fold-Enrichment@K (K = 10, 25, 50) for base vs ablation models.

Inputs:
  - Base comparison TSVs: validation_pdb_and_databases/cullin_*_database_comparison.tsv
  - Ablation all-pairs TSVs: script/results/cullin_*_no_*/cullin_*_no_*_all_pairs.tsv

Outputs:
  - validation_ablations/enrichment_comparison_mock.png
  - validation_ablations/enrichment_comparison_wt.png
  - validation_ablations/enrichment_comparison_vif.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_DIR = ROOT / "validation_ablations"


def normalize_protein_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def parse_base_fold_enrichment_at_k():
    """
    Compute Fold-Enrichment@K for K = 100, 200, 300 from *_database_comparison.tsv files
    for WT, MOCK, VIF. This matches the method used in validation_analysis.py for the main figure.
    
    FE@K = (Precision@K / 100) / baseline_rate
    where baseline_rate = known_positives / total_pairs
    
    This uses the same probabilities and method as the base model validation figure.
    """
    import pandas as pd
    
    base_comparison_dir = ROOT / "validation_pdb_and_databases"
    ks = [10, 25, 50]
    base_fe = {"WT": {}, "MOCK": {}, "VIF": {}}
    
    dataset_map = {"wt": "WT", "mock": "MOCK", "vif": "VIF"}
    
    for ds_key, ds_name in dataset_map.items():
        tsv_path = base_comparison_dir / f"cullin_{ds_key}_database_comparison.tsv"
        if not tsv_path.exists():
            continue
        
        # Load TSV file (same as validation_analysis.py)
        df = pd.read_csv(tsv_path, sep="\t")
        df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
        df = df.dropna(subset=["Probability"]).copy()
        df["Probability"] = df["Probability"].clip(0, 1)
        
        # PU definition (same as validation_analysis.py)
        df["KnownPositive"] = (
            df["In_Any_Database"].fillna(False).astype(bool).astype(int)
        )
        
        # Sort by probability (descending) - same as figure
        d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        N = len(d)
        P = d["KnownPositive"].sum()
        baseline_rate = P / N if N > 0 else 0
        
        # Compute Precision@K and FE@K for each K
        for k in ks:
            if k > N:
                continue
            sub = d.iloc[:k]
            cpos = sub["KnownPositive"].sum()
            prec_frac = cpos / k if k > 0 else 0
            fe = prec_frac / baseline_rate if baseline_rate > 0 else 0
            base_fe[ds_name][k] = fe
    
    return base_fe


def load_ablation_fold_enrichment_at_k_from_all_pairs():
    """
    Compute ablation Fold-Enrichment@K (K = 100, 200, 300) from *_all_pairs.tsv files.

    Returns dict: { condition: { model_label: {K: fold_enrichment} } }
    where condition in {"mock", "wt", "vif"} and model_label in
    {"base", "no_corr", "no_saint", "no_priors"} for plotting.
    """
    import pandas as pd

    ks = [10, 25, 50]
    data = {
        "mock": {},
        "wt": {},
        "vif": {},
    }

    dataset_map = {"wt": "wt", "mock": "mock", "vif": "vif"}

    for ds_key, condition in dataset_map.items():
        base_tsv = ROOT / "validation_pdb_and_databases" / f"cullin_{ds_key}_database_comparison.tsv"
        if not base_tsv.exists():
            continue
        df_base = pd.read_csv(base_tsv, sep="\t")
        df_base["Protein1_norm"] = df_base["Protein1"].apply(normalize_protein_name)
        df_base["Protein2_norm"] = df_base["Protein2"].apply(normalize_protein_name)
        df_base["PairKey"] = df_base.apply(
            lambda row: tuple(sorted([row["Protein1_norm"], row["Protein2_norm"]])), axis=1
        )
        base_lookup = dict(zip(df_base["PairKey"], df_base["In_Any_Database"]))

        for label in ["no_corr", "no_saint", "no_priors"]:
            ab_dir = ROOT / "script" / "results" / f"cullin_{ds_key}_{label}"
            ab_tsv = ab_dir / f"cullin_{ds_key}_{label}_all_pairs.tsv"
            if not ab_tsv.exists():
                continue

            df = pd.read_csv(ab_tsv, sep="\t")
            df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
            df = df.dropna(subset=["Probability"]).copy()
            df["Probability"] = df["Probability"].clip(0, 1)
            df["Protein1_norm"] = df["Protein1"].apply(normalize_protein_name)
            df["Protein2_norm"] = df["Protein2"].apply(normalize_protein_name)
            df["PairKey"] = df.apply(
                lambda row: tuple(sorted([row["Protein1_norm"], row["Protein2_norm"]])), axis=1
            )
            df["KnownPositive"] = df["PairKey"].map(lambda k: int(bool(base_lookup.get(k, False))))

            d = df.sort_values("Probability", ascending=False).reset_index(drop=True)
            N = len(d)
            P = d["KnownPositive"].sum()
            baseline_rate = P / N if N > 0 else 0

            data.setdefault(condition, {})
            data[condition].setdefault(label, {})
            for k in ks:
                if k > N:
                    continue
                sub = d.iloc[:k]
                cpos = sub["KnownPositive"].sum()
                prec_frac = cpos / k if k > 0 else 0
                fe = prec_frac / baseline_rate if baseline_rate > 0 else 0
                data[condition][label][k] = fe

    return data


def make_plots():
    base_fe = parse_base_fold_enrichment_at_k()
    ablation_fe = load_ablation_fold_enrichment_at_k_from_all_pairs()

    out_dir = ABLATIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [10, 25, 50]
    conditions = [("MOCK", "mock"), ("WT", "wt"), ("VIF", "vif")]

    for base_name, cond_key in conditions:
        models = ["base", "no_corr", "no_saint", "no_priors"]
        model_labels = {
            "base": "Full model",
            "no_corr": "No correlation",
            "no_saint": "No SAINT",
            "no_priors": "No priors",
        }

        # Prepare data matrix: rows = models, cols = Ks
        values = []
        for m in models:
            row = []
            for k in ks:
                if m == "base":
                    val = base_fe.get(base_name, {}).get(k, float("nan"))
                else:
                    val = ablation_fe.get(cond_key, {}).get(m, {}).get(k, float("nan"))
                row.append(val)
            values.append(row)

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(8, 5))

        x = range(len(ks))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

        for idx, (m, row) in enumerate(zip(models, values)):
            xpos = [xi + offsets[idx] for xi in x]
            ax.bar(xpos, row, width=width, label=model_labels[m])

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_ylabel("Fold-Enrichment@K")
        ax.set_title(f"Fold-Enrichment@K comparison ({base_name})")
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (1.0x)')
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = out_dir / f"enrichment_comparison_{cond_key}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    make_plots()


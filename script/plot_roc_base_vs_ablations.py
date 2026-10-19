#!/usr/bin/env python3
"""
Compute ROC curves / AUROC for base vs ablations using the same KnownPositive set.

KnownPositive definition:
  - From base comparison TSV: In_Any_Database (BioGRID, CORUM, or PDB)

Scores (two modes):
  - Default: Probability from all_pairs TSV (base + script/results ablations)
  - --edge-frequency: main-cluster edge frequency (mean M=1{Z>0} over draws in the
    labeled main cluster; default labels use the full posterior), from mean_edges_main_cluster.npy; base uses
    post_sampling_analysis/base_models/.../main_cluster_network/; ablations use
    script/results/.../main_cluster_network/ (run_ablation_main_cluster_network.sh).
  - --legacy-cluster1-edge-frequency: cluster-1 mean_edges_cluster_1.npy paths
    (updated_analysis + ablation clustering_k3).

We align ablation scores to the base pair list via normalized (Protein1, Protein2).

Outputs (--namesuffix default: "" for Probability; "_edgefreq" for --edge-frequency unless overridden):
  - validation_ablations/roc_base_vs_ablations_<cond><suffix>.png
  - validation_ablations/roc_base_vs_ablations_summary<suffix>.tsv
  - validation_ablations/roc_base_vs_ablations_curves<suffix>/<cond>__<model_slug>.tsv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validation_analysis import (  # noqa: E402
    _attach_edge_frequency,
    _model_data_pkl,
    build_pair_to_flat_index,
)

ROOT = Path(__file__).resolve().parents[1]

# Panel titles for publication (match infection labels used elsewhere).
CONDITION_TITLE = {
    "mock": "mock infection",
    "wt": "HIV WT infection",
    "vif": "HIV ΔVIF infection",
}

FULL_MODEL_RESULTS_SUBDIR = {
    "mock": "cullin_mock_config_multi_seeds",
    "wt": "cullin_wt_config_multi_seeds",
    "vif": "cullin_vif_config_multi_seeds",
}

# (directory slug under cullin_{cond}_{slug}, legend label in figures/summary TSV)
ABLATION_MODELS: tuple[tuple[str, str], ...] = (
    ("no_corr", "No AP-MS profile similarity score"),
    ("no_saint", "No SAINT-derived score"),
    ("no_priors", "No priors"),
)

# Typography: bold + larger everywhere except legend (legend stays regular weight).
TITLE_FS = 20
AXIS_LABEL_FS = 18
TICK_LABEL_FS = 16
LEGEND_FS = 13


def normalize_protein_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def make_pairkey(p1: str, p2: str) -> Tuple[str, str]:
    a = normalize_protein_name(p1)
    b = normalize_protein_name(p2)
    return (a, b) if a <= b else (b, a)


def resolve_model_data_pkl(dataset_dir: Path) -> Path:
    """model_data for ablation layouts (variant filename under seed_*)."""
    merged_file = dataset_dir / "merged" / "merged_chains.pkl"
    if not merged_file.is_file():
        raise FileNotFoundError(f"Missing merged chains: {merged_file}")
    with open(merged_file, "rb") as f:
        merged_data = pickle.load(f)
    seeds = merged_data["seeds"]
    seed0 = int(seeds[0])
    seed_dir = dataset_dir / f"seed_{seed0}"
    baseline = seed_dir / f"0_model23_n_{seed0}_model_data.pkl"
    if baseline.is_file():
        return baseline
    for pattern in (
        f"0_model23_n_no_corr_{seed0}_model_data.pkl",
        f"0_model23_n_no_saint_{seed0}_model_data.pkl",
        f"0_model23_n_no_priors_{seed0}_model_data.pkl",
    ):
        c = seed_dir / pattern
        if c.is_file():
            return c
    matches = sorted(seed_dir.glob("0_model23_n*_model_data.pkl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No model_data.pkl under {seed_dir}")


def pair_scores_from_mean_edges(mean_edges_path: Path, model_data_pkl: Path) -> pd.DataFrame:
    with open(model_data_pkl, "rb") as f:
        md = pickle.load(f)
    idx_map = build_pair_to_flat_index(md)
    vec = np.load(mean_edges_path).astype(float).ravel().clip(0, 1)
    if vec.size != len(idx_map):
        raise ValueError(
            f"{mean_edges_path}: vector length {vec.size} != model pair count {len(idx_map)}"
        )
    inv_pk: list = [None] * vec.size  # type: ignore[var-annotated]
    for pk, ix in idx_map.items():
        inv_pk[ix] = pk
    if any(p is None for p in inv_pk):
        raise ValueError(f"Non-contiguous flat indices in model_data for {mean_edges_path}")
    return pd.DataFrame({"PairKey": inv_pk, "Score": vec.astype(float)})


def load_base_labels_and_scores_prob(base_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(base_tsv, sep="\t")
    req = {"Protein1", "Protein2", "Probability", "In_Any_Database"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{base_tsv} missing columns: {missing}")

    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)
    df["KnownPositive"] = df["In_Any_Database"].fillna(False).astype(bool).astype(int)
    df["PairKey"] = df.apply(lambda r: make_pairkey(r["Protein1"], r["Protein2"]), axis=1)
    return df[["PairKey", "KnownPositive", "Probability"]].rename(columns={"Probability": "Score"})


def load_base_labels_and_scores_edgefreq(
    base_tsv: Path,
    mean_edges_path: Path,
    base_results_dir: Path,
    dataset_folder: str,
) -> pd.DataFrame:
    df = pd.read_csv(base_tsv, sep="\t")
    req = {"Protein1", "Protein2", "In_Any_Database"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{base_tsv} missing columns: {missing}")
    df["KnownPositive"] = df["In_Any_Database"].fillna(False).astype(bool).astype(int)
    with open(_model_data_pkl(base_results_dir, dataset_folder), "rb") as f:
        model_data = pickle.load(f)
    df = _attach_edge_frequency(df, mean_edges_path, model_data, dataset_folder)
    df["PairKey"] = df.apply(lambda r: make_pairkey(r["Protein1"], r["Protein2"]), axis=1)
    return df[["PairKey", "KnownPositive", "Edge_frequency"]].rename(
        columns={"Edge_frequency": "Score"}
    )


def load_ablation_scores_prob(ablation_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(ablation_tsv, sep="\t")
    req = {"Protein1", "Protein2", "Probability"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{ablation_tsv} missing columns: {missing}")

    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"]).copy()
    df["Probability"] = df["Probability"].clip(0, 1)
    df["PairKey"] = df.apply(lambda r: make_pairkey(r["Protein1"], r["Protein2"]), axis=1)
    return df[["PairKey", "Probability"]].rename(columns={"Probability": "Score"})


def slugify(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def compute_roc_with_thresholds(y_true: np.ndarray, y_score: np.ndarray):
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return fpr, tpr, thr, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation_ablations")
    ap.add_argument(
        "--edge-frequency",
        action="store_true",
        help="Use mean_edges_main_cluster.npy (main-cluster draw mask) instead of TSV Probability.",
    )
    ap.add_argument(
        "--legacy-cluster1-edge-frequency",
        action="store_true",
        help="With --edge-frequency: use cluster-1 mean_edges_cluster_1.npy (updated_analysis + ablation clustering_k3).",
    )
    ap.add_argument(
        "--base-results-dir",
        type=Path,
        default=ROOT / "results",
        help="For edge-frequency base model: folder with cullin_*_config_multi_seeds (model_data).",
    )
    ap.add_argument(
        "--ablation-results-dir",
        type=Path,
        default=ROOT / "script" / "results",
        help="Ablations live under <this>/cullin_<cond>_no_* .",
    )
    ap.add_argument(
        "--post-sampling-base-models",
        type=Path,
        default=ROOT / "post_sampling_analysis" / "base_models",
        help="Base main-cluster npy: <this>/cullin_<cond>_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy",
    )
    ap.add_argument(
        "--updated-analysis-root",
        type=Path,
        default=ROOT / "updated_analysis",
        help="Legacy cluster-1 base npy root: <root>/cullin_<cond>_config_multi_seeds_binary/clustering_k3/mean_edges_cluster_1.npy",
    )
    ap.add_argument(
        "--name-suffix",
        type=str,
        default="AUTO",
        help='Output filename suffix, e.g. "_edgefreq". AUTO: _edgefreq if --edge-frequency else "".',
    )
    ap.add_argument(
        "--no-overwrite",
        action="store_true",
        help="If set, do not overwrite existing output TSV/PNG files (fails fast).",
    )
    args = ap.parse_args()

    suffix = args.name_suffix
    if suffix == "AUTO":
        suffix = "_edgefreq" if args.edge_frequency else ""

    if args.legacy_cluster1_edge_frequency and not args.edge_frequency:
        ap.error("--legacy-cluster1-edge-frequency requires --edge-frequency")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    curves_dir = out_dir / f"roc_base_vs_ablations_curves{suffix}"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # Panel / processing order: HIV WT, HIV ΔVIF, mock (matches 3-panel combine).
    for cond in ["wt", "vif", "mock"]:
        base_tsv = ROOT / "validation_pdb_and_databases" / f"cullin_{cond}_database_comparison.tsv"

        if args.edge_frequency:
            if args.legacy_cluster1_edge_frequency:
                base_npy = (
                    args.updated_analysis_root
                    / f"cullin_{cond}_config_multi_seeds_binary"
                    / "clustering_k3"
                    / "mean_edges_cluster_1.npy"
                )
                base_hint = "Run script/run_cullin_cluster1_average_edges.sh or equivalent."
            else:
                base_npy = (
                    args.post_sampling_base_models
                    / FULL_MODEL_RESULTS_SUBDIR[cond]
                    / "main_cluster_network"
                    / "mean_edges_main_cluster.npy"
                )
                base_hint = (
                    "Run script/extract_main_cluster_network.py for the base model "
                    "(post_sampling_analysis/base_models/.../main_cluster_network)."
                )
            if not base_npy.is_file():
                raise FileNotFoundError(
                    f"Edge-frequency mode requires base npy: {base_npy}\n{base_hint}"
                )
            df_base = load_base_labels_and_scores_edgefreq(
                base_tsv,
                base_npy,
                args.base_results_dir,
                FULL_MODEL_RESULTS_SUBDIR[cond],
            )
        else:
            df_base = load_base_labels_and_scores_prob(base_tsv)

        y = df_base["KnownPositive"].to_numpy(dtype=int)
        base_score = df_base["Score"].to_numpy(dtype=float)
        fpr_b, tpr_b, thr_b, auc_b = compute_roc_with_thresholds(y, base_score)

        curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {
            "Full model": (fpr_b, tpr_b, auc_b)
        }
        pd.DataFrame({"FPR": fpr_b, "TPR": tpr_b, "Threshold": thr_b}).to_csv(
            curves_dir / f"{cond}__{slugify('Full model')}.tsv",
            sep="\t",
            index=False,
        )

        for slug, display_label in ABLATION_MODELS:
            ab_dir = args.ablation_results_dir / f"cullin_{cond}_{slug}"
            if args.edge_frequency:
                if args.legacy_cluster1_edge_frequency:
                    ab_npy = ab_dir / "clustering_k3" / "mean_edges_cluster_1.npy"
                else:
                    ab_npy = ab_dir / "main_cluster_network" / "mean_edges_main_cluster.npy"
                if not ab_npy.is_file():
                    print(f"WARN: skip {display_label} ({cond}): missing {ab_npy}")
                    continue
                try:
                    md_path = resolve_model_data_pkl(ab_dir)
                except FileNotFoundError as e:
                    print(f"WARN: skip {display_label} ({cond}): {e}")
                    continue
                df_ab = pair_scores_from_mean_edges(ab_npy, md_path)
            else:
                ab_tsv = ab_dir / f"cullin_{cond}_{slug}_all_pairs.tsv"
                if not ab_tsv.is_file():
                    continue
                df_ab = load_ablation_scores_prob(ab_tsv)

            merged = df_base[["PairKey", "KnownPositive"]].merge(
                df_ab, on="PairKey", how="left", validate="one_to_one"
            )
            merged["Score"] = merged["Score"].fillna(0.0)
            y_ab = merged["KnownPositive"].to_numpy(dtype=int)
            s_ab = merged["Score"].to_numpy(dtype=float)
            fpr, tpr, thr, auc = compute_roc_with_thresholds(y_ab, s_ab)
            curves[display_label] = (fpr, tpr, auc)
            pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr}).to_csv(
                curves_dir / f"{cond}__{slug}.tsv",
                sep="\t",
                index=False,
            )

        # Slightly wider/taller axes; two-line legend labels keep the box narrower.
        fig, ax = plt.subplots(figsize=(9.2, 7.2))
        for label, (fpr, tpr, auc) in curves.items():
            ax.plot(fpr, tpr, lw=3.0, label=f"{label}\n(AUROC={auc:.3f})")
            summary_rows.append(
                {
                    "Condition": cond,
                    "Model": label,
                    "Score_kind": (
                        "edge_frequency_main_cluster"
                        if args.edge_frequency and not args.legacy_cluster1_edge_frequency
                        else (
                            "edge_frequency_cluster1"
                            if args.edge_frequency
                            else "probability"
                        )
                    ),
                    "AUROC": auc,
                    "N_pairs": int(len(df_base)),
                    "N_known_positives": int(df_base["KnownPositive"].sum()),
                }
            )

        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=2.0, label="Random\n(AUROC=0.5)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(
            "False Positive Rate",
            fontsize=AXIS_LABEL_FS,
            fontweight="bold",
        )
        ax.set_ylabel(
            "True Positive Rate",
            fontsize=AXIS_LABEL_FS,
            fontweight="bold",
        )
        ax.set_title(
            CONDITION_TITLE[cond],
            fontsize=TITLE_FS,
            fontweight="bold",
            pad=12,
        )
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FS)
        for tl in ax.get_xticklabels():
            tl.set_fontweight("bold")
        for tl in ax.get_yticklabels():
            tl.set_fontweight("bold")
        ax.grid(alpha=0.25)
        leg = ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            borderaxespad=0.35,
            labelspacing=0.35,
            handlelength=2.2,
            handletextpad=0.45,
            fontsize=LEGEND_FS,
            prop={"size": LEGEND_FS, "weight": "normal"},
            frameon=True,
            fancybox=False,
            shadow=False,
        )
        for text in leg.get_texts():
            text.set_fontweight("normal")
        fig.tight_layout()

        out_png = out_dir / f"roc_base_vs_ablations_{cond}{suffix}.png"
        if args.no_overwrite and out_png.exists():
            raise SystemExit(
                f"Refusing to overwrite existing file (use without --no-overwrite): {out_png}"
            )
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Saved {out_png}")

    out_tsv = out_dir / f"roc_base_vs_ablations_summary{suffix}.tsv"
    if args.no_overwrite and out_tsv.exists():
        raise SystemExit(
            f"Refusing to overwrite existing file (use without --no-overwrite): {out_tsv}"
        )
    pd.DataFrame(summary_rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved {out_tsv}")


if __name__ == "__main__":
    main()

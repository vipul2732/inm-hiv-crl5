#!/usr/bin/env python3
"""
Compare the final binary main-cluster network (e.g. freq >= 0.9) to database labels
on the same candidate pair universe as validation_analysis (PU setting).

Universe: rows in validation_pdb_and_databases/cullin_{mock,wt,vif}_database_comparison.tsv
Known positive: In_Any_Database (BioGRID / CORUM / PDB-style flags already in TSV).

We do not have confirmed negatives; reported metrics:

  - Recall (recovery): fraction of known positives that the network includes (edge = 1).
  - Precision among predictions: known-positive rate among predicted edges (TP / #predicted);
    interpret as a PU lower bound / enrichment over random, not classical specificity.
  - Enrichment: P(Known+ | predicted) / P(Known+ | not predicted).
  - Counts: TP, predicted edges, known positives, predicted-but-not-in-DB (unlabeled).

Typical use (default paths for main_cluster_binary_freq_ge_0p6.npy):

  python script/compare_binary_network_to_database.py \\
    --validation-dir validation_pdb_and_databases \\
    --results-dir results

Writes by default:
  - main_cluster_binary_vs_database_summary.tsv / .json

Optional: --output-figure <path.png> for a 2-panel PU validation figure.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validation_analysis import (  # noqa: E402
    _model_data_pkl,
    build_pair_to_flat_index,
    normalize_protein_name,
    set_pub_style,
)

DATASET_FOLDER = {
    "MOCK": "cullin_mock_config_multi_seeds",
    "WT": "cullin_wt_config_multi_seeds",
    "VIF": "cullin_vif_config_multi_seeds",
}

TSV_KEY = {"MOCK": "mock", "WT": "wt", "VIF": "vif"}


def _cutoff_file_token(freq_cutoff: float) -> str:
    return f"{float(freq_cutoff):g}".replace(".", "p")


def default_binary_npy(
    dataset_folder: str,
    freq_cutoff: float,
    min_wilson_lb: Optional[float],
) -> Path:
    ft = _cutoff_file_token(freq_cutoff)
    stem = f"main_cluster_binary_freq_ge_{ft}"
    if min_wilson_lb is not None:
        stem += f"_wilson_lb_ge_{_cutoff_file_token(min_wilson_lb)}"
    return (
        REPO_ROOT
        / "post_sampling_analysis"
        / "base_models"
        / dataset_folder
        / "main_cluster_network"
        / f"{stem}.npy"
    )


def attach_binary_column(
    df: pd.DataFrame,
    binary_vec: np.ndarray,
    model_data: dict,
    label: str,
) -> pd.DataFrame:
    vec = np.asarray(binary_vec).astype(np.uint8).ravel()
    pair_idx = build_pair_to_flat_index(model_data)
    if vec.size != len(pair_idx):
        raise ValueError(f"{label}: binary len {vec.size} != pair map {len(pair_idx)}")

    preds: List[int] = []
    missing = 0
    for _, row in df.iterrows():
        a = normalize_protein_name(row["Protein1"])
        b = normalize_protein_name(row["Protein2"])
        key = tuple(sorted((a, b)))
        k = pair_idx.get(key)
        if k is None:
            missing += 1
            preds.append(-1)
        else:
            preds.append(int(vec[k]))
    out = df.copy()
    out["Predicted_edge"] = preds
    if missing:
        print(f"WARNING {label}: {missing} TSV pairs not in model pair map (dropped)")
    out = out[out["Predicted_edge"] >= 0].copy()
    out["Predicted_edge"] = out["Predicted_edge"].astype(int)
    return out


def pu_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    pred = df["Predicted_edge"].astype(bool)
    pos = df["KnownPositive"].astype(bool)
    n = int(len(df))
    n_pos = int(pos.sum())
    n_pred = int(pred.sum())
    tp = int((pred & pos).sum())
    pred_not_db = int((pred & ~pos).sum())
    not_pred_pos = int((~pred & pos).sum())

    recall = float(tp / n_pos) if n_pos else float("nan")
    prec_among_pred = float(tp / n_pred) if n_pred else float("nan")

    n_not_pred = int((~pred).sum())
    rate_pos_if_pred = float(tp / n_pred) if n_pred else float("nan")
    pos_in_not = int((~pred & pos).sum())
    rate_pos_if_not = float(pos_in_not / n_not_pred) if n_not_pred else float("nan")
    if rate_pos_if_not > 0:
        enrichment = rate_pos_if_pred / rate_pos_if_not
    else:
        enrichment = float("nan")

    return {
        "n_pairs_universe": n,
        "n_known_positive": n_pos,
        "n_predicted_edge": n_pred,
        "n_tp_predicted_and_db": tp,
        "n_predicted_not_in_db_unlabeled": pred_not_db,
        "n_known_positive_missed": not_pred_pos,
        "recall_known_positives": recall,
        "precision_among_predicted_edges": prec_among_pred,
        "rate_known_pos_given_predicted": rate_pos_if_pred,
        "rate_known_pos_given_not_predicted": rate_pos_if_not,
        "enrichment_predicted_vs_not": enrichment,
    }


def run_one(
    display_name: str,
    comparison_tsv: Path,
    binary_npy: Path,
    results_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(comparison_tsv, sep="\t")
    df["KnownPositive"] = (
        df["In_Any_Database"].fillna(False).astype(bool).astype(int)
    )
    folder = DATASET_FOLDER[display_name]
    md_path = _model_data_pkl(results_dir, folder)
    with open(md_path, "rb") as f:
        model_data = pickle.load(f)
    binary_vec = np.load(binary_npy)
    df2 = attach_binary_column(df, binary_vec, model_data, display_name)
    m = pu_metrics(df2)
    m["dataset"] = display_name
    m["comparison_tsv"] = str(comparison_tsv.resolve())
    m["binary_npy"] = str(binary_npy.resolve())
    m["model_data_pkl"] = str(md_path.resolve())
    return m, df2


def plot_binary_network_vs_database(
    rows: List[Dict[str, Any]],
    out_path: Path,
    freq_cutoff: float,
) -> None:
    """Two-panel figure: recall / DB hit rate among predictions; enrichment vs ¬predicted."""
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    datasets = [r["dataset"] for r in rows]
    x = np.arange(len(datasets))
    w = 0.36
    recall_pct = [100.0 * float(r["recall_known_positives"]) for r in rows]
    prec_pct = [100.0 * float(r["precision_among_predicted_edges"]) for r in rows]

    ax0 = axes[0]
    ax0.bar(x - w / 2, recall_pct, w, label="Recall of DB positives", color="#2a9d8f")
    ax0.bar(x + w / 2, prec_pct, w, label="DB rate among predicted edges", color="#e76f51")
    ax0.set_xticks(x)
    ax0.set_xticklabels(datasets)
    ax0.set_ylabel("Percent")
    ax0.set_title("Recovery and hit rate (PU)")
    ax0.set_ylim(0, min(105, max(max(recall_pct + prec_pct, default=0) * 1.15, 20)))
    ax0.legend(loc="upper right")
    ax0.grid(True, axis="y")

    ax1 = axes[1]
    enrich = [float(r["enrichment_predicted_vs_not"]) for r in rows]
    ax1.bar(x, enrich, color="#264653", width=0.55)
    ax1.axhline(1.0, ls="--", lw=2, color="gray", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("Fold enrichment")
    ax1.set_title(r"$P(\mathrm{DB+}\mid \mathrm{pred}) \,/\, P(\mathrm{DB+}\mid \neg \mathrm{pred})$")
    ax1.grid(True, axis="y")
    ymax = max(enrich) * 1.12 if enrich else 1.0
    ax1.set_ylim(0, ymax if ymax > 1 else 2)

    fig.suptitle(
        f"Main-cluster binary network (freq ≥ {freq_cutoff}) vs In_Any_Database · "
        f"N ≈ {rows[0].get('n_pairs_universe', '?')} candidate pairs per condition",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        0.01,
        "PU setting: pairs not in DB are unlabeled, not true negatives.",
        ha="center",
        fontsize=11,
        style="italic",
        color="#333333",
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare binary main-cluster network to database flags (PU metrics)."
    )
    ap.add_argument("--validation-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument(
        "--freq-cutoff",
        type=float,
        default=0.6,
        help="Mean-frequency token in default binary npy name (default 0.6).",
    )
    ap.add_argument(
        "--min-wilson-lb",
        type=float,
        default=None,
        help="If set, default binary path includes _wilson_lb_ge_<token> (match extract_main_cluster_network).",
    )
    ap.add_argument("--binary-npy-mock", type=Path, default=None)
    ap.add_argument("--binary-npy-wt", type=Path, default=None)
    ap.add_argument("--binary-npy-vif", type=Path, default=None)
    ap.add_argument(
        "--output-summary-tsv",
        type=Path,
        default=None,
        help="Default: <validation-dir>/main_cluster_binary_vs_database_summary.tsv",
    )
    ap.add_argument(
        "--output-summary-json",
        type=Path,
        default=None,
        help="Default: <validation-dir>/main_cluster_binary_vs_database_summary.json",
    )
    ap.add_argument(
        "--output-pairwise-dir",
        type=Path,
        default=None,
        help="If set, write per-dataset TSV with Predicted_edge and KnownPositive per pair.",
    )
    ap.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="If set, write a 2-panel PU validation figure to this PNG path.",
    )
    args = ap.parse_args()

    val_dir = args.validation_dir.resolve()
    res_dir = args.results_dir.resolve()

    rows: List[Dict[str, Any]] = []
    for display in ("MOCK", "WT", "VIF"):
        key = TSV_KEY[display]
        comp = val_dir / f"cullin_{key}_database_comparison.tsv"
        if not comp.is_file():
            print(f"SKIP {display}: missing {comp}")
            continue
        bin_arg = getattr(args, f"binary_npy_{key}")
        bin_path = (
            Path(bin_arg).resolve()
            if bin_arg is not None
            else default_binary_npy(
                DATASET_FOLDER[display],
                float(args.freq_cutoff),
                args.min_wilson_lb,
            )
        )
        if not bin_path.is_file():
            raise SystemExit(f"Missing binary npy for {display}: {bin_path}")
        meta, df_p = run_one(display, comp, bin_path, res_dir)
        rows.append(meta)
        if args.output_pairwise_dir is not None:
            out_dir = args.output_pairwise_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            slim = df_p[
                [
                    "Protein1",
                    "Protein2",
                    "KnownPositive",
                    "Predicted_edge",
                    "In_Any_Database",
                ]
            ].copy()
            slim.to_csv(
                out_dir / f"cullin_{key}_binary_network_vs_db_pairs.tsv",
                sep="\t",
                index=False,
            )

    if not rows:
        raise SystemExit("No datasets processed.")

    summary = pd.DataFrame(rows)
    out_tsv = args.output_summary_tsv or (val_dir / "main_cluster_binary_vs_database_summary.tsv")
    out_json = args.output_summary_json or (val_dir / "main_cluster_binary_vs_database_summary.json")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_tsv, sep="\t", index=False)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")

    print(summary.to_string(index=False))
    print(f"\nWrote {out_tsv}")
    print(f"Wrote {out_json}")

    if args.output_figure is not None:
        fig_path = args.output_figure.resolve()
        plot_binary_network_vs_database(rows, fig_path, float(args.freq_cutoff))
        print(f"Wrote {fig_path}")

    print(
        "\nNote: Unlabeled pairs are not true negatives; "
        "precision_among_predicted_edges is the DB hit rate among called edges, "
        "and recall_known_positives is recovery of database-supported pairs in this universe."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

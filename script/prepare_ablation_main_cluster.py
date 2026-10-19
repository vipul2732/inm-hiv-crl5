#!/usr/bin/env python3
"""
For one ablation multi-seeds directory under script/results/:

1) Ensure binary_edges_full.npy matches merged z row count (re-run Z→M if not).
2) Ensure draw labels exist (same pipeline as compute_post_sampling_imp_metrics):
   prefer posterior_hamming_linkage/draw_labels_sampling_precision.npy, else
   draw_labels_full_posterior.npy — same row count as merged z.
3) Run extract_main_cluster_network → main_cluster_network/ (M = 1{Z > 0.5}, centroid C at mean freq ≥ 0.6).

Usage:
  python script/prepare_ablation_main_cluster.py --multi-seeds-dir script/results/cullin_mock_no_corr

For all nine ablations (sampling precision + main-cluster + mentor), prefer:
  bash script/run_ablation_sampling_precision_pipeline.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binary_ensemble import load_z_from_results_pkl, run_pipeline  # noqa: E402
from compute_post_sampling_imp_metrics import analyze_one_dataset  # noqa: E402
from extract_main_cluster_network import run as extract_main_cluster_run  # noqa: E402


def _merged_z_rows(merged_pkl: Path) -> int:
    z = load_z_from_results_pkl(merged_pkl)
    return int(z.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--multi-seeds-dir",
        type=Path,
        required=True,
        help="e.g. script/results/cullin_mock_no_corr",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only sync binary + draw labels; do not write main_cluster_network.",
    )
    args = ap.parse_args()

    multi = args.multi_seeds_dir.resolve()
    merged = multi / "merged" / "merged_chains.pkl"
    if not merged.is_file():
        raise SystemExit(f"Missing {merged}")

    T = _merged_z_rows(merged)
    ens_dir = multi / "binary_ensemble_from_merged"
    bin_full = ens_dir / "binary_edges_full.npy"
    need_bin = True
    if bin_full.is_file():
        n_bin = int(np.load(bin_full, mmap_mode="r").shape[0])
        if n_bin == T:
            need_bin = False
        else:
            print(
                f"Binary ensemble rows {n_bin} != merged z rows {T}; regenerating Z→M …",
                flush=True,
            )
    else:
        print("Missing binary ensemble; running Z→M …", flush=True)

    if need_bin:
        run_pipeline(merged, model_data_pkl=None, output_dir=ens_dir)

    link_dir = multi / "posterior_hamming_linkage"
    lab_sp = link_dir / "draw_labels_sampling_precision.npy"
    lab_full = link_dir / "draw_labels_full_posterior.npy"
    lab_path = lab_sp if lab_sp.is_file() else lab_full
    need_hc = True
    if lab_path.is_file():
        n_lab = int(np.load(lab_path).shape[0])
        if n_lab == T:
            need_hc = False
        else:
            print(
                f"draw_labels length {n_lab} != merged z rows {T}; recomputing Hamming linkage + labels …",
                flush=True,
            )
    else:
        print("Missing draw labels; running post-sampling linkage (full merged draws) …", flush=True)

    if need_hc:
        results_root = multi.parent
        dataset = multi.name
        analyze_one_dataset(
            results_root,
            dataset,
            multi,
            cluster_max_draws=None,
            seed=0,
            min_cluster_size_mixing=50,
            mixing_test_alpha=0.05,
            n_tau_balance_scan=400,
            max_clusters_ab_per_tau_row=2500,
            reuse_linkage=False,
            z_binary_threshold=0.5,
        )
        if not lab_path.is_file():
            raise SystemExit(f"Expected {lab_path} after hierarchical clustering")
        if int(np.load(lab_path).shape[0]) != T:
            raise SystemExit(f"Label length mismatch after clustering: {lab_path}")

    if args.skip_extract:
        print(f"Done (skip extract): {multi}", flush=True)
        return

    out_net = multi / "main_cluster_network"
    extract_main_cluster_run(
        multi_seeds_dir=multi,
        labels_path=lab_path,
        out_dir=out_net,
        freq_cutoff=0.6,
        model_data_pkl=None,
        min_wilson_lb=None,
        wilson_alpha=0.05,
        z_binary_threshold=0.5,
    )
    print(f"Done: {out_net / 'mean_edges_main_cluster.npy'}", flush=True)


if __name__ == "__main__":
    main()

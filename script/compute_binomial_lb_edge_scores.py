#!/usr/bin/env python3
"""
Per-edge lower confidence bounds on the Bernoulli proportion P(edge=1) using only draws
in a fixed cluster (same mask as mean_edges_cluster_k.npy).

Methods:
  - wilson: Wilson score interval lower bound (vectorized, fast)
  - clopper: exact Clopper–Pearson (via scipy.stats.binomtest), slower

Two data sources (binary rule aligned with post_sampling_z_binary.Z_BINARY_THRESHOLD for merged z):

  A) Binary ensemble folder (same inputs as average_cluster_edges.py):
       binary_edges_full.npy + maxclust_labels.npy

  B) Merged chains pickle + draw labels (e.g. posterior_hamming_linkage):
       merged_chains.pkl z + draw_labels_sampling_precision.npy (or draw_labels_full_posterior.npy)

Output: one float vector in [0,1], upper-tri order matching mean_edges_cluster_*.npy.

Example (ensemble — matches cluster1_edgefreq updated_analysis layout):

  python script/compute_binomial_lb_edge_scores.py \\
    --ensemble-dir updated_analysis/cullin_mock_config_multi_seeds_binary \\
    --out-dir updated_analysis/cullin_mock_config_multi_seeds_binary/clustering_k3 \\
    --cluster-id 1 --method wilson

Example (merged + post_sampling labels):

  python script/compute_binomial_lb_edge_scores.py \\
    --merged-chains results/cullin_mock_config_multi_seeds/merged/merged_chains.pkl \\
    --labels post_sampling_analysis/base_models/cullin_mock_config_multi_seeds/\\
posterior_hamming_linkage/draw_labels_sampling_precision.npy \\
    --cluster-label 1 \\
    --out-npy /tmp/wilson_lb_mock_cluster1.npy --method wilson
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import binomtest, norm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binary_ensemble import load_binary_ensemble  # noqa: E402
from post_sampling_z_binary import binary_m_from_z, effective_z_binary_threshold  # noqa: E402


def wilson_lower_bound_all(k: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Vectorized Wilson score lower bound for Binomial(n, p); k = successes per edge."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    k = np.asarray(k, dtype=np.float64)
    p_hat = k / float(n)
    z2 = z * z
    nn = float(n)
    denom = 1.0 + z2 / nn
    center = p_hat + z2 / (2.0 * nn)
    inside = p_hat * (1.0 - p_hat) + z2 / (4.0 * nn)
    inside = np.maximum(inside, 0.0)
    rad = z * np.sqrt(inside / nn)
    lb = (center - rad) / denom
    return np.clip(lb, 0.0, 1.0).astype(np.float64)


def clopper_pearson_lower_bound_all(k: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Exact one-sided lower Clopper–Pearson bound at level alpha (two-sided CI symmetric)."""
    kk = np.asarray(k, dtype=np.int64).ravel()
    out = np.empty(kk.shape[0], dtype=np.float64)
    level = 1.0 - alpha
    for j in range(kk.shape[0]):
        r = binomtest(int(kk[j]), int(n))
        ci = r.proportion_ci(confidence_level=level, method="exact")
        out[j] = float(ci.low)
    return out


def run_ensemble(
    ensemble_dir: Path,
    out_dir: Path,
    labels_path: Path,
    cluster_id: int,
    method: str,
    alpha: float,
) -> Path:
    X_uint8, meta = load_binary_ensemble(ensemble_dir)
    X = np.asarray(X_uint8, dtype=np.uint8)
    lab = np.load(labels_path)
    lab = np.asarray(lab, dtype=np.int32).ravel()
    if lab.shape[0] != X.shape[0]:
        raise ValueError(f"labels length {lab.shape[0]} != draws {X.shape[0]}")
    mask = lab == int(cluster_id)
    n_used = int(mask.sum())
    if n_used == 0:
        raise ValueError(f"No draws with cluster_id={cluster_id}")
    sub = X[mask]
    k = sub.sum(axis=0).astype(np.float64)
    if method == "wilson":
        vec = wilson_lower_bound_all(k, n_used, alpha)
    elif method == "clopper":
        vec = clopper_pearson_lower_bound_all(k, n_used, alpha)
    else:
        raise ValueError(method)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "wilson_lb" if method == "wilson" else "clopper_pearson_lb"
    out_npy = out_dir / f"{tag}_edges_cluster_{int(cluster_id)}.npy"
    np.save(out_npy, vec)

    meta_out: Dict[str, Any] = {
        "source": "binary_ensemble",
        "ensemble_dir": str(ensemble_dir.resolve()),
        "labels": str(labels_path.resolve()),
        "cluster_id": int(cluster_id),
        "n_draws_used": n_used,
        "n_draws_total": int(X.shape[0]),
        "n_edges": int(vec.size),
        "method": method,
        "alpha_two_sided": float(alpha),
        "confidence_level": float(1.0 - alpha),
        "out_npy": out_npy.name,
        "rule": (
            "M_draw from binary_edges_full.npy; k_j = sum of M over cluster draws; "
            "LB on Binomial(n, p_j) (binary built with post_sampling_z_binary threshold)"
        ),
    }
    (out_dir / f"{tag}_cluster_{int(cluster_id)}_meta.json").write_text(
        json.dumps(meta_out, indent=2) + "\n"
    )
    return out_npy


def run_merged(
    merged_chains: Path,
    labels_path: Path,
    cluster_label: int | None,
    out_npy: Path,
    method: str,
    alpha: float,
    largest_cluster: bool = False,
    z_binary_threshold: float | None = None,
) -> Path:
    with open(merged_chains, "rb") as f:
        d = pickle.load(f)
    z = np.asarray(d["merged_samples"]["z"], dtype=np.float32)
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    M = binary_m_from_z(z.astype(np.float64), z_threshold=z_binary_threshold)
    lab = np.load(labels_path).astype(np.int64).ravel()
    if lab.shape[0] != M.shape[0]:
        raise ValueError(f"labels length {lab.shape[0]} != z rows {M.shape[0]}")
    if largest_cluster:
        uniq, cnts = np.unique(lab, return_counts=True)
        cl = int(uniq[int(np.argmax(cnts))])
    else:
        if cluster_label is None:
            raise ValueError("cluster_label required when largest_cluster is False")
        cl = int(cluster_label)
    mask = lab == cl
    n_used = int(mask.sum())
    if n_used == 0:
        raise ValueError(f"No draws with cluster_label={cl}")
    sub = M[mask]
    k = sub.sum(axis=0).astype(np.float64)
    if method == "wilson":
        vec = wilson_lower_bound_all(k, n_used, alpha)
    elif method == "clopper":
        vec = clopper_pearson_lower_bound_all(k, n_used, alpha)
    else:
        raise ValueError(method)

    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, vec)

    tag = "wilson_lb" if method == "wilson" else "clopper_pearson_lb"
    meta_out = {
        "source": "merged_chains",
        "merged_chains": str(merged_chains.resolve()),
        "labels": str(labels_path.resolve()),
        "cluster_label": cl,
        "largest_cluster_mode": bool(largest_cluster),
        "n_draws_used": n_used,
        "n_edges": int(vec.size),
        "method": method,
        "alpha_two_sided": float(alpha),
        "out_npy": str(out_npy.resolve()),
    }
    z_thr = float(effective_z_binary_threshold(z_binary_threshold))
    meta_out["z_binary_threshold"] = z_thr
    meta_out["per_draw_binary_rule"] = f"M = 1 if Z > {z_thr} else 0"
    out_npy.with_name(out_npy.stem + "_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    return out_npy


def main() -> int:
    ap = argparse.ArgumentParser(description="Binomial Wilson / Clopper–Pearson LB edge scores per cluster.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ensemble-dir",
        type=Path,
        default=None,
        help="Folder with binary_edges_full.npy (see average_cluster_edges / binary_ensemble).",
    )
    src.add_argument(
        "--merged-chains",
        type=Path,
        default=None,
        help="merged_chains.pkl (uses z, M=1{z>Z_thr} from post_sampling_z_binary).",
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="With --ensemble-dir: output directory (default: <ensemble-dir>/clustering_k3).",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="maxclust / hierarchical labels .npy (length = n draws).",
    )
    ap.add_argument("--cluster-id", type=int, default=1, help="Cluster label to keep (ensemble mode).")
    ap.add_argument(
        "--cluster-label",
        type=int,
        default=None,
        help="With --merged-chains: label value (default: same as --cluster-id).",
    )
    ap.add_argument(
        "--largest-cluster",
        action="store_true",
        help="With --merged-chains: use the cluster with the most draws (overrides --cluster-label).",
    )
    ap.add_argument(
        "--out-npy",
        type=Path,
        default=None,
        help="Required with --merged-chains: where to save the LB vector.",
    )
    ap.add_argument(
        "--method",
        choices=("wilson", "clopper"),
        default="wilson",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for Wilson / exact interval (default 0.05 → ~95%% CI).",
    )
    ap.add_argument(
        "--z-binary-threshold",
        type=float,
        default=None,
        help="Merged-chains mode only: threshold on Z for per-draw M (default: module constant).",
    )
    args = ap.parse_args()

    if args.ensemble_dir is not None:
        ens = args.ensemble_dir.resolve()
        out_dir = args.out_dir.resolve() if args.out_dir else (ens / "clustering_k3")
        labels = args.labels.resolve() if args.labels else (out_dir / "maxclust_labels.npy")
        if not labels.is_file():
            raise SystemExit(f"Labels not found: {labels} (pass --labels)")
        path = run_ensemble(ens, out_dir, labels, args.cluster_id, args.method, float(args.alpha))
        print(json.dumps({"wrote": str(path)}, indent=2))
        return 0

    # merged
    if args.out_npy is None:
        raise SystemExit("--out-npy required with --merged-chains")
    if args.labels is None:
        raise SystemExit("--labels required with --merged-chains")
    cl_opt: int | None = None
    if not args.largest_cluster:
        cl_opt = int(args.cluster_label if args.cluster_label is not None else args.cluster_id)
    path = run_merged(
        args.merged_chains.resolve(),
        args.labels.resolve(),
        cl_opt,
        args.out_npy.resolve(),
        args.method,
        float(args.alpha),
        largest_cluster=bool(args.largest_cluster),
        z_binary_threshold=args.z_binary_threshold,
    )
    print(json.dumps({"wrote": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

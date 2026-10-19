#!/usr/bin/env python3
"""
Edge-wise mean (and optional binary consensus) over draws in one maxclust group.

Uses binary_edges_full.npy + maxclust_labels.npy from clustering.py --k-clusters K.
If labels are missing, pass --recompute-linkage --k-clusters K (expensive) and optional
--write-labels to save maxclust_labels.npy next to outputs.

Mean vector is in [0,1]: fraction of draws in that cluster with edge = 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binary_ensemble import load_binary_ensemble  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_or_compute_labels(
    X: np.ndarray,
    labels_path: Path,
    recompute: bool,
    k: int,
    write_labels: bool,
) -> np.ndarray:
    if labels_path.exists() and not recompute:
        lab = np.load(labels_path)
        return np.asarray(lab, dtype=np.int32)
    if not recompute:
        raise FileNotFoundError(
            f"Missing {labels_path}; re-run clustering.py --k-clusters {k} "
            f"or pass --recompute-linkage --k-clusters {k}"
        )
    if not HAS_SCIPY:
        raise ImportError("scipy required for --recompute-linkage")
    Xf = np.asarray(X, dtype=np.float32, order="C")
    logger.info("pdist/linkage on %s x %s …", Xf.shape[0], Xf.shape[1])
    Z = linkage(pdist(Xf, metric="cityblock"), method="average")
    lab = fcluster(Z, t=int(k), criterion="maxclust").astype(np.int32)
    if write_labels:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, lab)
        logger.info("Wrote %s", labels_path)
    return lab


def run(
    ensemble_dir: Path,
    out_dir: Path,
    cluster_id: int,
    labels_npy: Optional[Path],
    recompute_linkage: bool,
    k_clusters: int,
    write_labels: bool,
    binary_threshold: Optional[float],
) -> None:
    ensemble_dir = Path(ensemble_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_uint8, _meta = load_binary_ensemble(ensemble_dir)
    X = np.asarray(X_uint8, dtype=np.float32, order="C")
    T, M = X.shape

    if labels_npy is None:
        labels_path = out_dir / "maxclust_labels.npy"
    else:
        labels_path = Path(labels_npy)

    labels = load_or_compute_labels(
        X, labels_path, recompute=recompute_linkage, k=k_clusters, write_labels=write_labels
    )
    if labels.shape[0] != T:
        raise ValueError(f"labels length {labels.shape[0]} != n_draws {T}")

    mask = labels == int(cluster_id)
    n_used = int(mask.sum())
    if n_used == 0:
        raise ValueError(f"No draws with cluster_id={cluster_id}")

    sub = X_uint8[mask].astype(np.float64, copy=False)
    mean_edges = sub.mean(axis=0)
    np.save(out_dir / f"mean_edges_cluster_{cluster_id}.npy", mean_edges.astype(np.float64))

    idx = np.flatnonzero(mask)
    np.save(out_dir / f"cluster_{cluster_id}_draw_indices.npy", idx.astype(np.int64))

    meta_out: Dict[str, Any] = {
        "ensemble_dir": str(ensemble_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "cluster_id": int(cluster_id),
        "n_draws_used": n_used,
        "n_draws_total": int(T),
        "m_edges": int(M),
        "mean_edges_npy": f"mean_edges_cluster_{cluster_id}.npy",
        "draw_indices_npy": f"cluster_{cluster_id}_draw_indices.npy",
        "rule": "mean_edges[j] = fraction of cluster draws with edge j == 1",
    }
    if binary_threshold is not None:
        binary = (mean_edges >= float(binary_threshold)).astype(np.uint8)
        bn = f"mean_edges_cluster_{cluster_id}_binary_t{binary_threshold:g}.npy"
        np.save(out_dir / bn, binary)
        meta_out["binary_threshold"] = float(binary_threshold)
        meta_out["binary_npy"] = bn

    with open(out_dir / "average_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    logger.info(
        "cluster %s: %s / %s draws, mean_edges → %s",
        cluster_id,
        n_used,
        T,
        out_dir,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Mean binary edges over one maxclust cluster")
    p.add_argument("--ensemble-dir", type=Path, required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write means (default: ensemble-dir/clustering_k3)",
    )
    p.add_argument("--cluster-id", type=int, default=1)
    p.add_argument("--labels-npy", type=Path, default=None, help="Override path to maxclust_labels.npy")
    p.add_argument(
        "--recompute-linkage",
        action="store_true",
        help="If no labels file, recompute linkage+maxclust (slow for 30k)",
    )
    p.add_argument("--k-clusters", type=int, default=3)
    p.add_argument(
        "--write-labels",
        action="store_true",
        help="With --recompute-linkage, save maxclust_labels.npy to out-dir",
    )
    p.add_argument(
        "--no-binary",
        action="store_true",
        help="Do not write thresholded binary consensus",
    )
    p.add_argument("--binary-threshold", type=float, default=0.5)
    args = p.parse_args()

    out_dir = args.out_dir or (Path(args.ensemble_dir) / "clustering_k3")
    binary_thr = None if args.no_binary else float(args.binary_threshold)

    run(
        ensemble_dir=args.ensemble_dir,
        out_dir=out_dir,
        cluster_id=args.cluster_id,
        labels_npy=args.labels_npy,
        recompute_linkage=args.recompute_linkage,
        k_clusters=args.k_clusters,
        write_labels=args.write_labels,
        binary_threshold=binary_thr,
    )
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()

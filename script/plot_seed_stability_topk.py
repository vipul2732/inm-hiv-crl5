#!/usr/bin/env python3
"""
Seed stability (reproducibility) of Top-K predicted edges for base vs ablated models.

For each condition (mock/wt/vif) and each model (full, no_corr, no_saint, no_priors),
we compute the Top-K edge set separately for each seed (13/42/71) and then report
the mean pairwise Jaccard overlap across seeds.

This is complementary to database/PDB enrichment: a model can have higher recovery
but lower stability, which is usually undesirable for discovery.
"""

from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def normalize_protein_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def find_seed_run_pkl(dataset_dir: Path, seed: int) -> Path:
    """
    Find the main per-seed run pickle containing samples (z) for a dataset/seed.
    Works for both base and ablation naming schemes.
    """
    seed_dir = dataset_dir / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed dir not found: {seed_dir}")

    candidates = sorted(seed_dir.glob(f"0_model23_n*_{seed}.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No per-seed run pkl found in {seed_dir} matching 0_model23_n*_{seed}.pkl")
    return candidates[0]


def find_model_data_pkl(dataset_dir: Path, seed: int) -> Path:
    seed_dir = dataset_dir / f"seed_{seed}"
    baseline = seed_dir / f"0_model23_n_{seed}_model_data.pkl"
    if baseline.exists():
        return baseline
    candidates = sorted(seed_dir.glob("0_model23_n*_model_data.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model_data pkl found in {seed_dir}")
    return candidates[0]


def load_mean_edge_probs_for_seed(dataset_dir: Path, seed: int) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Returns:
      - edges: list of (Protein1, Protein2) pairs (normalized), length = n_edges
      - probs: np.ndarray of mean probabilities for each edge, same length
    """
    run_pkl = find_seed_run_pkl(dataset_dir, seed)
    model_data_pkl = find_model_data_pkl(dataset_dir, seed)

    with open(run_pkl, "rb") as f:
        obj = pkl.load(f)
    samples = obj["samples"]
    z = samples["z"]

    # Convert JAX array -> numpy
    z_arr = np.array(z)
    if z_arr.ndim == 3:
        z_arr = z_arr.reshape(-1, z_arr.shape[-1])
    elif z_arr.ndim != 2:
        z_arr = z_arr.reshape(1, -1)

    # sigmoid and mean over samples
    # (avoid importing jax here; use numpy)
    probs = 1.0 / (1.0 + np.exp(-z_arr))
    mean_probs = probs.mean(axis=0)

    with open(model_data_pkl, "rb") as f:
        model_data = pkl.load(f)
    node_idx2name = model_data.get("node_idx2name", {})
    if not node_idx2name:
        raise ValueError(f"node_idx2name missing in {model_data_pkl}")

    n_nodes = len(node_idx2name)
    edges: List[Tuple[str, str]] = []
    k = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p1 = normalize_protein_name(node_idx2name[i])
            p2 = normalize_protein_name(node_idx2name[j])
            edges.append((p1, p2))
            k += 1

    if k != len(mean_probs):
        raise ValueError(
            f"Edge count mismatch for {dataset_dir.name} seed {seed}: "
            f"constructed {k} edges but have {len(mean_probs)} probabilities"
        )

    return edges, mean_probs


def topk_edge_set(edges: List[Tuple[str, str]], probs: np.ndarray, k: int) -> Set[Tuple[str, str]]:
    if k <= 0:
        return set()
    if k > len(edges):
        k = len(edges)
    idx = np.argsort(-probs)[:k]
    out: Set[Tuple[str, str]] = set()
    for i in idx:
        a, b = edges[int(i)]
        out.add(tuple(sorted((a, b))))
    return out


def jaccard(a: Set[Tuple[str, str]], b: Set[Tuple[str, str]]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def mean_pairwise_jaccard(sets: Dict[int, Set[Tuple[str, str]]]) -> float:
    seeds = sorted(sets.keys())
    if len(seeds) < 2:
        return float("nan")
    vals = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            vals.append(jaccard(sets[seeds[i]], sets[seeds[j]]))
    return float(np.mean(vals)) if vals else float("nan")


def compute_stability_for_model(dataset_dir: Path, seeds: List[int], ks: List[int]) -> Dict[int, float]:
    # Preload per seed mean probs once; keep only seeds with the modal edge size
    per_seed_all = {}
    edge_sizes = {}
    for s in seeds:
        try:
            edges, probs = load_mean_edge_probs_for_seed(dataset_dir, s)
        except Exception as e:
            print(f"[warn] {dataset_dir.name}: skipping seed {s} due to load error: {e}")
            continue
        per_seed_all[s] = (edges, probs)
        edge_sizes[s] = len(probs)

    # Select the most common edge size (mode)
    sizes = list(edge_sizes.values())
    if not sizes:
        return {k: float('nan') for k in ks}
    unique, counts = np.unique(sizes, return_counts=True)
    modal_size = int(unique[int(np.argmax(counts))])

    kept = [s for s in sorted(edge_sizes.keys()) if edge_sizes.get(s) == modal_size]
    dropped = [s for s in seeds if s not in kept]
    if dropped:
        print(f"[warn] {dataset_dir.name}: dropping seeds with non-modal edge size {modal_size}: {dropped} (sizes={edge_sizes})")

    per_seed = {s: per_seed_all[s] for s in kept}

    out = {}
    for k in ks:
        if len(per_seed) < 2:
            out[k] = float("nan")
            continue
        top_sets = {s: topk_edge_set(per_seed[s][0], per_seed[s][1], k) for s in kept}
        out[k] = mean_pairwise_jaccard(top_sets)
    return out


def plot_condition(condition: str, ks: List[int], out_path: Path):
    # dataset directories
    base_dir = ROOT / "results" / f"cullin_{condition}_config_multi_seeds"
    ab_root = ROOT / "script" / "results"

    model_dirs = {
        "Full model": base_dir,
        "No correlation": ab_root / f"cullin_{condition}_no_corr",
        "No SAINT": ab_root / f"cullin_{condition}_no_saint",
        "No priors": ab_root / f"cullin_{condition}_no_priors",
    }

    seeds = [13, 42, 71]

    stability: Dict[str, Dict[int, float]] = {}
    for name, ddir in model_dirs.items():
        stability[name] = compute_stability_for_model(ddir, seeds, ks)

    # Plot grouped bars: x = models, groups = K
    models = list(model_dirs.keys())
    x = np.arange(len(models))
    width = 0.18 if len(ks) >= 4 else 0.22
    offsets = (np.arange(len(ks)) - (len(ks) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(ks):
        vals = [stability[m].get(k, float("nan")) for m in models]
        ax.bar(x + offsets[i], vals, width=width, label=f"K={k}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Seed stability (mean pairwise Jaccard of Top-K edges)")
    ax.set_title(f"Top-K reproducibility across seeds (cullin_{condition})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=min(len(ks), 4), fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="10,25,50,100", help="Comma-separated K values (e.g. 25,50,100)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation_ablations", help="Output directory for PNGs")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    for cond in ["mock", "wt", "vif"]:
        out_path = args.out_dir / f"seed_stability_topk_{cond}.png"
        plot_condition(cond, ks, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


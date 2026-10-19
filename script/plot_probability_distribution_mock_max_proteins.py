#!/usr/bin/env python3
"""
Compare predicted probability distributions for Mock infection across max_proteins runs.

Specifically compares:
  - Base mock model: results/cullin_mock_config_multi_seeds
  - max_proteins variants (from script/input/...): max90, max100, max110

For each model and each seed (13/42/71):
  - load the per-seed model pickle (contains posterior samples z)
  - convert z -> probability via sigmoid(z)
  - average over posterior samples to get a probability per edge
Then aggregate a normalized probability histogram across edges:
  fraction_in_bin = bin_count / total_edges

Robustness:
  In your current folders, z dimensionality can differ across seeds for the same
  max_proteins setting. We keep only the seeds with the modal edge dimension
  (most common z length) to ensure a consistent histogram universe.

Outputs:
  - validation_base_models/cullin_mock/probability_distribution_compare_max_proteins_mock.png
  - validation_base_models/cullin_mock/probability_distribution_compare_max_proteins_mock.tsv
"""

from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_mean_probabilities_from_seed(seed_pkl: Path) -> np.ndarray:
    with open(seed_pkl, "rb") as f:
        obj = pkl.load(f)
    z = obj["samples"]["z"]
    z_arr = np.array(z)
    if z_arr.ndim == 3:
        z_arr = z_arr.reshape(-1, z_arr.shape[-1])
    elif z_arr.ndim != 2:
        z_arr = z_arr.reshape(1, -1)
    probs = sigmoid(z_arr).mean(axis=0)
    return probs


def modal_edge_length(seeds_to_lengths: Dict[int, int]) -> Tuple[int, List[int]]:
    lengths = list(seeds_to_lengths.values())
    unique, counts = np.unique(lengths, return_counts=True)
    modal = int(unique[int(np.argmax(counts))])
    kept = [s for s, L in seeds_to_lengths.items() if L == modal]
    return modal, sorted(kept)


def histogram_fractions(probs: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    # bins are in [0,1], last bin includes right edge by np.histogram convention
    counts, _ = np.histogram(probs, bins=bin_edges, range=(0.0, 1.0))
    total = counts.sum()
    if total <= 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    return counts.astype(float) / float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation_base_models" / "cullin_mock")
    ap.add_argument("--bins-step", type=float, default=0.1)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [13, 42, 71]

    variants: Dict[str, Path] = {
        "Base": ROOT / "results" / "cullin_mock_config_multi_seeds",
        "Max90": ROOT / "script" / "input" / "cullin" / "cullin_mock" / "results" / "cullin_mock_max90",
        "Max100": ROOT / "script" / "input" / "cullin" / "cullin_mock" / "results" / "cullin_mock_max100",
        "Max110": ROOT / "script" / "input" / "cullin" / "cullin_mock" / "results" / "cullin_mock_max110",
    }

    # probability bins: [0.0,0.1), ..., [0.9,1.0]
    step = float(args.bins_step)
    bin_edges = np.arange(0.0, 1.0 + 1e-9, step)
    if len(bin_edges) < 2:
        raise SystemExit("Need at least 2 bin edges.")

    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges) - 1)]

    # Collect per-variant histogram fractions
    records = []
    for variant_name, variant_dir in variants.items():
        # Determine modal edge dimension and keep only compatible seeds
        seeds_to_lengths: Dict[int, int] = {}
        seed_prob_vectors: Dict[int, np.ndarray] = {}

        for seed in seeds:
            if variant_name == "Base":
                seed_pkl = variant_dir / f"seed_{seed}" / f"0_model23_n_{seed}.pkl"
            else:
                seed_pkl = variant_dir / f"seed_{seed}" / f"0_model23_n_{seed}.pkl"
            if not seed_pkl.exists():
                print(f"[warn] Missing seed pkl for {variant_name} seed {seed}: {seed_pkl}")
                continue

            probs = load_mean_probabilities_from_seed(seed_pkl)
            seed_prob_vectors[seed] = probs
            seeds_to_lengths[seed] = int(len(probs))

        if not seeds_to_lengths:
            print(f"[warn] No seeds found for variant {variant_name}, skipping.")
            continue

        modal_len, kept_seeds = modal_edge_length(seeds_to_lengths)
        if len(kept_seeds) < len(seeds_to_lengths):
            dropped = [s for s in sorted(seeds_to_lengths.keys()) if s not in kept_seeds]
            print(f"[warn] {variant_name}: dropping non-modal seeds {dropped} (modal edge count={modal_len})")

        # build histogram fractions per kept seed
        hist_fracs = []
        for seed in kept_seeds:
            probs = seed_prob_vectors[seed]
            hist_fracs.append(histogram_fractions(probs, bin_edges))

        hist_fracs_arr = np.stack(hist_fracs, axis=0) if hist_fracs else np.zeros((0, len(bin_labels)))
        mean_frac = hist_fracs_arr.mean(axis=0) if len(hist_fracs_arr) else np.zeros(len(bin_labels))
        std_frac = hist_fracs_arr.std(axis=0) if len(hist_fracs_arr) else np.zeros(len(bin_labels))

        for i, label in enumerate(bin_labels):
            records.append({
                "Variant": variant_name,
                "ProbabilityBin": label,
                "BinCenter": (bin_edges[i] + bin_edges[i+1]) / 2.0,
                "FractionMean": mean_frac[i],
                "FractionStd": std_frac[i],
                "ModalEdgeCount": modal_len,
                "SeedsUsed": ",".join(map(str, kept_seeds)),
            })

    df_out = pd.DataFrame(records)
    tsv_out = out_dir / "probability_distribution_compare_max_proteins_mock.tsv"
    df_out.to_csv(tsv_out, sep="\t", index=False)
    print(f"Saved {tsv_out}")

    # Plot bar chart (fractions)
    variants_order = ["Base", "Max90", "Max100", "Max110"]
    present = [v for v in variants_order if v in set(df_out["Variant"].unique())]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(bin_labels))
    width = 0.2 if len(present) <= 4 else 0.15

    for idx, v in enumerate(present):
        sub = df_out[df_out["Variant"] == v].sort_values("BinCenter")
        y = sub["FractionMean"].to_numpy(dtype=float)
        yerr = sub["FractionStd"].to_numpy(dtype=float)
        ax.bar(x + (idx - (len(present) - 1) / 2.0) * width, y, width=width, label=v, yerr=yerr, capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=35, ha="right")
    ax.set_ylabel("Fraction of edges in probability bin")
    ax.set_xlabel("Predicted probability")
    ax.set_title("Mock: probability distribution vs max_proteins")
    ax.set_ylim(0.0, max(0.05, df_out["FractionMean"].max() * 1.25))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()

    png_out = out_dir / "probability_distribution_compare_max_proteins_mock.png"
    fig.savefig(png_out, dpi=300)
    plt.close(fig)
    print(f"Saved {png_out}")


if __name__ == "__main__":
    main()


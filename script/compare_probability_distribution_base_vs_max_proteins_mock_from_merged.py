#!/usr/bin/env python3
"""
Recompute probability distributions for Mock across different max_proteins values
following the merge logic:
  - For each max_proteins variant (90/100/110), load per-seed z samples from the
    corresponding run pkl files.
  - Drop seeds whose z edge dimension is incompatible (keep the modal edge count).
  - For each kept seed, compute mean probabilities over z samples.
  - Bin the edge-level mean probabilities into probability bins (0.0-0.1, ..., 0.9-1.0)
    and compute fraction mean/std across kept seeds.

Base model probabilities are taken from the existing base merged predictions:
  - results/cullin_mock_config_multi_seeds/all_pairs.tsv

Outputs:
  - validation_base_models/cullin_mock/probability_distribution_base_vs_max_proteins_mock_merged_recomputed.png
  - validation_base_models/cullin_mock/probability_distribution_base_vs_max_proteins_mock_merged_recomputed.tsv
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
    # Numerically stable sigmoid not necessary here; values are logits.
    return 1.0 / (1.0 + np.exp(-x))


def prob_bin_edges() -> List[Tuple[float, float]]:
    # 10 bins: [0.0-0.1], ..., [0.9-1.0]
    return [(i / 10.0, (i + 1) / 10.0) for i in range(10)]


def compute_fractions_in_bins(probs: np.ndarray) -> np.ndarray:
    edges = prob_bin_edges()
    fracs = []
    for low, high in edges:
        if high >= 1.0:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        fracs.append(mask.mean() if probs.size else 0.0)
    return np.array(fracs, dtype=float)  # shape (10,)


def load_seed_mean_probs(run_dir: Path, seed: int) -> Tuple[np.ndarray, int]:
    seed_dir = run_dir / f"seed_{seed}"
    pkl_path = seed_dir / f"0_model23_n_{seed}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing seed pkl: {pkl_path}")
    with open(pkl_path, "rb") as f:
        obj = pkl.load(f)
    z = np.array(obj["samples"]["z"])
    # Expect (n_samples, n_edges) but be robust.
    if z.ndim == 3:
        z = z.reshape(-1, z.shape[-1])
    elif z.ndim != 2:
        z = z.reshape(1, -1)
    mean_probs = sigmoid(z).mean(axis=0)
    return mean_probs, int(mean_probs.shape[0])


def modal_int(values: List[int]) -> int:
    # Return the mode; if tie, pick the smallest.
    uniq, counts = np.unique(values, return_counts=True)
    max_count = counts.max()
    candidates = uniq[counts == max_count]
    return int(candidates.min())


def recompute_variant_distribution(
    variant_dir: Path,
    seeds: List[int],
) -> Dict[str, object]:
    # Load seeds and keep modal edge dimension
    seed_mean_probs: Dict[int, np.ndarray] = {}
    edge_counts: Dict[int, int] = {}
    for s in seeds:
        mean_probs, n_edges = load_seed_mean_probs(variant_dir, s)
        seed_mean_probs[s] = mean_probs
        edge_counts[s] = n_edges

    modal_edges = modal_int([edge_counts[s] for s in seeds])
    kept_seeds = [s for s in seeds if edge_counts[s] == modal_edges]

    if len(kept_seeds) < 1:
        raise SystemExit(f"No seeds kept for {variant_dir.name}; modal edge count={modal_edges}")

    per_seed_fracs = np.stack([compute_fractions_in_bins(seed_mean_probs[s]) for s in kept_seeds], axis=0)
    frac_mean = per_seed_fracs.mean(axis=0)
    frac_std = per_seed_fracs.std(axis=0)

    # KDE works on edge-level probability samples
    kept_prob_samples = np.concatenate([seed_mean_probs[s] for s in kept_seeds], axis=0)

    return {
        "modal_edges": modal_edges,
        "kept_seeds": kept_seeds,
        "frac_mean": frac_mean,
        "frac_std": frac_std,
        "kept_prob_samples": kept_prob_samples,
    }


def load_base_distribution_from_all_pairs(all_pairs_tsv: Path) -> Dict[str, object]:
    df = pd.read_csv(all_pairs_tsv, sep="\t")
    if "Probability" not in df.columns:
        raise ValueError(f"{all_pairs_tsv} missing Probability column")
    probs = pd.to_numeric(df["Probability"], errors="coerce").dropna().to_numpy(dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    frac_mean = compute_fractions_in_bins(probs)
    frac_std = np.zeros_like(frac_mean)
    modal_edges = int(probs.shape[0])
    return {
        "modal_edges": modal_edges,
        "kept_seeds": ["13,42,71"],
        "frac_mean": frac_mean,
        "frac_std": frac_std,
        "prob_samples": probs,
    }


def kde_density(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde

    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 2:
        return np.zeros_like(x_grid)
    if np.all(s == s[0]):
        # Degenerate case: approximate a narrow peak
        center = float(s[0])
        sigma = 0.01
        dens = np.exp(-0.5 * ((x_grid - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return dens / dens.max() if dens.max() > 0 else dens

    kde = gaussian_kde(s, bw_method="scott")
    return kde.evaluate(x_grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", type=str, default="probability_distribution_base_vs_max_proteins_mock_merged_recomputed")
    args = ap.parse_args()

    out_dir = ROOT / "validation_base_models" / "cullin_mock"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base model (max_proteins=80) probabilities from existing all-pairs TSV
    base_all_pairs = ROOT / "results" / "cullin_mock_config_multi_seeds" / "all_pairs.tsv"
    base_info = load_base_distribution_from_all_pairs(base_all_pairs)

    base_label = "Base (max=80)"
    max_list = [90, 100, 110]
    # Seeds used by these mock max_proteins configs were originally {13,42,71}.
    default_seeds = [13, 42, 71]

    variant_infos: Dict[int, Dict[str, object]] = {}
    for m in max_list:
        variant_dir = ROOT / "script" / "input" / "cullin" / "cullin_mock" / "results" / f"cullin_mock_max{m}"
        if not variant_dir.exists():
            raise FileNotFoundError(f"Missing variant run dir: {variant_dir}")
        variant_infos[m] = recompute_variant_distribution(variant_dir, default_seeds)

    # Prepare TSV rows
    bins = prob_bin_edges()
    centers = [0.05 + 0.1 * i for i in range(10)]
    out_rows = []

    def add_rows(variant_name: str, frac_mean: np.ndarray, frac_std: np.ndarray, modal_edges: int, seeds_used) -> None:
        for i, (low, high) in enumerate(bins):
            bout = f"{low:.1f}-{high:.1f}" if high < 1.0 else f"{low:.1f}-1.0"
            out_rows.append({
                "Variant": variant_name,
                "ProbabilityBin": bout,
                "BinCenter": centers[i],
                "FractionMean": float(frac_mean[i]),
                "FractionStd": float(frac_std[i]),
                "ModalEdgeCount": int(modal_edges),
                "SeedsUsed": ",".join(seeds_used) if isinstance(seeds_used, list) else str(seeds_used),
            })

    add_rows(base_label, base_info["frac_mean"], base_info["frac_std"], base_info["modal_edges"], base_info["kept_seeds"])
    for m in max_list:
        info = variant_infos[m]
        add_rows(f"Max{m}", info["frac_mean"], info["frac_std"], info["modal_edges"], [str(s) for s in info["kept_seeds"]])

    tsv_out = out_dir / f"{args.out_prefix}.tsv"
    pd.DataFrame(out_rows).to_csv(tsv_out, sep="\t", index=False)

    # Plot: 1x3 panels Base vs each Max
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=True)
    variant_order = [90, 100, 110]
    x_grid = np.linspace(0.0, 1.0, 300)

    for ax, m in zip(axes, variant_order):
        info = variant_infos[m]
        dens_base = kde_density(base_info["prob_samples"], x_grid)
        dens_max = kde_density(info["kept_prob_samples"], x_grid)

        # Shade under each density curve to make overlap easier to read.
        # Keep labels on the line plots (not on fill) so legends stay clean.
        base_color = "C0"
        max_color = "C1"
        dens_base = np.maximum(dens_base, 0.0)
        dens_max = np.maximum(dens_max, 0.0)

        ax.fill_between(x_grid, 0.0, dens_base, color=base_color, alpha=0.18)
        ax.fill_between(x_grid, 0.0, dens_max, color=max_color, alpha=0.18)

        ax.plot(x_grid, dens_base, lw=3.0, color=base_color, label=base_label)
        ax.plot(x_grid, dens_max, lw=3.0, color=max_color, label=f"Max{m}")

        ax.set_xlabel("Predicted probability")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Density (KDE)")
    # Each panel has its own pair of curves (Base + Max{m}), so each panel
    # should get its own legend.
    for ax in axes:
        ax.legend(fontsize=12, frameon=True, loc="upper right")
    fig.suptitle("Mock infection", fontsize=26, fontweight="bold", y=0.98)
    fig.tight_layout()

    png_out = out_dir / f"{args.out_prefix}.png"
    fig.savefig(png_out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved {png_out}")
    print(f"Saved {tsv_out}")


if __name__ == "__main__":
    main()


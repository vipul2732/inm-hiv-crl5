#!/usr/bin/env python3
"""
Compare posterior edge probability distributions:
  - Base mock (merged multi-seed): results/cullin_mock_config_multi_seeds/all_pairs.tsv
  - Random-80 mock: results/cullin_mock_random80_custom/all_pairs.tsv (or custom paths)

Outputs (default under validation_base_models/cullin_mock/):
  - probability_distribution_mock_base_vs_random80.png
  - probability_distribution_mock_base_vs_random80_summary.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def kde_density(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde

    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 2:
        return np.zeros_like(x_grid)
    if np.all(s == s[0]):
        center = float(s[0])
        sigma = 0.01
        dens = np.exp(-0.5 * ((x_grid - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return dens / dens.max() if dens.max() > 0 else dens
    kde = gaussian_kde(s, bw_method="scott")
    return kde.evaluate(x_grid)


def summarize(name: str, probs: np.ndarray) -> dict:
    p = np.asarray(probs, dtype=float)
    return {
        "Dataset": name,
        "N_edges": int(p.size),
        "Mean": float(np.mean(p)),
        "Std": float(np.std(p)),
        "Min": float(np.min(p)),
        "Max": float(np.max(p)),
        "Median": float(np.median(p)),
        "P_gt_0.5": int(np.sum(p > 0.5)),
        "P_gt_0.6": int(np.sum(p > 0.6)),
        "P_gt_0.7": int(np.sum(p > 0.7)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-tsv",
        type=Path,
        default=ROOT / "results" / "cullin_mock_config_multi_seeds" / "all_pairs.tsv",
    )
    ap.add_argument(
        "--random-tsv",
        type=Path,
        default=ROOT / "results" / "cullin_mock_random80_custom" / "all_pairs.tsv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "validation_base_models" / "cullin_mock",
    )
    ap.add_argument("--out-prefix", type=str, default="probability_distribution_mock_base_vs_random80")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_b = pd.read_csv(args.base_tsv, sep="\t")
    df_r = pd.read_csv(args.random_tsv, sep="\t")
    for df, label in [(df_b, "base"), (df_r, "random")]:
        if "Probability" not in df.columns:
            raise ValueError(f"{label}: missing Probability column")

    pb = pd.to_numeric(df_b["Probability"], errors="coerce").dropna().clip(0, 1).to_numpy()
    pr = pd.to_numeric(df_r["Probability"], errors="coerce").dropna().clip(0, 1).to_numpy()

    rows = [
        summarize("Mock base (SAINT-ranked 80, merged seeds)", pb),
        summarize("Mock random 80 (merged seeds)", pr),
    ]
    summary_path = args.out_dir / f"{args.out_prefix}_summary.tsv"
    pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)

    x = np.linspace(0.0, 1.0, 400)
    kb = kde_density(pb, x)
    kr = kde_density(pr, x)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    c0, c1 = "C0", "C1"
    ax.fill_between(x, 0, kb, color=c0, alpha=0.2)
    ax.fill_between(x, 0, kr, color=c1, alpha=0.2)
    ax.plot(x, kb, lw=2.8, color=c0, label=f"Base mock (N={len(pb)}, mean={np.mean(pb):.3f})")
    ax.plot(x, kr, lw=2.8, color=c1, label=f"Random 80 (N={len(pr)}, mean={np.mean(pr):.3f})")
    ax.set_xlabel("Posterior edge probability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Density (KDE)", fontsize=14, fontweight="bold")
    ax.set_title("Mock: probability distribution — base vs random 80-protein subset", fontsize=15, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=True, loc="upper left")
    fig.tight_layout()
    png_path = args.out_dir / f"{args.out_prefix}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {summary_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

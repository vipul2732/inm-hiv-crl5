#!/usr/bin/env python3
"""
Compare posterior edge probability distributions: base multi-seed all_pairs.tsv vs
random-80 multi-seed all_pairs.tsv for mock / wt / vif.

Also supports **mock random-80 without topology priors** (``model23_n_no_priors`` + random
80 proteins): ``--condition mock_random80_no_priors`` (expects merged
``results/cullin_mock_random80_no_priors/all_pairs.tsv``).

Examples:
  python compare_probability_distribution_base_vs_random80.py --condition mock
  python compare_probability_distribution_base_vs_random80.py --condition wt
  python compare_probability_distribution_base_vs_random80.py --condition vif
  python compare_probability_distribution_base_vs_random80.py --condition mock_random80_no_priors

  # Custom paths:
  python compare_probability_distribution_base_vs_random80.py \\
    --base-tsv results/cullin_wt_config_multi_seeds/all_pairs.tsv \\
    --random-tsv results/cullin_wt_random80_custom/all_pairs.tsv \\
    --title "HIV WT type infection" \\
    --out-prefix probability_distribution_wt_base_vs_random80
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

CONDITION_DEFAULTS = {
    "mock": {
        "base_tsv": ROOT / "results" / "cullin_mock_config_multi_seeds" / "all_pairs.tsv",
        "random_tsv": ROOT / "results" / "cullin_mock_random80_custom" / "all_pairs.tsv",
        "title": "Mock infection: base vs random 80-protein subset",
        "out_prefix": "probability_distribution_mock_base_vs_random80",
        "out_subdir": "cullin_mock",
    },
    "wt": {
        "base_tsv": ROOT / "results" / "cullin_wt_config_multi_seeds" / "all_pairs.tsv",
        "random_tsv": ROOT / "results" / "cullin_wt_random80_custom" / "all_pairs.tsv",
        "title": "HIV WT type infection: base vs random 80-protein subset",
        "out_prefix": "probability_distribution_wt_base_vs_random80",
        "out_subdir": "cullin_wt",
    },
    "vif": {
        "base_tsv": ROOT / "results" / "cullin_vif_config_multi_seeds" / "all_pairs.tsv",
        "random_tsv": ROOT / "results" / "cullin_vif_random80_custom" / "all_pairs.tsv",
        "title": "HIV Δ VIF infection: base vs random 80-protein subset",
        "out_prefix": "probability_distribution_vif_base_vs_random80",
        "out_subdir": "cullin_vif",
    },
    "mock_random80_no_priors": {
        "base_tsv": ROOT / "results" / "cullin_mock_config_multi_seeds" / "all_pairs.tsv",
        # Default name from run_multiple_seeds.py (multi_seed folder)
        "random_tsv": ROOT
        / "results"
        / "cullin_mock_config_random80_no_priors_multi_seeds"
        / "all_pairs.tsv",
        "title": "Mock infection: base vs random 80 (no topology priors)",
        "out_prefix": "probability_distribution_mock_base_vs_random80_no_priors",
        "out_subdir": "cullin_mock",
        "random_plot_name": "Random 80 (no priors)",
        "random_summary_name": "Random 80, no topology priors (merged seeds)",
    },
}


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


def summarize(label: str, probs: np.ndarray) -> dict:
    p = np.asarray(probs, dtype=float)
    return {
        "Dataset": label,
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
        "--condition",
        choices=["mock", "wt", "vif", "mock_random80_no_priors"],
        default=None,
        help="Presets for paths/title; mock_random80_no_priors = random 80 + model23_n_no_priors",
    )
    ap.add_argument("--base-tsv", type=Path, default=None)
    ap.add_argument("--random-tsv", type=Path, default=None)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: validation_base_models/<condition>/ )",
    )
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument(
        "--random-label",
        type=str,
        default=None,
        help='Short name for the subset curve in the legend (default: "Random 80")',
    )
    args = ap.parse_args()

    cond_plot_name: str | None = None
    cond_summary_name: str | None = None
    if args.condition is not None:
        d = CONDITION_DEFAULTS[args.condition]
        base_tsv = args.base_tsv or d["base_tsv"]
        random_tsv = args.random_tsv or d["random_tsv"]
        title = args.title or d["title"]
        out_prefix = args.out_prefix or d["out_prefix"]
        out_dir = args.out_dir or (ROOT / "validation_base_models" / d["out_subdir"])
        cond_plot_name = d.get("random_plot_name")
        cond_summary_name = d.get("random_summary_name")
    else:
        if args.base_tsv is None or args.random_tsv is None:
            ap.error(
                "Provide --condition mock|wt|vif|mock_random80_no_priors or both --base-tsv and --random-tsv"
            )
        base_tsv = args.base_tsv
        random_tsv = args.random_tsv
        title = args.title or "Base vs random 80: edge probability distribution"
        out_prefix = args.out_prefix or "probability_distribution_base_vs_random80"
        out_dir = args.out_dir or (ROOT / "validation_base_models")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_tsv = Path(base_tsv).resolve()
    random_tsv = Path(random_tsv).resolve()
    print("compare_probability_distribution_base_vs_random80:")
    print(f"  base TSV:   {base_tsv}")
    print(f"  random TSV: {random_tsv}")

    df_b = pd.read_csv(base_tsv, sep="\t")
    df_r = pd.read_csv(random_tsv, sep="\t")
    for df, label in [(df_b, "base"), (df_r, "random")]:
        if "Probability" not in df.columns:
            raise ValueError(f"{label}: missing Probability column")

    pb = pd.to_numeric(df_b["Probability"], errors="coerce").dropna().clip(0, 1).to_numpy()
    pr = pd.to_numeric(df_r["Probability"], errors="coerce").dropna().clip(0, 1).to_numpy()

    random_plot_name = args.random_label or cond_plot_name or "Random 80"
    random_summary_name = (
        args.random_label or cond_summary_name or "Random 80 (multi-seed merged)"
    )
    rows = [
        summarize("Base (SAINT-ranked / standard multi-seed merged)", pb),
        summarize(random_summary_name, pr),
    ]
    summary_path = out_dir / f"{out_prefix}_summary.tsv"
    pd.DataFrame(rows).to_csv(summary_path, sep="\t", index=False)

    x = np.linspace(0.0, 1.0, 400)
    kb = kde_density(pb, x)
    kr = kde_density(pr, x)

    mb = float(np.mean(pb))
    mr = float(np.mean(pr))

    fig, ax = plt.subplots(figsize=(10, 6.2))
    c0, c1 = "C0", "C1"
    ax.fill_between(x, 0, kb, color=c0, alpha=0.2)
    ax.fill_between(x, 0, kr, color=c1, alpha=0.2)
    ax.plot(x, kb, lw=2.8, color=c0, label=f"Base (mean={mb:.4f})")
    ax.plot(x, kr, lw=2.8, color=c1, label=f"{random_plot_name} (mean={mr:.4f})")
    ax.set_xlabel("Posterior edge probability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Density (KDE)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{title}\n"
        f"(means: base μ={mb:.4f}, random μ={mr:.4f}, Δ={mb - mr:+.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=True, loc="upper left")
    fig.tight_layout()
    png_path = out_dir / f"{out_prefix}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {summary_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()

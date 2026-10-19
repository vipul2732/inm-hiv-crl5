#!/usr/bin/env python3
"""
For validation TSV pairs scored by edge frequency, sweep a hard threshold τ:

  "predicted edge" := Edge_frequency >= τ

Enrichment(τ) = rate(KnownPositive | freq >= τ) / rate(KnownPositive | freq < τ)

Requires enough pairs on each side (--min-per-split) for stable ratios.

Typical use with main-cluster frequencies:

  python script/sweep_edge_frequency_threshold_enrichment.py \\
    --input-dir validation_pdb_and_databases \\
    --results-dir results \\
    --edge-frequency-mock post_sampling_analysis/base_models/cullin_mock_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy \\
    --edge-frequency-wt   post_sampling_analysis/base_models/cullin_wt_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy \\
    --edge-frequency-vif  post_sampling_analysis/base_models/cullin_vif_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy \\
    --output-tsv validation_pdb_and_databases/main_cluster_edgefreq_threshold_sweep.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validation_analysis import load_validation_data  # noqa: E402


def enrichment_at_threshold(
    df: pd.DataFrame,
    tau: float,
    min_per_split: int,
) -> Tuple[float, int, int, float, float]:
    """Returns enrichment, n_ge, n_lt, rate_ge, rate_lt (or nan enrichment if invalid)."""
    ge = df["Edge_frequency"] >= float(tau)
    lt = ~ge
    n_ge = int(ge.sum())
    n_lt = int(lt.sum())
    if n_ge < min_per_split or n_lt < min_per_split:
        return float("nan"), n_ge, n_lt, float("nan"), float("nan")
    r_ge = float(df.loc[ge, "KnownPositive"].mean())
    r_lt = float(df.loc[lt, "KnownPositive"].mean())
    if r_lt <= 0:
        return float("nan"), n_ge, n_lt, r_ge, r_lt
    return r_ge / r_lt, n_ge, n_lt, r_ge, r_lt


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep edge-frequency threshold vs database enrichment.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--edge-frequency-mock", type=Path, required=True)
    ap.add_argument("--edge-frequency-wt", type=Path, required=True)
    ap.add_argument("--edge-frequency-vif", type=Path, required=True)
    ap.add_argument("--output-tsv", type=Path, required=True)
    ap.add_argument(
        "--fixed-thresholds",
        type=str,
        default=None,
        help='Comma-separated cutoffs (e.g. "0.9,0.8,0.7,0.6") instead of a dense grid (--tau-step ignored).',
    )
    ap.add_argument(
        "--output-best-tsv",
        type=Path,
        default=None,
        help="One row per dataset: best τ by enrichment (subject to min-per-split).",
    )
    ap.add_argument(
        "--min-per-split",
        type=int,
        default=100,
        help="Minimum pairs with freq>=τ and freq<τ required to report enrichment.",
    )
    ap.add_argument(
        "--tau-step",
        type=float,
        default=0.005,
        help="Grid step from 0 to 1 (inclusive of 0, exclusive upper bound 1).",
    )
    args = ap.parse_args()

    edge_npys = {
        "mock": args.edge_frequency_mock.resolve(),
        "wt": args.edge_frequency_wt.resolve(),
        "vif": args.edge_frequency_vif.resolve(),
    }
    for k, p in edge_npys.items():
        if not p.is_file():
            raise SystemExit(f"Missing npy for {k}: {p}")

    datasets, score_col = load_validation_data(
        args.input_dir,
        results_dir=args.results_dir.resolve(),
        edge_frequency_npys=edge_npys,
    )
    if score_col != "Edge_frequency":
        raise SystemExit("Internal error: expected Edge_frequency mode.")
    if not datasets:
        raise SystemExit("No datasets loaded.")

    if args.fixed_thresholds:
        taus_list = []
        for part in args.fixed_thresholds.split(","):
            part = part.strip()
            if not part:
                continue
            taus_list.append(float(part))
        if not taus_list:
            raise SystemExit("--fixed-thresholds had no valid numbers.")
        taus = sorted(set(taus_list), reverse=True)
    else:
        taus = list(np.arange(0.0, 1.0, float(args.tau_step)))
    rows: List[dict] = []
    for name, df in datasets.items():
        for tau in taus:
            enrich, n_ge, n_lt, r_ge, r_lt = enrichment_at_threshold(
                df, float(tau), int(args.min_per_split)
            )
            rows.append(
                {
                    "dataset": name,
                    "threshold": round(float(tau), 6),
                    "n_freq_ge_tau": n_ge,
                    "n_freq_lt_tau": n_lt,
                    "rate_known_pos_ge": r_ge,
                    "rate_known_pos_lt": r_lt,
                    "enrichment_ge_vs_lt": enrich,
                }
            )

    out_df = pd.DataFrame(rows)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_tsv, sep="\t", index=False)

    best_rows: List[dict] = []
    for name, df in datasets.items():
        sub = out_df[out_df["dataset"] == name].copy()
        valid = sub[np.isfinite(sub["enrichment_ge_vs_lt"])].copy()
        if valid.empty:
            best_rows.append(
                {
                    "dataset": name,
                    "best_threshold": float("nan"),
                    "best_enrichment": float("nan"),
                    "n_freq_ge_at_best": -1,
                    "n_freq_lt_at_best": -1,
                }
            )
            continue
        j = int(valid["enrichment_ge_vs_lt"].values.argmax())
        row = valid.iloc[j]
        best_rows.append(
            {
                "dataset": name,
                "best_threshold": float(row["threshold"]),
                "best_enrichment": float(row["enrichment_ge_vs_lt"]),
                "n_freq_ge_at_best": int(row["n_freq_ge_tau"]),
                "n_freq_lt_at_best": int(row["n_freq_lt_tau"]),
            }
        )

    best_df = pd.DataFrame(best_rows)
    if args.output_best_tsv is not None:
        args.output_best_tsv.parent.mkdir(parents=True, exist_ok=True)
        best_df.to_csv(args.output_best_tsv, sep="\t", index=False)

    print(f"Wrote sweep: {args.output_tsv.resolve()}")
    print("\nBest threshold per dataset (max enrichment, min split=%d):" % args.min_per_split)
    print(best_df.to_string(index=False))
    if args.output_best_tsv:
        print(f"\nWrote best summary: {args.output_best_tsv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

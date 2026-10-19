#!/usr/bin/env python3
"""
Plot probabilities vs sampling steps for:
  - Within-RNX edges: RUNX1–RUNX2, RUNX1–RUNX3, RUNX2–RUNX3
  - CBFB–RUNXx using CBFB composite membership:
      mean probability over all edges in CBFB composite edge sets that touch RUNXx

X-axis: sampling step / MCMC sample index (0..n_samples-1)
Y-axis: posterior edge probability = sigmoid(z)

This script averages across base model seeds (default: 13,42,71) and shows mean +/- std.

By default, curves and CSV rows use every 100th step (0, 100, 200, …); set
``--x-tick-step 50`` for finer points. X-axis **number labels** default to every
500 steps (``--x-label-step``) so ticks stay readable.

Examples::

  python script/plot_cbfb_rnx_within_rnx_vs_sampling_steps_base_mock.py --condition mock
  python script/plot_cbfb_rnx_within_rnx_vs_sampling_steps_base_mock.py --condition wt
  python script/plot_cbfb_rnx_within_rnx_vs_sampling_steps_base_mock.py --condition vif

  # Custom paths (omit --condition):
  python script/plot_cbfb_rnx_within_rnx_vs_sampling_steps_base_mock.py \\
      --base-dir results/cullin_wt_config_multi_seeds \\
      --out validation_base_models/cullin_wt/cbfb_rnx_within_rnx_vs_sampling_steps.png \\
      --infection-title "HIV WT type infection" --base-label "base wt"
"""

from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CONDITION_DEFAULTS = {
    "mock": {
        "base_dir": ROOT / "results" / "cullin_mock_config_multi_seeds",
        "out": ROOT
        / "validation_base_models"
        / "cullin_mock"
        / "cbfb_rnx_within_rnx_vs_sampling_steps.png",
        "infection_title": "Mock infection",
        "base_label": "base mock",
    },
    "wt": {
        "base_dir": ROOT / "results" / "cullin_wt_config_multi_seeds",
        "out": ROOT
        / "validation_base_models"
        / "cullin_wt"
        / "cbfb_rnx_within_rnx_vs_sampling_steps.png",
        "infection_title": "HIV WT type infection",
        "base_label": "base wt",
    },
    "vif": {
        "base_dir": ROOT / "results" / "cullin_vif_config_multi_seeds",
        "out": ROOT
        / "validation_base_models"
        / "cullin_vif"
        / "cbfb_rnx_within_rnx_vs_sampling_steps.png",
        "infection_title": "HIV Δ VIF infection",
        "base_label": "base vif",
    },
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_protein_name(name: str) -> str:
    """
    Model node names look like 'RUNX1_HUMAN'. Normalize by removing species suffixes.
    """
    s = str(name).strip().upper()
    suffixes = ["_HUMAN", "_MOUSE", "_RAT", "_CHICKEN", "_DROME", "_CAEEL", "_YEAST"]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def edge_idx_from_nodes(i: int, j: int, n_nodes: int) -> int:
    """
    Convert node-index pair (i,j) with i!=j into the corresponding edge index used by the model.

    The model stores undirected edges in an upper-triangle ordering:
      edge_idx = 0 for (0,1), 1 for (0,2), ... then (1,2) etc.
    """
    if j < i:
        i, j = j, i
    # edges for fixed i:
    # count edges for all a < i plus the offset within row i
    return sum((n_nodes - a - 1) for a in range(i)) + (j - i - 1)


def find_runx_node_indices(model_data: Dict[str, object]) -> Dict[str, int]:
    node_idx2name: Dict[int, str] = model_data["node_idx2name"]  # type: ignore[assignment]
    norm2idx = {normalize_protein_name(v): k for k, v in node_idx2name.items()}
    out = {
        "RUNX1": int(norm2idx["RUNX1"]),
        "RUNX2": int(norm2idx["RUNX2"]),
        "RUNX3": int(norm2idx["RUNX3"]),
    }
    return out


def find_cbfb_composite_key(model_data: Dict[str, object]) -> str:
    composite_membership: Dict[str, List[int]] = model_data["composite_membership"]  # type: ignore[assignment]
    for k in composite_membership.keys():
        if "CBFB" in str(k).upper():
            return str(k)
    raise KeyError("No CBFB composite key found in model_data['composite_membership']")


def get_within_rnx_edge_indices(model_data: Dict[str, object]) -> Dict[str, int]:
    n_nodes = int(model_data["N"])  # type: ignore[arg-type]
    runx = find_runx_node_indices(model_data)

    return {
        "RUNX1-RUNX2": edge_idx_from_nodes(runx["RUNX1"], runx["RUNX2"], n_nodes),
        "RUNX1-RUNX3": edge_idx_from_nodes(runx["RUNX1"], runx["RUNX3"], n_nodes),
        "RUNX2-RUNX3": edge_idx_from_nodes(runx["RUNX2"], runx["RUNX3"], n_nodes),
    }


def get_cbfb_rnx_edge_indices(model_data: Dict[str, object]) -> Dict[str, List[int]]:
    """
    CBFB is a composite label (not a node in the prey-prey graph), so we aggregate:
      for RUNXx, take all edge indices in CBFB composite edge sets that touch RUNXx.
    """
    n_nodes = int(model_data["N"])  # type: ignore[arg-type]
    runx = find_runx_node_indices(model_data)

    cb_key = find_cbfb_composite_key(model_data)
    composite_membership: Dict[str, List[int]] = model_data["composite_membership"]  # type: ignore[assignment]
    composite_edge_sets: Dict[int, Iterable[Tuple[int, int]]] = model_data["composite_edge_sets"]  # type: ignore[assignment]

    comp_ids = composite_membership[cb_key]
    if not comp_ids:
        raise ValueError(f"CBFB composite '{cb_key}' has no membership composite ids")

    out: Dict[str, List[int]] = {}
    for rn in ["RUNX1", "RUNX2", "RUNX3"]:
        rn_idx = runx[rn]
        edge_indices: set[int] = set()
        for cid in comp_ids:
            es = composite_edge_sets[cid]
            for i, j in es:
                if i == rn_idx or j == rn_idx:
                    edge_indices.add(edge_idx_from_nodes(i, j, n_nodes))
        out[f"CBFB-{rn}"] = sorted(edge_indices)

    return out


def prob_timeseries_for_edges(z: np.ndarray, edge_indices: List[int]) -> np.ndarray:
    """
    Return P(edge)(t) time series averaged over:
      - chains (if z is 3D: (n_chains, n_samples, M))
      - edges in `edge_indices`
    """
    edge_indices = [int(e) for e in edge_indices]
    if not edge_indices:
        raise ValueError("edge_indices is empty")

    z_arr = np.array(z)
    if z_arr.ndim == 3:
        # (n_chains, n_samples, M)
        z_sel = z_arr[:, :, edge_indices]  # (n_chains, n_samples, n_edges)
        return sigmoid(z_sel).mean(axis=(0, 2))  # (n_samples,)
    if z_arr.ndim == 2:
        # (n_samples, M)
        z_sel = z_arr[:, edge_indices]  # (n_samples, n_edges)
        return sigmoid(z_sel).mean(axis=1)  # (n_samples,)
    if z_arr.ndim == 1:
        # Single sample: (M,)
        z_sel = z_arr[edge_indices]  # (n_edges,)
        p = sigmoid(z_sel).mean()
        return np.array([p], dtype=float)

    raise ValueError(f"Unexpected z array ndim={z_arr.ndim}, shape={z_arr.shape}")


def load_seed_series(seed: int, base_dir: Path) -> Tuple[Dict[str, np.ndarray], int]:
    seed_dir = base_dir / f"seed_{seed}"
    run_pkl = seed_dir / f"0_model23_n_{seed}.pkl"
    model_data_pkl = seed_dir / f"0_model23_n_{seed}_model_data.pkl"

    if not run_pkl.exists():
        raise FileNotFoundError(run_pkl)
    if not model_data_pkl.exists():
        raise FileNotFoundError(model_data_pkl)

    with open(run_pkl, "rb") as f:
        obj = pkl.load(f)
    z = obj["samples"]["z"]

    with open(model_data_pkl, "rb") as f:
        model_data = pkl.load(f)

    within_edges = get_within_rnx_edge_indices(model_data)
    cbfb_edges = get_cbfb_rnx_edge_indices(model_data)

    series: Dict[str, np.ndarray] = {}
    for name, eidx in within_edges.items():
        series[name] = prob_timeseries_for_edges(z, [eidx])
    for name, eidxs in cbfb_edges.items():
        series[name] = prob_timeseries_for_edges(z, eidxs)

    n_samples = int(np.array(next(iter(series.values()))).shape[0])
    return series, n_samples


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--condition",
        choices=["mock", "wt", "vif"],
        default=None,
        help="Preset base-dir, --out, titles (mock / HIV WT / HIV ΔVIF). Optional if you pass paths explicitly.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="13,42,71",
        help="Comma-separated seed ids (default: 13,42,71)",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base model multi-seed directory (default: mock preset, or from --condition)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: mock preset, or from --condition)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: same basename as --out with .csv)",
    )
    ap.add_argument(
        "--x-tick-step",
        type=int,
        default=100,
        metavar="N",
        help="Plot and export probabilities only at steps 0, N, 2N, … (e.g. 50 or 100).",
    )
    ap.add_argument(
        "--x-label-step",
        type=int,
        default=500,
        metavar="N",
        help="X-axis tick **labels** every N steps only (default: 500; less crowded).",
    )
    ap.add_argument(
        "--infection-title",
        type=str,
        default=None,
        help="Figure suptitle (default: from --condition or mock preset).",
    )
    ap.add_argument(
        "--base-label",
        type=str,
        default=None,
        help="Label used inside panel subtitles (default: from --condition or mock preset).",
    )
    args = ap.parse_args()

    mock_defaults = CONDITION_DEFAULTS["mock"]
    if args.condition:
        cfg = CONDITION_DEFAULTS[args.condition]
        base_dir = Path(args.base_dir) if args.base_dir is not None else cfg["base_dir"]
        out_path = Path(args.out) if args.out is not None else cfg["out"]
        infection_title = (
            args.infection_title if args.infection_title is not None else cfg["infection_title"]
        )
        base_label = args.base_label if args.base_label is not None else cfg["base_label"]
    else:
        base_dir = Path(args.base_dir) if args.base_dir is not None else mock_defaults["base_dir"]
        out_path = Path(args.out) if args.out is not None else mock_defaults["out"]
        infection_title = (
            args.infection_title
            if args.infection_title is not None
            else mock_defaults["infection_title"]
        )
        base_label = (
            args.base_label if args.base_label is not None else mock_defaults["base_label"]
        )

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    within_labels = ["RUNX1-RUNX2", "RUNX1-RUNX3", "RUNX2-RUNX3"]
    cbfb_labels = ["CBFB-RUNX1", "CBFB-RUNX2", "CBFB-RUNX3"]

    # Visual styling to keep nearby curves readable.
    # - thinner mean lines
    # - lower-opacity std bands
    # - distinct line styles per curve
    style_map = {
        "RUNX1-RUNX2": "-",
        "RUNX1-RUNX3": "--",
        "RUNX2-RUNX3": ":",
        "CBFB-RUNX1": "-",
        "CBFB-RUNX2": "--",
        "CBFB-RUNX3": ":",
    }

    # Load per-seed time series
    per_seed: List[Dict[str, np.ndarray]] = []
    n_samples_list: List[int] = []
    for seed in seeds:
        series, n_samples = load_seed_series(seed, base_dir)
        per_seed.append(series)
        n_samples_list.append(n_samples)
        print(f"[seed {seed}] n_samples={n_samples}, series_keys={sorted(series.keys())}")

    min_len = int(min(n_samples_list))

    # Aggregate across seeds
    def stack_and_stats(label: str) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.stack([s[label][:min_len] for s in per_seed], axis=0)  # (n_seeds, n_samples)
        return arr.mean(axis=0), arr.std(axis=0)

    all_series_labels = within_labels + cbfb_labels
    stats_by_label: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        lab: stack_and_stats(lab) for lab in all_series_labels
    }

    step = max(1, int(args.x_tick_step))
    label_step = max(1, int(args.x_label_step))
    x_locator = MultipleLocator(label_step)
    idx = np.arange(0, min_len, step, dtype=np.int64)
    x_s = idx  # subsampled steps for plot + CSV

    fig, axes = plt.subplots(2, 1, figsize=(13.8, 9.2), sharex=True)

    ax0 = axes[0]
    for lab in within_labels:
        mean, std = stats_by_label[lab]
        m = mean[idx]
        s = std[idx]
        ax0.plot(
            x_s,
            m,
            lw=1.6,
            ls=style_map.get(lab, "-"),
            label=lab,
            marker="o",
            markersize=3.2,
            markeredgewidth=0.4,
        )
        ax0.fill_between(x_s, m - s, m + s, alpha=0.06)

    ax0.set_title(f"Within-RNX edge probabilities ({base_label})")
    ax0.set_ylabel("Posterior edge probability")
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper left", frameon=True)
    ax0.xaxis.set_major_locator(x_locator)

    ax1 = axes[1]
    for lab in cbfb_labels:
        mean, std = stats_by_label[lab]
        m = mean[idx]
        s = std[idx]
        ax1.plot(
            x_s,
            m,
            lw=1.6,
            ls=style_map.get(lab, "-"),
            label=lab,
            marker="o",
            markersize=3.2,
            markeredgewidth=0.4,
        )
        ax1.fill_between(x_s, m - s, m + s, alpha=0.06)

    ax1.set_title(f"CBFB composite vs RNX proteins ({base_label})")
    ax1.set_xlabel("Sampling step (MCMC sample index)")
    ax1.set_ylabel("Aggregated edge probability")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", frameon=True)
    ax1.xaxis.set_major_locator(x_locator)

    fig.suptitle(infection_title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved plot: {out_path}")

    # Probability time series (mean ± std across seeds) → CSV
    csv_path = args.out_csv
    if csv_path is None:
        csv_path = out_path.with_suffix(".csv")
    else:
        csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_cols: Dict[str, np.ndarray] = {"sampling_step": x_s.astype(np.int64)}
    for lab in all_series_labels:
        mean, std = stats_by_label[lab]
        base = lab.replace("-", "_")
        csv_cols[f"{base}_probability_mean_across_seeds"] = mean[idx].astype(np.float64)
        csv_cols[f"{base}_probability_std_across_seeds"] = std[idx].astype(np.float64)

    df_out = pd.DataFrame(csv_cols)
    df_out.to_csv(csv_path, index=False)
    print(
        f"Saved probabilities CSV: {csv_path}  "
        f"(rows={len(df_out)}, data_step={step}, x_labels_every={label_step}, seeds={seeds})"
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
From existing hierarchical labels + merged posterior z:

1) Take the cluster with the most draws (main cluster).
2) Report mixing: counts (and fractions) of those draws from each MCMC chain
   (merged order = seeds sorted ascending, equal-length concatenation).
3) Mean binary matrix over draws in that cluster → edge-wise frequency in [0,1].
   Sample std of that binary column across the same draws (ddof=1) per edge.
4) Hard network (default): edge on iff mean frequency >= cutoff (default 0.6).
   Optional --min-wilson-lb: also require Wilson LB on P(edge=1) (default: off; Wilson
   still written to TSV/npy for diagnostics when not used for C).
5) Cluster precision: over the same draws, mean/median/RMS Hamming distance from each
   draw's binary M to the centroid binary network C (same rule as step 4).

Binary per draw: M = 1{Z > Z_thr} (Z_thr from post_sampling_z_binary.py by default).

Typical use (labels from post_sampling_analysis Hamming-linkage run):

  python script/extract_main_cluster_network.py \\
    --multi-seeds-dir results/cullin_mock_config_multi_seeds \\
    --labels post_sampling_analysis/base_models/cullin_mock_config_multi_seeds/\\
posterior_hamming_linkage/draw_labels_sampling_precision.npy

Writes under post_sampling_analysis/base_models/<dataset_basename>/main_cluster_network/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validation_analysis import build_pair_to_flat_index, normalize_protein_name  # noqa: E402
from compute_binomial_lb_edge_scores import wilson_lower_bound_all  # noqa: E402
from post_sampling_z_binary import binary_m_from_z, effective_z_binary_threshold  # noqa: E402


def load_z_and_meta(merged_pkl: Path) -> Tuple[np.ndarray, List[int], int]:
    with open(merged_pkl, "rb") as f:
        d = pickle.load(f)
    z = np.asarray(d["merged_samples"]["z"], dtype=np.float32)
    seeds = [int(s) for s in d.get("seeds", [])]
    n_chains = int(d.get("n_chains", len(seeds)))
    return z, seeds, n_chains


def centroid_binary_npy_path(net_dir: Path) -> Path:
    """Consensus main-cluster binary: `binary_network_npy` in summary, else freq_ge_<cutoff> from meta, then legacy."""
    summ = net_dir / "main_cluster_summary.json"
    if not summ.is_file():
        raise FileNotFoundError(summ)
    meta = json.loads(summ.read_text())
    name = meta.get("binary_network_npy")
    candidates: List[Path] = []
    if name:
        candidates.append(net_dir / name)
    fc = meta.get("freq_cutoff_for_binary_network")
    if isinstance(fc, (int, float)):
        tok = _float_token(float(fc))
        p_fc = net_dir / f"main_cluster_binary_freq_ge_{tok}.npy"
        if p_fc not in candidates:
            candidates.append(p_fc)
    for legacy in (
        "main_cluster_binary_freq_ge_0p6.npy",
        "main_cluster_binary_freq_ge_0p9.npy",
    ):
        lp = net_dir / legacy
        if lp not in candidates:
            candidates.append(lp)
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    for p in uniq:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No centroid binary .npy under {net_dir} (tried: {', '.join(str(p) for p in uniq)})"
    )


def default_model_data_pkl(multi_seeds_dir: Path, seeds: List[int]) -> Path:
    seed0 = min(seeds)
    candidates = list((multi_seeds_dir / f"seed_{seed0}").glob("0_model23_n*_model_data.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model_data under {multi_seeds_dir}/seed_{seed0}")
    return candidates[0]


def _float_token(x: float) -> str:
    return f"{float(x):g}".replace(".", "p")


def _binary_column_name(freq_cutoff: float, min_wilson_lb: float | None) -> str:
    if min_wilson_lb is None:
        return f"binary_ge{_float_token(freq_cutoff)}"
    return (
        f"binary_mean_ge{_float_token(freq_cutoff)}_wilson_lb_ge{_float_token(min_wilson_lb)}"
    )


def _binary_npy_stem(freq_cutoff: float, min_wilson_lb: float | None) -> str:
    stem = f"main_cluster_binary_freq_ge_{_float_token(freq_cutoff)}"
    if min_wilson_lb is not None:
        stem += f"_wilson_lb_ge_{_float_token(min_wilson_lb)}"
    return stem


def edge_frequency_to_pair_tsv(
    freq: np.ndarray,
    freq_std: np.ndarray,
    wilson_lb: np.ndarray,
    binary: np.ndarray,
    model_data: dict,
    out_tsv: Path,
    binary_col_name: str,
) -> None:
    idx_map = build_pair_to_flat_index(model_data)
    if freq.size != len(idx_map):
        raise ValueError(f"freq len {freq.size} != pair map {len(idx_map)}")
    if freq_std.shape != freq.shape or wilson_lb.shape != freq.shape:
        raise ValueError("freq_std / wilson_lb shape must match freq")
    inv: List[Tuple[str, str]] = [("", "")] * len(idx_map)
    for pair, k in idx_map.items():
        inv[k] = pair
    lines = [
        "protein1\tprotein2\tfrequency\tfrequency_std\twilson_lb\t"
        f"{binary_col_name}\n"
    ]
    for k in range(len(inv)):
        a, b = inv[k]
        lines.append(
            f"{a}\t{b}\t{freq[k]:.10g}\t{freq_std[k]:.10g}\t{wilson_lb[k]:.10g}\t"
            f"{int(binary[k])}\n"
        )
    out_tsv.write_text("".join(lines))


def run(
    multi_seeds_dir: Path,
    labels_path: Path,
    out_dir: Path,
    freq_cutoff: float,
    model_data_pkl: Path | None,
    min_wilson_lb: float | None,
    wilson_alpha: float,
    z_binary_threshold: float | None = None,
) -> Dict[str, Any]:
    merged_pkl = multi_seeds_dir / "merged" / "merged_chains.pkl"
    if not merged_pkl.is_file():
        raise FileNotFoundError(merged_pkl)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    z, seeds, n_chains = load_z_and_meta(merged_pkl)
    T, M_edges = z.shape
    labels = np.load(labels_path).astype(np.int64).ravel()
    if labels.shape[0] != T:
        raise ValueError(f"labels length {labels.shape[0]} != z rows {T}")

    z_thr = effective_z_binary_threshold(z_binary_threshold)
    M = binary_m_from_z(z, z_threshold=z_binary_threshold)

    uniq, cnts = np.unique(labels, return_counts=True)
    main_cluster_id = int(uniq[np.argmax(cnts)])
    n_main = int(np.max(cnts))

    mask_main = labels == main_cluster_id
    M_main = M[mask_main]
    n_main_rows = int(M_main.shape[0])
    mean_freq = M_main.mean(axis=0).astype(np.float64)
    if n_main_rows < 2:
        std_freq = np.zeros(M_edges, dtype=np.float64)
    else:
        std_freq = M_main.astype(np.float64).std(axis=0, ddof=1)

    k_counts = M_main.sum(axis=0).astype(np.float64)
    if n_main_rows < 1:
        raise ValueError("main cluster has no draws")
    wilson_lb = wilson_lower_bound_all(k_counts, n_main_rows, float(wilson_alpha))

    mean_ok = mean_freq >= float(freq_cutoff)
    if min_wilson_lb is not None:
        binary = (mean_ok & (wilson_lb >= float(min_wilson_lb))).astype(np.uint8)
    else:
        binary = mean_ok.astype(np.uint8)

    # Cluster precision: Hamming distance from each draw's binary M to the centroid network C
    C = binary.astype(np.uint8).ravel()
    if C.size != M_edges:
        raise ValueError(f"centroid length {C.size} != n_edges {M_edges}")
    diff = M_main.astype(np.uint8) != C.reshape(1, -1)
    hamming_per_draw = diff.sum(axis=1).astype(np.float64)
    cluster_precision = {
        "representative_cluster": "largest cluster by draw count (label from --labels)",
        "centroid_binary_rule": (
            f"edge=1 iff mean_freq>={freq_cutoff}"
            + (
                f" AND wilson_lb>={min_wilson_lb} (alpha={wilson_alpha})"
                if min_wilson_lb is not None
                else "; Wilson LB in TSV/npy only (not used for C)"
            )
        ),
        "freq_cutoff": float(freq_cutoff),
        "min_wilson_lb_for_binary": float(min_wilson_lb) if min_wilson_lb is not None else None,
        "wilson_alpha_two_sided": float(wilson_alpha),
        "n_edges": int(M_edges),
        "n_draws_in_largest_cluster": int(n_main_rows),
        "mean_hamming_draw_to_centroid": float(np.mean(hamming_per_draw)),
        "median_hamming_draw_to_centroid": float(np.median(hamming_per_draw)),
        "rms_hamming_draw_to_centroid": float(np.sqrt(np.mean(hamming_per_draw**2))),
        "mean_hamming_divided_by_n_edges": float(np.mean(hamming_per_draw) / M_edges),
    }

    if T % n_chains != 0:
        raise ValueError(f"T={T} not divisible by n_chains={n_chains}")
    S = T // n_chains

    mixing_rows: List[Dict[str, Any]] = []
    flat_idx = np.arange(T)
    chain_ids = flat_idx // S
    for c in range(n_chains):
        in_chain = chain_ids == c
        in_main = mask_main & in_chain
        n_cm = int(in_main.sum())
        mixing_rows.append(
            {
                "chain_index": c,
                "seed": int(seeds[c]) if c < len(seeds) else c,
                "n_samples_in_main_cluster": n_cm,
                "n_samples_total_in_chain": int(S),
                "frac_chain_samples_in_main_cluster": float(n_cm / S) if S else 0.0,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mean_edges_main_cluster.npy", mean_freq)
    np.save(out_dir / "std_edges_main_cluster.npy", std_freq)
    np.save(out_dir / "wilson_lb_edges_main_cluster.npy", wilson_lb)
    binary_stem = _binary_npy_stem(float(freq_cutoff), min_wilson_lb)
    binary_npy_name = f"{binary_stem}.npy"
    np.save(out_dir / binary_npy_name, binary)
    np.save(out_dir / "main_cluster_draw_mask.npy", mask_main.astype(np.uint8))

    if min_wilson_lb is not None:
        bin_rule = (
            f"edge=1 iff mean_freq>={freq_cutoff} AND wilson_lb>={min_wilson_lb} "
            f"(Wilson LB two-sided alpha={wilson_alpha} on cluster draws)"
        )
    else:
        bin_rule = f"edge=1 iff mean_freq>={freq_cutoff} (no Wilson cutoff)"

    meta: Dict[str, Any] = {
        "multi_seeds_dir": str(multi_seeds_dir.resolve()),
        "merged_chains_pkl": str(merged_pkl.resolve()),
        "labels_source": str(labels_path.resolve()),
        "binary_rule_per_draw": f"M = 1 if Z > {z_thr} else 0",
        "z_binary_threshold": float(z_thr),
        "binary_network_rule": bin_rule,
        "main_cluster_id": main_cluster_id,
        "n_draws_in_main_cluster": int(mask_main.sum()),
        "n_draws_total": int(T),
        "n_chains": n_chains,
        "seeds_in_merge_order": seeds,
        "samples_per_chain": int(S),
        "n_edges": int(M_edges),
        "frequency_std_def": "sample std of binary M per edge over draws in main cluster (ddof=1)",
        "wilson_lb_def": (
            "Wilson score lower bound on P(M=1) per edge over main-cluster draws; "
            "same k/n as mean frequency"
        ),
        "wilson_alpha_two_sided": float(wilson_alpha),
        "min_wilson_lb_for_binary": float(min_wilson_lb) if min_wilson_lb is not None else None,
        "freq_cutoff_for_binary_network": float(freq_cutoff),
        "binary_network_npy": binary_npy_name,
        "mixing_by_chain": mixing_rows,
        "cluster_counts_all_labels": {int(u): int(n) for u, n in zip(uniq, cnts)},
        "cluster_precision_hamming_to_centroid": cluster_precision,
    }
    (out_dir / "cluster_precision.json").write_text(
        json.dumps(cluster_precision, indent=2) + "\n"
    )
    (out_dir / "main_cluster_summary.json").write_text(json.dumps(meta, indent=2) + "\n")

    md_path = model_data_pkl or default_model_data_pkl(multi_seeds_dir, seeds)
    with open(md_path, "rb") as f:
        model_data = pickle.load(f)
    meta["model_data_pkl"] = str(md_path.resolve())
    bcol = _binary_column_name(float(freq_cutoff), min_wilson_lb)
    edge_frequency_to_pair_tsv(
        mean_freq,
        std_freq,
        wilson_lb,
        binary,
        model_data,
        out_dir / "main_cluster_edges.tsv",
        bcol,
    )
    (out_dir / "main_cluster_summary.json").write_text(json.dumps(meta, indent=2) + "\n")

    # One-line mixing TSV
    mix_tsv = out_dir / "mixing_by_chain.tsv"
    mix_tsv.write_text(
        "chain_index\tseed\tn_in_main\tn_in_chain\tfrac_chain_in_main\n"
        + "".join(
            f"{r['chain_index']}\t{r['seed']}\t{r['n_samples_in_main_cluster']}\t"
            f"{r['n_samples_total_in_chain']}\t{r['frac_chain_samples_in_main_cluster']:.6f}\n"
            for r in mixing_rows
        )
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--multi-seeds-dir",
        type=Path,
        required=True,
        help="e.g. results/cullin_mock_config_multi_seeds",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="draw_labels .npy (same length as merged z), e.g. draw_labels_full_posterior.npy",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: post_sampling_analysis/base_models/<basename>/main_cluster_network",
    )
    ap.add_argument(
        "--freq-cutoff",
        type=float,
        default=0.6,
        help="Binary network: require mean edge frequency in main cluster >= this (default 0.6).",
    )
    ap.add_argument(
        "--min-wilson-lb",
        type=float,
        default=None,
        help="If set, also require Wilson lower bound on P(edge=1) >= this (e.g. 0.85 with 0.9 mean).",
    )
    ap.add_argument(
        "--wilson-alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for Wilson interval (default 0.05 ≈ 95%%).",
    )
    ap.add_argument("--model-data", type=Path, default=None)
    ap.add_argument(
        "--z-binary-threshold",
        type=float,
        default=None,
        help="M=1 iff Z>thr per draw (default: post_sampling_z_binary.Z_BINARY_THRESHOLD). Use 0 for legacy.",
    )
    args = ap.parse_args()

    multi = args.multi_seeds_dir.resolve()
    if args.out_dir is None:
        out = REPO_ROOT / "post_sampling_analysis" / "base_models" / multi.name / "main_cluster_network"
    else:
        out = args.out_dir.resolve()

    meta = run(
        multi,
        args.labels.resolve(),
        out,
        args.freq_cutoff,
        args.model_data,
        args.min_wilson_lb,
        args.wilson_alpha,
        z_binary_threshold=args.z_binary_threshold,
    )
    cp = meta.get("cluster_precision_hamming_to_centroid") or {}
    print(
        json.dumps(
            {
                "wrote": str(out),
                "main_cluster_id": meta["main_cluster_id"],
                "n_main": meta["n_draws_in_main_cluster"],
                "binary_npy": meta.get("binary_network_npy"),
                "mean_hamming_draw_to_centroid": cp.get("mean_hamming_draw_to_centroid"),
                "mean_hamming_divided_by_n_edges": cp.get("mean_hamming_divided_by_n_edges"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

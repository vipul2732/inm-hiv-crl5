#!/usr/bin/env python3
"""
Correlate main-cluster centroid edge frequencies with:

  1) AP–MS profile similarity (``apms_corr_flat`` in model_data), and
  2) SAINT: both the edge vector stored in ``model_data`` (what the sampler used) and, when
     composite.tsv is available, a **derived** SAINT edgelist from the composite table that still
     scores indirect prey–prey evidence when baits are **not** nodes in the filtered graph (relaxation
     only for this diagnostic; the strict model only fills ``bait_prey_scores`` when bait is a node).
  3) **Number of common baits** per prey–prey edge: count baits B for which both proteins appear as
     Prey (with MSscore > ``--common-baits-min-msscore``). Correlates centroid frequency with this
     count and reports mean counts above/below the centroid frequency cutoff from ``main_cluster_summary``.

Reads:
  - mean_edges_main_cluster.npy (or --mean-freq-npy): length M, same edge order as z / model_data.
  - model_data.pkl: saint_max_pair_score_edgelist, apms_corr_flat (same M; supports edge_filter_map).
  - Optional composite: ``<multi-seeds-dir>/seed_<min_seed>/preprocessed/composite_table.tsv`` or
    ``--composite-tsv``.

Writes JSON (default next to main_cluster_network/) with Pearson/Spearman + shuffle nulls.

Optional ``--saint-npy`` overrides the **model_data** SAINT vector only (derived composite is still
computed for comparison unless ``--no-derived-saint``).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validation_analysis import build_pair_to_flat_index, normalize_protein_name  # noqa: E402

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _as_float_vec(x: Any) -> np.ndarray:
    """Host numpy vector (handles JAX arrays from pickled model_data)."""
    try:
        import jax

        x = jax.device_get(x)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float64).ravel()


def _triu_flat_index(N: int, i: int, j: int) -> int:
    if i > j:
        i, j = j, i
    k = 0
    for r in range(i):
        k += N - r - 1
    k += j - i - 1
    return k


def _derived_saint_edgelist_from_composite(composite_tsv: Path, model_data: dict) -> np.ndarray:
    """
    Edge-wise SAINT summary for diagnostics: direct (bait+prey in graph) + indirect prey–prey
    under the same bait. Bait may be **outside** the node set (rows still contribute to indirect).
    """
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for derived SAINT from composite TSV")
    N = int(model_data["N"])
    M_full = N * (N - 1) // 2
    name2node_idx: Dict[str, int] = model_data["name2node_idx"]
    use_indirect = bool(model_data.get("use_indirect_saint", True))
    indirect_thr = float(model_data.get("indirect_saint_threshold", 0.3))

    df = pd.read_csv(composite_tsv, sep="\t")
    if "MSscore" not in df.columns:
        raise ValueError(f"{composite_tsv}: expected MSscore column")
    df = df[df["MSscore"] > 0.0].copy()

    saint_scores: Dict[Tuple[int, int], float] = {}
    bait_prey_scores: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        bait = str(row["Bait"])
        prey = str(row["Prey"])
        msscore = float(row.get("MSscore", 0.0) or 0.0)
        if bait == prey or msscore <= 0.0:
            continue
        if prey not in name2node_idx:
            continue
        bait_prey_scores.setdefault(bait, {})[prey] = max(
            msscore, bait_prey_scores.get(bait, {}).get(prey, 0.0)
        )
        if bait in name2node_idx:
            i = int(name2node_idx[bait])
            j = int(name2node_idx[prey])
            if i == j:
                continue
            if i > j:
                i, j = j, i
            pair = (i, j)
            prev = saint_scores.get(pair, -1.0)
            if msscore > prev:
                saint_scores[pair] = msscore

    if use_indirect:
        indirect_saint_scores: Dict[Tuple[int, int], float] = {}
        for _bait, prey_dict in bait_prey_scores.items():
            prey_list = list(prey_dict.keys())
            for ii, prey1 in enumerate(prey_list):
                saint1 = float(prey_dict[prey1])
                if saint1 < indirect_thr:
                    continue
                for prey2 in prey_list[ii + 1 :]:
                    saint2 = float(prey_dict[prey2])
                    if saint2 < indirect_thr:
                        continue
                    if prey1 not in name2node_idx or prey2 not in name2node_idx:
                        continue
                    idx1 = int(name2node_idx[prey1])
                    idx2 = int(name2node_idx[prey2])
                    if idx1 == idx2:
                        continue
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    pair = (idx1, idx2)
                    indirect_saint = float(np.sqrt(saint1 * saint2))
                    if indirect_saint < indirect_thr:
                        continue
                    prev = indirect_saint_scores.get(pair, -1.0)
                    if indirect_saint > prev:
                        indirect_saint_scores[pair] = indirect_saint

        for pair, indirect_score in indirect_saint_scores.items():
            if pair not in saint_scores:
                saint_scores[pair] = indirect_score
            else:
                saint_scores[pair] = max(saint_scores[pair], indirect_score)

    out = np.zeros(M_full, dtype=np.float64)
    for (i, j), score in saint_scores.items():
        k = _triu_flat_index(N, i, j)
        if 0 <= k < M_full:
            out[k] = max(out[k], score)
    return out


def _bait_to_prey_sets_from_composite(
    composite_tsv: Path,
    model_data: dict,
    min_msscore: float,
) -> Dict[str, Set[str]]:
    """Bait -> set of prey (graph nodes only) with MSscore > min in composite."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas required")
    name2node_idx: Dict[str, int] = model_data["name2node_idx"]
    df = pd.read_csv(composite_tsv, sep="\t")
    if "MSscore" not in df.columns:
        raise ValueError(f"{composite_tsv}: expected MSscore column")
    bait_to_preys: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        bait = str(row["Bait"])
        prey = str(row["Prey"])
        ms = float(row.get("MSscore", 0.0) or 0.0)
        if ms <= float(min_msscore) or bait == prey:
            continue
        if prey not in name2node_idx:
            continue
        bait_to_preys.setdefault(bait, set()).add(prey)
    return bait_to_preys


def _common_bait_counts_edgelist(model_data: dict, bait_to_preys: Dict[str, Set[str]]) -> np.ndarray:
    """
    For each upper-tri edge (protein i, protein j), count baits B such that both proteins
    appear as Prey with MSscore > threshold in rows with Bait B (same-bait co-occurrence).
    """
    N = int(model_data["N"])
    M_full = N * (N - 1) // 2
    names: Dict[int, str] = model_data["node_idx2name"]
    out = np.zeros(M_full, dtype=np.int32)
    # list baits once for inner loop
    bait_items = list(bait_to_preys.items())
    for i in range(N):
        pi = str(names[i])
        for j in range(i + 1, N):
            pj = str(names[j])
            k = _triu_flat_index(N, i, j)
            c = 0
            for _bait, preys in bait_items:
                if pi in preys and pj in preys:
                    c += 1
            out[k] = c
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan"), float("nan")
    r = np.corrcoef(a, b)[0, 1]
    # two-sided p-value (t approximation)
    df = a.size - 2
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1.0:
        return float(r), float("nan")
    t = r * np.sqrt(df / max(1e-15, (1.0 - r * r)))
    p = 2 * stats.t.sf(abs(t), df) if HAS_SCIPY else float("nan")
    return float(r), float(p)


def _spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan"), float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan"), float("nan")
    try:
        r, p = stats.spearmanr(a, b)
    except (ValueError, FloatingPointError):
        return float("nan"), float("nan")
    return float(r), float(p) if p is not None and np.isfinite(p) else float("nan")


def _shuffle_null_pearson(f: np.ndarray, x: np.ndarray, n_perm: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    f = np.asarray(f, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    mask = np.isfinite(f) & np.isfinite(x)
    f, x = f[mask], x[mask]
    if f.size < 10 or n_perm < 1:
        return {"mean_abs_r_perm": float("nan"), "frac_perm_ge_observed": float("nan")}
    r_obs, _ = _pearson(f, x)
    if not np.isfinite(r_obs):
        return {"mean_abs_r_perm": float("nan"), "frac_perm_ge_observed": float("nan")}
    abs_rs: List[float] = []
    for _ in range(n_perm):
        perm = rng.permutation(f.size)
        rp, _ = _pearson(f[perm], x)
        if np.isfinite(rp):
            abs_rs.append(abs(rp))
    abs_rs_arr = np.array(abs_rs, dtype=np.float64)
    if abs_rs_arr.size == 0:
        return {"mean_abs_r_perm": float("nan"), "frac_perm_ge_observed": float("nan")}
    frac = float("nan")
    if np.isfinite(r_obs):
        frac = float(np.mean(abs_rs_arr >= abs(r_obs)))
    return {
        "mean_abs_r_perm": float(np.mean(abs_rs_arr)),
        "frac_perm_ge_observed": frac,
    }


def _pair_names_for_edges(model_data: dict) -> List[Tuple[str, str]]:
    """Ordered list (protein1, protein2) for flat edge index k = 0..M_full-1."""
    idx = build_pair_to_flat_index(model_data)
    inv: List[Tuple[str, str]] = [("", "")] * len(idx)
    for pair, k in idx.items():
        inv[k] = pair
    return inv


def _filtered_pair_names(model_data: dict) -> List[Tuple[str, str]]:
    inv_full = _pair_names_for_edges(model_data)
    efm = model_data.get("edge_filter_map")
    if efm is None:
        return inv_full
    efm = np.asarray(efm, dtype=np.int64).ravel()
    return [inv_full[int(ko)] for ko in efm]


def _correlation_bundle(
    F: np.ndarray,
    X: np.ndarray,
    *,
    saint_positive_thr: float,
    n_perm: int,
    perm_seed: int,
    x_high_mask: Optional[np.ndarray] = None,
    include_input_gt_threshold_subset: bool = True,
) -> Dict[str, Any]:
    """Correlations of centroid frequency F vs input vector X (same length)."""
    F = np.asarray(F, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64).ravel()

    def one(mask: np.ndarray) -> Dict[str, Any]:
        m = np.asarray(mask, dtype=bool).ravel() & np.isfinite(F) & np.isfinite(X)
        f, x = F[m], X[m]
        pr, pp = _pearson(f, x)
        sp, spp = _spearman(f, x)
        return {
            "n": int(m.sum()),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sp,
            "spearman_p": spp,
        }

    out: Dict[str, Any] = {"all_edges": one(np.ones_like(F, dtype=bool))}
    if include_input_gt_threshold_subset:
        out["subset_input_gt_threshold"] = one(X > saint_positive_thr)
    if x_high_mask is not None:
        out["subset_input_top_quartile"] = one(np.asarray(x_high_mask, dtype=bool))
    out["shuffle_null_pearson"] = _shuffle_null_pearson(F, X, n_perm=n_perm, seed=perm_seed)
    return out


def run_one(
    main_cluster_dir: Path,
    multi_seeds_dir: Path,
    mean_freq_path: Path,
    model_data_path: Optional[Path],
    out_json: Optional[Path],
    out_tsv: Optional[Path],
    saint_positive_thr: float,
    n_perm: int,
    perm_seed: int,
    saint_npy: Optional[Path],
    composite_tsv: Optional[Path],
    no_derived_saint: bool,
    skip_common_baits: bool,
    common_baits_min_msscore: float,
) -> Dict[str, Any]:
    main_cluster_dir = main_cluster_dir.resolve()
    multi_seeds_dir = multi_seeds_dir.resolve()
    mean_freq_path = mean_freq_path.resolve()

    if not mean_freq_path.is_file():
        raise FileNotFoundError(mean_freq_path)

    summ_path = main_cluster_dir / "main_cluster_summary.json"
    if not summ_path.is_file():
        raise FileNotFoundError(summ_path)
    summ = json.loads(summ_path.read_text())

    if model_data_path is None:
        model_data_path = Path(summ.get("model_data_pkl", ""))
        if not model_data_path.is_file():
            seed0 = min(int(s) for s in summ.get("seeds_in_merge_order", [13]))
            cand = list((multi_seeds_dir / f"seed_{seed0}").glob("0_model23_n*_model_data.pkl"))
            if not cand:
                raise FileNotFoundError(
                    f"No model_data pkl; pass --model-data (tried summary + {multi_seeds_dir}/seed_{seed0})"
                )
            model_data_path = cand[0]
    model_data_path = model_data_path.resolve()
    if not model_data_path.is_file():
        raise FileNotFoundError(model_data_path)

    with open(model_data_path, "rb") as f:
        model_data = pickle.load(f)

    F = np.load(mean_freq_path).astype(np.float64).ravel()
    if saint_npy is not None:
        S_model = np.load(saint_npy).astype(np.float64).ravel()
        saint_model_note = f"override --saint-npy: {saint_npy.name}"
    else:
        S_model = _as_float_vec(model_data["saint_max_pair_score_edgelist"])
        saint_model_note = "model_data pickle (saint_max_pair_score_edgelist)"
    R = np.clip(_as_float_vec(model_data["apms_corr_flat"]), 0.0, 1.0)

    if F.size != S_model.size or F.size != R.size:
        raise ValueError(
            f"Length mismatch: centroid_freq {F.size}, saint_model {S_model.size}, corr {R.size} "
            f"(use same model_data / merged z as extract_main_cluster_network)"
        )

    seeds_order = summ.get("seeds_in_merge_order") or []
    seed_min = int(min(int(s) for s in seeds_order)) if seeds_order else 13
    default_composite = (
        multi_seeds_dir / f"seed_{seed_min}" / "preprocessed" / "composite_table.tsv"
    )
    comp_path = composite_tsv if composite_tsv is not None else default_composite

    efm = model_data.get("edge_filter_map")
    if efm is not None:
        efm = np.asarray(efm, dtype=np.int64).ravel()

    S_derived_full: Optional[np.ndarray] = None
    derived_meta: Optional[Dict[str, Any]] = None
    if not no_derived_saint and comp_path.is_file() and HAS_PANDAS:
        try:
            S_derived_full = _derived_saint_edgelist_from_composite(comp_path, model_data)
            if efm is not None:
                S_derived = S_derived_full[efm].astype(np.float64)
            else:
                S_derived = S_derived_full.astype(np.float64)
            if S_derived.shape[0] != F.shape[0]:
                raise ValueError("derived SAINT length mismatch after edge_filter_map")
            derived_meta = {
                "composite_tsv": str(comp_path.resolve()),
                "saint_max": float(np.nanmax(S_derived)),
                "definition": (
                    "Direct SAINT on edges where bait and prey are graph nodes; plus indirect "
                    "prey–prey under the same bait (geometric mean), with baits allowed off-graph "
                    "for building prey sets (diagnostic relaxation vs strict model_data pipeline)."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            derived_meta = {"composite_tsv": str(comp_path.resolve()), "error": str(exc)}
            S_derived_full = None

    thr_r = float(np.quantile(R, 0.75))
    mask_r_topq = R >= thr_r

    K_edges: Optional[np.ndarray] = None
    common_baits_error: Optional[str] = None
    if not skip_common_baits and comp_path.is_file() and HAS_PANDAS:
        try:
            btp = _bait_to_prey_sets_from_composite(
                comp_path, model_data, common_baits_min_msscore
            )
            K_full = _common_bait_counts_edgelist(model_data, btp)
            K_edges = (
                K_full[efm].astype(np.float64)
                if efm is not None
                else K_full.astype(np.float64)
            )
            if K_edges.shape[0] != F.shape[0]:
                raise ValueError("common bait count length mismatch")
            common_baits_meta_counts = {
                "n_baits_with_any_graph_prey": int(len(btp)),
                "min_msscore_for_prey_in_bait_set": float(common_baits_min_msscore),
            }
        except Exception as exc:  # noqa: BLE001
            common_baits_error = str(exc)
            K_edges = None
            common_baits_meta_counts = {}
    else:
        common_baits_meta_counts = {}

    freq_cutoff = float(summ.get("freq_cutoff_for_binary_network", 0.6))

    out: Dict[str, Any] = {
        "main_cluster_dir": str(main_cluster_dir),
        "mean_freq_npy": str(mean_freq_path),
        "model_data_pkl": str(model_data_path),
        "n_edges": int(F.size),
        "N": int(model_data.get("N", 0)),
        "has_edge_filter_map": efm is not None,
        "saint_positive_threshold": float(saint_positive_thr),
        "has_scipy": HAS_SCIPY,
        "has_pandas": HAS_PANDAS,
        "centroid_frequency_vs_apms_similarity": _correlation_bundle(
            F,
            R,
            saint_positive_thr=saint_positive_thr,
            n_perm=n_perm,
            perm_seed=perm_seed,
            x_high_mask=mask_r_topq,
            include_input_gt_threshold_subset=False,
        ),
        "centroid_frequency_vs_saint_stored_model_data": {
            "vector_note": saint_model_note,
            "saint_max": float(np.nanmax(S_model)),
            **_correlation_bundle(
                F,
                S_model,
                saint_positive_thr=saint_positive_thr,
                n_perm=n_perm,
                perm_seed=perm_seed + 10,
            ),
        },
    }
    if float(np.nanmax(S_model)) < 1e-12:
        out["note_saint_model_data_all_zero"] = (
            "Stored SAINT edgelist is all-zero (often when baits are not graph nodes). "
            "Use ``centroid_frequency_vs_saint_derived_from_composite`` or ``--saint-npy``."
        )

    if derived_meta is not None and S_derived_full is not None and "error" not in derived_meta:
        S_d = S_derived_full[efm] if efm is not None else S_derived_full
        out["centroid_frequency_vs_saint_derived_from_composite"] = {
            **derived_meta,
            **_correlation_bundle(
                F,
                S_d,
                saint_positive_thr=saint_positive_thr,
                n_perm=n_perm,
                perm_seed=perm_seed + 20,
            ),
        }
    elif derived_meta is not None:
        out["centroid_frequency_vs_saint_derived_from_composite"] = derived_meta
    else:
        out["centroid_frequency_vs_saint_derived_from_composite"] = {
            "skipped": True,
            "reason": "no composite TSV or pandas missing or --no-derived-saint",
            "composite_tsv_tried": str(comp_path.resolve()),
        }

    if common_baits_error is not None:
        out["centroid_frequency_vs_n_common_baits"] = {
            "composite_tsv": str(comp_path.resolve()),
            "error": common_baits_error,
        }
    elif K_edges is not None:
        cb = _correlation_bundle(
            F,
            K_edges,
            saint_positive_thr=1.0,
            n_perm=n_perm,
            perm_seed=perm_seed + 30,
            x_high_mask=(K_edges >= np.quantile(K_edges, 0.75)) if float(np.max(K_edges)) > 0 else None,
            include_input_gt_threshold_subset=False,
        )

        def _subset_block(mask: np.ndarray) -> Dict[str, Any]:
            m = (
                np.asarray(mask, dtype=bool).ravel()
                & np.isfinite(F)
                & np.isfinite(K_edges)
            )
            f, x = F[m], K_edges[m]
            pr, pp = _pearson(f, x)
            sp, spp = _spearman(f, x)
            return {
                "n": int(m.sum()),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sp,
                "spearman_p": spp,
            }

        cb["subset_n_common_baits_ge_1"] = _subset_block(K_edges >= 1.0)
        if float(np.max(K_edges)) >= 2.0:
            cb["subset_n_common_baits_ge_2"] = _subset_block(K_edges >= 2.0)
        mask_hi_f = F >= freq_cutoff
        out["centroid_frequency_vs_n_common_baits"] = {
            "definition": (
                "Per graph edge (prey–prey pair): number of baits B such that both proteins appear "
                "as Prey in composite_table with MSscore > min_msscore for that bait."
            ),
            "composite_tsv": str(comp_path.resolve()),
            **common_baits_meta_counts,
            **cb,
            "stratify_mean_n_common_baits_by_centroid": {
                "centroid_freq_cutoff": freq_cutoff,
                "mean_n_common_baits_when_freq_ge_cutoff": float(K_edges[mask_hi_f].mean())
                if bool(mask_hi_f.any())
                else None,
                "mean_n_common_baits_when_freq_lt_cutoff": float(K_edges[~mask_hi_f].mean())
                if bool((~mask_hi_f).any())
                else None,
                "n_edges_freq_ge_cutoff": int(mask_hi_f.sum()),
                "n_edges_freq_lt_cutoff": int((~mask_hi_f).sum()),
            },
        }
    else:
        out["centroid_frequency_vs_n_common_baits"] = {
            "skipped": True,
            "reason": "composite missing, pandas missing, or --skip-common-baits",
            "composite_tsv_tried": str(comp_path.resolve()),
        }

    if out_json is None:
        out_json = main_cluster_dir / "centroid_input_correlation.json"
    out_json = out_json.resolve()
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    out["wrote_json"] = str(out_json)

    if out_tsv is not None:
        pairs = _filtered_pair_names(model_data)
        if len(pairs) != F.size:
            raise ValueError(f"pair list len {len(pairs)} != freq {F.size}")
        S_d_col = S_derived_full[efm] if (S_derived_full is not None and efm is not None) else S_derived_full
        kc = K_edges if K_edges is not None else None
        if S_derived_full is None:
            if kc is None:
                lines = [
                    "protein1\tprotein2\tsaint_model_data\tapms_similarity\tcentroid_mean_freq\n"
                ]
                for (a, b), sm, r, f in zip(pairs, S_model, R, F):
                    na = normalize_protein_name(a)
                    nb = normalize_protein_name(b)
                    lines.append(f"{na}\t{nb}\t{sm:.10g}\t{r:.10g}\t{f:.10g}\n")
            else:
                lines = [
                    "protein1\tprotein2\tn_common_baits\tsaint_model_data\t"
                    "apms_similarity\tcentroid_mean_freq\n"
                ]
                for (a, b), kb, sm, r, f in zip(pairs, kc, S_model, R, F):
                    na = normalize_protein_name(a)
                    nb = normalize_protein_name(b)
                    lines.append(
                        f"{na}\t{nb}\t{int(kb)}\t{sm:.10g}\t{r:.10g}\t{f:.10g}\n"
                    )
        else:
            if S_d_col is None or S_d_col.shape[0] != F.size:
                raise ValueError("derived saint column mismatch for TSV")
            if kc is None:
                lines = [
                    "protein1\tprotein2\tsaint_model_data\tsaint_derived_composite\t"
                    "apms_similarity\tcentroid_mean_freq\n"
                ]
                for (a, b), sm, sd, r, f in zip(pairs, S_model, S_d_col, R, F):
                    na = normalize_protein_name(a)
                    nb = normalize_protein_name(b)
                    lines.append(f"{na}\t{nb}\t{sm:.10g}\t{sd:.10g}\t{r:.10g}\t{f:.10g}\n")
            else:
                lines = [
                    "protein1\tprotein2\tn_common_baits\tsaint_model_data\tsaint_derived_composite\t"
                    "apms_similarity\tcentroid_mean_freq\n"
                ]
                for (a, b), kb, sm, sd, r, f in zip(pairs, kc, S_model, S_d_col, R, F):
                    na = normalize_protein_name(a)
                    nb = normalize_protein_name(b)
                    lines.append(
                        f"{na}\t{nb}\t{int(kb)}\t{sm:.10g}\t{sd:.10g}\t{r:.10g}\t{f:.10g}\n"
                    )
        out_tsv = out_tsv.resolve()
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        out_tsv.write_text("".join(lines))
        out["wrote_tsv"] = str(out_tsv)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--main-cluster-dir",
        type=Path,
        required=True,
        help="Directory with main_cluster_summary.json and mean_edges_main_cluster.npy",
    )
    ap.add_argument(
        "--multi-seeds-dir",
        type=Path,
        required=True,
        help="e.g. results/cullin_wt_config_multi_seeds (fallback model_data discovery)",
    )
    ap.add_argument(
        "--mean-freq-npy",
        type=Path,
        default=None,
        help="Default: <main-cluster-dir>/mean_edges_main_cluster.npy",
    )
    ap.add_argument(
        "--model-data",
        type=Path,
        default=None,
        help="Override model_data.pkl (default: from main_cluster_summary.json)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: <main-cluster-dir>/centroid_input_correlation.json",
    )
    ap.add_argument(
        "--out-tsv",
        type=Path,
        default=None,
        help="Optional per-edge TSV: saint, correlation, centroid frequency",
    )
    ap.add_argument(
        "--saint-positive-thr",
        type=float,
        default=1e-6,
        help="Threshold for 'saint_gt_threshold' subset (default 1e-6)",
    )
    ap.add_argument("--n-perm", type=int, default=500, help="Permutations for shuffle null (default 500)")
    ap.add_argument("--perm-seed", type=int, default=42, help="RNG seed for shuffle null")
    ap.add_argument(
        "--saint-npy",
        type=Path,
        default=None,
        help="Optional SAINT target vector (float64, length = n_edges) overriding model_data",
    )
    ap.add_argument(
        "--composite-tsv",
        type=Path,
        default=None,
        help="Composite table for derived SAINT (default: seed_*/preprocessed/composite_table.tsv)",
    )
    ap.add_argument(
        "--no-derived-saint",
        action="store_true",
        help="Do not compute SAINT from composite (only model_data / --saint-npy)",
    )
    ap.add_argument(
        "--skip-common-baits",
        action="store_true",
        help="Skip counting baits shared by both proteins (saves time)",
    )
    ap.add_argument(
        "--common-baits-min-msscore",
        type=float,
        default=0.0,
        help="Prey counts toward a bait only if composite MSscore for that bait–prey row exceeds this (default 0)",
    )
    args = ap.parse_args()

    mean_path = args.mean_freq_npy or (args.main_cluster_dir / "mean_edges_main_cluster.npy")

    out = run_one(
        args.main_cluster_dir,
        args.multi_seeds_dir,
        mean_path,
        args.model_data,
        args.out_json,
        args.out_tsv,
        saint_positive_thr=args.saint_positive_thr,
        n_perm=args.n_perm,
        perm_seed=args.perm_seed,
        saint_npy=args.saint_npy,
        composite_tsv=args.composite_tsv,
        no_derived_saint=bool(args.no_derived_saint),
        skip_common_baits=bool(args.skip_common_baits),
        common_baits_min_msscore=float(args.common_baits_min_msscore),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

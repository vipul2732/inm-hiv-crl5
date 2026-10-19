#!/usr/bin/env python3
"""
Publication-quality 4-panel PU validation figure
(KnownPositive = In_Any_Database) for WT / MOCK / VIF.

Panels:
1) Enrichment (top vs bottom quantiles)
2) Cumulative recovery vs rank
3) Evidence rate in top-N% predictions
4) Precision@K (PU; lower bound)

No panel letters. Minimal titles. Clean annotations.
"""

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Dataset name mapping for display
DATASET_DISPLAY_NAMES = {
    'WT': 'HIV WT infection',
    'MOCK': 'mock infection',
    'VIF': 'HIV Δ VIF infection'  # Using delta symbol
}


# ----------------------------
# Global publication style
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 16,
        "axes.titlesize": 22,  # panel titles (Enrichment, Cumulative recovery, …)
        "axes.labelsize": 18,  # label_fontsize = 18
        "xtick.labelsize": 16,  # tick_fontsize = 16
        "ytick.labelsize": 16,  # tick_fontsize = 16
        "legend.fontsize": 16,  # legend_fontsize = 16
        "axes.linewidth": 1.6,
        "lines.linewidth": 3.0,
        "lines.markersize": 8,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": True,
    })


def bold_axes(ax):
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def quantile_enrichment_ratio(
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    pseudocount: float = 0.5,
) -> float:
    """
    Fold enrichment of top vs bottom quantile with Haldane-style pseudocounts.

    rate = (k + pseudocount) / (n + 2 * pseudocount) per quantile; returns rate_top / rate_bot.
    """
    k_top = float(top["KnownPositive"].sum())
    k_bot = float(bottom["KnownPositive"].sum())
    n_top = len(top)
    n_bot = len(bottom)
    rate_top = (k_top + pseudocount) / (n_top + 2 * pseudocount)
    rate_bot = (k_bot + pseudocount) / (n_bot + 2 * pseudocount)
    return rate_top / rate_bot


def _apply_staggered_annotations(
    ax,
    annotations: List[dict],
    *,
    base_offset_y: float = 10,
    step_y: float = 20,
    fontsize: int = 14,
) -> None:
    """Vertically stagger labels that share the same x to reduce overlap."""
    groups: dict[float, list] = defaultdict(list)
    for ann in annotations:
        groups[float(ann["x"])].append(ann)

    for group in groups.values():
        group.sort(key=lambda a: a["y"])
        n = len(group)
        for i, ann in enumerate(group):
            dy = base_offset_y + (i - (n - 1) / 2.0) * step_y
            ax.annotate(
                ann["text"],
                (ann["x"], ann["y"]),
                textcoords="offset points",
                xytext=(ann.get("dx", 0), dy),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                color=ann["color"],
                clip_on=True,
            )


# Dataset-specific horizontal nudge (secondary to vertical stagger)
_DATASET_DX = {"WT": -6, "VIF": 0, "MOCK": 6}

# ----------------------------
# Data loading
# ----------------------------
def normalize_protein_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def _model_data_pkl(results_dir: Path, dataset_folder: str) -> Path:
    merged = results_dir / dataset_folder / "merged" / "merged_chains.pkl"
    with open(merged, "rb") as f:
        merged_data = pickle.load(f)
    seeds = merged_data["seeds"]
    seed0 = int(seeds[0])
    pkl = results_dir / dataset_folder / f"seed_{seed0}" / f"0_model23_n_{seed0}_model_data.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"model_data not found: {pkl}")
    return pkl


def build_pair_to_flat_index(model_data: dict) -> Dict[Tuple[str, str], int]:
    """Upper-tri order i<j matches mean_edges_cluster_1.npy."""
    node_idx2name = model_data.get("node_idx2name") or {}
    N = int(model_data["N"])
    out: Dict[Tuple[str, str], int] = {}
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            ni = normalize_protein_name(node_idx2name.get(i, f"node_{i}"))
            nj = normalize_protein_name(node_idx2name.get(j, f"node_{j}"))
            out[tuple(sorted((ni, nj)))] = k
            k += 1
    return out


def _attach_edge_frequency(
    df: pd.DataFrame,
    mean_edges_path: Path,
    model_data: dict,
    label: str,
) -> pd.DataFrame:
    vec = np.load(mean_edges_path).astype(float).ravel()
    pair_idx = build_pair_to_flat_index(model_data)
    if vec.size != len(pair_idx):
        raise ValueError(
            f"{label}: mean_edges len {vec.size} != expected pairs {len(pair_idx)}"
        )

    freqs = []
    missing = 0
    for _, row in df.iterrows():
        a = normalize_protein_name(row["Protein1"])
        b = normalize_protein_name(row["Protein2"])
        key = tuple(sorted((a, b)))
        idx = pair_idx.get(key)
        if idx is None:
            missing += 1
            freqs.append(np.nan)
        else:
            freqs.append(float(vec[idx]))
    df = df.copy()
    df["Edge_frequency"] = freqs
    if missing:
        logger.warning(f"{label}: {missing} TSV pairs not in model upper-tri map (dropped)")
    df = df.dropna(subset=["Edge_frequency"]).copy()
    df["Edge_frequency"] = df["Edge_frequency"].clip(0, 1)
    return df


def _comparison_tsv_path(input_dir: Path, dataset: str, comparison_suffix: Optional[str]) -> Path:
    if comparison_suffix:
        return input_dir / f"cullin_{dataset}_database_comparison_{comparison_suffix}.tsv"
    return input_dir / f"cullin_{dataset}_database_comparison.tsv"


def load_validation_data(
    input_dir: Path,
    results_dir: Optional[Path] = None,
    edge_frequency_npys: Optional[Dict[str, Path]] = None,
    comparison_suffix: Optional[str] = None,
    known_positive_column: str = "In_Any_Database",
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Returns (datasets, score_column_name).
    score_column_name is \"Edge_frequency\" when npys provided, else \"Probability\".
    """
    dataset_folder = {
        "wt": "cullin_wt_config_multi_seeds",
        "mock": "cullin_mock_config_multi_seeds",
        "vif": "cullin_vif_config_multi_seeds",
    }
    use_ef = edge_frequency_npys is not None and len(edge_frequency_npys) > 0
    if use_ef and results_dir is None:
        raise ValueError("results_dir required when using edge-frequency npy files")

    datasets: Dict[str, pd.DataFrame] = {}
    # Insertion order controls plot/legend order: WT, ΔVIF, mock (manuscript convention).
    for ds in ["wt", "vif", "mock"]:
        fp = _comparison_tsv_path(input_dir, ds, comparison_suffix)
        if not fp.exists():
            logger.warning(f"Missing: {fp}")
            continue

        df = pd.read_csv(fp, sep="\t")
        if known_positive_column not in df.columns:
            raise KeyError(f"{fp} missing column {known_positive_column!r}")
        df["KnownPositive"] = (
            df[known_positive_column].fillna(False).astype(bool).astype(int)
        )

        if use_ef:
            npy_path = edge_frequency_npys.get(ds)
            if npy_path is None or not npy_path.is_file():
                raise FileNotFoundError(
                    f"Edge-frequency mode requires a valid npy for {ds}: {npy_path}"
                )
            with open(_model_data_pkl(results_dir, dataset_folder[ds]), "rb") as f:
                model_data = pickle.load(f)
            df = _attach_edge_frequency(df, npy_path, model_data, ds.upper())
            logger.info(
                f"{ds.upper()}: N={len(df)} | Pos={df['KnownPositive'].sum()} "
                "(ranked by edge frequency map)"
            )
        else:
            df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
            df = df.dropna(subset=["Probability"]).copy()
            df["Probability"] = df["Probability"].clip(0, 1)
            logger.info(f"{ds.upper()}: N={len(df)} | Pos={df['KnownPositive'].sum()}")

        datasets[ds.upper()] = df

    score_col = "Edge_frequency" if use_ef else "Probability"
    return datasets, score_col


# ----------------------------
# Panel 1 — Enrichment
# ----------------------------
def plot_enrichment(ax, datasets, score_col: str = "Probability"):
    ax.set_title("Enrichment of known interactions")
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("Enrichment ratio")

    quantiles = [5, 10, 20, 30, 50]
    annotations: List[dict] = []
    ymax = 1.0

    for name, df in datasets.items():
        d = df.sort_values(score_col, ascending=False)
        N = len(d)

        xs, ys = [], []
        for q in quantiles:
            n = max(1, int(N * q / 100))
            top = d.iloc[:n]
            bot = d.iloc[-n:]

            y = quantile_enrichment_ratio(top, bot)
            xs.append(q)
            ys.append(y)
            ymax = max(ymax, y)

        display_name = DATASET_DISPLAY_NAMES.get(name, name)
        line = ax.plot(xs, ys, marker="o", label=display_name)
        c = line[0].get_color()

        for x, y in zip(xs, ys):
            annotations.append(
                {
                    "x": x,
                    "y": y,
                    "text": f"{y:.1f}×",
                    "color": c,
                    "dx": _DATASET_DX.get(name, 0),
                }
            )

    _apply_staggered_annotations(ax, annotations, base_offset_y=12, step_y=22, fontsize=14)

    ax.axhline(1.0, ls="--", lw=2, color="gray", alpha=0.6)
    ax.set_ylim(0, max(ymax * 1.18, 1.5))
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Panel 2 — Cumulative recovery
# ----------------------------
def plot_cumulative(ax, datasets, score_col: str = "Probability"):
    ax.set_title("Cumulative recovery")
    ax.set_xlabel("Top ranked pairs")
    ax.set_ylabel("Recovered known interactions")

    for name, df in datasets.items():
        d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        N = len(d)
        P = d["KnownPositive"].sum()

        x = np.arange(1, N + 1)
        y = d["KnownPositive"].cumsum()

        display_name = DATASET_DISPLAY_NAMES.get(name, name)
        line = ax.plot(x, y, label=display_name)
        c = line[0].get_color()
        ax.plot(x, x * (P / N), ls="--", lw=2, alpha=0.5, color=c)

    ax.legend(loc="lower right")
    bold_axes(ax)


# ----------------------------
# Panel 3 — Top-N% evidence rate (FIXED)
# ----------------------------
def plot_top_quantiles(ax, datasets, score_col: str = "Probability"):
    ax.set_title("Evidence rate in top predictions")
    ax.set_xlabel("Top N% of ranked predictions")
    ax.set_ylabel("Evidence rate (%)")

    quantiles = [5, 10, 20, 30, 50, 100]
    annotations: List[dict] = []

    for name, df in datasets.items():
        d = df.sort_values(score_col, ascending=False)
        N = len(d)

        xs, ys, counts = [], [], []
        for q in quantiles:
            n = max(1, int(N * q / 100))
            top = d.iloc[:n]
            xs.append(q)
            ys.append(top["KnownPositive"].mean() * 100)
            counts.append(top["KnownPositive"].sum())

        display_name = DATASET_DISPLAY_NAMES.get(name, name)
        line = ax.plot(xs, ys, marker="s", label=display_name)
        c = line[0].get_color()

        for q, y, cnt in zip(xs, ys, counts):
            if q in (10, 100):
                annotations.append(
                    {
                        "x": q,
                        "y": y,
                        "text": f"{cnt}/{int(N * q / 100)}",
                        "color": c,
                        "dx": _DATASET_DX.get(name, 0),
                    }
                )

        overall = d["KnownPositive"].mean() * 100
        ax.axhline(overall, ls=":", lw=2, alpha=0.4, color=c)

    _apply_staggered_annotations(ax, annotations, base_offset_y=14, step_y=26, fontsize=14)

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Panel 4 — Precision@K
# ----------------------------
def plot_precision(ax, datasets, score_col: str = "Probability"):
    ax.set_title("Precision@K (PU)")
    ax.set_xlabel("Top K predictions")
    ax.set_ylabel("Precision (%)")

    topks = [10, 25, 50, 100, 200, 500, 1000]
    annotations: List[dict] = []

    for name, df in datasets.items():
        d = df.sort_values(score_col, ascending=False)
        ks, ps, cs = [], [], []

        for k in topks:
            if k > len(d):
                continue
            sub = d.iloc[:k]
            cpos = sub["KnownPositive"].sum()
            ks.append(k)
            ps.append(cpos / k * 100)
            cs.append(cpos)

        display_name = DATASET_DISPLAY_NAMES.get(name, name)
        line = ax.plot(ks, ps, marker="D", label=display_name)
        c = line[0].get_color()

        max_k = max(ks) if ks else None
        for k, p, cpos in zip(ks, ps, cs):
            if k in (25, 100) or k == max_k:
                annotations.append(
                    {
                        "x": k,
                        "y": p,
                        "text": f"{cpos}/{k}",
                        "color": c,
                        "dx": _DATASET_DX.get(name, 0),
                    }
                )

        overall = d["KnownPositive"].mean() * 100
        ax.axhline(overall, ls=":", lw=2, alpha=0.4, color=c)

    _apply_staggered_annotations(ax, annotations, base_offset_y=12, step_y=24, fontsize=14)

    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.legend()
    bold_axes(ax)


# ----------------------------
# Main
# ----------------------------
PANEL_CHOICES = ("enrichment", "cumulative", "top_quantiles", "precision")

PANEL_FUNCS = {
    "enrichment": plot_enrichment,
    "cumulative": plot_cumulative,
    "top_quantiles": plot_top_quantiles,
    "precision": plot_precision,
}


def _parse_panels(raw: str) -> list[str]:
    panels = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not panels:
        raise ValueError("At least one panel required")
    bad = [p for p in panels if p not in PANEL_CHOICES]
    if bad:
        raise ValueError(f"Unknown panel(s): {bad}. Choose from {PANEL_CHOICES}")
    return panels


def _figure_layout(n_panels: int) -> tuple:
    if n_panels == 1:
        return (1, 1), (10, 8)
    if n_panels == 2:
        return (1, 2), (18, 7)
    if n_panels == 4:
        return (2, 2), (18, 13)
    raise ValueError(f"Unsupported panel count: {n_panels} (use 1, 2, or 4)")


def main():
    set_pub_style()

    ap = argparse.ArgumentParser(
        description="4-panel validation figure; optional cluster-1 edge frequency scores."
    )
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results root (required with --edge-frequency-* for model_data.pkl)",
    )
    ap.add_argument(
        "--edge-frequency-mock",
        type=Path,
        default=None,
        help="mean_edges_cluster_1.npy for mock (with wt+vif enables edge-frequency mode)",
    )
    ap.add_argument("--edge-frequency-wt", type=Path, default=None)
    ap.add_argument("--edge-frequency-vif", type=Path, default=None)
    ap.add_argument(
        "--suptitle",
        type=str,
        default=None,
        help="Optional figure suptitle (edge-frequency mode only; overrides default).",
    )
    ap.add_argument(
        "--no-suptitle",
        action="store_true",
        help="Edge-frequency mode: do not draw a figure suptitle (overrides --suptitle and default).",
    )
    ap.add_argument(
        "--comparison-suffix",
        type=str,
        default=None,
        help=(
            "Load cullin_{ds}_database_comparison_<suffix>.tsv instead of "
            "cullin_{ds}_database_comparison.tsv"
        ),
    )
    ap.add_argument(
        "--known-positive-column",
        type=str,
        default="In_Any_Database",
        help="TSV column used as KnownPositive label (default: In_Any_Database).",
    )
    ap.add_argument(
        "--panels",
        type=str,
        default="enrichment,cumulative,top_quantiles,precision",
        help=(
            "Comma-separated panels: enrichment, cumulative, top_quantiles, precision "
            "(default: all four)"
        ),
    )
    args = ap.parse_args()

    ef = [args.edge_frequency_mock, args.edge_frequency_wt, args.edge_frequency_vif]
    if any(ef) and not all(ef):
        raise SystemExit(
            "Provide all three --edge-frequency-{mock,wt,vif} or none."
        )
    edge_npys = None
    if all(ef):
        edge_npys = {
            "mock": args.edge_frequency_mock.resolve(),
            "wt": args.edge_frequency_wt.resolve(),
            "vif": args.edge_frequency_vif.resolve(),
        }
        if args.results_dir is None:
            raise SystemExit("--results-dir required when using edge-frequency npy files.")

    datasets, score_col = load_validation_data(
        args.input_dir,
        results_dir=args.results_dir,
        edge_frequency_npys=edge_npys,
        comparison_suffix=args.comparison_suffix,
        known_positive_column=args.known_positive_column,
    )
    if not datasets:
        raise SystemExit("No datasets loaded.")

    panels = _parse_panels(args.panels)
    grid_shape, figsize = _figure_layout(len(panels))
    fig, axes = plt.subplots(*grid_shape, figsize=figsize)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes_flat = axes.ravel()

    if score_col == "Edge_frequency" and not args.no_suptitle:
        if args.suptitle is not None and str(args.suptitle).strip():
            fig.suptitle(str(args.suptitle).strip(), fontsize=18, fontweight="bold", y=1.02)
        else:
            fig.suptitle(
                "Ranked by cluster-1 edge frequency (fraction of binary samples with edge on)",
                fontsize=18,
                fontweight="bold",
                y=1.02,
            )

    for ax, panel in zip(axes_flat, panels):
        PANEL_FUNCS[panel](ax, datasets, score_col)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved: {args.output} (score column: {score_col}, panels: {panels})")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Split-reference validation figures for PI layout:

Figure A (main, 2×2):
  [0,0] Enrichment — BioGRID direct + CORUM
  [0,1] Cumulative recovery — BioGRID direct + CORUM
  [1,0] Enrichment — PDB direct contact (BSA ≥ 400 Å²)
  [1,1] Cumulative recovery — PDB direct contact

Figure B (supplementary, 2×2):
  [0,0] Evidence rate in top predictions — BioGRID direct + CORUM
  [0,1] Precision@K — BioGRID direct + CORUM
  [1,0] Evidence rate in top predictions — PDB direct contact
  [1,1] Precision@K — PDB direct contact

Figure C (PDB expanded, 2×2):
  All four panels using expanded PDB structural evidence
  (direct + homolog direct + co-complex experimental/homolog).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validation_analysis import (  # noqa: E402
    bold_axes,
    plot_cumulative,
    plot_enrichment,
    plot_precision,
    plot_top_quantiles,
    set_pub_style,
    load_validation_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REF_BIOGRID_CORUM = "BioGRID direct + CORUM"
REF_PDB_DIRECT = "PDB direct contact (BSA ≥ 400 Å²)"
REF_PDB_EXPANDED = "PDB structural evidence"


def _set_panel_title(ax, base_title: str, reference: str) -> None:
    ax.set_title(f"{base_title}\n({reference})")
    bold_axes(ax)


def _load_pair(
    input_dir: Path,
    results_dir: Path,
    edge_npys: dict,
    comparison_suffix: str,
    known_positive_column: str,
):
    return load_validation_data(
        input_dir,
        results_dir=results_dir,
        edge_frequency_npys=edge_npys,
        comparison_suffix=comparison_suffix,
        known_positive_column=known_positive_column,
    )[0]


def figure_main_enrichment_recovery(
    datasets_bg: dict,
    datasets_pdb: dict,
    score_col: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    plot_enrichment(axes[0, 0], datasets_bg, score_col)
    _set_panel_title(axes[0, 0], "Enrichment of known interactions", REF_BIOGRID_CORUM)

    plot_cumulative(axes[0, 1], datasets_bg, score_col)
    _set_panel_title(axes[0, 1], "Cumulative recovery", REF_BIOGRID_CORUM)

    plot_enrichment(axes[1, 0], datasets_pdb, score_col)
    _set_panel_title(axes[1, 0], "Enrichment of known interactions", REF_PDB_DIRECT)

    plot_cumulative(axes[1, 1], datasets_pdb, score_col)
    _set_panel_title(axes[1, 1], "Cumulative recovery", REF_PDB_DIRECT)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


def figure_supp_evidence_precision(
    datasets_bg: dict,
    datasets_pdb: dict,
    score_col: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    plot_top_quantiles(axes[0, 0], datasets_bg, score_col)
    _set_panel_title(axes[0, 0], "Evidence rate in top predictions", REF_BIOGRID_CORUM)

    plot_precision(axes[0, 1], datasets_bg, score_col)
    _set_panel_title(axes[0, 1], "Precision@K (PU)", REF_BIOGRID_CORUM)

    plot_top_quantiles(axes[1, 0], datasets_pdb, score_col)
    _set_panel_title(axes[1, 0], "Evidence rate in top predictions", REF_PDB_DIRECT)

    plot_precision(axes[1, 1], datasets_pdb, score_col)
    _set_panel_title(axes[1, 1], "Precision@K (PU)", REF_PDB_DIRECT)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


def figure_pdb_expanded_full(
    datasets_exp: dict,
    score_col: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    plot_enrichment(axes[0, 0], datasets_exp, score_col)
    _set_panel_title(axes[0, 0], "Enrichment of known interactions", REF_PDB_EXPANDED)

    plot_cumulative(axes[0, 1], datasets_exp, score_col)
    _set_panel_title(axes[0, 1], "Cumulative recovery", REF_PDB_EXPANDED)

    plot_top_quantiles(axes[1, 0], datasets_exp, score_col)
    _set_panel_title(axes[1, 0], "Evidence rate in top predictions", REF_PDB_EXPANDED)

    plot_precision(axes[1, 1], datasets_exp, score_col)
    _set_panel_title(axes[1, 1], "Precision@K (PU)", REF_PDB_EXPANDED)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


def main() -> int:
    set_pub_style()
    ap = argparse.ArgumentParser(description="PI split-reference validation figures.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--edge-frequency-mock", type=Path, required=True)
    ap.add_argument("--edge-frequency-wt", type=Path, required=True)
    ap.add_argument("--edge-frequency-vif", type=Path, required=True)
    args = ap.parse_args()

    edge_npys = {
        "mock": args.edge_frequency_mock.resolve(),
        "wt": args.edge_frequency_wt.resolve(),
        "vif": args.edge_frequency_vif.resolve(),
    }

    datasets_bg = _load_pair(
        args.input_dir,
        args.results_dir,
        edge_npys,
        "direct_biogrid_corum",
        "In_Any_Database",
    )
    datasets_pdb = _load_pair(
        args.input_dir,
        args.results_dir,
        edge_npys,
        "pdb_direct_contact",
        "In_PDB_Direct_Contact",
    )
    datasets_exp = _load_pair(
        args.input_dir,
        args.results_dir,
        edge_npys,
        "pdb_structural_evidence",
        "In_PDB_Structural_Evidence",
    )
    score_col = "Edge_frequency"

    out = args.output_dir
    figure_main_enrichment_recovery(
        datasets_bg,
        datasets_pdb,
        score_col,
        out / "validation_analysis_4panel_split_main_cluster1_wilson_lb.png",
    )
    figure_supp_evidence_precision(
        datasets_bg,
        datasets_pdb,
        score_col,
        out / "validation_analysis_4panel_split_supp_cluster1_wilson_lb.png",
    )
    figure_pdb_expanded_full(
        datasets_exp,
        score_col,
        out / "validation_analysis_4panel_pdb_structural_evidence_cluster1_wilson_lb.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Generate biological subgraph figures at multiple mean edge-frequency cutoffs.

Curated biological views (same as main script), filenames include Louvain community ID:
  wt_community_1_vif_crl5_runx_module.png
  wt_community_0_csn_cul2_elongin_layer.png
  ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from community_binary_graph import (  # noqa: E402
    build_graph_from_main_cluster_edges_tsv,
    protein_label_for_plot,
)
from compare_community_detection import detect_communities_louvain  # noqa: E402
from plot_wt_vif_biological_subgraphs import (  # noqa: E402
    EDGE_N_TIERS,
    GROUP_BORDERS_CSN,
    GROUP_BORDERS_WT_VIF_MODULE,
    GROUP_COLORS_CSN,
    GROUP_COLORS_WT_VIF_MODULE,
    GROUP_ORDER_CSN,
    GROUP_ORDER_WT_VIF,
    MIN_NODE_SEPARATION,
    WT_CSN_CUL2_CURATED,
    _remove_obsolete_figures,
    classify_csn_layer,
    classify_wt_vif_module,
    community_stem,
    hif_context_nodes,
    hif_group,
    induced_or_warn,
    plot_and_export,
    plot_mock_subgraphs,
    runx_group,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CUTOFFS = (0.60, 0.62, 0.64, 0.66, 0.68)


def cutoff_dir_name(freq: float) -> str:
    return f"freq_ge_{freq:.2f}".replace(".", "p")


def community_of_anchor(node_to_comm: Dict[str, int], anchor: str) -> Optional[Set[str]]:
    anchor = protein_label_for_plot(anchor)
    if anchor not in node_to_comm:
        return None
    cid = node_to_comm[anchor]
    return {n for n, c in node_to_comm.items() if c == cid}


def anchor_comm_id(node_to_comm: Dict[str, int], anchor: str) -> Optional[int]:
    anchor = protein_label_for_plot(anchor)
    return node_to_comm.get(anchor)


def generate_at_cutoff(
    cutoff: float,
    out_dir: Path,
    wt_edges_tsv: Path,
    vif_edges_tsv: Path,
    mock_edges_tsv: Path,
    *,
    dpi: int,
    resolution: float,
    min_node_separation: float = MIN_NODE_SEPARATION,
) -> dict:
    logger.info("=== Cutoff mean frequency >= %.2f ===", cutoff)
    G_wt = build_graph_from_main_cluster_edges_tsv(wt_edges_tsv, cutoff)
    G_vif = build_graph_from_main_cluster_edges_tsv(vif_edges_tsv, cutoff)
    G_mock = build_graph_from_main_cluster_edges_tsv(mock_edges_tsv, cutoff)

    wt_map, wt_comms = detect_communities_louvain(G_wt, resolution=resolution)
    vif_map, vif_comms = detect_communities_louvain(G_vif, resolution=resolution)
    mock_map, mock_comms = detect_communities_louvain(G_mock, resolution=resolution)

    cyto_dir = out_dir / "cytoscape"
    edge_note = (
        f"Mean edge frequency ≥ {cutoff:.2f}; "
        f"edge width ∝ frequency quantile within subgraph ({EDGE_N_TIERS} tiers)"
    )
    freq_suffix = f"freq ≥ {cutoff:.2f}"

    stats = {
        "cutoff": cutoff,
        "wt_nodes": G_wt.number_of_nodes(),
        "wt_edges": G_wt.number_of_edges(),
        "wt_communities": len(wt_comms),
        "vif_nodes": G_vif.number_of_nodes(),
        "vif_edges": G_vif.number_of_edges(),
        "vif_communities": len(vif_comms),
        "mock_nodes": G_mock.number_of_nodes(),
        "mock_edges": G_mock.number_of_edges(),
        "mock_communities": len(mock_comms),
    }

    _remove_obsolete_figures(out_dir, cyto_dir)

    # WT — Vif / CRL5 module
    wt_module_nodes = community_of_anchor(wt_map, "VIFPROTEIN")
    wt_cid = anchor_comm_id(wt_map, "VIFPROTEIN")
    if wt_module_nodes is None:
        wt_module_nodes = community_of_anchor(wt_map, "CUL5")
        wt_cid = anchor_comm_id(wt_map, "CUL5")
    if wt_module_nodes and wt_cid is not None:
        H = induced_or_warn(G_wt, wt_module_nodes)
        plot_and_export(
            H,
            community_stem("wt", wt_cid, "vif_crl5_runx_module"),
            f"WT — Vif · CRL5 · RBX · CBFβ · RUNX (Louvain comm. {wt_cid}; {freq_suffix})",
            edge_note,
            out_dir / f"{community_stem('wt', wt_cid, 'vif_crl5_runx_module')}.png",
            cyto_dir,
            condition="WT",
            node_group_fn=classify_wt_vif_module,
            group_colors=GROUP_COLORS_WT_VIF_MODULE,
            group_borders=GROUP_BORDERS_WT_VIF_MODULE,
            group_order=GROUP_ORDER_WT_VIF,
            community_map=wt_map,
            dpi=dpi,
            min_node_separation=min_node_separation,
        )
        stats["wt_vif_module_comm"] = wt_cid
        stats["wt_vif_module_nodes"] = H.number_of_nodes()

    # WT — CSN curated (anchor CSN1 for community id)
    wt_csn_cid = anchor_comm_id(wt_map, "CSN1")
    if wt_csn_cid is None:
        wt_csn_cid = anchor_comm_id(wt_map, "CUL2") or 0
    H = induced_or_warn(G_wt, WT_CSN_CUL2_CURATED)
    if H.number_of_nodes() > 0:
        plot_and_export(
            H,
            community_stem("wt", wt_csn_cid, "csn_cul2_elongin_layer"),
            f"WT — CSN · CRL2 · elongin · PAF1 / adaptors (curated; comm. {wt_csn_cid}; {freq_suffix})",
            edge_note,
            out_dir / f"{community_stem('wt', wt_csn_cid, 'csn_cul2_elongin_layer')}.png",
            cyto_dir,
            condition="WT",
            node_group_fn=classify_csn_layer,
            group_colors=GROUP_COLORS_CSN,
            group_borders=GROUP_BORDERS_CSN,
            group_order=GROUP_ORDER_CSN,
            community_map=wt_map,
            dpi=dpi,
            min_node_separation=min_node_separation,
        )
        stats["wt_csn_curated_comm"] = wt_csn_cid

    # VIF KO — CRL5
    vif_crl5 = community_of_anchor(vif_map, "CUL5")
    vif_crl5_cid = anchor_comm_id(vif_map, "CUL5")
    if vif_crl5 and vif_crl5_cid is not None:
        H = induced_or_warn(G_vif, vif_crl5)
        plot_and_export(
            H,
            community_stem("vifko", vif_crl5_cid, "crl5_without_vif"),
            f"VIF KO — CRL5 · RBX · adaptors (Louvain comm. {vif_crl5_cid}; {freq_suffix})",
            edge_note,
            out_dir / f"{community_stem('vifko', vif_crl5_cid, 'crl5_without_vif')}.png",
            cyto_dir,
            condition="VIF_KO",
            node_group_fn=lambda n: classify_wt_vif_module(n) if n != "VIFPROTEIN" else "Vif",
            group_colors={**GROUP_COLORS_WT_VIF_MODULE, "Vif": "#F1F5F9"},
            group_borders=GROUP_BORDERS_WT_VIF_MODULE,
            group_order=[g for g in GROUP_ORDER_WT_VIF if g != "Vif"],
            community_map=vif_map,
            dpi=dpi,
            min_node_separation=min_node_separation,
        )
        stats["vif_crl5_comm"] = vif_crl5_cid

    # VIF KO — RUNX / CBFB
    vif_runx = community_of_anchor(vif_map, "CBFB")
    vif_runx_cid = anchor_comm_id(vif_map, "CBFB")
    if vif_runx and vif_runx_cid is not None:
        H = induced_or_warn(G_vif, vif_runx)
        rc = {"RUNX": "#FBCFE8", "CBFβ": "#BFDBFE", "BC11B": "#FECACA", "Other": "#E2E8F0"}
        rb = {"RUNX": "#DB2777", "CBFβ": "#2563EB", "BC11B": "#DC2626", "Other": "#64748B"}
        plot_and_export(
            H,
            community_stem("vifko", vif_runx_cid, "runx_cbfb_bc11b"),
            f"VIF KO — CBFβ · RUNX · BC11B (Louvain comm. {vif_runx_cid}; {freq_suffix})",
            edge_note,
            out_dir / f"{community_stem('vifko', vif_runx_cid, 'runx_cbfb_bc11b')}.png",
            cyto_dir,
            condition="VIF_KO",
            node_group_fn=runx_group,
            group_colors=rc,
            group_borders=rb,
            group_order=["BC11B", "CBFβ", "RUNX"],
            community_map=vif_map,
            figsize=(9, 8),
            dpi=dpi,
            min_node_separation=min_node_separation,
        )
        stats["vif_runx_comm"] = vif_runx_cid

    # VIF KO — HIF1A context
    hif_comm = community_of_anchor(vif_map, "HIF1A")
    hif_cid = anchor_comm_id(vif_map, "HIF1A")
    if hif_comm and hif_cid is not None:
        nodes_hif = hif_context_nodes(G_vif, hif_comm)
        H = induced_or_warn(G_vif, nodes_hif)
        if H.number_of_nodes() > 0:
            hc = {
                "HIF1A": "#FECACA",
                "Elongin B/C": "#BFDBFE",
                "COP9 (CSN)": "#A7F3D0",
                "Neighbors (comm. 0)": "#F1F5F9",
            }
            hb = {
                "HIF1A": "#DC2626",
                "Elongin B/C": "#2563EB",
                "COP9 (CSN)": "#059669",
                "Neighbors (comm. 0)": "#94A3B8",
            }
            plot_and_export(
                H,
                community_stem("vifko", hif_cid, "hif1a_csn_elongin_context"),
                f"VIF KO — HIF1A neighborhood (Louvain comm. {hif_cid}; {freq_suffix})",
                f"HIF1A ego within its Louvain community. {edge_note}",
                out_dir / f"{community_stem('vifko', hif_cid, 'hif1a_csn_elongin_context')}.png",
                cyto_dir,
                condition="VIF_KO",
                node_group_fn=hif_group,
                group_colors=hc,
                group_borders=hb,
                group_order=["HIF1A", "COP9 (CSN)", "Elongin B/C", "Neighbors (comm. 0)"],
                community_map=vif_map,
                figsize=(12, 10),
                dpi=dpi,
                min_node_separation=min_node_separation,
            )
            stats["vif_hif_comm"] = hif_cid

    mock_crl5 = community_of_anchor(mock_map, "CUL5")
    mock_crl5_cid = anchor_comm_id(mock_map, "CUL5")
    mock_csn_core = community_of_anchor(mock_map, "CSN1")
    mock_csn_cid = anchor_comm_id(mock_map, "CSN1")
    if mock_csn_core is None:
        mock_csn_core = community_of_anchor(mock_map, "ELOB")
        mock_csn_cid = anchor_comm_id(mock_map, "ELOB")
    mock_hif = community_of_anchor(mock_map, "HIF1A")
    mock_hif_cid = anchor_comm_id(mock_map, "HIF1A")
    mock_runx = community_of_anchor(mock_map, "CBFB")
    mock_runx_cid = anchor_comm_id(mock_map, "CBFB")
    if mock_runx is None:
        mock_runx = community_of_anchor(mock_map, "PEBB")
        mock_runx_cid = anchor_comm_id(mock_map, "PEBB")

    mock_csn_curated_cid = mock_hif_cid if mock_hif_cid is not None else 0

    plot_mock_subgraphs(
        G_mock,
        out_dir,
        cyto_dir,
        crl5_nodes=mock_crl5,
        crl5_comm_id=mock_crl5_cid if mock_crl5_cid is not None else 1,
        csn_elongin_nodes=mock_csn_core,
        csn_elongin_comm_id=mock_csn_cid if mock_csn_cid is not None else 2,
        hif_comm_nodes=mock_hif,
        hif_comm_id=mock_hif_cid if mock_hif_cid is not None else 0,
        runx_nodes=mock_runx,
        runx_comm_id=mock_runx_cid,
        community_map=mock_map,
        edge_subtitle=edge_note,
        dpi=dpi,
        min_node_separation=min_node_separation,
        title_suffix=freq_suffix,
        fig02_comm_id=mock_csn_curated_cid,
    )

    part_dir = out_dir / "communities"
    part_dir.mkdir(parents=True, exist_ok=True)
    for label, nmap in (("wt", wt_map), ("vif_ko", vif_map), ("mock", mock_map)):
        rows = [{"Protein": p, "Community_ID": c} for p, c in sorted(nmap.items())]
        if rows:
            pd.DataFrame(rows).to_csv(part_dir / f"{label}_louvain.tsv", sep="\t", index=False)

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Biological subgraph cutoff sweep (curated views + comm ID in filename).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "wt_vif_comparison" / "subgraphs_biological" / "cutoff_sweep",
    )
    ap.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Mean edge frequency thresholds (default: 0.60 0.62 0.64 0.66 0.68)",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution")
    args = ap.parse_args()

    wt_edges = (
        ROOT
        / "post_sampling_analysis/base_models/cullin_wt_config_multi_seeds/main_cluster_network/main_cluster_edges.tsv"
    )
    vif_edges = (
        ROOT
        / "post_sampling_analysis/base_models/cullin_vif_config_multi_seeds/main_cluster_network/main_cluster_edges.tsv"
    )
    mock_edges = (
        ROOT
        / "post_sampling_analysis/base_models/cullin_mock_config_multi_seeds/main_cluster_network/main_cluster_edges.tsv"
    )

    all_stats: List[dict] = []
    for cutoff in sorted(args.cutoffs):
        sub = args.out_dir / cutoff_dir_name(cutoff)
        sub.mkdir(parents=True, exist_ok=True)
        all_stats.append(
            generate_at_cutoff(
                cutoff,
                sub,
                wt_edges,
                vif_edges,
                mock_edges,
                dpi=args.dpi,
                resolution=args.resolution,
            )
        )

    summary = pd.DataFrame(all_stats)
    summary_path = args.out_dir / "cutoff_sweep_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    logger.info("Wrote summary: %s", summary_path)
    logger.info("Done. Cutoff sweep in %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

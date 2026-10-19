#!/usr/bin/env python3
"""
Focused subgraph figures: curated biological views per Louvain community (WT, VIF KO, Mock).

Filenames include community ID, e.g.:
  wt_community_1_vif_crl5_runx_module.png
  wt_community_0_csn_cul2_elongin_layer.png
  vifko_community_2_runx_cbfb_bc11b.png
Also writes Cytoscape tables under subgraphs_biological/cytoscape/.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from community_binary_graph import (  # noqa: E402
    build_graph_from_main_cluster_binary,
    build_graph_from_main_cluster_edges_tsv,
    default_model_data_pkl,
    load_model_data,
    node_display_label,
    protein_label_for_plot,
    CBFB_GROUP_LABEL,
    CBFB_RUNX_GROUP_LABEL,
)

ROOT = Path(__file__).resolve().parents[1]

WT_CSN_CUL2_CURATED: Tuple[str, ...] = (
    "CSN1",
    "CSN3",
    "CSN4",
    "CSN5",
    "CSN7A",
    "CSN7B",
    "CSN8",
    "CUL2",
    "ELOB",
    "ELOC",
    "DCA11",
    "PDC6I",
    "AMRA1",
    "SOCS2",
    "SOCS6",
    "SPSB3",
    "ASB8",
    "ASB9",
    "WSB1",
    "PAF1",
    "CTR9",
    "LEO1",
    "WDR61",
    "FEM1B",
    "FEM1C",
    "NRBP",
    "NRBP2",
)

# Light fills (readable black labels) + matching darker borders.
GROUP_COLORS_WT_VIF_MODULE: Dict[str, str] = {
    "Vif": "#FECACA",
    "CRL5 / RBX / neddylation": "#DDD6FE",
    "Elongin B/C": "#BFDBFE",
    "CBFβ / RUNX": "#FBCFE8",
    "ASB / other": "#E2E8F0",
}

GROUP_BORDERS_WT_VIF_MODULE: Dict[str, str] = {
    "Vif": "#DC2626",
    "CRL5 / RBX / neddylation": "#7C3AED",
    "Elongin B/C": "#2563EB",
    "CBFβ / RUNX": "#DB2777",
    "ASB / other": "#64748B",
}

GROUP_COLORS_CSN: Dict[str, str] = {
    "COP9 (CSN)": "#A7F3D0",
    "CRL2 + elongin": "#BFDBFE",
    "DCA11 / PDC6I": "#FDE68A",
    "Adaptor (SOCS/ASB/WSB)": "#E2E8F0",
    "PAF1 module": "#FED7AA",
    "Other": "#F1F5F9",
}

GROUP_BORDERS_CSN: Dict[str, str] = {
    "COP9 (CSN)": "#059669",
    "CRL2 + elongin": "#2563EB",
    "DCA11 / PDC6I": "#CA8A04",
    "Adaptor (SOCS/ASB/WSB)": "#64748B",
    "PAF1 module": "#EA580C",
    "Other": "#94A3B8",
}

NODE_FILL_ALPHA = 0.82
# Minimum center-to-center distance when resolving label overlap (within each biological group).
MIN_NODE_SEPARATION = 0.36

GROUP_ORDER_WT_VIF = [
    "Vif",
    "CRL5 / RBX / neddylation",
    "Elongin B/C",
    "CBFβ / RUNX",
    "ASB / other",
]

GROUP_ORDER_CSN = [
    "COP9 (CSN)",
    "CRL2 + elongin",
    "DCA11 / PDC6I",
    "PAF1 module",
    "Adaptor (SOCS/ASB/WSB)",
    "Other",
]

COMM_BORDER_COLORS: Dict[int, str] = {
    0: "#1E293B",
    1: "#0F766E",
    2: "#9333EA",
}


def load_communities_tsv(path: Path) -> Dict[int, Set[str]]:
    df = pd.read_csv(path, sep="\t")
    out: Dict[int, Set[str]] = defaultdict(set)
    for _, r in df.iterrows():
        out[int(r["Community_ID"])].add(protein_label_for_plot(str(r["Protein"])))
    return dict(out)


def load_protein_community(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path, sep="\t")
    return {
        protein_label_for_plot(str(r["Protein"])): int(r["Community_ID"])
        for _, r in df.iterrows()
    }


def load_edge_attributes(edges_tsv: Path) -> Dict[Tuple[str, str], dict]:
    df = pd.read_csv(edges_tsv, sep="\t")
    out: Dict[Tuple[str, str], dict] = {}
    for _, r in df.iterrows():
        if int(r["binary_ge0p6"]) != 1:
            continue
        a = protein_label_for_plot(str(r["protein1"]))
        b = protein_label_for_plot(str(r["protein2"]))
        key = tuple(sorted((a, b)))
        out[key] = {
            "frequency": float(r["frequency"]),
            "wilson_lb": float(r["wilson_lb"]),
        }
    return out


def annotate_graph_edges(G: nx.Graph, edge_attrs: Dict[Tuple[str, str], dict]) -> None:
    for u, v, data in G.edges(data=True):
        key = tuple(sorted((u, v)))
        info = edge_attrs.get(key, {"frequency": 0.6, "wilson_lb": 0.6})
        data["frequency"] = info["frequency"]
        data["wilson_lb"] = info["wilson_lb"]


def classify_biological_group(n: str) -> str:
    """Unified functional coloring for full-community subgraphs."""
    u = n.upper()
    if u == "VIFPROTEIN":
        return "Vif"
    if u == "HIF1A":
        return "HIF1A"
    if u == "BC11B":
        return "BC11B"
    if u in {"CBFB", "PEBB"}:
        return CBFB_GROUP_LABEL
    if u.startswith("RUNX"):
        return "RUNX"
    if u.startswith("CSN"):
        return "COP9 (CSN)"
    if u in {"CUL2", "ELOB", "ELOC"}:
        return "CRL2 + elongin"
    if u in {
        "CUL5",
        "RBX1",
        "RBX2",
        "NEDD8",
        "APC11",
        "CAND1",
        "DCNL1",
        "KAT3",
        "INP4A",
        "TULP4",
        "PCMD1",
        "PCMD2",
        "RB40C",
    }:
        return "CRL5 / RBX / neddylation"
    if u in {"DCA11", "PDC6I"}:
        return "DCA11 / PDC6I"
    if u in {"PAF1", "CTR9", "LEO1", "WDR61"}:
        return "PAF1 module"
    if (
        u.startswith("SOCS")
        or u.startswith("ASB")
        or u.startswith("WSB")
        or u.startswith("SPSB")
        or u in {"AMRA1", "FEM1B", "FEM1C", "NRBP", "NRBP2"}
    ):
        return "Adaptor (SOCS/ASB/WSB)"
    if u in {"SGT1", "LLR1"}:
        return "RBX / SGT1 / LLR1"
    return "Other"


GROUP_COLORS_BIOLOGICAL: Dict[str, str] = {
    "Vif": "#FECACA",
    "HIF1A": "#FECACA",
    "BC11B": "#FECACA",
    "CBFβ": "#BFDBFE",
    "RUNX": "#FBCFE8",
    "COP9 (CSN)": "#A7F3D0",
    "CRL2 + elongin": "#BFDBFE",
    "CRL5 / RBX / neddylation": "#DDD6FE",
    "DCA11 / PDC6I": "#FDE68A",
    "PAF1 module": "#FED7AA",
    "Adaptor (SOCS/ASB/WSB)": "#E2E8F0",
    "RBX / SGT1 / LLR1": "#DDD6FE",
    "Other": "#F1F5F9",
}

GROUP_BORDERS_BIOLOGICAL: Dict[str, str] = {
    "Vif": "#DC2626",
    "HIF1A": "#DC2626",
    "BC11B": "#DC2626",
    "CBFβ": "#2563EB",
    "RUNX": "#DB2777",
    "COP9 (CSN)": "#059669",
    "CRL2 + elongin": "#2563EB",
    "CRL5 / RBX / neddylation": "#7C3AED",
    "DCA11 / PDC6I": "#CA8A04",
    "PAF1 module": "#EA580C",
    "Adaptor (SOCS/ASB/WSB)": "#64748B",
    "RBX / SGT1 / LLR1": "#7C3AED",
    "Other": "#94A3B8",
}

GROUP_ORDER_BIOLOGICAL = [
    "Vif",
    "HIF1A",
    "BC11B",
    "CBFβ",
    "RUNX",
    "CRL5 / RBX / neddylation",
    "CRL2 + elongin",
    "COP9 (CSN)",
    "RBX / SGT1 / LLR1",
    "PAF1 module",
    "DCA11 / PDC6I",
    "Adaptor (SOCS/ASB/WSB)",
    "Other",
]

# Legacy thematic classifiers (kept for reference / imports)
def classify_wt_vif_module(n: str) -> str:
    u = n.upper()
    if u == "VIFPROTEIN":
        return "Vif"
    if u in {
        "CUL5",
        "RBX1",
        "RBX2",
        "NEDD8",
        "APC11",
        "CAND1",
        "DCNL1",
        "KAT3",
        "INP4A",
        "TULP4",
        "PCMD1",
        "PCMD2",
        "RB40C",
    }:
        return "CRL5 / RBX / neddylation"
    if u in {"ELOB", "ELOC"}:
        return "Elongin B/C"
    if u in {"CBFB", "RUNX1", "RUNX2", "RUNX3"}:
        return CBFB_RUNX_GROUP_LABEL
    if u.startswith("ASB") or u.startswith("SOCS"):
        return "ASB / other"
    return "ASB / other"


def classify_csn_layer(n: str) -> str:
    u = n.upper()
    if u.startswith("CSN"):
        return "COP9 (CSN)"
    if u in {"CUL2", "ELOB", "ELOC"}:
        return "CRL2 + elongin"
    if u in {"DCA11", "PDC6I"}:
        return "DCA11 / PDC6I"
    if (
        u.startswith("SOCS")
        or u.startswith("ASB")
        or u.startswith("WSB")
        or u.startswith("SPSB")
        or u in {"AMRA1", "FEM1B", "FEM1C", "NRBP", "NRBP2"}
    ):
        return "Adaptor (SOCS/ASB/WSB)"
    if u in {"PAF1", "CTR9", "LEO1", "WDR61"}:
        return "PAF1 module"
    return "Other"


def classify_csn_elongin_core(n: str) -> str:
    u = n.upper()
    if u.startswith("CSN"):
        return "COP9 (CSN)"
    if u in {"ELOB", "ELOC"}:
        return "Elongin B/C"
    if u in {"RBX1", "SGT1", "LLR1"}:
        return "RBX / SGT1 / LLR1"
    return "Other"


GROUP_COLORS_CSN_ELONGIN_CORE: Dict[str, str] = {
    "COP9 (CSN)": "#A7F3D0",
    "Elongin B/C": "#BFDBFE",
    "RBX / SGT1 / LLR1": "#DDD6FE",
    "Other": "#F1F5F9",
}

GROUP_BORDERS_CSN_ELONGIN_CORE: Dict[str, str] = {
    "COP9 (CSN)": "#059669",
    "Elongin B/C": "#2563EB",
    "RBX / SGT1 / LLR1": "#7C3AED",
    "Other": "#94A3B8",
}

GROUP_ORDER_CSN_ELONGIN_CORE = [
    "COP9 (CSN)",
    "Elongin B/C",
    "RBX / SGT1 / LLR1",
    "Other",
]


def runx_group(n: str) -> str:
    u = n.upper()
    if u == "BC11B":
        return "BC11B"
    if u.startswith("RUNX"):
        return "RUNX"
    if u in {"CBFB", "PEBB"}:
        return CBFB_GROUP_LABEL
    return "Other"


def hif_group(n: str) -> str:
    u = n.upper()
    if u == "HIF1A":
        return "HIF1A"
    if u in {"ELOB", "ELOC"}:
        return "Elongin B/C"
    if u.startswith("CSN"):
        return "COP9 (CSN)"
    return "Neighbors (comm. 0)"


def community_of_anchor(node_to_comm: Dict[str, int], anchor: str) -> Optional[Set[str]]:
    anchor = protein_label_for_plot(anchor)
    if anchor not in node_to_comm:
        return None
    cid = node_to_comm[anchor]
    return {n for n, c in node_to_comm.items() if c == cid}


def induced_or_warn(G: nx.Graph, nodes: Iterable[str]) -> nx.Graph:
    present = [n for n in nodes if n in G]
    missing = sorted(set(nodes) - set(present))
    if missing:
        print(f"  (skip {len(missing)} nodes not in graph: {missing[:8]}{'…' if len(missing) > 8 else ''})")
    return G.subgraph(present).copy()


def _group_centers(n_groups: int, group_spacing: float) -> List[np.ndarray]:
    """Place cluster centers; two groups side-by-side to avoid tall vertical gaps."""
    if n_groups <= 0:
        return []
    if n_groups == 1:
        return [np.array([0.0, 0.0])]
    if n_groups == 2:
        d = group_spacing * 0.52
        return [np.array([-d, 0.0]), np.array([d, 0.0])]
    centers: List[np.ndarray] = []
    for i in range(n_groups):
        angle = 2 * np.pi * i / n_groups - np.pi / 2
        centers.append(group_spacing * np.array([np.cos(angle), np.sin(angle)]))
    return centers


def _compact_group_positions(
    sub: nx.Graph,
    nodes: List[str],
    center: np.ndarray,
    *,
    intra_radius: float,
    seed: int,
    k: float,
    prefer_circular: bool = False,
    min_separation: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Spring (or circular) layout within one biological group — keeps category nodes clustered."""
    n = len(nodes)
    if prefer_circular and n >= 8:
        sub_pos = nx.circular_layout(sub, scale=1.0)
        pts = np.array([sub_pos[node] for node in nodes])
        pts -= pts.mean(axis=0)
        span = max(float(np.max(np.abs(pts))), 0.15)
        if min_separation is not None and n >= 2:
            target = max(min_separation * 1.38, 2.9 * intra_radius / n)
            ring_r = target / (2.0 * np.sin(np.pi / n))
            ring_r = min(max(ring_r, intra_radius * 0.70), intra_radius * 1.55)
        else:
            ring_r = intra_radius
        pts = pts / span * ring_r
    else:
        sub_pos = nx.spring_layout(sub, seed=seed, k=k, iterations=200, weight="frequency")
        pts = np.array([sub_pos[n] for n in nodes])
        pts -= pts.mean(axis=0)
        span = max(float(np.max(np.abs(pts))), 0.15)
        pts = pts / span * intra_radius
    return {node: np.array(p, dtype=float) + center for node, p in zip(nodes, pts)}


def _separate_overlapping_only(
    pos: Dict[str, np.ndarray],
    nodes: List[str],
    center: np.ndarray,
    *,
    min_dist: float,
    max_iterations: int = 50,
) -> None:
    """Nudge overlapping nodes apart; re-center without expanding the group blob."""
    if len(nodes) < 2:
        return
    for _ in range(max_iterations):
        moved = False
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                delta = pos[b] - pos[a]
                dist = float(np.linalg.norm(delta))
                if dist >= min_dist:
                    continue
                if dist < 1e-8:
                    delta = np.array([1.0, 0.0])
                    dist = 1.0
                push = (min_dist - dist) * 0.45
                direction = delta / dist
                pos[a] -= direction * push
                pos[b] += direction * push
                moved = True
        if not moved:
            break
    pts = np.array([pos[n] for n in nodes])
    shift = center - pts.mean(axis=0)
    for n in nodes:
        pos[n] = pos[n] + shift


def _resolve_group_overlaps(
    local: Dict[str, np.ndarray],
    nodes: List[str],
    center: np.ndarray,
    *,
    intra_radius: float,
    min_node_separation: float,
    use_circular: bool = False,
) -> None:
    """Separate overlapping nodes inside one biological group; keep the cluster cohesive."""
    n = len(nodes)
    if n < 2 or use_circular:
        return
    pack_dist = (2.35 + 0.07 * max(n - 6, 0)) * intra_radius / n
    eff_min = min(min_node_separation, max(pack_dist, 0.13))
    expand = min(1.16, 1.05 + 0.018 * max(n - 5, 0))
    contract = max(0.90, 0.95 - 0.005 * max(n - 6, 0))
    for node in nodes:
        local[node] = center + expand * (local[node] - center)
    _separate_overlapping_only(
        local, nodes, center, min_dist=eff_min, max_iterations=55
    )
    for node in nodes:
        local[node] = center + contract * (local[node] - center)
    _separate_overlapping_only(
        local, nodes, center, min_dist=eff_min * 0.93, max_iterations=40
    )


def grouped_layout(
    G: nx.Graph,
    node_group_fn: Callable[[str], str],
    *,
    group_order: Optional[List[str]] = None,
    seed: int = 42,
    group_spacing: float = 2.4,
    min_node_separation: Optional[float] = None,
) -> dict:
    """Place biological groups in separate regions; spring-layout within each group."""
    by_group: Dict[str, List[str]] = defaultdict(list)
    for n in G.nodes():
        by_group[node_group_fn(n)].append(n)

    if group_order is None:
        group_order = sorted(by_group.keys(), key=lambda g: (-len(by_group[g]), g))
    else:
        group_order = [g for g in group_order if g in by_group] + sorted(
            g for g in by_group if g not in group_order
        )

    n_groups = len(group_order)
    centers_list = _group_centers(n_groups, group_spacing)
    centers: Dict[str, np.ndarray] = {
        g: centers_list[i] for i, g in enumerate(group_order)
    }

    pos: dict = {}
    separate = min_node_separation is not None
    for i, g in enumerate(group_order):
        nodes = by_group[g]
        if not nodes:
            continue
        center = centers[g]
        if len(nodes) == 1:
            pos[nodes[0]] = center.copy()
            continue
        n = len(nodes)
        intra_radius = group_spacing * (0.30 if n_groups <= 2 else 0.36)
        if separate and n >= 7:
            intra_radius *= 1.0 + 0.035 * min(n - 6, 8)
        k = 1.1 / max(np.sqrt(n), 1)
        sub = G.subgraph(nodes)
        prefer_circular = separate and n >= 7
        if separate:
            local = _compact_group_positions(
                sub,
                nodes,
                center,
                intra_radius=intra_radius,
                seed=seed + i,
                k=k,
                prefer_circular=prefer_circular,
                min_separation=min_node_separation,
            )
            _resolve_group_overlaps(
                local,
                nodes,
                center,
                intra_radius=intra_radius,
                min_node_separation=min_node_separation,
                use_circular=prefer_circular,
            )
            pos.update(local)
        else:
            local = _compact_group_positions(
                sub, nodes, center, intra_radius=intra_radius, seed=seed + i, k=k
            )
            pos.update(local)

    return pos


def _fit_square_limits(ax, pos: dict, *, pad: float = 1.35) -> None:
    """Center plot with square limits so large nodes are not clipped."""
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    cx = float(xs.mean())
    cy = float(ys.mean())
    half = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2.0 + pad
    half = max(half, 2.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


# Edge display: quantile tiers within each subgraph (spreads narrow frequency ranges).
EDGE_N_TIERS = 4
EDGE_WIDTH_BY_TIER = (0.35, 1.1, 2.0, 3.2)
EDGE_ALPHA_BY_TIER = (0.10, 0.20, 0.36, 0.52)


def _edge_strength_tiers(G: nx.Graph, *, n_tiers: int = EDGE_N_TIERS) -> Dict[Tuple[str, str], int]:
    """
    Assign each edge a tier 0..n_tiers-1 from frequency quantiles within this subgraph.
    Makes weak vs strong edges visually separable even when raw frequencies cluster (e.g. 0.68–0.72).
    """
    items = [(u, v, float(d.get("frequency", 0.6))) for u, v, d in G.edges(data=True)]
    if not items:
        return {}

    freqs = np.array([f for _, _, f in items], dtype=float)
    if float(np.ptp(freqs)) < 1e-12:
        mid = n_tiers // 2
        return {(u, v): mid for u, v, _ in items}

    # np.digitize with interior quantile breakpoints
    if len(np.unique(freqs)) < n_tiers:
        n_tiers = max(2, len(np.unique(freqs)))
    breaks = np.unique(np.quantile(freqs, np.linspace(0, 1, n_tiers + 1)[1:-1]))

    out: Dict[Tuple[str, str], int] = {}
    for u, v, f in items:
        tier = int(np.digitize(f, breaks, right=True))
        tier = min(tier, n_tiers - 1)
        out[(u, v)] = tier
        out[(v, u)] = tier
    return out


def _draw_weighted_edges(
    G: nx.Graph,
    pos: dict,
    ax,
    *,
    n_tiers: int = EDGE_N_TIERS,
) -> None:
    tier_map = _edge_strength_tiers(G, n_tiers=n_tiers)
    n_used = max(tier_map.values()) + 1 if tier_map else n_tiers
    widths = np.linspace(EDGE_WIDTH_BY_TIER[0], EDGE_WIDTH_BY_TIER[-1], n_used)
    alphas = np.linspace(EDGE_ALPHA_BY_TIER[0], EDGE_ALPHA_BY_TIER[-1], n_used)

    for u, v in G.edges():
        tier = tier_map.get((u, v), 0)
        width = float(widths[tier])
        alpha = float(alphas[tier])
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#64748B",
            linewidth=width,
            alpha=alpha,
            solid_capstyle="round",
            zorder=1,
        )


def _draw_grouped_nodes(
    G: nx.Graph,
    pos: dict,
    ax,
    *,
    node_group_fn: Callable[[str], str],
    group_colors: Dict[str, str],
    group_borders: Optional[Dict[str, str]] = None,
    node_size: float,
) -> None:
    for n in G.nodes():
        g = node_group_fn(n)
        fill = group_colors.get(g, "#F1F5F9")
        border = (group_borders or {}).get(g, "#475569")
        x, y = pos[n]
        ax.scatter(
            x,
            y,
            s=node_size,
            c=fill,
            alpha=NODE_FILL_ALPHA,
            edgecolors=border,
            linewidths=2.4,
            zorder=3,
        )


def plot_focus_subgraph(
    G: nx.Graph,
    title: str,
    subtitle: str,
    out: Path,
    *,
    node_group_fn: Callable[[str], str],
    group_colors: Dict[str, str],
    group_order: Optional[List[str]] = None,
    group_borders: Optional[Dict[str, str]] = None,
    community_map: Optional[Dict[str, int]] = None,
    figsize: Tuple[float, float] = (11, 10),
    dpi: int = 300,
    seed: int = 42,
    min_node_separation: Optional[float] = None,
):
    if G.number_of_nodes() == 0:
        print(f"Skip empty graph: {title}")
        return

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    pos = grouped_layout(
        G,
        node_group_fn,
        group_order=group_order,
        seed=seed,
        min_node_separation=min_node_separation,
    )
    node_size = max(1200, 6500 // max(G.number_of_nodes(), 1))

    _draw_weighted_edges(G, pos, ax)
    _draw_grouped_nodes(
        G,
        pos,
        ax,
        node_group_fn=node_group_fn,
        group_colors=group_colors,
        group_borders=group_borders,
        node_size=node_size,
    )

    fs = max(6.5, min(9.5, 200 / max(G.number_of_nodes(), 1)))
    for n, (x, y) in pos.items():
        ax.text(
            x,
            y,
            node_display_label(n),
            fontsize=fs,
            fontweight="bold",
            ha="center",
            va="center",
            color="#0F172A",
            zorder=4,
        )

    seen: List[str] = []
    handles = []
    for g in (group_order or []) + sorted(set(node_group_fn(n) for n in G.nodes())):
        if g in seen or g not in {node_group_fn(n) for n in G.nodes()}:
            continue
        seen.append(g)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=group_colors.get(g, "#F1F5F9"),
                markersize=10,
                markeredgecolor=(group_borders or {}).get(g, "#475569"),
                markeredgewidth=1.5,
                linestyle="None",
                label=g,
            )
            )

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=8.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12, color="#0F172A")
    fig.suptitle(subtitle, fontsize=10.5, color="#475569", y=0.02, ha="center")
    _fit_square_limits(ax, pos, pad=1.62 if min_node_separation else 1.5)
    ax.axis("off")
    fig.subplots_adjust(left=0.06, right=0.78, top=0.94, bottom=0.07)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {out}")


def export_cytoscape_tables(
    G: nx.Graph,
    stem: str,
    out_dir: Path,
    *,
    node_group_fn: Callable[[str], str],
    group_colors: Dict[str, str],
    group_borders: Optional[Dict[str, str]] = None,
    community_map: Optional[Dict[str, int]] = None,
    condition: str,
) -> None:
    """Write node/edge TSVs for Cytoscape import (File → Import → Network from File)."""
    if G.number_of_nodes() == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    node_rows = []
    for n in sorted(G.nodes()):
        g = node_group_fn(n)
        node_rows.append(
            {
                "id": n,
                "label": node_display_label(n),
                "biological_group": g,
                "louvain_comm": community_map.get(n, "") if community_map else "",
                "fill_color": group_colors.get(g, "#F1F5F9"),
                "border_color": (group_borders or {}).get(g, "#475569"),
                "degree": G.degree(n),
                "condition": condition,
            }
        )
    pd.DataFrame(node_rows).to_csv(out_dir / f"{stem}_nodes.tsv", sep="\t", index=False)

    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append(
            {
                "source": u,
                "target": v,
                "frequency": data.get("frequency"),
                "wilson_lb": data.get("wilson_lb"),
                "interaction": f"{u}_{v}",
            }
        )
    edge_df = pd.DataFrame(edge_rows)
    if not edge_df.empty:
        edge_df = edge_df.sort_values(["source", "target"])
    edge_df.to_csv(out_dir / f"{stem}_edges.tsv", sep="\t", index=False)
    print(f"Saved Cytoscape tables: {out_dir}/{stem}_nodes.tsv, {stem}_edges.tsv")


def plot_and_export(
    G: nx.Graph,
    stem: str,
    title: str,
    subtitle: str,
    png_out: Path,
    cyto_dir: Path,
    *,
    condition: str,
    node_group_fn: Callable[[str], str],
    group_colors: Dict[str, str],
    group_borders: Optional[Dict[str, str]] = None,
    group_order: Optional[List[str]] = None,
    community_map: Optional[Dict[str, int]] = None,
    figsize: Tuple[float, float] = (11, 10),
    dpi: int = 300,
    min_node_separation: Optional[float] = None,
) -> None:
    plot_focus_subgraph(
        G,
        title,
        subtitle,
        png_out,
        node_group_fn=node_group_fn,
        group_colors=group_colors,
        group_borders=group_borders,
        group_order=group_order,
        community_map=community_map,
        figsize=figsize,
        dpi=dpi,
        min_node_separation=min_node_separation,
    )
    export_cytoscape_tables(
        G,
        stem,
        cyto_dir,
        node_group_fn=node_group_fn,
        group_colors=group_colors,
        group_borders=group_borders,
        community_map=community_map,
        condition=condition,
    )


def community_stem(condition_tag: str, community_id: int, view: str) -> str:
    """Filename stem: {condition}_community_{id}_{view}."""
    return f"{condition_tag}_community_{community_id}_{view}"


def hif_context_nodes(G: nx.Graph, comm0: Set[str], hif: str = "HIF1A", max_neighbors: int = 18) -> Set[str]:
    if hif not in G:
        return {hif} if hif in comm0 else set()
    nbrs = [n for n in G.neighbors(hif) if n in comm0]
    priority = {"ELOB", "ELOC", "CSN1", "CSN5", "CSN4", "CTR9", "CDC73", "FEM1A"}
    nbrs_sorted = sorted(nbrs, key=lambda x: (0 if x in priority else 1, x))
    take = set(nbrs_sorted[:max_neighbors])
    take.add(hif)
    for p in ("ELOB", "ELOC"):
        if p in nbrs:
            take.add(p)
    return take


_BARE_COMMUNITY_PNG = re.compile(r"^(wt|vifko|mock)_community_\d+\.png$")

OBSOLETE_FIGURE_STEMS = (
    "01_wt_vif_crl5_runx_module",
    "02_wt_csn_cul2_elongin_layer",
    "03_vifko_crl5_without_vif",
    "04_vifko_runx_cbfb_bc11b",
    "05_vifko_hif1a_csn_elongin_context",
    "01_mock_crl5_adaptors_module",
    "02_mock_csn_cul2_elongin_layer",
    "03_mock_csn_elongin_rbx_core",
    "04_mock_runx_cbfb_bc11b",
    "05_mock_hif1a_csn_elongin_context",
)


def _remove_obsolete_figures(out_dir: Path, cyto_dir: Path) -> None:
    stems = set(OBSOLETE_FIGURE_STEMS)
    for png in out_dir.glob("*.png"):
        if _BARE_COMMUNITY_PNG.match(png.name):
            stems.add(png.stem)
    for stem in stems:
        png = out_dir / f"{stem}.png"
        if png.exists():
            png.unlink()
        for suffix in ("_nodes.tsv", "_edges.tsv"):
            tsv = cyto_dir / f"{stem}{suffix}"
            if tsv.exists():
                tsv.unlink()


def plot_mock_subgraphs(
    G: nx.Graph,
    out_dir: Path,
    cyto_dir: Path,
    *,
    crl5_nodes: Optional[Set[str]],
    crl5_comm_id: int,
    csn_elongin_nodes: Optional[Set[str]],
    csn_elongin_comm_id: int,
    hif_comm_nodes: Optional[Set[str]],
    hif_comm_id: int,
    runx_nodes: Optional[Set[str]],
    runx_comm_id: Optional[int],
    community_map: Dict[str, int],
    edge_subtitle: str,
    dpi: int,
    min_node_separation: Optional[float],
    skip_png: bool = False,
    title_suffix: str = "",
    fig02_comm_id: int = 0,
    fig02_note: str = "",
) -> None:
    suffix = title_suffix.strip()

    def go(
        H,
        stem,
        title,
        subtitle,
        *,
        node_group_fn,
        group_colors,
        group_borders=None,
        group_order=None,
        figsize=(11, 10),
    ):
        if H.number_of_nodes() == 0:
            print(f"Skip empty graph: {title}")
            return
        if suffix:
            title = f"{title} ({suffix})"
        if skip_png:
            export_cytoscape_tables(
                H,
                stem,
                cyto_dir,
                node_group_fn=node_group_fn,
                group_colors=group_colors,
                group_borders=group_borders,
                community_map=community_map,
                condition="Mock",
            )
        else:
            plot_and_export(
                H,
                stem,
                title,
                subtitle,
                out_dir / f"{stem}.png",
                cyto_dir,
                condition="Mock",
                node_group_fn=node_group_fn,
                group_colors=group_colors,
                group_borders=group_borders,
                group_order=group_order,
                community_map=community_map,
                figsize=figsize,
                dpi=dpi,
                min_node_separation=min_node_separation,
            )

    if crl5_nodes:
        H = induced_or_warn(G, crl5_nodes)
        go(
            H,
            community_stem("mock", crl5_comm_id, "crl5_adaptors_module"),
            f"Mock — CRL5 · RBX · neddylation / adaptors (Louvain community {crl5_comm_id})",
            edge_subtitle,
            node_group_fn=classify_wt_vif_module,
            group_colors=GROUP_COLORS_WT_VIF_MODULE,
            group_borders=GROUP_BORDERS_WT_VIF_MODULE,
            group_order=[g for g in GROUP_ORDER_WT_VIF if g != "Vif"],
        )

    H = induced_or_warn(G, WT_CSN_CUL2_CURATED)
    if H.number_of_nodes() > 0:
        sub = f"Curated subset for legibility. {fig02_note}{edge_subtitle}".strip()
        go(
            H,
            community_stem("mock", fig02_comm_id, "csn_cul2_elongin_layer"),
            f"Mock — CSN · CRL2 · elongin · PAF1 / adaptors (curated; community {fig02_comm_id})",
            sub,
            node_group_fn=classify_csn_layer,
            group_colors=GROUP_COLORS_CSN,
            group_borders=GROUP_BORDERS_CSN,
            group_order=GROUP_ORDER_CSN,
        )

    if csn_elongin_nodes:
        H = induced_or_warn(G, csn_elongin_nodes)
        go(
            H,
            community_stem("mock", csn_elongin_comm_id, "csn_elongin_rbx_core"),
            f"Mock — CSN · elongin · RBX core (Louvain community {csn_elongin_comm_id})",
            edge_subtitle,
            node_group_fn=classify_csn_elongin_core,
            group_colors=GROUP_COLORS_CSN_ELONGIN_CORE,
            group_borders=GROUP_BORDERS_CSN_ELONGIN_CORE,
            group_order=GROUP_ORDER_CSN_ELONGIN_CORE,
            figsize=(10, 9),
        )

    if runx_nodes and runx_comm_id is not None:
        H = induced_or_warn(G, runx_nodes)
        rc = {"RUNX": "#FBCFE8", "CBFβ": "#BFDBFE", "BC11B": "#FECACA", "Other": "#E2E8F0"}
        rb = {"RUNX": "#DB2777", "CBFβ": "#2563EB", "BC11B": "#DC2626", "Other": "#64748B"}
        go(
            H,
            community_stem("mock", runx_comm_id, "runx_cbfb_bc11b"),
            f"Mock — CBFβ · RUNX · BC11B (Louvain community {runx_comm_id})",
            edge_subtitle,
            node_group_fn=runx_group,
            group_colors=rc,
            group_borders=rb,
            group_order=["BC11B", "CBFβ", "RUNX"],
            figsize=(9, 8),
        )

    if hif_comm_nodes:
        nodes_hif = hif_context_nodes(G, hif_comm_nodes)
        H = induced_or_warn(G, nodes_hif)
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
        go(
            H,
            community_stem("mock", hif_comm_id, "hif1a_csn_elongin_context"),
            f"Mock — HIF1A with CSN / elongin-rich neighborhood (community {hif_comm_id})",
            f"HIF1A ego network within Louvain comm. {hif_comm_id}. {edge_subtitle}",
            node_group_fn=hif_group,
            group_colors=hc,
            group_borders=hb,
            group_order=["HIF1A", "COP9 (CSN)", "Elongin B/C", "Neighbors (comm. 0)"],
            figsize=(12, 10),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "wt_vif_comparison" / "subgraphs_biological")
    ap.add_argument(
        "--cytoscape-dir",
        type=Path,
        default=None,
        help="Cytoscape TSV output dir (default: <out-dir>/cytoscape)",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--min-node-separation",
        type=float,
        default=MIN_NODE_SEPARATION,
        help="Min center distance within biological groups (0 = compact, may overlap)",
    )
    ap.add_argument("--skip-png", action="store_true", help="Only write Cytoscape tables")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    cyto_dir: Path = args.cytoscape_dir or (out_dir / "cytoscape")
    min_sep = args.min_node_separation if args.min_node_separation > 0 else None

    wt_bin = ROOT / "post_sampling_analysis/base_models/cullin_wt_config_multi_seeds/main_cluster_network/main_cluster_binary_freq_ge_0p6.npy"
    vif_bin = ROOT / "post_sampling_analysis/base_models/cullin_vif_config_multi_seeds/main_cluster_network/main_cluster_binary_freq_ge_0p6.npy"
    wt_edges_tsv = ROOT / "post_sampling_analysis/base_models/cullin_wt_config_multi_seeds/main_cluster_network/main_cluster_edges.tsv"
    vif_edges_tsv = ROOT / "post_sampling_analysis/base_models/cullin_vif_config_multi_seeds/main_cluster_network/main_cluster_edges.tsv"
    wt_md_p = default_model_data_pkl(ROOT / "results", "cullin_wt_config_multi_seeds")
    vif_md_p = default_model_data_pkl(ROOT / "results", "cullin_vif_config_multi_seeds")

    wt_edge_attrs = load_edge_attributes(wt_edges_tsv)
    vif_edge_attrs = load_edge_attributes(vif_edges_tsv)

    G_wt = build_graph_from_main_cluster_binary(wt_bin, load_model_data(wt_md_p))
    G_vif = build_graph_from_main_cluster_binary(vif_bin, load_model_data(vif_md_p))
    annotate_graph_edges(G_wt, wt_edge_attrs)
    annotate_graph_edges(G_vif, vif_edge_attrs)

    dep_comm = load_communities_tsv(ROOT / "wt_vif_comparison/vif_dependent_communities.tsv")
    vif_comm = load_communities_tsv(ROOT / "wt_vif_comparison/vif_ko_communities.tsv")
    mock_comm = load_communities_tsv(ROOT / "wt_vif_comparison/mock_communities.tsv")
    wt_prot_comm = load_protein_community(ROOT / "wt_vif_comparison/vif_dependent_communities.tsv")
    vif_prot_comm = load_protein_community(ROOT / "wt_vif_comparison/vif_ko_communities.tsv")
    mock_prot_comm = load_protein_community(ROOT / "wt_vif_comparison/mock_communities.tsv")

    wt_c0, wt_c1 = dep_comm[0], dep_comm[1]
    vif_c0, vif_c1, vif_c2 = vif_comm[0], vif_comm[1], vif_comm[2]
    mock_c0, mock_c1, mock_c2 = mock_comm[0], mock_comm[1], mock_comm[2]

    edge_subtitle = (
        "Edge width ∝ frequency quantile within subgraph "
        f"({EDGE_N_TIERS} tiers; main-cluster mean frequency ≥ 0.6)"
    )

    _remove_obsolete_figures(out_dir, cyto_dir)

    def go(
        G,
        stem,
        title,
        subtitle,
        *,
        condition,
        node_group_fn,
        group_colors,
        group_borders=None,
        group_order=None,
        community_map=None,
        figsize=(11, 10),
    ):
        H = G
        if args.skip_png:
            export_cytoscape_tables(
                H,
                stem,
                cyto_dir,
                node_group_fn=node_group_fn,
                group_colors=group_colors,
                group_borders=group_borders,
                community_map=community_map,
                condition=condition,
            )
        else:
            plot_and_export(
                H,
                stem,
                title,
                subtitle,
                out_dir / f"{stem}.png",
                cyto_dir,
                condition=condition,
                node_group_fn=node_group_fn,
                group_colors=group_colors,
                group_borders=group_borders,
                group_order=group_order,
                community_map=community_map,
                figsize=figsize,
                dpi=args.dpi,
                min_node_separation=min_sep,
            )

    go(
        induced_or_warn(G_wt, wt_c1),
        community_stem("wt", 1, "vif_crl5_runx_module"),
        "WT — Vif · CRL5 · RBX · CBFβ · RUNX (Louvain community 1)",
        edge_subtitle,
        condition="WT",
        node_group_fn=classify_wt_vif_module,
        group_colors=GROUP_COLORS_WT_VIF_MODULE,
        group_borders=GROUP_BORDERS_WT_VIF_MODULE,
        group_order=GROUP_ORDER_WT_VIF,
        community_map=wt_prot_comm,
    )

    go(
        induced_or_warn(G_wt, WT_CSN_CUL2_CURATED),
        community_stem("wt", 0, "csn_cul2_elongin_layer"),
        "WT — CSN · CRL2 · elongin · PAF1 / adaptors (curated; community 0)",
        f"Curated subset for legibility (full comm. 0 = {len(wt_c0)} nodes). {edge_subtitle}",
        condition="WT",
        node_group_fn=classify_csn_layer,
        group_colors=GROUP_COLORS_CSN,
        group_borders=GROUP_BORDERS_CSN,
        group_order=GROUP_ORDER_CSN,
        community_map=wt_prot_comm,
    )

    go(
        induced_or_warn(G_vif, vif_c1),
        community_stem("vifko", 1, "crl5_without_vif"),
        "VIF KO — CRL5 · RBX · neddylation / adaptors (Louvain community 1)",
        edge_subtitle,
        condition="VIF_KO",
        node_group_fn=lambda n: classify_wt_vif_module(n) if n != "VIFPROTEIN" else "Vif",
        group_colors={**GROUP_COLORS_WT_VIF_MODULE, "Vif": "#F1F5F9"},
        group_borders=GROUP_BORDERS_WT_VIF_MODULE,
        group_order=[g for g in GROUP_ORDER_WT_VIF if g != "Vif"],
        community_map=vif_prot_comm,
    )

    rc = {"RUNX": "#FBCFE8", "CBFβ": "#BFDBFE", "BC11B": "#FECACA", "Other": "#E2E8F0"}
    rb = {"RUNX": "#DB2777", "CBFβ": "#2563EB", "BC11B": "#DC2626", "Other": "#64748B"}
    go(
        induced_or_warn(G_vif, vif_c2),
        community_stem("vifko", 2, "runx_cbfb_bc11b"),
        "VIF KO — CBFβ · RUNX · BC11B (Louvain community 2)",
        edge_subtitle,
        condition="VIF_KO",
        node_group_fn=runx_group,
        group_colors=rc,
        group_borders=rb,
        group_order=["BC11B", "CBFβ", "RUNX"],
        community_map=vif_prot_comm,
        figsize=(9, 8),
    )

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
    go(
        induced_or_warn(G_vif, hif_context_nodes(G_vif, vif_c0)),
        community_stem("vifko", 0, "hif1a_csn_elongin_context"),
        "VIF KO — HIF1A with CSN / elongin-rich neighborhood (community 0)",
        f"HIF1A ego network within Louvain comm. 0. {edge_subtitle}",
        condition="VIF_KO",
        node_group_fn=hif_group,
        group_colors=hc,
        group_borders=hb,
        group_order=["HIF1A", "COP9 (CSN)", "Elongin B/C", "Neighbors (comm. 0)"],
        community_map=vif_prot_comm,
        figsize=(12, 10),
    )

    mock_edges_tsv = (
        ROOT
        / "post_sampling_analysis/base_models/cullin_mock_config_multi_seeds/main_cluster_network/main_cluster_edges.tsv"
    )
    G_mock = build_graph_from_main_cluster_edges_tsv(mock_edges_tsv, 0.6)
    plot_mock_subgraphs(
        G_mock,
        out_dir,
        cyto_dir,
        crl5_nodes=mock_c1,
        crl5_comm_id=1,
        csn_elongin_nodes=mock_c2,
        csn_elongin_comm_id=2,
        hif_comm_nodes=mock_c0,
        hif_comm_id=0,
        runx_nodes=None,
        runx_comm_id=None,
        community_map=mock_prot_comm,
        edge_subtitle=edge_subtitle,
        dpi=args.dpi,
        min_node_separation=min_sep,
        skip_png=args.skip_png,
        fig02_comm_id=0,
        fig02_note=f"(full comm. 0 = {len(mock_c0)} nodes). ",
    )

    print(f"Done. Figures in {out_dir}; Cytoscape tables in {cyto_dir}")


if __name__ == "__main__":
    main()

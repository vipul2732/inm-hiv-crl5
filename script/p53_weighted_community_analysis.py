#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from detect_mock_communities import visualize_network_with_communities


def load_weighted_graph(edges_tsv: Path, min_frequency: float) -> nx.Graph:
    df = pd.read_csv(edges_tsv, sep="\t")
    need = {"protein1", "protein2", "frequency"}
    if not need.issubset(df.columns):
        raise ValueError(f"{edges_tsv} missing columns: {sorted(need - set(df.columns))}")
    df = df[df["frequency"].astype(float) >= float(min_frequency)].copy()
    g = nx.Graph()
    for _, r in df.iterrows():
        p1 = str(r["protein1"]).strip()
        p2 = str(r["protein2"]).strip()
        w = float(r["frequency"])
        if p1 and p2 and p1 != p2:
            g.add_edge(p1, p2, weight=w, frequency=w)
    return g


def detect_greedy_modularity(g: nx.Graph) -> Tuple[List[Set[str]], float]:
    comms = list(nx.community.greedy_modularity_communities(g, weight="weight"))
    mod = nx.community.modularity(g, comms, weight="weight") if comms else -1.0
    return [set(c) for c in comms], float(mod)


def save_communities(out_dir: Path, communities: List[Set[str]], dataset: str) -> None:
    rows = []
    for cid, nodes in enumerate(sorted(communities, key=lambda s: (-len(s), sorted(s)[0] if s else ""))):
        for n in sorted(nodes):
            rows.append(
                {
                    "dataset": dataset,
                    "protein": n,
                    "community_id": cid,
                    "community_size": len(nodes),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "communities.tsv", sep="\t", index=False)


def plot_network(
    g: nx.Graph,
    communities: List[Set[str]],
    out_png: Path,
    title: str,
    highlight_node: str = "OLA1",
) -> None:
    if g.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No edges after cutoff", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    node_to_c = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            node_to_c[n] = cid

    # Reuse the exact cullin-era visual style (spacing, palette, edge/node aesthetics).
    visualize_network_with_communities(
        g,
        node_to_c,
        communities,
        title,
        out_png,
        label_mode="all_bold",
        dpi=350,
        figsize=(16, 12),
        intra_k=1.35,
        intra_iter=180,
        cluster_scale=2.0,
        node_size=900,
        max_labels_per_community=14,
        force_labels=[highlight_node],
    )


def create_community_aware_layout(
    g: nx.Graph,
    communities: List[Set[str]],
    seed: int = 42,
    intra_k: float = 0.8,
    intra_iter: int = 120,
    cluster_scale: float = 1.25,
    gap: float = 1.8,
    pack_iter: int = 900,
) -> Dict[str, np.ndarray]:
    """Layout nodes per community, then pack community centers to avoid overlap."""
    rng = np.random.default_rng(seed)
    comm_local: List[Dict[str, np.ndarray]] = []
    comm_radii: List[float] = []
    comm_centers: List[np.ndarray] = []

    for cid, nodeset in enumerate(communities):
        nodes = list(nodeset)
        if len(nodes) == 0:
            comm_local.append({})
            comm_radii.append(0.0)
            comm_centers.append(np.array([0.0, 0.0], dtype=float))
            continue
        if len(nodes) == 1:
            local = {nodes[0]: np.array([0.0, 0.0], dtype=float)}
        else:
            sub = g.subgraph(nodes)
            local_nx = nx.spring_layout(
                sub,
                seed=seed + cid,
                k=intra_k,
                iterations=intra_iter,
                weight="weight",
            )
            local = {n: np.array(local_nx[n], dtype=float) for n in nodes}

        pts = np.vstack(list(local.values()))
        center = pts.mean(axis=0)
        pts = pts - center
        max_norm = float(np.max(np.linalg.norm(pts, axis=1))) if len(nodes) > 1 else 1.0
        max_norm = max(max_norm, 1e-6)
        pts = (pts / max_norm) * cluster_scale
        local2 = {n: pts[i] for i, n in enumerate(nodes)}
        comm_local.append(local2)
        radius = float(np.max(np.linalg.norm(pts, axis=1))) + 0.48
        comm_radii.append(radius)
        comm_centers.append(np.array([0.0, 0.0], dtype=float))

    n_c = len(communities)
    if n_c == 0:
        return {}

    total_r = sum(comm_radii) + gap * n_c
    circle_r = max(2.0, total_r / (2 * math.pi))
    for i in range(n_c):
        ang = 2 * math.pi * i / max(1, n_c)
        comm_centers[i] = np.array([circle_r * math.cos(ang), circle_r * math.sin(ang)], dtype=float)

    for _ in range(pack_iter):
        moved = False
        for i in range(n_c):
            for j in range(i + 1, n_c):
                delta = comm_centers[j] - comm_centers[i]
                dist = float(np.linalg.norm(delta))
                need = comm_radii[i] + comm_radii[j] + gap
                if dist < 1e-9:
                    direction = rng.normal(size=2)
                    direction /= np.linalg.norm(direction) + 1e-12
                    comm_centers[j] += direction * 0.01
                    moved = True
                    continue
                if dist < need:
                    direction = delta / dist
                    push = 0.5 * (need - dist + 1e-6)
                    comm_centers[i] -= direction * push
                    comm_centers[j] += direction * push
                    moved = True
        if not moved:
            break

    pos: Dict[str, np.ndarray] = {}
    for cid, nodeset in enumerate(communities):
        center = comm_centers[cid]
        for n in nodeset:
            pos[n] = comm_local[cid][n] + center
    return pos


def run_one(dataset_dir: Path, out_root: Path, min_frequency: float) -> Dict:
    ds_name = dataset_dir.name
    edges_tsv = dataset_dir / "main_cluster_network" / "main_cluster_edges.tsv"
    out_dir = out_root / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    g = load_weighted_graph(edges_tsv, min_frequency=min_frequency)
    if g.number_of_nodes() == 0:
        meta = {
            "dataset": ds_name,
            "min_frequency": min_frequency,
            "n_nodes": 0,
            "n_edges": 0,
            "n_communities": 0,
            "method": "greedy_modularity",
            "modularity": None,
            "ola1_in_graph": False,
        }
        (out_dir / "community_report.json").write_text(json.dumps(meta, indent=2) + "\n")
        plot_network(g, [], out_dir / "network_communities.png", f"{ds_name} (empty at cutoff)")
        return meta

    comms, mod = detect_greedy_modularity(g)
    save_communities(out_dir, comms, ds_name)
    plot_network(
        g,
        comms,
        out_dir / "network_communities.png",
        f"{ds_name}: weighted communities (freq >= {min_frequency})",
    )

    sizes = sorted([len(c) for c in comms], reverse=True)
    meta = {
        "dataset": ds_name,
        "min_frequency": min_frequency,
        "method": "greedy_modularity",
        "n_nodes": int(g.number_of_nodes()),
        "n_edges": int(g.number_of_edges()),
        "n_communities": int(len(comms)),
        "community_sizes": sizes,
        "modularity": float(mod),
        "ola1_in_graph": bool("OLA1" in g.nodes()),
        "ola1_degree": int(g.degree("OLA1")) if "OLA1" in g else 0,
    }
    (out_dir / "community_report.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Weighted community analysis for p53 main-cluster edge frequencies.")
    ap.add_argument("--p53-root", type=Path, default=Path("post_sampling_analysis/p53"))
    ap.add_argument("--out-dir", type=Path, default=Path("post_sampling_analysis/p53/community_analysis_weighted"))
    ap.add_argument("--min-frequency", type=float, default=0.80)
    args = ap.parse_args()

    p53_root = args.p53_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        p53_root / "p53_contact_mutations_only_config_multi_seeds",
        p53_root / "p53_structural_destabilization_only_config_multi_seeds",
        p53_root / "Serine_mutants_only_config_multi_seeds",
    ]
    all_rows: List[Dict] = []
    for d in datasets:
        all_rows.append(run_one(d, out_dir, args.min_frequency))

    pd.DataFrame(all_rows).to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    print(f"Wrote {out_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()

"""Build NetworkX graphs from main-cluster binary edge vectors (centroid over per-draw M)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validation_analysis import build_pair_to_flat_index, normalize_protein_name  # noqa: E402


CBFB_NODE_LABEL = "CBFβ"
CBFB_GROUP_LABEL = "CBFβ"
CBFB_RUNX_GROUP_LABEL = "CBFβ / RUNX"


def protein_label_for_plot(name: str) -> str:
    """Match community scripts: normalized symbol, PEBB → CBFB for display."""
    n = normalize_protein_name(name)
    return "CBFB" if n == "PEBB" else n


def node_display_label(name: str) -> str:
    """Human-readable node label for subgraph figures."""
    n = normalize_protein_name(name)
    if n == "VIFPROTEIN":
        return "Vif"
    if n in {"CBFB", "PEBB"}:
        return CBFB_NODE_LABEL
    return n


def default_model_data_pkl(results_dir: Path, dataset_folder: str) -> Path:
    merged = results_dir / dataset_folder / "merged" / "merged_chains.pkl"
    if not merged.is_file():
        raise FileNotFoundError(merged)
    with open(merged, "rb") as f:
        merged_data = pickle.load(f)
    seeds = merged_data["seeds"]
    seed0 = int(seeds[0])
    pkl = results_dir / dataset_folder / f"seed_{seed0}" / f"0_model23_n_{seed0}_model_data.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"model_data not found: {pkl}")
    return pkl


def pair_index_to_protein_pair(
    idx_map: Dict[Tuple[str, str], int],
) -> list:
    n = len(idx_map)
    inv: list = [None] * n
    for pk, ix in idx_map.items():
        inv[ix] = pk
    if any(x is None for x in inv):
        raise ValueError("Non-contiguous pair indices")
    return inv


def build_graph_from_main_cluster_binary(
    binary_npy: Path,
    model_data: dict,
) -> nx.Graph:
    """
    One edge per upper-tri pair with binary==1. Unit weights for community detection.
    Endpoints use protein_label_for_plot() for consistency with TSV-based graphs.
    """
    binary_npy = Path(binary_npy)
    vec = np.load(binary_npy).astype(np.uint8).ravel()
    idx_map = build_pair_to_flat_index(model_data)
    if vec.size != len(idx_map):
        raise ValueError(f"{binary_npy}: length {vec.size} != model pairs {len(idx_map)}")
    inv = pair_index_to_protein_pair(idx_map)

    G = nx.Graph()
    for i, bit in enumerate(vec):
        if int(bit) != 1:
            continue
        a_raw, b_raw = inv[i]
        p1 = protein_label_for_plot(str(a_raw))
        p2 = protein_label_for_plot(str(b_raw))
        if p1 == p2:
            continue
        G.add_edge(p1, p2, weight=1.0, probability=1.0)
    return G


def build_graph_from_main_cluster_edges_tsv(
    edges_tsv: Path,
    min_frequency: float,
) -> nx.Graph:
    """
    Undirected graph from main_cluster_edges.tsv with frequency >= min_frequency.
    Edge attrs: frequency, wilson_lb, weight=1 (for Louvain).
    """
    edges_tsv = Path(edges_tsv)
    df = pd.read_csv(edges_tsv, sep="\t")
    G = nx.Graph()
    for _, r in df.iterrows():
        freq = float(r["frequency"])
        if freq < min_frequency:
            continue
        p1 = protein_label_for_plot(str(r["protein1"]))
        p2 = protein_label_for_plot(str(r["protein2"]))
        if p1 == p2:
            continue
        G.add_edge(
            p1,
            p2,
            weight=1.0,
            frequency=freq,
            wilson_lb=float(r["wilson_lb"]),
        )
    return G


def load_model_data(pkl: Path) -> dict:
    with open(pkl, "rb") as f:
        return pickle.load(f)

#!/usr/bin/env python3
"""
Create a 3-panel figure for VIF-dependent interactions:
Panel 1: Heatmap
Panel 2: Network
Panel 3: Clustered network by probability bins (0.5-0.55, 0.55-0.60, etc.)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 16


def load_vif_dependent_interactions(file_path: Path) -> pd.DataFrame:
    """Load VIF-dependent interactions."""
    logger.info(f"Loading VIF-dependent interactions from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    # Replace PEBB with CBFB
    df['Protein1'] = df['Protein1'].replace('PEBB', 'CBFB')
    df['Protein2'] = df['Protein2'].replace('PEBB', 'CBFB')
    logger.info(f"  Loaded {len(df)} VIF-dependent interactions")
    return df


def create_interaction_matrix(df: pd.DataFrame) -> tuple:
    """Create a symmetric interaction matrix from interaction pairs."""
    all_proteins = set()
    for _, row in df.iterrows():
        all_proteins.add(str(row['Protein1']))
        all_proteins.add(str(row['Protein2']))
    
    protein_list = sorted(all_proteins)
    n_proteins = len(protein_list)
    protein_to_idx = {prot: idx for idx, prot in enumerate(protein_list)}
    
    matrix = np.zeros((n_proteins, n_proteins))
    
    for _, row in df.iterrows():
        prot1 = str(row['Protein1'])
        prot2 = str(row['Protein2'])
        prob = float(row['WT_Probability'])
        
        idx1 = protein_to_idx[prot1]
        idx2 = protein_to_idx[prot2]
        
        matrix[idx1, idx2] = prob
        matrix[idx2, idx1] = prob
    
    return matrix, protein_list, protein_to_idx


def highlight_key_proteins(protein_list: list) -> Dict[str, bool]:
    """Identify key proteins for highlighting."""
    key_proteins = {
        'vifprotein', 'CUL5', 'CUL2', 'DCA11', 'ELOB', 'ELOC', 
        'RBX1', 'RBX2', 'CBFB', 'PDC6I', 'NEDD8'
    }
    return {prot: prot in key_proteins for prot in protein_list}


def plot_panel1_heatmap(matrix: np.ndarray, protein_list: list, ax, cmap: str = 'YlOrRd'):
    """Panel 1: Heatmap of VIF-dependent interactions."""
    logger.info("Creating Panel 1: Heatmap")
    
    key_proteins = highlight_key_proteins(protein_list)
    key_prot_list = [p for p in protein_list if key_proteins[p]]
    other_prot_list = [p for p in protein_list if not key_proteins[p]]
    reordered_list = key_prot_list + other_prot_list
    
    protein_to_idx = {prot: idx for idx, prot in enumerate(protein_list)}
    reordered_indices = [protein_to_idx[p] for p in reordered_list]
    reordered_matrix = matrix[np.ix_(reordered_indices, reordered_indices)]
    
    im = ax.imshow(reordered_matrix, cmap=cmap, aspect='auto', 
                  vmin=0, vmax=matrix.max())
    
    n_key = len(key_prot_list)
    if n_key > 0 and n_key < len(protein_list):
        ax.axhline(y=n_key - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(x=n_key - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xticks(np.arange(len(reordered_list)))
    ax.set_yticks(np.arange(len(reordered_list)))
    ax.set_xticklabels(reordered_list, rotation=90, ha='right', fontsize=12, fontweight='bold')
    ax.set_yticklabels(reordered_list, fontsize=12, fontweight='bold')
    
    for i, protein in enumerate(reordered_list):
        if key_proteins[protein]:
            ax.get_xticklabels()[i].set_color('darkblue')
            ax.get_yticklabels()[i].set_color('darkblue')
    
    cbar = plt.colorbar(im, ax=ax, label='Interaction Probability (WT)',
                       fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label('Interaction Probability (WT)', fontsize=12, fontweight='bold')
    
    ax.set_title('VIF-dependent protein interactions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Proteins', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proteins', fontsize=12, fontweight='bold')
    
    ax.set_xticks(np.arange(len(reordered_list)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(reordered_list)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)


def build_network(df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from interaction data."""
    G = nx.Graph()
    
    for _, row in df.iterrows():
        protein1 = str(row['Protein1']).strip()
        protein2 = str(row['Protein2']).strip()
        prob = float(row['WT_Probability'])
        
        G.add_edge(protein1, protein2, weight=prob, probability=prob)
    
    return G


def identify_key_proteins() -> Set[str]:
    """Identify key VIF complex proteins."""
    return {
        'vifprotein', 'CUL5', 'CUL2', 'DCA11', 'ELOB', 'ELOC', 
        'RBX1', 'RBX2', 'CBFB', 'PDC6I', 'NEDD8'
    }


def calculate_node_sizes(G: nx.Graph, min_size: int = 200, max_size: int = 2000) -> Dict[str, float]:
    """Calculate node sizes based on degree."""
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    node_sizes = {}
    for node in G.nodes():
        degree = degrees[node]
        size = min_size + (max_size - min_size) * (degree / max_degree)
        node_sizes[node] = size
    
    return node_sizes


def assign_node_colors(G: nx.Graph) -> Dict[str, str]:
    """Assign colors to nodes based on protein type."""
    key_proteins = identify_key_proteins()
    
    node_colors = {}
    for node in G.nodes():
        if node == 'vifprotein':
            node_colors[node] = '#FF0000'
        elif node in ['CUL5', 'CUL2']:
            node_colors[node] = '#0066CC'
        elif node in ['DCA11']:
            node_colors[node] = '#00CC00'
        elif node in ['ELOB', 'ELOC', 'CBFB']:
            node_colors[node] = '#FF9900'
        elif node in ['RBX1', 'RBX2']:
            node_colors[node] = '#9900CC'
        elif node in ['PDC6I', 'NEDD8']:
            node_colors[node] = '#FF00FF'
        elif node in key_proteins:
            node_colors[node] = '#CCCC00'
        else:
            node_colors[node] = '#CCCCCC'
    
    return node_colors


def plot_panel2_network(G: nx.Graph, ax):
    """Panel 2: Network visualization."""
    logger.info("Creating Panel 2: Network")
    
    node_sizes = calculate_node_sizes(G)
    node_colors = assign_node_colors(G)
    
    edge_weights = [G[u][v]['probability'] for u, v in G.edges()]
    min_weight = min(edge_weights) if edge_weights else 0.5
    max_weight = max(edge_weights) if edge_weights else 0.7
    edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) 
                   if max_weight > min_weight else 2 
                   for w in edge_weights]
    
    core_complex = {'vifprotein', 'CUL5', 'DCA11', 'ELOB', 'ELOC', 'RBX2', 'CBFB'}
    
    pos = nx.spring_layout(G, k=3.5, iterations=200, seed=42)
    
    # Prevent node overlap
    min_distance = 0.3
    max_iterations = 50
    for iteration in range(max_iterations):
        moved = False
        nodes_list = list(G.nodes())
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                pos1 = np.array(pos[node1])
                pos2 = np.array(pos[node2])
                distance = np.linalg.norm(pos1 - pos2)
                if distance < min_distance:
                    direction = (pos1 - pos2) / (distance + 1e-6)
                    move_distance = (min_distance - distance) / 2
                    pos[node1] = tuple(pos1 + direction * move_distance)
                    pos[node2] = tuple(pos2 - direction * move_distance)
                    moved = True
        if not moved:
            break
    
    # Draw edges
    edge_colors = []
    for u, v in G.edges():
        if u in core_complex and v in core_complex:
            edge_colors.append('#FF0000')
        else:
            edge_colors.append('#CCCCCC')
    
    non_core_edges = [(u, v) for u, v in G.edges() 
                      if not (u in core_complex and v in core_complex)]
    core_edges = [(u, v) for u, v in G.edges() 
                  if u in core_complex and v in core_complex]
    
    if non_core_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=non_core_edges,
            width=[edge_widths[list(G.edges()).index(e)] for e in non_core_edges],
            alpha=0.3,
            edge_color='gray',
            ax=ax
        )
    
    if core_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=core_edges,
            width=[edge_widths[list(G.edges()).index(e)] for e in core_edges],
            alpha=0.8,
            edge_color='red',
            ax=ax
        )
    
    # Draw nodes
    node_list = list(G.nodes())
    node_color_list = [node_colors.get(node, '#CCCCCC') for node in node_list]
    node_size_list = [node_sizes.get(node, 500) for node in node_list]
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_color=node_color_list,
        node_size=node_size_list,
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Draw labels
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=10,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title('VIF-dependent protein interactions', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')


def get_probability_bin(prob: float, bin_size: float = 0.01, min_prob: float = None) -> Tuple[float, float]:
    """Get probability bin range for a given probability.
    
    If min_prob is provided, bins start from min_prob rounded down to nearest bin_size.
    Otherwise, bins start from 0.50.
    """
    # Round to avoid floating point issues
    prob = round(prob, 4)
    
    # If min_prob is provided, use it as the base
    if min_prob is not None:
        base = np.floor(min_prob / bin_size) * bin_size
    else:
        base = 0.50
    
    # Calculate which bin this probability falls into
    bin_start = np.floor((prob - base) / bin_size) * bin_size + base
    bin_end = bin_start + bin_size
    return (round(bin_start, 3), round(bin_end, 3))


def plot_panel3_clustered_network(df: pd.DataFrame, ax, bin_size: float = 0.01):
    """Panel 3: Clustered network by probability bins."""
    logger.info("Creating Panel 3: Clustered network by probability bins")
    
    # Get the actual probability range in the data
    probs = df['WT_Probability'].values
    min_prob = float(np.min(probs))
    max_prob = float(np.max(probs))
    
    # Adjust bin_size if needed to get reasonable number of bins
    # If data range is narrow, use smaller bins
    data_range = max_prob - min_prob
    if data_range < 0.05:
        bin_size = max(0.005, data_range / 10)  # Create ~10 bins
    elif data_range < 0.15:
        bin_size = 0.01
    else:
        bin_size = 0.05
    
    logger.info(f"  Probability range: {min_prob:.4f} - {max_prob:.4f}")
    logger.info(f"  Using bin size: {bin_size:.4f}")
    
    # Group edges by probability bins
    bins = defaultdict(list)
    for _, row in df.iterrows():
        prob = float(row['WT_Probability'])
        bin_start, bin_end = get_probability_bin(prob, bin_size, min_prob)
        bins[(bin_start, bin_end)].append((row['Protein1'], row['Protein2'], prob))
    
    # Create separate graphs for each bin
    bin_graphs = {}
    for bin_range, edges in bins.items():
        G_bin = nx.Graph()
        for prot1, prot2, prob in edges:
            G_bin.add_edge(prot1, prot2, weight=prob, probability=prob)
        bin_graphs[bin_range] = G_bin
    
    # Create combined graph for layout
    G_combined = nx.Graph()
    for _, row in df.iterrows():
        protein1 = str(row['Protein1']).strip()
        protein2 = str(row['Protein2']).strip()
        prob = float(row['WT_Probability'])
        G_combined.add_edge(protein1, protein2, weight=prob, probability=prob)
    
    # Calculate layout on combined graph
    pos = nx.spring_layout(G_combined, k=3.5, iterations=200, seed=42)
    
    # Prevent overlap
    min_distance = 0.3
    max_iterations = 50
    for iteration in range(max_iterations):
        moved = False
        nodes_list = list(G_combined.nodes())
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                pos1 = np.array(pos[node1])
                pos2 = np.array(pos[node2])
                distance = np.linalg.norm(pos1 - pos2)
                if distance < min_distance:
                    direction = (pos1 - pos2) / (distance + 1e-6)
                    move_distance = (min_distance - distance) / 2
                    pos[node1] = tuple(pos1 + direction * move_distance)
                    pos[node2] = tuple(pos2 - direction * move_distance)
                    moved = True
        if not moved:
            break
    
    # Generate color scheme dynamically based on number of bins
    # Use a colormap from light gray (low) to dark red (high)
    sorted_bin_ranges = sorted(bins.keys())
    n_bins = len(sorted_bin_ranges)
    
    # Create color gradient
    if n_bins == 1:
        bin_colors = {sorted_bin_ranges[0]: '#8C8C8C'}
    else:
        # Use colormap from light to dark
        from matplotlib import cm
        colors = cm.Reds(np.linspace(0.3, 0.9, n_bins))
        bin_colors = {}
        for bin_range, color in zip(sorted_bin_ranges, colors):
            # Convert RGBA to hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            bin_colors[bin_range] = hex_color
    
    # Draw edges grouped by bins (draw lower probability first, then higher)
    sorted_bins = sorted(bin_graphs.keys())
    for bin_range in sorted_bins:
        G_bin = bin_graphs[bin_range]
        color = bin_colors.get(bin_range, '#CCCCCC')
        
        edges = list(G_bin.edges())
        if edges:
            edge_widths = [1 + 2 * (G_bin[u][v]['probability'] - 0.5) / 0.5 
                          for u, v in edges]
            nx.draw_networkx_edges(
                G_bin, pos,
                edgelist=edges,
                width=edge_widths,
                alpha=0.6,
                edge_color=color,
                ax=ax
            )
    
    # Draw nodes
    node_sizes = calculate_node_sizes(G_combined)
    node_colors = assign_node_colors(G_combined)
    
    node_list = list(G_combined.nodes())
    node_color_list = [node_colors.get(node, '#CCCCCC') for node in node_list]
    node_size_list = [node_sizes.get(node, 500) for node in node_list]
    
    nx.draw_networkx_nodes(
        G_combined, pos,
        nodelist=node_list,
        node_color=node_color_list,
        node_size=node_size_list,
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # Draw labels
    labels = {node: node for node in G_combined.nodes()}
    nx.draw_networkx_labels(
        G_combined, pos,
        labels,
        font_size=10,
        font_weight='bold',
        ax=ax
    )
    
    # Create legend for probability bins
    legend_elements = []
    for bin_range in sorted(bins.keys()):
        bin_start, bin_end = bin_range
        color = bin_colors.get(bin_range, '#CCCCCC')
        count = len(bins[bin_range])
        # Use appropriate decimal places based on bin size
        if bin_size < 0.01:
            format_str = f'{bin_start:.3f}-{bin_end:.3f} (n={count})'
        else:
            format_str = f'{bin_start:.2f}-{bin_end:.2f} (n={count})'
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=format_str)
        )
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
    
    ax.set_title('VIF-dependent interactions clustered by probability', 
                fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser(
        description='Create 3-panel figure for VIF-dependent interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-file', type=Path,
                       default='wt_vif_comparison/vif_dependent_interactions.tsv',
                       help='VIF-dependent interactions TSV file')
    parser.add_argument('--output-file', type=Path,
                       default='wt_vif_comparison/vif_three_panel.png',
                       help='Output file for 3-panel figure')
    parser.add_argument('--bin-size', type=float, default=0.01,
                       help='Probability bin size for clustering (default: 0.01, auto-adjusted based on data range)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_vif_dependent_interactions(args.input_file)
    
    # Create interaction matrix
    matrix, protein_list, protein_to_idx = create_interaction_matrix(df)
    
    # Create 3-panel figure
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    plot_panel1_heatmap(matrix, protein_list, ax1)
    
    # Panel 2: Network
    ax2 = fig.add_subplot(gs[0, 1])
    G = build_network(df)
    plot_panel2_network(G, ax2)
    
    # Panel 3: Clustered network
    ax3 = fig.add_subplot(gs[0, 2])
    plot_panel3_clustered_network(df, ax3, bin_size=args.bin_size)
    
    plt.savefig(args.output_file, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    logger.info("=" * 80)
    logger.info("3-panel figure generation complete!")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

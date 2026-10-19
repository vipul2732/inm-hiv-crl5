#!/usr/bin/env python3
"""
Create interaction network diagram for VIF-dependent interactions.
Panel 2: Network visualization showing VIF-dependent protein-protein interactions.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
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


def build_network(df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from interaction data."""
    logger.info("Building interaction network")
    G = nx.Graph()
    
    for _, row in df.iterrows():
        protein1 = str(row['Protein1']).strip()
        protein2 = str(row['Protein2']).strip()
        prob = float(row['WT_Probability'])
        
        # Add edge with probability as weight
        G.add_edge(protein1, protein2, weight=prob, probability=prob)
    
    logger.info(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def identify_key_proteins() -> Set[str]:
    """Identify key VIF complex proteins."""
    return {
        'vifprotein', 'CUL5', 'CUL2', 'DCA11', 'ELOB', 'ELOC', 
        'RBX1', 'RBX2', 'CBFB', 'PDC6I', 'NEDD8'
    }


def calculate_node_sizes(G: nx.Graph, min_size: int = 200, max_size: int = 2000) -> Dict[str, float]:
    """Calculate node sizes based on degree and importance."""
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    node_sizes = {}
    for node in G.nodes():
        degree = degrees[node]
        # Scale by degree
        size = min_size + (max_size - min_size) * (degree / max_degree)
        node_sizes[node] = size
    
    return node_sizes


def assign_node_colors(G: nx.Graph) -> Dict[str, str]:
    """Assign colors to nodes based on protein type."""
    key_proteins = identify_key_proteins()
    
    node_colors = {}
    for node in G.nodes():
        if node == 'vifprotein':
            node_colors[node] = '#FF0000'  # Red for VIF
        elif node in ['CUL5', 'CUL2']:
            node_colors[node] = '#0066CC'  # Blue for cullins
        elif node in ['DCA11']:
            node_colors[node] = '#00CC00'  # Green for DCAF11
        elif node in ['ELOB', 'ELOC', 'CBFB']:
            node_colors[node] = '#FF9900'  # Orange for Elongin BC
        elif node in ['RBX1', 'RBX2']:
            node_colors[node] = '#9900CC'  # Purple for RBX
        elif node in ['PDC6I', 'NEDD8']:
            node_colors[node] = '#FF00FF'  # Magenta for other key proteins
        elif node in key_proteins:
            node_colors[node] = '#CCCC00'  # Yellow for other key
        else:
            node_colors[node] = '#CCCCCC'  # Gray for others
    
    return node_colors


def plot_network(G: nx.Graph, output_file: Path, figsize: tuple = (14, 12)):
    """Plot interaction network."""
    logger.info("Creating network visualization")
    
    # Calculate node properties
    node_sizes = calculate_node_sizes(G)
    node_colors = assign_node_colors(G)
    
    # Calculate edge widths based on probability
    edge_weights = [G[u][v]['probability'] for u, v in G.edges()]
    min_weight = min(edge_weights) if edge_weights else 0.5
    max_weight = max(edge_weights) if edge_weights else 0.7
    edge_widths = [1 + 3 * (w - min_weight) / (max_weight - min_weight) 
                   if max_weight > min_weight else 2 
                   for w in edge_weights]
    
    # Use spring layout with better parameters
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Try to position key proteins in center
    key_proteins = identify_key_proteins()
    key_nodes = [n for n in G.nodes() if n in key_proteins]
    if key_nodes:
        # Adjust positions to center key proteins
        center_pos = np.mean([pos[n] for n in key_nodes], axis=0)
        for node in key_nodes:
            pos[node] = pos[node] - center_pos + np.array([0, 0])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges first (so nodes appear on top)
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.4,
        edge_color='gray',
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
    
    # Draw labels - only for key proteins and high-degree nodes
    degrees = dict(G.degree())
    high_degree_threshold = np.percentile(list(degrees.values()), 70)
    
    labels = {}
    for node in G.nodes():
        if node in key_proteins or degrees[node] >= high_degree_threshold:
            labels[node] = node
    
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=14,
        font_weight='bold',
        ax=ax
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF0000', edgecolor='black', label='VIF protein'),
        mpatches.Patch(facecolor='#0066CC', edgecolor='black', label='Cullins (CUL2, CUL5)'),
        mpatches.Patch(facecolor='#00CC00', edgecolor='black', label='DCAF11'),
        mpatches.Patch(facecolor='#FF9900', edgecolor='black', label='Elongin BC (ELOB, ELOC, CBFB)'),
        mpatches.Patch(facecolor='#9900CC', edgecolor='black', label='RBX1/2 (RING-box proteins)'),
        mpatches.Patch(facecolor='#FF00FF', edgecolor='black', label='ALIX (PDC6I), NEDD8'),
        mpatches.Patch(facecolor='#CCCCCC', edgecolor='black', label='Other proteins'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.9)
    
    ax.set_title('VIF-dependent protein interactions', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved network plot to {output_file}")


def plot_network_highlighted_core(G: nx.Graph, output_file: Path, figsize: tuple = (14, 12)):
    """Plot network with VIF-CUL5-DCAF11-Elongin BC core complex highlighted."""
    logger.info("Creating network visualization with highlighted core complex")
    
    # Define core complex
    core_complex = {'vifprotein', 'CUL5', 'DCA11', 'ELOB', 'ELOC', 'RBX2', 'CBFB'}
    
    # Calculate node properties
    node_sizes = calculate_node_sizes(G, min_size=300, max_size=2500)
    node_colors = assign_node_colors(G)
    
    # All nodes follow the same size rule (based on degree)
    
    # Calculate edge widths
    edge_weights = [G[u][v]['probability'] for u, v in G.edges()]
    min_weight = min(edge_weights) if edge_weights else 0.5
    max_weight = max(edge_weights) if edge_weights else 0.7
    edge_widths = [1 + 4 * (w - min_weight) / (max_weight - min_weight) 
                   if max_weight > min_weight else 2 
                   for w in edge_weights]
    
    # Highlight core complex edges
    edge_colors = []
    for u, v in G.edges():
        if u in core_complex and v in core_complex:
            edge_colors.append('#FF0000')  # Red for core complex edges
        else:
            edge_colors.append('#CCCCCC')  # Gray for others
    
    # Use spring layout with better spacing to avoid overlap
    pos = nx.spring_layout(G, k=3.5, iterations=200, seed=42)
    
    # Apply additional separation to prevent overlap
    # Increase spacing between all nodes
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
                    # Push nodes apart
                    direction = (pos1 - pos2) / (distance + 1e-6)
                    move_distance = (min_distance - distance) / 2
                    pos[node1] = tuple(pos1 + direction * move_distance)
                    pos[node2] = tuple(pos2 - direction * move_distance)
                    moved = True
        if not moved:
            break
    
    # Center core complex
    core_nodes = [n for n in G.nodes() if n in core_complex]
    if core_nodes:
        center_pos = np.mean([pos[n] for n in core_nodes], axis=0)
        for node in core_nodes:
            pos[node] = tuple(np.array(pos[node]) - center_pos + np.array([0, 0]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw non-core edges first
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
            style='solid',
            ax=ax
        )
    
    # Draw non-core nodes
    non_core_nodes = [n for n in G.nodes() if n not in core_complex]
    if non_core_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=non_core_nodes,
            node_color=[node_colors.get(n, '#CCCCCC') for n in non_core_nodes],
            node_size=[node_sizes.get(n, 500) for n in non_core_nodes],
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            ax=ax
        )
    
    # Draw core complex nodes on top
    if core_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=core_nodes,
            node_color=[node_colors.get(n, '#FF0000') for n in core_nodes],
            node_size=[node_sizes.get(n, 2000) for n in core_nodes],
            alpha=0.9,
            edgecolors='red',
            linewidths=3,
            ax=ax
        )
    
    # Draw labels for all nodes
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=12,
        font_weight='bold',
        ax=ax
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#FF0000', edgecolor='red', linewidth=2, label='VIF-CUL5-DCAF11-Elongin BC core complex'),
        mpatches.Patch(facecolor='#0066CC', edgecolor='black', label='Cullins'),
        mpatches.Patch(facecolor='#00CC00', edgecolor='black', label='DCAF11'),
        mpatches.Patch(facecolor='#FF9900', edgecolor='black', label='Elongin BC (CBFB)'),
        mpatches.Patch(facecolor='#9900CC', edgecolor='black', label='RBX1/2 (RING-box proteins)'),
        mpatches.Patch(facecolor='#FF00FF', edgecolor='black', label='ALIX (PDC6I), NEDD8'),
        mpatches.Patch(facecolor='#CCCCCC', edgecolor='black', label='Other proteins'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.9)
    
    ax.set_title('VIF-dependent protein interactions', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved network plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create network diagram for VIF-dependent interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-file', type=Path,
                       default='wt_vif_comparison/vif_dependent_interactions.tsv',
                       help='VIF-dependent interactions TSV file')
    parser.add_argument('--output-dir', type=Path, default='wt_vif_comparison',
                       help='Output directory for plots')
    parser.add_argument('--highlight-core', action='store_true', default=True,
                       help='Highlight VIF-CUL5-DCAF11-Elongin BC core complex')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_vif_dependent_interactions(args.input_file)
    
    # Build network
    G = build_network(df)
    
    # Create plots
    if args.highlight_core:
        plot_network_highlighted_core(G, args.output_dir / 'vif_dependent_network.png')
    else:
        plot_network(G, args.output_dir / 'vif_dependent_network.png')
    
    logger.info("=" * 80)
    logger.info("Network visualization complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

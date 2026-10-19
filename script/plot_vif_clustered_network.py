#!/usr/bin/env python3
"""
Create a clustered network visualization for VIF-dependent interactions.
- Includes interactions with probability >= 0.6
- Clusters by protein complexes
- Colors edges by probability bins
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
from scipy.spatial import ConvexHull

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 16


def load_wt_interactions(wt_file: Path, vif_file: Path, min_prob: float = 0.6) -> pd.DataFrame:
    """Load WT interactions with probability >= min_prob that are VIF-dependent."""
    logger.info(f"Loading WT interactions from {wt_file}")
    logger.info(f"Comparing with VIF KO from {vif_file}")
    
    # Load full WT dataset
    df_wt = pd.read_csv(wt_file, sep='\t')
    df_wt['Protein1'] = df_wt['Protein1'].replace('PEBB', 'CBFB')
    df_wt['Protein2'] = df_wt['Protein2'].replace('PEBB', 'CBFB')
    
    # Load VIF KO dataset
    df_vif = pd.read_csv(vif_file, sep='\t')
    df_vif['Protein1'] = df_vif['Protein1'].replace('PEBB', 'CBFB')
    df_vif['Protein2'] = df_vif['Protein2'].replace('PEBB', 'CBFB')
    
    # Create pair identifiers
    df_wt['Pair'] = df_wt.apply(
        lambda row: tuple(sorted([str(row['Protein1']), str(row['Protein2'])])),
        axis=1
    )
    df_vif['Pair'] = df_vif.apply(
        lambda row: tuple(sorted([str(row['Protein1']), str(row['Protein2'])])),
        axis=1
    )
    
    # Get VIF KO pairs
    vif_pairs = set(df_vif['Pair'])
    
    # Filter WT by probability threshold
    df_wt_filtered = df_wt[df_wt['Probability'] >= min_prob].copy()
    
    logger.info(f"  Total WT pairs with prob >= {min_prob}: {len(df_wt_filtered)}")
    logger.info(f"  Probability range: {df_wt_filtered['Probability'].min():.4f} - {df_wt_filtered['Probability'].max():.4f}")
    
    # Identify VIF-dependent (in WT but not in VIF KO)
    df_wt_filtered['VIF_KO_Probability'] = df_wt_filtered['Pair'].apply(
        lambda pair: df_vif[df_vif['Pair'] == pair]['Probability'].values[0] 
        if pair in vif_pairs else 0.0
    )
    
    # Keep only VIF-dependent interactions (present in WT but absent or very low in VIF KO)
    df_vif_dependent = df_wt_filtered[
        (df_wt_filtered['VIF_KO_Probability'] < 0.1) | 
        (df_wt_filtered['Pair'].apply(lambda p: p not in vif_pairs))
    ].copy()
    
    df_vif_dependent = df_vif_dependent.rename(columns={'Probability': 'WT_Probability'})
    
    logger.info(f"  VIF-dependent pairs with prob >= {min_prob}: {len(df_vif_dependent)}")
    logger.info(f"  VIF-dependent probability range: {df_vif_dependent['WT_Probability'].min():.4f} - {df_vif_dependent['WT_Probability'].max():.4f}")
    
    return df_vif_dependent


def identify_protein_complexes() -> Dict[str, str]:
    """Identify which complex each protein belongs to."""
    complexes = {
        # VIF-CUL5-DCAF11-Elongin BC core complex
        'vifprotein': 'VIF-CUL5-DCAF11-Elongin BC',
        'CUL5': 'VIF-CUL5-DCAF11-Elongin BC',
        'DCA11': 'VIF-CUL5-DCAF11-Elongin BC',
        'ELOB': 'VIF-CUL5-DCAF11-Elongin BC',
        'ELOC': 'VIF-CUL5-DCAF11-Elongin BC',
        'CBFB': 'VIF-CUL5-DCAF11-Elongin BC',
        'RBX2': 'VIF-CUL5-DCAF11-Elongin BC',
        
        # CUL2 complex
        'CUL2': 'CUL2 Complex',
        'RBX1': 'CUL2 Complex',
        
        # Other key proteins
        'NEDD8': 'NEDD8 Modification',
        'PDC6I': 'ALIX/PDC6I',
    }
    return complexes


def get_probability_bin(prob: float, bin_size: float = 0.05) -> Tuple[float, float]:
    """Get probability bin range for a given probability."""
    prob = round(prob, 4)
    # Round down to nearest bin_size starting from 0.6
    bin_start = np.floor((prob - 0.6) / bin_size) * bin_size + 0.6
    bin_start = max(0.6, bin_start)
    bin_end = bin_start + bin_size
    return (round(bin_start, 2), round(bin_end, 2))


def assign_complex_to_node(node: str, complexes: Dict[str, str]) -> str:
    """Assign a complex name to a node."""
    return complexes.get(node, 'Other')


def build_network_with_complexes(df: pd.DataFrame) -> Tuple[nx.Graph, Dict[str, str], Dict[str, str]]:
    """Build network and assign complexes to nodes."""
    G = nx.Graph()
    complexes = identify_protein_complexes()
    
    node_to_complex = {}
    
    for _, row in df.iterrows():
        protein1 = str(row['Protein1']).strip()
        protein2 = str(row['Protein2']).strip()
        prob = float(row['WT_Probability'])
        
        G.add_edge(protein1, protein2, weight=prob, probability=prob)
        
        # Assign complexes
        node_to_complex[protein1] = assign_complex_to_node(protein1, complexes)
        node_to_complex[protein2] = assign_complex_to_node(protein2, complexes)
    
    return G, node_to_complex, complexes


def calculate_node_sizes(G: nx.Graph, min_size: int = 300, max_size: int = 2500) -> Dict[str, float]:
    """Calculate node sizes based on degree."""
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    node_sizes = {}
    for node in G.nodes():
        degree = degrees[node]
        size = min_size + (max_size - min_size) * (degree / max_degree)
        node_sizes[node] = size
    
    return node_sizes


def assign_node_colors(G: nx.Graph, node_to_complex: Dict[str, str]) -> Dict[str, str]:
    """Assign colors to nodes based on complex membership."""
    complex_colors = {
        'VIF-CUL5-DCAF11-Elongin BC': '#FF0000',  # Red
        'CUL2 Complex': '#0066CC',  # Blue
        'NEDD8 Modification': '#FF00FF',  # Magenta
        'ALIX/PDC6I': '#00CC00',  # Green
        'Other': '#CCCCCC',  # Gray
    }
    
    node_colors = {}
    for node in G.nodes():
        complex_name = node_to_complex.get(node, 'Other')
        node_colors[node] = complex_colors.get(complex_name, '#CCCCCC')
    
    return node_colors


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Detect communities using Louvain algorithm."""
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(G, seed=42)
        
        # Map nodes to community IDs
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        
        logger.info(f"  Detected {len(communities)} communities")
        return node_to_community
    except ImportError:
        logger.warning("  Community detection not available, using complex-based clustering")
        return {}


def create_clustered_layout(G: nx.Graph, node_to_complex: Dict[str, str]) -> Dict:
    """Create a layout that clusters nodes by complex using community detection and force-directed layout."""
    # Group nodes by complex
    complex_to_nodes = defaultdict(list)
    for node in G.nodes():
        complex_name = node_to_complex.get(node, 'Other')
        complex_to_nodes[complex_name].append(node)
    
    logger.info(f"  Clustering {len(G.nodes())} nodes into {len(complex_to_nodes)} complexes")
    
    # Detect communities for better clustering
    node_to_community = detect_communities(G)
    
    # Create a subgraph for each complex and compute local layouts
    complex_positions = {}
    complex_layouts = {}
    
    # Calculate node sizes to use for spacing
    node_sizes = calculate_node_sizes(G, min_size=300, max_size=2500)
    node_radii = {node: np.sqrt(size / np.pi) / 100.0 for node, size in node_sizes.items()}
    
    # Calculate positions for each complex separately
    for complex_name, nodes in complex_to_nodes.items():
        if len(nodes) == 0:
            continue
        
        # Create subgraph for this complex
        subgraph_nodes = [n for n in nodes if n in G.nodes()]
        if len(subgraph_nodes) < 2:
            # Single node - place at origin
            complex_layouts[complex_name] = {subgraph_nodes[0]: (0, 0)} if subgraph_nodes else {}
            continue
        
        # Get subgraph (nodes and edges within complex)
        subgraph = G.subgraph(subgraph_nodes).copy()
        
        # Use spring layout for subgraph with spacing based on node sizes
        if subgraph.number_of_edges() > 0:
            # Calculate average node radius for this complex
            avg_radius = np.mean([node_radii.get(n, 0.3) for n in subgraph_nodes])
            k_value = max(2.0, avg_radius * 4)  # Scale k based on node size
            sub_pos = nx.spring_layout(subgraph, k=k_value, iterations=300, seed=42)
        else:
            # If no edges, use circular layout with spacing based on node sizes
            avg_radius = np.mean([node_radii.get(n, 0.3) for n in subgraph_nodes])
            scale = max(2.0, avg_radius * 6)
            sub_pos = nx.circular_layout(subgraph, scale=scale)
        
        complex_layouts[complex_name] = sub_pos
    
    # Position complexes in a circle or grid
    n_complexes = len(complex_to_nodes)
    if n_complexes == 0:
        return nx.spring_layout(G, k=3.0, iterations=300, seed=42)
    
    # Calculate complex centers in a circular arrangement with more spacing
    center_pos = np.array([0, 0])
    radius = 6.0  # Distance from center for complex centers (increased)
    
    complex_centers = {}
    angle_step = 2 * np.pi / n_complexes if n_complexes > 1 else 0
    
    # Place VIF-CUL5-DCAF11-Elongin BC in center if it exists
    vif_complex_name = 'VIF-CUL5-DCAF11-Elongin BC'
    if vif_complex_name in complex_to_nodes and len(complex_to_nodes[vif_complex_name]) > 0:
        complex_centers[vif_complex_name] = center_pos
        angle = 0
        for complex_name in complex_to_nodes.keys():
            if complex_name != vif_complex_name:
                offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                complex_centers[complex_name] = center_pos + offset
                angle += angle_step
    else:
        # No VIF complex, arrange all in circle
        angle = 0
        for complex_name in complex_to_nodes.keys():
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            complex_centers[complex_name] = center_pos + offset
            angle += angle_step
    
    # Combine layouts: position each complex's nodes relative to its center
    pos = {}
    for complex_name, nodes in complex_to_nodes.items():
        if complex_name not in complex_centers:
            continue
        
        complex_center = complex_centers[complex_name]
        local_layout = complex_layouts.get(complex_name, {})
        
        if len(local_layout) == 0:
            # No layout computed, place nodes in small circle
            if len(nodes) == 1:
                pos[nodes[0]] = tuple(complex_center)
            else:
                local_angle = 0
                local_radius = 0.5
                for node in nodes:
                    if node in G.nodes():
                        offset = np.array([local_radius * np.cos(local_angle), 
                                          local_radius * np.sin(local_angle)])
                        pos[node] = tuple(complex_center + offset)
                        local_angle += 2 * np.pi / len(nodes)
        else:
            # Use computed layout, but center it
            local_center = np.mean([np.array(local_layout[n]) for n in nodes if n in local_layout], axis=0) if local_layout else np.array([0, 0])
            
            for node in nodes:
                if node in local_layout:
                    relative_pos = np.array(local_layout[node]) - local_center
                    pos[node] = tuple(complex_center + relative_pos * 1.2)  # Scale for better spacing
                elif node in G.nodes():
                    pos[node] = tuple(complex_center)
    
    # Ensure all nodes have positions
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, 0)
    
    # Apply force-directed refinement to improve spacing while maintaining clusters
    # Use a modified spring layout that respects community structure
    pos = refine_layout_with_communities(G, pos, node_to_complex, node_to_community, iterations=50)
    
    return pos


def refine_layout_with_communities(G: nx.Graph, initial_pos: Dict, 
                                   node_to_complex: Dict[str, str],
                                   node_to_community: Dict[str, int],
                                   iterations: int = 100) -> Dict:
    """Refine layout using force-directed algorithm that respects communities."""
    pos = initial_pos.copy()
    
    # Calculate node sizes to determine minimum distance
    node_sizes = calculate_node_sizes(G, min_size=300, max_size=2500)
    # Convert node size to radius (assuming size is area, radius is roughly sqrt(area/pi))
    node_radii = {node: np.sqrt(size / np.pi) / 100.0 for node, size in node_sizes.items()}
    
    # Parameters
    k = 2.0  # Optimal distance between nodes (increased)
    repulsion_strength = 1.0  # Strength of repulsion between different complexes (increased)
    attraction_strength = 0.3  # Strength of attraction within complexes
    
    for iteration in range(iterations):
        new_pos = {}
        for node in G.nodes():
            force = np.array([0.0, 0.0])
            
            # Repulsion from all other nodes
            for other_node in G.nodes():
                if node == other_node:
                    continue
                
                diff = np.array(pos[node]) - np.array(pos[other_node])
                distance = np.linalg.norm(diff) + 1e-6
                
                # Stronger repulsion if nodes are in different complexes
                node_complex = node_to_complex.get(node, 'Other')
                other_complex = node_to_complex.get(other_node, 'Other')
                
                if node_complex != other_complex:
                    # Different complexes - stronger repulsion
                    repulsion = repulsion_strength * k * k / distance
                else:
                    # Same complex - weaker repulsion
                    repulsion = 0.1 * k * k / distance
                
                force += repulsion * diff / distance
            
            # Attraction along edges
            for neighbor in G.neighbors(node):
                diff = np.array(pos[neighbor]) - np.array(pos[node])
                distance = np.linalg.norm(diff) + 1e-6
                
                # Stronger attraction if in same complex
                node_complex = node_to_complex.get(node, 'Other')
                neighbor_complex = node_to_complex.get(neighbor, 'Other')
                
                if node_complex == neighbor_complex:
                    attraction = attraction_strength * (distance - k) / k
                else:
                    attraction = 0.1 * (distance - k) / k
                
                force += attraction * diff / distance
            
            # Update position
            step_size = 0.1
            new_pos[node] = tuple(np.array(pos[node]) + step_size * force)
        
        pos = new_pos
        
        # Prevent nodes from overlapping based on their actual sizes
        nodes_list = list(G.nodes())
        for i, node1 in enumerate(nodes_list):
            for node2 in nodes_list[i+1:]:
                pos1 = np.array(pos[node1])
                pos2 = np.array(pos[node2])
                distance = np.linalg.norm(pos1 - pos2)
                
                # Calculate minimum distance based on node radii
                radius1 = node_radii.get(node1, 0.3)
                radius2 = node_radii.get(node2, 0.3)
                min_distance = (radius1 + radius2) * 1.3  # Add 30% padding
                
                if distance < min_distance:
                    direction = (pos1 - pos2) / (distance + 1e-6)
                    move_distance = (min_distance - distance) / 2
                    pos[node1] = tuple(pos1 + direction * move_distance)
                    pos[node2] = tuple(pos2 - direction * move_distance)
    
    return pos


def plot_clustered_network(df: pd.DataFrame, output_file: Path, bin_size: float = 0.05, 
                           min_prob: float = 0.6, figsize: tuple = (16, 12)):
    """Plot clustered network by complexes and probability bins."""
    logger.info("Creating clustered network visualization")
    
    # Build network
    G, node_to_complex, complexes = build_network_with_complexes(df)
    
    # Get probability range
    probs = df['WT_Probability'].values
    min_prob_data = float(np.min(probs))
    max_prob_data = float(np.max(probs))
    data_range = max_prob_data - min_prob_data
    
    # Adjust bin_size if data range is narrow
    if data_range < 0.05:
        bin_size = max(0.01, data_range / 10)
    elif data_range < 0.15:
        bin_size = 0.01
    else:
        bin_size = 0.05
    
    logger.info(f"  Probability range: {min_prob_data:.4f} - {max_prob_data:.4f}")
    logger.info(f"  Using bin size: {bin_size:.4f}")
    
    # Group edges by probability bins
    bins = defaultdict(list)
    for _, row in df.iterrows():
        prob = float(row['WT_Probability'])
        bin_start, bin_end = get_probability_bin(prob, bin_size)
        bins[(bin_start, bin_end)].append((row['Protein1'], row['Protein2'], prob))
    
    logger.info(f"  Created {len(bins)} probability bins")
    
    # Create layout clustered by complexes
    pos = create_clustered_layout(G, node_to_complex)
    
    # Generate color scheme for bins (lighter shades)
    sorted_bin_ranges = sorted(bins.keys())
    n_bins = len(sorted_bin_ranges)
    
    from matplotlib import cm
    if n_bins == 1:
        bin_colors = {sorted_bin_ranges[0]: '#FFB3B3'}  # Light red
    else:
        # Use lighter shades: start from 0.1 (very light) to 0.5 (medium)
        colors = cm.Reds(np.linspace(0.1, 0.5, n_bins))
        bin_colors = {}
        for bin_range, color in zip(sorted_bin_ranges, colors):
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
            bin_colors[bin_range] = hex_color
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges grouped by bins (lower probability first, then higher)
    for bin_range in sorted_bin_ranges:
        color = bin_colors.get(bin_range, '#CCCCCC')
        edges_in_bin = bins[bin_range]
        
        for prot1, prot2, prob in edges_in_bin:
            if G.has_edge(prot1, prot2):
                edge_width = 1 + 3 * (prob - min_prob) / (max_prob_data - min_prob) if max_prob_data > min_prob else 2
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(prot1, prot2)],
                    width=edge_width,
                    alpha=0.4,  # Lighter transparency
                    edge_color=color,
                    ax=ax
                )
    
    # Draw complex boundaries (convex hulls) to make complexes more distinct
    from matplotlib.patches import Polygon, Circle
    
    complex_to_nodes = defaultdict(list)
    for node in G.nodes():
        complex_name = node_to_complex.get(node, 'Other')
        complex_to_nodes[complex_name].append(node)
    
    # Draw convex hulls for each complex (except "Other" which is too scattered)
    for complex_name, nodes in complex_to_nodes.items():
        if complex_name == 'Other' or len(nodes) < 3:
            continue
        
        # Get positions of nodes in this complex
        node_positions = [np.array(pos[node]) for node in nodes if node in pos]
        if len(node_positions) < 3:
            continue
        
        try:
            # Create convex hull
            hull = ConvexHull(node_positions)
            hull_points = [node_positions[i] for i in hull.vertices]
            hull_points.append(hull_points[0])  # Close the polygon
            
            # Get complex color
            complex_colors = {
                'VIF-CUL5-DCAF11-Elongin BC': '#FF0000',
                'CUL2 Complex': '#0066CC',
                'NEDD8 Modification': '#FF00FF',
                'ALIX/PDC6I': '#00CC00',
            }
            color = complex_colors.get(complex_name, '#CCCCCC')
            
            # Draw filled polygon with transparency
            polygon = Polygon(hull_points, closed=True, 
                            facecolor=color, edgecolor=color, 
                            alpha=0.15, linewidth=2, linestyle='--')
            ax.add_patch(polygon)
        except:
            # If convex hull fails, draw a circle around the complex
            center = np.mean(node_positions, axis=0)
            max_dist = max([np.linalg.norm(pos - center) for pos in node_positions])
            circle = Circle(center, max_dist * 1.2, 
                          facecolor=color, edgecolor=color,
                          alpha=0.15, linewidth=2, linestyle='--')
            ax.add_patch(circle)
    
    # Draw nodes
    node_sizes = calculate_node_sizes(G)
    node_colors = assign_node_colors(G, node_to_complex)
    
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
        font_size=11,
        font_weight='bold',
        ax=ax
    )
    
    # Create legends
    # Legend 1: Complexes
    complex_colors = {
        'VIF-CUL5-DCAF11-Elongin BC': '#FF0000',
        'CUL2 Complex': '#0066CC',
        'NEDD8 Modification': '#FF00FF',
        'ALIX/PDC6I': '#00CC00',
        'Other': '#CCCCCC',
    }
    
    complex_legend_elements = []
    for complex_name, color in complex_colors.items():
        if any(node_to_complex.get(n) == complex_name for n in G.nodes()):
            complex_legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black', label=complex_name)
            )
    
    # Legend 2: Probability bins
    bin_legend_elements = []
    for bin_range in sorted(bins.keys()):
        bin_start, bin_end = bin_range
        color = bin_colors.get(bin_range, '#CCCCCC')
        count = len(bins[bin_range])
        if bin_size < 0.01:
            format_str = f'{bin_start:.3f}-{bin_end:.3f} (n={count})'
        else:
            format_str = f'{bin_start:.2f}-{bin_end:.2f} (n={count})'
        bin_legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=format_str)
        )
    
    # Place legends
    legend1 = ax.legend(handles=complex_legend_elements, loc='upper left', 
                       fontsize=12, framealpha=0.9, title='Protein Complexes', 
                       title_fontsize=13)
    ax.add_artist(legend1)
    ax.legend(handles=bin_legend_elements, loc='upper right', 
             fontsize=12, framealpha=0.9, title='Probability Bins', 
             title_fontsize=13, ncol=1)
    
    ax.set_title('VIF-dependent interactions clustered by complexes and probability\n(HIV WT type infection vs HIV Δ VIF infection)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    logger.info(f"Saved clustered network to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create clustered network for VIF-dependent interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--wt-file', type=Path,
                       default='results/cullin_wt_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='WT all pairs TSV file')
    parser.add_argument('--vif-file', type=Path,
                       default='results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='VIF KO all pairs TSV file')
    parser.add_argument('--output-file', type=Path,
                       default='wt_vif_comparison/vif_clustered_network.png',
                       help='Output file for clustered network')
    parser.add_argument('--min-prob', type=float, default=0.6,
                       help='Minimum probability threshold (default: 0.6)')
    parser.add_argument('--bin-size', type=float, default=0.05,
                       help='Probability bin size (default: 0.05, auto-adjusted)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_wt_interactions(args.wt_file, args.vif_file, min_prob=args.min_prob)
    
    if len(df) == 0:
        logger.error(f"No interactions found with probability >= {args.min_prob}")
        return
    
    # Create clustered network
    plot_clustered_network(df, args.output_file, bin_size=args.bin_size, 
                          min_prob=args.min_prob)
    
    logger.info("=" * 80)
    logger.info("Clustered network generation complete!")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

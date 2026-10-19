#!/usr/bin/env python3
"""
Analyze top predicted pairs and create network visualizations.

This script:
1. Loads merged MCMC chains
2. Extracts edge probabilities
3. Identifies top N pairs (default: top 100)
4. Creates network visualizations
5. Saves edge lists and network statistics

Usage:
    python analyze_top_pairs.py --merged-chains merged/merged_chains.pkl --model-data seed_13/0_model23_n_13_model_data.pkl --output networks/
    python analyze_top_pairs.py --merged-chains merged/merged_chains.pkl --model-data seed_13/0_model23_n_13_model_data.pkl --output networks/ --top-n 50
"""

import argparse
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import jax
import jax.numpy as jnp

# Try to import networkx and matplotlib for network visualization
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("networkx not available. Network visualization will be skipped.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Network plots will not be generated.")

# Import model_variations for edge probability extraction
import sys
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import model_variations as mv


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_merged_chains(merged_chains_file: Path, logger: logging.Logger) -> Dict:
    """Load merged chains from pickle file."""
    logger.info(f"Loading merged chains from {merged_chains_file}")
    with open(merged_chains_file, 'rb') as f:
        data = pkl.load(f)
    
    merged_samples = data.get('merged_samples', {})
    n_chains = data.get('n_chains', 1)
    seeds = data.get('seeds', [])
    burn_in = data.get('burn_in', 0)
    
    logger.info(f"Loaded merged chains: {n_chains} chains, seeds: {seeds}, burn-in: {burn_in}")
    
    return {
        'merged_samples': merged_samples,
        'samples': merged_samples,  # Also provide as 'samples' for compatibility
        'n_chains': n_chains,
        'seeds': seeds,
        'burn_in': burn_in
    }


def remove_human_suffix(protein_name: str) -> str:
    """Remove '_HUMAN' suffix from protein name."""
    if protein_name.endswith('_HUMAN'):
        return protein_name[:-6]  # Remove '_HUMAN' (6 characters)
    return protein_name


def extract_top_pairs(edge_dict: Dict[Tuple[str, str], float], top_n: int = 100) -> List[Tuple[str, str, float]]:
    """
    Extract top N pairs by probability.
    
    Args:
        edge_dict: Dictionary mapping (protein1, protein2) -> probability
        top_n: Number of top pairs to extract
    
    Returns:
        List of (protein1, protein2, probability) tuples, sorted by probability (descending)
        Protein names have '_HUMAN' suffix removed
    """
    # Sort edges by probability (descending)
    sorted_edges = sorted(edge_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Take top N and remove _HUMAN suffix
    top_pairs = [(remove_human_suffix(p1), remove_human_suffix(p2), prob) 
                 for (p1, p2), prob in sorted_edges[:top_n]]
    
    return top_pairs


def create_network_from_pairs(top_pairs: List[Tuple[str, str, float]]) -> nx.Graph:
    """
    Create NetworkX graph from top pairs.
    
    Args:
        top_pairs: List of (protein1, protein2, probability) tuples
    
    Returns:
        NetworkX graph with edges and probabilities as edge attributes
    """
    G = nx.Graph()
    
    for p1, p2, prob in top_pairs:
        G.add_edge(p1, p2, weight=prob, probability=prob)
    
    return G


def plot_network(G: nx.Graph, output_file: Path, title: str = "Top Predicted Pairs Network", 
                logger: Optional[logging.Logger] = None):
    """
    Create publication-quality network visualization.
    
    Args:
        G: NetworkX graph
        output_file: Path to save plot
        title: Plot title
        logger: Optional logger
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        if logger:
            logger.warning("matplotlib or networkx not available. Skipping network plot.")
        return
    
    # Set publication-quality style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create figure with better aspect ratio
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    
    # Calculate node sizes based on degree (with scaling)
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 1
    # Scale node sizes: min_size=300, max_size=2000
    node_sizes = [300 + (degrees[node] - min_degree) / (max_degree - min_degree + 1e-6) * 1700 
                  for node in G.nodes()]
    
    # Get edge weights (probabilities) for coloring and width
    edges = list(G.edges())
    edge_weights = [G[u][v]['probability'] for u, v in edges]
    min_prob = min(edge_weights) if edge_weights else 0
    max_prob = max(edge_weights) if edge_weights else 1
    
    # Try multiple layout algorithms and choose the best one
    layouts = {
        'spring': lambda: nx.spring_layout(G, k=2.5, iterations=100, seed=42),
        'kamada_kawai': lambda: nx.kamada_kawai_layout(G),
        'circular': lambda: nx.circular_layout(G),
        'shell': lambda: nx.shell_layout(G)
    }
    
    # Use spring layout as default, fallback to others if it fails
    pos = None
    for layout_name, layout_func in layouts.items():
        try:
            pos = layout_func()
            if logger:
                logger.info(f"Using {layout_name} layout")
            break
        except:
            continue
    
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Normalize edge weights for better visualization
    if max_prob > min_prob:
        normalized_weights = [(w - min_prob) / (max_prob - min_prob) for w in edge_weights]
    else:
        normalized_weights = [0.5] * len(edge_weights)
    
    # Draw edges with gradient coloring and width
    # Use a colormap that works well in print (viridis or plasma)
    import matplotlib.cm as cm
    edge_colormap = cm.plasma  # Good for print, colorblind-friendly
    
    # Draw edges
    for i, (u, v) in enumerate(edges):
        weight = edge_weights[i]
        norm_weight = normalized_weights[i]
        # Edge width: 0.5 to 3.0 based on probability
        width = 0.5 + norm_weight * 2.5
        # Edge color: use colormap
        color = edge_colormap(norm_weight)
        # Edge alpha: higher probability = more opaque
        alpha = 0.4 + norm_weight * 0.4
        
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
               color=color, linewidth=width, alpha=alpha, zorder=1)
    
    # Draw nodes with gradient coloring based on degree
    node_colormap = cm.viridis
    node_colors = []
    for node in G.nodes():
        degree_norm = (degrees[node] - min_degree) / (max_degree - min_degree + 1e-6)
        node_colors.append(node_colormap(degree_norm))
    
    # Draw nodes (zorder controlled by drawing order - nodes after edges)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.85, edgecolors='white', linewidths=1.5, ax=ax)
    
    # Draw labels - only for high-degree nodes or all if network is small
    if G.number_of_nodes() <= 30:
        # Show all labels for small networks
        labels = {node: node for node in G.nodes()}
        label_font_size = 8
    else:
        # Show labels only for nodes with degree >= median degree
        median_degree = np.median(list(degrees.values()))
        labels = {node: node if degrees[node] >= median_degree else '' 
                 for node in G.nodes()}
        label_font_size = 7
    
    # Draw labels with better positioning (zorder controlled by drawing order - labels last)
    nx.draw_networkx_labels(G, pos, labels, font_size=label_font_size, 
                           font_weight='bold', font_color='black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='none', alpha=0.7),
                           ax=ax)
    
    # Add colorbar for edge probabilities
    sm = plt.cm.ScalarMappable(cmap=edge_colormap, 
                               norm=plt.Normalize(vmin=min_prob, vmax=max_prob))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('Edge Probability', fontsize=10, fontweight='bold', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Remove axes
    ax.axis('off')
    
    # Add network statistics as text box
    stats_text = f"N={G.number_of_nodes()}, E={G.number_of_edges()}\n"
    stats_text += f"Density={nx.density(G):.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if logger:
        logger.info(f"Saved publication-quality network plot: {output_file}")


def save_edge_list(top_pairs: List[Tuple[str, str, float]], output_file: Path, 
                  logger: logging.Logger):
    """Save top pairs to TSV file."""
    import pandas as pd
    
    df_data = []
    for i, (p1, p2, prob) in enumerate(top_pairs, 1):
        df_data.append({
            'Rank': i,
            'Protein1': p1,
            'Protein2': p2,
            'Probability': prob
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Saved top {len(top_pairs)} pairs to {output_file}")


def calculate_network_statistics(G: nx.Graph) -> Dict:
    """Calculate network statistics."""
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
        'num_connected_components': nx.number_connected_components(G),
    }
    
    if stats['num_nodes'] > 0:
        degrees = [d for n, d in G.degree()]
        stats['mean_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['min_degree'] = min(degrees) if degrees else 0
    
    if stats['is_connected'] and stats['num_nodes'] > 1:
        stats['average_path_length'] = nx.average_shortest_path_length(G)
        stats['diameter'] = nx.diameter(G)
    else:
        stats['average_path_length'] = None
        stats['diameter'] = None
    
    # Clustering coefficient
    clustering = nx.clustering(G)
    stats['mean_clustering'] = np.mean(list(clustering.values())) if clustering else 0
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze top predicted pairs and create network visualizations'
    )
    parser.add_argument(
        '--merged-chains',
        type=Path,
        required=True,
        help='Path to merged_chains.pkl file (from merge_and_analyze_convergence.py)'
    )
    parser.add_argument(
        '--model-data',
        type=Path,
        required=True,
        help='Path to model_data.pkl file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for networks and edge lists'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='Number of top pairs to analyze (default: 100)'
    )
    parser.add_argument(
        '--burn-in',
        type=int,
        default=0,
        help='Additional burn-in to apply (default: 0, uses burn-in from merged chains)'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ANALYZING TOP PREDICTED PAIRS")
    logger.info("=" * 60)
    logger.info(f"Merged chains: {args.merged_chains}")
    logger.info(f"Model data: {args.model_data}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Top N pairs: {args.top_n}")
    
    # Load merged chains
    merged_data = load_merged_chains(args.merged_chains, logger)
    # merged_chains.pkl uses 'merged_samples' key
    merged_samples = merged_data.get('merged_samples', merged_data.get('samples', {}))
    
    # Load model data
    logger.info(f"Loading model data from {args.model_data}")
    with open(args.model_data, 'rb') as f:
        model_data = pkl.load(f)
    
    # Extract edge probabilities
    logger.info("Extracting edge probabilities from merged chains...")
    
    # z values are continuous (Normal distribution), but we need binary edges for probabilities
    # Apply sigmoid to convert continuous z to probabilities, then threshold
    z_samples = merged_samples['z']
    
    # Convert to numpy if needed
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    # Apply burn-in if specified
    if args.burn_in > 0:
        z_samples = z_samples[args.burn_in:]
    
    # Convert continuous z to probabilities using sigmoid
    # z ~ Normal(0.1, 1.0), so we use sigmoid to map to [0, 1]
    try:
        from scipy.special import expit  # sigmoid function
        z_probs = expit(z_samples)  # Maps z to probability space [0, 1]
    except ImportError:
        # Fallback to numpy implementation of sigmoid
        z_probs = 1.0 / (1.0 + np.exp(-z_samples))  # Sigmoid: 1/(1+exp(-z))
    
    # Threshold: edge exists if probability > 0.5 (or use raw probabilities as edge weights)
    # For edge probabilities, we'll use the mean probability across samples
    logger.info(f"Computing edge probabilities from {z_samples.shape[0]} samples")
    
    N = int(model_data['N'])
    M = model_data.get('M', N * (N - 1) // 2)
    edge_filter_map = model_data.get('edge_filter_map', None)
    
    # Average probabilities across samples
    mean_z_probs = np.mean(z_probs, axis=0)  # Average across samples
    
    # Convert to edge dictionary
    name2node_idx = model_data.get('name2node_idx', {})
    node_idx2name = model_data.get('node_idx2name', {})
    
    # Map probabilities back to (i, j) pairs
    edge_probs_matrix = np.zeros((N, N))
    edge_dict = {}
    
    if edge_filter_map is not None:
        # Handle filtered edges
        M_original = N * (N - 1) // 2
        triu_indices = np.triu_indices(N, k=1)
        z_full = np.zeros(M_original)
        z_full[edge_filter_map] = mean_z_probs
        edge_probs_matrix[triu_indices] = z_full
    else:
        triu_indices = np.triu_indices(N, k=1)
        edge_probs_matrix[triu_indices] = mean_z_probs
    
    # Make symmetric
    edge_probs_matrix = edge_probs_matrix + edge_probs_matrix.T
    
    # Create edge dictionary
    for i in range(N):
        for j in range(i + 1, N):
            prob = edge_probs_matrix[i, j]
            if prob > 0:
                name_i = node_idx2name.get(i, f"protein_{i}")
                name_j = node_idx2name.get(j, f"protein_{j}")
                edge_dict[(name_i, name_j)] = float(prob)
    
    edge_probs = edge_probs_matrix
    
    logger.info(f"Found {len(edge_dict)} edges with non-zero probability")
    
    # Get top pairs
    logger.info(f"Extracting top {args.top_n} pairs...")
    top_pairs = extract_top_pairs(edge_dict, top_n=args.top_n)
    
    if not top_pairs:
        logger.error("No top pairs found!")
        return
    
    logger.info(f"Top pair: {top_pairs[0][0]} - {top_pairs[0][1]} (prob={top_pairs[0][2]:.4f})")
    
    # Save edge list
    edge_list_file = args.output / f'top_{args.top_n}_pairs.tsv'
    save_edge_list(top_pairs, edge_list_file, logger)
    
    # Create network
    logger.info("Creating network from top pairs...")
    G = create_network_from_pairs(top_pairs)
    
    # Calculate and save network statistics
    stats = calculate_network_statistics(G)
    stats_file = args.output / 'network_statistics.json'
    with open(stats_file, 'w') as f:
        # Convert numpy types to native Python types for JSON
        stats_json = {}
        for k, v in stats.items():
            if isinstance(v, (np.integer, np.floating)):
                stats_json[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif v is None:
                stats_json[k] = None
            else:
                stats_json[k] = v
        json.dump(stats_json, f, indent=2)
    logger.info(f"Saved network statistics to {stats_file}")
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("NETWORK STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Number of nodes: {stats['num_nodes']}")
    logger.info(f"Number of edges: {stats['num_edges']}")
    logger.info(f"Network density: {stats['density']:.4f}")
    logger.info(f"Mean degree: {stats['mean_degree']:.2f}")
    logger.info(f"Mean clustering coefficient: {stats['mean_clustering']:.4f}")
    logger.info(f"Connected: {stats['is_connected']}")
    logger.info(f"Number of connected components: {stats['num_connected_components']}")
    if stats['average_path_length'] is not None:
        logger.info(f"Average path length: {stats['average_path_length']:.2f}")
        logger.info(f"Diameter: {stats['diameter']}")
    
    # Create network visualization
    if HAS_NETWORKX and HAS_MATPLOTLIB:
        logger.info("Creating network visualization...")
        network_plot_file = args.output / f'network_top_{args.top_n}.png'
        plot_network(G, network_plot_file, 
                    title=f'Top {args.top_n} Predicted Protein-Protein Interactions',
                    logger=logger)
    else:
        logger.warning("Skipping network visualization (networkx or matplotlib not available)")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis complete! Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()


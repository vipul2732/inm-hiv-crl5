#!/usr/bin/env python3
"""
Clustering analysis for cullin datasets.

This script performs:
1. Network-based clustering to identify protein complexes
2. Cross-condition comparison (WT, MOCK, VIF)
3. Protein participation analysis (hubs, multi-complex proteins)
4. Visualization of networks and complexes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx

# Try to import community detection algorithms
try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    try:
        import networkx.algorithms.community as nx_comm
        HAS_LOUVAIN = False
    except:
        pass

try:
    from cdlib import algorithms, evaluation
    HAS_CDLIB = True
except ImportError:
    HAS_CDLIB = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


class NetworkAnalyzer:
    """Analyze protein interaction networks and identify complexes."""
    
    def __init__(self, top_n: int = 500, probability_threshold: float = 0.5):
        """
        Initialize analyzer.
        
        Args:
            top_n: Number of top pairs to include in network
            probability_threshold: Minimum probability threshold for edges
        """
        self.top_n = top_n
        self.probability_threshold = probability_threshold
        self.networks: Dict[str, nx.Graph] = {}
        self.communities: Dict[str, Dict[int, Set[str]]] = {}
        self.protein_to_community: Dict[str, Dict[str, int]] = {}
        
    def load_data(self, file_path: Path, condition: str) -> pd.DataFrame:
        """Load interaction data from TSV file."""
        logger.info(f"Loading data for {condition} from {file_path}")
        df = pd.read_csv(file_path, sep='\t')
        
        # Filter by probability threshold
        df = df[df['Probability'] >= self.probability_threshold].copy()
        
        # Sort by probability descending and take top N
        df = df.sort_values('Probability', ascending=False).head(self.top_n).copy()
        
        logger.info(f"  Loaded {len(df)} pairs for {condition}")
        logger.info(f"  Probability range: {df['Probability'].min():.4f} - {df['Probability'].max():.4f}")
        
        return df
    
    def build_network(self, df: pd.DataFrame, condition: str) -> nx.Graph:
        """Build NetworkX graph from interaction data."""
        logger.info(f"Building network for {condition}")
        G = nx.Graph()
        
        for _, row in df.iterrows():
            protein1 = str(row['Protein1']).strip()
            protein2 = str(row['Protein2']).strip()
            prob = float(row['Probability'])
            
            # Add edge with probability as weight
            G.add_edge(protein1, protein2, weight=prob, probability=prob)
        
        logger.info(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        logger.info(f"  Connected components: {nx.number_connected_components(G)}")
        
        self.networks[condition] = G
        return G
    
    def detect_communities(self, G: nx.Graph, condition: str, method: str = 'louvain') -> Dict[int, Set[str]]:
        """
        Detect communities (complexes) in the network.
        
        Args:
            G: NetworkX graph
            condition: Condition name
            method: Community detection method ('louvain' or 'leiden')
        
        Returns:
            Dictionary mapping community ID to set of proteins
        """
        logger.info(f"Detecting communities for {condition} using {method}")
        
        communities = {}
        
        if method == 'louvain':
            if HAS_LOUVAIN:
                partition = community_louvain.best_partition(G, weight='weight')
            elif HAS_CDLIB:
                communities_dict = algorithms.louvain(G, weight='weight')
                partition = {}
                for idx, community in enumerate(communities_dict.communities):
                    for node in community:
                        partition[node] = idx
            else:
                # Use NetworkX's built-in Louvain
                try:
                    communities_dict = nx_comm.louvain_communities(G, weight='weight', seed=42)
                    partition = {}
                    for idx, community in enumerate(communities_dict):
                        for node in community:
                            partition[node] = idx
                except:
                    # Fallback to simple connected components
                    logger.warning("Louvain not available, using connected components")
                    partition = {}
                    for idx, component in enumerate(nx.connected_components(G)):
                        for node in component:
                            partition[node] = idx
            
            # Convert partition to community sets
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = set()
                communities[comm_id].add(node)
        
        elif method == 'leiden':
            if HAS_CDLIB:
                communities_dict = algorithms.leiden(G, weights='weight')
                for idx, community in enumerate(communities_dict.communities):
                    communities[idx] = set(community)
            else:
                logger.warning("Leiden not available, falling back to Louvain")
                return self.detect_communities(G, condition, method='louvain')
        
        # Filter out single-node communities (optional - you might want to keep them)
        communities = {cid: nodes for cid, nodes in communities.items() if len(nodes) > 1}
        
        logger.info(f"  Found {len(communities)} communities")
        logger.info(f"  Community sizes: {sorted([len(nodes) for nodes in communities.values()], reverse=True)[:10]}")
        
        self.communities[condition] = communities
        
        # Create protein-to-community mapping
        protein_to_comm = {}
        for comm_id, nodes in communities.items():
            for node in nodes:
                protein_to_comm[node] = comm_id
        self.protein_to_community[condition] = protein_to_comm
        
        return communities
    
    def analyze_protein_participation(self, condition: str) -> pd.DataFrame:
        """Analyze how many complexes each protein participates in."""
        G = self.networks[condition]
        communities = self.communities[condition]
        
        # Count degree (number of interactions)
        degrees = dict(G.degree())
        
        # Count community participation (should be 1 for non-overlapping communities)
        comm_participation = Counter()
        for comm_id, nodes in communities.items():
            for node in nodes:
                comm_participation[node] += 1
        
        # Calculate betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
        except:
            betweenness = {node: 0 for node in G.nodes()}
        
        # Build summary
        data = []
        for node in G.nodes():
            data.append({
                'Protein': node,
                'Degree': degrees.get(node, 0),
                'Communities': comm_participation.get(node, 0),
                'Betweenness': betweenness.get(node, 0),
                'Condition': condition
            })
        
        return pd.DataFrame(data)
    
    def compare_conditions(self) -> Dict[str, pd.DataFrame]:
        """Compare complexes and proteins across conditions."""
        conditions = list(self.networks.keys())
        
        # Get all proteins across all conditions
        all_proteins = set()
        for G in self.networks.values():
            all_proteins.update(G.nodes())
        
        # Build interaction matrices
        interaction_matrices = {}
        for condition in conditions:
            G = self.networks[condition]
            matrix = np.zeros((len(all_proteins), len(all_proteins)))
            protein_list = sorted(all_proteins)
            protein_to_idx = {p: i for i, p in enumerate(protein_list)}
            
            for u, v, data in G.edges(data=True):
                i = protein_to_idx[u]
                j = protein_to_idx[v]
                prob = data.get('probability', 0.5)
                matrix[i, j] = prob
                matrix[j, i] = prob  # Symmetric
            
            interaction_matrices[condition] = pd.DataFrame(
                matrix, index=protein_list, columns=protein_list
            )
        
        # Find shared vs unique interactions
        shared_interactions = set()
        condition_interactions = {cond: set() for cond in conditions}
        
        for condition in conditions:
            G = self.networks[condition]
            for u, v in G.edges():
                edge = tuple(sorted([u, v]))
                condition_interactions[condition].add(edge)
                if edge in shared_interactions or any(edge in condition_interactions[c] for c in conditions if c != condition):
                    shared_interactions.add(edge)
        
        # Build comparison summary
        comparison_data = []
        for condition in conditions:
            unique = condition_interactions[condition] - shared_interactions
            shared = condition_interactions[condition] & shared_interactions
            
            comparison_data.append({
                'Condition': condition,
                'Total_Interactions': len(condition_interactions[condition]),
                'Shared_Interactions': len(shared),
                'Unique_Interactions': len(unique),
                'Shared_Percentage': 100 * len(shared) / len(condition_interactions[condition]) if condition_interactions[condition] else 0
            })
        
        return {
            'summary': pd.DataFrame(comparison_data),
            'interaction_matrices': interaction_matrices,
            'shared_interactions': shared_interactions,
            'condition_interactions': condition_interactions
        }


def plot_network_with_communities(
    G: nx.Graph,
    communities: Dict[int, Set[str]],
    condition: str,
    output_file: Path,
    layout: str = 'spring',
    max_nodes: int = 200
):
    """Plot network with communities colored differently."""
    logger.info(f"Plotting network for {condition}")
    
    # Limit nodes if network is too large
    if G.number_of_nodes() > max_nodes:
        # Keep top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_set = set([n for n, _ in top_nodes])
        G = G.subgraph(top_node_set).copy()
        # Filter communities
        communities = {
            cid: nodes & top_node_set
            for cid, nodes in communities.items()
            if len(nodes & top_node_set) > 0
        }
    
    # Create color map for communities
    n_communities = len(communities)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_communities, 20)))
    if n_communities > 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = np.vstack([colors] * ((n_communities // 20) + 1))[:n_communities]
    
    node_colors = {}
    for comm_id, nodes in communities.items():
        color = colors[comm_id % len(colors)]
        for node in nodes:
            node_colors[node] = color
    
    # Default color for nodes not in any community
    default_color = [0.8, 0.8, 0.8, 1.0]
    for node in G.nodes():
        if node not in node_colors:
            node_colors[node] = default_color
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, alpha=0.3, width=0.5, ax=ax
    )
    
    # Draw nodes with community colors
    node_list = list(G.nodes())
    node_color_list = [node_colors.get(node, default_color) for node in node_list]
    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list, node_color=node_color_list,
        node_size=100, alpha=0.9, ax=ax
    )
    
    # Draw labels for high-degree nodes only
    degrees = dict(G.degree())
    high_degree_nodes = [n for n, d in degrees.items() if d >= np.percentile(list(degrees.values()), 75)]
    labels = {n: n for n in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f"{condition} - Protein Interaction Network\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities)",
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved network plot to {output_file}")


def plot_community_comparison(analyzer: NetworkAnalyzer, output_file: Path):
    """Plot comparison of communities across conditions."""
    logger.info("Creating community comparison plot")
    
    conditions = list(analyzer.communities.keys())
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(1, n_conditions, figsize=(6 * n_conditions, 5))
    if n_conditions == 1:
        axes = [axes]
    
    for idx, condition in enumerate(conditions):
        communities = analyzer.communities[condition]
        sizes = [len(nodes) for nodes in communities.values()]
        
        ax = axes[idx]
        ax.hist(sizes, bins=min(20, max(5, len(set(sizes)))), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Community Size (number of proteins)', fontsize=12)
        ax.set_ylabel('Number of Communities', fontsize=12)
        ax.set_title(f'{condition}\n({len(communities)} communities)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved community comparison to {output_file}")


def plot_cross_condition_heatmap(comparison_results: Dict, output_file: Path, top_n_proteins: int = 50):
    """Plot heatmap showing interaction probabilities across conditions."""
    logger.info("Creating cross-condition heatmap")
    
    interaction_matrices = comparison_results['interaction_matrices']
    conditions = list(interaction_matrices.keys())
    
    # Select top proteins by total interactions across all conditions
    all_proteins = set()
    for matrix in interaction_matrices.values():
        all_proteins.update(matrix.index)
    
    protein_totals = {}
    for protein in all_proteins:
        total = sum(
            matrix.loc[protein, :].sum() + matrix.loc[:, protein].sum()
            for matrix in interaction_matrices.values()
        )
        protein_totals[protein] = total
    
    top_proteins = sorted(protein_totals.items(), key=lambda x: x[1], reverse=True)[:top_n_proteins]
    top_protein_list = [p for p, _ in top_proteins]
    
    # Create combined matrix
    n_proteins = len(top_protein_list)
    n_conditions = len(conditions)
    
    # For each condition, extract submatrix
    fig, axes = plt.subplots(1, n_conditions, figsize=(5 * n_conditions, 5))
    if n_conditions == 1:
        axes = [axes]
    
    for idx, condition in enumerate(conditions):
        matrix = interaction_matrices[condition]
        submatrix = matrix.loc[top_protein_list, top_protein_list]
        
        ax = axes[idx]
        im = ax.imshow(submatrix.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'{condition}', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(top_protein_list)))
        ax.set_xticklabels(top_protein_list, rotation=90, fontsize=8)
        ax.set_yticks(range(len(top_protein_list)))
        ax.set_yticklabels(top_protein_list, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Probability')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved heatmap to {output_file}")


def plot_protein_participation(participation_dfs: Dict[str, pd.DataFrame], output_file: Path):
    """Plot protein participation analysis (hubs, multi-complex proteins)."""
    logger.info("Creating protein participation plot")
    
    # Combine all conditions
    all_data = pd.concat(list(participation_dfs.values()), ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Degree distribution
    ax = axes[0, 0]
    for condition in participation_dfs.keys():
        data = participation_dfs[condition]
        ax.hist(data['Degree'], bins=20, alpha=0.6, label=condition, edgecolor='black')
    ax.set_xlabel('Degree (number of interactions)', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Degree Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Top hubs by condition
    ax = axes[0, 1]
    top_hubs = all_data.nlargest(20, 'Degree')
    y_pos = np.arange(len(top_hubs))
    ax.barh(y_pos, top_hubs['Degree'], edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Protein']} ({row['Condition']})" for _, row in top_hubs.iterrows()], fontsize=9)
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_title('Top 20 Hub Proteins', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 3: Betweenness centrality
    ax = axes[1, 0]
    for condition in participation_dfs.keys():
        data = participation_dfs[condition]
        ax.hist(data['Betweenness'], bins=20, alpha=0.6, label=condition, edgecolor='black')
    ax.set_xlabel('Betweenness Centrality', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title('Betweenness Centrality Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Condition comparison
    ax = axes[1, 1]
    summary = all_data.groupby('Condition').agg({
        'Degree': ['mean', 'std'],
        'Betweenness': ['mean', 'std']
    }).reset_index()
    summary.columns = ['Condition', 'Mean_Degree', 'Std_Degree', 'Mean_Betweenness', 'Std_Betweenness']
    
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary['Mean_Degree'], width, label='Mean Degree', edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, summary['Mean_Betweenness'] * 100, width, label='Mean Betweenness (×100)', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Network Metrics by Condition', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Condition'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved participation plot to {output_file}")


def plot_shared_interactions(comparison_results: Dict, output_file: Path):
    """Plot Venn-like diagram or bar chart of shared vs unique interactions."""
    logger.info("Creating shared interactions plot")
    
    summary = comparison_results['summary']
    condition_interactions = comparison_results['condition_interactions']
    shared_interactions = comparison_results['shared_interactions']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Bar chart of interaction counts
    ax = axes[0]
    x = np.arange(len(summary))
    width = 0.35
    ax.bar(x - width/2, summary['Shared_Interactions'], width, label='Shared', edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, summary['Unique_Interactions'], width, label='Unique', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Condition', fontsize=12)
    ax.set_ylabel('Number of Interactions', fontsize=12)
    ax.set_title('Shared vs Unique Interactions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Condition'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Percentage shared
    ax = axes[1]
    ax.bar(summary['Condition'], summary['Shared_Percentage'], edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_ylabel('Percentage Shared (%)', fontsize=12)
    ax.set_title('Percentage of Interactions Shared Across Conditions', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved shared interactions plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Perform clustering analysis on cullin datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=Path, default='results',
                       help='Directory containing all_pairs_with_uniprot.tsv files')
    parser.add_argument('--output-dir', type=Path, default='clustering_analysis',
                       help='Output directory for results')
    parser.add_argument('--top-n', type=int, default=500,
                       help='Number of top pairs to include in network')
    parser.add_argument('--probability-threshold', type=float, default=0.5,
                       help='Minimum probability threshold for edges')
    parser.add_argument('--method', type=str, default='louvain', choices=['louvain', 'leiden'],
                       help='Community detection method')
    parser.add_argument('--max-nodes', type=int, default=200,
                       help='Maximum nodes to plot in network visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = NetworkAnalyzer(
        top_n=args.top_n,
        probability_threshold=args.probability_threshold
    )
    
    # Process each condition
    conditions = ['cullin_wt', 'cullin_mock', 'cullin_vif']
    condition_labels = ['WT', 'MOCK', 'VIF']
    
    for condition, label in zip(conditions, condition_labels):
        # Find input file
        input_file = args.input_dir / f"{condition}_config_multi_seeds" / "all_pairs_with_uniprot.tsv"
        
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            continue
        
        # Load data
        df = analyzer.load_data(input_file, label)
        
        # Build network
        G = analyzer.build_network(df, label)
        
        # Detect communities
        communities = analyzer.detect_communities(G, label, method=args.method)
    
    # Analyze protein participation
    logger.info("Analyzing protein participation")
    participation_dfs = {}
    for label in condition_labels:
        if label in analyzer.networks:
            participation_dfs[label] = analyzer.analyze_protein_participation(label)
    
    # Compare conditions
    logger.info("Comparing conditions")
    comparison_results = analyzer.compare_conditions()
    
    # Save summary statistics
    summary_file = args.output_dir / "clustering_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTERING ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for label in condition_labels:
            if label in analyzer.networks:
                G = analyzer.networks[label]
                communities = analyzer.communities[label]
                f.write(f"\n{label}:\n")
                f.write(f"  Nodes: {G.number_of_nodes()}\n")
                f.write(f"  Edges: {G.number_of_edges()}\n")
                f.write(f"  Communities: {len(communities)}\n")
                f.write(f"  Average community size: {np.mean([len(n) for n in communities.values()]):.2f}\n")
                f.write(f"  Largest community: {max([len(n) for n in communities.values()])}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CROSS-CONDITION COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_results['summary'].to_string(index=False))
        f.write("\n")
    
    logger.info(f"Saved summary to {summary_file}")
    
    # Save detailed results
    for label in condition_labels:
        if label in participation_dfs:
            output_file = args.output_dir / f"{label.lower()}_protein_participation.tsv"
            participation_dfs[label].to_csv(output_file, sep='\t', index=False)
            logger.info(f"Saved participation data to {output_file}")
    
    # Save community assignments
    for label in condition_labels:
        if label in analyzer.communities:
            communities = analyzer.communities[label]
            output_file = args.output_dir / f"{label.lower()}_communities.tsv"
            with open(output_file, 'w') as f:
                f.write("Community_ID\tProteins\n")
                for comm_id, nodes in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True):
                    f.write(f"{comm_id}\t{','.join(sorted(nodes))}\n")
            logger.info(f"Saved communities to {output_file}")
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Network plots
    for label in condition_labels:
        if label in analyzer.networks:
            output_file = args.output_dir / f"{label.lower()}_network.png"
            plot_network_with_communities(
                analyzer.networks[label],
                analyzer.communities[label],
                label,
                output_file,
                max_nodes=args.max_nodes
            )
    
    # Community comparison
    plot_community_comparison(analyzer, args.output_dir / "community_comparison.png")
    
    # Cross-condition heatmap
    plot_cross_condition_heatmap(comparison_results, args.output_dir / "cross_condition_heatmap.png")
    
    # Protein participation
    if participation_dfs:
        plot_protein_participation(participation_dfs, args.output_dir / "protein_participation.png")
    
    # Shared interactions
    plot_shared_interactions(comparison_results, args.output_dir / "shared_interactions.png")
    
    logger.info("=" * 80)
    logger.info("Clustering analysis complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

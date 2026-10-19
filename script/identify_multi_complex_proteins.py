#!/usr/bin/env python3
"""
Identify multi-complex proteins in the network.

This script:
1. Detects communities using multiple algorithms
2. Identifies proteins that participate in multiple complexes
3. Calculates participation coefficients
4. Analyzes bridge proteins
5. Generates a comprehensive report
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12


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
    
    # Identify VIF-dependent (in WT but not in VIF KO)
    df_wt_filtered['VIF_KO_Probability'] = df_wt_filtered['Pair'].apply(
        lambda pair: df_vif[df_vif['Pair'] == pair]['Probability'].values[0] 
        if pair in vif_pairs else 0.0
    )
    
    # Keep only VIF-dependent interactions
    df_vif_dependent = df_wt_filtered[
        (df_wt_filtered['VIF_KO_Probability'] < 0.1) | 
        (df_wt_filtered['Pair'].apply(lambda p: p not in vif_pairs))
    ].copy()
    
    df_vif_dependent = df_vif_dependent.rename(columns={'Probability': 'WT_Probability'})
    
    logger.info(f"  VIF-dependent pairs with prob >= {min_prob}: {len(df_vif_dependent)}")
    
    return df_vif_dependent


def build_network(df: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from interaction data."""
    G = nx.Graph()
    
    for _, row in df.iterrows():
        protein1 = str(row['Protein1']).strip()
        protein2 = str(row['Protein2']).strip()
        prob = float(row['WT_Probability'])
        
        G.add_edge(protein1, protein2, weight=prob, probability=prob)
    
    logger.info(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def detect_communities_louvain(G: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
    """Detect communities using Louvain algorithm."""
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(G, resolution=resolution, seed=42)
        
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        
        logger.info(f"  Detected {len(communities)} communities (resolution={resolution})")
        return node_to_community, communities
    except Exception as e:
        logger.warning(f"  Louvain failed: {e}")
        return {}, []


def detect_communities_leiden(G: nx.Graph, resolution: float = 1.0) -> Dict[str, int]:
    """Detect communities using Leiden algorithm."""
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.leiden_communities(G, resolution=resolution, seed=42)
        
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        
        logger.info(f"  Detected {len(communities)} communities (resolution={resolution})")
        return node_to_community, communities
    except Exception as e:
        logger.warning(f"  Leiden failed: {e}")
        return {}, []


def detect_communities_greedy_modularity(G: nx.Graph) -> Dict[str, int]:
    """Detect communities using greedy modularity maximization."""
    try:
        import networkx.algorithms.community as nx_comm
        communities = nx_comm.greedy_modularity_communities(G)
        
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        
        logger.info(f"  Detected {len(communities)} communities")
        return node_to_community, list(communities)
    except Exception as e:
        logger.warning(f"  Greedy modularity failed: {e}")
        return {}, []


def calculate_participation_coefficient(G: nx.Graph, node_to_community: Dict[str, int]) -> Dict[str, float]:
    """Calculate participation coefficient for each node.
    
    Participation coefficient measures how evenly a node's connections
    are distributed across communities.
    PC = 1 - sum((k_is/k_i)^2) where k_is is connections to community s, k_i is total degree
    PC = 0: all connections within one community
    PC = 1: connections evenly distributed across all communities
    """
    pc = {}
    
    # Get all communities
    all_communities = set(node_to_community.values())
    
    for node in G.nodes():
        degree = G.degree(node)
        if degree == 0:
            pc[node] = 0.0
            continue
        
        # Count connections to each community
        community_degrees = defaultdict(int)
        for neighbor in G.neighbors(node):
            neighbor_comm = node_to_community.get(neighbor, -1)
            community_degrees[neighbor_comm] += 1
        
        # Calculate participation coefficient
        pc_value = 1.0
        for comm_id, comm_degree in community_degrees.items():
            pc_value -= (comm_degree / degree) ** 2
        
        pc[node] = pc_value
    
    return pc


def calculate_within_module_degree(G: nx.Graph, node_to_community: Dict[str, int]) -> Dict[str, float]:
    """Calculate within-module degree z-score.
    
    Measures how well-connected a node is within its own community
    compared to other nodes in that community.
    """
    z_scores = {}
    
    # Calculate statistics for each community
    community_stats = defaultdict(lambda: {'degrees': [], 'mean': 0, 'std': 1})
    
    for node in G.nodes():
        comm_id = node_to_community.get(node, -1)
        degree = G.degree(node)
        community_stats[comm_id]['degrees'].append(degree)
    
    # Calculate mean and std for each community
    for comm_id, stats in community_stats.items():
        if len(stats['degrees']) > 1:
            stats['mean'] = np.mean(stats['degrees'])
            stats['std'] = np.std(stats['degrees'])
        elif len(stats['degrees']) == 1:
            stats['mean'] = stats['degrees'][0]
            stats['std'] = 1.0
    
    # Calculate z-scores
    for node in G.nodes():
        comm_id = node_to_community.get(node, -1)
        degree = G.degree(node)
        stats = community_stats[comm_id]
        
        if stats['std'] > 0:
            z_scores[node] = (degree - stats['mean']) / stats['std']
        else:
            z_scores[node] = 0.0
    
    return z_scores


def identify_multi_community_proteins(node_to_community: Dict[str, int], 
                                      communities: List[Set[str]]) -> Dict[str, List[int]]:
    """Identify proteins that appear in multiple communities.
    
    Note: In standard community detection, each node belongs to exactly one community.
    But we can identify proteins that:
    1. Have connections to multiple communities (high participation coefficient)
    2. Are in communities that share many connections
    """
    # For each protein, find which communities it connects to
    protein_communities = defaultdict(set)
    
    for node in node_to_community.keys():
        # Primary community
        primary_comm = node_to_community[node]
        protein_communities[node].add(primary_comm)
    
    return {node: list(comms) for node, comms in protein_communities.items()}


def analyze_bridge_proteins(G: nx.Graph, node_to_community: Dict[str, int]) -> Dict[str, Dict]:
    """Identify bridge proteins that connect different communities."""
    bridge_info = {}
    
    for node in G.nodes():
        primary_comm = node_to_community.get(node, -1)
        
        # Count connections to different communities
        comm_connections = defaultdict(int)
        for neighbor in G.neighbors(node):
            neighbor_comm = node_to_community.get(neighbor, -1)
            comm_connections[neighbor_comm] += 1
        
        # Calculate metrics
        total_connections = len(comm_connections)
        within_comm_connections = comm_connections.get(primary_comm, 0)
        between_comm_connections = sum(v for k, v in comm_connections.items() if k != primary_comm)
        
        bridge_info[node] = {
            'primary_community': primary_comm,
            'total_communities_connected': total_connections,
            'within_community_connections': within_comm_connections,
            'between_community_connections': between_comm_connections,
            'bridge_ratio': between_comm_connections / G.degree(node) if G.degree(node) > 0 else 0
        }
    
    return bridge_info


def generate_report(G: nx.Graph, node_to_community: Dict[str, int], 
                   communities: List[Set[str]], output_file: Path):
    """Generate comprehensive report on multi-complex proteins."""
    logger.info("Generating multi-complex protein analysis report")
    
    # Calculate metrics
    pc = calculate_participation_coefficient(G, node_to_community)
    z_scores = calculate_within_module_degree(G, node_to_community)
    bridge_info = analyze_bridge_proteins(G, node_to_community)
    
    # Create results dataframe
    results = []
    for node in G.nodes():
        comm_id = node_to_community.get(node, -1)
        degree = G.degree(node)
        
        results.append({
            'Protein': node,
            'Community_ID': comm_id,
            'Community_Size': len(communities[comm_id]) if comm_id < len(communities) else 0,
            'Degree': degree,
            'Participation_Coefficient': pc.get(node, 0.0),
            'Within_Module_Z_Score': z_scores.get(node, 0.0),
            'Bridge_Ratio': bridge_info[node]['bridge_ratio'],
            'Between_Community_Connections': bridge_info[node]['between_community_connections'],
            'Within_Community_Connections': bridge_info[node]['within_community_connections'],
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Participation_Coefficient', ascending=False)
    
    # Save results
    df_results.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Saved results to {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-COMPLEX PROTEIN ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    # High participation coefficient (multi-complex)
    high_pc = df_results[df_results['Participation_Coefficient'] > 0.3]
    logger.info(f"\nProteins with high participation coefficient (>0.3): {len(high_pc)}")
    logger.info("These proteins have connections distributed across multiple communities:")
    for _, row in high_pc.head(10).iterrows():
        logger.info(f"  {row['Protein']}: PC={row['Participation_Coefficient']:.3f}, "
                   f"Bridge_Ratio={row['Bridge_Ratio']:.3f}")
    
    # Bridge proteins
    high_bridge = df_results[df_results['Bridge_Ratio'] > 0.3]
    logger.info(f"\nBridge proteins (Bridge_Ratio > 0.3): {len(high_bridge)}")
    logger.info("These proteins connect different communities:")
    for _, row in high_bridge.head(10).iterrows():
        logger.info(f"  {row['Protein']}: Bridge_Ratio={row['Bridge_Ratio']:.3f}, "
                   f"Between_Comm={row['Between_Community_Connections']}, "
                   f"Within_Comm={row['Within_Community_Connections']}")
    
    # Hub proteins (high degree, high within-module z-score)
    hubs = df_results[(df_results['Degree'] > df_results['Degree'].quantile(0.75)) & 
                      (df_results['Within_Module_Z_Score'] > 1.0)]
    logger.info(f"\nHub proteins (high degree, high within-module connectivity): {len(hubs)}")
    for _, row in hubs.head(10).iterrows():
        logger.info(f"  {row['Protein']}: Degree={row['Degree']}, "
                   f"Z_Score={row['Within_Module_Z_Score']:.2f}")
    
    return df_results


def plot_analysis(df_results: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""
    logger.info("Creating visualization plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Participation Coefficient distribution
    ax = axes[0, 0]
    ax.hist(df_results['Participation_Coefficient'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(df_results['Participation_Coefficient'].median(), color='red', 
               linestyle='--', label=f'Median: {df_results["Participation_Coefficient"].median():.3f}')
    ax.set_xlabel('Participation Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Participation Coefficients', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Participation Coefficient vs Bridge Ratio
    ax = axes[0, 1]
    scatter = ax.scatter(df_results['Participation_Coefficient'], 
                        df_results['Bridge_Ratio'],
                        c=df_results['Degree'], 
                        s=df_results['Degree']*10,
                        alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Participation Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bridge Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Complex Proteins\n(Size = Degree, Color = Degree)', 
                fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Degree')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Within-Module Z-Score vs Participation Coefficient
    ax = axes[1, 0]
    scatter = ax.scatter(df_results['Within_Module_Z_Score'],
                        df_results['Participation_Coefficient'],
                        c=df_results['Bridge_Ratio'],
                        s=100, alpha=0.6, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Within-Module Z-Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Participation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Protein Roles\n(Color = Bridge Ratio)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Bridge Ratio')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Top multi-complex proteins
    ax = axes[1, 1]
    top_multi = df_results.nlargest(15, 'Participation_Coefficient')
    y_pos = np.arange(len(top_multi))
    ax.barh(y_pos, top_multi['Participation_Coefficient'], edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_multi['Protein'], fontsize=10)
    ax.set_xlabel('Participation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Multi-Complex Proteins', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_file = output_dir / 'multi_complex_protein_analysis.png'
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    logger.info(f"Saved plots to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Identify multi-complex proteins in the network',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--wt-file', type=Path,
                       default='results/cullin_wt_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='WT all pairs TSV file')
    parser.add_argument('--vif-file', type=Path,
                       default='results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='VIF KO all pairs TSV file')
    parser.add_argument('--output-dir', type=Path, default='wt_vif_comparison',
                       help='Output directory')
    parser.add_argument('--min-prob', type=float, default=0.6,
                       help='Minimum probability threshold')
    parser.add_argument('--method', type=str, default='louvain',
                       choices=['louvain', 'leiden', 'greedy'],
                       help='Community detection method')
    parser.add_argument('--resolution', type=float, default=1.0,
                       help='Resolution parameter for community detection')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_wt_interactions(args.wt_file, args.vif_file, min_prob=args.min_prob)
    
    if len(df) == 0:
        logger.error(f"No interactions found with probability >= {args.min_prob}")
        return
    
    # Build network
    G = build_network(df)
    
    # Detect communities
    logger.info(f"\nDetecting communities using {args.method} method...")
    if args.method == 'louvain':
        node_to_community, communities = detect_communities_louvain(G, args.resolution)
    elif args.method == 'leiden':
        node_to_community, communities = detect_communities_leiden(G, args.resolution)
    else:
        node_to_community, communities = detect_communities_greedy_modularity(G)
    
    if not node_to_community:
        logger.error("Community detection failed")
        return
    
    # Generate report
    output_file = args.output_dir / 'multi_complex_proteins.tsv'
    df_results = generate_report(G, node_to_community, communities, output_file)
    
    # Create plots
    plot_analysis(df_results, args.output_dir)
    
    logger.info("=" * 80)
    logger.info("Multi-complex protein identification complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

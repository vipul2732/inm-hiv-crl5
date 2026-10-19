#!/usr/bin/env python3
"""
Compare WT (with VIF) vs VIF knockout (ΔVIF) conditions.

This script identifies:
1. VIF-dependent interactions (present in WT but not in VIF knockout)
2. Network topology differences
3. Key protein differences
4. Interaction probability changes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Set, Tuple, List
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_interactions(file_path: Path, condition: str, top_n: int = None, 
                     probability_threshold: float = 0.5) -> pd.DataFrame:
    """Load interaction data from TSV file."""
    logger.info(f"Loading {condition} data from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    
    # Filter by probability threshold
    df = df[df['Probability'] >= probability_threshold].copy()
    
    # Sort by probability descending
    df = df.sort_values('Probability', ascending=False).copy()
    
    # Take top N if specified
    if top_n:
        df = df.head(top_n).copy()
    
    # Create normalized pair identifier (sorted alphabetically)
    df['Pair'] = df.apply(
        lambda row: tuple(sorted([str(row['Protein1']), str(row['Protein2'])])),
        axis=1
    )
    
    logger.info(f"  Loaded {len(df)} pairs for {condition}")
    logger.info(f"  Probability range: {df['Probability'].min():.4f} - {df['Probability'].max():.4f}")
    
    return df


def compare_conditions(wt_df: pd.DataFrame, vif_df: pd.DataFrame) -> Dict:
    """Compare WT and VIF knockout conditions."""
    logger.info("Comparing WT vs VIF knockout conditions")
    
    # Get all pairs
    wt_pairs = set(wt_df['Pair'])
    vif_pairs = set(vif_df['Pair'])
    
    # Find VIF-dependent interactions (in WT but not in VIF)
    vif_dependent = wt_pairs - vif_pairs
    logger.info(f"  VIF-dependent interactions: {len(vif_dependent)}")
    
    # Find shared interactions
    shared = wt_pairs & vif_pairs
    logger.info(f"  Shared interactions: {len(shared)}")
    
    # Find VIF knockout-specific (in VIF but not in WT)
    vif_ko_specific = vif_pairs - wt_pairs
    logger.info(f"  VIF knockout-specific interactions: {len(vif_ko_specific)}")
    
    # Create merged dataframe for shared interactions
    wt_dict = {row['Pair']: row for _, row in wt_df.iterrows()}
    vif_dict = {row['Pair']: row for _, row in vif_df.iterrows()}
    
    shared_data = []
    for pair in shared:
        wt_row = wt_dict[pair]
        vif_row = vif_dict[pair]
        shared_data.append({
            'Protein1': wt_row['Protein1'],
            'Protein2': wt_row['Protein2'],
            'WT_Probability': wt_row['Probability'],
            'VIF_KO_Probability': vif_row['Probability'],
            'Probability_Difference': wt_row['Probability'] - vif_row['Probability'],
            'Pair': pair
        })
    
    shared_df = pd.DataFrame(shared_data)
    
    # Get VIF-dependent interactions with details
    vif_dependent_data = []
    for pair in vif_dependent:
        wt_row = wt_dict[pair]
        vif_dependent_data.append({
            'Protein1': wt_row['Protein1'],
            'Protein1_UniProt': wt_row.get('Protein1_UniProt', ''),
            'Protein2': wt_row['Protein2'],
            'Protein2_UniProt': wt_row.get('Protein2_UniProt', ''),
            'WT_Probability': wt_row['Probability'],
            'VIF_KO_Probability': 0.0,
            'Probability_Difference': wt_row['Probability'],
            'Pair': pair
        })
    
    vif_dependent_df = pd.DataFrame(vif_dependent_data)
    vif_dependent_df = vif_dependent_df.sort_values('WT_Probability', ascending=False)
    
    return {
        'vif_dependent': vif_dependent_df,
        'shared': shared_df,
        'vif_ko_specific': vif_ko_specific,
        'stats': {
            'wt_total': len(wt_pairs),
            'vif_total': len(vif_pairs),
            'shared_count': len(shared),
            'vif_dependent_count': len(vif_dependent),
            'vif_ko_specific_count': len(vif_ko_specific)
        }
    }


def identify_key_proteins(comparison_results: Dict) -> Dict:
    """Identify key proteins involved in VIF-dependent interactions."""
    logger.info("Identifying key proteins in VIF-dependent interactions")
    
    vif_dependent = comparison_results['vif_dependent']
    
    # Count protein appearances
    protein_counts = Counter()
    for _, row in vif_dependent.iterrows():
        protein_counts[row['Protein1']] += 1
        protein_counts[row['Protein2']] += 1
    
    # Get top proteins
    top_proteins = protein_counts.most_common(20)
    
    # Calculate average probabilities for each protein
    protein_avg_prob = defaultdict(list)
    for _, row in vif_dependent.iterrows():
        prob = row['WT_Probability']
        protein_avg_prob[row['Protein1']].append(prob)
        protein_avg_prob[row['Protein2']].append(prob)
    
    protein_summary = []
    for protein, count in top_proteins:
        avg_prob = np.mean(protein_avg_prob[protein])
        max_prob = np.max(protein_avg_prob[protein])
        protein_summary.append({
            'Protein': protein,
            'VIF_Dependent_Interactions': count,
            'Average_Probability': avg_prob,
            'Max_Probability': max_prob
        })
    
    return pd.DataFrame(protein_summary)


def plot_comparison_summary(comparison_results: Dict, output_file: Path):
    """Plot summary comparison of WT vs VIF knockout."""
    logger.info("Creating comparison summary plot")
    
    stats = comparison_results['stats']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Interaction counts
    ax = axes[0]
    categories = ['WT Total', 'VIF KO Total', 'Shared', 'VIF-Dependent', 'VIF KO-Specific']
    counts = [
        stats['wt_total'],
        stats['vif_total'],
        stats['shared_count'],
        stats['vif_dependent_count'],
        stats['vif_ko_specific_count']
    ]
    colors = ['steelblue', 'coral', 'gray', 'red', 'orange']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Number of Interactions', fontsize=12)
    ax.set_title('Interaction Counts: WT vs VIF Knockout', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Probability distribution for shared interactions
    ax = axes[1]
    shared_df = comparison_results['shared']
    if len(shared_df) > 0:
        ax.scatter(shared_df['WT_Probability'], shared_df['VIF_KO_Probability'],
                  alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        min_prob = min(shared_df['WT_Probability'].min(), shared_df['VIF_KO_Probability'].min())
        max_prob = max(shared_df['WT_Probability'].max(), shared_df['VIF_KO_Probability'].max())
        ax.plot([min_prob, max_prob], [min_prob, max_prob], 'r--', alpha=0.5, label='y=x')
        
        ax.set_xlabel('WT Probability', fontsize=12)
        ax.set_ylabel('VIF KO Probability', fontsize=12)
        ax.set_title('Shared Interactions: Probability Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No shared interactions', ha='center', va='center', fontsize=12)
        ax.set_xlabel('WT Probability', fontsize=12)
        ax.set_ylabel('VIF KO Probability', fontsize=12)
        ax.set_title('Shared Interactions: Probability Comparison', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved comparison plot to {output_file}")


def plot_vif_dependent_interactions(vif_dependent_df: pd.DataFrame, output_file: Path, top_n: int = 50):
    """Plot top VIF-dependent interactions."""
    logger.info("Creating VIF-dependent interactions plot")
    
    top_interactions = vif_dependent_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(top_interactions) * 0.15)))
    
    # Create interaction labels
    labels = [f"{row['Protein1']} - {row['Protein2']}" for _, row in top_interactions.iterrows()]
    y_pos = np.arange(len(labels))
    probabilities = top_interactions['WT_Probability'].values
    
    ax.barh(y_pos, probabilities, edgecolor='black', alpha=0.7, color='red')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Interaction Probability (WT)', fontsize=12)
    ax.set_title(f'Top {top_n} VIF-Dependent Interactions\n(Present in WT but absent in VIF Knockout)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add probability values
    for i, prob in enumerate(probabilities):
        ax.text(prob, i, f' {prob:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved VIF-dependent interactions plot to {output_file}")


def plot_key_proteins(protein_summary: pd.DataFrame, output_file: Path):
    """Plot key proteins involved in VIF-dependent interactions."""
    logger.info("Creating key proteins plot")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Number of VIF-dependent interactions per protein
    ax = axes[0]
    top_proteins = protein_summary.head(15)
    y_pos = np.arange(len(top_proteins))
    
    ax.barh(y_pos, top_proteins['VIF_Dependent_Interactions'], 
           edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_proteins['Protein'], fontsize=10)
    ax.set_xlabel('Number of VIF-Dependent Interactions', fontsize=12)
    ax.set_title('Top Proteins by VIF-Dependent Interactions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Average probability vs interaction count
    ax = axes[1]
    ax.scatter(protein_summary['VIF_Dependent_Interactions'], 
              protein_summary['Average_Probability'],
              s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Label top proteins
    top_5 = protein_summary.head(5)
    for _, row in top_5.iterrows():
        ax.annotate(row['Protein'], 
                   (row['VIF_Dependent_Interactions'], row['Average_Probability']),
                   fontsize=9, ha='left')
    
    ax.set_xlabel('Number of VIF-Dependent Interactions', fontsize=12)
    ax.set_ylabel('Average Interaction Probability', fontsize=12)
    ax.set_title('Protein Importance in VIF-Dependent Network', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved key proteins plot to {output_file}")


def generate_report(comparison_results: Dict, protein_summary: pd.DataFrame, output_file: Path):
    """Generate text report of comparison."""
    logger.info("Generating comparison report")
    
    stats = comparison_results['stats']
    vif_dependent = comparison_results['vif_dependent']
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("WT vs VIF KNOCKOUT COMPARISON REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 100 + "\n\n")
        f.write(f"WT (with VIF) total interactions: {stats['wt_total']}\n")
        f.write(f"VIF Knockout total interactions: {stats['vif_total']}\n")
        f.write(f"Shared interactions: {stats['shared_count']} ({stats['shared_count']/stats['wt_total']*100:.1f}% of WT)\n")
        f.write(f"VIF-dependent interactions: {stats['vif_dependent_count']} ({stats['vif_dependent_count']/stats['wt_total']*100:.1f}% of WT)\n")
        f.write(f"VIF KO-specific interactions: {stats['vif_ko_specific_count']}\n\n")
        
        f.write("KEY PROTEINS IN VIF-DEPENDENT INTERACTIONS\n")
        f.write("-" * 100 + "\n\n")
        f.write(protein_summary.to_string(index=False))
        f.write("\n\n")
        
        f.write("TOP 20 VIF-DEPENDENT INTERACTIONS\n")
        f.write("-" * 100 + "\n\n")
        top_20 = vif_dependent.head(20)
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            f.write(f"{idx}. {row['Protein1']} - {row['Protein2']}\n")
            f.write(f"   Probability: {row['WT_Probability']:.4f}\n")
            if row.get('Protein1_UniProt'):
                f.write(f"   UniProt IDs: {row['Protein1_UniProt']} - {row.get('Protein2_UniProt', '')}\n")
            f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    logger.info(f"Saved report to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare WT vs VIF knockout conditions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--wt-file', type=Path, 
                       default='results/cullin_wt_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='WT condition interaction file')
    parser.add_argument('--vif-file', type=Path,
                       default='results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='VIF knockout condition interaction file')
    parser.add_argument('--output-dir', type=Path, default='wt_vif_comparison',
                       help='Output directory for results')
    parser.add_argument('--top-n', type=int, default=500,
                       help='Number of top pairs to analyze from each condition')
    parser.add_argument('--probability-threshold', type=float, default=0.5,
                       help='Minimum probability threshold for edges')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    wt_df = load_interactions(args.wt_file, 'WT', top_n=args.top_n, 
                             probability_threshold=args.probability_threshold)
    vif_df = load_interactions(args.vif_file, 'VIF Knockout', top_n=args.top_n,
                              probability_threshold=args.probability_threshold)
    
    # Compare conditions
    comparison_results = compare_conditions(wt_df, vif_df)
    
    # Identify key proteins
    protein_summary = identify_key_proteins(comparison_results)
    
    # Save results
    comparison_results['vif_dependent'].to_csv(
        args.output_dir / 'vif_dependent_interactions.tsv', sep='\t', index=False
    )
    logger.info(f"Saved VIF-dependent interactions to {args.output_dir / 'vif_dependent_interactions.tsv'}")
    
    comparison_results['shared'].to_csv(
        args.output_dir / 'shared_interactions.tsv', sep='\t', index=False
    )
    logger.info(f"Saved shared interactions to {args.output_dir / 'shared_interactions.tsv'}")
    
    protein_summary.to_csv(
        args.output_dir / 'key_proteins.tsv', sep='\t', index=False
    )
    logger.info(f"Saved key proteins to {args.output_dir / 'key_proteins.tsv'}")
    
    # Generate visualizations
    plot_comparison_summary(comparison_results, args.output_dir / 'comparison_summary.png')
    plot_vif_dependent_interactions(
        comparison_results['vif_dependent'], 
        args.output_dir / 'vif_dependent_interactions.png'
    )
    plot_key_proteins(protein_summary, args.output_dir / 'key_proteins.png')
    
    # Generate report
    generate_report(comparison_results, protein_summary, args.output_dir / 'comparison_report.txt')
    
    logger.info("=" * 80)
    logger.info("WT vs VIF Knockout comparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

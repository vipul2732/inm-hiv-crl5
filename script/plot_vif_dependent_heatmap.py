#!/usr/bin/env python3
"""
Create a heatmap/matrix plot showing VIF-dependent interactions.
Shows protein-protein interaction matrix for pairs present in WT but not in VIF knockout.
"""

import argparse
import logging
from pathlib import Path
from typing import Set, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


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
    """
    Create a symmetric interaction matrix from interaction pairs.
    
    Returns:
        matrix: numpy array of interaction probabilities
        protein_list: sorted list of all proteins
        protein_to_idx: mapping from protein name to matrix index
    """
    # Get all unique proteins
    all_proteins = set()
    for _, row in df.iterrows():
        all_proteins.add(str(row['Protein1']))
        all_proteins.add(str(row['Protein2']))
    
    protein_list = sorted(all_proteins)
    n_proteins = len(protein_list)
    protein_to_idx = {prot: idx for idx, prot in enumerate(protein_list)}
    
    logger.info(f"  Creating matrix for {n_proteins} proteins")
    
    # Initialize matrix with zeros
    matrix = np.zeros((n_proteins, n_proteins))
    
    # Fill matrix with interaction probabilities
    for _, row in df.iterrows():
        prot1 = str(row['Protein1'])
        prot2 = str(row['Protein2'])
        prob = float(row['WT_Probability'])
        
        idx1 = protein_to_idx[prot1]
        idx2 = protein_to_idx[prot2]
        
        # Make symmetric (both directions)
        matrix[idx1, idx2] = prob
        matrix[idx2, idx1] = prob
    
    return matrix, protein_list, protein_to_idx


def plot_heatmap(matrix: np.ndarray, protein_list: list, output_file: Path, 
                 figsize: tuple = (14, 12), cmap: str = 'YlOrRd'):
    """Plot interaction matrix as heatmap."""
    logger.info(f"Creating heatmap with {len(protein_list)} proteins")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=matrix.max())
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(protein_list)))
    ax.set_yticks(np.arange(len(protein_list)))
    ax.set_xticklabels(protein_list, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(protein_list, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Interaction Probability (HIV WT type infection)', 
                        fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    # Add title
    ax.set_title('VIF-Dependent Interactions\n(Present in HIV WT type infection, Absent in HIV Δ VIF infection)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(protein_list)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(protein_list)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add text annotations for non-zero values (optional, can be cluttered)
    # Only add for high-probability interactions to avoid clutter
    threshold = matrix.max() * 0.9  # Only annotate top 10%
    for i in range(len(protein_list)):
        for j in range(len(protein_list)):
            if matrix[i, j] > threshold and i != j:
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6,
                             fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved heatmap to {output_file}")


def plot_heatmap_clustered(matrix: np.ndarray, protein_list: list, output_file: Path,
                           figsize: tuple = (14, 12), cmap: str = 'YlOrRd'):
    """Plot clustered heatmap using seaborn."""
    logger.info(f"Creating clustered heatmap with {len(protein_list)} proteins")
    
    # Create DataFrame
    df_matrix = pd.DataFrame(matrix, index=protein_list, columns=protein_list)
    
    # Use seaborn's clustermap for better clustering
    g = sns.clustermap(df_matrix, 
                      cmap=cmap,
                      vmin=0,
                      vmax=matrix.max(),
                      figsize=figsize,
                      cbar_kws={'label': 'Interaction Probability (HIV WT type infection)'},
                      xticklabels=True,
                      yticklabels=True,
                      linewidths=0.5,
                      linecolor='gray')
    
    g.ax_row_dendrogram.remove()
    g.ax_col_dendrogram.remove()
    
    g.fig.suptitle('VIF-dependent protein interactions\n(HIV WT type infection vs HIV Δ VIF infection)',
                  fontsize=16, fontweight='bold', y=0.98)
    
    # Rotate labels and make them bold and bigger
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='right', fontsize=16, fontweight='bold')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16, fontweight='bold')
    
    # Make colorbar labels bold and bigger
    g.cax.tick_params(labelsize=16)
    for label in g.cax.get_yticklabels():
        label.set_fontweight('bold')
    g.cax.set_ylabel('Interaction Probability (HIV WT type infection)', fontsize=16, fontweight='bold')
    
    # Add axis labels
    g.ax_heatmap.set_xlabel('Proteins', fontsize=16, fontweight='bold')
    g.ax_heatmap.set_ylabel('Proteins', fontsize=16, fontweight='bold')
    
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved clustered heatmap to {output_file}")


def highlight_key_proteins(protein_list: list) -> Dict[str, bool]:
    """Identify key proteins for highlighting."""
    key_proteins = {
        'vifprotein', 'CUL5', 'CUL2', 'DCA11', 'ELOB', 'ELOC', 
        'RBX1', 'RBX2', 'CBFB', 'PDC6I', 'NEDD8'
    }
    return {prot: prot in key_proteins for prot in protein_list}


def plot_heatmap_with_highlights(matrix: np.ndarray, protein_list: list, 
                                 output_file: Path, figsize: tuple = (14, 12),
                                 cmap: str = 'YlOrRd'):
    """Plot heatmap with key proteins highlighted."""
    logger.info(f"Creating heatmap with highlights for {len(protein_list)} proteins")
    
    # Identify key proteins
    key_proteins = highlight_key_proteins(protein_list)
    
    # Reorder proteins: key proteins first, then others
    key_prot_list = [p for p in protein_list if key_proteins[p]]
    other_prot_list = [p for p in protein_list if not key_proteins[p]]
    reordered_list = key_prot_list + other_prot_list
    
    # Reorder matrix
    protein_to_idx = {prot: idx for idx, prot in enumerate(protein_list)}
    reordered_indices = [protein_to_idx[p] for p in reordered_list]
    reordered_matrix = matrix[np.ix_(reordered_indices, reordered_indices)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(reordered_matrix, cmap=cmap, aspect='auto', 
                  vmin=0, vmax=matrix.max())
    
    # Add divider line between key and other proteins
    n_key = len(key_prot_list)
    if n_key > 0 and n_key < len(protein_list):
        ax.axhline(y=n_key - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax.axvline(x=n_key - 0.5, color='blue', linewidth=2, linestyle='--', alpha=0.7)
    
    # Set ticks and labels - make them bold and bigger
    ax.set_xticks(np.arange(len(reordered_list)))
    ax.set_yticks(np.arange(len(reordered_list)))
    ax.set_xticklabels(reordered_list, rotation=90, ha='right', fontsize=16, fontweight='bold')
    ax.set_yticklabels(reordered_list, fontsize=16, fontweight='bold')
    
    # Highlight key protein labels
    for i, protein in enumerate(reordered_list):
        if key_proteins[protein]:
            ax.get_xticklabels()[i].set_color('darkblue')
            ax.get_yticklabels()[i].set_color('darkblue')
    
    # Add colorbar - position it properly
    cbar = plt.colorbar(im, ax=ax, label='Interaction Probability (HIV WT type infection)',
                       fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label('Interaction Probability (HIV WT type infection)', fontsize=16, fontweight='bold')
    
    # Add title - simplified
    ax.set_title('VIF-dependent protein interactions\n(HIV WT type infection vs HIV Δ VIF infection)',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(reordered_list)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(reordered_list)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add axis labels - make them bold and bigger
    ax.set_xlabel('Proteins', fontsize=16, fontweight='bold')
    ax.set_ylabel('Proteins', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved highlighted heatmap to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create heatmap of VIF-dependent interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-file', type=Path,
                       default='wt_vif_comparison/vif_dependent_interactions.tsv',
                       help='VIF-dependent interactions TSV file')
    parser.add_argument('--output-dir', type=Path, default='wt_vif_comparison',
                       help='Output directory for plots')
    parser.add_argument('--clustered', action='store_true',
                       help='Create clustered heatmap')
    parser.add_argument('--highlight', action='store_true', default=True,
                       help='Highlight key proteins')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_vif_dependent_interactions(args.input_file)
    
    # Create interaction matrix
    matrix, protein_list, protein_to_idx = create_interaction_matrix(df)
    
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Non-zero interactions: {np.count_nonzero(matrix)}")
    logger.info(f"Max probability: {matrix.max():.4f}")
    logger.info(f"Mean probability (non-zero): {matrix[matrix > 0].mean():.4f}")
    
    # Create plots
    if args.clustered:
        plot_heatmap_clustered(matrix, protein_list, 
                              args.output_dir / 'vif_dependent_heatmap_clustered.png')
    else:
        if args.highlight:
            plot_heatmap_with_highlights(matrix, protein_list,
                                       args.output_dir / 'vif_dependent_heatmap.png')
        else:
            plot_heatmap(matrix, protein_list,
                        args.output_dir / 'vif_dependent_heatmap.png')
    
    logger.info("=" * 80)
    logger.info("Heatmap generation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create a heatmap/matrix plot showing VIF knockout interactions.
Shows protein-protein interaction matrix for pairs present in VIF knockout condition.
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


def load_vif_ko_interactions(vif_file: Path, min_prob: float = 0.6) -> pd.DataFrame:
    """Load VIF KO interactions with probability >= min_prob."""
    logger.info(f"Loading VIF KO interactions from {vif_file}")
    df = pd.read_csv(vif_file, sep='\t')
    # Replace PEBB with CBFB
    df['Protein1'] = df['Protein1'].replace('PEBB', 'CBFB')
    df['Protein2'] = df['Protein2'].replace('PEBB', 'CBFB')
    
    # Filter by probability threshold
    df = df[df['Probability'] >= min_prob].copy()
    df = df.sort_values('Probability', ascending=False).copy()
    
    logger.info(f"  Loaded {len(df)} VIF KO interactions with prob >= {min_prob}")
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
        prob = float(row['Probability'])
        
        idx1 = protein_to_idx[prot1]
        idx2 = protein_to_idx[prot2]
        
        # Make symmetric (both directions)
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
    cbar = plt.colorbar(im, ax=ax, label='Interaction Probability (HIV Δ VIF infection)',
                       fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.set_label('Interaction Probability (HIV Δ VIF infection)', fontsize=16, fontweight='bold')
    
    # Add title - simplified
    ax.set_title('HIV Δ VIF infection',
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


def plot_heatmap_clustered(matrix: np.ndarray, protein_list: list, output_file: Path,
                           figsize: tuple = (14, 12), cmap: str = 'YlOrRd', title: str = None):
    """Plot clustered heatmap using seaborn."""
    logger.info(f"Creating clustered heatmap with {len(protein_list)} proteins")
    
    # Adjust figure size based on number of proteins to accommodate all labels
    n_proteins = len(protein_list)
    if n_proteins > 50:
        # For large networks, increase figure size significantly to fit all labels
        figsize = (max(24, n_proteins * 0.35), max(22, n_proteins * 0.35))
        # Use smaller but readable font size for labels
        label_fontsize = 7
        # Show ALL labels
        show_every_n = 1
    elif n_proteins > 30:
        figsize = (20, 18)
        label_fontsize = 8
        show_every_n = 1
    else:
        figsize = (14, 12)
        label_fontsize = 10
        show_every_n = 1
    
    # Create DataFrame
    df_matrix = pd.DataFrame(matrix, index=protein_list, columns=protein_list)
    
    # Key proteins to always show
    key_proteins = {'CUL5', 'CUL2', 'ELOB', 'ELOC', 'RBX1', 'RBX2', 
                    'CBFB', 'NEDD8', 'CAND1', 'CSN1', 'CSN2', 'CSN3', 
                    'CSN4', 'CSN5', 'CSN6', 'CSN7A', 'CSN7B', 'CSN8',
                    'PAF1', 'CTR9', 'LEO1', 'CDC73', 'WDR61', 'RUNX1', 
                    'RUNX2', 'RUNX3'}
    
    # Use seaborn's clustermap for better clustering
    # Start with no labels, we'll add them selectively after
    g = sns.clustermap(df_matrix, 
                      cmap=cmap,
                      vmin=0,
                      vmax=matrix.max(),
                      figsize=figsize,
                      cbar_kws={'label': 'Interaction Probability (HIV Δ VIF infection)'},
                      xticklabels=False,  # Hide initially
                      yticklabels=False,  # Hide initially
                      linewidths=0.3,  # Thinner lines for less visual clutter
                      linecolor='gray')
    
    g.ax_row_dendrogram.remove()
    g.ax_col_dendrogram.remove()
    
    if title is None:
        title = 'HIV Δ VIF infection'
    g.fig.suptitle(title,
                  fontsize=18, fontweight='bold', y=0.98)
    
    # Get reordered protein list from clustering
    try:
        # Get reordered indices from the dendrogram
        if hasattr(g, 'dendrogram_row') and hasattr(g.dendrogram_row, 'reordered_ind'):
            reordered_indices = g.dendrogram_row.reordered_ind
        elif hasattr(g, 'dendrogram_col') and hasattr(g.dendrogram_col, 'reordered_ind'):
            reordered_indices = g.dendrogram_col.reordered_ind
        else:
            reordered_indices = list(range(len(protein_list)))
        reordered_proteins = [protein_list[i] for i in reordered_indices]
    except:
        reordered_proteins = protein_list
    
    # Create labels - show ALL proteins
    x_labels = reordered_proteins
    y_labels = reordered_proteins
    
    # Set the labels
    g.ax_heatmap.set_xticks(range(len(reordered_proteins)))
    g.ax_heatmap.set_xticklabels(x_labels, rotation=90, ha='right', 
                                  fontsize=label_fontsize, fontweight='bold')
    g.ax_heatmap.set_yticks(range(len(reordered_proteins)))
    g.ax_heatmap.set_yticklabels(y_labels, rotation=0, 
                                 fontsize=label_fontsize, fontweight='bold')
    
    # Highlight key protein labels
    for i, protein in enumerate(reordered_proteins):
        if protein in key_proteins:
            try:
                g.ax_heatmap.get_xticklabels()[i].set_color('darkblue')
                g.ax_heatmap.get_yticklabels()[i].set_color('darkblue')
            except:
                pass
    
    # Make colorbar labels bold and bigger
    g.cax.tick_params(labelsize=14)
    for label in g.cax.get_yticklabels():
        label.set_fontweight('bold')
    g.cax.set_ylabel('Interaction Probability (HIV Δ VIF infection)', fontsize=14, fontweight='bold')
    
    # Add axis labels
    g.ax_heatmap.set_xlabel('Proteins', fontsize=14, fontweight='bold')
    g.ax_heatmap.set_ylabel('Proteins', fontsize=14, fontweight='bold')
    
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    logger.info(f"Saved clustered heatmap to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create heatmap of VIF knockout interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--vif-file', type=Path,
                       default='results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='VIF KO all pairs TSV file')
    parser.add_argument('--output-dir', type=Path, default='wt_vif_comparison',
                       help='Output directory for plots')
    parser.add_argument('--min-prob', type=float, default=0.6,
                       help='Minimum probability threshold')
    parser.add_argument('--clustered', action='store_true',
                       help='Create clustered heatmap')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_vif_ko_interactions(args.vif_file, min_prob=args.min_prob)
    
    if len(df) == 0:
        logger.error(f"No interactions found with probability >= {args.min_prob}")
        return
    
    # Create interaction matrix
    matrix, protein_list, protein_to_idx = create_interaction_matrix(df)
    
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Non-zero interactions: {np.count_nonzero(matrix)}")
    logger.info(f"Max probability: {matrix.max():.4f}")
    logger.info(f"Mean probability (non-zero): {matrix[matrix > 0].mean():.4f}")
    
    # Create plots
    if args.clustered:
        plot_heatmap_clustered(matrix, protein_list, 
                              args.output_dir / 'vif_ko_heatmap_clustered.png')
    else:
        plot_heatmap_with_highlights(matrix, protein_list,
                                   args.output_dir / 'vif_ko_heatmap.png')
    
    logger.info("=" * 80)
    logger.info("VIF KO Heatmap generation complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

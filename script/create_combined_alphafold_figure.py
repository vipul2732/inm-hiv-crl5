#!/usr/bin/env python3
"""
Create a single combined figure with 3 panels (WT, VIF, Mock) showing crucial AlphaFold prediction scores.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_summary_file(file_path: Path) -> pd.DataFrame:
    """Load AlphaFold summary TSV file."""
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    logger.info(f"  Loaded {len(df)} pairs")
    return df


def create_combined_figure(wt_pos: pd.DataFrame, wt_neg: pd.DataFrame,
                           vif_pos: pd.DataFrame, vif_neg: pd.DataFrame,
                           mock_pos: pd.DataFrame, mock_neg: pd.DataFrame,
                           output_dir: Path):
    """Create a single figure with 3 panels (WT, VIF, Mock) showing crucial prediction scores."""
    logger.info("Creating combined figure with 3 panels")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    conditions_data = [
        ('Wildtype', wt_pos, wt_neg, axes[0]),
        ('VIF', vif_pos, vif_neg, axes[1]),
        ('Mock', mock_pos, mock_neg, axes[2]),
    ]
    
    for condition, pos_df, neg_df, ax in conditions_data:
        # Box plot for ipTM, pLDDT, and PAE - the most crucial metrics
        # Note: PAE is "lower is better" metric, so positive set should have lower values
        data_to_plot = [
            pos_df['ipTM'].values,
            neg_df['ipTM'].values,
            pos_df['mean_pLDDT'].values / 100,  # Normalize for visibility
            neg_df['mean_pLDDT'].values / 100,
            pos_df['mean_interface_PAE'].values / 40,  # Normalize PAE (divide by ~40 to scale to 0-1 range)
            neg_df['mean_interface_PAE'].values / 40,
        ]
        
        bp = ax.boxplot(data_to_plot, 
                         labels=['ipTM\nPos', 'ipTM\nNeg', 'pLDDT\nPos', 'pLDDT\nNeg', 'PAE\nPos', 'PAE\nNeg'],
                         patch_artist=True, widths=0.5, showmeans=False)
        
        colors = ['lightgreen', 'lightcoral', 'lightblue', 'lightpink', 'lightyellow', 'wheat']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Make median lines thicker and more visible
        for median in bp['medians']:
            median.set_linewidth(2)
            median.set_color('black')
        
        ax.set_ylabel('Score', fontsize=20, fontweight='bold')
        ax.set_title(f'{condition}', fontsize=20, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='y')
        
        # Set x-axis labels with bold, size 20
        ax.tick_params(axis='x', labelsize=20, width=2)
        ax.tick_params(axis='y', labelsize=20, width=2)
        
        # Make all labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
    
    plt.tight_layout()
    output_file = output_dir / 'alphafold_combined_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined comparison plot to {output_file}")


def main():
    output_dir = Path("Alphafold_multimer")
    
    # Load all data files
    wt_pos = load_summary_file(output_dir / "colabfold_multimer_summary_wildtype_positive_set.tsv")
    wt_neg = load_summary_file(output_dir / "colabfold_multimer_summary_wildtype_negative_set.tsv")
    vif_pos = load_summary_file(output_dir / "colabfold_multimer_summary_vif_positive_Set.tsv")
    vif_neg = load_summary_file(output_dir / "colabfold_multimer_summary_vif_negative_set.tsv")
    mock_pos = load_summary_file(output_dir / "colabfold_multimer_summary_mock_positve_set.tsv")
    mock_neg = load_summary_file(output_dir / "colabfold_multimer_summary_mock_negative_set.tsv")
    
    # Create combined figure
    create_combined_figure(wt_pos, wt_neg, vif_pos, vif_neg, mock_pos, mock_neg, output_dir)
    
    print("\n" + "="*80)
    print("Combined figure created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

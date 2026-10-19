#!/usr/bin/env python3
"""
Analyze trends in AlphaFold multimer predictions comparing positive (high probability)
and negative (low probability) pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_summary_file(file_path: Path) -> pd.DataFrame:
    """Load AlphaFold summary TSV file."""
    logger.info(f"Loading {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    logger.info(f"  Loaded {len(df)} pairs")
    return df


def calculate_statistics(df: pd.DataFrame, label: str) -> dict:
    """Calculate summary statistics for key metrics."""
    stats = {
        'label': label,
        'n_pairs': len(df),
        'ipTM_mean': df['ipTM'].mean(),
        'ipTM_median': df['ipTM'].median(),
        'ipTM_std': df['ipTM'].std(),
        'pTM_mean': df['pTM'].mean(),
        'pTM_median': df['pTM'].median(),
        'pTM_std': df['pTM'].std(),
        'mean_pLDDT_mean': df['mean_pLDDT'].mean(),
        'mean_pLDDT_median': df['mean_pLDDT'].median(),
        'mean_pLDDT_std': df['mean_pLDDT'].std(),
        'mean_interface_PAE_mean': df['mean_interface_PAE'].mean(),
        'mean_interface_PAE_median': df['mean_interface_PAE'].median(),
        'mean_interface_PAE_std': df['mean_interface_PAE'].std(),
        'frac_interface_PAE_lt10_mean': df['frac_interface_PAE_lt10'].mean(),
        'frac_interface_PAE_lt10_median': df['frac_interface_PAE_lt10'].median(),
        'frac_interface_PAE_lt15_mean': df['frac_interface_PAE_lt15'].mean(),
        'frac_interface_PAE_lt15_median': df['frac_interface_PAE_lt15'].median(),
    }
    return stats


def create_single_category_plot(df: pd.DataFrame, category: str, output_dir: Path, condition: str):
    """Create plots for a single category (positive or negative)."""
    logger.info(f"Creating {category} set plots for {condition}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'AlphaFold Multimer Predictions: {condition} - {category.title()} Set (n={len(df)})', 
                 fontsize=16, fontweight='bold')
    
    color = 'green' if category == 'positive' else 'red'
    
    # 1. ipTM distribution
    ax = axes[0, 0]
    ax.hist(df['ipTM'], bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel('ipTM (Interface Predicted TM-score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Interface TM-score Distribution\nMean: {df["ipTM"].mean():.3f}, Median: {df["ipTM"].median():.3f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 2. pTM distribution
    ax = axes[0, 1]
    ax.hist(df['pTM'], bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel('pTM (Predicted TM-score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Predicted TM-score Distribution\nMean: {df["pTM"].mean():.3f}, Median: {df["pTM"].median():.3f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 3. mean_pLDDT distribution
    ax = axes[0, 2]
    ax.hist(df['mean_pLDDT'], bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel('mean pLDDT (Confidence Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Confidence Score Distribution\nMean: {df["mean_pLDDT"].mean():.1f}, Median: {df["mean_pLDDT"].median():.1f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 4. mean_interface_PAE distribution
    ax = axes[1, 0]
    ax.hist(df['mean_interface_PAE'], bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel('Mean Interface PAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Interface Precision Distribution\nMean: {df["mean_interface_PAE"].mean():.2f}, Median: {df["mean_interface_PAE"].median():.2f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 5. Fraction with PAE < 10
    ax = axes[1, 1]
    ax.hist(df['frac_interface_PAE_lt10'], bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel('Fraction Interface PAE < 10', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'High Precision Interface Fraction\nMean: {df["frac_interface_PAE_lt10"].mean():.4f}, Median: {df["frac_interface_PAE_lt10"].median():.4f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 6. Fraction with PAE < 15
    ax = axes[1, 2]
    ax.hist(df['frac_interface_PAE_lt15'], bins=30, alpha=0.7, color=color, edgecolor='black')
    ax.set_xlabel('Fraction Interface PAE < 15', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Medium Precision Interface Fraction\nMean: {df["frac_interface_PAE_lt15"].mean():.4f}, Median: {df["frac_interface_PAE_lt15"].median():.4f}', 
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / f'alphafold_{category}_{condition.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {category} plot to {output_file}")


def create_combined_figure(wt_pos: pd.DataFrame, wt_neg: pd.DataFrame,
                           vif_pos: pd.DataFrame, vif_neg: pd.DataFrame,
                           mock_pos: pd.DataFrame, mock_neg: pd.DataFrame,
                           output_dir: Path):
    """Create a single figure with 3 panels (WT, VIF, Mock) showing crucial prediction scores."""
    logger.info("Creating combined figure with 3 panels")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('AlphaFold Multimer Predictions: Positive vs Negative Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    conditions = [
        ('Wildtype', wt_pos, wt_neg, axes[0]),
        ('VIF', vif_pos, vif_neg, axes[1]),
        ('Mock', mock_pos, mock_neg, axes[2]),
    ]
    
    for condition, pos_df, neg_df, ax in conditions:
        # Create a 2x2 subplot within each panel for key metrics
        ax.axis('off')
        gs = ax.get_gridspec()
        
        # Create sub-axes for this condition
        ax1 = fig.add_subplot(gs[0, conditions.index((condition, pos_df, neg_df, ax))])
        
        # Show the most crucial metrics: pTM and mean_pLDDT
        # Use violin plots or box plots to show distribution
        
        # Box plot for pTM and pLDDT
        data_to_plot = [
            pos_df['pTM'].values,
            neg_df['pTM'].values,
            pos_df['mean_pLDDT'].values / 100,  # Normalize for visibility
            neg_df['mean_pLDDT'].values / 100,
        ]
        
        bp = ax1.boxplot(data_to_plot, 
                         labels=['Pos\npTM', 'Neg\npTM', 'Pos\npLDDT/100', 'Neg\npLDDT/100'],
                         patch_artist=True, widths=0.6)
        
        colors = ['lightgreen', 'lightcoral', 'lightblue', 'lightpink']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'{condition}\n(n={len(pos_df)} pos, {len(neg_df)} neg)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')
        ax1.tick_params(axis='x', labelsize=10)
        
        # Add mean values as text
        pos_pTM_mean = pos_df['pTM'].mean()
        neg_pTM_mean = neg_df['pTM'].mean()
        pos_pLDDT_mean = pos_df['mean_pLDDT'].mean()
        neg_pLDDT_mean = neg_df['mean_pLDDT'].mean()
        
        ax1.text(0.5, 0.95, f'pTM: {pos_pTM_mean:.3f} vs {neg_pTM_mean:.3f}', 
                transform=ax1.transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.text(0.5, 0.88, f'pLDDT: {pos_pLDDT_mean:.1f} vs {neg_pLDDT_mean:.1f}', 
                transform=ax1.transAxes, ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / 'alphafold_combined_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined comparison plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trends in AlphaFold multimer predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--positive-file",
        type=Path,
        required=True,
        help="Path to positive set summary TSV",
    )
    parser.add_argument(
        "--negative-file",
        type=Path,
        required=True,
        help="Path to negative set summary TSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="Alphafold_multimer",
        help="Output directory",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="",
        help="Condition name (e.g., 'Wildtype', 'VIF')",
    )
    
    args = parser.parse_args()
    
    # Load data
    positive_df = load_summary_file(args.positive_file)
    negative_df = load_summary_file(args.negative_file)
    
    # Calculate statistics
    pos_stats = calculate_statistics(positive_df, "Positive")
    neg_stats = calculate_statistics(negative_df, "Negative")
    
    # Print comparison
    print("\n" + "="*80)
    print(f"ALPHAFOLD MULTIMER PREDICTION ANALYSIS: {args.condition}")
    print("="*80)
    print(f"\nPositive Set (High Probability): {pos_stats['n_pairs']} pairs")
    print(f"Negative Set (Low Probability): {neg_stats['n_pairs']} pairs")
    print("\n" + "-"*80)
    print("METRIC COMPARISON")
    print("-"*80)
    
    metrics = [
        ('ipTM', 'Interface TM-score (higher is better)'),
        ('pTM', 'Predicted TM-score (higher is better)'),
        ('mean_pLDDT', 'Confidence Score (higher is better)'),
        ('mean_interface_PAE', 'Interface PAE (lower is better)'),
        ('frac_interface_PAE_lt10', 'Fraction PAE < 10 (higher is better)'),
        ('frac_interface_PAE_lt15', 'Fraction PAE < 15 (higher is better)'),
    ]
    
    for metric, description in metrics:
        pos_mean = pos_stats[f'{metric}_mean']
        neg_mean = neg_stats[f'{metric}_mean']
        diff = pos_mean - neg_mean
        pct_diff = (diff / neg_mean * 100) if neg_mean != 0 else 0
        
        print(f"\n{description}:")
        print(f"  Positive: {pos_mean:.4f} (median: {pos_stats[f'{metric}_median']:.4f})")
        print(f"  Negative: {neg_mean:.4f} (median: {neg_stats[f'{metric}_median']:.4f})")
        print(f"  Difference: {diff:+.4f} ({pct_diff:+.2f}%)")
    
    # Create comparison plot
    condition_name = args.condition or args.positive_file.stem.replace('colabfold_multimer_summary_', '').replace('_', ' ').title()
    create_comparison_plot(positive_df, negative_df, args.output_dir, condition_name)
    
    # Save statistics
    stats_df = pd.DataFrame([pos_stats, neg_stats])
    stats_file = args.output_dir / f'alphafold_statistics_{condition_name.lower().replace(" ", "_")}.tsv'
    stats_df.to_csv(stats_file, sep='\t', index=False)
    logger.info(f"Saved statistics to {stats_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()

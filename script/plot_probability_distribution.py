#!/usr/bin/env python3
"""
Plot probability distribution for ALL edge pairs (not just top N).

This script:
1. Loads merged MCMC chains
2. Extracts edge probabilities for ALL pairs
3. Creates distribution plots (histogram, density, etc.)
4. Saves plots and statistics

Usage:
    python plot_probability_distribution.py --merged-chains merged/merged_chains.pkl --model-data seed_13/0_model23_n_13_model_data.pkl --output merged/
"""

import argparse
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, Optional
import warnings

import numpy as np
import jax
import jax.numpy as jnp

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")

try:
    from scipy.special import expit  # sigmoid function
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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


def extract_all_probabilities(merged_samples: Dict, model_data: Dict, logger: logging.Logger) -> np.ndarray:
    """
    Extract probabilities for ALL edge pairs.
    
    Args:
        merged_samples: Dictionary with 'z' key containing samples
        model_data: Model data dictionary with 'N', 'M', etc.
        logger: Logger instance
    
    Returns:
        probabilities: 1D array of probabilities for all edges
    """
    logger.info("Extracting edge probabilities from merged chains...")
    
    # Get z samples
    z_samples = merged_samples['z']
    
    # Convert to numpy if needed
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    # Handle shape: might be (n_samples, M) or (n_chains, n_samples, M)
    if len(z_samples.shape) == 3:
        # Multiple chains: flatten to (n_chains * n_samples, M)
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif len(z_samples.shape) == 2:
        # Single chain: (n_samples, M)
        pass
    else:
        raise ValueError(f"Unexpected z_samples shape: {z_samples.shape}")
    
    # Convert continuous z to probabilities using sigmoid
    if HAS_SCIPY:
        z_probs = expit(z_samples)  # Maps z to probability space [0, 1]
    else:
        # Fallback to numpy implementation of sigmoid
        z_probs = 1.0 / (1.0 + np.exp(-z_samples))  # Sigmoid: 1/(1+exp(-z))
    
    # Compute mean probability across samples
    logger.info(f"Computing mean probabilities from {z_samples.shape[0]} samples")
    mean_z_probs = np.mean(z_probs, axis=0)
    
    logger.info(f"Extracted {len(mean_z_probs)} edge probabilities")
    logger.info(f"  Min: {mean_z_probs.min():.4f}")
    logger.info(f"  Max: {mean_z_probs.max():.4f}")
    logger.info(f"  Mean: {mean_z_probs.mean():.4f}")
    logger.info(f"  Median: {np.median(mean_z_probs):.4f}")
    logger.info(f"  Std: {mean_z_probs.std():.4f}")
    
    return mean_z_probs


def plot_probability_distribution(probabilities: np.ndarray, output_file: Path, 
                                 title: str = "Edge Probability Distribution",
                                 logger: Optional[logging.Logger] = None):
    """
    Create probability distribution plots.
    
    Args:
        probabilities: 1D array of probabilities
        output_file: Path to save plot
        title: Plot title
        logger: Optional logger
    """
    if not HAS_MATPLOTLIB:
        if logger:
            logger.warning("matplotlib not available. Skipping plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Histogram
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(probabilities, bins=50, alpha=0.7, color='#2E86AB', 
                                edgecolor='#1B5E7A', linewidth=1.5, density=False)
    ax1.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Histogram', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axvline(np.mean(probabilities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(probabilities):.4f}')
    ax1.axvline(np.median(probabilities), color='green', linestyle='--', linewidth=2, 
                label=f'Median = {np.median(probabilities):.4f}')
    ax1.legend(fontsize=10)
    
    # Add statistics text
    stats_text = f'N = {len(probabilities)}\n'
    stats_text += f'Min = {np.min(probabilities):.4f}\n'
    stats_text += f'Max = {np.max(probabilities):.4f}\n'
    stats_text += f'Mean = {np.mean(probabilities):.4f}\n'
    stats_text += f'Median = {np.median(probabilities):.4f}\n'
    stats_text += f'Std = {np.std(probabilities):.4f}'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Density plot (normalized histogram)
    ax2 = axes[0, 1]
    ax2.hist(probabilities, bins=50, alpha=0.7, color='#2E86AB', 
             edgecolor='#1B5E7A', linewidth=1.5, density=True)
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Density Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axvline(np.mean(probabilities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {np.mean(probabilities):.4f}')
    ax2.axvline(np.median(probabilities), color='green', linestyle='--', linewidth=2, 
                label=f'Median = {np.median(probabilities):.4f}')
    ax2.legend(fontsize=10)
    
    # 3. Cumulative distribution function
    ax3 = axes[1, 0]
    sorted_probs = np.sort(probabilities)
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax3.plot(sorted_probs, cumulative, linewidth=2, color='#2E86AB')
    ax3.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 1])
    
    # Add percentile lines
    for percentile in [10, 25, 50, 75, 90, 95, 99]:
        prob_val = np.percentile(probabilities, percentile)
        ax3.axvline(prob_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax3.text(prob_val, 0.02, f'{percentile}%', rotation=90, 
                verticalalignment='bottom', fontsize=8)
    
    # 4. Box plot
    ax4 = axes[1, 1]
    bp = ax4.boxplot(probabilities, vert=True, patch_artist=True, 
                     boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='#1B5E7A', linewidth=1.5),
                     capprops=dict(color='#1B5E7A', linewidth=1.5))
    ax4.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax4.set_title('Box Plot', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax4.set_xticklabels(['All Pairs'])
    
    # Add statistics as text
    q1, median, q3 = np.percentile(probabilities, [25, 50, 75])
    iqr = q3 - q1
    stats_text = f'Q1 = {q1:.4f}\n'
    stats_text += f'Median = {median:.4f}\n'
    stats_text += f'Q3 = {q3:.4f}\n'
    stats_text += f'IQR = {iqr:.4f}'
    ax4.text(0.98, 0.02, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Saved probability distribution plot: {output_file}")


def save_probability_statistics(probabilities: np.ndarray, output_file: Path, 
                               logger: Optional[logging.Logger] = None):
    """Save probability statistics to JSON file."""
    stats = {
        'n_pairs': int(len(probabilities)),
        'min': float(np.min(probabilities)),
        'max': float(np.max(probabilities)),
        'mean': float(np.mean(probabilities)),
        'median': float(np.median(probabilities)),
        'std': float(np.std(probabilities)),
        'percentiles': {
            'p1': float(np.percentile(probabilities, 1)),
            'p5': float(np.percentile(probabilities, 5)),
            'p10': float(np.percentile(probabilities, 10)),
            'p25': float(np.percentile(probabilities, 25)),
            'p50': float(np.percentile(probabilities, 50)),
            'p75': float(np.percentile(probabilities, 75)),
            'p90': float(np.percentile(probabilities, 90)),
            'p95': float(np.percentile(probabilities, 95)),
            'p99': float(np.percentile(probabilities, 99)),
        },
        'counts_by_threshold': {
            'p_gt_0.9': int(np.sum(probabilities > 0.9)),
            'p_gt_0.8': int(np.sum(probabilities > 0.8)),
            'p_gt_0.7': int(np.sum(probabilities > 0.7)),
            'p_gt_0.6': int(np.sum(probabilities > 0.6)),
            'p_gt_0.5': int(np.sum(probabilities > 0.5)),
            'p_gt_0.4': int(np.sum(probabilities > 0.4)),
            'p_gt_0.3': int(np.sum(probabilities > 0.3)),
            'p_gt_0.2': int(np.sum(probabilities > 0.2)),
            'p_gt_0.1': int(np.sum(probabilities > 0.1)),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if logger:
        logger.info(f"Saved probability statistics: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot probability distribution for ALL edge pairs'
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
        help='Output directory for plots and statistics'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Edge Probability Distribution',
        help='Plot title (default: "Edge Probability Distribution")'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("PROBABILITY DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Merged chains: {args.merged_chains}")
    logger.info(f"Model data: {args.model_data}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load merged chains
    chains_data = load_merged_chains(args.merged_chains, logger)
    merged_samples = chains_data['merged_samples']
    
    # Load model data
    logger.info(f"Loading model data from {args.model_data}")
    with open(args.model_data, 'rb') as f:
        model_data = pkl.load(f)
    
    # Extract all probabilities
    probabilities = extract_all_probabilities(merged_samples, model_data, logger)
    
    # Create plots
    plot_file = args.output / 'probability_distribution.png'
    plot_probability_distribution(probabilities, plot_file, title=args.title, logger=logger)
    
    # Save statistics
    stats_file = args.output / 'probability_statistics.json'
    save_probability_statistics(probabilities, stats_file, logger=logger)
    
    logger.info("=" * 60)
    logger.info("Analysis complete! Results saved to:")
    logger.info(f"  Plot: {plot_file}")
    logger.info(f"  Statistics: {stats_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()








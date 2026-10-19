#!/usr/bin/env python3
"""
Compare ablated models against baseline and each other.

This script:
1. Loads predictions from baseline and ablated models
2. Compares edge probabilities (Spearman correlation)
3. Compares top-K prediction overlap
4. Runs validation analysis for each model
5. Generates comparison figures

Usage:
    python compare_ablated_models.py \
        --baseline results/cullin_mock_config_multi_seeds/merged \
        --no-corr results/cullin_mock_no_corr/merged \
        --no-saint results/cullin_mock_no_saint/merged \
        --no-priors results/cullin_mock_no_priors/merged \
        --output comparison_ablated_models/
"""

import argparse
import logging
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Tuple, Set, List
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jaccard

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(merged_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load edge predictions from merged chains."""
    merged_file = merged_dir / "merged_chains.pkl"
    
    if not merged_file.exists():
        logger.error(f"Merged chains file not found: {merged_file}")
        return pd.DataFrame()
    
    # Load merged chains
    with open(merged_file, 'rb') as f:
        merged_data = pkl.load(f)
    
    merged_samples = merged_data['merged_samples']
    seeds = merged_data['seeds']
    
    # Load model data to get protein names
    dataset_dir = merged_dir.parent.parent
    model_data_file = dataset_dir / f"seed_{seeds[0]}" / f"0_model23_n_{seeds[0]}_model_data.pkl"
    
    if not model_data_file.exists():
        # Try alternative path
        model_data_file = dataset_dir / f"seed_{seeds[0]}" / "model_data.pkl"
    
    if not model_data_file.exists():
        logger.error(f"Model data file not found: {model_data_file}")
        return pd.DataFrame()
    
    with open(model_data_file, 'rb') as f:
        model_data = pkl.load(f)
    
    node_idx2name = model_data.get('node_idx2name', {})
    if not node_idx2name:
        logger.error("No node_idx2name in model_data")
        return pd.DataFrame()
    
    # Get z samples
    if 'z' not in merged_samples:
        logger.error("No 'z' samples in merged data")
        return pd.DataFrame()
    
    z_samples = merged_samples['z']
    
    # Convert to numpy if JAX array
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    # Handle multiple chains
    if len(z_samples.shape) == 3:
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif len(z_samples.shape) == 2:
        pass
    else:
        z_samples = z_samples.reshape(1, -1)
    
    # Calculate mean probability for each edge
    mean_probs = np.mean(1 / (1 + np.exp(-z_samples)), axis=0)
    
    # Create predictions DataFrame
    N = len(node_idx2name)
    predictions = []
    
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(mean_probs[k])
            p1 = str(node_idx2name[i]).strip().upper()
            p2 = str(node_idx2name[j]).strip().upper()
            
            # Normalize protein names
            if p1.endswith("_HUMAN"):
                p1 = p1[:-6]
            if p2.endswith("_HUMAN"):
                p2 = p2[:-6]
            
            predictions.append({
                'Protein1': p1,
                'Protein2': p2,
                'Probability': prob
            })
            k += 1
    
    df = pd.DataFrame(predictions)
    logger.info(f"Loaded {len(df)} predictions from {merged_dir}")
    return df


def normalize_protein_pair(p1: str, p2: str) -> Tuple[str, str]:
    """Normalize protein pair for comparison."""
    p1 = str(p1).strip().upper()
    p2 = str(p2).strip().upper()
    if p1.endswith("_HUMAN"):
        p1 = p1[:-6]
    if p2.endswith("_HUMAN"):
        p2 = p2[:-6]
    return tuple(sorted([p1, p2]))


def create_pair_dict(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """Create dictionary mapping protein pairs to probabilities."""
    pair_dict = {}
    for _, row in df.iterrows():
        pair = normalize_protein_pair(row['Protein1'], row['Protein2'])
        pair_dict[pair] = row['Probability']
    return pair_dict


def compare_predictions(
    baseline_dict: Dict[Tuple[str, str], float],
    ablated_dict: Dict[Tuple[str, str], float],
    model_name: str
) -> Dict:
    """Compare ablated model predictions against baseline."""
    # Find common pairs
    common_pairs = set(baseline_dict.keys()) & set(ablated_dict.keys())
    
    if len(common_pairs) == 0:
        logger.warning(f"No common pairs between baseline and {model_name}")
        return {}
    
    # Extract probabilities for common pairs
    baseline_probs = np.array([baseline_dict[p] for p in sorted(common_pairs)])
    ablated_probs = np.array([ablated_dict[p] for p in sorted(common_pairs)])
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(baseline_probs, ablated_probs)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(baseline_probs, ablated_probs)
    
    # Mean absolute difference
    mad = np.mean(np.abs(baseline_probs - ablated_probs))
    
    # Root mean squared difference
    rmse = np.sqrt(np.mean((baseline_probs - ablated_probs) ** 2))
    
    return {
        'model': model_name,
        'n_common_pairs': len(common_pairs),
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'mad': mad,
        'rmse': rmse,
        'baseline_probs': baseline_probs,
        'ablated_probs': ablated_probs,
        'common_pairs': sorted(common_pairs)
    }


def compare_top_k(
    baseline_dict: Dict[Tuple[str, str], float],
    ablated_dict: Dict[Tuple[str, str], float],
    k_values: List[int] = [50, 100, 200, 500, 1000]
) -> Dict:
    """Compare top-K predictions between models."""
    # Sort by probability
    baseline_sorted = sorted(baseline_dict.items(), key=lambda x: x[1], reverse=True)
    ablated_sorted = sorted(ablated_dict.items(), key=lambda x: x[1], reverse=True)
    
    results = {}
    for k in k_values:
        if k > len(baseline_sorted) or k > len(ablated_sorted):
            continue
        
        baseline_top_k = {pair for pair, _ in baseline_sorted[:k]}
        ablated_top_k = {pair for pair, _ in ablated_sorted[:k]}
        
        overlap = len(baseline_top_k & ablated_top_k)
        jaccard_sim = len(baseline_top_k & ablated_top_k) / len(baseline_top_k | ablated_top_k) if (baseline_top_k | ablated_top_k) else 0.0
        
        results[k] = {
            'overlap': overlap,
            'jaccard': jaccard_sim,
            'baseline_size': len(baseline_top_k),
            'ablated_size': len(ablated_top_k)
        }
    
    return results


def plot_comparison(comparison_results: Dict, output_dir: Path, logger: logging.Logger):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping plots.")
        return
    
    logger.info("Generating comparison plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    n_models = len(comparison_results)
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Scatter plots of probabilities
    for idx, (model_name, results) in enumerate(comparison_results.items(), 1):
        ax = plt.subplot(2, 2, idx)
        
        baseline_probs = results['baseline_probs']
        ablated_probs = results['ablated_probs']
        
        ax.scatter(baseline_probs, ablated_probs, alpha=0.3, s=10)
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='y=x')
        
        spearman_r = results['spearman_r']
        ax.set_xlabel('Baseline Probability', fontweight='bold', fontsize=12)
        ax.set_ylabel(f'{model_name} Probability', fontweight='bold', fontsize=12)
        ax.set_title(f'{model_name}\nSpearman r = {spearman_r:.3f}', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_file = output_dir / "probability_scatter_comparison.png"
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scatter plot: {scatter_file}")
    
    # Plot 2: Correlation summary
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(comparison_results.keys())
    spearman_rs = [comparison_results[m]['spearman_r'] for m in models]
    
    bars = ax.barh(models, spearman_rs, color=sns.color_palette("husl", len(models)))
    ax.set_xlabel('Spearman Correlation with Baseline', fontweight='bold', fontsize=14)
    ax.set_title('Model Comparison: Correlation with Baseline', fontweight='bold', fontsize=16)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, r in zip(bars, spearman_rs):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{r:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    corr_file = output_dir / "correlation_summary.png"
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation summary: {corr_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare ablated models against baseline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--baseline', type=Path, required=True,
                       help='Path to baseline merged results directory')
    parser.add_argument('--no-corr', type=Path, required=True,
                       help='Path to no_corr merged results directory')
    parser.add_argument('--no-saint', type=Path, required=True,
                       help='Path to no_saint merged results directory')
    parser.add_argument('--no-priors', type=Path, required=True,
                       help='Path to no_priors merged results directory')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Comparing Ablated Models")
    logger.info("=" * 60)
    
    # Load predictions
    logger.info("\nLoading predictions...")
    baseline_df = load_predictions(args.baseline, logger)
    no_corr_df = load_predictions(args.no_corr, logger)
    no_saint_df = load_predictions(args.no_saint, logger)
    no_priors_df = load_predictions(args.no_priors, logger)
    
    if baseline_df.empty:
        logger.error("Failed to load baseline predictions")
        return 1
    
    # Create pair dictionaries
    logger.info("\nCreating pair dictionaries...")
    baseline_dict = create_pair_dict(baseline_df)
    
    ablated_models = {
        'no_corr': (no_corr_df, args.no_corr),
        'no_saint': (no_saint_df, args.no_saint),
        'no_priors': (no_priors_df, args.no_priors),
    }
    
    # Compare each ablated model
    comparison_results = {}
    summary_data = []
    
    logger.info("\nComparing models...")
    for model_name, (df, path) in ablated_models.items():
        if df.empty:
            logger.warning(f"Skipping {model_name} - no predictions loaded")
            continue
        
        ablated_dict = create_pair_dict(df)
        results = compare_predictions(baseline_dict, ablated_dict, model_name)
        
        if results:
            comparison_results[model_name] = results
            summary_data.append({
                'Model': model_name,
                'N_Pairs': results['n_common_pairs'],
                'Spearman_r': results['spearman_r'],
                'Spearman_p': results['spearman_p'],
                'Pearson_r': results['pearson_r'],
                'MAD': results['mad'],
                'RMSE': results['rmse']
            })
            
            # Top-K comparison
            top_k_results = compare_top_k(baseline_dict, ablated_dict)
            logger.info(f"\n{model_name} Top-K overlap:")
            for k, metrics in top_k_results.items():
                logger.info(f"  Top {k}: {metrics['overlap']}/{k} overlap, Jaccard = {metrics['jaccard']:.3f}")
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = args.output / "comparison_summary.tsv"
        summary_df.to_csv(summary_file, sep='\t', index=False)
        logger.info(f"\nSaved summary: {summary_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Comparison Summary")
        logger.info("=" * 60)
        print(summary_df.to_string(index=False))
    
    # Generate plots
    if comparison_results and HAS_MATPLOTLIB:
        plot_comparison(comparison_results, args.output, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Comparison complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)
    
    logger.info("\nNext steps:")
    logger.info("  1. Run validation analysis for each ablated model")
    logger.info("  2. Compare validation metrics across models")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

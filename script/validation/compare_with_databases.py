#!/usr/bin/env python3
"""
Compare predicted protein interactions with reference databases (CORUM and HuMAP2).

This script:
1. Loads CORUM and HuMAP2 databases
2. Extracts all protein pairs from each database
3. Loads predicted top pairs from all datasets
4. Computes overlap statistics
5. Generates comparison visualizations

Usage:
    python compare_with_databases.py --results-dir ../results --database-dir ../database --output comparison_results/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List
import warnings

import pandas as pd
import numpy as np

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def normalize_protein_name(name: str) -> str:
    """
    Normalize protein name for comparison.
    
    Handles:
    - UniProt format: "ELOB_HUMAN" -> "ELOB"
    - Gene names: "ELOB" -> "ELOB"
    - Uppercase and strip whitespace
    """
    name = name.strip().upper()
    # Remove "_HUMAN" suffix if present (UniProt format)
    if name.endswith("_HUMAN"):
        name = name[:-6]  # Remove "_HUMAN"
    return name


def get_corum_pairs(corum_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """
    Extract all protein pairs from CORUM database.
    
    CORUM contains complexes, so we extract all pairs within each complex.
    """
    logger.info(f"Loading CORUM database from {corum_path}")
    
    df = pd.read_csv(corum_path, sep='\t', low_memory=False)
    logger.info(f"Loaded {len(df)} CORUM complexes")
    
    pairs = set()
    
    for idx, row in df.iterrows():
        # Get gene names from subunits_gene_name column
        gene_names_str = str(row.get('subunits_gene_name', ''))
        if pd.isna(gene_names_str) or gene_names_str == '':
            continue
        
        # Split by semicolon to get individual gene names
        gene_names = [normalize_protein_name(g.strip()) for g in gene_names_str.split(';')]
        gene_names = [g for g in gene_names if g and g != 'NAN']
        
        # Create all pairs within this complex
        if len(gene_names) >= 2:
            for i in range(len(gene_names)):
                for j in range(i + 1, len(gene_names)):
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([gene_names[i], gene_names[j]]))
                    pairs.add(pair)
    
    logger.info(f"Extracted {len(pairs)} unique protein pairs from CORUM")
    return pairs


def get_humap2_pairs(humap2_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """
    Extract all protein pairs from HuMAP2 database.
    
    HuMAP2 contains complexes, so we extract all pairs within each complex.
    """
    logger.info(f"Loading HuMAP2 database from {humap2_path}")
    
    # HuMAP2 uses comma-separated format
    df = pd.read_csv(humap2_path, sep=',', low_memory=False)
    logger.info(f"Loaded {len(df)} HuMAP2 complexes")
    
    pairs = set()
    
    for idx, row in df.iterrows():
        # Get gene names from genenames column
        gene_names_str = str(row.get('genenames', ''))
        if pd.isna(gene_names_str) or gene_names_str == '':
            continue
        
        # Split by space to get individual gene names
        gene_names = [normalize_protein_name(g.strip()) for g in gene_names_str.split()]
        gene_names = [g for g in gene_names if g and g != 'NAN']
        
        # Create all pairs within this complex
        if len(gene_names) >= 2:
            for i in range(len(gene_names)):
                for j in range(i + 1, len(gene_names)):
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([gene_names[i], gene_names[j]]))
                    pairs.add(pair)
    
    logger.info(f"Extracted {len(pairs)} unique protein pairs from HuMAP2")
    return pairs


def load_all_model_pairs(results_dir: Path, logger: logging.Logger) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Load ALL prey-prey pairs from merged chains (not just top 100).
    
    This extracts all edge probabilities from the model predictions.
    
    Returns:
        Dictionary mapping dataset name to dict of (protein1, protein2) -> probability
    """
    logger.info(f"Loading ALL model predictions from merged chains in {results_dir}")
    
    import pickle as pkl
    import jax.numpy as jnp
    import sys
    from pathlib import Path
    
    # Add script directory to path to import model_variations
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    import model_variations as mv
    
    all_model_pairs = {}
    
    # Find merged_chains.pkl files
    for merged_file in results_dir.rglob("merged/merged_chains.pkl"):
        # Extract dataset name from path
        parts = merged_file.parts
        dataset_name = None
        for i, part in enumerate(parts):
            if part.endswith("_config_multi_seeds"):
                dataset_name = part
                break
        
        if dataset_name is None:
            continue
        
        logger.info(f"Loading all model pairs from {dataset_name}")
        
        try:
            # Load merged chains
            with open(merged_file, 'rb') as f:
                merged_data = pkl.load(f)
            
            # The merged file structure: {'merged_samples': {...}, 'n_chains': ..., 'seeds': ..., 'burn_in': ...}
            merged_samples = merged_data['merged_samples']
            seeds = merged_data['seeds']
            model_data_file = merged_file.parent.parent / f"seed_{seeds[0]}/0_model23_n_{seeds[0]}_model_data.pkl"
            
            # Get z samples from merged_samples
            if 'z' not in merged_samples:
                logger.warning(f"No 'z' samples in merged data for {dataset_name}, skipping...")
                continue
            
            z_samples = merged_samples['z']
            
            # Load model data to get protein names
            with open(model_data_file, 'rb') as f:
                model_data = pkl.load(f)
            
            node_idx2name = model_data.get('node_idx2name', {})
            if not node_idx2name:
                logger.warning(f"No node_idx2name in model_data for {dataset_name}, skipping...")
                continue
            
            # Convert to numpy if JAX array
            if hasattr(z_samples, 'block_until_ready'):
                z_samples = np.array(z_samples)
            
            # Handle multiple chains: shape might be (n_chains, n_samples, M) or (n_samples, M)
            if len(z_samples.shape) == 3:
                # Multiple chains: flatten to (n_chains * n_samples, M)
                z_samples = z_samples.reshape(-1, z_samples.shape[-1])
            elif len(z_samples.shape) == 2:
                # Single chain: (n_samples, M)
                pass
            else:
                # Single sample: (M,)
                z_samples = z_samples.reshape(1, -1)
            
            n_samples = z_samples.shape[0]
            M = z_samples.shape[1]
            
            # Calculate N from M: M = N*(N-1)//2
            N = int((1 + np.sqrt(1 + 8 * M)) / 2)
            
            logger.info(f"  Processing {n_samples} samples, N={N} proteins, M={M} edges")
            
            # Convert z to probabilities using sigmoid
            z_probs = 1 / (1 + np.exp(-z_samples))  # sigmoid
            
            # Average probabilities across all samples
            mean_probs = np.mean(z_probs, axis=0)
            
            # Create mapping from edge index to protein pair
            edge_to_pair = {}
            edge_idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    prot1 = normalize_protein_name(node_idx2name.get(i, f"protein_{i}"))
                    prot2 = normalize_protein_name(node_idx2name.get(j, f"protein_{j}"))
                    pair = tuple(sorted([prot1, prot2]))
                    edge_to_pair[edge_idx] = pair
                    edge_idx += 1
            
            # Create dictionary of all pairs with probabilities
            pairs_dict = {}
            for edge_idx, prob in enumerate(mean_probs):
                if edge_idx in edge_to_pair:
                    pair = edge_to_pair[edge_idx]
                    pairs_dict[pair] = float(prob)
            
            all_model_pairs[dataset_name] = pairs_dict
            logger.info(f"  Extracted {len(pairs_dict)} prey-prey pairs from model")
            
        except Exception as e:
            logger.error(f"Error loading {merged_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Loaded all model pairs from {len(all_model_pairs)} datasets")
    return all_model_pairs


def load_predicted_pairs(results_dir: Path, logger: logging.Logger) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Load predicted top pairs from all datasets.
    
    Returns:
        Dictionary mapping dataset name to set of protein pairs
    """
    logger.info(f"Loading predicted pairs from {results_dir}")
    
    predicted_pairs = {}
    
    # Find all top_100_pairs.tsv files
    for top_pairs_file in results_dir.rglob("top_100_pairs.tsv"):
        # Extract dataset name from path
        # e.g., results/cullin_mock_config_multi_seeds/networks/top_100_pairs.tsv
        # -> cullin_mock_config_multi_seeds
        parts = top_pairs_file.parts
        dataset_name = None
        for i, part in enumerate(parts):
            if part.endswith("_config_multi_seeds"):
                dataset_name = part
                break
        
        if dataset_name is None:
            logger.warning(f"Could not extract dataset name from {top_pairs_file}")
            continue
        
        logger.info(f"Loading pairs from {dataset_name}")
        
        try:
            df = pd.read_csv(top_pairs_file, sep='\t')
            pairs = set()
            
            for _, row in df.iterrows():
                prot1 = normalize_protein_name(str(row['Protein1']))
                prot2 = normalize_protein_name(str(row['Protein2']))
                if prot1 and prot2 and prot1 != prot2:
                    pair = tuple(sorted([prot1, prot2]))
                    pairs.add(pair)
            
            predicted_pairs[dataset_name] = pairs
            logger.info(f"  Loaded {len(pairs)} pairs from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error loading {top_pairs_file}: {e}")
            continue
    
    logger.info(f"Loaded predicted pairs from {len(predicted_pairs)} datasets")
    return predicted_pairs


def compute_validation_stats(
    all_model_pairs: Dict[Tuple[str, str], float],
    top100_pairs: Set[Tuple[str, str]],
    corum_pairs: Set[Tuple[str, str]],
    humap2_pairs: Set[Tuple[str, str]],
    logger: logging.Logger
) -> Dict:
    """
    Compute proper validation statistics.
    
    The correct approach:
    1. Get ALL prey-prey pairs from model (not just top 100)
    2. Compare those with databases to find validated pairs (ground truth)
    3. Check how many validated pairs appear in top 100 predictions
    4. Calculate recall: validated pairs in top 100 / validated pairs in all model predictions
    """
    
    # Combined database (union of CORUM and HuMAP2)
    combined_db = corum_pairs | humap2_pairs
    
    # Step 1: Find validated pairs from ALL model predictions (pairs that are in databases)
    all_model_pairs_set = set(all_model_pairs.keys())
    validated_corum = all_model_pairs_set & corum_pairs
    validated_humap2 = all_model_pairs_set & humap2_pairs
    validated_combined = all_model_pairs_set & combined_db
    
    # Step 2: Check how many validated pairs appear in top 100 predictions
    predicted_validated_corum = top100_pairs & validated_corum
    predicted_validated_humap2 = top100_pairs & validated_humap2
    predicted_validated_combined = top100_pairs & validated_combined
    
    # Step 3: Calculate recall (the key metric)
    # Recall = (validated pairs in top 100) / (validated pairs in all model predictions)
    recall_corum = len(predicted_validated_corum) / len(validated_corum) if len(validated_corum) > 0 else 0
    recall_humap2 = len(predicted_validated_humap2) / len(validated_humap2) if len(validated_humap2) > 0 else 0
    recall_combined = len(predicted_validated_combined) / len(validated_combined) if len(validated_combined) > 0 else 0
    
    # Also calculate precision (for completeness)
    precision_corum = len(predicted_validated_corum) / len(top100_pairs) if len(top100_pairs) > 0 else 0
    precision_humap2 = len(predicted_validated_humap2) / len(top100_pairs) if len(top100_pairs) > 0 else 0
    precision_combined = len(predicted_validated_combined) / len(top100_pairs) if len(top100_pairs) > 0 else 0
    
    # Compute statistics
    stats = {
        'num_all_model_pairs': len(all_model_pairs_set),
        'num_top100_pairs': len(top100_pairs),
        'num_validated_corum': len(validated_corum),
        'num_validated_humap2': len(validated_humap2),
        'num_validated_combined': len(validated_combined),
        'num_predicted_validated_corum': len(predicted_validated_corum),
        'num_predicted_validated_humap2': len(predicted_validated_humap2),
        'num_predicted_validated_combined': len(predicted_validated_combined),
        'recall_corum': recall_corum,
        'recall_humap2': recall_humap2,
        'recall_combined': recall_combined,
        'precision_corum': precision_corum,
        'precision_humap2': precision_humap2,
        'precision_combined': precision_combined,
        'coverage_corum': len(validated_corum) / len(all_model_pairs_set) if len(all_model_pairs_set) > 0 else 0,
        'coverage_humap2': len(validated_humap2) / len(all_model_pairs_set) if len(all_model_pairs_set) > 0 else 0,
        'coverage_combined': len(validated_combined) / len(all_model_pairs_set) if len(all_model_pairs_set) > 0 else 0,
    }
    
    return stats, validated_combined, predicted_validated_combined


def plot_comparison(
    all_stats: Dict[str, Dict],
    output_dir: Path,
    logger: logging.Logger
):
    """Create validation comparison visualizations."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping plots.")
        return
    
    logger.info("Creating validation plots...")
    
    datasets = list(all_stats.keys())
    
    # Figure 1: Recall and Precision
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validation: Recall and Precision', fontsize=16, fontweight='bold')
    
    # Recall (key metric)
    rec_corum = [all_stats[d]['recall_corum'] * 100 for d in datasets]
    rec_humap2 = [all_stats[d]['recall_humap2'] * 100 for d in datasets]
    rec_combined = [all_stats[d]['recall_combined'] * 100 for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    axes[0, 0].bar(x - width, rec_corum, width, label='CORUM', alpha=0.8)
    axes[0, 0].bar(x, rec_humap2, width, label='HuMAP2', alpha=0.8)
    axes[0, 0].bar(x + width, rec_combined, width, label='Combined', alpha=0.8)
    axes[0, 0].set_ylabel('Recall (%)')
    axes[0, 0].set_title('Recall: Validated Pairs in Top 100 / Validated Pairs in Initial Data')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([d.replace('_config_multi_seeds', '') for d in datasets], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Precision
    prec_corum = [all_stats[d]['precision_corum'] * 100 for d in datasets]
    prec_humap2 = [all_stats[d]['precision_humap2'] * 100 for d in datasets]
    prec_combined = [all_stats[d]['precision_combined'] * 100 for d in datasets]
    
    axes[0, 1].bar(x - width, prec_corum, width, label='CORUM', alpha=0.8)
    axes[0, 1].bar(x, prec_humap2, width, label='HuMAP2', alpha=0.8)
    axes[0, 1].bar(x + width, prec_combined, width, label='Combined', alpha=0.8)
    axes[0, 1].set_ylabel('Precision (%)')
    axes[0, 1].set_title('Precision: Validated Pairs in Top 100 / Total Top 100')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([d.replace('_config_multi_seeds', '') for d in datasets], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Validated pairs counts
    validated_corum = [all_stats[d]['num_validated_corum'] for d in datasets]
    validated_humap2 = [all_stats[d]['num_validated_humap2'] for d in datasets]
    validated_combined = [all_stats[d]['num_validated_combined'] for d in datasets]
    
    axes[1, 0].bar(x - width, validated_corum, width, label='CORUM', alpha=0.8)
    axes[1, 0].bar(x, validated_humap2, width, label='HuMAP2', alpha=0.8)
    axes[1, 0].bar(x + width, validated_combined, width, label='Combined', alpha=0.8)
    axes[1, 0].set_ylabel('Number of Pairs')
    axes[1, 0].set_title('Validated Pairs in ALL Model Predictions')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([d.replace('_config_multi_seeds', '') for d in datasets], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Predicted validated pairs
    pred_validated_corum = [all_stats[d]['num_predicted_validated_corum'] for d in datasets]
    pred_validated_humap2 = [all_stats[d]['num_predicted_validated_humap2'] for d in datasets]
    pred_validated_combined = [all_stats[d]['num_predicted_validated_combined'] for d in datasets]
    
    axes[1, 1].bar(x - width, pred_validated_corum, width, label='CORUM', alpha=0.8)
    axes[1, 1].bar(x, pred_validated_humap2, width, label='HuMAP2', alpha=0.8)
    axes[1, 1].bar(x + width, pred_validated_combined, width, label='Combined', alpha=0.8)
    axes[1, 1].set_ylabel('Number of Pairs')
    axes[1, 1].set_title('Validated Pairs in Top 100 Predictions')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([d.replace('_config_multi_seeds', '') for d in datasets], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved validation plot to {output_dir / 'validation_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare predicted interactions with CORUM and HuMAP2 databases'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('../results'),
        help='Directory containing results with top_100_pairs.tsv files'
    )
    parser.add_argument(
        '--database-dir',
        type=Path,
        default=Path('../database'),
        help='Directory containing database files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('comparison_results'),
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("DATABASE COMPARISON")
    logger.info("=" * 60)
    
    # Load databases
    corum_path = args.database_dir / 'corum_humanComplexes.tsv'
    humap2_path = args.database_dir / 'humap2.text'
    
    if not corum_path.exists():
        logger.error(f"CORUM file not found: {corum_path}")
        return
    
    if not humap2_path.exists():
        logger.error(f"HuMAP2 file not found: {humap2_path}")
        return
    
    corum_pairs = get_corum_pairs(corum_path, logger)
    humap2_pairs = get_humap2_pairs(humap2_path, logger)
    
    # Load ALL model pairs (all prey-prey predictions, not just top 100)
    all_model_pairs_dict = load_all_model_pairs(args.results_dir, logger)
    
    # Load top 100 predicted pairs
    top100_pairs_dict = load_predicted_pairs(args.results_dir, logger)
    
    if not top100_pairs_dict:
        logger.error("No top 100 pairs found!")
        return
    
    if not all_model_pairs_dict:
        logger.error("No model pairs found!")
        return
    
    # Compute statistics for each dataset
    all_stats = {}
    all_overlaps = {}
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS (Correct Approach)")
    logger.info("=" * 60)
    logger.info("Step 1: Get ALL prey-prey pairs from model predictions")
    logger.info("Step 2: Compare with databases to find validated pairs (ground truth)")
    logger.info("Step 3: Check how many validated pairs appear in top 100")
    logger.info("Step 4: Calculate recall = validated pairs in top 100 / validated pairs in all model predictions")
    logger.info("=" * 60)
    
    for dataset_name in top100_pairs_dict.keys():
        if dataset_name not in all_model_pairs_dict:
            logger.warning(f"No model pairs found for {dataset_name}, skipping...")
            continue
        
        all_model_pairs = all_model_pairs_dict[dataset_name]
        top100_pairs = top100_pairs_dict[dataset_name]
        
        logger.info(f"\n--- {dataset_name} ---")
        stats, validated_combined, predicted_validated_combined = compute_validation_stats(
            all_model_pairs, top100_pairs, corum_pairs, humap2_pairs, logger
        )
        
        all_stats[dataset_name] = stats
        all_overlaps[dataset_name] = {
            'validated_combined': validated_combined,
            'predicted_validated_combined': predicted_validated_combined,
        }
        
        logger.info(f"All model pairs (all prey-prey predictions): {stats['num_all_model_pairs']}")
        logger.info(f"Top 100 pairs: {stats['num_top100_pairs']}")
        logger.info(f"\nValidated pairs in ALL model predictions (in databases):")
        logger.info(f"  CORUM: {stats['num_validated_corum']} ({stats['coverage_corum']*100:.2f}% of all model pairs)")
        logger.info(f"  HuMAP2: {stats['num_validated_humap2']} ({stats['coverage_humap2']*100:.2f}% of all model pairs)")
        logger.info(f"  Combined: {stats['num_validated_combined']} ({stats['coverage_combined']*100:.2f}% of all model pairs)")
        logger.info(f"\nValidated pairs in top 100 predictions:")
        logger.info(f"  CORUM: {stats['num_predicted_validated_corum']} / {stats['num_validated_corum']}")
        logger.info(f"  HuMAP2: {stats['num_predicted_validated_humap2']} / {stats['num_validated_humap2']}")
        logger.info(f"  Combined: {stats['num_predicted_validated_combined']} / {stats['num_validated_combined']}")
        logger.info(f"\nRECALL (key metric): Of validated pairs, how many are in top 100?")
        logger.info(f"  CORUM: {stats['recall_corum']*100:.1f}% ({stats['num_predicted_validated_corum']}/{stats['num_validated_corum']})")
        logger.info(f"  HuMAP2: {stats['recall_humap2']*100:.1f}% ({stats['num_predicted_validated_humap2']}/{stats['num_validated_humap2']})")
        logger.info(f"  Combined: {stats['recall_combined']*100:.1f}% ({stats['num_predicted_validated_combined']}/{stats['num_validated_combined']})")
        logger.info(f"\nPrecision: Of top 100, how many are validated?")
        logger.info(f"  CORUM: {stats['precision_corum']*100:.1f}%")
        logger.info(f"  HuMAP2: {stats['precision_humap2']*100:.1f}%")
        logger.info(f"  Combined: {stats['precision_combined']*100:.1f}%")
    
    # Save statistics
    stats_file = args.output / 'comparison_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_file}")
    
    # Save overlap details
    for dataset_name, overlaps in all_overlaps.items():
        # Save validated pairs from initial data
        if 'validated_combined' in overlaps and overlaps['validated_combined']:
            pairs_list = [{'Protein1': p[0], 'Protein2': p[1]} for p in sorted(overlaps['validated_combined'])]
            df = pd.DataFrame(pairs_list)
            output_file = args.output / f"{dataset_name}_validated_pairs.tsv"
            df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Saved {len(overlaps['validated_combined'])} validated pairs to {output_file}")
        
        # Save predicted validated pairs
        if 'predicted_validated_combined' in overlaps and overlaps['predicted_validated_combined']:
            pairs_list = [{'Protein1': p[0], 'Protein2': p[1]} for p in sorted(overlaps['predicted_validated_combined'])]
            df = pd.DataFrame(pairs_list)
            output_file = args.output / f"{dataset_name}_predicted_validated_pairs.tsv"
            df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Saved {len(overlaps['predicted_validated_combined'])} predicted validated pairs to {output_file}")
    
    # Create plots
    if HAS_MATPLOTLIB:
        plot_comparison(all_stats, args.output, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Comparison complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Identify novel interactions by comparing:
1. Directly observed bait-prey pairs (composite_table)
2. Spectral counts (spec_table) - to see which were "obvious"
3. Predicted interactions (all_pairs) - to find truly novel ones

Novel interactions = High probability but:
- Not directly observed as bait-prey
- Low/no spectral counts (not obvious from data)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List
from collections import defaultdict

import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_composite_table(composite_file: Path) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], float]]:
    """
    Load directly observed bait-prey pairs from composite_table.
    
    Returns:
        observed_pairs: Set of (protein1, protein2) tuples (normalized)
        observed_scores: Dict mapping pairs to SAINT scores
    """
    logger.info(f"Loading composite table from {composite_file}")
    df = pd.read_csv(composite_file, sep='\t')
    
    observed_pairs = set()
    observed_scores = {}
    
    for _, row in df.iterrows():
        bait = str(row['Bait']).strip()
        prey = str(row['Prey']).strip()
        score = float(row.get('MSscore', 0))
        
        # Remove _HUMAN suffix if present
        bait = bait.replace('_HUMAN', '')
        prey = prey.replace('_HUMAN', '')
        
        # Normalize pair (alphabetically sorted)
        pair = tuple(sorted([bait, prey]))
        observed_pairs.add(pair)
        if pair not in observed_scores or score > observed_scores[pair]:
            observed_scores[pair] = score
    
    logger.info(f"  Found {len(observed_pairs)} directly observed bait-prey pairs")
    return observed_pairs, observed_scores


def load_spec_table(spec_file: Path) -> Dict[Tuple[str, str], float]:
    """
    Load spectral counts from spec_table.
    Calculate average spectral count for each protein pair across all baits.
    
    Returns:
        spec_counts: Dict mapping (protein1, protein2) to average spectral count
    """
    logger.info(f"Loading spec table from {spec_file}")
    df = pd.read_csv(spec_file, sep='\t', index_col=0)
    
    # Get protein names (remove _HUMAN suffix)
    proteins = [p.replace('_HUMAN', '') for p in df.index]
    
    # Get bait columns (experimental replicates, not controls)
    bait_columns = [col for col in df.columns if '_r' in col or '_c' in col]
    
    spec_counts = defaultdict(list)
    
    # For each protein pair, calculate average spectral count
    # when both proteins are detected in the same bait experiment
    for i, prot1 in enumerate(proteins):
        for j, prot2 in enumerate(proteins):
            if i >= j:  # Only upper triangle
                continue
            
            pair = tuple(sorted([prot1, prot2]))
            
            # Get spectral counts for both proteins across all baits
            spec1 = df.iloc[i][bait_columns].values
            spec2 = df.iloc[j][bait_columns].values
            
            # Calculate co-occurrence: when both have non-zero counts
            co_occurrence = np.sum((spec1 > 0) & (spec2 > 0))
            avg_spec1 = np.mean(spec1[spec1 > 0]) if np.any(spec1 > 0) else 0
            avg_spec2 = np.mean(spec2[spec2 > 0]) if np.any(spec2 > 0) else 0
            max_spec = max(avg_spec1, avg_spec2)
            
            if co_occurrence > 0 or max_spec > 0:
                spec_counts[pair] = {
                    'co_occurrence': co_occurrence,
                    'avg_spec1': avg_spec1,
                    'avg_spec2': avg_spec2,
                    'max_spec': max_spec,
                    'total_spec': avg_spec1 + avg_spec2
                }
    
    logger.info(f"  Calculated spec counts for {len(spec_counts)} protein pairs")
    return spec_counts


def normalize_protein_name(name: str) -> str:
    """Normalize protein name by removing common suffixes."""
    name = str(name).strip()
    for suffix in ['_HUMAN', '_MOUSE', '_RAT']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def identify_novel_interactions(
    predicted_file: Path,
    composite_file: Path,
    spec_file: Path,
    output_file: Path,
    probability_threshold: float = 0.6,
    spec_count_threshold: float = 2.0
):
    """
    Identify novel interactions by comparing predictions with observed data.
    
    Args:
        predicted_file: File with predicted interactions (all_pairs.tsv)
        composite_file: File with directly observed bait-prey pairs
        spec_file: File with spectral counts
        output_file: Output file for novel interactions
        probability_threshold: Minimum probability for novel interactions
        spec_count_threshold: Maximum spec count to consider as "novel" (low spec = novel)
    """
    logger.info("Identifying novel interactions")
    
    # Load data
    observed_pairs, observed_scores = load_composite_table(composite_file)
    spec_counts = load_spec_table(spec_file)
    
    # Load predicted interactions
    logger.info(f"Loading predicted interactions from {predicted_file}")
    df_pred = pd.read_csv(predicted_file, sep='\t')
    
    # Normalize protein names
    df_pred['Protein1_norm'] = df_pred['Protein1'].apply(normalize_protein_name)
    df_pred['Protein2_norm'] = df_pred['Protein2'].apply(normalize_protein_name)
    df_pred['Pair'] = df_pred.apply(
        lambda row: tuple(sorted([row['Protein1_norm'], row['Protein2_norm']])),
        axis=1
    )
    
    # Filter by probability threshold
    df_pred = df_pred[df_pred['Probability'] >= probability_threshold].copy()
    
    # Categorize interactions
    results = []
    
    for _, row in df_pred.iterrows():
        pair = row['Pair']
        prob = row['Probability']
        prot1 = row['Protein1_norm']
        prot2 = row['Protein2_norm']
        
        # Check if directly observed
        is_observed = pair in observed_pairs
        observed_score = observed_scores.get(pair, 0)
        
        # Get spec count information
        spec_info = spec_counts.get(pair, {})
        max_spec = spec_info.get('max_spec', 0) if spec_info else 0
        co_occurrence = spec_info.get('co_occurrence', 0) if spec_info else 0
        total_spec = spec_info.get('total_spec', 0) if spec_info else 0
        
        # Categorize
        if is_observed:
            category = 'Directly_Observed'
        elif max_spec >= spec_count_threshold or co_occurrence >= 3:
            category = 'High_Spec_Count'  # Obvious from data
        elif max_spec > 0 or co_occurrence > 0:
            category = 'Low_Spec_Count'  # Some evidence but not obvious
        else:
            category = 'Novel'  # No spec count evidence
        
        results.append({
            'Protein1': prot1,
            'Protein2': prot2,
            'Probability': prob,
            'Category': category,
            'Directly_Observed': is_observed,
            'Observed_SAINT_Score': observed_score,
            'Max_Spectral_Count': max_spec,
            'Co_Occurrence_Count': co_occurrence,
            'Total_Spectral_Count': total_spec,
            'Protein1_UniProt': row.get('Protein1_UniProt', ''),
            'Protein2_UniProt': row.get('Protein2_UniProt', ''),
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Category', 'Probability'], ascending=[True, False])
    
    # Save results
    results_df.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Saved results to {output_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("NOVEL INTERACTIONS SUMMARY")
    logger.info("=" * 80)
    
    for category in ['Directly_Observed', 'High_Spec_Count', 'Low_Spec_Count', 'Novel']:
        count = len(results_df[results_df['Category'] == category])
        if count > 0:
            avg_prob = results_df[results_df['Category'] == category]['Probability'].mean()
            logger.info(f"{category}: {count} interactions (avg probability: {avg_prob:.4f})")
    
    # Top novel interactions
    novel_df = results_df[results_df['Category'] == 'Novel'].head(20)
    if len(novel_df) > 0:
        logger.info(f"\nTop 20 Novel Interactions (no spec count evidence):")
        for idx, (_, row) in enumerate(novel_df.iterrows(), 1):
            logger.info(f"  {idx}. {row['Protein1']} - {row['Protein2']}: {row['Probability']:.4f}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Identify novel interactions by comparing predictions with observed data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--predicted-file', type=Path,
                       default='results/cullin_wt_config_multi_seeds/all_pairs_with_uniprot.tsv',
                       help='Predicted interactions file')
    parser.add_argument('--composite-file', type=Path,
                       default='results/cullin_wt_config_multi_seeds/seed_13/composite_table.tsv',
                       help='Composite table with directly observed bait-prey pairs')
    parser.add_argument('--spec-file', type=Path,
                       default='results/cullin_wt_config_multi_seeds/seed_13/spec_table.tsv',
                       help='Spec table with spectral counts')
    parser.add_argument('--output-file', type=Path,
                       default='wt_vif_comparison/novel_interactions.tsv',
                       help='Output file for novel interactions')
    parser.add_argument('--probability-threshold', type=float, default=0.6,
                       help='Minimum probability for novel interactions')
    parser.add_argument('--spec-threshold', type=float, default=2.0,
                       help='Maximum spec count to consider as novel (above = obvious)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Identify novel interactions
    identify_novel_interactions(
        args.predicted_file,
        args.composite_file,
        args.spec_file,
        args.output_file,
        args.probability_threshold,
        args.spec_threshold
    )
    
    logger.info("=" * 80)
    logger.info("Novel interaction identification complete!")
    logger.info(f"Results saved to: {args.output_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Aggregate validation results from multiple datasets into a summary metrics table.

Usage:
    python aggregate_validation_results.py --validation-dir ../validation_all_cullin --output summary_table.tsv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_validation_stats(validation_dir: Path, dataset_name: str, logger: logging.Logger) -> Dict:
    """Load validation statistics for a dataset."""
    stats_file = validation_dir / dataset_name / f"{dataset_name}_validation_stats.json"
    
    if not stats_file.exists():
        logger.warning(f"Stats file not found: {stats_file}")
        return None
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        logger.error(f"Error loading stats for {dataset_name}: {e}")
        return None


def load_enrichment_stats(validation_dir: Path, dataset_name: str, logger: logging.Logger) -> Dict:
    """Load enrichment analysis for a dataset."""
    enrichment_file = validation_dir / dataset_name / f"{dataset_name}_enrichment_analysis.json"
    
    if not enrichment_file.exists():
        logger.warning(f"Enrichment file not found: {enrichment_file}")
        return None
    
    try:
        with open(enrichment_file, 'r') as f:
            enrichment = json.load(f)
        return enrichment
    except Exception as e:
        logger.error(f"Error loading enrichment for {dataset_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate validation results from multiple datasets'
    )
    parser.add_argument(
        '--validation-dir',
        type=Path,
        required=True,
        help='Directory containing validation results for multiple datasets'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('validation_summary_table.tsv'),
        help='Output file for summary table'
    )
    parser.add_argument(
        '--enrichment-output',
        type=Path,
        default=Path('enrichment_summary_table.tsv'),
        help='Output file for enrichment summary table'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("AGGREGATING VALIDATION RESULTS")
    logger.info("=" * 60)
    
    # Find all dataset directories
    dataset_dirs = [d for d in args.validation_dir.iterdir() if d.is_dir()]
    
    if not dataset_dirs:
        logger.error(f"No dataset directories found in {args.validation_dir}")
        return
    
    logger.info(f"Found {len(dataset_dirs)} datasets")
    
    # Collect validation stats
    all_validation_stats = []
    all_enrichment_stats = []
    
    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        logger.info(f"\nProcessing: {dataset_name}")
        
        # Load validation stats
        validation_stats = load_validation_stats(args.validation_dir, dataset_name, logger)
        if validation_stats:
            all_validation_stats.append(validation_stats)
        
        # Load enrichment stats
        enrichment_stats = load_enrichment_stats(args.validation_dir, dataset_name, logger)
        if enrichment_stats:
            # Extract key percentages (1%, 5%, 10%, 20%, 50%)
            key_percentages = [1, 5, 10, 20, 50]
            enrichment_summary = {'dataset': dataset_name}
            
            for result in enrichment_stats.get('enrichment_at_percentages', []):
                pct = result['top_percent']
                if pct in key_percentages:
                    # Handle both old (enrichment_factor) and new (fold_enrichment) metric names
                    enrichment_summary[f'fold_enrichment_{pct}pct'] = result.get('fold_enrichment', result.get('enrichment_factor', 0))
                    enrichment_summary[f'recall_{pct}pct'] = result.get('recall_percent', 0)
                    enrichment_summary[f'positive_set_fraction_{pct}pct'] = result.get('positive_set_fraction_percent', result.get('precision_percent', 0))
                    enrichment_summary[f'validated_{pct}pct'] = result.get('validated_in_top', 0)
                    enrichment_summary[f'n_pairs_{pct}pct'] = result.get('n_pairs', 0)
            
            enrichment_summary['total_pairs'] = enrichment_stats.get('total_pairs', 0)
            enrichment_summary['total_validated'] = enrichment_stats.get('total_validated', 0)
            enrichment_summary['baseline_rate'] = enrichment_stats.get('baseline_rate', 0)
            
            # Also add recall@K and fold-enrichment@K from validation stats if available
            validation_stats = load_validation_stats(args.validation_dir, dataset_name, logger)
            if validation_stats:
                if 'recall_at_k' in validation_stats:
                    for K in [50, 100, 200, 500]:
                        if K in validation_stats['recall_at_k']:
                            enrichment_summary[f'recall_at_{K}'] = validation_stats['recall_at_k'][K]
                if 'fold_enrichment_at_k' in validation_stats:
                    for K in [50, 100, 200, 500]:
                        if K in validation_stats['fold_enrichment_at_k']:
                            enrichment_summary[f'fold_enrichment_at_{K}'] = validation_stats['fold_enrichment_at_k'][K]
                if 'positive_set_fraction_at_k' in validation_stats:
                    for K in [50, 100, 200, 500]:
                        if K in validation_stats['positive_set_fraction_at_k']:
                            enrichment_summary[f'positive_set_fraction_at_{K}'] = validation_stats['positive_set_fraction_at_k'][K]
            
            all_enrichment_stats.append(enrichment_summary)
    
    # Create validation summary table
    if all_validation_stats:
        logger.info(f"\nCreating validation summary table...")
        validation_df = pd.DataFrame(all_validation_stats)
        
        # Reorder columns for better readability
        columns_order = [
            'dataset',
            'total_predicted_pairs',
            'top100_pairs',
            'validated_pairs',
            'validated_in_top100',
            'top100_validated',
            'recall_percent',
            'precision_percent'
        ]
        
        # Flatten nested validated_pairs dict
        if 'validated_pairs' in validation_df.columns:
            for key in ['pdb_only', 'database_only', 'both', 'total']:
                validation_df[f'validated_{key}'] = validation_df['validated_pairs'].apply(
                    lambda x: x.get(key, 0) if isinstance(x, dict) else 0
                )
            validation_df = validation_df.drop('validated_pairs', axis=1)
        
        # Select and reorder columns
        available_columns = [c for c in columns_order if c in validation_df.columns]
        other_columns = [c for c in validation_df.columns if c not in available_columns]
        validation_df = validation_df[available_columns + other_columns]
        
        # Save validation summary
        validation_df.to_csv(args.output, sep='\t', index=False)
        logger.info(f"Saved validation summary to {args.output}")
        print("\nValidation Summary:")
        print(validation_df.to_string(index=False))
    
    # Create enrichment summary table
    if all_enrichment_stats:
        logger.info(f"\nCreating enrichment summary table...")
        enrichment_df = pd.DataFrame(all_enrichment_stats)
        
        # Reorder columns
        enrichment_columns = ['dataset', 'total_pairs', 'total_validated', 'baseline_rate']
        # Add Recall@K metrics first
        for K in [50, 100, 200, 500]:
            enrichment_columns.extend([
                f'recall_at_{K}',
                f'fold_enrichment_at_{K}',
                f'positive_set_fraction_at_{K}'
            ])
        # Then add percentage-based metrics
        for pct in [1, 5, 10, 20, 50]:
            enrichment_columns.extend([
                f'validated_{pct}pct',
                f'fold_enrichment_{pct}pct',
                f'recall_{pct}pct',
                f'positive_set_fraction_{pct}pct'
            ])
        
        available_enrichment_cols = [c for c in enrichment_columns if c in enrichment_df.columns]
        other_enrichment_cols = [c for c in enrichment_df.columns if c not in available_enrichment_cols]
        enrichment_df = enrichment_df[available_enrichment_cols + other_enrichment_cols]
        
        # Save enrichment summary
        enrichment_df.to_csv(args.enrichment_output, sep='\t', index=False)
        logger.info(f"Saved enrichment summary to {args.enrichment_output}")
        print("\nEnrichment Summary:")
        print(enrichment_df.to_string(index=False))
    
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

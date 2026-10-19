#!/usr/bin/env python3
"""
Extract negative control sets from database comparison files.

This script extracts the last 100 pairs (lowest probability) that don't match
any database (BioGRID, CORUM, or PDB) as negative controls for each dataset.

Usage:
    python extract_negative_controls.py \
        --input-dir validation_pdb_and_databases \
        --output-dir validation_pdb_and_databases/negative_controls
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_negative_controls(input_file: Path, output_file: Path, n_pairs: int = 100):
    """
    Extract negative control pairs (not in any database).
    
    Args:
        input_file: Path to database comparison TSV file
        output_file: Path to output TSV file
        n_pairs: Number of pairs to extract (default: 100)
    """
    logger.info(f"Loading data from {input_file}")
    
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    logger.info(f"Total pairs in file: {len(df)}")
    
    # Filter for pairs not in any database
    # Check if In_Any_Database column exists, otherwise check individual columns
    if 'In_Any_Database' in df.columns:
        negative_df = df[df['In_Any_Database'] == False].copy()
    else:
        # Fallback: check individual database columns
        negative_df = df[
            (df.get('In_BioGRID', pd.Series([False] * len(df))) == False) &
            (df.get('In_CORUM', pd.Series([False] * len(df))) == False) &
            (df.get('Has_Shared_PDB', pd.Series([False] * len(df))) == False)
        ].copy()
    
    logger.info(f"Pairs not in any database: {len(negative_df)}")
    
    if len(negative_df) == 0:
        logger.warning("No negative control pairs found!")
        return
    
    # Sort by probability (ascending to get lowest probability first)
    negative_df = negative_df.sort_values('Probability', ascending=True)
    
    # Take the first n_pairs (lowest probability)
    negative_controls = negative_df.head(n_pairs).copy()
    
    # Keep only columns up to Probability
    columns_to_keep = ['Protein1', 'Protein1_UniProt', 'Protein2', 'Protein2_UniProt', 'Probability']
    negative_controls = negative_controls[columns_to_keep].copy()
    
    logger.info(f"Extracted {len(negative_controls)} negative control pairs")
    logger.info(f"  Probability range: {negative_controls['Probability'].min():.6f} - {negative_controls['Probability'].max():.6f}")
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    negative_controls.to_csv(output_file, sep='\t', index=False)
    
    logger.info(f"Saved negative controls to {output_file}")
    
    return negative_controls


def main():
    parser = argparse.ArgumentParser(
        description='Extract negative control pairs from database comparison files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('validation_pdb_and_databases'),
        help='Directory containing database comparison TSV files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('validation_pdb_and_databases/negative_controls'),
        help='Directory to save negative control files'
    )
    parser.add_argument(
        '--n-pairs',
        type=int,
        default=100,
        help='Number of pairs to extract (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Process each dataset
    datasets = ['WT', 'MOCK', 'VIF']
    
    for dataset_name in datasets:
        input_file = args.input_dir / f'cullin_{dataset_name.lower()}_database_comparison.tsv'
        output_file = args.output_dir / f'cullin_{dataset_name.lower()}_negative_controls.tsv'
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_name} dataset")
        logger.info(f"{'='*60}")
        
        extract_negative_controls(input_file, output_file, args.n_pairs)
    
    logger.info("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())

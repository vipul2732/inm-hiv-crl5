#!/usr/bin/env python3
"""
Validate predicted interactions using Cross-Linking Mass Spectrometry (XL-MS) data.

Cross-linking MS provides direct proximity evidence (< 30 Å) by creating covalent
bonds between nearby residues. This is strong direct contact evidence.

Usage:
    python validate_crosslinking.py \
        --predictions-tsv all_pairs.tsv \
        --xlms-database data/xlms_database.tsv \
        --output validation_results/
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import json

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def normalize_protein_name(name: str) -> str:
    """Normalize protein name for matching."""
    name = str(name).strip().upper()
    for suffix in ['_HUMAN', '_MOUSE', '_RAT', '_CHICKEN']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def load_xlms_database(xlms_file: Path, logger: Optional[logging.Logger] = None) -> Set[Tuple[str, str]]:
    """
    Load cross-linking MS data.
    
    Expected format: TSV with columns:
    - Protein1, Protein2 (or similar)
    - Or: UniProt_ID1, UniProt_ID2
    
    Args:
        xlms_file: Path to XL-MS database file
        logger: Optional logger
    
    Returns:
        Set of (protein1, protein2) tuples with cross-links
    """
    if not xlms_file.exists():
        if logger:
            logger.warning(f"XL-MS file not found: {xlms_file}")
        return set()
    
    interactions = set()
    
    try:
        df = pd.read_csv(xlms_file, sep='\t', low_memory=False)
        
        # Try to find protein columns
        protein_cols = None
        for col_pair in [
            ('Protein1', 'Protein2'),
            ('UniProt_ID1', 'UniProt_ID2'),
            ('protein1', 'protein2'),
            ('Protein_A', 'Protein_B'),
        ]:
            if col_pair[0] in df.columns and col_pair[1] in df.columns:
                protein_cols = col_pair
                break
        
        if protein_cols is None:
            if logger:
                logger.warning(f"Could not find protein columns in XL-MS file. Available: {df.columns.tolist()}")
            return set()
        
        for _, row in df.iterrows():
            p1 = str(row[protein_cols[0]]).strip()
            p2 = str(row[protein_cols[1]]).strip()
            
            if pd.isna(p1) or pd.isna(p2) or p1 == 'nan' or p2 == 'nan':
                continue
            
            # Normalize
            p1 = normalize_protein_name(p1)
            p2 = normalize_protein_name(p2)
            
            if p1 != p2:
                pair = tuple(sorted([p1, p2]))
                interactions.add(pair)
        
        if logger:
            logger.info(f"Loaded {len(interactions)} cross-linked pairs from XL-MS database")
    
    except Exception as e:
        if logger:
            logger.error(f"Error loading XL-MS database: {e}")
    
    return interactions


def validate_with_xlms(predictions: Dict[Tuple[str, str], float],
                       xlms_interactions: Set[Tuple[str, str]],
                       logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Validate predictions against XL-MS data.
    
    Args:
        predictions: Dict mapping (protein1, protein2) -> probability
        xlms_interactions: Set of cross-linked pairs
        logger: Optional logger
    
    Returns:
        DataFrame with validated pairs
    """
    validated_pairs = []
    
    for (p1, p2), prob in predictions.items():
        p1_norm = normalize_protein_name(p1)
        p2_norm = normalize_protein_name(p2)
        pair_norm = tuple(sorted([p1_norm, p2_norm]))
        
        if pair_norm in xlms_interactions:
            validated_pairs.append({
                'Protein1': p1,
                'Protein2': p2,
                'Probability': prob,
                'Evidence': 'Cross-Linking MS'
            })
    
    if validated_pairs:
        df = pd.DataFrame(validated_pairs)
        df = df.sort_values('Probability', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        if logger:
            logger.info(f"Found {len(df)} pairs with XL-MS cross-linking evidence")
    
    return pd.DataFrame(validated_pairs) if validated_pairs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Validate predictions using Cross-Linking Mass Spectrometry data'
    )
    parser.add_argument(
        '--predictions-tsv',
        type=Path,
        required=True,
        help='Path to TSV file with predictions'
    )
    parser.add_argument(
        '--xlms-database',
        type=Path,
        required=True,
        help='Path to XL-MS database file (TSV format)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("CROSS-LINKING MS VALIDATION")
    logger.info("=" * 70)
    
    # Load predictions
    pred_df = pd.read_csv(args.predictions_tsv, sep='\t')
    predictions = {}
    for _, row in pred_df.iterrows():
        pair = tuple(sorted([str(row['Protein1']), str(row['Protein2'])]))
        predictions[pair] = float(row['Probability'])
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load XL-MS data
    xlms_interactions = load_xlms_database(args.xlms_database, logger)
    
    if len(xlms_interactions) == 0:
        logger.warning("No XL-MS interactions loaded. Cannot validate.")
        return
    
    # Validate
    validated_df = validate_with_xlms(predictions, xlms_interactions, logger)
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    if len(validated_df) > 0:
        output_file = args.output / 'xlms_validated_pairs.tsv'
        validated_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved {len(validated_df)} validated pairs to {output_file}")
    
    logger.info("=" * 70)


if __name__ == '__main__':
    main()








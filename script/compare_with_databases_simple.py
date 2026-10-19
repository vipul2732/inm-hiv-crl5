#!/usr/bin/env python3
"""
Simplified version: Compare validated protein pairs against BioGRID, CORUM databases.
STRING comparison is skipped due to complexity of Ensembl ID mapping.

Usage:
    python compare_with_databases_simple.py \
        --input validation_pdb_and_databases/cullin_wt_pdb_validation.tsv \
        --output validation_pdb_and_databases/cullin_wt_database_comparison.tsv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Set, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_biogrid_interactions(biogrid_file: Path) -> Set[Tuple[str, str]]:
    """Load interactions from BioGRID MITAB file."""
    logger.info(f"Loading BioGRID interactions from {biogrid_file}")
    interactions = set()
    
    if not biogrid_file.exists():
        logger.warning(f"BioGRID file not found: {biogrid_file}")
        return interactions
    
    try:
        uniprot_pattern = re.compile(r'uniprot/(?:swiss-prot|trEMBL):([A-Z0-9-]+)')
        
        with open(biogrid_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            if not first_line.startswith('#'):
                f.seek(0)
            
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 4:
                    continue
                
                alt_ids_a = fields[2]
                alt_ids_b = fields[3]
                
                uniprot_ids_a = set(uniprot_pattern.findall(alt_ids_a))
                uniprot_ids_b = set(uniprot_pattern.findall(alt_ids_b))
                
                for up_a in uniprot_ids_a:
                    for up_b in uniprot_ids_b:
                        if up_a and up_b and up_a != up_b:
                            pair = tuple(sorted([up_a, up_b]))
                            interactions.add(pair)
        
        logger.info(f"Loaded {len(interactions)} unique BioGRID interactions")
        
    except Exception as e:
        logger.error(f"Error loading BioGRID: {e}")
    
    return interactions


def load_corum_interactions(corum_file: Path) -> Set[Tuple[str, str]]:
    """Load interactions from CORUM complexes file."""
    logger.info(f"Loading CORUM interactions from {corum_file}")
    interactions = set()
    
    if not corum_file.exists():
        logger.warning(f"CORUM file not found: {corum_file}")
        return interactions
    
    try:
        df = pd.read_csv(corum_file, sep='\t', low_memory=False)
        
        if 'subunits_uniprot_id' not in df.columns:
            logger.warning("CORUM file missing 'subunits_uniprot_id' column")
            return interactions
        
        for _, row in df.iterrows():
            subunits_str = str(row['subunits_uniprot_id'])
            if pd.isna(subunits_str) or subunits_str == 'nan':
                continue
            
            subunits = [s.strip() for s in subunits_str.split(';') if s.strip()]
            
            for i, sub1 in enumerate(subunits):
                for sub2 in subunits[i+1:]:
                    if sub1 and sub2 and sub1 != sub2:
                        pair = tuple(sorted([sub1, sub2]))
                        interactions.add(pair)
        
        logger.info(f"Loaded {len(interactions)} unique CORUM interactions")
        
    except Exception as e:
        logger.error(f"Error loading CORUM: {e}")
    
    return interactions


def compare_pairs_with_databases(
    input_file: Path,
    output_file: Path,
    biogrid_file: Path = None,
    corum_file: Path = None
) -> pd.DataFrame:
    """Compare validated pairs against databases."""
    logger.info(f"Loading validated pairs from {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    
    # Load database interactions
    biogrid_interactions = load_biogrid_interactions(biogrid_file) if biogrid_file else set()
    corum_interactions = load_corum_interactions(corum_file) if corum_file else set()
    
    # Compare pairs
    logger.info("Comparing pairs against databases...")
    results = []
    
    for idx, row in df.iterrows():
        uniprot_1 = str(row['Protein1_UniProt']).strip()
        uniprot_2 = str(row['Protein2_UniProt']).strip()
        
        if not uniprot_1 or not uniprot_2 or uniprot_1 == 'nan' or uniprot_2 == 'nan':
            continue
        
        pair = tuple(sorted([uniprot_1, uniprot_2]))
        
        in_biogrid = pair in biogrid_interactions
        in_corum = pair in corum_interactions
        
        db_count = sum([in_biogrid, in_corum])
        in_any_db = db_count > 0
        
        results.append({
            'Protein1': row['Protein1'],
            'Protein1_UniProt': uniprot_1,
            'Protein2': row['Protein2'],
            'Protein2_UniProt': uniprot_2,
            'Probability': row['Probability'],
            'Has_Shared_PDB': row.get('Has_Shared_PDB', False),
            'In_BioGRID': in_biogrid,
            'In_CORUM': in_corum,
            'In_Any_Database': in_any_db,
            'Database_Count': db_count
        })
    
    results_df = pd.DataFrame(results)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, sep='\t', index=False)
    
    # Statistics
    total_pairs = len(results_df)
    in_biogrid_count = results_df['In_BioGRID'].sum()
    in_corum_count = results_df['In_CORUM'].sum()
    in_any_db_count = results_df['In_Any_Database'].sum()
    
    logger.info(f"Comparison complete!")
    logger.info(f"  Total pairs: {total_pairs}")
    logger.info(f"  In BioGRID: {in_biogrid_count} ({in_biogrid_count/total_pairs*100:.2f}%)")
    logger.info(f"  In CORUM: {in_corum_count} ({in_corum_count/total_pairs*100:.2f}%)")
    logger.info(f"  In any database: {in_any_db_count} ({in_any_db_count/total_pairs*100:.2f}%)")
    logger.info(f"Saved results to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Compare validated pairs against databases')
    parser.add_argument('--input', type=Path, required=True, help='Input validation TSV file')
    parser.add_argument('--output', type=Path, required=True, help='Output TSV file')
    parser.add_argument('--biogrid', type=Path, default=Path('database/BIOGRID-ALL-5.0.253.mitab.txt'))
    parser.add_argument('--corum', type=Path, default=Path('database/corum_humanComplexes.tsv'))
    
    args = parser.parse_args()
    compare_pairs_with_databases(args.input, args.output, args.biogrid, args.corum)
    return 0


if __name__ == '__main__':
    exit(main())

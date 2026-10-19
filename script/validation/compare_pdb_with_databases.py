#!/usr/bin/env python3
"""
Compare PDB-validated pairs with downloaded databases (CORUM, HuMAP2, STRING, IntAct, BioGRID).

This script:
1. Loads PDB-validated pairs for a dataset
2. Loads all database pairs (CORUM, HuMAP2, STRING, IntAct, BioGRID)
3. Compares overlap and generates statistics

Usage:
    python compare_pdb_with_databases.py --pdb-results pdb_validation_test --database-dir ../database --dataset cullin_mock_config_multi_seeds --output comparison_results/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set, Tuple
import warnings

import pandas as pd

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
    """Normalize protein name for comparison."""
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def load_corum_pairs(corum_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load CORUM pairs."""
    logger.info(f"Loading CORUM from {corum_path}")
    df = pd.read_csv(corum_path, sep='\t', low_memory=False)
    pairs = set()
    
    for _, row in df.iterrows():
        gene_names_str = str(row.get('subunits_gene_name', ''))
        if pd.isna(gene_names_str) or gene_names_str == '':
            continue
        gene_names = [normalize_protein_name(g.strip()) for g in gene_names_str.split(';')]
        gene_names = [g for g in gene_names if g and g != 'NAN']
        if len(gene_names) >= 2:
            for i in range(len(gene_names)):
                for j in range(i + 1, len(gene_names)):
                    pairs.add(tuple(sorted([gene_names[i], gene_names[j]])))
    
    logger.info(f"  Loaded {len(pairs)} CORUM pairs")
    return pairs


def load_humap2_pairs(humap2_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load HuMAP2 pairs."""
    logger.info(f"Loading HuMAP2 from {humap2_path}")
    df = pd.read_csv(humap2_path, sep=',', low_memory=False)
    pairs = set()
    
    for _, row in df.iterrows():
        gene_names_str = str(row.get('genenames', ''))
        if pd.isna(gene_names_str) or gene_names_str == '':
            continue
        gene_names = [normalize_protein_name(g.strip()) for g in gene_names_str.split()]
        gene_names = [g for g in gene_names if g and g != 'NAN']
        if len(gene_names) >= 2:
            for i in range(len(gene_names)):
                for j in range(i + 1, len(gene_names)):
                    pairs.add(tuple(sorted([gene_names[i], gene_names[j]])))
    
    logger.info(f"  Loaded {len(pairs)} HuMAP2 pairs")
    return pairs


def load_string_pairs(string_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load STRING pairs (human interactions, combined score >= 400)."""
    logger.info(f"Loading STRING from {string_path}")
    pairs = set()
    
    try:
        # STRING format: protein1 protein2 combined_score ...
        # Protein IDs are in format "9606.ENSP00000000233" (Ensembl protein IDs)
        # We'll skip STRING for now as it uses Ensembl protein IDs, not gene names
        # Converting would require mapping Ensembl protein IDs to gene names
        logger.warning("  STRING uses Ensembl protein IDs, skipping (would need ID mapping)")
        return pairs
        
        # df = pd.read_csv(string_path, sep=' ', low_memory=False)
        # logger.info(f"  Read {len(df)} STRING interactions")
        # # ... rest of code ...
    except Exception as e:
        logger.warning(f"Error loading STRING: {e}")
    
    return pairs


def load_intact_pairs(intact_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load IntAct pairs."""
    logger.info(f"Loading IntAct from {intact_path}")
    pairs = set()
    
    try:
        # IntAct MITAB format: tab-separated
        # Gene names are in "Alias(es) interactor A" column (5th column, 0-indexed)
        # Format: psi-mi:SYNJ1(gene name)|uniprotkb:SYNJ1(gene name)|...
        df = pd.read_csv(intact_path, sep='\t', low_memory=False, comment='#', on_bad_lines='skip')
        logger.info(f"  Read {len(df)} IntAct interactions")
        
        # Get gene names from alias columns (columns 4 and 5, 0-indexed)
        alias_col_a = df.columns[4] if len(df.columns) > 4 else None
        alias_col_b = df.columns[5] if len(df.columns) > 5 else None
        
        def extract_gene_name(alias_str):
            """Extract gene name from alias string."""
            if pd.isna(alias_str) or alias_str == '':
                return None
            alias_str = str(alias_str)
            # Look for (gene name) pattern
            import re
            matches = re.findall(r'\(gene name\)\|?([^|()]+)', alias_str)
            if matches:
                return normalize_protein_name(matches[0].split('(')[0].strip())
            # Alternative: try to extract from format like "SYNJ1(gene name)"
            matches = re.findall(r'([A-Z0-9_-]+)\(gene name\)', alias_str)
            if matches:
                return normalize_protein_name(matches[0])
            return None
        
        if alias_col_a and alias_col_b:
            for _, row in df.iterrows():
                p1 = extract_gene_name(row[alias_col_a])
                p2 = extract_gene_name(row[alias_col_b])
                if p1 and p2 and p1 != p2 and p1 != 'NAN' and p2 != 'NAN':
                    pairs.add(tuple(sorted([p1, p2])))
        
        logger.info(f"  Loaded {len(pairs)} IntAct pairs")
    except Exception as e:
        logger.warning(f"Error loading IntAct: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return pairs


def load_biogrid_pairs(biogrid_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load BioGRID pairs (human interactions only)."""
    logger.info(f"Loading BioGRID from {biogrid_path}")
    pairs = set()
    
    try:
        # BioGRID MITAB format: tab-separated
        # Gene names are in "Aliases Interactor A" column (5th column, 0-indexed)
        # Format: entrez gene/locuslink:JNKK(gene name synonym)|entrez gene/locuslink:MAP2K4|...
        df = pd.read_csv(biogrid_path, sep='\t', low_memory=False, comment='#', on_bad_lines='skip')
        logger.info(f"  Read {len(df)} BioGRID interactions")
        
        # Filter for human interactions only (taxid 9606)
        if 'Taxid Interactor A' in df.columns:
            df = df[(df['Taxid Interactor A'] == 'taxid:9606') & (df['Taxid Interactor B'] == 'taxid:9606')]
            logger.info(f"  Filtered to {len(df)} human interactions")
        
        # Get gene names from alias columns (columns 4 and 5, 0-indexed)
        alias_col_a = df.columns[4] if len(df.columns) > 4 else None
        alias_col_b = df.columns[5] if len(df.columns) > 5 else None
        
        def extract_gene_name(alias_str):
            """Extract gene name from alias string."""
            if pd.isna(alias_str) or alias_str == '':
                return None
            alias_str = str(alias_str)
            # Look for (gene name synonym) or direct gene names
            import re
            # Try to find pattern like "MAP2K4" (gene name before parentheses or at start)
            # BioGRID format: entrez gene/locuslink:MAP2K4|uniprot/swiss-prot:P45985
            # Extract the part after colon and before pipe or parenthesis
            parts = alias_str.split('|')
            for part in parts:
                if ':' in part:
                    gene_candidate = part.split(':')[1].split('(')[0].strip()
                    if gene_candidate and gene_candidate != 'NAN':
                        return normalize_protein_name(gene_candidate)
            return None
        
        if alias_col_a and alias_col_b:
            for _, row in df.iterrows():
                p1 = extract_gene_name(row[alias_col_a])
                p2 = extract_gene_name(row[alias_col_b])
                if p1 and p2 and p1 != p2 and p1 != 'NAN' and p2 != 'NAN':
                    pairs.add(tuple(sorted([p1, p2])))
        
        logger.info(f"  Loaded {len(pairs)} BioGRID pairs")
    except Exception as e:
        logger.warning(f"Error loading BioGRID: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return pairs


def load_pdb_validated_pairs(pdb_results_dir: Path, dataset_name: str, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load PDB-validated pairs for a dataset."""
    pdb_file = pdb_results_dir / f"{dataset_name}_pdb_validated_pairs.tsv"
    
    if not pdb_file.exists():
        logger.error(f"PDB validated pairs file not found: {pdb_file}")
        return set()
    
    logger.info(f"Loading PDB-validated pairs from {pdb_file}")
    df = pd.read_csv(pdb_file, sep='\t')
    pairs = set()
    
    for _, row in df.iterrows():
        p1 = normalize_protein_name(str(row['Protein1']))
        p2 = normalize_protein_name(str(row['Protein2']))
        pairs.add(tuple(sorted([p1, p2])))
    
    logger.info(f"  Loaded {len(pairs)} PDB-validated pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Compare PDB-validated pairs with databases'
    )
    parser.add_argument(
        '--pdb-results',
        type=Path,
        required=True,
        help='Directory containing PDB validation results'
    )
    parser.add_argument(
        '--database-dir',
        type=Path,
        required=True,
        help='Directory containing database files'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., cullin_mock_config_multi_seeds)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('pdb_database_comparison'),
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PDB VALIDATED PAIRS vs DATABASES COMPARISON")
    logger.info("=" * 60)
    
    # Load PDB-validated pairs
    pdb_pairs = load_pdb_validated_pairs(args.pdb_results, args.dataset, logger)
    
    if not pdb_pairs:
        logger.error("No PDB-validated pairs found!")
        return
    
    # Load all databases
    logger.info("\n" + "=" * 60)
    logger.info("Loading databases...")
    logger.info("=" * 60)
    
    all_db_pairs = {}
    
    # CORUM
    corum_file = args.database_dir / "corum_humanComplexes.tsv"
    if corum_file.exists():
        all_db_pairs['CORUM'] = load_corum_pairs(corum_file, logger)
    else:
        logger.warning(f"CORUM file not found: {corum_file}")
        all_db_pairs['CORUM'] = set()
    
    # HuMAP2
    humap2_file = args.database_dir / "humap2.text"
    if humap2_file.exists():
        all_db_pairs['HuMAP2'] = load_humap2_pairs(humap2_file, logger)
    else:
        logger.warning(f"HuMAP2 file not found: {humap2_file}")
        all_db_pairs['HuMAP2'] = set()
    
    # STRING
    string_file = args.database_dir / "string_human_protein_links_detailed.txt"
    if string_file.exists():
        all_db_pairs['STRING'] = load_string_pairs(string_file, logger)
    else:
        logger.warning(f"STRING file not found: {string_file}")
        all_db_pairs['STRING'] = set()
    
    # IntAct
    intact_file = args.database_dir / "intact_human_interactions.txt"
    if intact_file.exists():
        all_db_pairs['IntAct'] = load_intact_pairs(intact_file, logger)
    else:
        logger.warning(f"IntAct file not found: {intact_file}")
        all_db_pairs['IntAct'] = set()
    
    # BioGRID
    biogrid_file = args.database_dir / "BIOGRID-ALL-5.0.253.mitab.txt"
    if biogrid_file.exists():
        all_db_pairs['BioGRID'] = load_biogrid_pairs(biogrid_file, logger)
    else:
        logger.warning(f"BioGRID file not found: {biogrid_file}")
        all_db_pairs['BioGRID'] = set()
    
    # Compute overlap
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    
    stats = {
        'dataset': args.dataset,
        'num_pdb_validated_pairs': len(pdb_pairs),
        'database_overlap': {}
    }
    
    # Combined database (union of all databases)
    all_combined = set()
    for db_name, db_pairs in all_db_pairs.items():
        all_combined.update(db_pairs)
    
    all_db_pairs['Any_Database'] = all_combined
    
    logger.info(f"\nPDB-validated pairs: {len(pdb_pairs)}")
    logger.info(f"\nDatabase sizes:")
    for db_name, db_pairs in all_db_pairs.items():
        logger.info(f"  {db_name}: {len(db_pairs)} pairs")
    
    logger.info(f"\nOverlap of PDB-validated pairs with databases:")
    for db_name, db_pairs in all_db_pairs.items():
        overlap = pdb_pairs.intersection(db_pairs)
        overlap_count = len(overlap)
        overlap_pct = (overlap_count / len(pdb_pairs) * 100) if pdb_pairs else 0
        
        stats['database_overlap'][db_name] = {
            'overlap_count': overlap_count,
            'overlap_percentage': overlap_pct,
            'database_size': len(db_pairs)
        }
        
        logger.info(f"  {db_name}: {overlap_count}/{len(pdb_pairs)} ({overlap_pct:.1f}%)")
        
        # Save overlapping pairs
        if overlap:
            overlap_df = pd.DataFrame([list(pair) for pair in sorted(overlap)], columns=['Protein1', 'Protein2'])
            overlap_file = args.output / f"{args.dataset}_pdb_overlap_{db_name.lower()}.tsv"
            overlap_df.to_csv(overlap_file, sep='\t', index=False)
            logger.info(f"    Saved {overlap_count} overlapping pairs to {overlap_file}")
    
    # Save statistics
    stats_file = args.output / f"{args.dataset}_pdb_database_comparison_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

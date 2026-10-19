#!/usr/bin/env python3
"""
Compare validated protein pairs against BioGRID, CORUM, and STRING databases.

This script:
1. Loads validated pairs with UniProt IDs
2. Compares against BioGRID (extracts UniProt IDs from MITAB format)
3. Compares against CORUM (extracts pairs from complex subunits)
4. Compares against STRING (uses gene names for matching)
5. Saves comparison results

Usage:
    python compare_with_databases.py \
        --input validation_pdb_and_databases/cullin_wt_pdb_validation.tsv \
        --output validation_pdb_and_databases/cullin_wt_database_comparison.tsv \
        --biogrid database/BIOGRID-ALL-5.0.253.mitab.txt \
        --corum database/corum_humanComplexes.tsv \
        --string database/string_human_protein_links_detailed.txt
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
    """
    Load interactions from BioGRID MITAB file.
    
    Args:
        biogrid_file: Path to BioGRID MITAB file
    
    Returns:
        Set of (uniprot_id1, uniprot_id2) tuples (sorted)
    """
    logger.info(f"Loading BioGRID interactions from {biogrid_file}")
    interactions = set()
    
    if not biogrid_file.exists():
        logger.warning(f"BioGRID file not found: {biogrid_file}")
        return interactions
    
    try:
        # BioGRID MITAB format: UniProt IDs are in columns 2 and 3 (Alt IDs)
        # Format: uniprot/swiss-prot:P45985 or uniprot/swiss-prot:P45985|uniprot/trEMBL:Q12345
        uniprot_pattern = re.compile(r'uniprot/(?:swiss-prot|trEMBL):([A-Z0-9-]+)')
        
        with open(biogrid_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header if it starts with #
            first_line = f.readline()
            if not first_line.startswith('#'):
                f.seek(0)
            
            for line_num, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 3:
                    continue
                
                # Extract UniProt IDs from Alt IDs columns (index 2 and 3)
                alt_ids_a = fields[2] if len(fields) > 2 else ''
                alt_ids_b = fields[3] if len(fields) > 3 else ''
                
                # Find all UniProt IDs in each field
                uniprot_ids_a = set(uniprot_pattern.findall(alt_ids_a))
                uniprot_ids_b = set(uniprot_pattern.findall(alt_ids_b))
                
                # Create pairs from all combinations (prefer Swiss-Prot over TrEMBL)
                for up_a in uniprot_ids_a:
                    for up_b in uniprot_ids_b:
                        if up_a and up_b and up_a != up_b:
                            # Sort to ensure consistent ordering
                            pair = tuple(sorted([up_a, up_b]))
                            interactions.add(pair)
        
        logger.info(f"Loaded {len(interactions)} unique BioGRID interactions")
        
    except Exception as e:
        logger.error(f"Error loading BioGRID: {e}")
    
    return interactions


def load_corum_interactions(corum_file: Path) -> Set[Tuple[str, str]]:
    """
    Load interactions from CORUM complexes file.
    
    Args:
        corum_file: Path to CORUM TSV file
    
    Returns:
        Set of (uniprot_id1, uniprot_id2) tuples (sorted)
    """
    logger.info(f"Loading CORUM interactions from {corum_file}")
    interactions = set()
    
    if not corum_file.exists():
        logger.warning(f"CORUM file not found: {corum_file}")
        return interactions
    
    try:
        df = pd.read_csv(corum_file, sep='\t', low_memory=False)
        
        # CORUM has subunits_uniprot_id column with semicolon-separated UniProt IDs
        if 'subunits_uniprot_id' not in df.columns:
            logger.warning("CORUM file missing 'subunits_uniprot_id' column")
            return interactions
        
        for _, row in df.iterrows():
            subunits_str = str(row['subunits_uniprot_id'])
            if pd.isna(subunits_str) or subunits_str == 'nan':
                continue
            
            # Split by semicolon and create all pairs within complex
            subunits = [s.strip() for s in subunits_str.split(';') if s.strip()]
            
            # Create all pairs within the complex
            for i, sub1 in enumerate(subunits):
                for sub2 in subunits[i+1:]:
                    if sub1 and sub2 and sub1 != sub2:
                        pair = tuple(sorted([sub1, sub2]))
                        interactions.add(pair)
        
        logger.info(f"Loaded {len(interactions)} unique CORUM interactions")
        
    except Exception as e:
        logger.error(f"Error loading CORUM: {e}")
    
    return interactions


def load_string_interactions_simple(string_file: Path, gene_to_uniprot: Dict[str, str] = None) -> Set[Tuple[str, str]]:
    """
    Load interactions from STRING file using gene name matching.
    
    Since STRING uses Ensembl IDs and mapping is complex, we use a gene-name based approach:
    1. Map our UniProt IDs to gene names
    2. For STRING, we'll match by gene names (simplified approach)
    
    Args:
        string_file: Path to STRING file
        gene_to_uniprot: Mapping from gene name to UniProt ID
    
    Returns:
        Set of (uniprot_id1, uniprot_id2) tuples (sorted)
    """
    logger.info(f"Loading STRING interactions from {string_file}")
    interactions = set()
    
    if not string_file.exists():
        logger.warning(f"STRING file not found: {string_file}")
        return interactions
    
    if gene_to_uniprot is None or len(gene_to_uniprot) == 0:
        logger.warning("No gene to UniProt mapping provided. STRING comparison skipped.")
        return interactions
    
    try:
        # STRING format: protein1 protein2 ... combined_score
        # protein1/protein2 are Ensembl IDs like 9606.ENSP00000000233
        # We'll use a gene-name based matching approach
        
        # Create reverse mapping: UniProt -> gene name
        uniprot_to_gene = {v: k for k, v in gene_to_uniprot.items()}
        
        # We need to map Ensembl IDs to gene names
        # Use Ensembl REST API to get gene names from protein IDs
        import requests
        import time
        
        logger.info("Reading STRING file and mapping Ensembl IDs to gene names...")
        logger.info("Note: This uses Ensembl API and may take a while...")
        
        # Cache for Ensembl ID -> gene name mapping
        ensembl_to_gene = {}
        string_gene_pairs = set()
        
        # First pass: read STRING file and collect unique Ensembl IDs
        logger.info("Collecting unique Ensembl IDs from STRING file...")
        unique_ensembl = set()
        line_count = 0
        
        with open(string_file, 'r') as f:
            first_line = f.readline()
            if not first_line.startswith('protein'):
                f.seek(0)
            
            for line in f:
                line_count += 1
                if line_count % 500000 == 0:
                    logger.info(f"  Processed {line_count} lines...")
                
                fields = line.strip().split()
                if len(fields) < 2:
                    continue
                
                ensembl_1 = fields[0]
                ensembl_2 = fields[1]
                unique_ensembl.add(ensembl_1)
                unique_ensembl.add(ensembl_2)
        
        logger.info(f"Found {len(unique_ensembl)} unique Ensembl IDs")
        logger.info("Mapping Ensembl IDs to gene names via Ensembl API...")
        
        # Map Ensembl protein IDs to gene names
        mapped_count = 0
        for i, ensembl_id_full in enumerate(unique_ensembl, 1):
            if i % 100 == 0:
                logger.info(f"  Mapped {mapped_count}/{i} Ensembl IDs...")
            
            # Extract protein ID (remove species prefix 9606.)
            ensembl_protein_id = ensembl_id_full.split('.')[1] if '.' in ensembl_id_full else ensembl_id_full
            
            try:
                # Query Ensembl for gene name
                url = f"https://rest.ensembl.org/lookup/id/{ensembl_protein_id}"
                params = {"expand": "0", "content-type": "application/json"}
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Try to get gene name from various fields
                    gene_name = None
                    if 'display_name' in data:
                        gene_name = data['display_name']
                    elif 'name' in data:
                        gene_name = data['name']
                    
                    if gene_name:
                        ensembl_to_gene[ensembl_id_full] = gene_name
                        mapped_count += 1
                
                time.sleep(0.1)  # Rate limiting
            except:
                pass
        
        logger.info(f"Mapped {mapped_count} Ensembl IDs to gene names")
        
        # Second pass: read STRING file and create gene-based pairs
        logger.info("Creating gene-based interaction pairs from STRING...")
        with open(string_file, 'r') as f:
            first_line = f.readline()
            if not first_line.startswith('protein'):
                f.seek(0)
            
            for line in f:
                fields = line.strip().split()
                if len(fields) < 2:
                    continue
                
                ensembl_1 = fields[0]
                ensembl_2 = fields[1]
                
                gene_1 = ensembl_to_gene.get(ensembl_1)
                gene_2 = ensembl_to_gene.get(ensembl_2)
                
                if gene_1 and gene_2:
                    # Get UniProt IDs for these genes
                    uniprot_1 = gene_to_uniprot.get(gene_1)
                    uniprot_2 = gene_to_uniprot.get(gene_2)
                    
                    if uniprot_1 and uniprot_2 and uniprot_1 != uniprot_2:
                        pair = tuple(sorted([uniprot_1, uniprot_2]))
                        interactions.add(pair)
        
        logger.info(f"Created {len(interactions)} STRING interactions with UniProt IDs")
        
    except Exception as e:
        logger.error(f"Error loading STRING: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return interactions


def compare_pairs_with_databases(
    input_file: Path,
    output_file: Path,
    biogrid_file: Path = None,
    corum_file: Path = None,
    string_file: Path = None
) -> pd.DataFrame:
    """
    Compare validated pairs against databases.
    
    Args:
        input_file: Path to validation TSV file
        output_file: Path to output TSV file with comparison results
        biogrid_file: Path to BioGRID MITAB file
        corum_file: Path to CORUM TSV file
        string_file: Path to STRING file
    
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Loading validated pairs from {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    
    # Load database interactions
    biogrid_interactions = set()
    corum_interactions = set()
    string_interactions = set()
    
    if biogrid_file:
        biogrid_interactions = load_biogrid_interactions(biogrid_file)
    
    if corum_file:
        corum_interactions = load_corum_interactions(corum_file)
    
    if string_file:
        # Create gene to UniProt mapping from our data
        gene_to_uniprot = {}
        for _, row in df.iterrows():
            gene1 = row['Protein1']
            gene2 = row['Protein2']
            uniprot1 = str(row['Protein1_UniProt']).strip()
            uniprot2 = str(row['Protein2_UniProt']).strip()
            if uniprot1 and uniprot1 != 'nan':
                gene_to_uniprot[gene1] = uniprot1
            if uniprot2 and uniprot2 != 'nan':
                gene_to_uniprot[gene2] = uniprot2
        
        string_interactions = load_string_interactions(string_file, gene_to_uniprot)
    
    # Compare pairs
    logger.info("Comparing pairs against databases...")
    results = []
    
    for idx, row in df.iterrows():
        uniprot_1 = str(row['Protein1_UniProt']).strip()
        uniprot_2 = str(row['Protein2_UniProt']).strip()
        
        if not uniprot_1 or not uniprot_2 or uniprot_1 == 'nan' or uniprot_2 == 'nan':
            continue
        
        # Create sorted pair for comparison
        pair = tuple(sorted([uniprot_1, uniprot_2]))
        
        # Check against databases
        in_biogrid = pair in biogrid_interactions
        in_corum = pair in corum_interactions
        in_string = pair in string_interactions
        
        # Count how many databases contain this pair
        db_count = sum([in_biogrid, in_corum, in_string])
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
            'In_STRING': in_string,
            'In_Any_Database': in_any_db,
            'Database_Count': db_count
        })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, sep='\t', index=False)
    
    # Statistics
    total_pairs = len(results_df)
    in_biogrid_count = results_df['In_BioGRID'].sum()
    in_corum_count = results_df['In_CORUM'].sum()
    in_string_count = results_df['In_STRING'].sum()
    in_any_db_count = results_df['In_Any_Database'].sum()
    
    logger.info(f"Comparison complete!")
    logger.info(f"  Total pairs: {total_pairs}")
    logger.info(f"  In BioGRID: {in_biogrid_count} ({in_biogrid_count/total_pairs*100:.2f}%)")
    logger.info(f"  In CORUM: {in_corum_count} ({in_corum_count/total_pairs*100:.2f}%)")
    logger.info(f"  In STRING: {in_string_count} ({in_string_count/total_pairs*100:.2f}%)")
    logger.info(f"  In any database: {in_any_db_count} ({in_any_db_count/total_pairs*100:.2f}%)")
    logger.info(f"Saved results to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Compare validated protein pairs against BioGRID, CORUM, and STRING databases',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input validation TSV file with UniProt IDs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output TSV file with comparison results'
    )
    parser.add_argument(
        '--biogrid',
        type=Path,
        default=Path('database/BIOGRID-ALL-5.0.253.mitab.txt'),
        help='Path to BioGRID MITAB file'
    )
    parser.add_argument(
        '--corum',
        type=Path,
        default=Path('database/corum_humanComplexes.tsv'),
        help='Path to CORUM TSV file'
    )
    parser.add_argument(
        '--string',
        type=Path,
        default=Path('database/string_human_protein_links_detailed.txt'),
        help='Path to STRING file'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results_df = compare_pairs_with_databases(
        args.input,
        args.output,
        args.biogrid,
        args.corum,
        args.string
    )
    
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())

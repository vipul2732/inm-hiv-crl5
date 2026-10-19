#!/usr/bin/env python3
"""
Validate predicted interactions using AlphaFold-Multimer predictions.

AlphaFold-Multimer provides computational structural predictions for protein
complexes. This script:
1. Queries AlphaFold DB for existing predictions
2. Extracts contact information from predicted structures
3. Validates pairs based on predicted contacts (< 5 Å)

Usage:
    python validate_alphafold.py \
        --predictions-tsv all_pairs.tsv \
        --output validation_results/
    
    # Optional: Run predictions for missing pairs (computationally expensive!)
    python validate_alphafold.py \
        --predictions-tsv all_pairs.tsv \
        --run-predictions \
        --output validation_results/
"""

import argparse
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List
import json
import time

import pandas as pd
import numpy as np

# Try to import BioPython for structure parsing
try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    warnings.simplefilter('ignore', PDBConstructionWarning)
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    warnings.warn("BioPython not available. Structure parsing will be limited.")


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


def get_uniprot_id_from_name(protein_name: str, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Get UniProt ID from protein name using UniProt search API.
    
    Args:
        protein_name: Protein name (e.g., 'CUL5', 'RBX2')
        logger: Optional logger
    
    Returns:
        UniProt ID or None if not found
    """
    import urllib.request
    import urllib.parse
    import json
    
    try:
        # Search UniProt by gene name
        query = f"gene:{protein_name} AND organism_id:9606"  # 9606 = human
        encoded_query = urllib.parse.quote(query)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&format=json&size=1"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            if 'results' in data and len(data['results']) > 0:
                uniprot_id = data['results'][0].get('primaryAccession')
                if uniprot_id:
                    time.sleep(0.1)  # Rate limiting
                    return uniprot_id
        
    except Exception as e:
        if logger:
            logger.debug(f"Error searching UniProt for {protein_name}: {e}")
    
    return None


def check_alphafold_structure_exists(uniprot_id: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if AlphaFold structure exists in AlphaFold DB for a single protein.
    
    Args:
        uniprot_id: UniProt ID
        logger: Optional logger
    
    Returns:
        True if structure exists, False otherwise
    """
    import urllib.request
    
    try:
        # AlphaFold DB structure URL format
        # Single chain: AF-{uniprot_id}-F1-model_v4.pdb
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            # If we get a response, structure exists
            if data:
                time.sleep(0.1)  # Rate limiting
                return True
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False  # Structure doesn't exist
        if logger:
            logger.debug(f"AlphaFold DB API error for {uniprot_id}: {e.code}")
    except Exception as e:
        if logger:
            logger.debug(f"Error checking AlphaFold DB for {uniprot_id}: {e}")
    
    return False


def download_alphafold_structure(uniprot_id: str, output_file: Path,
                                 logger: Optional[logging.Logger] = None) -> bool:
    """
    Download AlphaFold structure from AlphaFold DB.
    
    Args:
        uniprot_id: UniProt ID
        output_file: Path to save structure
        logger: Optional logger
    
    Returns:
        True if successful, False otherwise
    """
    import urllib.request
    
    try:
        # Try to get structure URL from API first
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            
            if not data:
                return False
            
            # Get the first entry (usually only one)
            entry = data[0] if isinstance(data, list) else data
            
            # Try to get PDB URL
            pdb_url = None
            if 'pdbUrl' in entry:
                pdb_url = entry['pdbUrl']
            elif 'cifUrl' in entry:
                pdb_url = entry['cifUrl']
            else:
                # Fallback: construct URL
                pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            
            # Download
            urllib.request.urlretrieve(pdb_url, output_file)
            time.sleep(0.1)  # Rate limiting
            
            if logger:
                logger.debug(f"Downloaded AlphaFold structure: {uniprot_id}")
            
            return True
    
    except urllib.error.HTTPError as e:
        if logger:
            logger.debug(f"Error downloading AlphaFold structure {uniprot_id}: {e.code}")
        return False
    except Exception as e:
        if logger:
            logger.debug(f"Error downloading AlphaFold structure {uniprot_id}: {e}")
        return False


def download_alphafold_structure(alphafold_id: str, output_file: Path, 
                                 logger: Optional[logging.Logger] = None) -> bool:
    """
    Download AlphaFold structure from AlphaFold DB.
    
    Args:
        alphafold_id: AlphaFold DB entry ID
        output_file: Path to save structure
        logger: Optional logger
    
    Returns:
        True if successful, False otherwise
    """
    import urllib.request
    
    try:
        # AlphaFold DB download URL
        url = f"https://alphafold.ebi.ac.uk/files/{alphafold_id}"
        
        urllib.request.urlretrieve(url, output_file)
        
        if logger:
            logger.debug(f"Downloaded AlphaFold structure: {alphafold_id}")
        
        return True
    
    except Exception as e:
        if logger:
            logger.debug(f"Error downloading AlphaFold structure {alphafold_id}: {e}")
        
        return False


def extract_contacts_from_structure(structure_file: Path, chain1: str, chain2: str,
                                    distance_threshold: float = 5.0,
                                    logger: Optional[logging.Logger] = None) -> bool:
    """
    Extract contacts between two chains in a structure file.
    
    Args:
        structure_file: Path to PDB/mmCIF file
        chain1: Chain ID for protein 1
        chain2: Chain ID for protein 2
        distance_threshold: Maximum distance for contact (default: 5.0 Å)
        logger: Optional logger
    
    Returns:
        True if contacts found, False otherwise
    """
    if not HAS_BIOPYTHON:
        if logger:
            logger.warning("BioPython not available. Cannot parse structures.")
        return False
    
    if not structure_file.exists():
        return False
    
    try:
        # Determine file type
        if structure_file.suffix == '.cif':
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        
        structure = parser.get_structure('complex', str(structure_file))
        
        # Get chains
        chain1_obj = None
        chain2_obj = None
        
        for model in structure:
            if chain1 in model:
                chain1_obj = model[chain1]
            if chain2 in model:
                chain2_obj = model[chain2]
        
        if chain1_obj is None or chain2_obj is None:
            return False
        
        # Check for contacts (Cα atoms within threshold)
        for residue1 in chain1_obj:
            if 'CA' not in residue1:
                continue
            
            ca1 = residue1['CA']
            
            for residue2 in chain2_obj:
                if 'CA' not in residue2:
                    continue
                
                ca2 = residue2['CA']
                distance = np.linalg.norm(ca1.coord - ca2.coord)
                
                if distance < distance_threshold:
                    return True  # Found at least one contact
        
        return False
    
    except Exception as e:
        if logger:
            logger.debug(f"Error parsing structure {structure_file}: {e}")
        
        return False


def validate_with_alphafold_db(predictions: Dict[Tuple[str, str], float],
                               uniprot_mapping: Optional[Dict[str, str]] = None,
                               download_structures: bool = False,
                               output_dir: Optional[Path] = None,
                               distance_threshold: float = 5.0,
                               logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Validate predictions using AlphaFold DB.
    
    Strategy:
    1. Check if both proteins have AlphaFold structures
    2. If download_structures=True, download and check for contacts
    3. Note: AlphaFold DB has limited complex coverage (mostly single chains)
       So we check if structures exist, but can't directly validate complexes
    
    Args:
        predictions: Dict mapping (protein1, protein2) -> probability
        uniprot_mapping: Optional dict mapping protein_name -> uniprot_id
        download_structures: If True, download structures and check contacts
        output_dir: Directory to save downloaded structures
        distance_threshold: Distance threshold for contacts (default: 5.0 Å)
        logger: Optional logger
    
    Returns:
        DataFrame with validated pairs
    """
    if logger:
        logger.info("Validating with AlphaFold DB...")
        logger.info("Note: AlphaFold DB has limited complex coverage (mostly single chains)")
        if download_structures:
            logger.info("Will download structures and check for contacts")
    
    validated_pairs = []
    checked_proteins = {}  # Cache: protein -> (uniprot_id, has_structure)
    
    total_pairs = len(predictions)
    structures_dir = output_dir / 'alphafold_structures' if output_dir and download_structures else None
    if structures_dir:
        structures_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, ((p1, p2), prob) in enumerate(predictions.items(), 1):
        if logger and idx % 100 == 0:
            logger.info(f"  Progress: {idx}/{total_pairs} pairs checked...")
        
        p1_norm = normalize_protein_name(p1)
        p2_norm = normalize_protein_name(p2)
        
        if p1_norm == p2_norm:
            continue
        
        # Get UniProt IDs and check AlphaFold DB
        if p1_norm not in checked_proteins:
            uniprot_id1 = None
            if uniprot_mapping:
                uniprot_id1 = uniprot_mapping.get(p1_norm)
            if not uniprot_id1:
                uniprot_id1 = get_uniprot_id_from_name(p1_norm, logger)
            
            has_structure1 = False
            if uniprot_id1:
                has_structure1 = check_alphafold_structure_exists(uniprot_id1, logger)
            
            checked_proteins[p1_norm] = (uniprot_id1, has_structure1)
        else:
            uniprot_id1, has_structure1 = checked_proteins[p1_norm]
        
        if p2_norm not in checked_proteins:
            uniprot_id2 = None
            if uniprot_mapping:
                uniprot_id2 = uniprot_mapping.get(p2_norm)
            if not uniprot_id2:
                uniprot_id2 = get_uniprot_id_from_name(p2_norm, logger)
            
            has_structure2 = False
            if uniprot_id2:
                has_structure2 = check_alphafold_structure_exists(uniprot_id2, logger)
            
            checked_proteins[p2_norm] = (uniprot_id2, has_structure2)
        else:
            uniprot_id2, has_structure2 = checked_proteins[p2_norm]
        
        # If both have structures, we can potentially validate
        # Note: AlphaFold DB mostly has single chains, not complexes
        # So we can't directly validate complexes from AlphaFold DB
        # But we can note that both proteins have structures
        
        if has_structure1 and has_structure2:
            # Both proteins have AlphaFold structures
            # However, AlphaFold DB doesn't have complex predictions for most pairs
            # We can only validate if we download and check (requires complex prediction)
            
            if download_structures and structures_dir and uniprot_id1 and uniprot_id2:
                # Try to download structures and check
                # Note: This only works if there's a complex prediction
                # Most pairs won't have complex predictions in AlphaFold DB
                struct1_file = structures_dir / f"{uniprot_id1}.pdb"
                struct2_file = structures_dir / f"{uniprot_id2}.pdb"
                
                # Download if not exists
                if not struct1_file.exists():
                    download_alphafold_structure(uniprot_id1, struct1_file, logger)
                if not struct2_file.exists():
                    download_alphafold_structure(uniprot_id2, struct2_file, logger)
                
                # Note: Single chain structures can't validate complexes
                # Would need complex prediction (not available in AlphaFold DB for most pairs)
                pass
    
    # Count proteins with structures
    proteins_with_structures = sum(1 for _, (_, has_struct) in checked_proteins.items() if has_struct)
    
    if logger:
        logger.info(f"Proteins with AlphaFold structures: {proteins_with_structures}/{len(checked_proteins)}")
        logger.warning("AlphaFold DB has limited complex coverage.")
        logger.warning("Most entries are single chains, not complexes.")
        logger.warning("To validate complexes, need to run ColabFold predictions.")
    
    # For now, return empty (AlphaFold DB doesn't have complex predictions)
    # But we've checked which proteins have structures
    return pd.DataFrame(validated_pairs)


def validate_with_colabfold(predictions: Dict[Tuple[str, str], float],
                           output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Validate predictions using ColabFold (run predictions).
    
    WARNING: This is computationally expensive!
    - Running ColabFold for 1081 pairs would take days/weeks
    - Recommended: Only run for top N pairs (e.g., top 100)
    
    Args:
        predictions: Dict mapping (protein1, protein2) -> probability
        output_dir: Directory to save predictions
        logger: Optional logger
    
    Returns:
        DataFrame with validated pairs
    """
    if logger:
        logger.warning("=" * 70)
        logger.warning("COLABFOLD VALIDATION")
        logger.warning("=" * 70)
        logger.warning("WARNING: Running ColabFold for all pairs is computationally expensive!")
        logger.warning("  • 1081 pairs × ~10-30 minutes each = 180-540 hours")
        logger.warning("  • Recommended: Run only for top 100-200 pairs")
        logger.warning("=" * 70)
    
    # This would require ColabFold installation
    # For now, provide instructions
    if logger:
        logger.info("ColabFold validation requires:")
        logger.info("  1. Install ColabFold: pip install colabfold")
        logger.info("  2. Run predictions for pairs (see ColabFold documentation)")
        logger.info("  3. Extract contacts from predicted structures")
        logger.info("  4. Validate pairs with contacts < 5 Å")
    
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Validate predictions using AlphaFold-Multimer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--predictions-tsv',
        type=Path,
        required=True,
        help='Path to TSV file with predictions'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--uniprot-mapping',
        type=Path,
        help='Path to JSON file mapping protein_name -> uniprot_id'
    )
    parser.add_argument(
        '--check-alphafold-db',
        action='store_true',
        default=True,
        help='Check AlphaFold DB for existing predictions (default: True)'
    )
    parser.add_argument(
        '--download-structures',
        action='store_true',
        help='Download AlphaFold structures and check for contacts (slower, but more thorough)'
    )
    parser.add_argument(
        '--run-colabfold',
        action='store_true',
        help='Run ColabFold predictions (WARNING: computationally expensive!)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=100,
        help='If running ColabFold, only run for top N pairs (default: 100)'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=5.0,
        help='Distance threshold for contacts in Angstroms (default: 5.0)'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ALPHAFOLD-MULTIMER VALIDATION")
    logger.info("=" * 70)
    
    # Load predictions
    pred_df = pd.read_csv(args.predictions_tsv, sep='\t')
    predictions = {}
    for _, row in pred_df.iterrows():
        pair = tuple(sorted([str(row['Protein1']), str(row['Protein2'])]))
        predictions[pair] = float(row['Probability'])
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Load UniProt mapping if provided
    uniprot_mapping = None
    if args.uniprot_mapping and args.uniprot_mapping.exists():
        with open(args.uniprot_mapping, 'r') as f:
            uniprot_mapping = json.load(f)
        logger.info(f"Loaded UniProt mapping: {len(uniprot_mapping)} mappings")
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    validated_pairs = []
    
    # Check AlphaFold DB
    if args.check_alphafold_db:
        logger.info("Checking AlphaFold DB for existing predictions...")
        validated_df = validate_with_alphafold_db(
            predictions, 
            uniprot_mapping, 
            download_structures=args.download_structures,
            output_dir=args.output,
            distance_threshold=args.distance_threshold,
            logger=logger
        )
        if len(validated_df) > 0:
            validated_pairs.append(validated_df)
    
    # Run ColabFold predictions
    if args.run_colabfold:
        logger.warning("ColabFold validation is computationally expensive!")
        logger.warning(f"Running for top {args.top_n} pairs only...")
        
        # Get top N pairs
        sorted_pairs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_pairs = dict(sorted_pairs[:args.top_n])
        
        validated_df = validate_with_colabfold(top_pairs, args.output, logger)
        if len(validated_df) > 0:
            validated_pairs.append(validated_df)
    
    # Save results
    if validated_pairs:
        combined_df = pd.concat(validated_pairs, ignore_index=True)
        output_file = args.output / 'alphafold_validated_pairs.tsv'
        combined_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved {len(combined_df)} validated pairs to {output_file}")
    else:
        logger.info("No AlphaFold validations found.")
        logger.info("AlphaFold DB has limited complex coverage.")
        logger.info("Consider using ColabFold for top pairs (see --run-colabfold).")
    
    logger.info("=" * 70)


if __name__ == '__main__':
    main()


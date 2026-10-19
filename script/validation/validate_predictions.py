#!/usr/bin/env python3
"""
Validate predicted protein-protein interactions against PDB co-complexes and other databases.

This script:
1. Loads predicted interactions from merged MCMC chains
2. Extracts co-complex pairs from PDB structures
3. Loads known interactions from CORUM and other databases
4. Calculates validation metrics (precision, recall, F1, etc.)
5. Generates validation report and plots

Usage:
    python validate_predictions.py \
        --merged-chains merged/merged_chains.pkl \
        --model-data seed_13/0_model23_n_13_model_data.pkl \
        --pdb-dir data/pdb/ \
        --corum-file data/corum.tsv \
        --output validation_results/
"""

import argparse
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import warnings

import numpy as np
import pandas as pd

# Try to import BioPython for PDB parsing
try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    HAS_BIOPYTHON = True
    warnings.simplefilter('ignore', PDBConstructionWarning)
except ImportError:
    HAS_BIOPYTHON = False
    warnings.warn("BioPython not available. PDB parsing will be limited.")

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. ROC/PR curves will not be generated.")


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_merged_chains(merged_chains_file: Path, logger: logging.Logger) -> Dict:
    """Load merged chains from pickle file."""
    logger.info(f"Loading merged chains from {merged_chains_file}")
    with open(merged_chains_file, 'rb') as f:
        data = pkl.load(f)
    
    merged_samples = data.get('merged_samples', {})
    n_chains = data.get('n_chains', 1)
    seeds = data.get('seeds', [])
    burn_in = data.get('burn_in', 0)
    
    logger.info(f"Loaded merged chains: {n_chains} chains, seeds: {seeds}, burn-in: {burn_in}")
    
    return {
        'merged_samples': merged_samples,
        'n_chains': n_chains,
        'seeds': seeds,
        'burn_in': burn_in
    }


def load_predictions_from_tsv(tsv_file: Path, logger: logging.Logger) -> Dict[Tuple[str, str], float]:
    """
    Load predictions from TSV file (e.g., from analyze_top_pairs.py output).
    
    Args:
        tsv_file: Path to TSV file with columns: Protein1, Protein2, Probability
        logger: Logger instance
    
    Returns:
        Dictionary mapping (protein1, protein2) -> probability
    """
    logger.info(f"Loading predictions from TSV file: {tsv_file}")
    
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Check for required columns
    required_cols = ['Protein1', 'Protein2', 'Probability']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"TSV file must contain columns: {required_cols}. Found: {df.columns.tolist()}")
    
    predictions = {}
    for _, row in df.iterrows():
        p1 = str(row['Protein1']).strip()
        p2 = str(row['Protein2']).strip()
        prob = float(row['Probability'])
        
        # Normalize names
        p1 = normalize_protein_name(p1)
        p2 = normalize_protein_name(p2)
        
        pair = tuple(sorted([p1, p2]))
        predictions[pair] = prob
    
    logger.info(f"Loaded {len(predictions)} predictions from TSV file")
    logger.info(f"  Probability range: {min(predictions.values()):.4f} - {max(predictions.values()):.4f}")
    
    return predictions


def extract_all_predictions(merged_samples: Dict, model_data: Dict, logger: logging.Logger) -> Dict[Tuple[str, str], float]:
    """
    Extract edge probabilities for ALL pairs from merged chains.
    
    Args:
        merged_samples: Dictionary with 'z' key containing samples
        model_data: Model data dictionary with 'N', 'node_idx2name', etc.
        logger: Logger instance
    
    Returns:
        Dictionary mapping (protein1, protein2) -> probability
    """
    logger.info("Extracting ALL predictions from merged chains...")
    
    # Get z samples
    z_samples = merged_samples['z']
    
    # Convert to numpy if needed
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    # Handle shape: might be (n_samples, M) or (n_chains, n_samples, M)
    if len(z_samples.shape) == 3:
        # Multiple chains: flatten to (n_chains * n_samples, M)
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif len(z_samples.shape) == 2:
        # Single chain: (n_samples, M)
        pass
    else:
        raise ValueError(f"Unexpected z_samples shape: {z_samples.shape}")
    
    # Convert continuous z to probabilities using sigmoid
    try:
        from scipy.special import expit
        z_probs = expit(z_samples)
    except ImportError:
        z_probs = 1.0 / (1.0 + np.exp(-z_samples))
    
    # Compute mean probability across samples
    logger.info(f"Computing mean probabilities from {z_samples.shape[0]} samples")
    mean_z_probs = np.mean(z_probs, axis=0)
    
    # Get protein name mappings
    node_idx2name = model_data.get('node_idx2name', {})
    N = int(model_data['N'])
    M = model_data.get('M', N * (N - 1) // 2)
    edge_filter_map = model_data.get('edge_filter_map', None)
    
    # Map probabilities back to protein pairs
    predictions = {}
    
    if edge_filter_map is not None:
        # Handle filtered edges
        M_original = N * (N - 1) // 2
        triu_indices = np.triu_indices(N, k=1)
        z_full = np.zeros(M_original)
        z_full[edge_filter_map] = mean_z_probs
        
        for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
            if idx < len(z_full):
                prob = z_full[idx]
                if prob > 0:
                    name_i = node_idx2name.get(i, f"protein_{i}")
                    name_j = node_idx2name.get(j, f"protein_{j}")
                    # Remove _HUMAN suffix if present
                    name_i = name_i.replace('_HUMAN', '')
                    name_j = name_j.replace('_HUMAN', '')
                    pair = tuple(sorted([name_i, name_j]))
                    predictions[pair] = float(prob)
    else:
        # Standard case: no filtering
        triu_indices = np.triu_indices(N, k=1)
        for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
            if idx < len(mean_z_probs):
                prob = mean_z_probs[idx]
                if prob > 0:
                    name_i = node_idx2name.get(i, f"protein_{i}")
                    name_j = node_idx2name.get(j, f"protein_{j}")
                    # Remove _HUMAN suffix if present
                    name_i = name_i.replace('_HUMAN', '')
                    name_j = name_j.replace('_HUMAN', '')
                    pair = tuple(sorted([name_i, name_j]))
                    predictions[pair] = float(prob)
    
    logger.info(f"Extracted {len(predictions)} predicted interactions")
    logger.info(f"  Probability range: {min(predictions.values()):.4f} - {max(predictions.values()):.4f}")
    
    return predictions


def normalize_protein_name(name: str) -> str:
    """
    Normalize protein name for matching.
    
    Removes common suffixes and converts to uppercase.
    """
    name = str(name).strip().upper()
    # Remove common suffixes
    for suffix in ['_HUMAN', '_MOUSE', '_RAT', '_CHICKEN']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def get_uniprot_pdb_ids(uniprot_id: str, experimental_methods: Optional[List[str]] = None,
                       logger: Optional[logging.Logger] = None) -> Set[str]:
    """
    Get PDB IDs associated with a UniProt ID using UniProt API.
    
    Args:
        uniprot_id: UniProt ID (e.g., 'Q93034' for CUL5)
        experimental_methods: Optional list of methods to filter by
                             ['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR', etc.]
                             If None, returns all PDB IDs regardless of method
        logger: Optional logger
    
    Returns:
        Set of PDB IDs (e.g., {'1LDK', '2XYY'})
    """
    import urllib.request
    import json
    import time
    
    pdb_ids = set()
    
    try:
        # UniProt REST API
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            # Extract PDB IDs from cross-references
            if 'uniProtKBCrossReferences' in data:
                for xref in data['uniProtKBCrossReferences']:
                    if xref.get('database') == 'PDB':
                        pdb_id = xref.get('id', '').upper()
                        if pdb_id:
                            # If filtering by method, check the method
                            if experimental_methods:
                                # Get method from properties if available
                                properties = xref.get('properties', [])
                                method = None
                                for prop in properties:
                                    if prop.get('key') == 'Method':
                                        method = prop.get('value')
                                        break
                                
                                # If method matches or no method filter, add it
                                if method in experimental_methods or method is None:
                                    pdb_ids.add(pdb_id)
                            else:
                                # No filtering, add all PDB IDs
                                pdb_ids.add(pdb_id)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
    except urllib.error.HTTPError as e:
        if logger:
            logger.debug(f"UniProt API error for {uniprot_id}: {e.code}")
    except Exception as e:
        if logger:
            logger.debug(f"Error fetching PDB IDs for {uniprot_id}: {e}")
    
    return pdb_ids


def get_protein_uniprot_id(protein_name: str, logger: Optional[logging.Logger] = None) -> Optional[str]:
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
    import time
    
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


def load_pdb_co_complexes_from_uniprot(predictions: Dict[Tuple[str, str], float],
                                       uniprot_mapping: Optional[Dict[str, str]] = None,
                                       experimental_methods: Optional[List[str]] = None,
                                       logger: Optional[logging.Logger] = None) -> Set[Tuple[str, str]]:
    """
    Validate each predicted pair individually by checking if both proteins appear in same PDB structure.
    
    For each pair in predictions:
    1. Get UniProt ID for protein1
    2. Get PDB IDs for protein1 (optionally filtered by experimental method)
    3. Get UniProt ID for protein2
    4. Get PDB IDs for protein2 (optionally filtered by experimental method)
    5. If they share any PDB ID → co-complex (direct contact)
    
    Args:
        predictions: Dict mapping (protein1, protein2) -> probability
        uniprot_mapping: Optional dict mapping protein_name -> uniprot_id
        experimental_methods: Optional list of PDB experimental methods to include
                             ['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR']
                             If None, includes all methods
        logger: Optional logger
    
    Returns:
        Set of (protein1, protein2) tuples that are co-complex
    """
    if logger:
        method_str = ", ".join(experimental_methods) if experimental_methods else "all methods"
        logger.info(f"Validating pairs individually using UniProt API (PDB methods: {method_str})...")
        logger.info(f"Checking {len(predictions)} predicted pairs for PDB co-complex evidence...")
    
    co_complex_pairs = set()
    checked_pairs = {}  # Cache for protein -> (uniprot_id, pdb_ids)
    
    # Process each predicted pair
    total_pairs = len(predictions)
    for idx, ((p1, p2), prob) in enumerate(predictions.items(), 1):
        if logger and idx % 100 == 0:
            logger.info(f"  Progress: {idx}/{total_pairs} pairs checked...")
        
        # Normalize protein names
        p1_norm = normalize_protein_name(p1)
        p2_norm = normalize_protein_name(p2)
        
        # Skip if same protein
        if p1_norm == p2_norm:
            continue
        
        # Get UniProt ID and PDB IDs for protein1
        if p1_norm not in checked_pairs:
            uniprot_id1 = None
            if uniprot_mapping:
                uniprot_id1 = uniprot_mapping.get(p1_norm)
            if not uniprot_id1:
                uniprot_id1 = get_protein_uniprot_id(p1_norm, logger)
            
            pdb_ids1 = set()
            if uniprot_id1:
                pdb_ids1 = get_uniprot_pdb_ids(uniprot_id1, experimental_methods, logger)
            
            checked_pairs[p1_norm] = (uniprot_id1, pdb_ids1)
        else:
            uniprot_id1, pdb_ids1 = checked_pairs[p1_norm]
        
        # Get UniProt ID and PDB IDs for protein2
        if p2_norm not in checked_pairs:
            uniprot_id2 = None
            if uniprot_mapping:
                uniprot_id2 = uniprot_mapping.get(p2_norm)
            if not uniprot_id2:
                uniprot_id2 = get_protein_uniprot_id(p2_norm, logger)
            
            pdb_ids2 = set()
            if uniprot_id2:
                pdb_ids2 = get_uniprot_pdb_ids(uniprot_id2, experimental_methods, logger)
            
            checked_pairs[p2_norm] = (uniprot_id2, pdb_ids2)
        else:
            uniprot_id2, pdb_ids2 = checked_pairs[p2_norm]
        
        # Check if they share any PDB ID (co-complex)
        if pdb_ids1 and pdb_ids2:
            shared_pdbs = pdb_ids1 & pdb_ids2  # Set intersection
            if shared_pdbs:
                pair = tuple(sorted([p1_norm, p2_norm]))
                co_complex_pairs.add(pair)
                if logger:
                    logger.debug(f"  ✓ Co-complex: {p1_norm} - {p2_norm} (shared PDBs: {', '.join(sorted(shared_pdbs))})")
    
    if logger:
        logger.info(f"Found {len(co_complex_pairs)}/{len(predictions)} pairs with PDB co-complex evidence")
        logger.info(f"  Proteins with UniProt IDs: {sum(1 for _, (uid, _) in checked_pairs.items() if uid)}/{len(checked_pairs)}")
        logger.info(f"  Proteins with PDB structures: {sum(1 for _, (_, pdbs) in checked_pairs.items() if pdbs)}/{len(checked_pairs)}")
    
    return co_complex_pairs


def load_pdb_co_complexes(pdb_dir: Path, protein_mapping: Optional[Dict[str, str]] = None, 
                          chain_to_protein: Optional[Dict[str, str]] = None,
                          logger: Optional[logging.Logger] = None) -> Set[Tuple[str, str]]:
    """
    Extract co-complex pairs from PDB structures.
    
    If two proteins appear in the same PDB structure, they are considered
    to be in a co-complex. This function extracts all chain pairs from each
    PDB file and maps them to protein names.
    
    Args:
        pdb_dir: Directory containing PDB files
        protein_mapping: Optional dict mapping PDB names to your protein names
        chain_to_protein: Optional dict mapping (pdb_id, chain_id) -> protein_name
                         Example: {('1LDK', 'A'): 'CUL5', ('1LDK', 'B'): 'RBX2'}
        logger: Optional logger
    
    Returns:
        Set of (protein1, protein2) tuples (normalized names)
    """
    if not HAS_BIOPYTHON:
        if logger:
            logger.warning("BioPython not available. Cannot parse PDB files.")
        return set()
    
    if not pdb_dir.exists():
        if logger:
            logger.warning(f"PDB directory not found: {pdb_dir}")
        return set()
    
    co_complex_pairs = set()
    pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.pdb.gz")) + \
                list(pdb_dir.glob("*.cif")) + list(pdb_dir.glob("*.cif.gz"))
    
    if logger:
        logger.info(f"Found {len(pdb_files)} PDB/mmCIF files in {pdb_dir}")
    
    parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True) if HAS_BIOPYTHON else None
    
    for pdb_file in pdb_files:
        try:
            # Get PDB ID from filename
            pdb_id = pdb_file.stem.replace('.pdb', '').replace('.cif', '').upper()
            
            # Determine file type
            if pdb_file.suffix == '.cif' or (pdb_file.suffix == '.gz' and pdb_file.stem.endswith('.cif')):
                if cif_parser:
                    structure = cif_parser.get_structure('complex', str(pdb_file))
                else:
                    continue
            else:
                structure = parser.get_structure('complex', str(pdb_file))
            
            # Extract all chains (proteins) in this structure
            chains = []
            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    protein_name = None
                    
                    # First, try chain_to_protein mapping (most reliable)
                    if chain_to_protein:
                        key = (pdb_id, chain_id)
                        protein_name = chain_to_protein.get(key)
                    
                    # If not found, try to get from header/annotation
                    if not protein_name:
                        try:
                            # Check if there's a compound name
                            if hasattr(structure.header, 'compound'):
                                for comp in structure.header['compound']:
                                    if chain_id in comp.get('chain', []):
                                        protein_name = comp.get('molecule', [None])[0]
                                        break
                        except:
                            pass
                    
                    # If still not found, try to get from mmCIF data
                    if not protein_name and hasattr(structure, 'header'):
                        try:
                            # Try to get entity name from mmCIF
                            if hasattr(structure.header, 'name'):
                                protein_name = structure.header.get('name', [None])[0]
                        except:
                            pass
                    
                    # If no name found, use chain ID as fallback
                    if not protein_name:
                        protein_name = f"{pdb_id}_{chain_id}"
                    
                    # Normalize name
                    protein_name = normalize_protein_name(protein_name)
                    
                    # Apply mapping if provided
                    if protein_mapping:
                        protein_name = protein_mapping.get(protein_name, protein_name)
                    
                    chains.append(protein_name)
            
            # Create all pairs within this complex
            chains = list(set(chains))  # Remove duplicates
            if len(chains) > 1:
                for i, chain1 in enumerate(chains):
                    for chain2 in chains[i+1:]:
                        pair = tuple(sorted([chain1, chain2]))
                        co_complex_pairs.add(pair)
                
                if logger:
                    logger.debug(f"  {pdb_file.name}: {len(chains)} chains, {len(chains)*(len(chains)-1)//2} pairs")
        
        except Exception as e:
            if logger:
                logger.warning(f"Error parsing {pdb_file.name}: {e}")
            continue
    
    if logger:
        logger.info(f"Extracted {len(co_complex_pairs)} co-complex pairs from PDB structures")
    
    return co_complex_pairs


def load_pdb_from_rcsb(pdb_ids: List[str], output_dir: Path, logger: Optional[logging.Logger] = None) -> Path:
    """
    Download PDB structures from RCSB PDB.
    
    Args:
        pdb_ids: List of PDB IDs (e.g., ['1LDK', '2XYY'])
        output_dir: Directory to save PDB files
        logger: Optional logger
    
    Returns:
        Path to directory containing downloaded files
    """
    import urllib.request
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info(f"Downloading {len(pdb_ids)} PDB structures...")
    
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.upper().strip()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = output_dir / f"{pdb_id}.pdb"
        
        if output_file.exists():
            if logger:
                logger.debug(f"  {pdb_id} already exists, skipping")
            continue
        
        try:
            urllib.request.urlretrieve(url, output_file)
            if logger:
                logger.info(f"  Downloaded {pdb_id}")
        except Exception as e:
            if logger:
                logger.warning(f"  Failed to download {pdb_id}: {e}")
    
    return output_dir


def download_corum(output_file: Path, logger: Optional[logging.Logger] = None) -> Path:
    """
    Download CORUM database from official website.
    
    Args:
        output_file: Path to save CORUM file
        logger: Optional logger
    
    Returns:
        Path to downloaded file
    """
    import urllib.request
    import ssl
    import zipfile
    
    if output_file.exists():
        if logger:
            logger.info(f"CORUM file already exists: {output_file}")
        return output_file
    
    if logger:
        logger.info("Downloading CORUM database...")
    
    # CORUM download URL (all complexes)
    url = "https://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip"
    
    try:
        zip_file = output_file.parent / "corum_temp.zip"
        
        # Create SSL context that doesn't verify certificates (for CORUM download)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download using urlopen with SSL context
        if logger:
            logger.info(f"  Downloading from {url}...")
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(zip_file, 'wb') as out_file:
                out_file.write(response.read())
        
        # Check if it's actually a zip file
        try:
            # Extract
            if logger:
                logger.info(f"  Extracting to {output_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Find the txt file inside
                files = zip_ref.namelist()
                txt_file = [f for f in files if f.endswith('.txt')][0]
                zip_ref.extract(txt_file, output_file.parent)
                
                # Rename to desired output name
                extracted_file = output_file.parent / txt_file
                if extracted_file != output_file:
                    extracted_file.rename(output_file)
            
            # Clean up zip file
            zip_file.unlink()
        except zipfile.BadZipFile:
            # If not a zip file, check if it's already a text file
            if logger:
                logger.warning(f"  Downloaded file is not a zip. Checking if it's already text...")
            # Check if it's HTML (error page) or text
            with open(zip_file, 'rb') as f:
                content = f.read(1024)
                if content.startswith(b'<!') or b'<html' in content.lower():
                    if logger:
                        logger.error(f"  Server returned HTML instead of zip file. URL may have changed.")
                    zip_file.unlink()
                    raise ValueError("CORUM download returned HTML instead of zip file. Please download manually.")
                elif b'\t' in content or b'Complex' in content:
                    # Looks like TSV file directly
                    if logger:
                        logger.info(f"  Downloaded file appears to be TSV format. Using directly...")
                    zip_file.rename(output_file)
                else:
                    # Unknown format
                    if logger:
                        logger.error(f"  Unknown file format downloaded.")
                    zip_file.unlink()
                    raise ValueError("Unknown file format downloaded from CORUM.")
        
        if logger:
            logger.info(f"  CORUM database downloaded: {output_file}")
        
        return output_file
    
    except Exception as e:
        if logger:
            logger.error(f"Error downloading CORUM: {e}")
        raise


def load_corum_interactions(corum_file: Path, protein_mapping: Optional[Dict[str, str]] = None,
                            logger: Optional[logging.Logger] = None) -> Set[Tuple[str, str]]:
    """
    Load known interactions from CORUM database.
    
    Args:
        corum_file: Path to CORUM TSV file
        protein_mapping: Optional dict mapping CORUM names to your protein names
        logger: Optional logger
    
    Returns:
        Set of (protein1, protein2) tuples
    """
    if not corum_file.exists():
        if logger:
            logger.warning(f"CORUM file not found: {corum_file}")
        return set()
    
    interactions = set()
    
    try:
        # CORUM format: tab-separated
        # Try different possible column names
        df = pd.read_csv(corum_file, sep='\t', low_memory=False)
        
        # Look for subunit column (could be 'subunits', 'Subunits', 'Subunit name(s)', etc.)
        subunit_col = None
        for col in df.columns:
            if 'subunit' in col.lower() or 'protein' in col.lower():
                subunit_col = col
                break
        
        if subunit_col is None:
            if logger:
                logger.warning(f"Could not find subunit column in CORUM file. Available columns: {df.columns.tolist()}")
            return set()
        
        for _, row in df.iterrows():
            subunits_str = str(row[subunit_col])
            if pd.isna(subunits_str) or subunits_str == 'nan':
                continue
            
            # Split by semicolon or comma
            subunits = []
            for sep in [';', ',', '|']:
                if sep in subunits_str:
                    subunits = [s.strip() for s in subunits_str.split(sep)]
                    break
            
            if not subunits:
                subunits = [subunits_str.strip()]
            
            # Normalize and map
            subunits_normalized = []
            for sub in subunits:
                sub = normalize_protein_name(sub)
                if protein_mapping:
                    sub = protein_mapping.get(sub, sub)
                subunits_normalized.append(sub)
            
            # Create all pairs within complex
            subunits_normalized = list(set(subunits_normalized))  # Remove duplicates
            for i, sub1 in enumerate(subunits_normalized):
                for sub2 in subunits_normalized[i+1:]:
                    pair = tuple(sorted([sub1, sub2]))
                    interactions.add(pair)
        
        if logger:
            logger.info(f"Loaded {len(interactions)} interactions from CORUM")
    
    except Exception as e:
        if logger:
            logger.error(f"Error loading CORUM: {e}")
    
    return interactions


def calculate_metrics(predictions: Dict[Tuple[str, str], float], 
                    known_interactions: Set[Tuple[str, str]],
                    threshold: float = 0.5) -> Dict:
    """
    Calculate precision, recall, F1, etc. at given threshold.
    
    Args:
        predictions: Dict mapping (protein1, protein2) -> probability
        known_interactions: Set of known (protein1, protein2) tuples
        threshold: Probability threshold for predictions
    
    Returns:
        Dictionary with metrics
    """
    # Normalize prediction pairs
    predicted_pairs = set()
    for (p1, p2), prob in predictions.items():
        if prob >= threshold:
            pair = tuple(sorted([normalize_protein_name(p1), normalize_protein_name(p2)]))
            predicted_pairs.add(pair)
    
    # Normalize known interactions
    known_normalized = set()
    for pair in known_interactions:
        if isinstance(pair, tuple) and len(pair) == 2:
            p1, p2 = pair
            pair_norm = tuple(sorted([normalize_protein_name(str(p1)), normalize_protein_name(str(p2))]))
            known_normalized.add(pair_norm)
    
    # Calculate confusion matrix
    TP = len(predicted_pairs & known_normalized)
    FP = len(predicted_pairs - known_normalized)
    FN = len(known_normalized - predicted_pairs)
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_predicted': len(predicted_pairs),
        'n_known': len(known_normalized),
        'n_overlap': TP
    }


def calculate_all_metrics(predictions: Dict[Tuple[str, str], float],
                          known_interactions: Set[Tuple[str, str]],
                          thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> pd.DataFrame:
    """
    Calculate metrics at multiple thresholds.
    
    Returns:
        DataFrame with metrics at each threshold
    """
    results = []
    for threshold in thresholds:
        metrics = calculate_metrics(predictions, known_interactions, threshold)
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_validation_curves(predictions: Dict[Tuple[str, str], float],
                           known_interactions: Set[Tuple[str, str]],
                           output_file: Path,
                           logger: Optional[logging.Logger] = None):
    """
    Generate ROC and Precision-Recall curves.
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        if logger:
            logger.warning("matplotlib or scikit-learn not available. Skipping curves.")
        return
    
    # Prepare data
    y_true = []
    y_scores = []
    
    # Get all unique pairs
    all_pairs = set(predictions.keys())
    for pair in known_interactions:
        if isinstance(pair, tuple) and len(pair) == 2:
            p1, p2 = pair
            pair_norm = tuple(sorted([normalize_protein_name(str(p1)), normalize_protein_name(str(p2))]))
            all_pairs.add(pair_norm)
    
    for pair in all_pairs:
        # Normalize pair
        if isinstance(pair, tuple) and len(pair) == 2:
            p1, p2 = pair
            pair_norm = tuple(sorted([normalize_protein_name(str(p1)), normalize_protein_name(str(p2))]))
        else:
            continue
        
        # Check if known
        is_known = pair_norm in known_interactions or any(
            tuple(sorted([normalize_protein_name(str(p1)), normalize_protein_name(str(p2))])) == pair_norm
            for (p1, p2) in known_interactions
        )
        
        y_true.append(1 if is_known else 0)
        y_scores.append(predictions.get(pair, 0.0))
    
    if len(y_true) == 0:
        if logger:
            logger.warning("No data for validation curves")
        return
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})', color='#2E86AB')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # PR
    ax2.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})', color='#E63946')
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Saved validation curves: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate predicted protein-protein interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--predictions-tsv',
        type=Path,
        help='Path to TSV file with predictions (columns: Protein1, Protein2, Probability). '
             'Alternative to --merged-chains/--model-data'
    )
    parser.add_argument(
        '--merged-chains',
        type=Path,
        help='Path to merged_chains.pkl file (required if --predictions-tsv not provided)'
    )
    parser.add_argument(
        '--model-data',
        type=Path,
        help='Path to model_data.pkl file (required if --predictions-tsv not provided). '
             'Note: model_data is identical across all seeds, so any seed\'s model_data works.'
    )
    parser.add_argument(
        '--auto-find-model-data',
        action='store_true',
        help='Automatically find model_data.pkl from merged directory or seed directories'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for validation results'
    )
    parser.add_argument(
        '--pdb-dir',
        type=Path,
        help='Directory containing PDB files (legacy - not used if using UniProt API)'
    )
    parser.add_argument(
        '--use-uniprot',
        action='store_true',
        default=True,
        help='Use UniProt API to find co-complex pairs (default: True)'
    )
    parser.add_argument(
        '--pdb-methods',
        type=str,
        nargs='+',
        choices=['X-RAY', 'CRYO-EM', 'NMR', 'ALL'],
        default=['ALL'],
        help='PDB experimental methods to include: X-RAY (crystallography), CRYO-EM (electron microscopy), NMR (solution NMR), ALL (all methods). Default: ALL'
    )
    parser.add_argument(
        '--uniprot-mapping',
        type=Path,
        help='Path to JSON file mapping protein_name -> uniprot_id'
    )
    parser.add_argument(
        '--corum-file',
        type=Path,
        help='Path to CORUM database file (will download if not exists)'
    )
    parser.add_argument(
        '--download-corum',
        action='store_true',
        help='Download CORUM database if not provided'
    )
    parser.add_argument(
        '--protein-mapping',
        type=Path,
        help='Path to JSON file mapping protein names (PDB/CORUM -> your names)'
    )
    parser.add_argument(
        '--chain-mapping',
        type=Path,
        help='Path to JSON file mapping (pdb_id, chain_id) -> protein_name. '
             'Format: {"1LDK": {"A": "CUL5", "B": "RBX2"}}'
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help='Probability thresholds for metrics (default: 0.5 0.6 0.7 0.8 0.9)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("VALIDATION OF PREDICTED INTERACTIONS")
    logger.info("=" * 70)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load protein mapping if provided
    protein_mapping = None
    if args.protein_mapping and args.protein_mapping.exists():
        with open(args.protein_mapping, 'r') as f:
            protein_mapping = json.load(f)
        logger.info(f"Loaded protein mapping from {args.protein_mapping}")
    
    # Load UniProt mapping if provided
    uniprot_mapping = None
    if args.uniprot_mapping and args.uniprot_mapping.exists():
        with open(args.uniprot_mapping, 'r') as f:
            uniprot_mapping = json.load(f)
        logger.info(f"Loaded UniProt mapping from {args.uniprot_mapping} ({len(uniprot_mapping)} mappings)")
    
    # Auto-find model_data if requested
    model_data_path = args.model_data
    if args.auto_find_model_data and not model_data_path:
        # Try to find model_data automatically
        if args.merged_chains:
            # Look in merged directory
            merged_dir = args.merged_chains.parent
            model_data_candidates = [
                merged_dir / 'model_data.pkl',
                merged_dir.parent / 'seed_13' / '0_model23_n_13_model_data.pkl',
                merged_dir.parent / 'seed_42' / '0_model23_n_42_model_data.pkl',
                merged_dir.parent / 'seed_71' / '0_model23_n_71_model_data.pkl',
            ]
            for candidate in model_data_candidates:
                if candidate.exists():
                    model_data_path = candidate
                    logger.info(f"Auto-found model_data: {model_data_path}")
                    break
    
    # Load predictions - either from TSV file or from merged chains
    if args.predictions_tsv:
        # Load directly from TSV file (all pairs with probabilities)
        predictions = load_predictions_from_tsv(args.predictions_tsv, logger)
    else:
        # Extract from merged chains (requires both merged-chains and model-data)
        if not args.merged_chains:
            logger.error("--merged-chains is required when not using --predictions-tsv")
            return
        
        if not model_data_path:
            logger.error("--model-data is required when not using --predictions-tsv. "
                        "Or use --auto-find-model-data to search automatically.")
            return
        
        chains_data = load_merged_chains(args.merged_chains, logger)
        
        logger.info(f"Loading model data from {model_data_path}")
        logger.info("  Note: model_data is identical across all seeds (same proteins/structure)")
        with open(model_data_path, 'rb') as f:
            model_data = pkl.load(f)
        
        predictions = extract_all_predictions(chains_data['merged_samples'], model_data, logger)
    
    # Load known interactions
    known_interactions = set()
    
    # PDB co-complexes (using UniProt API)
    if args.use_uniprot:
        # Convert method names to UniProt format
        method_mapping = {
            'X-RAY': 'X-RAY DIFFRACTION',
            'CRYO-EM': 'ELECTRON MICROSCOPY',
            'NMR': 'SOLUTION NMR',
            'ALL': None  # None means all methods
        }
        
        experimental_methods = None
        if 'ALL' not in args.pdb_methods:
            experimental_methods = [method_mapping[m] for m in args.pdb_methods if m in method_mapping]
            if logger:
                logger.info(f"Filtering PDB structures by methods: {experimental_methods}")
        
        pdb_interactions = load_pdb_co_complexes_from_uniprot(
            predictions, uniprot_mapping, experimental_methods, logger
        )
        known_interactions.update(pdb_interactions)
        logger.info(f"Found {len(pdb_interactions)} PDB co-complex pairs via UniProt")
    elif args.pdb_dir:
        # Legacy: parse PDB files directly
        pdb_interactions = load_pdb_co_complexes(args.pdb_dir, protein_mapping, None, logger)
        known_interactions.update(pdb_interactions)
        logger.info(f"Loaded {len(pdb_interactions)} PDB co-complex pairs from files")
    
    # CORUM
    corum_file = args.corum_file
    if args.download_corum or (corum_file and not corum_file.exists()):
        # Download CORUM if needed
        if not corum_file:
            corum_file = args.output / 'corum_allComplexes.txt'
        corum_file = download_corum(corum_file, logger)
    
    if corum_file and corum_file.exists():
        corum_interactions = load_corum_interactions(corum_file, protein_mapping, logger)
        known_interactions.update(corum_interactions)
        logger.info(f"Loaded {len(corum_interactions)} CORUM interactions")
    
    if len(known_interactions) == 0:
        logger.warning("No known interactions loaded! Cannot perform validation.")
        logger.warning("Please provide --pdb-dir or --corum-file")
        return
    
    logger.info(f"Total known interactions: {len(known_interactions)}")
    
    # Identify validated pairs (TP) and save them
    logger.info("=" * 70)
    logger.info("IDENTIFYING VALIDATED PAIRS")
    logger.info("=" * 70)
    
    # Get validated pairs (those in predictions AND known interactions)
    validated_pairs = []
    for (p1, p2), prob in predictions.items():
        pair_norm = tuple(sorted([normalize_protein_name(p1), normalize_protein_name(p2)]))
        # Check if this pair is in known interactions
        is_known = False
        for known_pair in known_interactions:
            if isinstance(known_pair, tuple) and len(known_pair) == 2:
                known_norm = tuple(sorted([normalize_protein_name(str(known_pair[0])), 
                                           normalize_protein_name(str(known_pair[1]))]))
                if pair_norm == known_norm:
                    is_known = True
                    break
        
        if is_known:
            validated_pairs.append({
                'Protein1': p1,
                'Protein2': p2,
                'Probability': prob,
                'Rank': None  # Will fill from predictions
            })
    
    # Sort by probability (descending)
    validated_pairs.sort(key=lambda x: x['Probability'], reverse=True)
    
    # Add rank
    for i, pair in enumerate(validated_pairs, 1):
        pair['Rank'] = i
    
    # Save validated pairs
    if validated_pairs:
        validated_df = pd.DataFrame(validated_pairs)
        validated_file = args.output / 'validated_pairs.tsv'
        validated_df.to_csv(validated_file, sep='\t', index=False)
        logger.info(f"Found {len(validated_pairs)} validated pairs")
        logger.info(f"Saved to {validated_file}")
        logger.info("\nValidated pairs:")
        logger.info(validated_df[['Rank', 'Protein1', 'Protein2', 'Probability']].to_string(index=False))
    
    # Calculate metrics
    logger.info("=" * 70)
    logger.info("CALCULATING METRICS")
    logger.info("=" * 70)
    
    metrics_df = calculate_all_metrics(predictions, known_interactions, args.thresholds)
    
    # Save metrics
    metrics_file = args.output / 'validation_metrics.tsv'
    metrics_df.to_csv(metrics_file, sep='\t', index=False)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info("\nMetrics at different thresholds:")
    logger.info(metrics_df.to_string(index=False))
    
    # Find best threshold (max F1)
    best_idx = metrics_df['f1_score'].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'threshold']
    logger.info(f"\nBest F1-score: {metrics_df.loc[best_idx, 'f1_score']:.4f} at threshold {best_threshold:.2f}")
    
    # Generate plots
    if HAS_MATPLOTLIB and HAS_SKLEARN:
        curves_file = args.output / 'validation_curves.png'
        plot_validation_curves(predictions, known_interactions, curves_file, logger)
    
    # Save summary JSON
    summary = {
        'n_predictions': len(predictions),
        'n_known_interactions': len(known_interactions),
        'best_threshold': float(best_threshold),
        'best_f1': float(metrics_df.loc[best_idx, 'f1_score']),
        'best_precision': float(metrics_df.loc[best_idx, 'precision']),
        'best_recall': float(metrics_df.loc[best_idx, 'recall']),
        'metrics': metrics_df.to_dict('records')
    }
    
    summary_file = args.output / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSaved summary to {summary_file}")
    logger.info("=" * 70)
    logger.info("Validation complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()


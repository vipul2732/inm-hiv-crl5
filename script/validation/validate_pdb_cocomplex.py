#!/usr/bin/env python3
"""
PDB Co-Complex Validation using UniProt Mappings.

This script:
1. Maps gene names to UniProt IDs (via UniProt API or mapping files)
2. Extracts PDB IDs from UniProt entries
3. Identifies pairs with shared PDB structures (co-complex evidence)
4. Validates predicted interactions against PDB co-complexes

Usage:
    python validate_pdb_cocomplex.py --results-dir ../results --output validation_results/
    python validate_pdb_cocomplex.py --results-dir ../results --output validation_results/ --use-api
"""

import argparse
import json
import logging
import pickle as pkl
import sys
import time
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np

# Try to import requests for API calls
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    warnings.warn("requests not available. Install with: pip install requests")

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
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
    """Normalize protein name (uppercase, strip whitespace, remove _HUMAN suffix)."""
    name = name.strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def get_uniprot_id_from_gene_name(gene_name: str, logger: logging.Logger, cache: Dict[str, Optional[str]] = None) -> Optional[str]:
    """
    Map gene name to UniProt ID using UniProt REST API.
    
    Args:
        gene_name: Gene name (e.g., 'ELOB')
        logger: Logger instance
        cache: Optional cache dictionary to avoid repeated API calls
    
    Returns:
        UniProt ID or None if not found
    """
    if cache is not None and gene_name in cache:
        return cache[gene_name]
    
    if not HAS_REQUESTS:
        logger.warning("requests not available. Cannot query UniProt API.")
        return None
    
    try:
        # Query UniProt API for human proteins
        url = f"https://rest.uniprot.org/uniprotkb/search"
        params = {
            'query': f'gene_exact:{gene_name} AND organism_id:9606',
            'format': 'json',
            'fields': 'accession',
            'size': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get('results') and len(data['results']) > 0:
            uniprot_id = data['results'][0]['primaryAccession']
            if cache is not None:
                cache[gene_name] = uniprot_id
            return uniprot_id
        
        # If exact match fails, try without exact
        params['query'] = f'gene:{gene_name} AND organism_id:9606'
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('results') and len(data['results']) > 0:
            uniprot_id = data['results'][0]['primaryAccession']
            if cache is not None:
                cache[gene_name] = uniprot_id
            return uniprot_id
        
        if cache is not None:
            cache[gene_name] = None
        return None
        
    except Exception as e:
        logger.debug(f"Error querying UniProt for {gene_name}: {e}")
        if cache is not None:
            cache[gene_name] = None
        return None


def get_pdb_ids_from_uniprot(uniprot_id: str, logger: logging.Logger, cache: Dict[str, Set[str]] = None) -> Set[str]:
    """
    Get PDB structure IDs for a UniProt ID using UniProt REST API.
    
    Args:
        uniprot_id: UniProt accession ID
        logger: Logger instance
        cache: Optional cache dictionary
    
    Returns:
        Set of PDB IDs (e.g., {'1ABC', '2DEF'})
    """
    if cache is not None and uniprot_id in cache:
        return cache[uniprot_id]
    
    if not HAS_REQUESTS:
        logger.warning("requests not available. Cannot query UniProt API.")
        return set()
    
    try:
        # Use UniProt REST API with xref_pdb field
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        params = {
            'fields': 'xref_pdb'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        pdb_ids = set()
        
        # Extract PDB IDs from crossReferences field
        xrefs = data.get('uniProtKBCrossReferences', [])
        for xref in xrefs:
            if xref.get('database') == 'PDB':
                pdb_id = xref.get('id')
                if pdb_id:
                    pdb_ids.add(pdb_id)
        
        if cache is not None:
            cache[uniprot_id] = pdb_ids
        return pdb_ids
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # Protein not found
            if cache is not None:
                cache[uniprot_id] = set()
            return set()
        logger.debug(f"HTTP error getting PDB IDs for {uniprot_id}: {e}")
        if cache is not None:
            cache[uniprot_id] = set()
        return set()
    except Exception as e:
        logger.debug(f"Error getting PDB IDs for {uniprot_id}: {e}")
        if cache is not None:
            cache[uniprot_id] = set()
        return set()


def batch_map_proteins_to_uniprot(
    gene_names: Set[str],
    logger: logging.Logger,
    cache_file: Optional[Path] = None,
    delay: float = 0.1
) -> Dict[str, Optional[str]]:
    """
    Batch map gene names to UniProt IDs with caching.
    
    Args:
        gene_names: Set of gene names to map
        logger: Logger instance
        cache_file: Optional cache file path to save/load mappings
        delay: Delay between API calls (seconds)
    
    Returns:
        Dictionary mapping gene_name -> UniProt ID (or None)
    """
    # Load cache if exists
    cache = {}
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached UniProt mappings from {cache_file}")
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    
    # Map remaining proteins
    unmapped = {g for g in gene_names if g not in cache}
    if unmapped:
        logger.info(f"Mapping {len(unmapped)} proteins to UniProt IDs (using API)...")
        for i, gene_name in enumerate(unmapped, 1):
            if i % 100 == 0:
                logger.info(f"  Mapped {i}/{len(unmapped)} proteins...")
            uniprot_id = get_uniprot_id_from_gene_name(gene_name, logger, cache)
            if delay > 0:
                time.sleep(delay)  # Rate limiting
    
    # Save cache
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(cache)} UniProt mappings to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache file: {e}")
    
    return cache


def batch_get_pdb_ids(
    uniprot_ids: Set[str],
    logger: logging.Logger,
    cache_file: Optional[Path] = None,
    delay: float = 0.1
) -> Dict[str, Set[str]]:
    """
    Batch get PDB IDs for UniProt IDs with caching.
    
    Args:
        uniprot_ids: Set of UniProt IDs
        logger: Logger instance
        cache_file: Optional cache file path
        delay: Delay between API calls (seconds)
    
    Returns:
        Dictionary mapping UniProt ID -> Set of PDB IDs
    """
    # Load cache if exists
    cache = {}
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_json = json.load(f)
                cache = {k: set(v) if v else set() for k, v in cache_json.items()}
            logger.info(f"Loaded {len(cache)} cached PDB mappings from {cache_file}")
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    
    # Get PDB IDs for remaining proteins
    unmapped = {u for u in uniprot_ids if u not in cache}
    if unmapped:
        logger.info(f"Getting PDB IDs for {len(unmapped)} UniProt IDs (using API)...")
        for i, uniprot_id in enumerate(unmapped, 1):
            if i % 100 == 0:
                logger.info(f"  Processed {i}/{len(unmapped)} proteins...")
            pdb_ids = get_pdb_ids_from_uniprot(uniprot_id, logger, cache)
            if delay > 0:
                time.sleep(delay)  # Rate limiting
    
    # Save cache
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_json = {k: list(v) for k, v in cache.items()}
            with open(cache_file, 'w') as f:
                json.dump(cache_json, f, indent=2)
            logger.info(f"Saved {len(cache)} PDB mappings to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache file: {e}")
    
    return cache


def load_all_model_pairs(results_dir: Path, logger: logging.Logger) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Load ALL prey-prey pairs from merged chains (from compare_with_databases.py).
    """
    import jax.numpy as jnp
    
    # Add script directory to path
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    import model_variations as mv
    
    all_model_pairs = {}
    
    # Find merged_chains.pkl files
    for merged_file in results_dir.rglob("merged/merged_chains.pkl"):
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
            with open(merged_file, 'rb') as f:
                merged_data = pkl.load(f)
            
            merged_samples = merged_data['merged_samples']
            seeds = merged_data['seeds']
            model_data_file = merged_file.parent.parent / f"seed_{seeds[0]}/0_model23_n_{seeds[0]}_model_data.pkl"
            
            with open(model_data_file, 'rb') as f:
                model_data = pkl.load(f)
            
            node_idx2name = model_data.get('node_idx2name', {})
            if not node_idx2name:
                logger.warning(f"No node_idx2name in model_data for {dataset_name}")
                continue
            
            z_samples = merged_samples['z']
            
            if hasattr(z_samples, 'block_until_ready'):
                z_samples = np.array(z_samples)
            
            if len(z_samples.shape) == 3:
                z_samples = z_samples.reshape(-1, z_samples.shape[-1])
            elif len(z_samples.shape) == 2:
                pass
            else:
                z_samples = z_samples.reshape(1, -1)
            
            n_samples = z_samples.shape[0]
            M = z_samples.shape[1]
            N = int((1 + np.sqrt(1 + 8 * M)) / 2)
            
            z_probs = 1 / (1 + np.exp(-z_samples))
            mean_probs = np.mean(z_probs, axis=0)
            
            edge_to_pair = {}
            edge_idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    prot1 = normalize_protein_name(node_idx2name.get(i, f"protein_{i}"))
                    prot2 = normalize_protein_name(node_idx2name.get(j, f"protein_{j}"))
                    pair = tuple(sorted([prot1, prot2]))
                    edge_to_pair[edge_idx] = pair
                    edge_idx += 1
            
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
            logger.debug(traceback.format_exc())
            continue
    
    return all_model_pairs


def load_top100_pairs(results_dir: Path, logger: logging.Logger) -> Dict[str, Set[Tuple[str, str]]]:
    """Load top 100 predicted pairs."""
    predicted_pairs = {}
    
    for top_pairs_file in results_dir.rglob("top_100_pairs.tsv"):
        parts = top_pairs_file.parts
        dataset_name = None
        for i, part in enumerate(parts):
            if part.endswith("_config_multi_seeds"):
                dataset_name = part
                break
        
        if dataset_name is None:
            continue
        
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
            logger.info(f"Loaded {len(pairs)} top pairs from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error loading {top_pairs_file}: {e}")
            continue
    
    return predicted_pairs


def find_pdb_cocomplex_pairs(
    all_model_pairs: Dict[Tuple[str, str], float],
    uniprot_mapping: Dict[str, Optional[str]],
    pdb_mapping: Dict[str, Set[str]],
    logger: logging.Logger
) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Set[str]]]:
    """
    Find pairs with shared PDB structures (co-complex evidence).
    
    Args:
        all_model_pairs: Dictionary of all predicted pairs -> probability
        uniprot_mapping: Gene name -> UniProt ID mapping
        pdb_mapping: UniProt ID -> Set of PDB IDs mapping
        logger: Logger instance
    
    Returns:
        Tuple of (validated_pairs_set, pair_to_pdb_dict)
    """
    validated_pairs = set()
    pair_to_pdb = {}
    
    logger.info("Finding pairs with shared PDB structures (co-complex evidence)...")
    
    for pair, prob in all_model_pairs.items():
        prot1, prot2 = pair
        
        # Get UniProt IDs
        uniprot1 = uniprot_mapping.get(prot1)
        uniprot2 = uniprot_mapping.get(prot2)
        
        if not uniprot1 or not uniprot2:
            continue
        
        # Get PDB IDs
        pdb_ids1 = pdb_mapping.get(uniprot1, set())
        pdb_ids2 = pdb_mapping.get(uniprot2, set())
        
        # Find shared PDB structures
        shared_pdb = pdb_ids1 & pdb_ids2
        
        if shared_pdb:
            validated_pairs.add(pair)
            pair_to_pdb[pair] = shared_pdb
    
    logger.info(f"Found {len(validated_pairs)} pairs with shared PDB structures (co-complex evidence)")
    
    return validated_pairs, pair_to_pdb


def compute_pdb_validation_stats(
    all_model_pairs: Dict[Tuple[str, str], float],
    top100_pairs: Set[Tuple[str, str]],
    validated_pairs: Set[Tuple[str, str]],
    logger: logging.Logger
) -> Dict:
    """Compute validation statistics."""
    
    # Validated pairs in top 100
    top100_validated = top100_pairs & validated_pairs
    
    # Statistics
    stats = {
        'num_all_model_pairs': len(all_model_pairs),
        'num_top100_pairs': len(top100_pairs),
        'num_validated_pdb_cocomplex': len(validated_pairs),
        'num_top100_validated': len(top100_validated),
        'recall': len(top100_validated) / len(validated_pairs) if len(validated_pairs) > 0 else 0,
        'precision': len(top100_validated) / len(top100_pairs) if len(top100_pairs) > 0 else 0,
        'coverage': len(validated_pairs) / len(all_model_pairs) if len(all_model_pairs) > 0 else 0,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='PDB Co-Complex Validation using UniProt Mappings'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('../results'),
        help='Directory containing results with merged chains'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('pdb_validation_results'),
        help='Output directory for validation results'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path('.cache'),
        help='Directory for caching UniProt/PDB mappings'
    )
    parser.add_argument(
        '--api-delay',
        type=float,
        default=0.1,
        help='Delay between API calls (seconds) to respect rate limits'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Process only a specific dataset (e.g., cullin_mock_config_multi_seeds). If not specified, processes all datasets.'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    if not HAS_REQUESTS:
        logger.error("requests library not available. Install with: pip install requests")
        return
    
    logger.info("=" * 60)
    logger.info("PDB CO-COMPLEX VALIDATION")
    logger.info("=" * 60)
    
    # Load all model pairs
    logger.info("Loading all model predictions...")
    all_model_pairs_dict = load_all_model_pairs(args.results_dir, logger)
    
    if not all_model_pairs_dict:
        logger.error("No model pairs found!")
        return
    
    # Filter to specific dataset if requested
    if args.dataset:
        if args.dataset not in all_model_pairs_dict:
            logger.error(f"Dataset '{args.dataset}' not found in results directory!")
            logger.info(f"Available datasets: {list(all_model_pairs_dict.keys())}")
            return
        all_model_pairs_dict = {args.dataset: all_model_pairs_dict[args.dataset]}
        logger.info(f"Filtering to dataset: {args.dataset}")
    
    # Load top 100 pairs
    logger.info("Loading top 100 predictions...")
    top100_pairs_dict = load_top100_pairs(args.results_dir, logger)
    
    # Filter top100 pairs to match filtered datasets
    if args.dataset:
        top100_pairs_dict = {k: v for k, v in top100_pairs_dict.items() if k in all_model_pairs_dict}
    
    # Process each dataset
    all_stats = {}
    all_validated = {}
    
    for dataset_name, all_model_pairs in all_model_pairs_dict.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'='*60}")
        
        top100_pairs = top100_pairs_dict.get(dataset_name, set())
        
        # Collect all unique proteins
        all_proteins = set()
        for pair in all_model_pairs.keys():
            all_proteins.add(pair[0])
            all_proteins.add(pair[1])
        
        logger.info(f"Found {len(all_proteins)} unique proteins")
        
        # Step 1: Map gene names to UniProt IDs
        logger.info("\nStep 1: Mapping gene names to UniProt IDs...")
        uniprot_cache_file = args.cache_dir / 'uniprot_mapping.json'
        uniprot_mapping = batch_map_proteins_to_uniprot(
            all_proteins, logger, uniprot_cache_file, delay=args.api_delay
        )
        
        mapped_count = sum(1 for v in uniprot_mapping.values() if v is not None)
        logger.info(f"Mapped {mapped_count}/{len(all_proteins)} proteins to UniProt IDs")
        
        # Step 2: Get PDB IDs for UniProt IDs
        logger.info("\nStep 2: Getting PDB structure IDs from UniProt...")
        uniprot_ids = {v for v in uniprot_mapping.values() if v is not None}
        pdb_cache_file = args.cache_dir / 'pdb_mapping.json'
        pdb_mapping = batch_get_pdb_ids(
            uniprot_ids, logger, pdb_cache_file, delay=args.api_delay
        )
        
        proteins_with_pdb = sum(1 for v in pdb_mapping.values() if v)
        logger.info(f"Found PDB structures for {proteins_with_pdb}/{len(uniprot_ids)} proteins")
        
        # Step 3: Find pairs with shared PDB structures
        logger.info("\nStep 3: Finding pairs with shared PDB structures (co-complex evidence)...")
        validated_pairs, pair_to_pdb = find_pdb_cocomplex_pairs(
            all_model_pairs, uniprot_mapping, pdb_mapping, logger
        )
        
        # Step 4: Compute statistics
        logger.info("\nStep 4: Computing validation statistics...")
        stats = compute_pdb_validation_stats(
            all_model_pairs, top100_pairs, validated_pairs, logger
        )
        
        all_stats[dataset_name] = stats
        all_validated[dataset_name] = {
            'validated_pairs': validated_pairs,
            'pair_to_pdb': {str(k): list(v) for k, v in pair_to_pdb.items()}
        }
        
        # Print statistics
        logger.info(f"\nValidation Statistics for {dataset_name}:")
        logger.info(f"  All model pairs: {stats['num_all_model_pairs']}")
        logger.info(f"  Top 100 pairs: {stats['num_top100_pairs']}")
        logger.info(f"  PDB co-complex validated pairs: {stats['num_validated_pdb_cocomplex']}")
        logger.info(f"  Validated pairs in top 100: {stats['num_top100_validated']}")
        logger.info(f"  Coverage: {stats['coverage']*100:.2f}% of all model pairs have PDB co-complex evidence")
        logger.info(f"  Recall: {stats['recall']*100:.1f}% ({stats['num_top100_validated']}/{stats['num_validated_pdb_cocomplex']})")
        logger.info(f"  Precision: {stats['precision']*100:.1f}% ({stats['num_top100_validated']}/{stats['num_top100_pairs']})")
    
    # Save results
    logger.info(f"\n{'='*60}")
    logger.info("Saving results...")
    
    # Save statistics
    stats_file = args.output / 'pdb_validation_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    # Save validated pairs with PDB evidence
    for dataset_name, validated_data in all_validated.items():
        validated_pairs = validated_data['validated_pairs']
        pair_to_pdb = validated_data['pair_to_pdb']
        
        if validated_pairs:
            # Create DataFrame
            rows = []
            for pair in sorted(validated_pairs):
                prot1, prot2 = pair
                pdb_ids = pair_to_pdb.get(str(pair), [])
                rows.append({
                    'Protein1': prot1,
                    'Protein2': prot2,
                    'PDB_IDs': ';'.join(sorted(pdb_ids)),
                    'Num_PDB_Structures': len(pdb_ids)
                })
            
            df = pd.DataFrame(rows)
            df = df.sort_values('Num_PDB_Structures', ascending=False)
            
            output_file = args.output / f"{dataset_name}_pdb_validated_pairs.tsv"
            df.to_csv(output_file, sep='\t', index=False)
            logger.info(f"Saved {len(validated_pairs)} PDB-validated pairs to {output_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("PDB co-complex validation complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
GO Term and Compartment Analysis for Top 100 Predicted Pairs.

This script:
1. Loads top 100 predicted pairs for each dataset
2. Maps proteins to UniProt IDs
3. Retrieves GO terms (Molecular Function, Biological Process, Cellular Component) from UniProt
4. Analyzes GO term overlap/similarity between pairs
5. Calculates enrichment metrics and creates visualizations

Usage:
    python analyze_go_terms_top100.py --results-dir ../results --output go_analysis/ --dataset cullin_mock_config_multi_seeds
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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")


def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / "go_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def normalize_protein_name(name: str) -> str:
    """Normalize protein name for comparison."""
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def load_top100_pairs(results_dir: Path, dataset_name: str, logger: logging.Logger) -> List[Tuple[str, str, float]]:
    """
    Load top 100 pairs with probabilities for a dataset.
    
    Returns:
        List of (protein1, protein2, probability) tuples, sorted by probability descending
    """
    logger.info(f"Loading top 100 pairs for {dataset_name}")
    
    # Try to load from top_100_pairs.tsv
    top_pairs_file = results_dir / dataset_name / "networks" / "top_100_pairs.tsv"
    
    if top_pairs_file.exists():
        try:
            df = pd.read_csv(top_pairs_file, sep='\t')
            pairs = []
            for _, row in df.iterrows():
                p1 = normalize_protein_name(str(row['Protein1']))
                p2 = normalize_protein_name(str(row['Protein2']))
                prob = float(row.get('Probability', row.get('Posterior_Prob', 0.0)))
                pairs.append((p1, p2, prob))
            logger.info(f"  Loaded {len(pairs)} pairs from {top_pairs_file}")
            return sorted(pairs, key=lambda x: x[2], reverse=True)
        except Exception as e:
            logger.warning(f"Error loading from {top_pairs_file}: {e}")
    
    # Fallback: Load from merged chains and get top 100
    logger.info("  Falling back to loading from merged chains...")
    try:
        merged_file = results_dir / dataset_name / "merged" / "merged_chains.pkl"
        if not merged_file.exists():
            logger.error(f"Merged chains file not found: {merged_file}")
            return []
        
        import jax.numpy as jnp
        import jax.nn
        
        with open(merged_file, 'rb') as f:
            merged_data = pkl.load(f)
        
        merged_samples = merged_data['merged_samples']
        seeds = merged_data['seeds']
        model_data_file = results_dir / dataset_name / f"seed_{seeds[0]}" / f"0_model23_n_{seeds[0]}_model_data.pkl"
        
        if 'z' not in merged_samples:
            logger.error(f"No 'z' samples in merged data")
            return []
        
        z_samples = merged_samples['z']
        
        with open(model_data_file, 'rb') as f:
            model_data = pkl.load(f)
        
        node_idx2name = model_data.get('node_idx2name', {})
        if not node_idx2name:
            logger.error(f"No node_idx2name in model_data")
            return []
        
        if hasattr(z_samples, 'block_until_ready'):
            z_samples = np.array(z_samples)
        
        if len(z_samples.shape) == 3:
            z_samples = z_samples.reshape(-1, z_samples.shape[-1])
        
        mean_probs = jax.nn.sigmoid(jnp.array(z_samples)).mean(axis=0)
        
        N = len(node_idx2name)
        all_pairs = []
        
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                prob = float(mean_probs[k])
                p1 = normalize_protein_name(node_idx2name[i])
                p2 = normalize_protein_name(node_idx2name[j])
                all_pairs.append((p1, p2, prob))
                k += 1
        
        # Sort by probability and take top 100
        all_pairs.sort(key=lambda x: x[2], reverse=True)
        top100 = all_pairs[:100]
        logger.info(f"  Loaded top 100 pairs from merged chains")
        return top100
        
    except Exception as e:
        logger.error(f"Error loading pairs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def get_uniprot_id_from_gene_name(gene_name: str, logger: logging.Logger, cache: Dict[str, Optional[str]] = None) -> Optional[str]:
    """Map gene name to UniProt ID using UniProt REST API."""
    if cache is not None and gene_name in cache:
        return cache[gene_name]
    
    if not HAS_REQUESTS:
        logger.warning("requests not available. Cannot query UniProt API.")
        return None
    
    try:
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
        
        # Try without exact
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


def get_go_terms_from_uniprot(uniprot_id: str, logger: logging.Logger, cache: Dict[str, Dict[str, Dict[str, str]]] = None) -> Dict[str, Dict[str, str]]:
    """
    Get GO terms (MF, BP, CC) for a UniProt ID with both IDs and names.
    
    Returns:
        Dictionary with keys 'MF', 'BP', 'CC', each containing a dict mapping GO ID to GO name
    """
    if cache is not None and uniprot_id in cache:
        return cache[uniprot_id]
    
    if not HAS_REQUESTS:
        logger.warning("requests not available. Cannot query UniProt API.")
        return {'MF': {}, 'BP': {}, 'CC': {}}
    
    try:
        # Get full entry - GO terms are in crossReferences
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        go_terms = {'MF': {}, 'BP': {}, 'CC': {}}
        
        # Extract GO terms from crossReferences (standard location)
        xrefs = data.get('uniProtKBCrossReferences', [])
        for xref in xrefs:
            if xref.get('database') == 'GO':
                go_id = xref.get('id')
                if go_id:
                    # Get GO term type and name from properties
                    # Properties is now a list of dicts with 'key' and 'value'
                    properties = xref.get('properties', [])
                    go_term_type = ''
                    go_term_name = ''
                    
                    # Find GoTerm property and extract type from prefix (C:, P:, F:)
                    for prop in properties:
                        if prop.get('key') == 'GoTerm':
                            go_term_value = prop.get('value', '')
                            # Format is like "C:cell junction", "P:process", "F:function"
                            if go_term_value.startswith('C:'):
                                go_term_type = 'C'
                                go_term_name = go_term_value[2:]  # Remove "C:" prefix
                            elif go_term_value.startswith('P:'):
                                go_term_type = 'P'
                                go_term_name = go_term_value[2:]  # Remove "P:" prefix
                            elif go_term_value.startswith('F:'):
                                go_term_type = 'F'
                                go_term_name = go_term_value[2:]  # Remove "F:" prefix
                            break
                    
                    if go_term_type == 'F' and go_term_name:  # Molecular Function
                        go_terms['MF'][go_id] = go_term_name
                    elif go_term_type == 'P' and go_term_name:  # Biological Process
                        go_terms['BP'][go_id] = go_term_name
                    elif go_term_type == 'C' and go_term_name:  # Cellular Component
                        go_terms['CC'][go_id] = go_term_name
        
        # Also check comments for function annotations (may have GO terms)
        comments = data.get('comments', [])
        for comment in comments:
            if comment.get('commentType') == 'FUNCTION':
                # Some function comments have GO terms
                go_term_list = comment.get('goTerms', [])
                if go_term_list:
                    for go_term in go_term_list:
                        go_id = go_term.get('id')
                        go_name = go_term.get('name', '')
                        if go_id and go_name:
                            go_terms['MF'][go_id] = go_name
        
        if cache is not None:
            cache[uniprot_id] = go_terms
        return go_terms
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            if cache is not None:
                cache[uniprot_id] = {'MF': {}, 'BP': {}, 'CC': {}}
            return {'MF': {}, 'BP': {}, 'CC': {}}
        logger.debug(f"HTTP error getting GO terms for {uniprot_id}: {e}")
        if cache is not None:
            cache[uniprot_id] = {'MF': {}, 'BP': {}, 'CC': {}}
        return {'MF': {}, 'BP': {}, 'CC': {}}
    except Exception as e:
        logger.debug(f"Error getting GO terms for {uniprot_id}: {e}")
        if cache is not None:
            cache[uniprot_id] = {'MF': {}, 'BP': {}, 'CC': {}}
        return {'MF': {}, 'BP': {}, 'CC': {}}


def calculate_go_similarity(go1: Dict[str, Dict[str, str]], go2: Dict[str, Dict[str, str]]) -> Dict[str, float]:
    """
    Calculate GO term similarity between two proteins.
    
    Returns:
        Dictionary with 'MF', 'BP', 'CC' similarity scores (Jaccard index)
    """
    similarities = {}
    for go_type in ['MF', 'BP', 'CC']:
        dict1 = go1.get(go_type, {})
        dict2 = go2.get(go_type, {})
        
        set1 = set(dict1.keys())
        set2 = set(dict2.keys())
        
        if not set1 and not set2:
            similarities[go_type] = 0.0
        elif not set1 or not set2:
            similarities[go_type] = 0.0
        else:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            similarities[go_type] = intersection / union if union > 0 else 0.0
    
    return similarities


def batch_map_proteins_to_uniprot(
    gene_names: Set[str],
    logger: logging.Logger,
    cache_file: Optional[Path] = None,
    delay: float = 0.1
) -> Dict[str, Optional[str]]:
    """Batch map gene names to UniProt IDs with caching."""
    cache = {}
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} cached UniProt mappings")
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    
    unmapped = {g for g in gene_names if g not in cache}
    if unmapped:
        logger.info(f"Mapping {len(unmapped)} proteins to UniProt IDs...")
        for i, gene_name in enumerate(unmapped, 1):
            if i % 50 == 0:
                logger.info(f"  Mapped {i}/{len(unmapped)} proteins...")
            uniprot_id = get_uniprot_id_from_gene_name(gene_name, logger, cache)
            if delay > 0:
                time.sleep(delay)
    
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(cache)} UniProt mappings")
        except Exception as e:
            logger.warning(f"Could not save cache file: {e}")
    
    return cache


def batch_get_go_terms(
    uniprot_ids: Set[str],
    logger: logging.Logger,
    cache_file: Optional[Path] = None,
    delay: float = 0.1
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Batch get GO terms for UniProt IDs with caching."""
    cache = {}
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_json = json.load(f)
                # Cache already stores as dicts, no conversion needed
                cache = cache_json
            logger.info(f"Loaded {len(cache)} cached GO term mappings")
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    
    unmapped = {uid for uid in uniprot_ids if uid and uid not in cache}
    if unmapped:
        logger.info(f"Retrieving GO terms for {len(unmapped)} UniProt IDs...")
        for i, uniprot_id in enumerate(unmapped, 1):
            if i % 50 == 0:
                logger.info(f"  Retrieved {i}/{len(unmapped)} GO term sets...")
            go_terms = get_go_terms_from_uniprot(uniprot_id, logger, cache)
            if delay > 0:
                time.sleep(delay)
    
    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(cache)} GO term mappings")
        except Exception as e:
            logger.warning(f"Could not save cache file: {e}")
    
    return cache


def analyze_go_terms_for_pairs(
    top100_pairs: List[Tuple[str, str, float]],
    output_dir: Path,
    dataset_name: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Analyze GO terms for top 100 pairs."""
    logger.info(f"\nAnalyzing GO terms for {len(top100_pairs)} pairs...")
    
    # Collect all unique proteins
    all_proteins = set()
    for p1, p2, _ in top100_pairs:
        all_proteins.add(p1)
        all_proteins.add(p2)
    
    logger.info(f"  Found {len(all_proteins)} unique proteins")
    
    # Map proteins to UniProt IDs
    uniprot_cache_file = output_dir / f"{dataset_name}_uniprot_cache.json"
    gene_to_uniprot = batch_map_proteins_to_uniprot(all_proteins, logger, uniprot_cache_file)
    
    # Get UniProt IDs
    uniprot_ids = {uid for uid in gene_to_uniprot.values() if uid}
    logger.info(f"  Mapped {len(uniprot_ids)} proteins to UniProt IDs")
    
    # Get GO terms for all UniProt IDs
    go_cache_file = output_dir / f"{dataset_name}_go_cache.json"
    uniprot_to_go = batch_get_go_terms(uniprot_ids, logger, go_cache_file)
    
    # Analyze pairs
    results = []
    for p1, p2, prob in top100_pairs:
        uniprot1 = gene_to_uniprot.get(p1)
        uniprot2 = gene_to_uniprot.get(p2)
        
        go1 = uniprot_to_go.get(uniprot1, {'MF': {}, 'BP': {}, 'CC': {}}) if uniprot1 else {'MF': {}, 'BP': {}, 'CC': {}}
        go2 = uniprot_to_go.get(uniprot2, {'MF': {}, 'BP': {}, 'CC': {}}) if uniprot2 else {'MF': {}, 'BP': {}, 'CC': {}}
        
        similarities = calculate_go_similarity(go1, go2)
        
        # Count GO terms
        mf1_count = len(go1.get('MF', {}))
        mf2_count = len(go2.get('MF', {}))
        bp1_count = len(go1.get('BP', {}))
        bp2_count = len(go2.get('BP', {}))
        cc1_count = len(go1.get('CC', {}))
        cc2_count = len(go2.get('CC', {}))
        
        # Check if they share any GO terms
        mf1_ids = set(go1.get('MF', {}).keys())
        mf2_ids = set(go2.get('MF', {}).keys())
        bp1_ids = set(go1.get('BP', {}).keys())
        bp2_ids = set(go2.get('BP', {}).keys())
        cc1_ids = set(go1.get('CC', {}).keys())
        cc2_ids = set(go2.get('CC', {}).keys())
        
        shared_mf = len(mf1_ids & mf2_ids)
        shared_bp = len(bp1_ids & bp2_ids)
        shared_cc = len(cc1_ids & cc2_ids)
        
        # Get shared GO terms (IDs)
        shared_mf_terms = sorted(mf1_ids & mf2_ids)
        shared_bp_terms = sorted(bp1_ids & bp2_ids)
        shared_cc_terms = sorted(cc1_ids & cc2_ids)
        
        # Convert GO term dicts to semicolon-separated strings (IDs)
        mf1_terms = sorted(mf1_ids)
        mf2_terms = sorted(mf2_ids)
        bp1_terms = sorted(bp1_ids)
        bp2_terms = sorted(bp2_ids)
        cc1_terms = sorted(cc1_ids)
        cc2_terms = sorted(cc2_ids)
        
        results.append({
            'Protein1': p1,
            'Protein2': p2,
            'Probability': prob,
            'UniProt1': uniprot1 or 'N/A',
            'UniProt2': uniprot2 or 'N/A',
            'MF_Similarity': similarities['MF'],
            'BP_Similarity': similarities['BP'],
            'CC_Similarity': similarities['CC'],
            'MF1_Count': mf1_count,
            'MF2_Count': mf2_count,
            'BP1_Count': bp1_count,
            'BP2_Count': bp2_count,
            'CC1_Count': cc1_count,
            'CC2_Count': cc2_count,
            'Shared_MF': shared_mf,
            'Shared_BP': shared_bp,
            'Shared_CC': shared_cc,
            'Has_Shared_MF': shared_mf > 0,
            'Has_Shared_BP': shared_bp > 0,
            'Has_Shared_CC': shared_cc > 0,
            'MF1_GO_Terms': ';'.join(mf1_terms) if mf1_terms else '',
            'MF2_GO_Terms': ';'.join(mf2_terms) if mf2_terms else '',
            'BP1_GO_Terms': ';'.join(bp1_terms) if bp1_terms else '',
            'BP2_GO_Terms': ';'.join(bp2_terms) if bp2_terms else '',
            'CC1_GO_Terms': ';'.join(cc1_terms) if cc1_terms else '',
            'CC2_GO_Terms': ';'.join(cc2_terms) if cc2_terms else '',
            'Shared_MF_GO_Terms': ';'.join(shared_mf_terms) if shared_mf_terms else '',
            'Shared_BP_GO_Terms': ';'.join(shared_bp_terms) if shared_bp_terms else '',
            'Shared_CC_GO_Terms': ';'.join(shared_cc_terms) if shared_cc_terms else '',
        })
    
    df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / f"{dataset_name}_go_analysis.tsv"
    df.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Saved GO analysis results to {output_file}")
    
    # Generate summary with percentages
    generate_go_summary(df, uniprot_to_go, output_dir, dataset_name, logger)
    
    return df


def generate_go_summary(
    df: pd.DataFrame,
    uniprot_to_go: Dict[str, Dict[str, Dict[str, str]]],
    output_dir: Path,
    dataset_name: str,
    logger: logging.Logger
):
    """Generate summary statistics with GO term names and percentages."""
    from collections import Counter
    
    # Collect all GO terms from all pairs
    all_mf_terms = Counter()
    all_bp_terms = Counter()
    all_cc_terms = Counter()
    
    # Get UniProt to gene mapping from the dataframe
    uniprot_to_gene = {}
    for _, row in df.iterrows():
        if row['UniProt1'] != 'N/A':
            uniprot_to_gene[row['UniProt1']] = row['Protein1']
        if row['UniProt2'] != 'N/A':
            uniprot_to_gene[row['UniProt2']] = row['Protein2']
    
    # Count occurrences of each GO term across all pairs
    # Track which pairs have each GO term
    pairs_with_mf = {}  # GO term name -> set of pair indices
    pairs_with_bp = {}
    pairs_with_cc = {}
    
    for idx, row in df.iterrows():
        uniprot1 = row['UniProt1'] if row['UniProt1'] != 'N/A' else None
        uniprot2 = row['UniProt2'] if row['UniProt2'] != 'N/A' else None
        
        go1 = uniprot_to_go.get(uniprot1, {'MF': {}, 'BP': {}, 'CC': {}}) if uniprot1 else {'MF': {}, 'BP': {}, 'CC': {}}
        go2 = uniprot_to_go.get(uniprot2, {'MF': {}, 'BP': {}, 'CC': {}}) if uniprot2 else {'MF': {}, 'BP': {}, 'CC': {}}
        
        # Track which pairs have each GO term
        for go_id, go_name in go1.get('MF', {}).items():
            if go_name not in pairs_with_mf:
                pairs_with_mf[go_name] = set()
            pairs_with_mf[go_name].add(idx)
        for go_id, go_name in go2.get('MF', {}).items():
            if go_name not in pairs_with_mf:
                pairs_with_mf[go_name] = set()
            pairs_with_mf[go_name].add(idx)
        
        for go_id, go_name in go1.get('BP', {}).items():
            if go_name not in pairs_with_bp:
                pairs_with_bp[go_name] = set()
            pairs_with_bp[go_name].add(idx)
        for go_id, go_name in go2.get('BP', {}).items():
            if go_name not in pairs_with_bp:
                pairs_with_bp[go_name] = set()
            pairs_with_bp[go_name].add(idx)
        
        for go_id, go_name in go1.get('CC', {}).items():
            if go_name not in pairs_with_cc:
                pairs_with_cc[go_name] = set()
            pairs_with_cc[go_name].add(idx)
        for go_id, go_name in go2.get('CC', {}).items():
            if go_name not in pairs_with_cc:
                pairs_with_cc[go_name] = set()
            pairs_with_cc[go_name].add(idx)
    
    total_pairs = len(df)
    
    # Create summary dataframes
    summary_data = []
    
    # Molecular Function - count pairs, not proteins
    for go_name in sorted(pairs_with_mf.keys(), key=lambda x: len(pairs_with_mf[x]), reverse=True):
        pair_count = len(pairs_with_mf[go_name])
        percentage = (pair_count / total_pairs) * 100
        summary_data.append({
            'Category': 'Molecular Function',
            'GO Term Name': go_name,
            'Number of Pairs': pair_count,
            'Percentage of Pairs': f"{percentage:.2f}%"
        })
    
    # Biological Process
    for go_name in sorted(pairs_with_bp.keys(), key=lambda x: len(pairs_with_bp[x]), reverse=True):
        pair_count = len(pairs_with_bp[go_name])
        percentage = (pair_count / total_pairs) * 100
        summary_data.append({
            'Category': 'Biological Process',
            'GO Term Name': go_name,
            'Number of Pairs': pair_count,
            'Percentage of Pairs': f"{percentage:.2f}%"
        })
    
    # Cellular Component
    for go_name in sorted(pairs_with_cc.keys(), key=lambda x: len(pairs_with_cc[x]), reverse=True):
        pair_count = len(pairs_with_cc[go_name])
        percentage = (pair_count / total_pairs) * 100
        summary_data.append({
            'Category': 'Cellular Component',
            'GO Term Name': go_name,
            'Number of Pairs': pair_count,
            'Percentage of Pairs': f"{percentage:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = output_dir / f"{dataset_name}_go_summary.tsv"
    summary_df.to_csv(summary_file, sep='\t', index=False)
    logger.info(f"Saved GO term summary to {summary_file}")
    
    # Also create a more readable text summary
    summary_text_file = output_dir / f"{dataset_name}_go_summary.txt"
    with open(summary_text_file, 'w') as f:
        f.write(f"GO Term Analysis Summary for {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total pairs analyzed: {total_pairs}\n\n")
        
        f.write("Cellular Component (Compartment) Distribution:\n")
        f.write("-" * 80 + "\n")
        cc_terms = summary_df[summary_df['Category'] == 'Cellular Component']
        for _, row in cc_terms.iterrows():
            f.write(f"  {row['GO Term Name']:50s} {row['Percentage of Pairs']:>15s} ({row['Number of Pairs']} pairs)\n")
        
        f.write("\n\nMolecular Function Distribution:\n")
        f.write("-" * 80 + "\n")
        mf_terms = summary_df[summary_df['Category'] == 'Molecular Function']
        for _, row in mf_terms.iterrows():
            f.write(f"  {row['GO Term Name']:50s} {row['Percentage of Pairs']:>15s} ({row['Number of Pairs']} pairs)\n")
        
        f.write("\n\nBiological Process Distribution:\n")
        f.write("-" * 80 + "\n")
        bp_terms = summary_df[summary_df['Category'] == 'Biological Process']
        for _, row in bp_terms.iterrows():
            f.write(f"  {row['GO Term Name']:50s} {row['Percentage of Pairs']:>15s} ({row['Number of Pairs']} pairs)\n")
    
    logger.info(f"Saved readable GO term summary to {summary_text_file}")


def plot_go_analysis(df: pd.DataFrame, output_dir: Path, dataset_name: str, logger: logging.Logger):
    """Create visualizations for GO term analysis."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping plots.")
        return
    
    logger.info("Creating GO term analysis plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'GO Term Analysis: Top 100 Pairs\n{dataset_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: GO Similarity Distribution
    ax = axes[0, 0]
    ax.hist(df['MF_Similarity'], bins=20, alpha=0.6, label='Molecular Function', color='blue')
    ax.hist(df['BP_Similarity'], bins=20, alpha=0.6, label='Biological Process', color='green')
    ax.hist(df['CC_Similarity'], bins=20, alpha=0.6, label='Cellular Component', color='red')
    ax.set_xlabel('GO Term Similarity (Jaccard Index)', fontweight='bold')
    ax.set_ylabel('Number of Pairs', fontweight='bold')
    ax.set_title('Distribution of GO Term Similarities', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Shared GO Terms
    ax = axes[0, 1]
    shared_data = {
        'MF': df['Has_Shared_MF'].sum(),
        'BP': df['Has_Shared_BP'].sum(),
        'CC': df['Has_Shared_CC'].sum()
    }
    colors = ['blue', 'green', 'red']
    bars = ax.bar(shared_data.keys(), shared_data.values(), color=colors, alpha=0.7)
    ax.set_ylabel('Number of Pairs', fontweight='bold')
    ax.set_title('Pairs with Shared GO Terms', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, (key, val) in zip(bars, shared_data.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({val/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Similarity vs Probability
    ax = axes[1, 0]
    ax.scatter(df['Probability'], df['CC_Similarity'], alpha=0.6, label='CC', color='red', s=30)
    ax.scatter(df['Probability'], df['BP_Similarity'], alpha=0.6, label='BP', color='green', s=30)
    ax.scatter(df['Probability'], df['MF_Similarity'], alpha=0.6, label='MF', color='blue', s=30)
    ax.set_xlabel('Posterior Probability', fontweight='bold')
    ax.set_ylabel('GO Term Similarity', fontweight='bold')
    ax.set_title('GO Similarity vs Prediction Probability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate statistics
    stats_text = []
    stats_text.append("Summary Statistics:")
    stats_text.append("")
    stats_text.append(f"Total Pairs: {len(df)}")
    stats_text.append(f"Pairs with UniProt IDs: {(df['UniProt1'] != 'N/A').sum() & (df['UniProt2'] != 'N/A').sum()}")
    stats_text.append("")
    stats_text.append("GO Term Similarity (Mean):")
    stats_text.append(f"  MF: {df['MF_Similarity'].mean():.3f}")
    stats_text.append(f"  BP: {df['BP_Similarity'].mean():.3f}")
    stats_text.append(f"  CC: {df['CC_Similarity'].mean():.3f}")
    stats_text.append("")
    stats_text.append("Pairs with Shared GO Terms:")
    stats_text.append(f"  MF: {df['Has_Shared_MF'].sum()} ({df['Has_Shared_MF'].sum()/len(df)*100:.1f}%)")
    stats_text.append(f"  BP: {df['Has_Shared_BP'].sum()} ({df['Has_Shared_BP'].sum()/len(df)*100:.1f}%)")
    stats_text.append(f"  CC: {df['Has_Shared_CC'].sum()} ({df['Has_Shared_CC'].sum()/len(df)*100:.1f}%)")
    stats_text.append("")
    stats_text.append("Correlation (Probability vs Similarity):")
    mf_corr = df['Probability'].corr(df['MF_Similarity'])
    bp_corr = df['Probability'].corr(df['BP_Similarity'])
    cc_corr = df['Probability'].corr(df['CC_Similarity'])
    stats_text.append(f"  MF: {mf_corr:.3f}")
    stats_text.append(f"  BP: {bp_corr:.3f}")
    stats_text.append(f"  CC: {cc_corr:.3f}")
    
    ax.text(0.1, 0.5, '\n'.join(stats_text), fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_file = output_dir / f"{dataset_name}_go_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved GO analysis plot to {plot_file}")


def plot_go_summary(output_dir: Path, dataset_name: str, logger: logging.Logger):
    """Create plots from the GO summary TSV file."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping summary plots.")
        return
    
    summary_file = output_dir / f"{dataset_name}_go_summary.tsv"
    if not summary_file.exists():
        logger.warning(f"Summary file not found: {summary_file}")
        return
    
    logger.info("Creating GO term summary plots...")
    
    # Load summary data
    summary_df = pd.read_csv(summary_file, sep='\t')
    
    # Separate by category
    cc_df = summary_df[summary_df['Category'] == 'Cellular Component'].copy()
    mf_df = summary_df[summary_df['Category'] == 'Molecular Function'].copy()
    bp_df = summary_df[summary_df['Category'] == 'Biological Process'].copy()
    
    # Extract percentage values (remove % sign and convert to float)
    cc_df['Percentage'] = cc_df['Percentage of Pairs'].str.rstrip('%').astype(float)
    mf_df['Percentage'] = mf_df['Percentage of Pairs'].str.rstrip('%').astype(float)
    bp_df['Percentage'] = bp_df['Percentage of Pairs'].str.rstrip('%').astype(float)
    
    # Sort by percentage and take top N for each category
    top_n = 10
    cc_top = cc_df.nlargest(top_n, 'Percentage')
    mf_top = mf_df.nlargest(top_n, 'Percentage')
    bp_top = bp_df.nlargest(top_n, 'Percentage')
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create figure with subplots - optimized for publication
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.patch.set_facecolor('white')
    
    # Define professional color scheme
    colors = {
        'CC': '#2E86AB',  # Professional blue
        'MF': '#A23B72',  # Professional purple
        'BP': '#F18F01'   # Professional orange
    }
    
    # Plot 1: Cellular Component (Compartment)
    ax = axes[0]
    bars = ax.barh(range(len(cc_top)), cc_top['Percentage'].values, 
                   color=colors['CC'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(cc_top)))
    ax.set_yticklabels(cc_top['GO Term Name'].values, fontsize=13, fontweight='bold')
    ax.set_xlabel('Percentage of Pairs (%)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    # Set x-axis tick labels to bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.invert_yaxis()
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, cc_top['Percentage'].values)):
        width = bar.get_width()
        label_x = min(width + 2, 98)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold', fontsize=12, color='black')
    
    # Plot 2: Molecular Function
    ax = axes[1]
    bars = ax.barh(range(len(mf_top)), mf_top['Percentage'].values, 
                   color=colors['MF'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(mf_top)))
    ax.set_yticklabels(mf_top['GO Term Name'].values, fontsize=13, fontweight='bold')
    ax.set_xlabel('Percentage of Pairs (%)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    # Set x-axis tick labels to bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.invert_yaxis()
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, mf_top['Percentage'].values)):
        width = bar.get_width()
        label_x = min(width + 2, 98)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold', fontsize=12, color='black')
    
    # Plot 3: Biological Process
    ax = axes[2]
    bars = ax.barh(range(len(bp_top)), bp_top['Percentage'].values, 
                   color=colors['BP'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(bp_top)))
    ax.set_yticklabels(bp_top['GO Term Name'].values, fontsize=13, fontweight='bold')
    ax.set_xlabel('Percentage of Pairs (%)', fontweight='bold', fontsize=14, labelpad=10)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.4, axis='x', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    # Set x-axis tick labels to bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    ax.invert_yaxis()
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, bp_top['Percentage'].values)):
        width = bar.get_width()
        label_x = min(width + 2, 98)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold', fontsize=12, color='black')
    
    plt.tight_layout(pad=2.0)
    plot_file = output_dir / f"{dataset_name}_go_summary.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Saved GO summary plot to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze GO terms for top 100 predicted pairs')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing results (with top_100_pairs.tsv or merged chains)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., cullin_mock_config_multi_seeds)')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between API calls (seconds, default: 0.1)')
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info(f"Starting GO term analysis for {args.dataset}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load top 100 pairs
    top100_pairs = load_top100_pairs(results_dir, args.dataset, logger)
    
    if not top100_pairs:
        logger.error("No top 100 pairs found!")
        return
    
    # Analyze GO terms
    df = analyze_go_terms_for_pairs(top100_pairs, output_dir, args.dataset, logger)
    
    # Create plots
    plot_go_analysis(df, output_dir, args.dataset, logger)
    
    # Create summary plots
    plot_go_summary(output_dir, args.dataset, logger)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("GO TERM ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total pairs analyzed: {len(df)}")
    logger.info(f"Pairs with UniProt IDs: {((df['UniProt1'] != 'N/A') & (df['UniProt2'] != 'N/A')).sum()}")
    logger.info("")
    logger.info("Mean GO Term Similarity:")
    logger.info(f"  Molecular Function: {df['MF_Similarity'].mean():.3f}")
    logger.info(f"  Biological Process: {df['BP_Similarity'].mean():.3f}")
    logger.info(f"  Cellular Component: {df['CC_Similarity'].mean():.3f}")
    logger.info("")
    logger.info("Pairs with Shared GO Terms:")
    logger.info(f"  MF: {df['Has_Shared_MF'].sum()} ({df['Has_Shared_MF'].sum()/len(df)*100:.1f}%)")
    logger.info(f"  BP: {df['Has_Shared_BP'].sum()} ({df['Has_Shared_BP'].sum()/len(df)*100:.1f}%)")
    logger.info(f"  CC: {df['Has_Shared_CC'].sum()} ({df['Has_Shared_CC'].sum()/len(df)*100:.1f}%)")
    logger.info("")
    logger.info("Correlation (Probability vs Similarity):")
    logger.info(f"  MF: {df['Probability'].corr(df['MF_Similarity']):.3f}")
    logger.info(f"  BP: {df['Probability'].corr(df['BP_Similarity']):.3f}")
    logger.info(f"  CC: {df['Probability'].corr(df['CC_Similarity']):.3f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

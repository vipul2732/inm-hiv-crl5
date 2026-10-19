#!/usr/bin/env python3
"""
Validate all predicted pairs (not just top 100) against PDB co-complexes and databases.

This script:
1. Loads ALL predicted pairs with probabilities for a dataset
2. Checks each pair for:
   - PDB co-complex evidence (shared PDB structures)
   - Database evidence (CORUM, HuMAP2, IntAct, BioGRID)
3. Marks pairs as "validated" if they have ANY evidence
4. Computes recall: How many validated pairs appear in top 100?
5. Also computes precision: How many top 100 pairs are validated?

Usage:
    python validate_all_pairs_with_databases.py --results-dir ../results --database-dir ../database --pdb-results pdb_validation_test --dataset cullin_mock_config_multi_seeds --output validation_results/
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
import re

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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")


def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / "validation.log"
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


def load_all_pairs_with_probabilities(results_dir: Path, dataset_name: str, logger: logging.Logger) -> Dict[Tuple[str, str], float]:
    """
    Load ALL predicted pairs with probabilities for a dataset.
    
    Returns:
        Dictionary mapping (protein1, protein2) -> probability
    """
    logger.info(f"Loading all predicted pairs with probabilities for {dataset_name}")
    
    # Find the merged chains file
    dataset_dir = results_dir / dataset_name
    merged_file = dataset_dir / "merged" / "merged_chains.pkl"
    
    if not merged_file.exists():
        logger.error(f"Merged chains file not found: {merged_file}")
        return {}
    
    try:
        # Import JAX
        import jax.numpy as jnp
        import jax.nn
        
        # Add script directory to path
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        
        # Load merged chains
        with open(merged_file, 'rb') as f:
            merged_data = pkl.load(f)
        
        merged_samples = merged_data['merged_samples']
        seeds = merged_data['seeds']

        # Robustly locate model_data file.
        # For baseline runs this is typically:
        #   0_model23_n_<seed>_model_data.pkl
        # For ablated models it may be:
        #   0_model23_n_no_corr_<seed>_model_data.pkl, etc.
        seed_dir = dataset_dir / f"seed_{seeds[0]}"
        # Prefer the old baseline pattern if present
        baseline_candidate = seed_dir / f"0_model23_n_{seeds[0]}_model_data.pkl"
        if baseline_candidate.exists():
            model_data_file = baseline_candidate
        else:
            # Fallback: glob for any model23_n*model_data.pkl in the seed directory
            matches = list(seed_dir.glob("0_model23_n*_model_data.pkl"))
            if not matches:
                logger.error(
                    f"No model_data.pkl found in {seed_dir}. "
                    "Expected files like 0_model23_n_<seed>_model_data.pkl or "
                    "0_model23_n_<variant>_<seed>_model_data.pkl."
                )
                return {}
            # Use the first match (there should normally be exactly one)
            model_data_file = matches[0]
            logger.info(f"Using model_data file: {model_data_file}")
        
        if 'z' not in merged_samples:
            logger.error(f"No 'z' samples in merged data for {dataset_name}")
            return {}
        
        z_samples = merged_samples['z']
        
        # Load model data to get protein names
        with open(model_data_file, 'rb') as f:
            model_data = pkl.load(f)
        
        node_idx2name = model_data.get('node_idx2name', {})
        if not node_idx2name:
            logger.error(f"No node_idx2name in model_data for {dataset_name}")
            return {}
        
        # Convert to numpy if JAX array
        if hasattr(z_samples, 'block_until_ready'):
            z_samples = np.array(z_samples)
        
        # Handle multiple chains: shape might be (n_chains, n_samples, M) or (n_samples, M)
        if len(z_samples.shape) == 3:
            z_samples = z_samples.reshape(-1, z_samples.shape[-1])
        elif len(z_samples.shape) == 2:
            pass
        else:
            z_samples = z_samples.reshape(1, -1)
        
        # Calculate mean probability for each edge
        mean_probs = jax.nn.sigmoid(jnp.array(z_samples)).mean(axis=0)
        
        # Reconstruct full adjacency matrix to get all pairs
        N = len(node_idx2name)
        all_pairs = {}
        
        # Iterate through all possible unique pairs (upper triangle)
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                prob = float(mean_probs[k])
                p1 = normalize_protein_name(node_idx2name[i])
                p2 = normalize_protein_name(node_idx2name[j])
                all_pairs[tuple(sorted([p1, p2]))] = prob
                k += 1
        
        logger.info(f"  Loaded {len(all_pairs)} pairs with probabilities")
        return all_pairs
        
    except Exception as e:
        logger.error(f"Error loading pairs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def load_top100_pairs(results_dir: Path, dataset_name: str, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load top 100 pairs for a dataset."""
    top_pairs_file = results_dir / dataset_name / "networks" / "top_100_pairs.tsv"
    
    if not top_pairs_file.exists():
        logger.warning(f"Top 100 pairs file not found: {top_pairs_file}")
        return set()
    
    try:
        df = pd.read_csv(top_pairs_file, sep='\t')
        pairs = set()
        for _, row in df.iterrows():
            p1 = normalize_protein_name(str(row['Protein1']))
            p2 = normalize_protein_name(str(row['Protein2']))
            pairs.add(tuple(sorted([p1, p2])))
        logger.info(f"  Loaded {len(pairs)} top 100 pairs")
        return pairs
    except Exception as e:
        logger.error(f"Error loading top 100 pairs: {e}")
        return set()


def load_pdb_validated_pairs(pdb_results_dir: Path, dataset_name: str, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load PDB-validated pairs for a dataset."""
    pdb_file = pdb_results_dir / f"{dataset_name}_pdb_validated_pairs.tsv"
    
    if not pdb_file.exists():
        logger.warning(f"PDB validated pairs file not found: {pdb_file}")
        return set()
    
    try:
        df = pd.read_csv(pdb_file, sep='\t')
        pairs = set()
        for _, row in df.iterrows():
            p1 = normalize_protein_name(str(row['Protein1']))
            p2 = normalize_protein_name(str(row['Protein2']))
            pairs.add(tuple(sorted([p1, p2])))
        logger.info(f"  Loaded {len(pairs)} PDB-validated pairs")
        return pairs
    except Exception as e:
        logger.error(f"Error loading PDB-validated pairs: {e}")
        return set()


def load_validated_pairs_from_comparison_tsv(
    comparison_tsv: Path,
    logger: logging.Logger
) -> Set[Tuple[str, str]]:
    """
    Load validated pairs from a base-model database comparison TSV.

    The expected columns are:
      Protein1, Protein2, Has_Shared_PDB, In_Any_Database, ...

    To exactly match the base PU analysis (validation_statistics.txt),
    a pair is treated as a known positive iff:
      In_Any_Database == True
    """
    if not comparison_tsv.exists():
        logger.error(f"Comparison TSV not found: {comparison_tsv}")
        return set()

    try:
        df = pd.read_csv(comparison_tsv, sep='\t')
    except Exception as e:
        logger.error(f"Error reading comparison TSV {comparison_tsv}: {e}")
        return set()

    required_cols = {'Protein1', 'Protein2', 'Has_Shared_PDB', 'In_Any_Database'}
    if not required_cols.issubset(df.columns):
        logger.error(
            f"Comparison TSV {comparison_tsv} is missing required columns. "
            f"Found columns: {list(df.columns)}"
        )
        return set()

    positives = set()
    for _, row in df.iterrows():
        # Match base model definition: KnownPositive = In_Any_Database
        in_any_db = bool(row['In_Any_Database'])
        if not in_any_db:
            continue
        p1 = normalize_protein_name(str(row['Protein1']))
        p2 = normalize_protein_name(str(row['Protein2']))
        positives.add(tuple(sorted([p1, p2])))

    logger.info(
        f"Loaded {len(positives)} validated pairs from base comparison TSV: {comparison_tsv}"
    )
    return positives


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


def load_intact_pairs(intact_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load IntAct pairs."""
    logger.info(f"Loading IntAct from {intact_path}")
    pairs = set()
    
    try:
        df = pd.read_csv(intact_path, sep='\t', low_memory=False, comment='#', on_bad_lines='skip')
        logger.info(f"  Read {len(df)} IntAct interactions")
        
        alias_col_a = df.columns[4] if len(df.columns) > 4 else None
        alias_col_b = df.columns[5] if len(df.columns) > 5 else None
        
        def extract_gene_name(alias_str):
            if pd.isna(alias_str) or alias_str == '':
                return None
            alias_str = str(alias_str)
            matches = re.findall(r'\(gene name\)\|?([^|()]+)', alias_str)
            if matches:
                return normalize_protein_name(matches[0].split('(')[0].strip())
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
    
    return pairs


def load_biogrid_pairs(biogrid_path: Path, logger: logging.Logger) -> Set[Tuple[str, str]]:
    """Load BioGRID pairs."""
    logger.info(f"Loading BioGRID from {biogrid_path}")
    pairs = set()
    
    try:
        df = pd.read_csv(biogrid_path, sep='\t', low_memory=False, comment='#', on_bad_lines='skip')
        logger.info(f"  Read {len(df)} BioGRID interactions")
        
        if 'Taxid Interactor A' in df.columns:
            df = df[(df['Taxid Interactor A'] == 'taxid:9606') & (df['Taxid Interactor B'] == 'taxid:9606')]
            logger.info(f"  Filtered to {len(df)} human interactions")
        
        alias_col_a = df.columns[4] if len(df.columns) > 4 else None
        alias_col_b = df.columns[5] if len(df.columns) > 5 else None
        
        def extract_gene_name(alias_str):
            if pd.isna(alias_str) or alias_str == '':
                return None
            alias_str = str(alias_str)
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
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Validate all predicted pairs against PDB and databases'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        required=True,
        help='Directory containing results with merged chains'
    )
    parser.add_argument(
        '--database-dir',
        type=Path,
        required=True,
        help='Directory containing database files'
    )
    parser.add_argument(
        '--pdb-results',
        type=Path,
        required=True,
        help='Directory containing PDB validation results'
    )
    parser.add_argument(
        '--use-base-comparison',
        action='store_true',
        help=(
            'For ablation datasets, reuse the base model database comparison TSV '
            'to define the set of known positives (ensures identical positive set '
            'between base and ablations).'
        )
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
        default=Path('validation_results'),
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output)
    
    logger.info("=" * 60)
    logger.info("VALIDATE ALL PAIRS WITH PDB AND DATABASES")
    logger.info("=" * 60)
    
    # Step 1: Load all predicted pairs with probabilities
    logger.info(f"\nStep 1: Loading all predicted pairs for {args.dataset}...")
    all_pairs = load_all_pairs_with_probabilities(args.results_dir, args.dataset, logger)
    
    if not all_pairs:
        logger.error("No pairs found!")
        return
    
    # Step 2: Load top 100 pairs
    logger.info(f"\nStep 2: Loading top 100 pairs...")
    top100_pairs = load_top100_pairs(args.results_dir, args.dataset, logger)
    
    # Decide how to construct the set of known positives
    use_base_comparison = args.use_base_comparison and '_no_' in args.dataset
    pdb_validated_pairs: Set[Tuple[str, str]] = set()
    all_db_pairs: Set[Tuple[str, str]] = set()

    if use_base_comparison:
        # For ablation datasets, reuse the exact base-model comparison TSV
        logger.info("\nStep 3 (Ablation mode): Using base-model comparison TSV for known positives")

        # Infer which base dataset this ablation belongs to
        base_tag = None
        if 'mock' in args.dataset:
            base_tag = 'cullin_mock'
        elif 'wt' in args.dataset:
            base_tag = 'cullin_wt'
        elif 'vif' in args.dataset:
            base_tag = 'cullin_vif'

        if base_tag is None:
            logger.error(
                f"Could not infer base dataset (mock/wt/vif) from ablation name: {args.dataset}"
            )
            return

        # Assume comparison TSVs live alongside the original database directory
        comparison_dir = args.database_dir.parent / "validation_pdb_and_databases"
        comparison_tsv = comparison_dir / f"{base_tag}_database_comparison.tsv"

        base_validated_pairs = load_validated_pairs_from_comparison_tsv(
            comparison_tsv, logger
        )

        if not base_validated_pairs:
            logger.error(
                "No validated pairs loaded from base comparison TSV; "
                "cannot proceed with consistent ablation evaluation."
            )
            return

        # Restrict to pairs that actually appear in this model's prediction universe
        validated_pairs = base_validated_pairs & set(all_pairs.keys())
        logger.info(
            f"After intersecting with ablation predictions, "
            f"{len(validated_pairs)} validated pairs remain "
            f"(out of {len(base_validated_pairs)} from base model)."
        )
    else:
        # Original behavior: derive positives from PDB + databases
        # Step 3: Load PDB-validated pairs
        logger.info(f"\nStep 3: Loading PDB-validated pairs...")
        pdb_validated_pairs = load_pdb_validated_pairs(args.pdb_results, args.dataset, logger)
        
        # Step 4: Load all databases
        logger.info(f"\nStep 4: Loading databases...")
        
        # CORUM
        corum_file = args.database_dir / "corum_humanComplexes.tsv"
        corum_pairs = load_corum_pairs(corum_file, logger) if corum_file.exists() else set()
        
        # HuMAP2
        humap2_file = args.database_dir / "humap2.text"
        humap2_pairs = load_humap2_pairs(humap2_file, logger) if humap2_file.exists() else set()
        
        # IntAct
        intact_file = args.database_dir / "intact_human_interactions.txt"
        intact_pairs = load_intact_pairs(intact_file, logger) if intact_file.exists() else set()
        
        # BioGRID
        biogrid_file = args.database_dir / "BIOGRID-ALL-5.0.253.mitab.txt"
        biogrid_pairs = load_biogrid_pairs(biogrid_file, logger) if biogrid_file.exists() else set()
        
        # Combined database pairs (union of all databases)
        all_db_pairs = corum_pairs | humap2_pairs | intact_pairs | biogrid_pairs
        
        # Step 5: Mark pairs as validated
        logger.info(f"\nStep 5: Identifying validated pairs...")
        validated_pairs = set()
        
        # A pair is validated if:
        # 1. It has PDB co-complex evidence, OR
        # 2. It exists in any database
        validated_pairs = pdb_validated_pairs | (all_pairs.keys() & all_db_pairs)
    
    logger.info(f"  Total predicted pairs: {len(all_pairs)}")
    if use_base_comparison:
        logger.info(
            "  Known positives source: base-model database comparison TSV "
            f"({len(validated_pairs)} pairs after intersection)"
        )
    else:
        logger.info(f"  PDB-validated pairs: {len(pdb_validated_pairs)}")
        logger.info(f"  Database pairs (in our predictions): {len(all_pairs.keys() & all_db_pairs)}")
        logger.info(f"  Validated pairs (PDB OR database): {len(validated_pairs)}")
    
    # Step 6: Compute statistics (enrichment/retrieval focused, NOT treating non-annotated as negatives)
    logger.info(f"\nStep 6: Computing enrichment/retrieval statistics...")
    logger.info("  NOTE: We evaluate enrichment of known positives, NOT precision over all predictions.")
    logger.info("  Non-annotated edges are NOT treated as negatives (incomplete positive set).")
    
    # Sort all pairs by probability (descending) for Recall@K analysis
    sorted_pairs = sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)
    total_pairs = len(sorted_pairs)
    
    # Define K values for Recall@K / Precision@K analysis
    K_values = [10, 25, 50, 100, 200, 300, 500, 1000]
    
    # Compute Recall@K (hit-rate@K): how many known positives in top K
    recall_at_k = {}
    fold_enrichment_at_k = {}
    positive_set_fraction_at_k = {}  # Fraction of top K supported by reference positives
    precision_at_k = {}  # Lower-bound Precision@K in PU setting
    
    baseline_rate = len(validated_pairs) / total_pairs if total_pairs > 0 else 0
    
    for K in K_values:
        if K > total_pairs:
            continue
        
        # Get top K pairs
        top_k_pairs = {pair for pair, prob in sorted_pairs[:K]}
        validated_in_top_k = len(top_k_pairs & validated_pairs)
        
        # Recall@K: fraction of known positives recovered in top K
        recall_k = validated_in_top_k / len(validated_pairs) * 100 if validated_pairs else 0
        recall_at_k[K] = recall_k
        
        # Fold-enrichment@K: (positives in top K / K) / (positives in all / N)
        expected_random = K * baseline_rate
        fold_enrichment = validated_in_top_k / expected_random if expected_random > 0 else 0
        fold_enrichment_at_k[K] = fold_enrichment
        
        # Positive-set fraction: fraction of top K that are validated
        # (carefully worded: "database-supported fraction" not "precision")
        positive_set_fraction = validated_in_top_k / K * 100 if K > 0 else 0
        positive_set_fraction_at_k[K] = positive_set_fraction

        # Precision@K (lower bound in PU setting): validated_in_top_k / K
        precision_k = validated_in_top_k / K * 100 if K > 0 else 0
        precision_at_k[K] = precision_k
    
    # Also compute for top 100 (for compatibility)
    top100_pairs_set = set(top100_pairs)
    validated_in_top100_set = top100_pairs_set & validated_pairs
    validated_in_top100_count = len(validated_in_top100_set)
    recall_at_100 = validated_in_top100_count / len(validated_pairs) * 100 if validated_pairs else 0
    positive_set_fraction_100 = validated_in_top100_count / len(top100_pairs) * 100 if top100_pairs else 0
    
    stats = {
        'dataset': args.dataset,
        'total_predicted_pairs': len(all_pairs),
        'known_positives_total': len(validated_pairs),
        'baseline_rate': baseline_rate,
        'validated_pairs': {
            'pdb_only': len(pdb_validated_pairs - all_db_pairs),
            'database_only': len((all_pairs.keys() & all_db_pairs) - pdb_validated_pairs),
            'both': len(pdb_validated_pairs & all_db_pairs),
            'total': len(validated_pairs)
        },
        'recall_at_k': recall_at_k,
        'fold_enrichment_at_k': fold_enrichment_at_k,
        'positive_set_fraction_at_k': positive_set_fraction_at_k,
        'precision_at_k': precision_at_k,
        'top100': {
            'validated_count': validated_in_top100_count,
            'recall_percent': recall_at_100,
            'positive_set_fraction_percent': positive_set_fraction_100,
            'fold_enrichment': validated_in_top100_count / (100 * baseline_rate) if baseline_rate > 0 else 0
        }
    }
    
    logger.info(f"\n{'='*60}")
    logger.info("ENRICHMENT/RETRIEVAL RESULTS")
    logger.info("(Evaluating enrichment of known positives, not precision over all predictions)")
    logger.info(f"{'='*60}")
    logger.info(f"Total predicted pairs: {stats['total_predicted_pairs']}")
    logger.info(f"Known positives (validated): {stats['known_positives_total']}")
    logger.info(f"Baseline rate (positives/all): {baseline_rate*100:.3f}%")
    logger.info(f"\nValidated pairs breakdown:")
    logger.info(f"  PDB only: {stats['validated_pairs']['pdb_only']}")
    logger.info(f"  Database only: {stats['validated_pairs']['database_only']}")
    logger.info(f"  Both PDB and database: {stats['validated_pairs']['both']}")
    logger.info(f"  Total validated: {stats['validated_pairs']['total']}")
    logger.info(f"\nRecall@K (Hit-rate@K): Fraction of known positives recovered in top K predictions:")
    for K in sorted(K_values):
        if K in recall_at_k:
            logger.info(f"  Recall@{K}: {recall_at_k[K]:.1f}% ({stats['recall_at_k'].get(K, 0):.1f}% of known positives in top {K})")
    logger.info(f"\nFold-Enrichment@K (enrichment of known positives in top K vs. random):")
    for K in sorted(K_values):
        if K in fold_enrichment_at_k:
            logger.info(f"  FE@{K}: {fold_enrichment_at_k[K]:.2f}x ({fold_enrichment_at_k[K]:.2f}x enrichment over random)")
    logger.info(f"\nPositive-set fraction@K (fraction of top K supported by reference positives):")
    for K in sorted(K_values):
        if K in positive_set_fraction_at_k:
            logger.info(f"  Top {K}: {positive_set_fraction_at_k[K]:.1f}% database-supported")
    logger.info(f"\nPrecision@K (lower bound in PU setting):")
    for K in sorted(K_values):
        if K in precision_at_k:
            logger.info(f"  Precision@{K}: {precision_at_k[K]:.2f}% ({precision_at_k[K]:.2f}% validated in top {K})")
    logger.info(f"\nTop 100 summary:")
    logger.info(f"  Known positives in top 100: {validated_in_top100_count}/{len(validated_pairs)}")
    logger.info(f"  Recall@100: {recall_at_100:.1f}%")
    logger.info(f"  Fold-enrichment@100: {stats['top100']['fold_enrichment']:.2f}x")
    logger.info(f"  Positive-set fraction: {positive_set_fraction_100:.1f}% (fraction supported by reference positives)")
    
    # Step 7: Save detailed results
    logger.info(f"\nStep 7: Saving detailed results...")
    
    # Save statistics
    stats_file = args.output / f"{args.dataset}_validation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    # Save validated pairs with probabilities
    validated_pairs_data = []
    for pair in sorted(validated_pairs):
        prob = all_pairs.get(pair, 0.0)
        in_top100 = pair in top100_pairs
        has_pdb = pair in pdb_validated_pairs
        in_database = pair in all_db_pairs
        
        validated_pairs_data.append({
            'Protein1': pair[0],
            'Protein2': pair[1],
            'Probability': prob,
            'In_Top100': in_top100,
            'Has_PDB_Evidence': has_pdb,
            'In_Database': in_database
        })
    
    validated_df = pd.DataFrame(validated_pairs_data)
    validated_df = validated_df.sort_values('Probability', ascending=False)
    validated_file = args.output / f"{args.dataset}_validated_pairs.tsv"
    validated_df.to_csv(validated_file, sep='\t', index=False)
    logger.info(f"Saved {len(validated_pairs_data)} validated pairs to {validated_file}")
    
    # Save validated pairs in top 100
    top100_validated_data = []
    for pair in sorted(validated_in_top100_set):
        prob = all_pairs.get(pair, 0.0)
        has_pdb = pair in pdb_validated_pairs
        in_database = pair in all_db_pairs
        
        top100_validated_data.append({
            'Protein1': pair[0],
            'Protein2': pair[1],
            'Probability': prob,
            'Has_PDB_Evidence': has_pdb,
            'In_Database': in_database
        })
    
    top100_validated_df = pd.DataFrame(top100_validated_data)
    if not top100_validated_df.empty and 'Probability' in top100_validated_df.columns:
        top100_validated_df = top100_validated_df.sort_values('Probability', ascending=False)
    top100_validated_file = args.output / f"{args.dataset}_validated_in_top100.tsv"
    top100_validated_df.to_csv(top100_validated_file, sep='\t', index=False)
    logger.info(f"Saved {len(top100_validated_data)} validated pairs in top 100 to {top100_validated_file}")
    
    # Step 8: Enrichment analysis (top % analysis)
    logger.info(f"\nStep 8: Performing enrichment analysis...")
    enrichment_stats = calculate_enrichment(
        all_pairs, validated_pairs, args.output, args.dataset, logger
    )
    
    # Step 9: Create probability distribution plots
    if HAS_MATPLOTLIB:
        logger.info(f"\nStep 9: Creating probability distribution plots...")
        plot_probability_distribution(
            all_pairs, validated_pairs, top100_pairs, args.output, args.dataset, logger
        )
        # Create enrichment plot
        plot_enrichment_curve(
            all_pairs, validated_pairs, args.output, args.dataset, logger
        )
    else:
        logger.warning("matplotlib not available. Skipping plots.")
    
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION COMPLETE")
    logger.info(f"{'='*60}")


def calculate_enrichment(
    all_pairs: Dict[Tuple[str, str], float],
    validated_pairs: Set[Tuple[str, str]],
    output_dir: Path,
    dataset_name: str,
    logger: logging.Logger
) -> Dict:
    """
    Calculate enrichment at different top percentages (like virtual screening).
    
    Returns enrichment statistics at top 1%, 5%, 10%, 15%, 20%, 25%, 50%, 100%
    """
    logger.info("Calculating enrichment at different top percentages...")
    
    # Sort all pairs by probability (descending)
    sorted_pairs = sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)
    total_pairs = len(sorted_pairs)
    total_validated = len(validated_pairs)
    
    # Calculate baseline: random expectation
    baseline_rate = total_validated / total_pairs if total_pairs > 0 else 0
    
    # Percentages to analyze
    percentages = [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    enrichment_results = []
    
    for pct in percentages:
        # Calculate how many pairs in top X%
        n_top = int(total_pairs * pct / 100)
        top_pairs = {pair for pair, prob in sorted_pairs[:n_top]}
        
        # Count validated pairs in top X%
        validated_in_top = len(top_pairs & validated_pairs)
        
        # Expected number if random
        expected_random = n_top * baseline_rate
        
        # Enrichment factor
        enrichment_factor = validated_in_top / expected_random if expected_random > 0 else 0
        
        # Recall (of all validated, how many are in top X%?)
        recall = validated_in_top / total_validated * 100 if total_validated > 0 else 0
        
        # Precision (of top X%, how many are validated?)
        precision = validated_in_top / n_top * 100 if n_top > 0 else 0
        
        enrichment_results.append({
            'top_percent': pct,
            'n_pairs': n_top,
            'validated_in_top': validated_in_top,
            'expected_random': expected_random,
            'enrichment_factor': enrichment_factor,
            'recall_percent': recall,
            'precision_percent': precision
        })
        
        logger.info(f"  Top {pct:3d}%: {validated_in_top:3d}/{n_top:4d} validated "
                   f"(enrichment: {enrichment_factor:.2f}x, recall: {recall:.1f}%)")
    
    # Save enrichment table
    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_file = output_dir / f"{dataset_name}_enrichment_analysis.tsv"
    enrichment_df.to_csv(enrichment_file, sep='\t', index=False)
    logger.info(f"Saved enrichment analysis to {enrichment_file}")
    
    # Save as JSON too
    enrichment_json = output_dir / f"{dataset_name}_enrichment_analysis.json"
    with open(enrichment_json, 'w') as f:
        json.dump({
            'total_pairs': total_pairs,
            'total_validated': total_validated,
            'baseline_rate': baseline_rate,
            'enrichment_at_percentages': enrichment_results
        }, f, indent=2)
    
    return {
        'total_pairs': total_pairs,
        'total_validated': total_validated,
        'baseline_rate': baseline_rate,
        'enrichment_results': enrichment_results
    }


def plot_enrichment_curve(
    all_pairs: Dict[Tuple[str, str], float],
    validated_pairs: Set[Tuple[str, str]],
    output_dir: Path,
    dataset_name: str,
    logger: logging.Logger
):
    """Plot enrichment curve showing validated pairs at different top percentages."""
    
    # Sort all pairs by probability (descending)
    sorted_pairs = sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)
    total_pairs = len(sorted_pairs)
    total_validated = len(validated_pairs)
    
    # Calculate cumulative validated pairs at different percentages
    percentages = np.linspace(0, 100, 101)  # 0% to 100% in 1% increments
    cumulative_validated = []
    enrichment_factors = []
    recall_values = []
    
    baseline_rate = total_validated / total_pairs if total_pairs > 0 else 0
    
    for pct in percentages:
        n_top = int(total_pairs * pct / 100)
        top_pairs = {pair for pair, prob in sorted_pairs[:n_top]}
        validated_in_top = len(top_pairs & validated_pairs)
        cumulative_validated.append(validated_in_top)
        
        # Enrichment factor
        expected_random = n_top * baseline_rate
        ef = validated_in_top / expected_random if expected_random > 0 else 0
        enrichment_factors.append(ef)
        
        # Recall
        recall = validated_in_top / total_validated * 100 if total_validated > 0 else 0
        recall_values.append(recall)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Enrichment Analysis: {dataset_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Cumulative validated pairs (enrichment curve)
    ax1 = axes[0, 0]
    ax1.plot(percentages, cumulative_validated, 'b-', linewidth=2, label='Validated pairs')
    # Random line
    random_line = [total_validated * p / 100 for p in percentages]
    ax1.plot(percentages, random_line, 'r--', linewidth=1.5, label='Random expectation')
    ax1.set_xlabel('Top Percentage of Predictions (%)', fontsize=11)
    ax1.set_ylabel('Cumulative Validated Pairs', fontsize=11)
    ax1.set_title('Enrichment Curve: Cumulative Validated Pairs', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, max(cumulative_validated) * 1.1)
    
    # Plot 2: Enrichment factor
    ax2 = axes[0, 1]
    ax2.plot(percentages, enrichment_factors, 'g-', linewidth=2, label='Enrichment factor')
    ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Random (1.0x)')
    ax2.set_xlabel('Top Percentage of Predictions (%)', fontsize=11)
    ax2.set_ylabel('Enrichment Factor', fontsize=11)
    ax2.set_title('Enrichment Factor at Different Top Percentages', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, max(enrichment_factors) * 1.1)
    
    # Plot 3: Recall (how many validated pairs are recovered)
    ax3 = axes[1, 0]
    ax3.plot(percentages, recall_values, 'purple', linewidth=2, label='Recall (%)')
    ax3.set_xlabel('Top Percentage of Predictions (%)', fontsize=11)
    ax3.set_ylabel('Recall (%)', fontsize=11)
    ax3.set_title('Recall: Validated Pairs Recovered', fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    
    # Plot 4: Bar chart of key percentages
    ax4 = axes[1, 1]
    key_percentages = [1, 5, 10, 15, 20, 25, 50]
    key_enrichment = []
    for pct in key_percentages:
        idx = int(pct)
        if idx < len(enrichment_factors):
            key_enrichment.append(enrichment_factors[idx])
    
    bars = ax4.bar(range(len(key_percentages)), key_enrichment, color='steelblue', alpha=0.7)
    ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Random (1.0x)')
    ax4.set_xticks(range(len(key_percentages)))
    ax4.set_xticklabels([f'{p}%' for p in key_percentages], fontsize=9)
    ax4.set_ylabel('Enrichment Factor', fontsize=11)
    ax4.set_title('Enrichment Factor at Key Percentages', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, ef) in enumerate(zip(bars, key_enrichment)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{ef:.2f}x', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_file = output_dir / f"{dataset_name}_enrichment_curve.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved enrichment curve to {plot_file}")


def plot_probability_distribution(
    all_pairs: Dict[Tuple[str, str], float],
    validated_pairs: Set[Tuple[str, str]],
    top100_pairs: Set[Tuple[str, str]],
    output_dir: Path,
    dataset_name: str,
    logger: logging.Logger
):
    """Plot probability distribution of all pairs with validated pairs highlighted."""
    
    # Extract probabilities
    all_probs = list(all_pairs.values())
    validated_probs = [all_pairs[pair] for pair in validated_pairs if pair in all_pairs]
    top100_probs = [all_pairs[pair] for pair in top100_pairs if pair in all_pairs]
    validated_in_top100_probs = [
        all_pairs[pair] for pair in validated_pairs & top100_pairs if pair in all_pairs
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Probability Distribution: {dataset_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Histogram overlay
    ax1 = axes[0]
    n_bins = 50
    ax1.hist(all_probs, bins=n_bins, alpha=0.6, label=f'All pairs (n={len(all_probs)})', 
             color='lightblue', edgecolor='black', linewidth=0.5)
    ax1.hist(validated_probs, bins=n_bins, alpha=0.8, label=f'Validated pairs (n={len(validated_probs)})',
             color='orange', edgecolor='darkorange', linewidth=1)
    ax1.set_xlabel('Probability', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Probability Distribution: All Pairs vs Validated Pairs', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_xlim(0, 1)
    
    # Add vertical line for median
    median_all = np.median(all_probs)
    median_validated = np.median(validated_probs) if validated_probs else 0
    ax1.axvline(median_all, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median (all): {median_all:.3f}')
    ax1.axvline(median_validated, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median (validated): {median_validated:.3f}')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot 2: Density/KDE plot
    ax2 = axes[1]
    try:
        import seaborn as sns
        sns.histplot(all_probs, bins=n_bins, alpha=0.5, label=f'All pairs (n={len(all_probs)})',
                    color='lightblue', kde=True, ax=ax2, stat='density')
        sns.histplot(validated_probs, bins=n_bins, alpha=0.7, label=f'Validated pairs (n={len(validated_probs)})',
                    color='orange', kde=True, ax=ax2, stat='density')
    except ImportError:
        # Fallback to matplotlib if seaborn not available
        ax2.hist(all_probs, bins=n_bins, alpha=0.5, label=f'All pairs (n={len(all_probs)})',
                color='lightblue', density=True, edgecolor='black', linewidth=0.5)
        ax2.hist(validated_probs, bins=n_bins, alpha=0.7, label=f'Validated pairs (n={len(validated_probs)})',
                color='orange', density=True, edgecolor='darkorange', linewidth=1)
    
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Probability Density: All Pairs vs Validated Pairs', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plot_file = output_dir / f"{dataset_name}_probability_distribution.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved probability distribution plot to {plot_file}")
    
    # Create additional plot: Validated pairs in top 100
    if validated_in_top100_probs:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.hist(all_probs, bins=n_bins, alpha=0.5, label=f'All pairs (n={len(all_probs)})',
               color='lightblue', edgecolor='black', linewidth=0.5)
        ax.hist(validated_probs, bins=n_bins, alpha=0.6, label=f'All validated (n={len(validated_probs)})',
               color='orange', edgecolor='darkorange', linewidth=1)
        ax.hist(validated_in_top100_probs, bins=n_bins, alpha=0.9, 
               label=f'Validated in top 100 (n={len(validated_in_top100_probs)})',
               color='red', edgecolor='darkred', linewidth=1.5)
        
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Probability Distribution: Validated Pairs in Top 100\n{dataset_name}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plot_file2 = output_dir / f"{dataset_name}_validated_top100_distribution.png"
        plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validated top100 distribution plot to {plot_file2}")


if __name__ == '__main__':
    main()

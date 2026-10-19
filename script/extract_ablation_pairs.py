#!/usr/bin/env python3
"""
Extract all pairs with probabilities from merged chains for each ablation run.
Saves as TSV files: Protein1, Protein2, Probability
"""

import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    import jax.numpy as jnp
    import jax.nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("Warning: JAX not available, using numpy sigmoid")


def normalize_protein_name(name: str) -> str:
    """Normalize protein name for comparison."""
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def extract_pairs_from_merged_chains(merged_file: Path) -> pd.DataFrame:
    """
    Extract all pairs with probabilities from merged chains.
    
    Returns DataFrame with columns: Protein1, Protein2, Probability
    """
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged chains file not found: {merged_file}")
    
    # Load merged chains
    with open(merged_file, 'rb') as f:
        merged_data = pkl.load(f)
    
    merged_samples = merged_data['merged_samples']
    seeds = merged_data['seeds']
    
    # Find model_data file
    dataset_dir = merged_file.parent.parent
    seed_dir = dataset_dir / f"seed_{seeds[0]}"
    
    # Try to find model_data file (could be baseline or ablation naming)
    model_data_file = None
    baseline_candidate = seed_dir / f"0_model23_n_{seeds[0]}_model_data.pkl"
    if baseline_candidate.exists():
        model_data_file = baseline_candidate
    else:
        # Try ablation naming patterns
        for pattern in [
            f"0_model23_n_no_corr_{seeds[0]}_model_data.pkl",
            f"0_model23_n_no_saint_{seeds[0]}_model_data.pkl",
            f"0_model23_n_no_priors_{seeds[0]}_model_data.pkl",
        ]:
            candidate = seed_dir / pattern
            if candidate.exists():
                model_data_file = candidate
                break
        
        # Fallback: glob for any model23_n*model_data.pkl
        if model_data_file is None:
            matches = list(seed_dir.glob("0_model23_n*_model_data.pkl"))
            if matches:
                model_data_file = matches[0]
    
    if model_data_file is None or not model_data_file.exists():
        raise FileNotFoundError(f"Model data file not found in {seed_dir}")
    
    # Load model data to get protein names
    with open(model_data_file, 'rb') as f:
        model_data = pkl.load(f)
    
    node_idx2name = model_data.get('node_idx2name', {})
    if not node_idx2name:
        raise ValueError(f"No node_idx2name in model_data")
    
    # Get z samples
    if 'z' not in merged_samples:
        raise ValueError("No 'z' samples in merged data")
    
    z_samples = merged_samples['z']
    
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
    if HAS_JAX:
        mean_probs = jax.nn.sigmoid(jnp.array(z_samples)).mean(axis=0)
        mean_probs = np.array(mean_probs)
    else:
        # Fallback to numpy sigmoid
        mean_probs = 1.0 / (1.0 + np.exp(-np.clip(z_samples, -500, 500)))
        mean_probs = np.mean(mean_probs, axis=0)
    
    # Reconstruct all pairs
    N = len(node_idx2name)
    pairs_data = []
    
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(mean_probs[k])
            p1 = normalize_protein_name(node_idx2name[i])
            p2 = normalize_protein_name(node_idx2name[j])
            pairs_data.append({
                'Protein1': p1,
                'Protein2': p2,
                'Probability': prob
            })
            k += 1
    
    df = pd.DataFrame(pairs_data)
    # Sort by probability descending
    df = df.sort_values('Probability', ascending=False).reset_index(drop=True)
    
    return df


def main():
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "script" / "results"
    
    # Find all ablation merged chains
    ablation_patterns = ["*_no_corr", "*_no_saint", "*_no_priors"]
    merged_files = []
    
    for pattern in ablation_patterns:
        merged_files.extend(results_dir.glob(f"{pattern}/merged/merged_chains.pkl"))
    
    if not merged_files:
        print("No ablation merged chains found!")
        return
    
    print(f"Found {len(merged_files)} ablation merged chain files")
    
    for merged_file in sorted(merged_files):
        dataset_name = merged_file.parent.parent.name
        output_file = merged_file.parent.parent / f"{dataset_name}_all_pairs.tsv"
        
        print(f"\nProcessing: {dataset_name}")
        print(f"  Input: {merged_file}")
        
        try:
            df = extract_pairs_from_merged_chains(merged_file)
            df.to_csv(output_file, sep='\t', index=False)
            print(f"  Output: {output_file}")
            print(f"  Total pairs: {len(df)}")
            print(f"  Probability range: {df['Probability'].min():.4f} - {df['Probability'].max():.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

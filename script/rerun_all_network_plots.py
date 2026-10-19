#!/usr/bin/env python3
"""
Rerun network plot analysis for all datasets with enhanced publication-quality aesthetics.

This script runs analyze_top_pairs.py for all multi-seed result directories.
"""

import subprocess
import sys
from pathlib import Path

# Define all datasets to process
DATASETS = [
    {
        'name': 'Cullin Mock',
        'merged_chains': 'results/cullin_mock_config_multi_seeds/merged/merged_chains.pkl',
        'model_data': 'results/cullin_mock_config_multi_seeds/seed_13/0_model23_n_13_model_data.pkl',
        'output': 'results/cullin_mock_config_multi_seeds/merged'
    },
    {
        'name': 'Cullin VIF',
        'merged_chains': 'results/cullin_vif_config_multi_seeds/merged/merged_chains.pkl',
        'model_data': 'results/cullin_vif_config_multi_seeds/seed_13/0_model23_n_13_model_data.pkl',
        'output': 'results/cullin_vif_config_multi_seeds/merged'
    },
    {
        'name': 'Cullin Wildtype',
        'merged_chains': 'results/cullin_wt_config_multi_seeds/merged/merged_chains.pkl',
        'model_data': 'results/cullin_wt_config_multi_seeds/seed_13/0_model23_n_13_model_data.pkl',
        'output': 'results/cullin_wt_config_multi_seeds/merged'
    },
    {
        'name': 'P53 Contact Mutations',
        'merged_chains': 'results/p53_contact_mutations_only_config_multi_seeds/merged/merged_chains.pkl',
        'model_data': 'results/p53_contact_mutations_only_config_multi_seeds/seed_13/0_model23_n_13_model_data.pkl',
        'output': 'results/p53_contact_mutations_only_config_multi_seeds/merged'
    },
    {
        'name': 'P53 Serine Mutants',
        'merged_chains': 'results/Serine_mutants_only_config_multi_seeds/merged/merged_chains.pkl',
        'model_data': 'results/Serine_mutants_only_config_multi_seeds/seed_13/0_model23_n_13_model_data.pkl',
        'output': 'results/Serine_mutants_only_config_multi_seeds/merged'
    },
    {
        'name': 'P53 Wildtype',
        'merged_chains': 'results/WT_only_config_multi_seeds/merged/merged_chains.pkl',
        'model_data': 'results/WT_only_config_multi_seeds/seed_13/0_model23_n_13_model_data.pkl',
        'output': 'results/WT_only_config_multi_seeds/merged'
    }
]

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 80)
    print("RERUNNING NETWORK PLOT ANALYSIS FOR ALL DATASETS")
    print("=" * 80)
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Total datasets: {len(DATASETS)}\n")
    
    successful = []
    failed = []
    
    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing: {dataset['name']}")
        print("-" * 80)
        
        # Check if files exist
        merged_chains_path = project_root / dataset['merged_chains']
        model_data_path = project_root / dataset['model_data']
        output_dir = project_root / dataset['output']
        
        if not merged_chains_path.exists():
            print(f"  ❌ ERROR: Merged chains file not found: {merged_chains_path}")
            failed.append((dataset['name'], 'Merged chains file not found'))
            continue
        
        if not model_data_path.exists():
            print(f"  ❌ ERROR: Model data file not found: {model_data_path}")
            failed.append((dataset['name'], 'Model data file not found'))
            continue
        
        print(f"  ✓ Merged chains: {merged_chains_path}")
        print(f"  ✓ Model data: {model_data_path}")
        print(f"  ✓ Output directory: {output_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analyze_top_pairs.py
        cmd = [
            sys.executable,
            str(script_dir / 'analyze_top_pairs.py'),
            '--merged-chains', str(merged_chains_path),
            '--model-data', str(model_data_path),
            '--output', str(output_dir),
            '--top-n', '100'
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✓ SUCCESS: {dataset['name']}")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    print("  Output (last 5 lines):")
                    for line in lines[-5:]:
                        print(f"    {line}")
                else:
                    print("  Output:")
                    for line in lines:
                        print(f"    {line}")
            successful.append(dataset['name'])
        except subprocess.CalledProcessError as e:
            print(f"  ❌ FAILED: {dataset['name']}")
            print(f"  Exit code: {e.returncode}")
            if e.stdout:
                print(f"  STDOUT:\n{e.stdout}")
            if e.stderr:
                print(f"  STDERR:\n{e.stderr}")
            failed.append((dataset['name'], f"Exit code {e.returncode}"))
        except Exception as e:
            print(f"  ❌ ERROR: {dataset['name']}")
            print(f"  Exception: {e}")
            failed.append((dataset['name'], str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{len(DATASETS)}")
    for name in successful:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(DATASETS)}")
        for name, error in failed:
            print(f"  ❌ {name}: {error}")
    
    print("\n" + "=" * 80)
    print("All network plots have been regenerated with publication-quality aesthetics!")
    print("=" * 80)

if __name__ == '__main__':
    main()



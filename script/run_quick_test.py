#!/usr/bin/env python3
"""
Quick test run with minimal samples to see results fast.

Usage:
    python run_quick_test.py --config config.json
    python run_quick_test.py --data input/data.xlsx --sheets 0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Run quick test with minimal samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with existing config
  python run_quick_test.py --config ../input/cullin/config.json
  
  # Quick test with specific data file (uses first sheet only)
  python run_quick_test.py --data ../input/cullin/data.xlsx
        """
    )
    
    parser.add_argument('--config', type=Path, help='Path to config file')
    parser.add_argument('--data', type=Path, help='Input Excel file (will use first sheet only)')
    parser.add_argument('--sheets', type=int, nargs='+', default=[0], help='Sheet indices (default: [0])')
    parser.add_argument('--output', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    # If data file provided, create quick config
    if args.data:
        xlsx_path = Path(args.data)
        if not xlsx_path.is_absolute():
            xlsx_path = (script_dir / xlsx_path).resolve()
        
        if not xlsx_path.exists():
            print(f"ERROR: Data file not found: {xlsx_path}")
            sys.exit(1)
        
        # Create quick test config
        import json
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = script_dir.parent / 'results' / 'quick_test'
        
        config = {
            'data': {
                'input_file': str(xlsx_path),
                'prey_column': 'PreyGene',
                'filter': 'all',
                'enforce_bait_remapping': True,
                'sheet_nums': len(args.sheets),
                'selected_sheet_indices': args.sheets,
            },
            'model': {
                'name': 'model23_n',
                'rseed': 13,
                'model_id': 0,
            },
            'sampling': {
                'num_warmup': 1,      # Very small for quick test
                'num_samples': 5,      # Very small for quick test
                'thinning': 1,
                'target_accept_prob': 0.8,
                'progress_bar': True,
            },
            'hyperparameters': {
                'alpha': 0.02,
                'beta': 0.1,
                'thresholds': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'n_null_bins': 80,
                'disconnectivity_distance': 10,
                'max_distance': 11,
            },
            'output': {
                'output_dir': str(output_dir),
                'save_warmup': False,    # Skip to save time
                'include_extra_fields': False,  # Skip to save time
            },
            'figures': {
                'generate': False,  # Skip figures for speed
            },
            'imp': {
                'generate': False,
            }
        }
        
        config_file = script_dir / 'quick_test_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Created quick test config: {config_file}")
        print(f"  Using sheet(s): {args.sheets}")
        print(f"  Warmup: 100 steps (quick)")
        print(f"  Samples: 500 steps (quick)")
        print()
        
        config_path = config_file
    elif args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (script_dir / config_path).resolve()
    else:
        parser.error("Must provide either --config or --data")
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Modify config for quick test
    import json
    with open(config_path) as f:
        config = json.load(f)
    
    # Override with quick test settings (reduced for faster testing)
    # Note: First run includes JAX compilation which can take 10-30+ minutes for large models
    # Subsequent runs will be much faster
    config['sampling']['num_warmup'] = 50  # Reduced from 100 for faster quick test
    config['sampling']['num_samples'] = 100  # Reduced from 500 for faster quick test
    config['output']['save_warmup'] = False
    config['output']['include_extra_fields'] = False
    config['figures']['generate'] = False
    
    # Enable indirect SAINT scores (now safe since we filter to top 50 proteins)
    # With top 50 proteins, indirect SAINT scores won't create too many pairs
    if 'hyperparameters' not in config:
        config['hyperparameters'] = {}
    # Don't override use_indirect_saint - use default (True) from run.py
    # This allows indirect prey-prey pairs to be included in the model
    
    # CRITICAL PERFORMANCE FIXES:
    # 1. Filter proteins by SAINT score to reduce N (the real bottleneck)
    # 2. Filter edges to reduce M (number of dimensions being sampled)
    if 'hyperparameters' not in config:
        config['hyperparameters'] = {}
    
    # FILTERING: Keep only top 50 proteins by SAINT score for all simulations
    # This reduces N from 177 to 50, making compilation fast enough
    # N² reduction: 177² = 31,329 → 50² = 2,500 (12.5x reduction)
    # This makes JAX compilation fast enough on CPU while keeping reasonable coverage
    config['hyperparameters']['max_proteins'] = 50
    
    # Filter edges: keep top 150 edges (reduces M from 15,576 to 150)
    # This helps with sampling speed, but N reduction is more important for compilation
    config['hyperparameters']['max_edges'] = 150
    
    # Also set expected edges for the restraint
    config['hyperparameters']['hardcoded_expected_edges'] = 300
    config['hyperparameters']['hardcoded_expected_edges_std'] = 100  # ±100 as user mentioned
    
    # Save modified config
    quick_config_file = script_dir / 'quick_test_config.json'
    with open(quick_config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)
    print(f"Config: {quick_config_file}")
    print(f"Warmup: {config['sampling']['num_warmup']} steps (normally 1000)")
    print(f"Samples: {config['sampling']['num_samples']} steps (normally 10000)")
    print(f"Indirect SAINT scores: ENABLED (safe with top 50 protein filtering)")
    print(f"NOTE: First run includes JAX compilation which can take 10-30+ minutes")
    print(f"      Subsequent runs will be much faster (seconds to minutes)")
    print(f"Expected time: 10-30 minutes (first run with compilation)")
    print("=" * 60)
    print()
    
    # Run single seed with quick settings
    cmd = [
        sys.executable,
        "run.py",
        "--config", str(quick_config_file),
        "--rseed", "13",
        "--no-figures"
    ]
    
    if args.output:
        cmd.extend(["--output", str(args.output)])
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Set JAX to use CPU
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'
    
    try:
        subprocess.run(cmd, cwd=script_dir, check=True, env=env)
        print()
        print("=" * 60)
        print("✓ Quick test complete!")
        print("=" * 60)
        print(f"Results saved to: {config['output']['output_dir']}")
        print()
        print("Note: This was a quick test with minimal samples.")
        print("For production, use full settings (1000 warmup, 10000 samples)")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Test failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()


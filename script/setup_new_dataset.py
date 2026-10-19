#!/usr/bin/env python3
"""
Setup workflow for a new dataset.

This script helps organize the workflow by:
1. Creating a standardized directory structure
2. Generating a config template
3. Validating input data
4. Setting up the workflow environment
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import shutil

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def create_directory_structure(workflow_dir: Path, dataset_name: str) -> dict:
    """Create organized directory structure for new dataset."""
    dirs = {
        'input': workflow_dir / 'input' / dataset_name,
        'config': workflow_dir / 'scripts' / 'configs' / dataset_name,
        'results': workflow_dir / 'results' / dataset_name,
        'logs': workflow_dir / 'results' / dataset_name,  # Logs go with results
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def generate_config_template(dataset_name: str, input_file: str, output_dir: str, 
                            format: str = 'yaml') -> dict:
    """Generate a config template for the new dataset."""
    config = {
        'data': {
            'input_file': input_file,
            'prey_column': 'PreyGene',
            'filter': 'all',
            'enforce_bait_remapping': False,
        },
        'model': {
            'name': 'model23_n',
            'rseed': 13,
            'model_id': 0,
        },
        'sampling': {
            'num_warmup': 1000,
            'num_samples': 10000,
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
            'output_dir': output_dir,
            'save_warmup': True,
            'include_extra_fields': True,
        },
        'figures': {
            'generate': True,
        },
        'metadata': {
            'dataset_name': dataset_name,
            'created': datetime.now().isoformat(),
            'description': f'Configuration for {dataset_name} dataset',
        }
    }
    
    return config


def save_config(config: dict, filepath: Path, format: str = 'yaml'):
    """Save config to file in specified format."""
    if format.lower() == 'yaml':
        if not HAS_YAML:
            print("⚠ PyYAML not available. Saving as JSON instead.")
            format = 'json'
        
        if format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return
    
    # JSON format
    with open(filepath.with_suffix('.json'), 'w') as f:
        json.dump(config, f, indent=2)


def create_readme(dirs: dict, dataset_name: str, config_file: Path):
    """Create a README for the dataset."""
    readme_content = f"""# Dataset: {dataset_name}

## Directory Structure

```
workflow/
├── input/
│   └── {dataset_name}/    # Input data files
├── scripts/
│   └── configs/
│       └── {dataset_name}/  # Configuration files
└── results/
    └── {dataset_name}/     # Output results and logs
```

## Quick Start

1. **Place your data file** in: `{dirs['input']}`

2. **Edit the config file**: `{config_file}`

3. **Validate your data**:
   ```bash
   python validation/validate_input_data.py {dirs['input']}/your_data.xlsx
   ```

4. **Run the workflow**:
   ```bash
   python run.py --config {config_file}
   ```

## Configuration

The config file is located at: `{config_file}`

Key settings to adjust:
- `data.input_file`: Path to your Excel file
- `data.filter`: Filter condition ('all', 'wt', 'vif', 'mock')
- `sampling.num_samples`: Number of MCMC samples (10000+ for production)
- `sampling.num_warmup`: Warmup steps (1000-5000 recommended)
- `model.name`: Model to use (model23_n recommended)

## Results

Results will be saved to: `{dirs['results']}`

After running, you can:
- Extract top edges: `python get_top_edges.py results/.../0_model23_n_13.pkl 100`
- Inspect results: `python inspect_results.py results/.../`

## Notes

- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- See `USING_NEW_DATASET.md` for detailed documentation
"""
    
    readme_path = dirs['config'] / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path


def main():
    parser = argparse.ArgumentParser(
        description='Setup workflow for a new dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup new dataset with data file
  python setup_new_dataset.py my_dataset --data-file ../data/my_data.xlsx
  
  # Setup with custom output directory
  python setup_new_dataset.py my_dataset --data-file ../data/my_data.xlsx --output-dir results/my_custom_dir
  
  # Setup with JSON config
  python setup_new_dataset.py my_dataset --data-file ../data/my_data.xlsx --format json
        """
    )
    
    parser.add_argument('dataset_name', type=str,
                       help='Name of the dataset (used for directories and config)')
    parser.add_argument('--data-file', type=Path,
                       help='Path to input Excel file (optional, can set later in config)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (default: results/DATASET_NAME)')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                       help='Config file format (default: yaml)')
    parser.add_argument('--base-dir', type=Path,
                       help='Base directory for project (default: scripts/ parent)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate input data file if provided')
    
    args = parser.parse_args()
    
    # Determine workflow directory
    if args.base_dir:
        workflow_dir = args.base_dir.resolve()
    else:
        script_dir = Path(__file__).parent
        workflow_dir = script_dir.parent  # workflow/
    
    print("=" * 60)
    print(f"Setting up workflow for: {args.dataset_name}")
    print("=" * 60)
    print()
    
    # Create directory structure
    print("Creating directory structure...")
    dirs = create_directory_structure(workflow_dir, args.dataset_name)
    for name, path in dirs.items():
        print(f"  ✓ {name}: {path}")
    print()
    
    # Determine input file path
    if args.data_file:
        input_file = args.data_file
        # Copy to input directory if it exists
        if input_file.exists():
            dest = dirs['input'] / input_file.name
            if not dest.exists():
                shutil.copy(input_file, dest)
                print(f"  ✓ Copied data file to: {dest}")
            # Path relative to scripts directory
            input_file_str = f"../input/{args.dataset_name}/{input_file.name}"
        else:
            print(f"  ⚠ Data file not found: {input_file} (set manually in config)")
            input_file_str = str(args.data_file)
    else:
        input_file_str = f"../input/{args.dataset_name}/your_data.xlsx"
        print(f"  ⚠ No data file provided. Set manually in config.")
    print()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"../results/{args.dataset_name}"
    
    # Generate config
    print("Generating config template...")
    config = generate_config_template(
        args.dataset_name,
        input_file_str,
        output_dir,
        args.format
    )
    
    config_filename = f"config_{args.dataset_name}.{args.format}"
    config_path = dirs['config'] / config_filename
    
    # Also create in scripts directory for easy access
    scripts_config_path = Path(__file__).parent / config_filename
    save_config(config, scripts_config_path, args.format)
    save_config(config, config_path, args.format)
    
    print(f"  ✓ Config saved to: {scripts_config_path}")
    print(f"  ✓ Config saved to: {config_path}")
    print()
    
    # Create README
    print("Creating README...")
    readme_path = create_readme(dirs, args.dataset_name, scripts_config_path)
    print(f"  ✓ README saved to: {readme_path}")
    print()
    
    # Validate data if requested
    if args.validate and args.data_file and args.data_file.exists():
        print("Validating input data...")
        import subprocess
        result = subprocess.run(
            ['python', 'validation/validate_input_data.py', str(args.data_file)],
            cwd=Path(__file__).parent
        )
        if result.returncode != 0:
            print("  ⚠ Validation found issues. Please review and fix.")
        else:
            print("  ✓ Data validation passed!")
        print()
    
    # Summary
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Edit config: {scripts_config_path}")
    if not args.data_file or not args.data_file.exists():
        print(f"  2. Place your data file in: {dirs['input']}")
    print(f"  3. Validate data: python validation/validate_input_data.py <your_data.xlsx>")
    print(f"  4. Run workflow: python run.py --config {scripts_config_path}")
    print()
    print(f"See README for more details: {readme_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Run validation_analysis_ablations.py for all conditions (combining all three ablations per condition).
"""

from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
results_dir = root / "script" / "results"
base_comparison_dir = root / "validation_pdb_and_databases"
output_dir = root / "validation_ablations_figures"

output_dir.mkdir(parents=True, exist_ok=True)

# Process each condition (mock, wt, vif)
for condition in ["mock", "wt", "vif"]:
    output_file = output_dir / f"cullin_{condition}_ablations_validation_4panel.png"
    
    print(f"\nProcessing condition: {condition}")
    print(f"  Output: {output_file}")
    
    cmd = [
        "python",
        str(root / "script" / "validation_analysis_ablations.py"),
        "--condition", condition,
        "--results-dir", str(results_dir),
        "--base-comparison-dir", str(base_comparison_dir),
        "--output", str(output_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Success")
    else:
        print(f"  ✗ Error:")
        print(result.stderr)

print(f"\nDone! Figures saved to: {output_dir}")

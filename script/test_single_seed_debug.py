#!/usr/bin/env python3
"""
Test a single seed with enhanced debugging to diagnose segfault issues.

This script runs a single seed with:
- Python faulthandler enabled
- Increased stack size
- JAX memory safeguards
- Better error reporting

Usage:
    python test_single_seed_debug.py --config config.json --seed 13
"""

import argparse
import faulthandler
import os
import resource
import subprocess
import sys
from pathlib import Path

# Enable faulthandler immediately
faulthandler.enable()

def main():
    parser = argparse.ArgumentParser(
        description='Test single seed with enhanced debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=13, help='Random seed to use')
    parser.add_argument('--output', type=Path, help='Output directory')
    
    args = parser.parse_args()
    
    # Try to increase stack size
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        new_soft = min(64 * 1024 * 1024, hard)  # 64MB
        resource.setrlimit(resource.RLIMIT_STACK, (new_soft, hard))
        print(f"✓ Stack size: {soft / 1024 / 1024:.1f}MB → {new_soft / 1024 / 1024:.1f}MB")
    except (ValueError, OSError) as e:
        print(f"⚠ Could not increase stack size: {e}")
    
    # Set JAX environment variables
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    # Enable JAX debugging
    env['JAX_ENABLE_X64'] = 'false'  # Keep float32 for speed
    env['JAX_TRACEBACK_FILTERING'] = 'off'  # Show full tracebacks
    
    # Build command
    script_dir = Path(__file__).parent
    cmd = [
        sys.executable,
        "run.py",
        "--config", str(args.config),
        "--rseed", str(args.seed),
        "--no-figures"
    ]
    
    if args.output:
        cmd.extend(["--output", str(args.output)])
    
    print("=" * 70)
    print("DEBUGGING SINGLE SEED RUN")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Stack size: {resource.getrlimit(resource.RLIMIT_STACK)[0] / 1024 / 1024:.1f}MB")
    print("=" * 70)
    print()
    
    # Run with faulthandler
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            env=env,
            check=False
        )
        
        if result.returncode == 0:
            print("\n✓ Run completed successfully!")
            return 0
        else:
            print(f"\n✗ Run failed with exit code {result.returncode}")
            if result.returncode == -11:
                print("\n⚠ SEGMENTATION FAULT DETECTED (exit code -11)")
                print("\nPossible causes:")
                print("  1. Stack overflow during JAX compilation")
                print("  2. Memory corruption in JAX/NumPyro")
                print("  3. Incompatible JAX/NumPyro versions")
                print("  4. System-level issue (check dmesg for kernel errors)")
                print("\nSuggestions:")
                print("  • Try reducing num_warmup (e.g., 500 instead of 1000)")
                print("  • Try reducing max_proteins in config")
                print("  • Check JAX/NumPyro versions: pip list | grep -E 'jax|numpyro'")
                print("  • Try running with: ulimit -s 65536  # 64MB stack")
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())





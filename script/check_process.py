#!/usr/bin/env python3
"""
Quick script to check if MCMC process is running and monitor progress.
"""

import subprocess
import sys
from pathlib import Path

def check_process():
    """Check if the process is running and show CPU/memory usage."""
    try:
        # Find the process
        result = subprocess.run(
            ['pgrep', '-f', 'run_multiple_seeds'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ No run_multiple_seeds process found")
            return
        
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p]
        
        if not pids:
            print("❌ No run_multiple_seeds process found")
            return
        
        print(f"✓ Found process(es): {', '.join(pids)}")
        print()
        
        # Check CPU and memory usage
        for pid in pids:
            try:
                result = subprocess.run(
                    ['ps', '-p', pid, '-o', 'pid,%cpu,%mem,etime,command'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(result.stdout)
            except:
                pass
        
        # Check for log files
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / 'results'
        
        if results_dir.exists():
            log_files = list(results_dir.rglob('run.log'))
            if log_files:
                print("\n📄 Log files found:")
                for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    print(f"  {log_file}")
                    # Show last few lines
                    try:
                        with open(log_file) as f:
                            lines = f.readlines()
                            if lines:
                                print(f"    Last line: {lines[-1].strip()}")
                    except:
                        pass
        
        # Check for output files
        if results_dir.exists():
            pkl_files = list(results_dir.rglob('*.pkl'))
            if pkl_files:
                print(f"\n💾 Output files found: {len(pkl_files)}")
                for pkl_file in sorted(pkl_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    size_mb = pkl_file.stat().st_size / (1024 * 1024)
                    mtime = pkl_file.stat().st_mtime
                    import time
                    age_min = (time.time() - mtime) / 60
                    print(f"  {pkl_file.name} ({size_mb:.1f} MB, modified {age_min:.1f} min ago)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check_process()


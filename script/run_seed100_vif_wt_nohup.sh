#!/usr/bin/env bash
# Sequential nohup: Cullin VIF seed 100, then WT seed 100 (same CLI as mock seed 100).
# Logs:
#   results/cullin_vif_config_multi_seeds/nohup_seed100.log
#   results/cullin_wt_config_multi_seeds/nohup_seed100.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/script"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_STACK_SIZE=8388608

VIF_LOG="$ROOT/results/cullin_vif_config_multi_seeds/nohup_seed100.log"
WT_LOG="$ROOT/results/cullin_wt_config_multi_seeds/nohup_seed100.log"

{
  echo "======== $(date -Is) VIF seed 100 start ========="
  python -X faulthandler run.py \
    --config "$ROOT/results/cullin_vif_config_multi_seeds/seed_13/config.json" \
    --rseed 100 \
    --output "$ROOT/results/cullin_vif_config_multi_seeds/seed_100" \
    --no-figures
  echo "======== $(date -Is) VIF seed 100 done ========="
} >>"$VIF_LOG" 2>&1

{
  echo "======== $(date -Is) WT seed 100 start ========="
  python -X faulthandler run.py \
    --config "$ROOT/results/cullin_wt_config_multi_seeds/seed_13/config.json" \
    --rseed 100 \
    --output "$ROOT/results/cullin_wt_config_multi_seeds/seed_100" \
    --no-figures
  echo "======== $(date -Is) WT seed 100 done ========="
} >>"$WT_LOG" 2>&1

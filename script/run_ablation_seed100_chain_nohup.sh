#!/usr/bin/env bash
# Run MCMC seed 100 for each cullin_* ablation under script/results, sequentially.
# Logs: script/results/nohup_ablation_seed100_chain.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/script"
RES="$SCRIPT/results"
LOG="$RES/nohup_ablation_seed100_chain.log"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_STACK_SIZE=8388608

ORDER=(
  cullin_mock_no_corr
  cullin_mock_no_priors
  cullin_mock_no_saint
  cullin_vif_no_corr
  cullin_vif_no_priors
  cullin_vif_no_saint
  cullin_wt_no_corr
  cullin_wt_no_priors
  cullin_wt_no_saint
)

{
  echo "======== $(date -Is) ablation seed-100 batch start ========="
  cd "$SCRIPT"
  for name in "${ORDER[@]}"; do
    D="$RES/$name"
    if [[ ! -f "$D/seed_13/config.json" ]]; then
      echo "SKIP $name: missing seed_13/config.json"
      continue
    fi
    echo ""
    echo "======== $(date -Is) $name seed 100 start ========="
    python -X faulthandler run.py \
      --config "$D/seed_13/config.json" \
      --rseed 100 \
      --output "$D/seed_100" \
      --no-figures
    echo "======== $(date -Is) $name seed 100 done ========="
  done
  echo ""
  echo "======== $(date -Is) all ablation seed-100 runs finished ========="
} >>"$LOG" 2>&1

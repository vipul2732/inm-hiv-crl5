#!/usr/bin/env bash
# Full merged-draw hierarchical clustering (40k × M edges) — high RAM / long CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/post_sampling_analysis/nohup_imp_metrics.log"
mkdir -p "$(dirname "$LOG")"
{
  echo "======== $(date -Is) start ========="
  cd "$ROOT"
  python script/compute_post_sampling_imp_metrics.py "$@"
  echo "======== $(date -Is) done ========="
} >>"$LOG" 2>&1 &
echo "nohup PID=$! — log: $LOG"

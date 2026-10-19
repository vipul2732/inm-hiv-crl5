#!/usr/bin/env bash
# Z -> M for Cullin base multi-seed runs. Writes under updated_analysis/ by default.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Base Cullin models live under <repo>/results/ (not script/results/)
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/updated_analysis}"

for name in cullin_wt_config_multi_seeds cullin_vif_config_multi_seeds cullin_mock_config_multi_seeds; do
  merged="${RESULTS_ROOT}/${name}/merged/merged_chains.pkl"
  if [[ -f "$merged" ]]; then
    echo "=== $name ==="
    python "$SCRIPT_DIR/run_z_to_m.py" \
      --merged "$merged" \
      --out "${OUT_ROOT}/${name}_binary" \
      "$@"
  else
    echo "Skip (missing): $merged"
  fi
done

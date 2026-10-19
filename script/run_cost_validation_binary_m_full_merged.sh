#!/usr/bin/env bash
# Cost vs accuracy perturbation validation using mean(1{Z>0}) over ALL merged draws
# (binary edge rule, full ensemble — not cluster-restricted, not mean(sigmoid(z))).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="${RESULTS_DIR:-$ROOT/results}"
OUT_BASE="${OUTPUT_BASE_DIR:-$ROOT/cost_validation}"

declare -a DATASETS=(
  "cullin_mock_config_multi_seeds"
  "cullin_wt_config_multi_seeds"
  "cullin_vif_config_multi_seeds"
)

for ds in "${DATASETS[@]}"; do
  out="${OUT_BASE}/${ds}"
  mkdir -p "$out"
  echo "=== $ds -> $out ==="
  python "$ROOT/script/validate_model_cost_perturbations.py" \
    --results-dir "$RESULTS" \
    --dataset "$ds" \
    --output "$out" \
    --binary-m-mean-full-merged \
    --reference-label "true model"
done

echo "Done. Outputs under $OUT_BASE"

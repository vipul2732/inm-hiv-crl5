#!/usr/bin/env bash
# Re-run post_sampling_analysis for all three base models (same settings as ablations):
#   - Per-draw binary M = 1{Z > 0.5}  (--z-binary-threshold 0.5; must match post_sampling_z_binary.py)
#   - Hamming linkage + τ* sampling precision (compute_post_sampling_imp_metrics)
#   - Main-cluster centroid C: mean edge frequency >= 0.6 (--freq-cutoff 0.6)
#     → cluster_precision.json (Hamming draws vs C)
#
# Do NOT use --reuse-linkage: Hamming linkage must be recomputed when M or Z_thr changes.
#
# Usage (from repo root):
#   bash script/run_base_models_post_sampling_z_binary.sh
# Long runs — log:
#   nohup bash script/run_base_models_post_sampling_z_binary.sh >> post_sampling_analysis/rerun_base_z_binary.log 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASETS=(
  cullin_mock_config_multi_seeds
  cullin_wt_config_multi_seeds
  cullin_vif_config_multi_seeds
)

echo "Using Z_BINARY_THRESHOLD from script/post_sampling_z_binary.py (edit file to change)."
python -c "import sys; sys.path.insert(0,'script'); import post_sampling_z_binary as b; print('Z_BINARY_THRESHOLD =', b.Z_BINARY_THRESHOLD)"

for ds in "${DATASETS[@]}"; do
  echo "======== $(date -Is) compute_post_sampling_imp_metrics $ds ========="
  python script/compute_post_sampling_imp_metrics.py \
    --results-root "$ROOT/results" \
    --out-dir "$ROOT/post_sampling_analysis/base_models" \
    --dataset "$ds" \
    --z-binary-threshold 0.5
done

for ds in "${DATASETS[@]}"; do
  echo "======== $(date -Is) extract_main_cluster_network $ds ========="
  python script/extract_main_cluster_network.py \
    --multi-seeds-dir "$ROOT/results/$ds" \
    --labels "$ROOT/post_sampling_analysis/base_models/$ds/posterior_hamming_linkage/draw_labels_sampling_precision.npy" \
    --out-dir "$ROOT/post_sampling_analysis/base_models/$ds/main_cluster_network" \
    --z-binary-threshold 0.5 \
    --freq-cutoff 0.6
done

echo "======== $(date -Is) done all base models ========="

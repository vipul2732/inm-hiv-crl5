#!/usr/bin/env bash
# Full ablation post-sampling (same settings as run_base_models_post_sampling_z_binary.sh):
#   - Per-draw M = 1{Z > 0.5}  (--z-binary-threshold 0.5 on compute + extract)
#   - Centroid C for cluster precision: mean frequency >= 0.6 (--freq-cutoff 0.6)
#
# Steps:
#   1) Hamming linkage + τ* sampling precision (compute_post_sampling_imp_metrics)
#   2) Main-cluster network + cluster_precision.json (extract_main_cluster_network)
#   3) Optional ROC (SKIP_ROC=1 to skip)
#
# Writes under script/results/<cullin_*_no_*>/ and merged summaries:
#   script/results/summary_ablations_sampling_precision.tsv
#
# Linkage reuse: default OFF (recomputes pdist when M=1{Z>Z_thr} or other rules change).
#   REUSE_LINKAGE=1 bash script/run_ablation_sampling_precision_pipeline.sh
#
# Usage (from repo root):
#   bash script/run_ablation_sampling_precision_pipeline.sh
# Long runs:
#   nohup bash script/run_ablation_sampling_precision_pipeline.sh >> validation_ablations/ablation_sampling_precision.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RES="$SCRIPT_DIR/results"
cd "$REPO"

ABLATIONS=(
  cullin_mock_no_corr cullin_mock_no_saint cullin_mock_no_priors
  cullin_wt_no_corr cullin_wt_no_saint cullin_wt_no_priors
  cullin_vif_no_corr cullin_vif_no_saint cullin_vif_no_priors
)

echo "Using Z_BINARY_THRESHOLD from script/post_sampling_z_binary.py"
python -c "import sys; sys.path.insert(0,'script'); import post_sampling_z_binary as b; print('Z_BINARY_THRESHOLD =', b.Z_BINARY_THRESHOLD)"

missing=()
for d in "${ABLATIONS[@]}"; do
  if [[ ! -f "$RES/$d/merged/merged_chains.pkl" ]]; then
    missing+=("$d")
  fi
done
if ((${#missing[@]})); then
  echo "ERROR: missing merged_chains.pkl for: ${missing[*]}" >&2
  exit 1
fi

echo "========== $(date -Is) compute_post_sampling_imp_metrics (all ablations, one pass) ==========" >&2
cmd=(
  python "$SCRIPT_DIR/compute_post_sampling_imp_metrics.py"
  --results-root "$RES"
  --out-dir "$RES"
  --merged-summary-tsv "$RES/summary_ablations_sampling_precision.tsv"
  --z-binary-threshold 0.5
)
if [[ "${REUSE_LINKAGE:-0}" == "1" ]]; then
  cmd+=(--reuse-linkage)
  echo "  REUSE_LINKAGE=1: reusing existing linkage_matrix.npy where present" >&2
fi
for d in "${ABLATIONS[@]}"; do
  cmd+=(--dataset "$d")
done
"${cmd[@]}"

for d in "${ABLATIONS[@]}"; do
  labels="$RES/$d/posterior_hamming_linkage/draw_labels_sampling_precision.npy"
  if [[ ! -f "$labels" ]]; then
    echo "ERROR $d: missing $labels" >&2
    exit 1
  fi
  echo "========== $(date -Is) extract_main_cluster_network $d ==========" >&2
  python "$SCRIPT_DIR/extract_main_cluster_network.py" \
    --multi-seeds-dir "$RES/$d" \
    --labels "$labels" \
    --out-dir "$RES/$d/main_cluster_network" \
    --z-binary-threshold 0.5 \
    --freq-cutoff 0.6
done

if [[ "${SKIP_ROC:-0}" == "1" ]]; then
  echo "SKIP_ROC=1 — skipping plot_roc_base_vs_ablations + combine_roc_three_panel." >&2
else
  echo "========== ROC + 3-panel ==========" >&2
  python "$SCRIPT_DIR/plot_roc_base_vs_ablations.py" --edge-frequency
  python "$SCRIPT_DIR/combine_roc_three_panel.py" --suffix _edgefreq
fi

echo "Done. Key outputs: $RES/summary_ablations_sampling_precision.tsv and per-dataset main_cluster_network/cluster_precision.json" >&2

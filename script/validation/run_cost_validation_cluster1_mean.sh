#!/usr/bin/env bash
# Cost vs accuracy perturbation validation (same engine as validate_model_cost_perturbations.py).
#
# REFERENCE_MODE (default: sigmoid) — pick the vector used as "true model" / reference p_base:
#
#   sigmoid (DEFAULT, closest to original cost_validation/ behavior)
#       mean σ(z) over ALL merged draws — same as run_cost_validation_all.sh with no extra flags.
#       Legend label: "true model" (default in Python). Good cost–accuracy corner behavior.
#
#   sigmoid_mean_z_main
#       σ(mean z) over draws in the largest τ* cluster only (main_cluster_summary.json + labels).
#       Different from sigmoid: uses mean z first, then σ (not mean of σ(z)).
#
#   binary_m_full
#       mean per-draw binary M = 1{Z>Z_thr} over all merged draws (Z_thr from post_sampling_z_binary).
#       Can break "lowest cost" monotonicity vs the surrogate cost (see earlier discussion).
#
#   main_cluster
#       mean binary M only in draws of the largest τ* cluster (mean_edges_main_cluster.npy).
#
# Back-compat: USE_MAIN_CLUSTER_MEAN=1 forces main_cluster (mean binary M) mode.
#
# Outputs: OUTPUT_BASE_DIR (default: cost_validation_cluster1_mean/) per dataset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
BM="${BM:-$PROJECT_ROOT/post_sampling_analysis/base_models}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$PROJECT_ROOT/cost_validation_cluster1_mean}"

REF="${REFERENCE_MODE:-sigmoid}"
if [[ "${USE_MAIN_CLUSTER_MEAN:-0}" == "1" ]]; then
  REF=main_cluster
fi

mkdir -p "$OUTPUT_BASE_DIR"

declare -a datasets=(
  "cullin_mock_config_multi_seeds"
  "cullin_vif_config_multi_seeds"
  "cullin_wt_config_multi_seeds"
)

for dataset in "${datasets[@]}"; do
  out="${OUTPUT_BASE_DIR}/${dataset}"
  mkdir -p "$out"
  echo "=== $dataset (REFERENCE_MODE=$REF) -> $out ==="
  case "$REF" in
    sigmoid)
      python "$PROJECT_ROOT/script/validate_model_cost_perturbations.py" \
        --results-dir "$RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$out"
      ;;
    sigmoid_mean_z_main)
      summary="${BM}/${dataset}/main_cluster_network/main_cluster_summary.json"
      if [[ ! -f "$summary" ]]; then
        echo "Skip $dataset (missing $summary)" >&2
        continue
      fi
      python "$PROJECT_ROOT/script/validate_model_cost_perturbations.py" \
        --results-dir "$RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$out" \
        --sigmoid-mean-z-main-cluster \
        --main-cluster-summary-json "$summary" \
        --reference-label "σ(mean z) main τ* cluster"
      ;;
    binary_m_full)
      merged="${RESULTS_DIR}/${dataset}/merged/merged_chains.pkl"
      if [[ ! -f "$merged" ]]; then
        echo "Skip $dataset (missing $merged)" >&2
        continue
      fi
      python "$PROJECT_ROOT/script/validate_model_cost_perturbations.py" \
        --results-dir "$RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$out" \
        --binary-m-mean-full-merged \
        --reference-label "mean binary M (full merged)"
      ;;
    main_cluster)
      mean_npy="${BM}/${dataset}/main_cluster_network/mean_edges_main_cluster.npy"
      if [[ ! -f "$mean_npy" ]]; then
        echo "Skip $dataset (missing $mean_npy)" >&2
        continue
      fi
      python "$PROJECT_ROOT/script/validate_model_cost_perturbations.py" \
        --results-dir "$RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$out" \
        --mean-edges-npy "$mean_npy" \
        --reference-label "mean M (main τ* cluster)"
      ;;
    *)
      echo "Unknown REFERENCE_MODE='$REF' (use sigmoid | sigmoid_mean_z_main | binary_m_full | main_cluster)" >&2
      exit 1
      ;;
  esac
done

echo "Done. Outputs under $OUTPUT_BASE_DIR (REFERENCE_MODE=$REF)"

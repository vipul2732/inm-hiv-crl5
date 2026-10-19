#!/usr/bin/env bash
# 4-panel validation ranked by Wilson lower bound on P(edge=1) on draws in the
# representative (largest) cluster from draw_labels_sampling_precision.npy (τ* partition).
#
# Uses merged z + post_sampling posterior_hamming_linkage labels. If your cluster1_edgefreq
# figure was built from updated_analysis/.../binary ensemble + maxclust_labels.npy, run
# compute_binomial_lb_edge_scores.py with --ensemble-dir ... instead for an exact match.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAL="$ROOT/validation_pdb_and_databases"
BM="$ROOT/post_sampling_analysis/base_models"

for ds in cullin_mock_config_multi_seeds cullin_wt_config_multi_seeds cullin_vif_config_multi_seeds; do
  python "$ROOT/script/compute_binomial_lb_edge_scores.py" \
    --merged-chains "$ROOT/results/$ds/merged/merged_chains.pkl" \
    --labels "$BM/$ds/posterior_hamming_linkage/draw_labels_sampling_precision.npy" \
    --largest-cluster \
    --out-npy "$BM/$ds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy" \
    --method wilson --alpha 0.05 \
    --z-binary-threshold 0.5
done

MOCK_NPY="$BM/cullin_mock_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"
WT_NPY="$BM/cullin_wt_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"
VIF_NPY="$BM/cullin_vif_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"

LABEL='Wilson 95% score lower bound on edge proportion (largest τ* cluster draws, M=1{Z>0.5})'

python "$ROOT/script/validation_analysis.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_analysis_4panel_any_database_cluster1_wilson_lb.png" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --no-suptitle

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_statistics_cluster1_wilson_lb.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --ranking-score-label "$LABEL"

echo "Wrote $VAL/validation_analysis_4panel_any_database_cluster1_wilson_lb.png"
echo "Wrote $VAL/validation_statistics_cluster1_wilson_lb.txt"

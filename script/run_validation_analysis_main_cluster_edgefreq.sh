#!/usr/bin/env bash
# 4-panel validation figure using main-cluster edge frequencies (mean binary M over
# draws in the largest hierarchical cluster). Optional threshold sweep for enrichment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAL="$ROOT/validation_pdb_and_databases"
BM="$ROOT/post_sampling_analysis/base_models"

MOCK_NPY="$BM/cullin_mock_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy"
WT_NPY="$BM/cullin_wt_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy"
VIF_NPY="$BM/cullin_vif_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy"

python "$ROOT/script/validation_analysis.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_analysis_4panel_any_database_main_cluster_edgefreq.png" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --suptitle "Ranked by main-cluster edge frequency (fraction of binary samples with edge on)"

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_statistics_main_cluster_edgefreq.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY"

python "$ROOT/script/sweep_edge_frequency_threshold_enrichment.py" \
  --input-dir "$VAL" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --output-tsv "$VAL/main_cluster_edgefreq_threshold_sweep.tsv" \
  --output-best-tsv "$VAL/main_cluster_edgefreq_threshold_best.tsv" \
  --min-per-split 100 \
  --tau-step 0.005

echo "Wrote $VAL/validation_analysis_4panel_any_database_main_cluster_edgefreq.png"
echo "Wrote $VAL/validation_statistics_main_cluster_edgefreq.txt"
echo "Wrote $VAL/main_cluster_edgefreq_threshold_sweep.tsv"
echo "Wrote $VAL/main_cluster_edgefreq_threshold_best.tsv"

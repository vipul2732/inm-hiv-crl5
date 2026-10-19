#!/usr/bin/env bash
# 4-panel figure like validation_analysis.py, ranked by cluster-1 edge frequency.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPD="$ROOT/updated_analysis"
VAL="$ROOT/validation_pdb_and_databases"

MOCK_NPY="$UPD/cullin_mock_config_multi_seeds_binary/clustering_k3/mean_edges_cluster_1.npy"
WT_NPY="$UPD/cullin_wt_config_multi_seeds_binary/clustering_k3/mean_edges_cluster_1.npy"
VIF_NPY="$UPD/cullin_vif_config_multi_seeds_binary/clustering_k3/mean_edges_cluster_1.npy"

python "$ROOT/script/validation_analysis.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_analysis_4panel_any_database_cluster1_edgefreq.png" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY"

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_statistics_cluster1_edgefreq.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY"

echo "Wrote $VAL/validation_analysis_4panel_any_database_cluster1_edgefreq.png"
echo "Wrote $VAL/validation_statistics_cluster1_edgefreq.txt"

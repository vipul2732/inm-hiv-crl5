#!/usr/bin/env bash
# Enrichment vs database at fixed binary edge cutoffs (freq >= τ → edge = 1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAL="$ROOT/validation_pdb_and_databases"
BM="$ROOT/post_sampling_analysis/base_models"

python "$ROOT/script/sweep_edge_frequency_threshold_enrichment.py" \
  --input-dir "$VAL" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$BM/cullin_mock_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy" \
  --edge-frequency-wt "$BM/cullin_wt_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy" \
  --edge-frequency-vif "$BM/cullin_vif_config_multi_seeds/main_cluster_network/mean_edges_main_cluster.npy" \
  --fixed-thresholds "0.9,0.8,0.7,0.6" \
  --min-per-split 50 \
  --output-tsv "$VAL/main_cluster_cutoffs_0p6_0p9_enrichment.tsv" \
  --output-best-tsv "$VAL/main_cluster_cutoffs_0p6_0p9_best.tsv"

echo "Wrote $VAL/main_cluster_cutoffs_0p6_0p9_enrichment.tsv"
echo "Wrote $VAL/main_cluster_cutoffs_0p6_0p9_best.tsv"

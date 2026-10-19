#!/usr/bin/env bash
# 4-panel validation (Wilson LB ranking) with direct-contact BioGRID + CORUM only.
# Ground truth: In_Any_Database from comparison TSVs built with
#   database/BIOGRID-ALL-5.0.253.direct_contact.mitab.txt + corum_humanComplexes.tsv
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAL="$ROOT/validation_pdb_and_databases"
BM="$ROOT/post_sampling_analysis/base_models"
SUFFIX="direct_biogrid_corum"
BIOGRID="$ROOT/database/BIOGRID-ALL-5.0.253.direct_contact.mitab.txt"
CORUM="$ROOT/database/corum_humanComplexes.tsv"

for ds in wt mock vif; do
  python "$ROOT/script/compare_with_databases_simple.py" \
    --input "$VAL/cullin_${ds}_pdb_validation.tsv" \
    --output "$VAL/cullin_${ds}_database_comparison_${SUFFIX}.tsv" \
    --biogrid "$BIOGRID" \
    --corum "$CORUM"
done

MOCK_NPY="$BM/cullin_mock_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"
WT_NPY="$BM/cullin_wt_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"
VIF_NPY="$BM/cullin_vif_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"

for npy in "$MOCK_NPY" "$WT_NPY" "$VIF_NPY"; do
  if [[ ! -f "$npy" ]]; then
    echo "Missing Wilson LB scores: $npy" >&2
    echo "Run script/run_validation_analysis_cluster1_wilson_lb.sh first." >&2
    exit 1
  fi
done

RANK_LABEL='Wilson 95% score lower bound on edge proportion (largest τ* cluster draws, M=1{Z>0.5})'
POS_LABEL='In_Any_Database (BioGRID direct contact MI:0407, or CORUM; human only for BioGRID)'

python "$ROOT/script/validation_analysis.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_analysis_4panel_direct_biogrid_corum_cluster1_wilson_lb.png" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix "$SUFFIX" \
  --no-suptitle

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_statistics_cluster1_wilson_lb_direct_biogrid_corum.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix "$SUFFIX" \
  --ranking-score-label "$RANK_LABEL" \
  --known-positive-label "$POS_LABEL"

echo "Wrote $VAL/validation_analysis_4panel_direct_biogrid_corum_cluster1_wilson_lb.png"
echo "Wrote $VAL/validation_statistics_cluster1_wilson_lb_direct_biogrid_corum.txt"

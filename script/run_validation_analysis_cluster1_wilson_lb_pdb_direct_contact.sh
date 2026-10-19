#!/usr/bin/env bash
# 4-panel validation (Wilson LB ranking) with PDB direct-contact ground truth only.
# Positive = BSA >= 400 Å² (experimental shared PDB, homolog, or structural recovery).
# Excludes co-complex pairs that fail the BSA threshold.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAL="$ROOT/validation_pdb_and_databases"
BM="$ROOT/post_sampling_analysis/base_models"
SUFFIX="pdb_direct_contact"

python "$ROOT/script/build_pdb_direct_contact_comparison.py" \
  --pdb-output-dir "$ROOT/pdb/output" \
  --output-dir "$VAL" \
  --suffix "$SUFFIX"

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
POS_LABEL='In_PDB_Direct_Contact (BSA >= 400 Å²: experimental, homolog, or structural recovery; excludes co-complex no-contact)'

python "$ROOT/script/validation_analysis.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_analysis_4panel_pdb_direct_contact_cluster1_wilson_lb.png" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix "$SUFFIX" \
  --known-positive-column "In_PDB_Direct_Contact" \
  --no-suptitle

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" \
  --output "$VAL/validation_statistics_cluster1_wilson_lb_pdb_direct_contact.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix "$SUFFIX" \
  --known-positive-column "In_PDB_Direct_Contact" \
  --ranking-score-label "$RANK_LABEL" \
  --known-positive-label "$POS_LABEL"

echo "Wrote $VAL/validation_analysis_4panel_pdb_direct_contact_cluster1_wilson_lb.png"
echo "Wrote $VAL/validation_statistics_cluster1_wilson_lb_pdb_direct_contact.txt"

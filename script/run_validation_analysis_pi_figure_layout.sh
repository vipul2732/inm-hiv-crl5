#!/usr/bin/env bash
# PI figure layout (split BioGRID+CORUM vs PDB direct in 4-panel figures):
#   Main:     enrichment + cumulative (BioGRID+CORUM | PDB direct)
#   Supp:     evidence rate + precision@K (BioGRID+CORUM | PDB direct)
#   Separate: full 4-panel expanded PDB structural evidence
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VAL="$ROOT/validation_pdb_and_databases"
BM="$ROOT/post_sampling_analysis/base_models"
BIOGRID="$ROOT/database/BIOGRID-ALL-5.0.253.direct_contact.mitab.txt"
CORUM="$ROOT/database/corum_humanComplexes.tsv"

for ds in wt mock vif; do
  python "$ROOT/script/compare_with_databases_simple.py" \
    --input "$VAL/cullin_${ds}_pdb_validation.tsv" \
    --output "$VAL/cullin_${ds}_database_comparison_direct_biogrid_corum.tsv" \
    --biogrid "$BIOGRID" \
    --corum "$CORUM"
done

python "$ROOT/script/build_pdb_direct_contact_comparison.py" \
  --pdb-output-dir "$ROOT/pdb/output" --output-dir "$VAL" --mode direct

python "$ROOT/script/build_pdb_direct_contact_comparison.py" \
  --pdb-output-dir "$ROOT/pdb/output" --output-dir "$VAL" --mode expanded

MOCK_NPY="$BM/cullin_mock_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"
WT_NPY="$BM/cullin_wt_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"
VIF_NPY="$BM/cullin_vif_config_multi_seeds/posterior_hamming_linkage/wilson_lb_edges_cluster_1.npy"

for npy in "$MOCK_NPY" "$WT_NPY" "$VIF_NPY"; do
  [[ -f "$npy" ]] || { echo "Missing Wilson LB: $npy"; exit 1; }
done

python "$ROOT/script/validation_analysis_split_figures.py" \
  --input-dir "$VAL" \
  --output-dir "$VAL" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" \
  --edge-frequency-wt "$WT_NPY" \
  --edge-frequency-vif "$VIF_NPY"

RANK_LABEL='Wilson 95% score lower bound on edge proportion (largest τ* cluster draws, M=1{Z>0.5})'

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" --output "$VAL/validation_statistics_direct_biogrid_corum_cluster1_wilson_lb.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" --edge-frequency-wt "$WT_NPY" --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix direct_biogrid_corum \
  --known-positive-label "BioGRID direct (MI:0407) or CORUM" \
  --ranking-score-label "$RANK_LABEL"

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" --output "$VAL/validation_statistics_pdb_direct_contact_cluster1_wilson_lb.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" --edge-frequency-wt "$WT_NPY" --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix pdb_direct_contact \
  --known-positive-column In_PDB_Direct_Contact \
  --known-positive-label "PDB direct contact (BSA >= 400 Å²)" \
  --ranking-score-label "$RANK_LABEL"

python "$ROOT/script/generate_validation_stats.py" \
  --input-dir "$VAL" --output "$VAL/validation_statistics_pdb_structural_evidence_cluster1_wilson_lb.txt" \
  --results-dir "$ROOT/results" \
  --edge-frequency-mock "$MOCK_NPY" --edge-frequency-wt "$WT_NPY" --edge-frequency-vif "$VIF_NPY" \
  --comparison-suffix pdb_structural_evidence \
  --known-positive-column In_PDB_Structural_Evidence \
  --known-positive-label "PDB structural evidence (direct + co-complex)" \
  --ranking-score-label "$RANK_LABEL"

echo "Wrote $VAL/validation_analysis_4panel_split_main_cluster1_wilson_lb.png"
echo "Wrote $VAL/validation_analysis_4panel_split_supp_cluster1_wilson_lb.png"
echo "Wrote $VAL/validation_analysis_4panel_pdb_structural_evidence_cluster1_wilson_lb.png"

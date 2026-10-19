#!/usr/bin/env bash
# Post-sampling for p53 multi-seed models (Hamming linkage + main cluster + centroid vs inputs).
# Heavy step: pdist on merged draws (~6 GiB condensed distances for 40k × 3160 edges).
#
# Run in background with nohup (from repo root):
#   nohup bash script/run_p53_post_sampling_nohup.sh >> post_sampling_analysis/p53/nohup_post_sampling.log 2>&1 &
#   echo $!
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======== $(date -Is) start ========="

python script/compute_post_sampling_imp_metrics.py \
  --results-root "$ROOT/results" \
  --out-dir "$ROOT/post_sampling_analysis/p53" \
  --dataset p53_contact_mutations_only_config_multi_seeds \
  --dataset p53_structural_destabilization_only_config_multi_seeds \
  --dataset Serine_mutants_only_config_multi_seeds \
  --z-binary-threshold 0.5 \
  --merged-summary-tsv "$ROOT/post_sampling_analysis/p53/summary_p53_models.tsv"

for ds in \
  p53_contact_mutations_only_config_multi_seeds \
  p53_structural_destabilization_only_config_multi_seeds \
  Serine_mutants_only_config_multi_seeds
do
  echo "======== $(date -Is) extract_main_cluster_network $ds ========="
  python script/extract_main_cluster_network.py \
    --multi-seeds-dir "$ROOT/results/$ds" \
    --labels "$ROOT/post_sampling_analysis/p53/$ds/posterior_hamming_linkage/draw_labels_sampling_precision.npy" \
    --out-dir "$ROOT/post_sampling_analysis/p53/$ds/main_cluster_network" \
    --z-binary-threshold 0.5 \
    --freq-cutoff 0.6
done

for ds in \
  p53_contact_mutations_only_config_multi_seeds \
  p53_structural_destabilization_only_config_multi_seeds \
  Serine_mutants_only_config_multi_seeds
do
  echo "======== $(date -Is) correlate_centroid_vs_inputs $ds ========="
  python script/correlate_centroid_vs_inputs.py \
    --main-cluster-dir "$ROOT/post_sampling_analysis/p53/$ds/main_cluster_network" \
    --multi-seeds-dir "$ROOT/results/$ds"
done

echo "======== $(date -Is) done ========="

#!/usr/bin/env bash
# Mirror of run_base_models_post_sampling_z_binary.sh for ablations under script/results/.
# Delegates to run_ablation_sampling_precision_pipeline.sh (sampling precision + main-cluster
# network + mentor cluster precision + model_precision TSV + optional ROC).
exec "$(cd "$(dirname "$0")" && pwd)/run_ablation_sampling_precision_pipeline.sh"

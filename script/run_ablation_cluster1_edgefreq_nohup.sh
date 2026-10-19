#!/usr/bin/env bash
# Nohup: all ablations — refresh binary from merged → cluster-1 means → main-cluster
# extract (heavy) → ROC edge-frequency PNGs + 3-panel combine.
# Log: validation_ablations/nohup_ablation_pipeline.log (appended)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p validation_ablations
LOG="$ROOT/validation_ablations/nohup_ablation_pipeline.log"
nohup bash -c "
  set -euo pipefail
  cd \"$ROOT\"
  echo \"======== \$(date -Is) start PID=\$\$ =========\"
  bash script/run_ablation_regenerate_binary_from_merged.sh
  bash script/run_ablation_cluster1_edgefreq.sh
  bash script/run_ablation_main_cluster_network.sh
  python script/plot_roc_base_vs_ablations.py --edge-frequency
  python script/combine_roc_three_panel.py --suffix _edgefreq
  echo \"======== \$(date -Is) done =========\"
" >>"$LOG" 2>&1 &
echo "Launched nohup job PID=$! (log appended to $LOG)"
echo "  tail -f $LOG"

#!/usr/bin/env bash
# For each ablation: full merged draws → binary ensemble → hierarchical k=3 labels →
# main_cluster_network/mean_edges_main_cluster.npy (for ROC with plot_roc_base_vs_ablations --edge-frequency).
# Uses script/prepare_ablation_main_cluster.py (handles 20k/30k label mismatches vs merged z).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RES="${REPO_ROOT}/script/results"

for cond in mock wt vif; do
  for slug in no_corr no_saint no_priors; do
    name="cullin_${cond}_${slug}"
    d="${RES}/${name}"
    if [[ ! -f "${d}/merged/merged_chains.pkl" ]]; then
      echo "SKIP $name (no merged_chains.pkl)"
      continue
    fi
    echo "=== $name ==="
    python "${SCRIPT_DIR}/prepare_ablation_main_cluster.py" --multi-seeds-dir "$d"
  done
done
echo "All ablation main_cluster_network steps done."

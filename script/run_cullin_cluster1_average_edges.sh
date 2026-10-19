#!/usr/bin/env bash
# Mean edge vector over maxclust cluster 1 for each Cullin base binary ensemble.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/updated_analysis}"

for name in \
  cullin_mock_config_multi_seeds_binary \
  cullin_wt_config_multi_seeds_binary \
  cullin_vif_config_multi_seeds_binary; do
  ens="${OUT_ROOT}/${name}"
  cdir="${ens}/clustering_k3"
  label="${cdir}/maxclust_labels.npy"
  extra=()
  if [[ ! -f "$label" ]]; then
    echo "=== $name (recompute linkage + save labels) ==="
    extra=(--recompute-linkage --write-labels)
  else
    echo "=== $name (using existing maxclust_labels.npy) ==="
  fi
  python "$SCRIPT_DIR/average_cluster_edges.py" \
    --ensemble-dir "$ens" \
    --out-dir "$cdir" \
    --cluster-id 1 \
    "${extra[@]}"
done
echo "All done."

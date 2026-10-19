#!/usr/bin/env bash
# Build mean_edges_cluster_1.npy for each ablation under script/results from merged_chains.pkl.
# Steps per dataset: Z -> binary ensemble (run_z_to_m), then average_cluster_edges (k=3, cluster 1).
# If merged gains draws (e.g. new chain merged) but binary_ensemble is stale, we re-run Z→M and
# --recompute-linkage so maxclust_labels.npy matches the new row count.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RES="${REPO_ROOT}/script/results"

# Exit 0 iff binary exists and its row count equals merged z rows.
binary_rows_match_merged() {
  local merged="$1" bin="$2"
  python3 - "${merged}" "${bin}" <<'PY'
import pickle
import sys
from pathlib import Path

import numpy as np

merged = Path(sys.argv[1])
bin_path = Path(sys.argv[2])
with open(merged, "rb") as f:
    z = pickle.load(f)["merged_samples"]["z"]
if hasattr(z, "block_until_ready"):
    z = np.asarray(z)
else:
    z = np.asarray(z)
T = int(z.shape[0])
if not bin_path.is_file():
    sys.exit(1)
n = int(np.load(bin_path, mmap_mode="r").shape[0])
sys.exit(0 if n == T else 1)
PY
}

# Exit 0 iff maxclust label vector length equals binary ensemble rows.
label_rows_match_binary() {
  local label="$1" bin="$2"
  python3 - "${label}" "${bin}" <<'PY'
import sys
from pathlib import Path

import numpy as np

lab = Path(sys.argv[1])
binp = Path(sys.argv[2])
L = int(np.load(lab).shape[0])
T = int(np.load(binp, mmap_mode="r").shape[0])
sys.exit(0 if L == T else 1)
PY
}

for cond in mock wt vif; do
  for slug in no_corr no_saint no_priors; do
    name="cullin_${cond}_${slug}"
    d="${RES}/${name}"
    merged="${d}/merged/merged_chains.pkl"
    if [[ ! -f "$merged" ]]; then
      echo "SKIP $name (no merged_chains.pkl)"
      continue
    fi
    ens="${d}/binary_ensemble_from_merged"
    cdir="${d}/clustering_k3"
    label="${cdir}/maxclust_labels.npy"
    bin_full="${ens}/binary_edges_full.npy"
    extra=()
    binary_regen=false

    if binary_rows_match_merged "$merged" "$bin_full"; then
      :
    else
      echo "=== $name: Z→binary (sync binary row count to merged z) ==="
      python "${SCRIPT_DIR}/run_z_to_m.py" --merged "$merged" --out "$ens"
      binary_regen=true
    fi

    if [[ ! -f "$label" ]]; then
      echo "=== $name: average_cluster_edges (first labels: linkage+k=3) ==="
      extra=(--recompute-linkage --write-labels)
    elif [[ "$binary_regen" == true ]]; then
      echo "=== $name: average_cluster_edges (re-linkage after Z→binary) ==="
      extra=(--recompute-linkage --write-labels)
    elif label_rows_match_binary "$label" "$bin_full"; then
      echo "=== $name: average_cluster_edges (existing labels) ==="
    else
      echo "=== $name: average_cluster_edges (label length != binary; re-linkage) ==="
      extra=(--recompute-linkage --write-labels)
    fi

    python "${SCRIPT_DIR}/average_cluster_edges.py" \
      --ensemble-dir "$ens" \
      --out-dir "$cdir" \
      --cluster-id 1 \
      --k-clusters 3 \
      "${extra[@]}"
  done
done
echo "All ablation cluster-1 edge vectors done."

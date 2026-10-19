#!/usr/bin/env bash
# Rebuild binary_edges_full.npy from each ablation's merged_chains.pkl (M = 1{Z>0}).
# Use after re-merging chains (e.g. 4th chain added) so the ensemble row count matches z.
#
# Output: <dataset>/binary_ensemble_from_merged/binary_edges_full.npy (+ metadata).
#
# Next (clustering / scores must match the new draw count):
#   bash script/run_ablation_cluster1_edgefreq.sh
#   bash script/run_ablation_main_cluster_network.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RES="${REPO_ROOT}/script/results"

shopt -s nullglob
for d in "${RES}"/cullin_*/; do
  [[ -d "$d" ]] || continue
  d="${d%/}"
  name="$(basename "$d")"
  merged="${d}/merged/merged_chains.pkl"
  if [[ ! -f "$merged" ]]; then
    echo "SKIP ${name} (no merged/merged_chains.pkl)"
    continue
  fi
  ens="${d}/binary_ensemble_from_merged"
  echo "=== ${name}: Z→binary → ${ens} ==="
  python "${SCRIPT_DIR}/run_z_to_m.py" --merged "$merged" --out "$ens"
done
shopt -u nullglob
echo "Done: binary ensembles refreshed from merged chains."

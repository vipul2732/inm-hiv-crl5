#!/usr/bin/env bash
# Rerun MCMC for cullin_mock_no_corr seed 13 using seed_42 config (same graph / M=2926),
# then merge all four chains. Log: script/results/cullin_mock_no_corr/rerun_seed13_merge_nohup.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/script"
RES="$SCRIPT/results"
D="$RES/cullin_mock_no_corr"
LOG="$D/rerun_seed13_merge_nohup.log"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_STACK_SIZE=8388608

{
  echo "======== $(date -Is) rerun seed 13 start (config from seed_42) ========"
  cd "$SCRIPT"
  python -X faulthandler run.py \
    --config "$D/seed_42/config.json" \
    --rseed 13 \
    --output "$D/seed_13" \
    --no-figures

  echo "======== $(date -Is) verify z.shape[1] == 2926 ========"
  python3 << PY
import pickle
from pathlib import Path
import numpy as np
pkl = sorted(Path(r"$D/seed_13").glob("0_model23_n_no_corr_13.pkl"))
assert pkl, "missing 0_model23_n_no_corr_13.pkl"
with open(pkl[0], "rb") as f:
    z = np.asarray(pickle.load(f)["samples"]["z"])
print("z.shape", z.shape)
if z.shape[1] != 2926:
    raise SystemExit(f"BAD z second dim {z.shape[1]}, expected 2926; merge would drop this chain")
PY

  echo "======== $(date -Is) merge_and_analyze_convergence ========"
  cd "$ROOT"
  python script/merge_and_analyze_convergence.py "$D" --no-plots

  echo "======== $(date -Is) verify merged_chains.pkl ========"
  python3 << PY
import pickle
from pathlib import Path
p = Path(r"$D/merged/merged_chains.pkl")
with open(p, "rb") as f:
    d = pickle.load(f)
print("n_chains", d["n_chains"], "seeds", d["seeds"])
z = d["merged_samples"]["z"]
import numpy as np
z = np.asarray(z)
print("merged z.shape", z.shape)
PY

  echo "======== $(date -Is) all done ========"
} >>"$LOG" 2>&1

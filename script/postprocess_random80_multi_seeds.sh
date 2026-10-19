#!/usr/bin/env bash
# Merge chains, extract all_pairs.tsv, compare to base distribution.
# Usage: ./postprocess_random80_multi_seeds.sh <mock|wt|vif>
# Expects: results/cullin_<cond>_random80_custom/seed_{13,42,71}/

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COND="${1:?usage: $0 mock|wt|vif}"

case "$COND" in
  mock) RDIR="$ROOT/results/cullin_mock_random80_custom" ;;
  wt)   RDIR="$ROOT/results/cullin_wt_random80_custom" ;;
  vif)  RDIR="$ROOT/results/cullin_vif_random80_custom" ;;
  *) echo "Unknown condition: $COND"; exit 1 ;;
esac

MD="$RDIR/seed_13/0_model23_n_13_model_data.pkl"
if [[ ! -f "$MD" ]]; then
  echo "Missing model_data (run multi-seed first): $MD"
  exit 1
fi

echo "==> merge: $RDIR"
python3 "$ROOT/script/merge_and_analyze_convergence.py" "$RDIR" --no-plots --burn-in 0

echo "==> extract all_pairs"
python3 "$ROOT/script/extract_all_pairs.py" \
  --results "$RDIR/merged/merged_chains.pkl" \
  --model-data "$MD" \
  --output "$RDIR/all_pairs.tsv"

echo "==> compare distribution"
python3 "$ROOT/script/compare_probability_distribution_base_vs_random80.py" --condition "$COND"

echo "Done."

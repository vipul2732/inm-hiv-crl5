#!/usr/bin/env bash
# Merge chains, extract all_pairs.tsv, plot base vs random-80-no-priors distribution.
# Prereq: multi-seed runs in results/cullin_mock_config_random80_no_priors_multi_seeds/seed_{13,42,71}/
#   (see input/cullin/cullin_mock/README_random80_no_priors.md)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RDIR="$ROOT/results/cullin_mock_config_random80_no_priors_multi_seeds"

if [[ ! -d "$RDIR/seed_13" ]]; then
  echo "Missing seed runs: $RDIR/seed_13"
  exit 1
fi

shopt -s nullglob
mds=( "$RDIR"/seed_13/*_model_data.pkl )
if [[ ${#mds[@]} -eq 0 ]]; then
  echo "No *_model_data.pkl under $RDIR/seed_13"
  exit 1
fi
MD="${mds[0]}"

echo "==> merge: $RDIR"
python3 "$ROOT/script/merge_and_analyze_convergence.py" "$RDIR" --no-plots --burn-in 0

echo "==> extract all_pairs (model_data: $MD)"
python3 "$ROOT/script/extract_all_pairs.py" \
  --results "$RDIR/merged/merged_chains.pkl" \
  --model-data "$MD" \
  --output "$RDIR/all_pairs.tsv"

echo "==> compare distribution vs base mock"
python3 "$ROOT/script/compare_probability_distribution_base_vs_random80.py" \
  --condition mock_random80_no_priors

echo "Done. Figure: validation_base_models/cullin_mock/probability_distribution_mock_base_vs_random80_no_priors.png"

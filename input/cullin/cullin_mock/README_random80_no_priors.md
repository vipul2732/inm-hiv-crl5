# Random 80-protein subset + **no topology priors** (mock)

Same random 80-prey sampling as `cullin_mock_config_random80.json`, but uses the ablation model **`model23_n_no_priors`** (no degree power-law prior, no edge-count prior), matching the “without prior” ablation style.

## Config

- `cullin_mock_config_random80_no_priors.json` → `results/cullin_mock_config_random80_no_priors_multi_seeds/` (via `run_multiple_seeds.py`)

## Run (three seeds, from repo root)

```bash
cd script
python3 run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config_random80_no_priors.json \
  --seeds 13 42 71
```

## Post-process (merge + `all_pairs.tsv` + figure)

**Required after MCMC finishes** — the plot is not created until you merge seeds and extract pairs:

```bash
bash script/postprocess_random80_no_priors_multi_seeds.sh
```

Or manually: merge `cullin_mock_config_random80_no_priors_multi_seeds`, then `extract_all_pairs.py`, then:

```bash
python3 script/compare_probability_distribution_base_vs_random80.py --condition mock_random80_no_priors
```

## Output figure

- `validation_base_models/cullin_mock/probability_distribution_mock_base_vs_random80_no_priors.png`

Baseline for comparison remains **standard mock multi-seed** `results/cullin_mock_config_multi_seeds/all_pairs.tsv` (same as the original random-80 comparison figure).

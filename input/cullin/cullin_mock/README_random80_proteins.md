# Random 80-protein subset (no ranking, no pre-subset protein filters)

The default base pipeline keeps **at most 80 prey proteins** chosen by **combined SAINT + robustness score** (`protein_selection`: implicit `"score"`).

This config uses **`"protein_selection": "random"`** instead. For random mode the loader **does not**:

- Apply **`remove_prey`** (no MS-threshold stripping of composite rows before protein choice).
- Apply **`filter_minimal_ids`** (no restriction of `spec_table` to proteins that appear in the composite table).

**SAINT = 0:** Preprocess and `model23_data_transformer` still **drop** rows with SAINT score 0 (same as base), so only positive-SAINT composite rows enter the model after Excel → TSV.

Then it **uniformly samples** `max_proteins` (e.g. 80 or 150) rows from the **full** `spec_table` index **without replacement**. After that, `composite_table` is restricted to bait–prey pairs where **both** `Prey` and `Bait` are in the sampled node set (induced subgraph for the model).

**Still applied (same as other runs):** columns are restricted to experimental replicates (`*_r*`), not controls, before correlation — so cosine similarity matches the rest of the pipeline.

The usual SAINT / correlation / model machinery then runs on the **reduced** protein set.

## Reproducibility

- Subset RNG seed defaults to the **MCMC `rseed`** for that run (13, 42, 71 → three different random subsets).
- Optional override: set `hyperparameters.random_proteins_seed` to the same integer for every seed if you want **identical** protein sets across chains.

## Outputs

Each run directory gets `random_protein_selection.txt` listing the chosen prey names.

## Run (mock, three seeds)

From the repo root:

```bash
cd script
python run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config_random80.json \
  --seeds 13 42 71
```

Default output base: `results/cullin_mock_config_random80_multi_seeds/` with `seed_13`, `seed_42`, `seed_71`.

For **WT** or **VIF**, copy `cullin_mock_config_random80.json`, point `data.input_file` / sheets to the corresponding Excel + sheet, set `data.filter` to `wt` or `vif`, and use a distinct `output` / multi-seed folder name.

## Random **150** proteins (mock only config provided)

Config: `cullin_mock_config_random150.json` (`max_proteins`: 150). Model has **M = 150×149/2 = 11 175** possible edges (vs 3160 for N=80), so runs are **much slower** and use **more GPU memory**.

```bash
cd script
mkdir -p ../results/cullin_mock_random150_custom
nohup python3 run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config_random150.json \
  --seeds 13 42 71 \
  --output-dir ../results/cullin_mock_random150_custom \
  > ../results/cullin_mock_random150_custom/nohup.log 2>&1 &
```

Add `--cpu` if you force CPU (see `run_multiple_seeds.py --help`). After completion: merge with `merge_and_analyze_convergence.py`, then `extract_all_pairs.py` pointing at `seed_13/0_model23_n_13_model_data.pkl` and the merged pickle.

Compare to base mock distribution with custom paths, e.g.:

```bash
python3 script/compare_probability_distribution_base_vs_random80.py \
  --base-tsv results/cullin_mock_config_multi_seeds/all_pairs.tsv \
  --random-tsv results/cullin_mock_random150_custom/all_pairs.tsv \
  --title "Mock infection: base vs random 150-protein subset" \
  --out-prefix probability_distribution_mock_base_vs_random150 \
  --out-dir validation_base_models/cullin_mock \
  --random-label "Random 150 (merged seeds)"
```

## Random mode, **all** prey proteins (`max_proteins`: null)

Set **`"max_proteins": null`** with **`"protein_selection": "random"`** to **skip** the random subsample step: the model uses **every row** in `spec_table` after experimental-column filtering (typically **~1.5k–1.6k** proteins for mock).

**Implications**

- **M ≈ N(N−1)/2** — for N≈1595, **M ≈ 1.27×10⁶** edges. JAX compile + each HMC step scale badly; you may hit **GPU OOM** or **multi-day** runs unless you cut `num_warmup` / `num_samples` for a pilot.
- Same random-mode data rules as above: **no `remove_prey`**, **no `filter_minimal_ids`** before building the graph.
- Config example: `cullin_mock_config_random_all.json` → output dir `results/cullin_mock_random_all_custom`.

```bash
cd script
mkdir -p ../results/cullin_mock_random_all_custom
# Strongly consider --test first or reduced samples in JSON
nohup python3 run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config_random_all.json \
  --seeds 13 42 71 \
  --output-dir ../results/cullin_mock_random_all_custom \
  > ../results/cullin_mock_random_all_custom/nohup.log 2>&1 &
```

**Note:** Omitting `max_proteins` from JSON does **not** mean “all proteins” — `run.py` defaults missing keys to **80**. You must set **`max_proteins` to `null` explicitly**.

**Alternative (same effect as “keep everyone”):** set `max_proteins` to any integer **≥** the number of prey rows (e.g. `10000`); `filter_proteins_random_subset` then keeps the full table and still applies the induced-subgraph composite restriction for that full node set.

## Stratified SAINT (`protein_selection`: `"stratified_saint"`)

Mix **high-** and **low-ranking** prey (by the **same combined SAINT + robustness score** as the base model), instead of uniform random from the full spec.

- Uses **base-style** steps: **`remove_prey`**, **`filter_minimal_ids`**, and the same ranking as **top-by-score**.
- Sorts all remaining prey by combined score (high → low), splits the list into **upper half** and **lower half** by rank (not by tied score).
- Draws **`max_proteins // 2`** uniformly from the upper half and **`max_proteins - max_proteins // 2`** from the lower half (without replacement within each half; fills from the rest if a half is too small).
- Restricts **`composite_table`** to bait–prey pairs with **both** ends in the chosen set (same induced subgraph as `random`).

Preprocess **drops SAINT = 0** for every run (including **`random`**). Stratified differs from **`random`** by also applying **`remove_prey`** and **`filter_minimal_ids`** before protein choice.

Example configs:
- `cullin_mock_config_stratified80.json` → `results/cullin_mock_stratified80_custom/` (N=80, M=3160)
- `cullin_mock_config_stratified150.json` → `results/cullin_mock_stratified150_custom/` (N=150)

```bash
cd script
mkdir -p ../results/cullin_mock_stratified80_custom
python3 run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config_stratified80.json \
  --seeds 13 42 71 \
  --output-dir ../results/cullin_mock_stratified80_custom

mkdir -p ../results/cullin_mock_stratified150_custom
python3 run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config_stratified150.json \
  --seeds 13 42 71 \
  --output-dir ../results/cullin_mock_stratified150_custom
```

Chosen proteins and tier labels (`TOP` / `BOT` / `FILL`) are written to `stratified_protein_selection.txt` under each run directory.

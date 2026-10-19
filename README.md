# INM HIV-1 CRL5 manuscript: data and scripts

Data, analysis scripts, and deposited results for:

> Kumar V, Palar A, Krogan NJ, Echeverria I, Sali A. Modeling networks of direct protein–protein contacts based on affinity purification–mass spectrometry data.

INM reads AP-MS spectral counts and SAINT scores and samples networks of direct protein contacts. The modeling code is the companion repository [vipul2732/integrative_network_modeling](https://github.com/vipul2732/integrative_network_modeling). The copy of that code under `script/` in this repository is the snapshot that produced the deposited results. Use this snapshot to reproduce the paper. Use `vipul2732/integrative_network_modeling` to run INM on a new dataset. 


## Repository layout

| Path | Contents |
| --- | --- |
| `input/cullin/` | AP-MS workbooks and JSON configs for HIV-1 WT, HIV-1 ΔVif, and mock |
| `input/cullin/1-s2.0-S1931312819302537-mmc2.xlsx` | Published supplementary table from Hüttenhain et al., Cell Host & Microbe 2019 |
| `script/` | Preprocessing, sampling, benchmark, and figure scripts |
| `results/cullin_{wt,vif,mock}_config_multi_seeds/` | Deposited chain summaries and `all_pairs.tsv` (posterior samples omitted; see below) |
| `validation_pdb_and_databases/` | BioGRID, CORUM, and PDB benchmark tables and figure panels |
| `validation_ablations/` | Full model versus leave-one-term-out ROC curves |
| `cost_validation/` | Scoring-function perturbation results |
| `post_sampling_analysis/base_models/` | Sampling precision, model precision, and main-cluster edge frequencies |
| `wt_vif_comparison/` | Louvain community networks for Figure 5 and Figure S10 |
| `Alphafold_multimer/` | AlphaFold 3 summary metrics for high- and low-probability pairs |
| `database/corum_humanComplexes.tsv` | CORUM table used in the benchmarks |
| `pdb/` | Direct-contact validation code and the deposited contact tables |

## Installation

```bash
conda env create -f environment.yml
conda activate inm-py310
python -c "import numpyro, jax, pandas, openpyxl, sklearn, networkx; print('ok', jax.devices())"
```

CPU-only JAX is installed by default. For CUDA 12, run `pip install "jax[cuda12]" jax-cuda12-plugin` after activating the environment.

PDB contact validation also needs FreeSASA and Biopython (`pip install biopython freesasa`). AlphaFold 3 metrics in this repository are the deposited summaries. Recomputing those structures requires access to AlphaFold 3, which is not redistributed here.

## Input data

Each condition is one workbook. The modeled sheet is `Table_S1A` (the first sheet). Columns are `Bait`, `PreyGene`, `SaintScore`, `Spec`, and `ctrlCounts`.

| Condition | Workbook | Config |
| --- | --- | --- |
| HIV-1 WT (Vif expressed) | `input/cullin/cullin_wt/cullin_wt.xlsx` | `input/cullin/cullin_wt/cullin_wt_config.json` |
| HIV-1 ΔVif | `input/cullin/cullin_vif/cullin_vif.xlsx` | `input/cullin/cullin_vif/cullin_vif_config.json` |
| Mock | `input/cullin/cullin_mock/cullin_mock.xlsx` | `input/cullin/cullin_mock/cullin_mock_config.json` |

The source experiment is Hüttenhain et al., Cell Host & Microbe 26:555–570.e7 (2019), supplementary file mmc2. The three workbooks above are that table split by infection condition.

Configs set `model23_n`, 1,000 warmup steps, 10,000 sampling steps, SAINT thresholds from 0.5 to 1.0, and indirect SAINT scores (geometric mean, minimum 0.3). `max_proteins` defaults to 80 in `script/run.py`. After the SAINT ≥ 0.5 filter, 77 proteins remain in each condition (2,926 pairs).

Ablation configs in the same folders set the model name to `model23_n_no_corr`, `model23_n_no_saint`, or `model23_n_no_priors`.

## Smoke test

This checks preprocessing and sampling. It does not reproduce the manuscript posteriors.

```bash
cd script
python run_multiple_seeds.py \
  --config ../input/cullin/cullin_mock/cullin_mock_config.json \
  --seeds 13 \
  --output-dir ../results/smoke_mock \
  --test \
  --cpu
```

`--test` uses 100 warmup steps and 500 samples. A finished run writes `results/smoke_mock/seed_13/config.json` and a pickle of samples.

## Reproduce the manuscript sampling

Run three chains for each condition. On a GPU this is hours per chain. On a CPU it is substantially longer. The first chain pays a one-time JAX compilation cost.

```bash
cd script
for cond in cullin_wt cullin_vif cullin_mock; do
  python run_multiple_seeds.py \
    --config ../input/cullin/${cond}/${cond}_config.json \
    --seeds 13 42 71 \
    --output-dir ../results/${cond}_config_multi_seeds \
    --cpu
done
```

Drop `--cpu` when JAX can see a GPU.

Then, for each condition, merge chains, write diagnostics, and export pair probabilities:

```bash
for run in cullin_wt_config_multi_seeds cullin_vif_config_multi_seeds cullin_mock_config_multi_seeds; do
  python merge_and_analyze_convergence.py ../results/${run}
  python extract_all_pairs.py \
    --results ../results/${run}/merged/merged_chains.pkl \
    --output ../results/${run}/all_pairs.tsv
done
```

Leave-one-term-out models (Figure 4) use `script/run_ablated_models.py`. Run it from the repository root. The script forces CPU sampling (`JAX_PLATFORMS=cpu`).

```bash
python script/run_ablated_models.py \
  --config input/cullin/cullin_wt/cullin_wt_config.json \
  --base-dataset cullin_wt_config_multi_seeds \
  --seeds 13 42 71 \
  --results-dir script/results
```

Repeat with the `cullin_vif` and `cullin_mock` configs and base-dataset names `cullin_vif_config_multi_seeds` and `cullin_mock_config_multi_seeds`. Outputs are written to `script/results/cullin_<condition>_no_corr`, `_no_saint`, and `_no_priors`.

### Files that are not in git

GitHub rejects files larger than 100 MB. These products of a full run are omitted:

- Posterior sample pickles (`.pkl`), including `merged/merged_chains.pkl`. The three base-model runs together are about 4 GB, and the ablation runs are larger.
- `binary_edges_full.npy` matrices from the ablation ensembles.
- The full BioGRID, IntAct, and STRING dumps, and the PDB mmCIF cache (`pdb/cache/`).

The tables and figures listed below were exported from those samples and are in the repository. Regenerating them from scratch requires the sampling commands above.

## Deposited results and manuscript figures

Pair probabilities for the consensus analysis, one row per protein pair:

- `results/cullin_wt_config_multi_seeds/all_pairs.tsv`
- `results/cullin_vif_config_multi_seeds/all_pairs.tsv`
- `results/cullin_mock_config_multi_seeds/all_pairs.tsv`

Columns are `Protein1`, `Protein2`, and `Probability`.

| Manuscript item | Where to look |
| --- | --- |
| Figure 2 and Figures S5–S6, database and PDB benchmarks | `validation_pdb_and_databases/`. Start with `validation_analysis_4panel_any_database.png`, `validation_analysis_4panel_direct_biogrid_corum_cluster1_wilson_lb.png`, `validation_analysis_4panel_pdb_direct_contact_cluster1_wilson_lb.png`, and `database_statistics_chart.png`. Pair-level calls are the `cullin_*_database_comparison_*.tsv` files. |
| Figure 3, score perturbations, HIV-1 WT | `cost_validation/cullin_wt_config_multi_seeds/cost_vs_accuracy.png` and `perturbation_results.tsv` |
| Figures S8 and S9 | `cost_validation/cullin_vif_config_multi_seeds/` and `cost_validation/cullin_mock_config_multi_seeds/` |
| Figure 4, full model versus ablations | `validation_ablations/roc_base_vs_ablations_3panel.png` and `roc_base_vs_ablations_summary.tsv`. Per-condition curves are `roc_base_vs_ablations_{wt,vif,mock}.png` and the TSVs in `roc_base_vs_ablations_curves/`. |
| Figure 5, WT and ΔVif communities | `wt_vif_comparison/vif_dependent_network_communities.png`, `vif_ko_network_communities.png`, `vif_dependent_communities.tsv`, `vif_ko_communities.tsv` |
| Figure S2–S4, sampling diagnostics | `results/cullin_{wt,vif,mock}_config_multi_seeds/merged/figure4_exhaustive_sampling.png` (also `cost_vs_steps.png` and `rhat_convergence.png`) |
| Figure S7, AlphaFold 3 | `Alphafold_multimer/alphafold_low_vs_high_probability_mock_wt_vif.png` and `af3_summary_metrics_with_probability_{wt,vif,mock}_{positive,negative}.csv` |
| Figure S10, mock communities | `wt_vif_comparison/mock_network_communities.png` and `mock_communities.tsv` |
| Sampling precision and model precision | `post_sampling_analysis/base_models/summary_all_base_models.tsv` |

Redraw the ROC panels from the deposited curve tables, without sampling:

```bash
cd script
python plot_roc_curves_from_saved_tsv.py
```

That script reads `validation_ablations/roc_base_vs_ablations_curves/*.tsv` and writes the per-condition ROC figures.

Other figure scripts in `script/` (`validation_analysis.py`, `plot_roc_base_vs_ablations.py`, `validate_model_cost_perturbations.py`, `plot_wt_vif_biological_subgraphs.py`, `pdb/validate_direct_contact.py`) rebuild the remaining panels from posterior samples or from the pair TSVs. Their `--help` text lists the input paths. The shell scripts `script/run_validation_analysis_pi_figure_layout.sh` and `pdb/run_all_datasets.sh` are the figure pipelines used for the database and PDB panels. They expect BioGRID and a PDB cache on local disk; see `database/README.md`.

## Reference databases

See [database/README.md](database/README.md). CORUM is included. BioGRID 5.0.253 and PDB structures are downloaded locally. The deposited comparison TSVs already contain the database labels used for the figures, so the enrichment and precision numbers can be checked without those downloads.

## Data citation

AP-MS measurements: Hüttenhain R, et al. Cell Host & Microbe 26, 555–570.e7 (2019). Please also follow the CORUM and BioGRID licenses when you reuse those files.

## License

Code in this repository is released under the MIT license. See [LICENSE](LICENSE). The AP-MS data remain subject to the terms of the original publication.

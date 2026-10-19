#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
DATABASE_DIR="$PROJECT_ROOT/database"
PDB_RESULTS_DIR="$PROJECT_ROOT/pdb_validation_all_cullin"
OUTPUT_DIR="$PROJECT_ROOT/validation_all_cullin_reframed"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define cullin datasets
declare -a datasets=(
    "cullin_mock_config_multi_seeds"
    "cullin_vif_config_multi_seeds"
    "cullin_wt_config_multi_seeds"
)

echo "============================================================"
echo "Running validation pipeline for all cullin datasets"
echo "============================================================"
echo ""

# Run validation for each dataset
for dataset in "${datasets[@]}"; do
    echo "============================================================"
    echo "Processing: $dataset"
    echo "============================================================"
    
    python "$SCRIPT_DIR/validate_all_pairs_with_databases.py" \
        --results-dir "$RESULTS_DIR" \
        --database-dir "$DATABASE_DIR" \
        --pdb-results "$PDB_RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$OUTPUT_DIR/$dataset"
    
    echo ""
done

echo "============================================================"
echo "Validation complete for all cullin datasets!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Now aggregating results..."
echo "============================================================"

# Aggregate results
python "$SCRIPT_DIR/aggregate_validation_results.py" \
    --validation-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/validation_summary_table.tsv" \
    --enrichment-output "$OUTPUT_DIR/enrichment_summary_table.tsv"

echo ""
echo "============================================================"
echo "All done! Summary tables saved to:"
echo "  - $OUTPUT_DIR/validation_summary_table.tsv"
echo "  - $OUTPUT_DIR/enrichment_summary_table.tsv"
echo "============================================================"

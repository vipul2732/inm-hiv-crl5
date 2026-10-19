#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$PROJECT_ROOT/results"
OUTPUT_BASE_DIR="$PROJECT_ROOT/cost_validation"

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Define cullin datasets
declare -a datasets=(
    "cullin_mock_config_multi_seeds"
    "cullin_vif_config_multi_seeds"
    "cullin_wt_config_multi_seeds"
)

echo "============================================================"
echo "Running cost validation for all cullin datasets"
echo "============================================================"
echo ""

# Run validation for each dataset
for dataset in "${datasets[@]}"; do
    echo "============================================================"
    echo "Processing: $dataset"
    echo "============================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$dataset"
    mkdir -p "$OUTPUT_DIR"
    
    python "$SCRIPT_DIR/validate_model_cost_perturbations.py" \
        --results-dir "$RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$OUTPUT_DIR"
    
    echo ""
done

echo "============================================================"
echo "Cost validation complete for all datasets!"
echo "Results saved to: $OUTPUT_BASE_DIR"
echo "============================================================"

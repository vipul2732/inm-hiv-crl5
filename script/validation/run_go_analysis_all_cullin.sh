#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define base directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_DIR="$PROJECT_ROOT/results"
OUTPUT_DIR="$PROJECT_ROOT/go_analysis_all_cullin"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define cullin datasets
declare -a datasets=(
    "cullin_mock_config_multi_seeds"
    "cullin_vif_config_multi_seeds"
    "cullin_wt_config_multi_seeds"
)

echo "============================================================"
echo "Running GO term analysis for all cullin datasets"
echo "============================================================"
echo ""

# Run GO analysis for each dataset
for dataset in "${datasets[@]}"; do
    echo "============================================================"
    echo "Processing: $dataset"
    echo "============================================================"
    
    python "$SCRIPT_DIR/analyze_go_terms_top100.py" \
        --results-dir "$RESULTS_DIR" \
        --dataset "$dataset" \
        --output "$OUTPUT_DIR/$dataset" \
        --delay 0.1
    
    echo ""
done

echo "============================================================"
echo "GO term analysis complete for all cullin datasets!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"

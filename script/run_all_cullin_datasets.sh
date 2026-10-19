#!/bin/bash
# Script to run Bayesian network inference on all cullin datasets sequentially
# For each dataset: select sheet 0, then run multiple seeds

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Datasets to process
DATASETS=(
    "cullin_mock"
    "cullin_vif"
    "cullin_wt"
)

echo "=========================================="
echo "Running analysis on all cullin datasets"
echo "=========================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing: $dataset"
    echo "=========================================="
    
    # Paths
    DATA_DIR="$PROJECT_ROOT/input/cullin/$dataset"
    XLSX_FILE="$DATA_DIR/${dataset}.xlsx"
    
    # Check if Excel file exists
    if [ ! -f "$XLSX_FILE" ]; then
        echo "ERROR: Excel file not found: $XLSX_FILE"
        echo "Skipping $dataset..."
        echo ""
        continue
    fi
    
    echo "Step 1: Selecting sheet 0 for $dataset..."
    cd "$SCRIPT_DIR"
    python select_sheets.py \
        --data "$XLSX_FILE" \
        --sheets 0 \
        --output-config "$DATA_DIR/${dataset}_config.json"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to select sheets for $dataset"
        echo "Skipping $dataset..."
        echo ""
        continue
    fi
    
    CONFIG_FILE="$DATA_DIR/${dataset}_config.json"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not created: $CONFIG_FILE"
        echo "Skipping $dataset..."
        echo ""
        continue
    fi
    
    echo "✓ Config file created: $CONFIG_FILE"
    echo ""
    
    echo "Step 2: Running multiple seeds for $dataset..."
    echo "This will run 3 seeds sequentially..."
    echo ""
    
    cd "$SCRIPT_DIR"
    python run_multiple_seeds.py \
        --config "$CONFIG_FILE" \
        --n-runs 3
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to run multiple seeds for $dataset"
        echo "Continuing to next dataset..."
        echo ""
        continue
    fi
    
    echo ""
    echo "✓ Completed: $dataset"
    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "=========================================="
echo "All datasets processed!"
echo "=========================================="



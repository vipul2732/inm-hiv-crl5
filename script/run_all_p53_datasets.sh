#!/bin/bash
# Script to run Bayesian network inference on all p53 datasets sequentially
# For each dataset: select sheet 0, then run multiple seeds

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Datasets to process
# Format: "subdirectory:xlsx_filename"
DATASETS=(
    "contact_mutation:p53_contact_mutations_only"
    "serine_mutation:Serine_mutants_only"
    "structural_mutation:p53_structural_destabilization_only"
    "wt:WT_only"
)

echo "=========================================="
echo "Running analysis on all p53 datasets"
echo "=========================================="
echo ""

for dataset_entry in "${DATASETS[@]}"; do
    # Parse subdirectory and filename
    IFS=':' read -r subdir filename <<< "$dataset_entry"
    
    echo "=========================================="
    echo "Processing: $subdir / $filename"
    echo "=========================================="
    
    # Paths
    DATA_DIR="$PROJECT_ROOT/input/p53/$subdir"
    XLSX_FILE="$DATA_DIR/${filename}.xlsx"
    
    # Check if Excel file exists
    if [ ! -f "$XLSX_FILE" ]; then
        echo "ERROR: Excel file not found: $XLSX_FILE"
        echo "Skipping $subdir/$filename..."
        echo ""
        continue
    fi
    
    echo "Step 1: Selecting sheet 0 for $filename..."
    cd "$SCRIPT_DIR"
    python select_sheets.py \
        --data "$XLSX_FILE" \
        --sheets 0 \
        --output-config "$DATA_DIR/${filename}_config.json"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to select sheets for $subdir/$filename"
        echo "Skipping $subdir/$filename..."
        echo ""
        continue
    fi
    
    CONFIG_FILE="$DATA_DIR/${filename}_config.json"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not created: $CONFIG_FILE"
        echo "Skipping $subdir/$filename..."
        echo ""
        continue
    fi
    
    echo "✓ Config file created: $CONFIG_FILE"
    echo ""
    
    echo "Step 2: Running multiple seeds for $filename..."
    echo "This will run 3 seeds sequentially..."
    echo ""
    
    cd "$SCRIPT_DIR"
    python run_multiple_seeds.py \
        --config "$CONFIG_FILE" \
        --n-runs 3
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to run multiple seeds for $subdir/$filename"
        echo "Continuing to next dataset..."
        echo ""
        continue
    fi
    
    echo ""
    echo "✓ Completed: $subdir/$filename"
    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "=========================================="
echo "All p53 datasets processed!"
echo "=========================================="



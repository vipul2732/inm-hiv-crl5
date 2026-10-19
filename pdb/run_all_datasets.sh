#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/pdb/validate_direct_contact.py"
VAL_DIR="$ROOT/validation_pdb_and_databases"
OUT_DIR="$ROOT/pdb/output"

mkdir -p "$OUT_DIR"

for dataset in cullin_wt cullin_mock cullin_vif; do
  echo "=== $dataset ==="
  python3 "$SCRIPT" \
    --input "$VAL_DIR/${dataset}_pdb_validation.tsv" \
    --output "$OUT_DIR/${dataset}_pdb_contact_validation.tsv" \
    --pdb-cache "$VAL_DIR/${dataset}_pdb_validation_pdb_cache.json" \
    --cache-dir "$ROOT/pdb/cache" \
    --bsa-threshold 400 \
    --max-pdbs-per-pair 20
done

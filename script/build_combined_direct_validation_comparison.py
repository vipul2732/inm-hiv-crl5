#!/usr/bin/env python3
"""
Merge BioGRID direct + CORUM + PDB direct-contact labels into one comparison TSV per condition.

Requires prior outputs from compare_with_databases_simple.py (direct_biogrid_corum suffix)
and build_pdb_direct_contact_comparison.py (pdb_direct_contact suffix).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build combined direct-evidence comparison TSVs (BioGRID+CORUM+PDB direct)."
    )
    ap.add_argument("--input-dir", type=Path, default=Path("validation_pdb_and_databases"))
    ap.add_argument(
        "--biogrid-corum-suffix",
        type=str,
        default="direct_biogrid_corum",
    )
    ap.add_argument(
        "--pdb-suffix",
        type=str,
        default="pdb_direct_contact",
    )
    ap.add_argument(
        "--output-suffix",
        type=str,
        default="combined_direct",
    )
    args = ap.parse_args()

    for ds in ("wt", "mock", "vif"):
        bg_path = args.input_dir / f"cullin_{ds}_database_comparison_{args.biogrid_corum_suffix}.tsv"
        pdb_path = args.input_dir / f"cullin_{ds}_database_comparison_{args.pdb_suffix}.tsv"
        if not bg_path.is_file() or not pdb_path.is_file():
            logger.warning("Missing inputs for %s: %s / %s", ds, bg_path, pdb_path)
            continue

        bg = pd.read_csv(bg_path, sep="\t")
        pdb = pd.read_csv(pdb_path, sep="\t")
        key_cols = ["Protein1_UniProt", "Protein2_UniProt"]
        merged = bg.merge(
            pdb[key_cols + ["In_PDB_Direct_Contact", "Contact_Validation_Status"]].rename(
                columns={"Contact_Validation_Status": "PDB_Contact_Validation_Status"}
            ),
            on=key_cols,
            how="left",
        )
        merged["In_PDB_Direct_Contact"] = merged["In_PDB_Direct_Contact"].fillna(False).astype(bool)
        merged["In_Combined_Direct_Evidence"] = (
            merged["In_BioGRID"].fillna(False).astype(bool)
            | merged["In_CORUM"].fillna(False).astype(bool)
            | merged["In_PDB_Direct_Contact"]
        )
        merged["In_Any_Database"] = merged["In_Combined_Direct_Evidence"]

        out_path = args.input_dir / f"cullin_{ds}_database_comparison_{args.output_suffix}.tsv"
        merged.to_csv(out_path, sep="\t", index=False)
        n = int(merged["In_Combined_Direct_Evidence"].sum())
        logger.info(
            "%s: %d combined direct positives (BioGRID=%d, CORUM=%d, PDB direct=%d) -> %s",
            ds.upper(),
            n,
            int(merged["In_BioGRID"].sum()),
            int(merged["In_CORUM"].sum()),
            int(merged["In_PDB_Direct_Contact"].sum()),
            out_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Build database-comparison TSVs from PDB BSA contact validation output.

Known positive = direct interface only (BSA >= threshold), including homolog recovery:
  - Direct_Contact_Experimental
  - Direct_Contact_Homolog
  - Direct_Contact_Structural_Recovery

Excluded (co-complex without sufficient BSA, no evidence, etc.):
  - Co_Complex_No_Contact
  - Homolog_Co_Complex_No_Contact
  - Structural_Recovery_No_Contact
  - No_Structural_Evidence, Shared_PDB_No_Chain_Map, ...

Example:
  python script/build_pdb_direct_contact_comparison.py \\
    --pdb-output-dir pdb/output \\
    --output-dir validation_pdb_and_databases
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIRECT_CONTACT_STATUSES = frozenset(
    {
        "Direct_Contact_Experimental",
        "Direct_Contact_Homolog",
        "Direct_Contact_Structural_Recovery",
    }
)

EXPANDED_PDB_STATUSES = frozenset(
    {
        "Direct_Contact_Experimental",
        "Direct_Contact_Homolog",
        "Direct_Contact_Structural_Recovery",
        "Co_Complex_No_Contact",
        "Homolog_Co_Complex_No_Contact",
        "Structural_Recovery_No_Contact",
    }
)


def build_comparison_df(contact_df: pd.DataFrame, mode: str = "direct") -> pd.DataFrame:
    status = contact_df["Contact_Validation_Status"].astype(str)
    if mode == "direct":
        positive = status.isin(DIRECT_CONTACT_STATUSES)
        positive_col = "In_PDB_Direct_Contact"
    elif mode == "expanded":
        positive = status.isin(EXPANDED_PDB_STATUSES)
        positive_col = "In_PDB_Structural_Evidence"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = pd.DataFrame(
        {
            "Protein1": contact_df["Protein1"],
            "Protein1_UniProt": contact_df["Protein1_UniProt"],
            "Protein2": contact_df["Protein2"],
            "Protein2_UniProt": contact_df["Protein2_UniProt"],
            "Probability": contact_df["Probability"],
            "Has_Shared_PDB": contact_df.get("Has_Shared_PDB", False),
            "Contact_Validation_Status": status,
            "In_PDB_Direct_Experimental": status == "Direct_Contact_Experimental",
            "In_PDB_Direct_Homolog": status == "Direct_Contact_Homolog",
            "In_PDB_Direct_Structural_Recovery": status == "Direct_Contact_Structural_Recovery",
            "In_PDB_Co_Complex": status == "Co_Complex_No_Contact",
            "In_PDB_Homolog_Co_Complex": status == "Homolog_Co_Complex_No_Contact",
            "In_PDB_Structural_Recovery_No_Contact": status == "Structural_Recovery_No_Contact",
            positive_col: positive,
            "In_Any_Database": positive,
        }
    )
    if mode == "direct":
        out["In_PDB_Structural_Evidence"] = status.isin(EXPANDED_PDB_STATUSES)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build comparison TSVs from PDB direct-contact validation output."
    )
    ap.add_argument(
        "--pdb-output-dir",
        type=Path,
        default=Path("pdb/output"),
        help="Directory with cullin_*_pdb_contact_validation.tsv files",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_pdb_and_databases"),
        help="Where to write cullin_*_database_comparison_pdb_direct_contact.tsv",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Output filename suffix (default: pdb_direct_contact or pdb_structural_evidence by mode)",
    )
    ap.add_argument(
        "--mode",
        choices=("direct", "expanded"),
        default="direct",
        help=(
            "direct: BSA>=threshold only; expanded: direct + co-complex (experimental/homolog) "
            "+ structural recovery no-contact"
        ),
    )
    args = ap.parse_args()

    suffix = args.suffix or (
        "pdb_direct_contact" if args.mode == "direct" else "pdb_structural_evidence"
    )
    positive_col = (
        "In_PDB_Direct_Contact" if args.mode == "direct" else "In_PDB_Structural_Evidence"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for ds in ("wt", "mock", "vif"):
        inp = args.pdb_output_dir / f"cullin_{ds}_pdb_contact_validation.tsv"
        if not inp.is_file():
            logger.warning("Missing: %s", inp)
            continue

        contact_df = pd.read_csv(inp, sep="\t")
        out_df = build_comparison_df(contact_df, mode=args.mode)
        out_path = args.output_dir / f"cullin_{ds}_database_comparison_{suffix}.tsv"
        out_df.to_csv(out_path, sep="\t", index=False)

        n_pos = int(out_df[positive_col].sum())
        logger.info(
            "%s [%s]: %d positives -> %s",
            ds.upper(),
            args.mode,
            n_pos,
            out_path,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

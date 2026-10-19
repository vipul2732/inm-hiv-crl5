#!/usr/bin/env python3
"""
Filter BioGRID PSI-MITAB to direct physical contact interactions only.

Keeps rows whose Interaction Types field is PSI-MI direct interaction (MI:0407).
Optionally restricts to human-human pairs (taxid:9606), matching other project loaders.

Example:
  python script/filter_biogrid_direct_contact.py \\
    --input database/BIOGRID-ALL-5.0.253.mitab.txt \\
    --output database/BIOGRID-ALL-5.0.253.direct_contact.mitab.txt
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIRECT_INTERACTION_MARKERS = (
    'psi-mi:"MI:0407"(direct interaction)',
    "MI:0407",
    "(direct interaction)",
)
UNIPROT_PATTERN = re.compile(r"uniprot/(?:swiss-prot|trEMBL):([A-Z0-9-]+)")


def is_direct_interaction(interaction_types: str) -> bool:
    text = interaction_types.strip().lower()
    return any(marker.lower() in text for marker in DIRECT_INTERACTION_MARKERS)


def is_human_pair(taxid_a: str, taxid_b: str) -> bool:
    return taxid_a.strip() == "taxid:9606" and taxid_b.strip() == "taxid:9606"


def extract_uniprot_pairs(alt_a: str, alt_b: str) -> set[tuple[str, str]]:
    ids_a = set(UNIPROT_PATTERN.findall(alt_a))
    ids_b = set(UNIPROT_PATTERN.findall(alt_b))
    pairs: set[tuple[str, str]] = set()
    for up_a in ids_a:
        for up_b in ids_b:
            if up_a and up_b and up_a != up_b:
                pairs.add(tuple(sorted((up_a, up_b))))
    return pairs


def filter_biogrid_mitab(
    input_file: Path,
    output_file: Path,
    human_only: bool = True,
) -> dict:
    input_file = input_file.resolve()
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    unique_pairs: set[tuple[str, str]] = set()

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, open(
        output_file, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue

            total_rows += 1
            if not is_direct_interaction(fields[11]):
                continue
            if human_only and not is_human_pair(fields[9], fields[10]):
                continue

            fout.write(line)
            kept_rows += 1
            unique_pairs.update(extract_uniprot_pairs(fields[2], fields[3]))

            if kept_rows % 50000 == 0:
                logger.info("  kept %d rows so far...", kept_rows)

    stats = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "human_only": human_only,
        "interaction_type": "direct interaction (PSI-MI:MI:0407)",
        "total_data_rows_read": total_rows,
        "rows_kept": kept_rows,
        "unique_uniprot_pairs": len(unique_pairs),
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter BioGRID MITAB to direct interaction (MI:0407) rows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("database/BIOGRID-ALL-5.0.253.mitab.txt"),
        help="Input BioGRID MITAB file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("database/BIOGRID-ALL-5.0.253.direct_contact.mitab.txt"),
        help="Output filtered MITAB file",
    )
    parser.add_argument(
        "--all-species",
        action="store_true",
        help="Keep all species (default: human-human only, taxid:9606)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    logger.info("Filtering %s", args.input)
    stats = filter_biogrid_mitab(
        args.input,
        args.output,
        human_only=not args.all_species,
    )

    logger.info("Done.")
    logger.info("  Rows read: %d", stats["total_data_rows_read"])
    logger.info("  Rows kept: %d", stats["rows_kept"])
    logger.info("  Unique UniProt pairs: %d", stats["unique_uniprot_pairs"])
    logger.info("  Wrote: %s", stats["output_file"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

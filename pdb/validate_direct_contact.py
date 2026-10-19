#!/usr/bin/env python3
"""
Validate protein pairs for direct PDB interface evidence using buried surface area.

Pipeline per pair:
1. Shared PDB: map chains, compute BSA, label Direct_Contact_Experimental.
2. No shared PDB: homolog search in partner PDBs via RCSB sequence + entity alignment.
3. Label Direct_Contact_Homolog when homolog interface BSA passes threshold.

Usage:
    python pdb/validate_direct_contact.py \\
        --input validation_pdb_and_databases/cullin_wt_pdb_validation.tsv \\
        --output pdb/output/cullin_wt_pdb_contact_validation.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from chain_mapping import chain_pairs_for_uniprots
from homolog import evaluate_pair_homolog, load_json_cache
from interface_bsa import InterfaceResult, has_direct_interface
from structure_io import download_structure, load_structure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATUS_DIRECT = "Direct_Contact_Experimental"
STATUS_HOMOLOG = "Direct_Contact_Homolog"
STATUS_STRUCTURAL_RECOVERY = "Direct_Contact_Structural_Recovery"
STATUS_NO_CONTACT = "Co_Complex_No_Contact"
STATUS_HOMOLOG_NO_CONTACT = "Homolog_Co_Complex_No_Contact"
STATUS_STRUCTURAL_NO_CONTACT = "Structural_Recovery_No_Contact"
STATUS_NO_EVIDENCE = "No_Structural_Evidence"
STATUS_MISSING_UNIPROT = "Missing_UniProt_ID"
STATUS_NO_CHAIN_MAP = "Shared_PDB_No_Chain_Map"
STATUS_STRUCTURE_FAIL = "Shared_PDB_Structure_Unavailable"


def parse_shared_pdb_ids(raw: object) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [p.strip().upper() for p in text.split(",") if p.strip()]


def resolve_pdb_cache_path(input_file: Path, pdb_cache_arg: Optional[Path]) -> Optional[Path]:
    if pdb_cache_arg is not None:
        return pdb_cache_arg
    auto = input_file.parent / f"{input_file.stem}_pdb_cache.json"
    return auto if auto.exists() else None


def evaluate_pair_shared_pdbs(
    shared_pdbs: List[str],
    uniprot_a: str,
    uniprot_b: str,
    session: requests.Session,
    chain_cache: Dict,
    chain_cache_path: Path,
    structure_cache_dir: Path,
    bsa_threshold: float,
    max_pdbs_per_pair: int,
) -> Tuple[str, Dict]:
    best_no_contact: Optional[Dict] = None

    for pdb_id in shared_pdbs[:max_pdbs_per_pair]:
        chain_pairs = chain_pairs_for_uniprots(
            pdb_id, uniprot_a, uniprot_b, session, chain_cache, chain_cache_path
        )
        if not chain_pairs:
            continue

        struct_path = download_structure(pdb_id, structure_cache_dir, session)
        if struct_path is None:
            continue

        try:
            structure = load_structure(struct_path)
            model = structure[0]
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", pdb_id, exc)
            continue

        for chain_a, chain_b in chain_pairs:
            ok, result = has_direct_interface(model, chain_a, chain_b, bsa_threshold=bsa_threshold)
            if result is None:
                continue

            record = _contact_record(
                pdb_id=pdb_id,
                chain_a=chain_a,
                chain_b=chain_b,
                result=result,
                bsa_threshold=bsa_threshold,
                evidence_type="experimental",
            )

            if ok:
                record["Contact_Validation_Status"] = STATUS_DIRECT
                return STATUS_DIRECT, record

            if best_no_contact is None or result.bsa_ang2 > best_no_contact["Interface_BSA_Ang2"]:
                record["Contact_Validation_Status"] = STATUS_NO_CONTACT
                best_no_contact = record

    if best_no_contact is not None:
        return STATUS_NO_CONTACT, best_no_contact

    if not shared_pdbs:
        return STATUS_NO_EVIDENCE, _empty_record(evidence_type="none")

    had_chain_map = False
    for pdb_id in shared_pdbs[:max_pdbs_per_pair]:
        if chain_pairs_for_uniprots(pdb_id, uniprot_a, uniprot_b, session, chain_cache, chain_cache_path):
            had_chain_map = True
            break

    if not had_chain_map:
        return STATUS_NO_CHAIN_MAP, _empty_record(evidence_type="experimental")

    return STATUS_STRUCTURE_FAIL, _empty_record(evidence_type="experimental")


def _empty_record(evidence_type: str) -> Dict:
    return {
        "Contact_Validation_Status": STATUS_NO_EVIDENCE,
        "Contact_Evidence_Type": evidence_type,
        "Contact_PDB_ID": "",
        "Contact_Chain_A": "",
        "Contact_Chain_B": "",
        "Interface_BSA_Ang2": "",
        "SASA_Chain_A": "",
        "SASA_Chain_B": "",
        "SASA_Complex": "",
        "BSA_Threshold_Ang2": "",
        "SASA_Method": "",
        "PDBs_Checked": 0,
        "Homolog_Query_UniProt": "",
        "Homolog_Experimental_UniProt": "",
        "Homolog_Chain_UniProt": "",
        "Homolog_Sequence_Identity_Pct": "",
        "Homolog_Alignment_Length": "",
        "Homolog_Entity_ID": "",
    }


def _contact_record(
    pdb_id: str,
    chain_a: str,
    chain_b: str,
    result: InterfaceResult,
    bsa_threshold: float,
    evidence_type: str,
) -> Dict:
    record = _empty_record(evidence_type=evidence_type)
    record.update(
        {
            "Contact_PDB_ID": pdb_id,
            "Contact_Chain_A": chain_a,
            "Contact_Chain_B": chain_b,
            "Interface_BSA_Ang2": round(result.bsa_ang2, 2),
            "SASA_Chain_A": round(result.sasa_chain_a, 2),
            "SASA_Chain_B": round(result.sasa_chain_b, 2),
            "SASA_Complex": round(result.sasa_complex, 2),
            "BSA_Threshold_Ang2": bsa_threshold,
            "SASA_Method": result.method,
            "PDBs_Checked": 1,
        }
    )
    return record


def validate_dataframe(
    df: pd.DataFrame,
    output_file: Path,
    cache_root: Path,
    pdb_cache: Dict[str, List[str]],
    bsa_threshold: float = 400.0,
    max_pdbs_per_pair: int = 20,
    only_shared: bool = False,
    enable_homolog: bool = True,
    homolog_min_identity_pct: float = 30.0,
    homolog_min_alignment_length: int = 50,
    homolog_search_identity_cutoff: float = 0.3,
) -> pd.DataFrame:
    chain_cache_path = cache_root / "chain_mapping.json"
    structure_cache_dir = cache_root / "structures"
    uniprot_seq_cache_path = cache_root / "uniprot_sequences.json"
    entity_seq_cache_path = cache_root / "entity_sequences.json"
    search_cache_path = cache_root / "rcsb_sequence_search.json"

    chain_cache = load_json_cache(chain_cache_path)
    uniprot_seq_cache = load_json_cache(uniprot_seq_cache_path)
    entity_seq_cache = load_json_cache(entity_seq_cache_path)
    search_cache = load_json_cache(search_cache_path)

    logger.info("Loaded chain mapping cache (%d PDB entries)", len(chain_cache))
    if pdb_cache:
        logger.info("Loaded UniProt PDB cache (%d accessions)", len(pdb_cache))

    session = requests.Session()
    session.headers.update({"User-Agent": "pdb-direct-contact-validator/1.0 (research)"})

    rows = []
    total = len(df)
    shared_count = int(df.get("Has_Shared_PDB", pd.Series([False] * total)).sum())

    for idx, row in df.iterrows():
        if (idx + 1) % 25 == 0 or idx == 0:
            logger.info("Processing pair %d/%d", idx + 1, total)

        u1 = str(row.get("Protein1_UniProt", "")).strip()
        u2 = str(row.get("Protein2_UniProt", "")).strip()
        has_shared = bool(row.get("Has_Shared_PDB", False))
        shared_pdbs = parse_shared_pdb_ids(row.get("Shared_PDB_IDs", ""))

        base = row.to_dict()

        if not u1 or not u2 or u1 == "nan" or u2 == "nan":
            base.update(_empty_record(evidence_type="none"))
            base["Contact_Validation_Status"] = STATUS_MISSING_UNIPROT
            rows.append(base)
            continue

        if only_shared and not has_shared:
            base.update(_empty_record(evidence_type="none"))
            base["Contact_Validation_Status"] = STATUS_NO_EVIDENCE
            rows.append(base)
            continue

        if has_shared and shared_pdbs:
            status, contact_info = evaluate_pair_shared_pdbs(
                shared_pdbs=shared_pdbs,
                uniprot_a=u1,
                uniprot_b=u2,
                session=session,
                chain_cache=chain_cache,
                chain_cache_path=chain_cache_path,
                structure_cache_dir=structure_cache_dir,
                bsa_threshold=bsa_threshold,
                max_pdbs_per_pair=max_pdbs_per_pair,
            )
            contact_info["Contact_Validation_Status"] = status
            contact_info["PDBs_Checked"] = min(len(shared_pdbs), max_pdbs_per_pair)
            base.update(contact_info)
            rows.append(base)
            continue

        if enable_homolog and pdb_cache:
            homolog_status, homolog_info = evaluate_pair_homolog(
                uniprot_a=u1,
                uniprot_b=u2,
                pdb_cache=pdb_cache,
                session=session,
                chain_cache=chain_cache,
                chain_cache_path=chain_cache_path,
                structure_cache_dir=structure_cache_dir,
                uniprot_seq_cache=uniprot_seq_cache,
                uniprot_seq_cache_path=uniprot_seq_cache_path,
                entity_seq_cache=entity_seq_cache,
                entity_seq_cache_path=entity_seq_cache_path,
                search_cache=search_cache,
                search_cache_path=search_cache_path,
                bsa_threshold=bsa_threshold,
                min_identity_pct=homolog_min_identity_pct,
                min_alignment_length=homolog_min_alignment_length,
                search_identity_cutoff=homolog_search_identity_cutoff,
                max_pdbs_per_pair=max_pdbs_per_pair,
            )
            if homolog_status:
                homolog_info["Contact_Validation_Status"] = homolog_status
                base.update(homolog_info)
                rows.append(base)
                continue

            if homolog_info:
                homolog_info.setdefault("Contact_Validation_Status", STATUS_NO_EVIDENCE)
                homolog_info.setdefault("Contact_Evidence_Type", "homolog")
                base.update(homolog_info)
                rows.append(base)
                continue

        base.update(_empty_record(evidence_type="none"))
        base["Contact_Validation_Status"] = STATUS_NO_EVIDENCE
        rows.append(base)

    out_df = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, sep="\t", index=False)

    direct = int((out_df["Contact_Validation_Status"] == STATUS_DIRECT).sum())
    homolog = int((out_df["Contact_Validation_Status"] == STATUS_HOMOLOG).sum())
    recovered = int((out_df["Contact_Validation_Status"] == STATUS_STRUCTURAL_RECOVERY).sum())
    no_contact = int((out_df["Contact_Validation_Status"] == STATUS_NO_CONTACT).sum())
    homolog_no_contact = int((out_df["Contact_Validation_Status"] == STATUS_HOMOLOG_NO_CONTACT).sum())
    structural_no_contact = int((out_df["Contact_Validation_Status"] == STATUS_STRUCTURAL_NO_CONTACT).sum())

    logger.info("Validation complete")
    logger.info("  Total pairs: %d", len(out_df))
    logger.info("  Input pairs with shared PDB: %d", shared_count)
    logger.info("  Direct_Contact_Experimental: %d", direct)
    logger.info("  Direct_Contact_Structural_Recovery: %d", recovered)
    logger.info("  Direct_Contact_Homolog: %d", homolog)
    logger.info("  Co_Complex_No_Contact: %d", no_contact)
    logger.info("  Homolog_Co_Complex_No_Contact: %d", homolog_no_contact)
    logger.info("  Structural_Recovery_No_Contact: %d", structural_no_contact)
    logger.info("  Saved: %s", output_file)
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description="PDB direct-contact validation via buried surface area")
    parser.add_argument("--input", type=Path, required=True, help="Input TSV from validate_with_pdb.py")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV with contact validation")
    parser.add_argument(
        "--pdb-cache",
        type=Path,
        help="UniProt->PDB JSON cache (default: <input_stem>_pdb_cache.json beside input)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cache",
        help="Cache directory for structures and chain mappings",
    )
    parser.add_argument("--bsa-threshold", type=float, default=400.0, help="Minimum BSA (Å²) for direct contact")
    parser.add_argument(
        "--max-pdbs-per-pair",
        type=int,
        default=20,
        help="Max PDB structures to check per pair",
    )
    parser.add_argument(
        "--only-shared",
        action="store_true",
        help="Only run interface checks on pairs that already have shared PDB IDs",
    )
    parser.add_argument(
        "--no-homolog",
        action="store_true",
        help="Disable homolog fallback for pairs without shared PDB",
    )
    parser.add_argument(
        "--homolog-min-identity",
        type=float,
        default=30.0,
        help="Minimum sequence identity (%%) for homolog chain in PDB",
    )
    parser.add_argument(
        "--homolog-min-alignment",
        type=int,
        default=50,
        help="Minimum alignment length for homolog chain match",
    )
    parser.add_argument(
        "--homolog-search-identity",
        type=float,
        default=0.3,
        help="RCSB sequence search identity cutoff (0-1)",
    )
    args = parser.parse_args()

    pdb_cache_path = resolve_pdb_cache_path(args.input, args.pdb_cache)
    pdb_cache: Dict[str, List[str]] = {}
    if pdb_cache_path and pdb_cache_path.exists():
        logger.info("Loading PDB cache from %s", pdb_cache_path)
        pdb_cache = load_json_cache(pdb_cache_path)
    elif not args.no_homolog:
        logger.warning("PDB cache not found; homolog fallback will be skipped")

    logger.info("Loading pairs from %s", args.input)
    df = pd.read_csv(args.input, sep="\t")
    validate_dataframe(
        df,
        args.output,
        cache_root=args.cache_dir,
        pdb_cache=pdb_cache,
        bsa_threshold=args.bsa_threshold,
        max_pdbs_per_pair=args.max_pdbs_per_pair,
        only_shared=args.only_shared,
        enable_homolog=not args.no_homolog,
        homolog_min_identity_pct=args.homolog_min_identity,
        homolog_min_alignment_length=args.homolog_min_alignment,
        homolog_search_identity_cutoff=args.homolog_search_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

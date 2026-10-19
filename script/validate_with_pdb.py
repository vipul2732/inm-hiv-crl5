#!/usr/bin/env python3
"""
Validate protein pairs by checking for shared PDB structures in UniProt.

This script:
1. Reads protein pairs with UniProt IDs
2. For each UniProt ID, queries UniProt REST to obtain PDB cross-references
3. Checks if both proteins share any common PDB IDs (co-occurrence in a structure)
4. Saves validation results

Usage:
    python validate_with_pdb.py \
        --input results/cullin_wt_config_multi_seeds/all_pairs_with_uniprot.tsv \
        --output validation_pdb_and_databases/cullin_wt_pdb_validation.tsv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Set, Tuple, Dict, Any

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def normalize_uniprot_id(raw: Any, keep_isoform: bool = True) -> str:
    """Return a cleaned UniProt accession string or empty string."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # If the TSV accidentally contains multiple IDs in one field, keep the first.
    # (You can change this to split on ',' as well if needed.)
    for sep in [";", " ", "|", ","]:
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    if not keep_isoform and "-" in s:
        s = s.split("-", 1)[0].strip()
    return s


def extract_pdb_ids_from_uniprot_json(data: Dict[str, Any]) -> Set[str]:
    """
    UniProt JSON commonly stores PDB xrefs in `uniProtKBCrossReferences`.
    Each entry looks like: {"database":"PDB","id":"1ABC", ...}
    """
    pdb_ids: Set[str] = set()

    xrefs = data.get("uniProtKBCrossReferences", [])
    if isinstance(xrefs, list):
        for x in xrefs:
            try:
                if x.get("database") == "PDB":
                    pdb_id = x.get("id")
                    if pdb_id:
                        pdb_ids.add(str(pdb_id).upper())
            except AttributeError:
                # In case a malformed xref is not a dict
                continue

    return pdb_ids


def get_pdb_ids_from_uniprot(
    session: requests.Session,
    uniprot_id: str,
    retry_count: int = 4,
    base_wait: float = 1.0,
) -> Set[str]:
    """
    Get PDB IDs for a UniProt accession using UniProt REST.

    Endpoint:
      https://rest.uniprot.org/uniprotkb/{accession}.json

    Returns:
      Set of PDB IDs (e.g., {"1ABC","2DEF"})
    """
    uniprot_id = normalize_uniprot_id(uniprot_id, keep_isoform=True)
    if not uniprot_id:
        return set()

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    for attempt in range(retry_count):
        try:
            resp = session.get(url, timeout=20)

            if resp.status_code == 200:
                data = resp.json()
                pdb_ids = extract_pdb_ids_from_uniprot_json(data)
                return pdb_ids

            if resp.status_code == 404:
                logger.debug(f"UniProt ID not found: {uniprot_id}")
                return set()

            # Rate limiting / transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = base_wait * (2 ** attempt)
                logger.debug(
                    f"HTTP {resp.status_code} for {uniprot_id} (attempt {attempt+1}/{retry_count}); "
                    f"waiting {wait:.1f}s"
                )
                time.sleep(wait)
                continue

            # Other non-success codes: treat as failure for this ID
            logger.debug(f"HTTP {resp.status_code} for {uniprot_id}; giving up.")
            return set()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = base_wait * (2 ** attempt)
            logger.debug(
                f"Network error for {uniprot_id} (attempt {attempt+1}/{retry_count}): {e}; "
                f"waiting {wait:.1f}s"
            )
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            logger.debug(f"RequestException for {uniprot_id}: {e}; giving up.")
            return set()

    return set()


def check_shared_pdb(pdb_ids_1: Set[str], pdb_ids_2: Set[str]) -> Tuple[bool, Set[str]]:
    shared = pdb_ids_1 & pdb_ids_2
    return (len(shared) > 0), shared


def validate_pairs_with_pdb(input_file: Path, output_file: Path, cache_file: Optional[Path]) -> pd.DataFrame:
    logger.info(f"Loading pairs from {input_file}")
    df = pd.read_csv(input_file, sep="\t")

    # Defensive: ensure expected columns exist
    required_cols = ["Protein1", "Protein2", "Protein1_UniProt", "Protein2_UniProt", "Probability"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input TSV: {missing}")

    # Normalize UniProt IDs early
    df["Protein1_UniProt"] = df["Protein1_UniProt"].apply(lambda x: normalize_uniprot_id(x, keep_isoform=True))
    df["Protein2_UniProt"] = df["Protein2_UniProt"].apply(lambda x: normalize_uniprot_id(x, keep_isoform=True))

    # Load cache
    pdb_cache: Dict[str, list] = {}
    if cache_file and cache_file.exists():
        logger.info(f"Loading PDB cache from {cache_file}")
        with open(cache_file, "r") as f:
            pdb_cache = json.load(f)
        logger.info(f"Loaded {len(pdb_cache)} cached UniProt -> PDB mappings")

    # Unique UniProt IDs to query
    unique_uniprot_ids = sorted(
        set(df["Protein1_UniProt"].dropna().tolist()) | set(df["Protein2_UniProt"].dropna().tolist())
    )
    unique_uniprot_ids = [u for u in unique_uniprot_ids if u]  # drop empty strings

    logger.info(f"Found {len(unique_uniprot_ids)} unique UniProt IDs to query")

    # Use a single session for efficiency
    session = requests.Session()
    session.headers.update({"User-Agent": "pdb-pair-validator/1.0 (research)"})

    logger.info("Querying UniProt for PDB cross-references...")
    for i, uid in enumerate(unique_uniprot_ids, 1):
        if uid in pdb_cache:
            continue
        if i % 25 == 0:
            logger.info(f"Progress: {i}/{len(unique_uniprot_ids)} UniProt IDs processed...")

        pdb_ids = get_pdb_ids_from_uniprot(session, uid)
        pdb_cache[uid] = sorted(pdb_ids)

        # Gentle pacing (UniProt is generally fine with reasonable throughput)
        time.sleep(0.05)

    # Save cache
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(pdb_cache, f, indent=2)
        logger.info(f"Saved PDB cache to {cache_file}")

    logger.info("Validating pairs for shared PDB structures...")
    results = []

    for _, row in df.iterrows():
        u1 = row["Protein1_UniProt"]
        u2 = row["Protein2_UniProt"]

        if not u1 or not u2:
            results.append(
                {
                    "Protein1": row["Protein1"],
                    "Protein1_UniProt": u1,
                    "Protein2": row["Protein2"],
                    "Protein2_UniProt": u2,
                    "Probability": row["Probability"],
                    "Has_Shared_PDB": False,
                    "Shared_PDB_Count": 0,
                    "Shared_PDB_IDs": "",
                    "Protein1_PDB_Count": 0,
                    "Protein2_PDB_Count": 0,
                    "Validation_Status": "Missing_UniProt_ID",
                }
            )
            continue

        pdb1 = set(pdb_cache.get(u1, []))
        pdb2 = set(pdb_cache.get(u2, []))
        has_shared, shared = check_shared_pdb(pdb1, pdb2)

        results.append(
            {
                "Protein1": row["Protein1"],
                "Protein1_UniProt": u1,
                "Protein2": row["Protein2"],
                "Protein2_UniProt": u2,
                "Probability": row["Probability"],
                "Has_Shared_PDB": has_shared,
                "Shared_PDB_Count": len(shared),
                "Shared_PDB_IDs": ",".join(sorted(shared)) if shared else "",
                "Protein1_PDB_Count": len(pdb1),
                "Protein2_PDB_Count": len(pdb2),
                "Validation_Status": "Validated" if has_shared else "No_Shared_PDB",
            }
        )

    results_df = pd.DataFrame(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, sep="\t", index=False)

    # Stats: count only non-empty UniProt IDs
    mask_both_uniprot = (results_df["Protein1_UniProt"].astype(str).str.len() > 0) & (
        results_df["Protein2_UniProt"].astype(str).str.len() > 0
    )

    total_pairs = len(results_df)
    pairs_with_both = int(mask_both_uniprot.sum())
    pairs_with_shared = int(results_df.loc[mask_both_uniprot, "Has_Shared_PDB"].sum())
    pairs_with_any_pdb = int(
        ((results_df["Protein1_PDB_Count"] > 0) | (results_df["Protein2_PDB_Count"] > 0)).sum()
    )

    logger.info("Validation complete!")
    logger.info(f"  Total pairs: {total_pairs}")
    logger.info(f"  Pairs with both UniProt IDs: {pairs_with_both}")
    logger.info(f"  Pairs with any PDB annotation: {pairs_with_any_pdb}")
    logger.info(f"  Pairs with shared PDB structures: {pairs_with_shared}")
    if pairs_with_both > 0:
        logger.info(f"  Validation rate (shared PDB | both UniProt): {pairs_with_shared / pairs_with_both * 100:.2f}%")
    logger.info(f"Saved results to {output_file}")

    return results_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate protein pairs by checking for shared PDB structures in UniProt"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input TSV with pairs and UniProt IDs")
    parser.add_argument("--output", type=Path, required=True, help="Output TSV with validation results")
    parser.add_argument("--cache", type=Path, help="Optional cache JSON for UniProt -> PDB mappings")
    args = parser.parse_args()

    if not args.cache:
        args.cache = args.output.parent / f"{args.output.stem}_pdb_cache.json"

    validate_pairs_with_pdb(args.input, args.output, args.cache)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


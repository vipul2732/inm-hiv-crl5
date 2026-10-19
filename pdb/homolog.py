"""Homolog-based PDB interface search when no shared UniProt PDB xref exists."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from chain_mapping import (
    chain_pairs_for_uniprots,
    chains_for_uniprot_in_pdb,
    get_pdb_entity_mapping,
)
from interface_bsa import InterfaceResult, has_direct_interface
from structure_io import download_structure, load_structure

logger = logging.getLogger(__name__)

RCSB_SEQUENCE_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"


@dataclass
class HomologChainMatch:
    pdb_id: str
    experimental_uniprot: str
    query_uniprot: str
    experimental_chains: List[str]
    homolog_chains: List[str]
    mapped_uniprot: str
    identity_pct: float
    alignment_length: int
    entity_id: int


def _normalize_uniprot(uid: str) -> str:
    return str(uid).strip().split("-")[0].upper()


def _normalize_pdb(pdb_id: str) -> str:
    return str(pdb_id).strip().upper()


def load_json_cache(path: Path) -> Dict:
    if path.exists():
        with open(path) as handle:
            return json.load(handle)
    return {}


def save_json_cache(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def fetch_uniprot_sequence(
    uniprot_id: str,
    session: requests.Session,
    cache: Dict[str, str],
    cache_path: Optional[Path] = None,
) -> str:
    uid = _normalize_uniprot(uniprot_id)
    if uid in cache:
        return cache[uid]

    url = f"https://rest.uniprot.org/uniprotkb/{uid}.fasta"
    try:
        resp = session.get(url, timeout=30)
    except requests.RequestException as exc:
        logger.debug("UniProt sequence fetch failed for %s: %s", uid, exc)
        cache[uid] = ""
        return ""

    if resp.status_code in (400, 404):
        cache[uid] = ""
        return ""
    if resp.status_code != 200:
        logger.debug("UniProt sequence HTTP %s for %s", resp.status_code, uid)
        cache[uid] = ""
        return ""
    seq = "".join(line.strip() for line in resp.text.splitlines() if not line.startswith(">"))
    cache[uid] = seq

    if cache_path is not None:
        save_json_cache(cache_path, cache)
    return seq


def fetch_entity_sequence(
    pdb_id: str,
    entity_id: int,
    session: requests.Session,
    cache: Dict[str, str],
    cache_path: Optional[Path] = None,
) -> str:
    key = f"{_normalize_pdb(pdb_id)}_{entity_id}"
    if key in cache:
        return cache[key]

    url = RCSB_POLYMER_URL.format(pdb_id=_normalize_pdb(pdb_id), entity_id=entity_id)
    resp = session.get(url, timeout=20)
    if resp.status_code != 200:
        cache[key] = ""
        return ""

    seq = resp.json().get("entity_poly", {}).get("pdbx_seq_one_letter_code", "") or ""
    cache[key] = seq
    if cache_path is not None:
        save_json_cache(cache_path, cache)
    return seq


def sequence_identity(seq_query: str, seq_target: str) -> Tuple[float, int]:
    if not seq_query or not seq_target:
        return 0.0, 0
    if max(len(seq_query), len(seq_target)) / max(min(len(seq_query), len(seq_target)), 1) > 3.0:
        return 0.0, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from Bio import pairwise2
        except ImportError:
            return 0.0, 0

        alignments = pairwise2.align.localxx(seq_query, seq_target, one_alignment_only=True)
        if not alignments:
            return 0.0, 0

        alignment = alignments[0]
        aligned_query = alignment[0]
        aligned_target = alignment[1]
        matches = sum(1 for a, b in zip(aligned_query, aligned_target) if a == b)
        aln_len = sum(1 for a, b in zip(aligned_query, aligned_target) if a != "-" or b != "-")
        if aln_len == 0:
            return 0.0, 0
        return 100.0 * matches / aln_len, aln_len


def rcsb_sequence_search_pdbs(
    sequence: str,
    session: requests.Session,
    cache: Dict[str, List[str]],
    cache_path: Optional[Path] = None,
    identity_cutoff: float = 0.3,
) -> Set[str]:
    if not sequence:
        return set()

    cache_key = f"{identity_cutoff}:{hash(sequence)}"
    if cache_key in cache:
        return set(cache[cache_key])

    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 10,
                "identity_cutoff": identity_cutoff,
                "target": "pdb_protein_sequence",
                "value": sequence,
            },
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 500},
        },
    }

    try:
        resp = session.post(RCSB_SEQUENCE_SEARCH_URL, json=query, timeout=90)
        resp.raise_for_status()
        pdb_ids = [
            _normalize_pdb(item["identifier"])
            for item in resp.json().get("result_set", [])
            if item.get("identifier")
        ]
    except requests.RequestException as exc:
        logger.debug("RCSB sequence search failed: %s", exc)
        pdb_ids = []

    cache[cache_key] = sorted(set(pdb_ids))
    if cache_path is not None:
        save_json_cache(cache_path, cache)
    return set(pdb_ids)


def pdb_ids_for_uniprot(uniprot_id: str, pdb_cache: Dict[str, List[str]]) -> Set[str]:
    uid = _normalize_uniprot(uniprot_id)
    return {_normalize_pdb(p) for p in pdb_cache.get(uid, [])}


def find_homolog_chains_in_pdb(
    pdb_id: str,
    experimental_uniprot: str,
    query_uniprot: str,
    query_sequence: str,
    session: requests.Session,
    chain_cache: Dict,
    entity_seq_cache: Dict[str, str],
    entity_seq_cache_path: Optional[Path],
    min_identity_pct: float,
    min_alignment_length: int,
) -> List[HomologChainMatch]:
    exp_uid = _normalize_uniprot(experimental_uniprot)
    query_uid = _normalize_uniprot(query_uniprot)
    exp_chains = set(
        chains_for_uniprot_in_pdb(pdb_id, exp_uid, session, chain_cache, None)
    )
    if not exp_chains:
        return []

    matches: List[HomologChainMatch] = []
    for entity in get_pdb_entity_mapping(pdb_id, session, chain_cache, None):
        entity_id = entity["entity_id"]
        entity_chains = list(entity.get("auth_chains", []))
        entity_uniprots = [_normalize_uniprot(u) for u in entity.get("uniprot_ids", [])]

        if exp_uid in entity_uniprots:
            continue
        if query_uid in entity_uniprots:
            continue

        entity_seq = fetch_entity_sequence(
            pdb_id, entity_id, session, entity_seq_cache, entity_seq_cache_path
        )
        identity_pct, aln_len = sequence_identity(query_sequence, entity_seq)
        if identity_pct < min_identity_pct or aln_len < min_alignment_length:
            continue

        homolog_chains = [c for c in entity_chains if c not in exp_chains]
        if not homolog_chains:
            continue

        mapped_uniprot = entity_uniprots[0] if entity_uniprots else ""
        matches.append(
            HomologChainMatch(
                pdb_id=_normalize_pdb(pdb_id),
                experimental_uniprot=exp_uid,
                query_uniprot=query_uid,
                experimental_chains=sorted(exp_chains),
                homolog_chains=homolog_chains,
                mapped_uniprot=mapped_uniprot,
                identity_pct=identity_pct,
                alignment_length=aln_len,
                entity_id=entity_id,
            )
        )

    return matches


def candidate_homolog_pdbs(
    uniprot_a: str,
    uniprot_b: str,
    pdb_cache: Dict[str, List[str]],
    seq_a: str,
    seq_b: str,
    session: requests.Session,
    search_cache: Dict[str, List[str]],
    search_cache_path: Optional[Path],
    identity_cutoff: float,
    max_pdbs: int,
) -> List[Tuple[str, str, str]]:
    """
    Return candidate tuples of (pdb_id, experimental_uniprot, query_uniprot).

    Uses partner PDB annotations plus RCSB sequence-search overlaps.
    """
    uid_a = _normalize_uniprot(uniprot_a)
    uid_b = _normalize_uniprot(uniprot_b)
    pdb_a = pdb_ids_for_uniprot(uid_a, pdb_cache)
    pdb_b = pdb_ids_for_uniprot(uid_b, pdb_cache)
    if not pdb_a and not pdb_b:
        return []

    hits_b = rcsb_sequence_search_pdbs(
        seq_b, session, search_cache, search_cache_path, identity_cutoff=identity_cutoff
    )
    hits_a = rcsb_sequence_search_pdbs(
        seq_a, session, search_cache, search_cache_path, identity_cutoff=identity_cutoff
    )

    def rank_partner_pdbs(partner_pdbs: Set[str], sequence_hits: Set[str]) -> List[str]:
        boosted = sorted(partner_pdbs & sequence_hits)
        remainder = sorted(partner_pdbs - set(boosted))
        return (boosted + remainder)[:max_pdbs]

    ranked_a = rank_partner_pdbs(pdb_a, hits_b)
    ranked_b = rank_partner_pdbs(pdb_b, hits_a)

    candidates: List[Tuple[str, str, str]] = []
    for pdb_id in ranked_a:
        candidates.append((pdb_id, uid_a, uid_b))
    for pdb_id in ranked_b:
        candidates.append((pdb_id, uid_b, uid_a))

    deduped: List[Tuple[str, str, str]] = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def evaluate_pair_homolog(
    uniprot_a: str,
    uniprot_b: str,
    pdb_cache: Dict[str, List[str]],
    session: requests.Session,
    chain_cache: Dict,
    chain_cache_path: Path,
    structure_cache_dir: Path,
    uniprot_seq_cache: Dict[str, str],
    uniprot_seq_cache_path: Path,
    entity_seq_cache: Dict[str, str],
    entity_seq_cache_path: Path,
    search_cache: Dict[str, List[str]],
    search_cache_path: Path,
    bsa_threshold: float,
    min_identity_pct: float,
    min_alignment_length: int,
    search_identity_cutoff: float,
    max_pdbs_per_pair: int,
) -> Tuple[Optional[str], Dict]:
    seq_a = fetch_uniprot_sequence(uniprot_a, session, uniprot_seq_cache, uniprot_seq_cache_path)
    seq_b = fetch_uniprot_sequence(uniprot_b, session, uniprot_seq_cache, uniprot_seq_cache_path)
    if not seq_a and not seq_b:
        return None, {}

    candidates = candidate_homolog_pdbs(
        uniprot_a,
        uniprot_b,
        pdb_cache,
        seq_a,
        seq_b,
        session,
        search_cache,
        search_cache_path,
        identity_cutoff=search_identity_cutoff,
        max_pdbs=max_pdbs_per_pair,
    )

    best_no_contact: Optional[Dict] = None
    pdbs_checked = 0

    for pdb_id, experimental_uniprot, query_uniprot in candidates:
        pdbs_checked += 1
        struct_path = download_structure(pdb_id, structure_cache_dir, session)
        if struct_path is None:
            continue

        try:
            structure = load_structure(struct_path)
            model = structure[0]
        except Exception as exc:
            logger.debug("Failed to parse homolog structure %s: %s", pdb_id, exc)
            continue

        direct_pairs = chain_pairs_for_uniprots(
            pdb_id,
            experimental_uniprot,
            query_uniprot,
            session,
            chain_cache,
            chain_cache_path,
        )
        for chain_a, chain_b in direct_pairs:
            ok, result = has_direct_interface(model, chain_a, chain_b, bsa_threshold=bsa_threshold)
            if result is None:
                continue
            record = _homolog_contact_record(
                match=HomologChainMatch(
                    pdb_id=_normalize_pdb(pdb_id),
                    experimental_uniprot=experimental_uniprot,
                    query_uniprot=query_uniprot,
                    experimental_chains=[chain_a],
                    homolog_chains=[chain_b],
                    mapped_uniprot=query_uniprot,
                    identity_pct=100.0,
                    alignment_length=0,
                    entity_id=0,
                ),
                exp_chain=chain_a,
                hom_chain=chain_b,
                result=result,
                bsa_threshold=bsa_threshold,
                pdbs_checked=pdbs_checked,
            )
            record["Contact_Evidence_Type"] = "structural_recovery"
            record["Homolog_Sequence_Identity_Pct"] = 100.0
            if ok:
                record["Contact_Validation_Status"] = "Direct_Contact_Structural_Recovery"
                return "Direct_Contact_Structural_Recovery", record
            if best_no_contact is None or result.bsa_ang2 > best_no_contact.get("Interface_BSA_Ang2", -1):
                record["Contact_Validation_Status"] = "Structural_Recovery_No_Contact"
                best_no_contact = record

        query_sequence = seq_b if query_uniprot == _normalize_uniprot(uniprot_b) else seq_a
        homolog_matches = find_homolog_chains_in_pdb(
            pdb_id=pdb_id,
            experimental_uniprot=experimental_uniprot,
            query_uniprot=query_uniprot,
            query_sequence=query_sequence,
            session=session,
            chain_cache=chain_cache,
            entity_seq_cache=entity_seq_cache,
            entity_seq_cache_path=entity_seq_cache_path,
            min_identity_pct=min_identity_pct,
            min_alignment_length=min_alignment_length,
        )
        if not homolog_matches:
            continue

        for match in homolog_matches:
            for exp_chain in match.experimental_chains:
                for hom_chain in match.homolog_chains:
                    ok, result = has_direct_interface(
                        model, exp_chain, hom_chain, bsa_threshold=bsa_threshold
                    )
                    if result is None:
                        continue

                    record = _homolog_contact_record(
                        match=match,
                        exp_chain=exp_chain,
                        hom_chain=hom_chain,
                        result=result,
                        bsa_threshold=bsa_threshold,
                        pdbs_checked=pdbs_checked,
                    )
                    if ok:
                        record["Contact_Validation_Status"] = "Direct_Contact_Homolog"
                        return "Direct_Contact_Homolog", record

                    if (
                        best_no_contact is None
                        or result.bsa_ang2 > best_no_contact.get("Interface_BSA_Ang2", -1)
                    ):
                        record["Contact_Validation_Status"] = "Homolog_Co_Complex_No_Contact"
                        best_no_contact = record

    if best_no_contact is not None:
        return "Homolog_Co_Complex_No_Contact", best_no_contact
    return None, {"PDBs_Checked": pdbs_checked}


def _homolog_contact_record(
    match: HomologChainMatch,
    exp_chain: str,
    hom_chain: str,
    result: InterfaceResult,
    bsa_threshold: float,
    pdbs_checked: int,
) -> Dict:
    return {
        "Contact_Evidence_Type": "homolog",
        "Contact_PDB_ID": match.pdb_id,
        "Contact_Chain_A": exp_chain,
        "Contact_Chain_B": hom_chain,
        "Interface_BSA_Ang2": round(result.bsa_ang2, 2),
        "SASA_Chain_A": round(result.sasa_chain_a, 2),
        "SASA_Chain_B": round(result.sasa_chain_b, 2),
        "SASA_Complex": round(result.sasa_complex, 2),
        "BSA_Threshold_Ang2": bsa_threshold,
        "SASA_Method": result.method,
        "PDBs_Checked": pdbs_checked,
        "Homolog_Query_UniProt": match.query_uniprot,
        "Homolog_Experimental_UniProt": match.experimental_uniprot,
        "Homolog_Chain_UniProt": match.mapped_uniprot,
        "Homolog_Sequence_Identity_Pct": round(match.identity_pct, 2),
        "Homolog_Alignment_Length": match.alignment_length,
        "Homolog_Entity_ID": match.entity_id,
    }

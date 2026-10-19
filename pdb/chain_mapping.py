"""Map UniProt accessions to PDB auth chain IDs via RCSB polymer entities."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"


def _normalize_uniprot(uid: str) -> str:
    return str(uid).strip().split("-")[0].upper()


def _normalize_pdb(pdb_id: str) -> str:
    return str(pdb_id).strip().upper()


def fetch_pdb_entities(pdb_id: str, session: requests.Session) -> List[Dict]:
    pdb_id = _normalize_pdb(pdb_id)
    entities: List[Dict] = []

    for entity_id in range(1, 64):
        url = RCSB_POLYMER_URL.format(pdb_id=pdb_id, entity_id=entity_id)
        try:
            resp = session.get(url, timeout=20)
        except requests.RequestException as exc:
            logger.debug("RCSB request failed for %s entity %s: %s", pdb_id, entity_id, exc)
            break

        if resp.status_code == 404:
            break
        if resp.status_code != 200:
            logger.debug("RCSB HTTP %s for %s entity %s", resp.status_code, pdb_id, entity_id)
            break

        data = resp.json()
        ids = data.get("rcsb_polymer_entity_container_identifiers", {})
        uniprot_ids = [_normalize_uniprot(u) for u in ids.get("uniprot_ids", []) or []]
        auth_chains = list(ids.get("auth_asym_ids", []) or [])
        if not auth_chains:
            continue

        entities.append(
            {
                "entity_id": entity_id,
                "auth_chains": auth_chains,
                "uniprot_ids": uniprot_ids,
            }
        )

    return entities


def get_pdb_entity_mapping(
    pdb_id: str,
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    cache_path: Optional[Path] = None,
) -> List[Dict]:
    pdb_key = _normalize_pdb(pdb_id)
    if pdb_key in cache:
        return cache[pdb_key]

    entities = fetch_pdb_entities(pdb_key, session)
    cache[pdb_key] = entities

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump(cache, handle, indent=2)

    time.sleep(0.05)
    return entities


def chains_for_uniprot_in_pdb(
    pdb_id: str,
    uniprot_id: str,
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    cache_path: Optional[Path] = None,
) -> List[str]:
    uid = _normalize_uniprot(uniprot_id)
    chains: List[str] = []

    for entity in get_pdb_entity_mapping(pdb_id, session, cache, cache_path):
        entity_uniprots = {_normalize_uniprot(u) for u in entity.get("uniprot_ids", [])}
        if uid not in entity_uniprots:
            continue
        for chain in entity.get("auth_chains", []):
            if chain not in chains:
                chains.append(chain)

    return chains


def chain_pairs_for_uniprots(
    pdb_id: str,
    uniprot_a: str,
    uniprot_b: str,
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    cache_path: Optional[Path] = None,
) -> List[tuple]:
    chains_a = chains_for_uniprot_in_pdb(pdb_id, uniprot_a, session, cache, cache_path)
    chains_b = chains_for_uniprot_in_pdb(pdb_id, uniprot_b, session, cache, cache_path)
    pairs: List[tuple] = []

    for ca in chains_a:
        for cb in chains_b:
            if ca == cb:
                continue
            pairs.append((ca, cb))

    return pairs

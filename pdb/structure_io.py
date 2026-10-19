"""Download and parse PDB/mmCIF structures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.simplefilter("ignore", PDBConstructionWarning)

logger = logging.getLogger(__name__)

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def normalize_pdb_id(pdb_id: str) -> str:
    return str(pdb_id).strip().upper()


def structure_cache_path(cache_dir: Path, pdb_id: str) -> Path:
    return cache_dir / f"{normalize_pdb_id(pdb_id)}.cif"


def download_structure(
    pdb_id: str,
    cache_dir: Path,
    session: requests.Session,
    force: bool = False,
) -> Optional[Path]:
    pdb_id = normalize_pdb_id(pdb_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = structure_cache_path(cache_dir, pdb_id)

    if out_path.exists() and not force:
        return out_path

    for url_template in (RCSB_CIF_URL, RCSB_PDB_URL):
        url = url_template.format(pdb_id=pdb_id)
        try:
            resp = session.get(url, timeout=120)
        except requests.RequestException as exc:
            logger.debug("Download failed for %s: %s", pdb_id, exc)
            continue

        if resp.status_code != 200 or not resp.text.strip():
            continue

        suffix = ".cif" if url.endswith(".cif") else ".pdb"
        out_path = cache_dir / f"{pdb_id}{suffix}"
        out_path.write_text(resp.text)
        return out_path

    logger.warning("Could not download structure for %s", pdb_id)
    return None


def load_structure(path: Path):
    pdb_id = path.stem
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, str(path))

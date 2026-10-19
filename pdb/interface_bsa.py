"""Buried surface area (BSA) interface detection between PDB chains."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from Bio.PDB import PDBIO
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

logger = logging.getLogger(__name__)

try:
    import freesasa

    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False

try:
    from Bio.PDB.SASA import ShrakeRupley

    HAS_SHRAKERUPLEY = True
except ImportError:
    HAS_SHRAKERUPLEY = False


@dataclass
class InterfaceResult:
    bsa_ang2: float
    sasa_chain_a: float
    sasa_chain_b: float
    sasa_complex: float
    method: str


def _save_pdb(structure: Structure, path: Path) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def _extract_chain(model: Model, chain_id: str) -> Structure:
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id} not in model")
    sub_model = Model(0)
    sub_structure = Structure("chain")
    sub_model.add(model[chain_id].copy())
    sub_structure.add(sub_model)
    return sub_structure


def _build_complex(model: Model, chain_a: str, chain_b: str) -> Structure:
    if chain_a not in model or chain_b not in model:
        raise KeyError(f"Chains {chain_a}/{chain_b} missing from model")
    sub_model = Model(0)
    sub_structure = Structure("complex")
    sub_model.add(model[chain_a].copy())
    sub_model.add(model[chain_b].copy())
    sub_structure.add(sub_model)
    return sub_structure


def _total_sasa_freesasa(structure: Structure) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as handle:
        path = Path(handle.name)
    try:
        _save_pdb(structure, path)
        fs_struct = freesasa.Structure(str(path))
        result = freesasa.calc(fs_struct)
        return float(result.totalArea())
    finally:
        path.unlink(missing_ok=True)


def _total_sasa_shrakerupley(structure: Structure, probe_radius: float = 1.4) -> float:
    sr = ShrakeRupley(probe_radius=probe_radius)
    sr.compute(structure, level="S")
    total = 0.0
    for model in structure:
        for chain in model:
            for residue in chain:
                total += float(getattr(residue, "sasa", 0.0))
    return total


def total_sasa(structure: Structure, probe_radius: float = 1.4) -> Tuple[float, str]:
    if HAS_FREESASA:
        return _total_sasa_freesasa(structure), "freesasa"
    if HAS_SHRAKERUPLEY:
        return _total_sasa_shrakerupley(structure, probe_radius=probe_radius), "shrake_rupley"
    raise RuntimeError("Neither freesasa nor BioPython ShrakeRupley is available for SASA calculation")


def compute_interface_bsa(
    model: Model,
    chain_a: str,
    chain_b: str,
    probe_radius: float = 1.4,
) -> Optional[InterfaceResult]:
    try:
        sasa_a, method = total_sasa(_extract_chain(model, chain_a), probe_radius=probe_radius)
        sasa_b, _ = total_sasa(_extract_chain(model, chain_b), probe_radius=probe_radius)
        sasa_ab, _ = total_sasa(_build_complex(model, chain_a, chain_b), probe_radius=probe_radius)
    except KeyError as exc:
        logger.debug("BSA skipped: %s", exc)
        return None
    except Exception as exc:
        logger.debug("BSA calculation failed for %s-%s: %s", chain_a, chain_b, exc)
        return None

    bsa = (sasa_a + sasa_b - sasa_ab) / 2.0
    return InterfaceResult(
        bsa_ang2=float(bsa),
        sasa_chain_a=float(sasa_a),
        sasa_chain_b=float(sasa_b),
        sasa_complex=float(sasa_ab),
        method=method,
    )


def has_direct_interface(
    model: Model,
    chain_a: str,
    chain_b: str,
    bsa_threshold: float = 400.0,
) -> Tuple[bool, Optional[InterfaceResult]]:
    result = compute_interface_bsa(model, chain_a, chain_b)
    if result is None:
        return False, None
    return result.bsa_ang2 >= bsa_threshold, result

#!/usr/bin/env python3
"""
For each base model under post_sampling_analysis/base_models/*/main_cluster_network/,
read main_cluster_edges.tsv (main-cluster mean edge frequency) and write:

  main_cluster_top25_edges.tsv    — 25 highest frequency edges
  main_cluster_bottom25_edges.tsv — 25 lowest frequency edges

Rank columns: rank_desc — for top TSV, 1 = highest frequency in that file; for bottom TSV,
1 = lowest frequency in that file.

UniProt accessions come from results/<dataset>/protein_to_uniprot.json (same mapping
as all_pairs_with_uniprot_* inputs), using validation gene normalization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from validation_analysis import normalize_protein_name  # noqa: E402


def _build_uniprot_lookup(raw: Dict[str, Any]) -> Dict[str, str]:
    """Map normalized gene symbol -> single UniProt accession (last key wins on ties)."""
    lut: Dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none"):
            continue
        nk = normalize_protein_name(k)
        lut[nk] = s.upper()
    return lut


def _uniprot_for(symbol: str, lut: Dict[str, str]) -> str:
    n = normalize_protein_name(symbol)
    return lut.get(n, "") or ""


def _finalize_columns(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Order: rank, proteins + UniProt, then frequency stats and binary column."""
    bin_cols = [c for c in df.columns if c.startswith("binary_")]
    if len(bin_cols) != 1:
        raise ValueError(f"Expected exactly one binary_* column, got {bin_cols}")
    binary = bin_cols[0]
    want = [
        rank_col,
        "protein1",
        "Protein1_UniProt",
        "protein2",
        "Protein2_UniProt",
        "frequency",
        "frequency_std",
        "wilson_lb",
        binary,
    ]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    extra = [c for c in df.columns if c not in want]
    if extra:
        raise ValueError(f"Unexpected extra columns: {extra}")
    return df[want].copy()


def _insert_uniprot_columns(df: pd.DataFrame, lut: Dict[str, str]) -> pd.DataFrame:
    """After columns protein1, protein2 add Protein1_UniProt, Protein2_UniProt."""
    out = df.copy()
    if "protein1" not in out.columns or "protein2" not in out.columns:
        return out
    u1 = [_uniprot_for(x, lut) for x in out["protein1"].astype(str)]
    u2 = [_uniprot_for(x, lut) for x in out["protein2"].astype(str)]
    cols = list(out.columns)
    i2 = cols.index("protein2") + 1
    out.insert(i2, "Protein2_UniProt", u2)
    out.insert(cols.index("protein1") + 1, "Protein1_UniProt", u1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-models-root",
        type=Path,
        default=REPO / "post_sampling_analysis" / "base_models",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=REPO / "results",
        help="Directory containing <dataset>/protein_to_uniprot.json per base model.",
    )
    ap.add_argument("--k", type=int, default=25, help="How many top/bottom edges per dataset.")
    args = ap.parse_args()
    root: Path = args.base_models_root
    results_root: Path = args.results_root
    if not root.is_dir():
        print(f"Missing {root}", file=sys.stderr)
        sys.exit(1)

    k = int(args.k)
    for ds_dir in sorted(root.glob("cullin_*_config_multi_seeds")):
        if not ds_dir.is_dir():
            continue
        net = ds_dir / "main_cluster_network"
        edges_tsv = net / "main_cluster_edges.tsv"
        summ_json = net / "main_cluster_summary.json"
        if not edges_tsv.is_file():
            print(f"SKIP {ds_dir.name} (no {edges_tsv.name})", flush=True)
            continue

        map_path = results_root / ds_dir.name / "protein_to_uniprot.json"
        lut: Dict[str, str] = {}
        if map_path.is_file():
            lut = _build_uniprot_lookup(json.loads(map_path.read_text()))
        else:
            print(
                f"WARN {ds_dir.name}: no {map_path} — UniProt columns will be empty",
                file=sys.stderr,
            )

        df = pd.read_csv(edges_tsv, sep="\t")
        if "frequency" not in df.columns:
            print(f"SKIP {ds_dir.name}: no 'frequency' column", file=sys.stderr)
            continue

        df = df.sort_values("frequency", ascending=False, kind="mergesort").reset_index(drop=True)
        top = df.head(k).copy()
        top.insert(0, "rank_desc", range(1, len(top) + 1))
        top = _insert_uniprot_columns(top, lut)
        top = _finalize_columns(top, "rank_desc")

        bot = df.sort_values("frequency", ascending=True, kind="mergesort").head(k).copy()
        bot.insert(0, "rank_desc", range(1, len(bot) + 1))
        bot = _insert_uniprot_columns(bot, lut)
        bot = _finalize_columns(bot, "rank_desc")

        meta = {}
        if summ_json.is_file():
            try:
                meta = json.loads(summ_json.read_text())
            except json.JSONDecodeError:
                pass
        main_id = meta.get("main_cluster_id")
        n_main = meta.get("n_draws_in_main_cluster")
        cond = ds_dir.name
        print(
            f"{cond}: main_cluster_id={main_id} n_draws={n_main} → top{k}/bottom{k} ({len(df)} edges)",
            flush=True,
        )

        out_top = net / f"main_cluster_top{k}_edges.tsv"
        out_bot = net / f"main_cluster_bottom{k}_edges.tsv"
        top.to_csv(out_top, sep="\t", index=False)
        bot.to_csv(out_bot, sep="\t", index=False)
        print(f"  Wrote {out_top.name}", flush=True)
        print(f"  Wrote {out_bot.name}", flush=True)


if __name__ == "__main__":
    main()

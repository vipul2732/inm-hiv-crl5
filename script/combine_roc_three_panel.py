#!/usr/bin/env python3
"""
Combine the three ROC PNGs (HIV WT, HIV ΔVIF, mock) into a single 3-panel figure for a paper.

Inputs (default):
  validation_ablations/roc_base_vs_ablations_wt.png
  validation_ablations/roc_base_vs_ablations_vif.png
  validation_ablations/roc_base_vs_ablations_mock.png

Output (default):
  validation_ablations/roc_base_vs_ablations_3panel.png

Panels are placed left-to-right in that order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help='Stem suffix for defaults, e.g. "_edgefreq" → roc_base_vs_ablations_mock_edgefreq.png',
    )
    base = ROOT / "validation_ablations"
    ap.add_argument("--mock", type=Path, default=None)
    ap.add_argument("--wt", type=Path, default=None)
    ap.add_argument("--vif", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--figsize",
        type=str,
        default="22,6.8",
        help="Figure size in inches: W,H (wider default matches wider single ROC panels)",
    )
    args = ap.parse_args()

    suf = args.suffix
    mock = args.mock or (base / f"roc_base_vs_ablations_mock{suf}.png")
    wt = args.wt or (base / f"roc_base_vs_ablations_wt{suf}.png")
    vif = args.vif or (base / f"roc_base_vs_ablations_vif{suf}.png")
    output = args.output or (base / f"roc_base_vs_ablations_3panel{suf}.png")

    for fp in [mock, wt, vif]:
        if not fp.exists():
            raise SystemExit(f"Missing input: {fp}")

    w, h = (float(x.strip()) for x in args.figsize.split(","))

    ims = [mpimg.imread(p) for p in [wt, vif, mock]]

    fig, axes = plt.subplots(1, 3, figsize=(w, h))
    for ax, im in zip(axes, ims):
        ax.imshow(im)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()


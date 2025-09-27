#!/usr/bin/env python3
"""
CLI: Entropy and sequence complexity metrics for 12-TET (or N-TET) scales.

Outputs (under --out-dir):
  - entropy_metrics.csv
  - hist_entropy.svg
  - scatter_entropy_vs_k.svg
  - scatter_entropy_vs_arrangement.svg
  - scatter_lz_vs_entropy.svg

Figures are publication-friendly SVGs (text preserved, grid on).

Examples:
  python scripts/scales_entropy_cli.py --out-dir out/entropy --max-gap 4
  python scripts/scales_entropy_cli.py --out-dir out/entropy --n 12 --min-k 5 --max-k 8
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from music_analysis.plotting import apply_pub_style, save_svg
from music_analysis.theory import (
    N_PCS_DEFAULT,
    arrangement_defect,
    iter_scale_masks,
    pcs_from_mask,
    shannon_entropy_bits,
    lz76_complexity_norm,
    step_vector_from_pcs,
)


def compute_table(n: int, require_root: bool, min_k: int, max_k: int | None, max_gap: int | None):
    rows = []
    for m in iter_scale_masks(n=n, require_root=require_root, min_k=min_k, max_k=max_k, max_gap=max_gap):
        pcs = pcs_from_mask(m, n)
        steps = step_vector_from_pcs(pcs, n)
        k = len(pcs)
        H, Hn = shannon_entropy_bits(steps, n)
        lz = lz76_complexity_norm(steps)
        arr = arrangement_defect(steps)
        rows.append({
            "mask": m,
            "n": n,
            "k": k,
            "entropy_bits": H,
            "entropy_norm": Hn,
            "lz_norm": lz,
            "arrangement_defect": arr,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Entropy and complexity metrics for N-TET scales")
    ap.add_argument("--out-dir", required=True, help="Directory to write CSVs and SVG figures")
    ap.add_argument("--n", type=int, default=N_PCS_DEFAULT, help="Number of pitch classes (default 12)")
    ap.add_argument("--min-k", type=int, default=1, help="Minimum scale size to include")
    ap.add_argument("--max-k", type=int, default=None, help="Maximum scale size to include (default n)")
    ap.add_argument("--max-gap", type=int, default=None, help="Max circular step allowed (semitones)")
    ap.add_argument("--no-require-root", action="store_true", help="Do NOT require pc 0 to be present")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    require_root = not args.no_require_root

    df = compute_table(
        n=args.n,
        require_root=require_root,
        min_k=args.min_k,
        max_k=args.max_k,
        max_gap=args.max_gap,
    )
    df.to_csv(os.path.join(out, "entropy_metrics.csv"), index=False)

    apply_pub_style()
    # Histogram of entropy_norm
    plt.figure()
    plt.hist(df["entropy_norm"], bins=30)
    plt.xlabel("Normalized Shannon entropy (steps)")
    plt.ylabel("Count of scales")
    plt.title("Distribution of entropy across scales")
    save_svg(os.path.join(out, "hist_entropy.svg"))

    # Scatter entropy vs k
    plt.figure()
    plt.scatter(df["k"], df["entropy_norm"], s=6, alpha=0.7)
    plt.xlabel("Scale size k")
    plt.ylabel("Normalized entropy")
    plt.title("Entropy vs scale size")
    save_svg(os.path.join(out, "scatter_entropy_vs_k.svg"))

    # Scatter entropy vs arrangement defect
    plt.figure()
    plt.scatter(df["arrangement_defect"], df["entropy_norm"], s=6, alpha=0.7)
    plt.xlabel("Arrangement defect")
    plt.ylabel("Normalized entropy")
    plt.title("Entropy vs arrangement defect")
    save_svg(os.path.join(out, "scatter_entropy_vs_arrangement.svg"))

    # Scatter LZ vs entropy
    plt.figure()
    plt.scatter(df["entropy_norm"], df["lz_norm"], s=6, alpha=0.7)
    plt.xlabel("Normalized entropy")
    plt.ylabel("Normalized LZ complexity")
    plt.title("Sequence complexity vs entropy")
    save_svg(os.path.join(out, "scatter_lz_vs_entropy.svg"))

    print(f"Wrote metrics CSV and figures to: {out}")


if __name__ == "__main__":
    main()


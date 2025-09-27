#!/usr/bin/env python3
"""
CLI: Collapse rotation/dihedral-equivalent modes; report class counts and symmetry.

Outputs (under --out-dir):
  - mode_classes.csv (canonical pattern, k, class_size, symmetry_order)
  - classes_by_k.csv (#classes per k)
  - bar_classes_by_k.svg
  - hist_symmetry_order.svg

Examples:
  python scripts/scales_mode_equivalence_cli.py --out-dir out/modes --max-gap 4
  python scripts/scales_mode_equivalence_cli.py --out-dir out/modes --dihedral
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from music_analysis.plotting import apply_pub_style, save_svg
from music_analysis.theory import (
    N_PCS_DEFAULT,
    canonical_step_pattern,
    iter_scale_masks,
    pcs_from_mask,
    step_vector_from_pcs,
)


def symmetry_order(steps: List[int], dihedral: bool) -> int:
    k = len(steps)
    if k == 0:
        return 0
    # Count group elements that fix the pattern
    count = 0
    # rotations
    for t in range(k):
        if steps == steps[t:] + steps[:t]:
            count += 1
    if dihedral:
        rev = list(reversed(steps))
        for t in range(k):
            if steps == rev[t:] + rev[:t]:
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Mode equivalence classes and symmetry analysis")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=N_PCS_DEFAULT)
    ap.add_argument("--max-gap", type=int, default=None)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--dihedral", action="store_true", help="Identify modes up to rotation AND reflection")
    ap.add_argument("--no-require-root", action="store_true")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    require_root = not args.no_require_root

    classes: Dict[Tuple[int, ...], Dict[str, int]] = {}
    details = []

    for m in iter_scale_masks(n=args.n, require_root=require_root, min_k=args.min_k, max_k=args.max_k, max_gap=args.max_gap):
        pcs = pcs_from_mask(m, args.n)
        steps = step_vector_from_pcs(pcs, args.n)
        canon = canonical_step_pattern(steps, dihedral=args.dihedral)
        if canon not in classes:
            classes[canon] = {"k": len(steps), "count": 0, "sym": symmetry_order(list(canon), args.dihedral)}
        classes[canon]["count"] += 1
        details.append({
            "mask": m,
            "k": len(steps),
            "canonical": "-".join(map(str, canon)),
        })

    # Build per-class table
    rows = []
    for canon, info in classes.items():
        rows.append({
            "canonical": "-".join(map(str, canon)),
            "k": info["k"],
            "class_size": info["count"],
            "symmetry_order": info["sym"],
        })
    df_classes = pd.DataFrame(rows).sort_values(["k", "class_size"], ascending=[True, False])
    df_classes.to_csv(os.path.join(out, "mode_classes.csv"), index=False)

    # Count classes per k
    by_k = df_classes.groupby("k").size().reset_index(name="num_classes")
    by_k.to_csv(os.path.join(out, "classes_by_k.csv"), index=False)

    # Plots
    apply_pub_style()
    plt.figure()
    plt.bar(by_k["k"], by_k["num_classes"])
    plt.xlabel("Scale size k")
    plt.ylabel("Number of equivalence classes")
    plt.title("Mode classes by k")
    save_svg(os.path.join(out, "bar_classes_by_k.svg"))

    plt.figure()
    plt.hist(df_classes["symmetry_order"], bins=range(0, df_classes["symmetry_order"].max() + 2))
    plt.xlabel("Symmetry order (stabilizer size)")
    plt.ylabel("Class count")
    plt.title("Distribution of symmetry orders")
    save_svg(os.path.join(out, "hist_symmetry_order.svg"))

    print(f"Found {len(df_classes)} classes across k. Results in {out}")


if __name__ == "__main__":
    main()


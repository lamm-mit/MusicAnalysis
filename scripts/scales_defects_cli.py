#!/usr/bin/env python3
"""
CLI: Enumerate 12-TET scales (with a single max-gap constraint), compute
'defect' metrics, and save plots + CSVs into a prescribed directory.

Outputs (all saved under --out-dir):
  - scale_metrics.csv
      mask,k,steps,unique_intervals,evenness_defect,arrangement_defect
  - unique_intervals_agg.csv
      k,mean,std,count,stderr
  - unique_intervals_vs_k.svg
      line+errorbars: mean unique-interval count vs k (± s.e.m.)
  - evenness_defect_vs_k.svg
      boxplots of evenness defect vs k
  - arrangement_defect_vs_k.svg
      boxplots of arrangement (mirror) defect vs k

Conventions:
  - 12 pitch classes (0..11), root is pc=0.
  - Step vector g = circular intervals between consecutive scale notes; sum(g)=12.
  - Max circular step (gap) ≤ K is enforced (K = --max-gap).
  - Evenness defect: normalized std of step sizes vs. ideal 12/k, normalized
    by the maximum std observed within valid scales of the same k.
  - Arrangement defect: 1 - max cosine similarity with reversed step vector
    under any rotation (0 = palindromic up to rotation, 1 = most asymmetric).

No seaborn; matplotlib only; one figure per chart; no explicit colors.

Usage examples:
  python scales_defects_cli.py --out-dir out --max-gap 4
  python scales_defects_cli.py --out-dir out --max-gap 4 --no-require-root
"""

import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_PCS = 12  # 12-TET


# ---------------------------
# Helpers
# ---------------------------
def pcs_from_mask(mask: int):
    return [i for i in range(N_PCS) if (mask >> i) & 1]


def max_circular_gap(pcs):
    if len(pcs) <= 1:
        return 12
    gaps = []
    for a, b in zip(pcs, pcs[1:]):
        gaps.append(b - a)
    gaps.append((pcs[0] + N_PCS) - pcs[-1])  # wrap-around
    return max(gaps)


def step_vector_from_pcs(pcs):
    """Intervals between consecutive pcs on the circle (wrap-around included). Sum = 12."""
    if len(pcs) <= 1:
        return []
    steps = []
    for a, b in zip(pcs, pcs[1:]):
        steps.append(b - a)
    steps.append((pcs[0] + N_PCS) - pcs[-1])
    return steps


def rotate(lst, tau):
    if not lst:
        return lst
    tau %= len(lst)
    return lst[tau:] + lst[:tau]


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def arrangement_defect(steps):
    """
    Mirror/arrangement defect:
    1 - max cosine similarity between steps and reversed steps under any rotation.
    0 = palindromic up to rotation; 1 = most asymmetric.
    """
    k = len(steps)
    if k == 0:
        return 0.0
    rev = list(reversed(steps))
    best = -1.0
    for tau in range(k):
        sim = cosine_similarity(steps, rotate(rev, tau))
        if sim > best:
            best = sim
    best = max(0.0, min(1.0, best))  # numeric clamp
    return 1.0 - best


def evenness_defect(steps, k, per_k_max_std_cache):
    """
    Size-sensitive evenness: normalized std dev of steps relative to ideal μ = 12/k.
    Normalize by the maximum std observed among valid scales of the same k.
    """
    if k <= 1:
        return 0.0
    mu = 12.0 / k
    arr = np.asarray(steps, dtype=float)
    std = float(np.sqrt(np.mean((arr - mu) ** 2)))
    max_std = per_k_max_std_cache.get(k, None)
    if max_std is None or max_std == 0:
        return 0.0
    return std / max_std


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute and plot scale defects for a single max-gap.")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Directory to save ALL images (SVG) and CSVs.")
    ap.add_argument("--max-gap", type=int, default=4,
                    help="Maximum allowed circular step (semitones). Default: 4")
    ap.add_argument("--no-require-root", action="store_true",
                    help="If set, do NOT require the root pc=0 to be present.")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    max_gap = int(args.max_gap)
    require_root = not args.no_require_root

    # Enumerate valid scales
    valid = []
    for mask in range(1 << N_PCS):
        if require_root and not (mask & 1):  # require pc 0
            continue
        pcs = pcs_from_mask(mask)
        if max_circular_gap(pcs) <= max_gap:
            steps = step_vector_from_pcs(pcs)
            if steps:  # skip degenerate k<=1
                valid.append((mask, pcs, steps))

    if not valid:
        print("No valid scales found under the given constraints.")
        return

    # Group by k
    by_k_steps = defaultdict(list)
    for _, pcs, steps in valid:
        by_k_steps[len(pcs)].append(steps)

    # Precompute per-k max std (for evenness normalization)
    per_k_max_std = {}
    for k, step_list in by_k_steps.items():
        mu = 12.0 / k
        max_std = 0.0
        for steps in step_list:
            arr = np.asarray(steps, dtype=float)
            std = float(np.sqrt(np.mean((arr - mu) ** 2)))
            if std > max_std:
                max_std = std
        per_k_max_std[k] = max_std if max_std > 0 else 1.0

    # Compute metrics per scale
    rows = []
    for mask, pcs, steps in valid:
        k = len(pcs)
        uniq_intervals = len(set(steps))
        d_arr = arrangement_defect(steps)
        d_even = evenness_defect(steps, k, per_k_max_std)
        rows.append({
            "mask": mask,
            "k": k,
            "steps": "-".join(map(str, steps)),
            "unique_intervals": uniq_intervals,
            "evenness_defect": d_even,
            "arrangement_defect": d_arr,
        })
    df = pd.DataFrame(rows).sort_values(["k", "mask"])
    df.to_csv(os.path.join(out_dir, "scale_metrics.csv"), index=False)

    # 1) Unique interval counts vs k (mean ± s.e.m.)
    agg_unique = df.groupby("k")["unique_intervals"].agg(["mean", "std", "count"]).reset_index()
    agg_unique["stderr"] = agg_unique["std"] / np.sqrt(agg_unique["count"].clip(lower=1))
    agg_unique.to_csv(os.path.join(out_dir, "unique_intervals_agg.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.errorbar(agg_unique["k"], agg_unique["mean"],
                 yerr=agg_unique["stderr"], marker="o", linestyle="-")
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Mean number of unique interval sizes (± s.e.m.)")
    plt.title(f"Unique Interval Counts vs. k (max circular step ≤ {max_gap}, require_root={require_root})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "unique_intervals_vs_k.svg"))
    plt.close()

    # Prepare labels
    ks_sorted = sorted(by_k_steps.keys())
    data_even = [df[df["k"] == k]["evenness_defect"].values for k in ks_sorted]
    data_arr = [df[df["k"] == k]["arrangement_defect"].values for k in ks_sorted]

    # 2) Evenness defect boxplots vs k
    plt.figure(figsize=(9, 5.5))
    plt.boxplot(data_even, positions=range(len(ks_sorted)), showfliers=False)
    plt.xticks(range(len(ks_sorted)), ks_sorted)
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Evenness defect (0 = perfectly even, 1 = most uneven in class)")
    plt.title(f"Evenness Defect vs. k (max circular step ≤ {max_gap}, require_root={require_root})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evenness_defect_vs_k.svg"))
    plt.close()

    # 3) Arrangement (mirror) defect boxplots vs k
    plt.figure(figsize=(9, 5.5))
    plt.boxplot(data_arr, positions=range(len(ks_sorted)), showfliers=False)
    plt.xticks(range(len(ks_sorted)), ks_sorted)
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Arrangement defect (0 = palindromic under rotation, 1 = most asymmetric)")
    plt.title(f"Arrangement Defect vs. k (max circular step ≤ {max_gap}, require_root={require_root})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arrangement_defect_vs_k.svg"))
    plt.close()

    # Console summary
    print("Wrote files to:", os.path.abspath(out_dir))
    print("  - scale_metrics.csv")
    print("  - unique_intervals_agg.csv")
    print("  - unique_intervals_vs_k.svg")
    print("  - evenness_defect_vs_k.svg")
    print("  - arrangement_defect_vs_k.svg")


if __name__ == "__main__":
    main()

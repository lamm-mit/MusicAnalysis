#!/usr/bin/env python3
"""
CLI: Enumerate 12-TET scales, compute “defect” metrics, and save plots + CSVs.

Adds:
  • --no-gap-filter : disable the max-gap constraint entirely
  • Composite defect: D_total = w1*evenness + w2*arrangement + w3*unique_intervals_defect
  • Extra plots:
      - evenness_vs_arrangement_scatter.svg
      - evenness_vs_unique_intervals_scatter.svg
      - composite_defect_boxplot_vs_k.svg
      - composite_defect_median_vs_k.svg
      - heatmap_evenness_by_k.svg (+ _colnorm)
      - heatmap_unique_by_k.svg (+ _colnorm)
      - evenness_vs_unique_2dhist.svg

Outputs (all under --out-dir):
  CSV
    - scale_metrics.csv
    - unique_intervals_agg.csv
    - composite_defect_agg.csv
  SVG figures
    - unique_intervals_vs_k.svg
    - evenness_defect_vs_k.svg
    - arrangement_defect_vs_k.svg
    - evenness_vs_arrangement_scatter.svg
    - evenness_vs_unique_intervals_scatter.svg
    - composite_defect_boxplot_vs_k.svg
    - composite_defect_median_vs_k.svg
    - heatmap_evenness_by_k.svg
    - heatmap_evenness_by_k_colnorm.svg
    - heatmap_unique_by_k.svg
    - heatmap_unique_by_k_colnorm.svg
    - evenness_vs_unique_2dhist.svg

Conventions:
  - 12 pitch classes (0..11), root is pc=0.
  - Step vector g = circular intervals between consecutive scale notes; sum(g) = 12.
  - “Max circular step” (gap) ≤ K enforced unless --no-gap-filter.
  - Evenness defect: std(|g - 12/k|) normalized by max std observed at that k.
  - Arrangement defect: 1 − max_τ cosine(g, rotate(reverse(g), τ))  (0 = palindromic under rotation).
  - Unique-intervals defect: normalized distinct step count in [0,1].

No seaborn; matplotlib only; one figure per chart; no explicit colors.

Examples:
  python scripts/scales_defects_cli.py --out-dir out/maxgap4 --max-gap 4
  python scripts/scales_defects_cli.py --out-dir out/no_gap --no-gap-filter
  python scripts/scales_defects_cli.py --out-dir out/maxgap4 --max-gap 4 --w1 0.6 --w2 0.2 --w3 0.2
"""

import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_PCS = 12  # 12-TET


# ---------- Helpers ----------
def pcs_from_mask(mask: int):
    return [i for i in range(N_PCS) if (mask >> i) & 1]


def max_circular_gap(pcs):
    if len(pcs) <= 1:
        return 12
    gaps = [b - a for a, b in zip(pcs, pcs[1:])]
    gaps.append((pcs[0] + N_PCS) - pcs[-1])  # wrap-around
    return max(gaps)


def step_vector_from_pcs(pcs):
    if len(pcs) <= 1:
        return []
    steps = [b - a for a, b in zip(pcs, pcs[1:])]
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
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def arrangement_defect(steps):
    k = len(steps)
    if k == 0:
        return 0.0
    rev = list(reversed(steps))
    best = -1.0
    for tau in range(k):
        sim = cosine_similarity(steps, rotate(rev, tau))
        if sim > best:
            best = sim
    best = max(0.0, min(1.0, best))
    return 1.0 - best


def evenness_defect(steps, k, per_k_max_std_cache):
    if k <= 1:
        return 0.0
    mu = 12.0 / k
    arr = np.asarray(steps, dtype=float)
    std = float(np.sqrt(np.mean((arr - mu) ** 2)))
    max_std = per_k_max_std_cache.get(k, None)
    if max_std is None or max_std == 0:
        return 0.0
    return std / max_std


def unique_intervals_defect(steps, k):
    if k <= 1:
        return 0.0
    distinct = len(set(steps))
    max_distinct = min(k, 6)
    if max_distinct <= 1:
        return 0.0
    return (distinct - 1) / (max_distinct - 1)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Compute and plot scale defects for one run.")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory for images/CSVs.")
    ap.add_argument("--max-gap", type=int, default=4, help="Max circular step (ignored if --no-gap-filter).")
    ap.add_argument("--no-gap-filter", action="store_true", help="Disable the max-gap filter.")
    ap.add_argument("--no-require-root", action="store_true", help="Do NOT require pc=0.")
    ap.add_argument("--w1", type=float, default=0.5, help="Weight for evenness_defect.")
    ap.add_argument("--w2", type=float, default=0.3, help="Weight for arrangement_defect.")
    ap.add_argument("--w3", type=float, default=0.2, help="Weight for unique_intervals_defect.")
    ap.add_argument("--heatmap-bins", type=int, default=12, help="Bins in [0,1] for heatmaps (rows).")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    require_root = not args.no_require_root

    # Enumerate valid scales
    valid = []
    for mask in range(1 << N_PCS):
        if require_root and not (mask & 1):
            continue
        pcs = pcs_from_mask(mask)
        if not args.no_gap_filter and max_circular_gap(pcs) > int(args.max_gap):
            continue
        steps = step_vector_from_pcs(pcs)
        if steps:
            valid.append((mask, pcs, steps))

    if not valid:
        print("No valid scales found under the given constraints.")
        return

    # Group by k
    by_k_steps = defaultdict(list)
    for _, pcs, steps in valid:
        by_k_steps[len(pcs)].append(steps)

    # Precompute per-k max std for evenness normalization
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

    # Per-scale metrics
    rows = []
    for mask, pcs, steps in valid:
        k = len(pcs)
        d_arr = arrangement_defect(steps)
        d_even = evenness_defect(steps, k, per_k_max_std)
        d_uniq = unique_intervals_defect(steps, k)
        d_total = args.w1 * d_even + args.w2 * d_arr + args.w3 * d_uniq
        rows.append({
            "mask": mask,
            "k": k,
            "steps": "-".join(map(str, steps)),
            "unique_intervals": len(set(steps)),
            "unique_intervals_defect": d_uniq,
            "evenness_defect": d_even,
            "arrangement_defect": d_arr,
            "composite_defect": d_total,
        })
    df = pd.DataFrame(rows).sort_values(["k", "mask"])
    df.to_csv(os.path.join(out_dir, "scale_metrics.csv"), index=False)

    # Unique intervals (mean ± s.e.m.) vs k
    agg_unique = df.groupby("k")["unique_intervals"].agg(["mean", "std", "count"]).reset_index()
    agg_unique["stderr"] = agg_unique["std"] / np.sqrt(agg_unique["count"].clip(lower=1))
    agg_unique.to_csv(os.path.join(out_dir, "unique_intervals_agg.csv"), index=False)

    suffix = "(no-gap)" if args.no_gap_filter else f"(max gap ≤ {int(args.max_gap)})"

    plt.figure(figsize=(8, 5))
    plt.errorbar(agg_unique["k"], agg_unique["mean"], yerr=agg_unique["stderr"], marker="o", linestyle="-")
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Mean number of unique interval sizes (± s.e.m.)")
    plt.title(f"Unique Interval Counts vs. k {suffix}, require_root={require_root}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "unique_intervals_vs_k.svg"))
    plt.close()

    # Prepare arrays per k
    ks_sorted = sorted(by_k_steps.keys())
    data_even = [df[df["k"] == k]["evenness_defect"].values for k in ks_sorted]
    data_arr = [df[df["k"] == k]["arrangement_defect"].values for k in ks_sorted]
    data_comp = [df[df["k"] == k]["composite_defect"].values for k in ks_sorted]

    # Boxplots
    plt.figure(figsize=(9, 5.5))
    plt.boxplot(data_even, positions=range(len(ks_sorted)), showfliers=False)
    plt.xticks(range(len(ks_sorted)), ks_sorted)
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Evenness defect (0 = perfectly even, 1 = most uneven in class)")
    plt.title(f"Evenness Defect vs. k {suffix}, require_root={require_root}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evenness_defect_vs_k.svg"))
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.boxplot(data_arr, positions=range(len(ks_sorted)), showfliers=False)
    plt.xticks(range(len(ks_sorted)), ks_sorted)
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Arrangement defect (0 = palindromic under rotation, 1 = most asymmetric)")
    plt.title(f"Arrangement Defect vs. k {suffix}, require_root={require_root}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "arrangement_defect_vs_k.svg"))
    plt.close()

    # Scatter: evenness vs arrangement
    plt.figure(figsize=(7, 6))
    for k in ks_sorted:
        sub = df[df["k"] == k]
        plt.scatter(sub["evenness_defect"], sub["arrangement_defect"], s=10, label=str(k))
    plt.xlabel("Evenness defect")
    plt.ylabel("Arrangement defect")
    plt.title(f"Evenness vs Arrangement {suffix}, require_root={require_root}")
    plt.legend(title="k", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evenness_vs_arrangement_scatter.svg"))
    plt.close()

    # Scatter: evenness vs unique-interval defect
    plt.figure(figsize=(7, 6))
    for k in ks_sorted:
        sub = df[df["k"] == k]
        plt.scatter(sub["evenness_defect"], sub["unique_intervals_defect"], s=10, label=str(k))
    plt.xlabel("Evenness defect")
    plt.ylabel("Unique-interval defect")
    plt.title(f"Evenness vs Unique-Interval Defect {suffix}, require_root={require_root}")
    plt.legend(title="k", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evenness_vs_unique_intervals_scatter.svg"))
    plt.close()

    # Composite: boxplots
    plt.figure(figsize=(9, 5.5))
    plt.boxplot(data_comp, positions=range(len(ks_sorted)), showfliers=False)
    plt.xticks(range(len(ks_sorted)), ks_sorted)
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Composite defect D_total")
    plt.title(
        f"Composite Defect vs. k {suffix}, require_root={require_root}\n"
        f"(weights w1={args.w1}, w2={args.w2}, w3={args.w3})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "composite_defect_boxplot_vs_k.svg"))
    plt.close()

    # Composite: median line with IQR band
    comp_stats = df.groupby("k")["composite_defect"].agg(["median", "count"]).reset_index()
    q1s, q3s = [], []
    for k in ks_sorted:
        vals = df[df["k"] == k]["composite_defect"].values
        if len(vals) > 0:
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
        else:
            q1 = q3 = 0.0
        q1s.append(q1)
        q3s.append(q3)
    comp_stats["q1"] = q1s
    comp_stats["q3"] = q3s
    comp_stats.to_csv(os.path.join(out_dir, "composite_defect_agg.csv"), index=False)

    plt.figure(figsize=(8.5, 5.2))
    xs = ks_sorted
    med = [float(comp_stats[comp_stats["k"] == k]["median"].values[0]) for k in ks_sorted]
    plt.plot(xs, med, marker="o")
    plt.fill_between(xs, q1s, q3s, alpha=0.2)
    plt.xlabel("Number of notes in the scale (k)")
    plt.ylabel("Composite defect D_total (median ± IQR)")
    plt.title(
        f"Composite Defect (median ± IQR) vs. k {suffix}, require_root={require_root}\n"
        f"(weights w1={args.w1}, w2={args.w2}, w3={args.w3})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "composite_defect_median_vs_k.svg"))
    plt.close()

    # -------- Heatmaps: distribution vs k --------
    def make_heatmap(metric_col: str, fname: str):
        """Build a (bins x K) matrix counting values in [0,1] per k, then plot imshow."""
        ks = ks_sorted
        nb = max(2, int(args.heatmap_bins))
        edges = np.linspace(0.0, 1.0, nb + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mat = np.zeros((nb, len(ks)), dtype=float)

        for j, k in enumerate(ks):
            vals = df[df["k"] == k][metric_col].values
            if len(vals) == 0:
                continue
            hist, _ = np.histogram(vals, bins=edges, range=(0.0, 1.0))
            mat[:, j] = hist

        # Raw counts heatmap
        plt.figure(figsize=(9, 5.8))
        plt.imshow(mat, aspect="auto", origin="lower", extent=[ks[0]-0.5, ks[-1]+0.5, 0.0, 1.0])
        plt.colorbar(label="Count")
        plt.xlabel("Number of notes in the scale (k)")
        plt.ylabel(f"{metric_col} (binned)")
        #plt.title(f"Distribution of {metric_col} vs. k {suffix}, require_root={require_root}")
        plt.title(f"Distribution of {metric_col} vs. k {suffix}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname}.svg"))
        plt.close()

        # Column-normalized (proportions) heatmap
        colsum = mat.sum(axis=0, keepdims=True)
        colsum[colsum == 0] = 1.0
        mat_norm = mat / colsum
        plt.figure(figsize=(9, 5.8))
        plt.imshow(mat_norm, aspect="auto", origin="lower", extent=[ks[0]-0.5, ks[-1]+0.5, 0.0, 1.0])
        plt.colorbar(label="Proportion")
        plt.xlabel("Number of notes in the scale (k)")
        plt.ylabel(f"{metric_col} (binned)")
        plt.title(f"Distribution of {metric_col} vs. k (column-normalized) {suffix}, require_root={require_root}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname}_colnorm.svg"))
        plt.close()

    make_heatmap("evenness_defect", "heatmap_evenness_by_k")
    make_heatmap("unique_intervals_defect", "heatmap_unique_by_k")

    # 2D histogram: evenness vs unique-interval defect (all scales)
    plt.figure(figsize=(7.2, 6.2))
    plt.hist2d(df["evenness_defect"].values, df["unique_intervals_defect"].values,
               bins=40, range=[[0, 1], [0, 1]])
    plt.colorbar(label="Count")
    plt.xlabel("Evenness defect")
    plt.ylabel("Unique-interval defect")
    plt.title(f"2D Histogram: Evenness vs Unique-Interval Defect {suffix}, require_root={require_root}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evenness_vs_unique_2dhist.svg"))
    plt.close()

    # Console summary
    print("Wrote files to:", os.path.abspath(out_dir))
    for name in [
        "scale_metrics.csv",
        "unique_intervals_agg.csv",
        "composite_defect_agg.csv",
        "unique_intervals_vs_k.svg",
        "evenness_defect_vs_k.svg",
        "arrangement_defect_vs_k.svg",
        "evenness_vs_arrangement_scatter.svg",
        "evenness_vs_unique_intervals_scatter.svg",
        "composite_defect_boxplot_vs_k.svg",
        "composite_defect_median_vs_k.svg",
        "heatmap_evenness_by_k.svg",
        "heatmap_evenness_by_k_colnorm.svg",
        "heatmap_unique_by_k.svg",
        "heatmap_unique_by_k_colnorm.svg",
        "evenness_vs_unique_2dhist.svg",
    ]:
        print("  -", name)


if __name__ == "__main__":
    main()

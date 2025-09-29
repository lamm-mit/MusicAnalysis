#!/usr/bin/env python3
"""
CLI: Entropy and sequence complexity metrics for 12-TET (or N-TET) scales.

Outputs (under --out-dir):
  - entropy_metrics.csv
  - hist_entropy.svg
  - scatter_entropy_vs_k.svg
  - scatter_entropy_vs_arrangement.svg
  - scatter_lz_vs_entropy.svg
  - heatmap_entropy_vs_evenness.svg
  - cultural_scales_metrics.csv (optional, if --overlay-cultural)
  - hist_entropy_with_cultural.svg (optional)
  - scatter_entropy_vs_k_cultural.svg (optional)
  - scatter_entropy_vs_arrangement_cultural.svg (optional)
  - mean_entropy_vs_k.svg (raw scatter + mean line ±1 SD)
  - mean_entropy_vs_k_cultural.svg (raw scatter + mean line ±1 SD + cultural overlays)
  - heatmap_entropy_vs_evenness_cultural.svg (optional)

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
from matplotlib import cm
from matplotlib.lines import Line2D

from music_analysis.plotting import apply_pub_style, save_svg
from music_analysis.theory import (
    N_PCS_DEFAULT,
    arrangement_defect,
    iter_scale_masks,
    pcs_from_mask,
    shannon_entropy_bits,
    lz76_complexity_norm,
    step_vector_from_pcs,
    parse_scale_spec,
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
        # evenness magnitude (std of deviations from n/k)
        if k <= 1:
            even_std = 0.0
        else:
            mu = n / k
            diffs = np.array(steps, dtype=float) - mu
            even_std = float(np.sqrt(np.mean(diffs * diffs)))
        rows.append({
            "mask": m,
            "n": n,
            "k": k,
            "entropy_bits": H,
            "entropy_norm": Hn,
            "lz_norm": lz,
            "arrangement_defect": arr,
            "evenness_std": even_std,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        per_k_max = df.groupby("k")["evenness_std"].transform(lambda s: s.max() if s.max() > 0 else 1.0)
        df["evenness_norm"] = (df["evenness_std"] / per_k_max).fillna(0.0).clip(0.0, 1.0)
    else:
        df["evenness_norm"] = []
    return df


## (no smoothing helper; mean line is plotted directly)


def cultural_catalog() -> dict:
    """Return a small catalog of culturally salient 12-TET approximations.

    Note: These are illustrative 12-TET step patterns for demonstration.
    """
    return {
        "major (ionian)": "major",
        "natural minor (aeolian)": "minor",
        "major pentatonic": "pentatonic-major",
        "minor pentatonic": "pentatonic-minor",
        "raga bhairav (≈ hijaz)": "raga-bhairav",
        "raga kalyani (lydian)": "raga-kalyani",
        "maqam bayati (approx)": "maqam-bayati",
        "maqam hijaz": "maqam-hijaz",
        "diminished octatonic (W–H)": "octatonic-whole-half",
        "diminished octatonic (H–W)": "octatonic-half-whole",
        "harmonic minor (7)": "harmonic-minor",
        "bebop dominant (8)": "bebop-dominant",
        "bebop major (8)": "bebop-major",
        "bebop harmonic minor (8)": "bebop-harmonic-minor",
    }


def main():
    ap = argparse.ArgumentParser(description="Entropy and complexity metrics for N-TET scales")
    ap.add_argument("--out-dir", required=True, help="Directory to write CSVs and SVG figures")
    ap.add_argument("--n", type=int, default=N_PCS_DEFAULT, help="Number of pitch classes (default 12)")
    ap.add_argument("--min-k", type=int, default=1, help="Minimum scale size to include")
    ap.add_argument("--max-k", type=int, default=None, help="Maximum scale size to include (default n)")
    ap.add_argument("--max-gap", type=int, default=None, help="Max circular step allowed (semitones)")
    ap.add_argument("--no-require-root", action="store_true", help="Do NOT require pc 0 to be present")
    ap.add_argument("--overlay-cultural", action="store_true", help="Overlay selected cultural scales on the plots (12-TET approximations)")
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
    plt.figure(figsize=(9, 5.5))
    plt.scatter(df["k"], df["entropy_norm"], s=6, alpha=0.7)
    plt.xlabel("Scale size k")
    plt.ylabel("Normalized entropy")
    plt.title("Entropy vs scale size")
    save_svg(os.path.join(out, "scatter_entropy_vs_k.svg"))

    # Mean entropy by k with raw data background
    apply_pub_style()
    plt.figure()
    plt.scatter(df["k"], df["entropy_norm"], s=6, alpha=0.18, label="All scales")
    grp = df.groupby("k")["entropy_norm"]
    mean_by_k = grp.mean().sort_index()
    std_by_k = grp.std(ddof=1).reindex(mean_by_k.index).fillna(0.0)
    x = mean_by_k.index.values.astype(float)
    m = mean_by_k.values.astype(float)
    s = std_by_k.values.astype(float)
    lower = np.clip(m - s, 0.0, 1.0)
    upper = np.clip(m + s, 0.0, 1.0)
    line, = plt.plot(x, m, linewidth=2.0, label="Mean by k")
    plt.fill_between(x, lower, upper, color=line.get_color(), alpha=0.18, label="±1 SD")
    plt.xlabel("Scale size k")
    plt.ylabel("Normalized entropy")
    plt.title("Entropy vs k (raw + mean ±1 SD)")
    plt.legend(frameon=False)
    save_svg(os.path.join(out, "mean_entropy_vs_k.svg"))

    # Scatter entropy vs arrangement defect
    plt.figure()
    plt.figure(figsize=(9, 5.5))
    
    plt.scatter(df["arrangement_defect"], df["entropy_norm"], s=6, alpha=0.7)
    plt.xlabel("Arrangement defect")
    plt.ylabel("Normalized entropy")
    plt.title("Entropy vs arrangement defect")
    save_svg(os.path.join(out, "scatter_entropy_vs_arrangement.svg"))

    # Scatter LZ vs entropy
    plt.figure()
    plt.figure(figsize=(9, 5.5))
    
    plt.scatter(df["entropy_norm"], df["lz_norm"], s=6, alpha=0.7)
    plt.xlabel("Normalized entropy")
    plt.ylabel("Normalized LZ complexity")
    plt.title("Sequence complexity vs entropy")
    save_svg(os.path.join(out, "scatter_lz_vs_entropy.svg"))

    # Heatmap: normalized entropy vs normalized evenness (standard colormap)
    apply_pub_style()
    plt.figure(figsize=(9.2, 6.2))
    h, xe, ye, im = plt.hist2d(
        df["evenness_norm"].values,
        df["entropy_norm"].values,
        bins=30,
        range=[[0, 1], [0, 1]],
    )
    # Remove tile borders and rasterize the mesh in SVG to avoid seam lines
    try:
        im.set_linewidth(0.0)
        im.set_edgecolor('none')
        im.set_antialiaseds(False)
        im.set_rasterized(True)
    except Exception:
        pass
    plt.grid(False)
    plt.xlabel("Evenness defect (normalized)")
    plt.ylabel("Normalized entropy")
    plt.title("Entropy vs evenness (density)")
    cbar = plt.colorbar()
    cbar.set_label("Count")
    plt.tight_layout()
    save_svg(os.path.join(out, "heatmap_entropy_vs_evenness.svg"))

    # Cultural overlays (12-TET approximations)
    if args.overlay_cultural and args.n == 12:
        cat = cultural_catalog()
        rows = []
        for name, spec in cat.items():
            try:
                m = parse_scale_spec(spec, n=args.n)
            except Exception:
                continue
            pcs = pcs_from_mask(m, args.n)
            if require_root and 0 not in pcs:
                pcs = sorted([0] + [p for p in pcs if p != 0])
            steps = step_vector_from_pcs(pcs, args.n)
            H, Hn = shannon_entropy_bits(steps, args.n)
            lz = lz76_complexity_norm(steps)
            arr = arrangement_defect(steps)
            rows.append({
                "name": name,
                "mask": m,
                "k": len(pcs),
                "entropy_bits": H,
                "entropy_norm": Hn,
                "lz_norm": lz,
                "arrangement_defect": arr,
            })
        if rows:
            cdf = pd.DataFrame(rows)
            cdf.to_csv(os.path.join(out, "cultural_scales_metrics.csv"), index=False)

            # Histogram with vertical lines for cultural scales + legend to avoid overlap
            apply_pub_style()
            plt.figure()
            plt.hist(df["entropy_norm"], bins=30, #color="lightblue",
                     
                     color="#A1C9F4",
                     )
            markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", ">", "<", "8"]
            colors = list(cm.get_cmap("tab10").colors)
            y0, y1 = plt.ylim()
            rug_y = y0 + 0.035 * (y1 - y0)
            legend_handles = []
            legend_labels = []
            for i, r in enumerate(cdf.itertuples(index=False)):
                x = float(r.entropy_norm)
                name = getattr(r, "name")
                col = colors[i % len(colors)]
                mk = markers[i % len(markers)]
                # vertical line in distinct color
                plt.axvline(x, color=col, linestyle="--", linewidth=1.2)
                # small rug marker with slight jitter to hint duplicates
                jitter = ((i % 5) - 2) * 0.002
                plt.scatter([x + jitter], [rug_y], color=col, marker=mk, s=40, zorder=3)
                # legend proxy showing both line style and marker symbol
                handle = Line2D([0], [0], color=col, linestyle="--", linewidth=1.2,
                                 marker=mk, markersize=6, markerfacecolor=col, markeredgecolor=col)
                legend_handles.append(handle)
                legend_labels.append(name)
            plt.xlabel("Normalized Shannon entropy (steps)")
            plt.ylabel("Count of scales")
            plt.title("Entropy distribution with cultural examples (12-TET approx.)")
            plt.legend(legend_handles, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Cultural scales", fontsize=9)
            save_svg(os.path.join(out, "hist_entropy_with_cultural.svg"))

            # Scatter entropy vs k with cultural labels
            apply_pub_style()
            plt.figure(figsize=(10, 5.6))
            plt.scatter(df["k"], df["entropy_norm"], s=6, alpha=0.35, label="All scales")
            markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", ">", "<", "8"]
            colors = list(cm.get_cmap("tab10").colors)
            # Plot each cultural point with distinct color/marker and slight x-jitter for visibility when overlapping
            for i, r in enumerate(cdf.itertuples(index=False)):
                col = colors[i % len(colors)]
                mk = markers[i % len(markers)]
                jitter_x = ((i % 5) - 2) * 0.06  # ±0.12 range around integer k
                plt.scatter([r.k + jitter_x], [r.entropy_norm], s=48, marker=mk, color=col, label=getattr(r, "name"))
            plt.xlabel("Scale size k")
            plt.ylabel("Normalized entropy")
            plt.title("Entropy vs k with cultural examples")
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Cultural scales", fontsize=9)
            save_svg(os.path.join(out, "scatter_entropy_vs_k_cultural.svg"))

            # Mean line with cultural overlays and raw background
            apply_pub_style()
            plt.figure(figsize=(9.5, 5))
            plt.scatter(df["k"], df["entropy_norm"], s=6, alpha=0.18, label="All scales")
            grp = df.groupby("k")["entropy_norm"]
            mean_by_k = grp.mean().sort_index()
            std_by_k = grp.std(ddof=1).reindex(mean_by_k.index).fillna(0.0)
            x = mean_by_k.index.values.astype(float)
            m = mean_by_k.values.astype(float)
            s = std_by_k.values.astype(float)
            lower = np.clip(m - s, 0.0, 1.0)
            upper = np.clip(m + s, 0.0, 1.0)
            line, = plt.plot(x, m, linewidth=2.0, label="Mean by k")
            plt.fill_between(x, lower, upper, color=line.get_color(), alpha=0.18, label="±1 SD")
            markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", ">", "<", "8"]
            colors = list(cm.get_cmap("tab10").colors)
            for i, r in enumerate(cdf.itertuples(index=False)):
                col = colors[i % len(colors)]
                mk = markers[i % len(markers)]
                jitter_x = ((i % 5) - 2) * 0.06
                plt.scatter([r.k + jitter_x], [r.entropy_norm], s=56, marker=mk, color=col, label=getattr(r, "name"))
            plt.xlabel("Scale size k")
            plt.ylabel("Normalized entropy")
            plt.title("Entropy vs k (raw + mean ±1 SD) with cultural examples")
            plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Cultural scales", fontsize=9)
            save_svg(os.path.join(out, "mean_entropy_vs_k_cultural.svg"))

            # Heatmap with cultural overlays (entropy vs evenness)
            apply_pub_style()
            plt.figure(figsize=(11, 6.5))
            h, xe, ye, im = plt.hist2d(
                df["evenness_norm"].values,
                df["entropy_norm"].values,
                bins=30,
                range=[[0, 1], [0, 1]],
            )
            try:
                im.set_linewidth(0.0)
                im.set_edgecolor('none')
                im.set_antialiaseds(False)
                im.set_rasterized(True)
            except Exception:
                pass
            plt.grid(False)
            # normalize cultural evenness using same per-k maxima
            per_k_max_map = df.groupby("k")["evenness_std"].max().to_dict()
            markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", ">", "<", "8"]
            colors = list(cm.get_cmap("tab10").colors)
            for i, r in enumerate(cdf.itertuples(index=False)):
                col = colors[i % len(colors)]
                mk = markers[i % len(markers)]
                k_val = int(r.k)
                max_std = per_k_max_map.get(k_val, 1.0) or 1.0
                pcs_c = pcs_from_mask(int(r.mask), args.n)
                steps_c = step_vector_from_pcs(pcs_c, args.n)
                if k_val <= 1:
                    even_norm = 0.0
                else:
                    mu_c = args.n / k_val
                    d = np.array(steps_c, dtype=float) - mu_c
                    ev_std = float(np.sqrt(np.mean(d * d)))
                    even_norm = 0.0 if max_std == 0 else float(np.clip(ev_std / max_std, 0.0, 1.0))
                jx = ((i % 5) - 2) * 0.004
                jy = ((i % 3) - 1) * 0.004
                plt.scatter([even_norm + jx], [float(r.entropy_norm) + jy], s=120, marker=mk, color=col, label=getattr(r, "name"))
            plt.xlabel("Evenness defect (normalized)")
            plt.ylabel("Normalized entropy")
            plt.title("Entropy vs evenness with cultural examples")
            handles, labels = plt.gca().get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            plt.legend(uniq.values(), uniq.keys(), loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False, title="Cultural scales", fontsize=11)
            cbar = plt.colorbar()
            cbar.set_label("Count")
            plt.tight_layout()
            save_svg(os.path.join(out, "heatmap_entropy_vs_evenness_cultural.svg"))

            # Scatter entropy vs arrangement with cultural labels
            apply_pub_style()
            plt.figure(figsize=(9.8, 5.6))
            plt.scatter(df["arrangement_defect"], df["entropy_norm"], s=12, alpha=0.35, label="All scales")
            markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", ">", "<", "8"]
            colors = list(cm.get_cmap("tab10").colors)
            for i, r in enumerate(cdf.itertuples(index=False)):
                col = colors[i % len(colors)]
                mk = markers[i % len(markers)]
                jitter_x = ((i % 5) - 2) * 0.005  # small horizontal jitter
                jitter_y = ((i % 3) - 1) * 0.004  # minimal vertical jitter
                plt.scatter([r.arrangement_defect + jitter_x], [r.entropy_norm + jitter_y], s=48, marker=mk, color=col, label=getattr(r, "name"))
            plt.xlabel("Arrangement defect")
            plt.ylabel("Normalized entropy")
            plt.title("Entropy vs arrangement with cultural examples")
            plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5), frameon=False, title="Cultural scales", fontsize=9)
            save_svg(os.path.join(out, "scatter_entropy_vs_arrangement_cultural.svg"))


    print(f"Wrote metrics CSV and figures to: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enumerate 12-TET scales and compare size distributions under different
maximum circular step (gap) constraints.

By default we require the root pitch class (0) to be present in every scale.
For each provided --max-gaps K (e.g., 3 4 5), we:
  - count valid scales by size (1..12),
  - save <out-dir>/counts_gapK.csv,
  - save <out-dir>/hist_gapK.png and <out-dir>/hist_gapK.svg.

If --include-no-gap is passed, we also:
  - count the 'no max-gap constraint' baseline,
  - save <out-dir>/counts_nogap.csv,
  - save <out-dir>/hist_nogap.png and <out-dir>/hist_nogap.svg.

Finally we save:
  - <out-dir>/counts_combined.csv  (columns for each series)
  - <out-dir>/hist_comparison.png and <out-dir>/hist_comparison.svg (overlay)

Usage examples:
  python scales_compare.py --out-dir results --max-gaps 4
  python scales_compare.py --out-dir results --max-gaps 3 4 5 --include-no-gap
  python scales_compare.py --out-dir results --max-gaps 4 --no-require-root

All plots use matplotlib (no seaborn, no explicit colors).
"""

import argparse
import os
from collections import Counter, OrderedDict
from typing import List, Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt

N_PCS = 12  # number of pitch classes in 12-TET


def pcs_from_mask(mask: int) -> List[int]:
    """Return sorted list of pitch classes (0..11) included in the bitmask."""
    return [i for i in range(N_PCS) if (mask >> i) & 1]


def max_circular_gap(pcs: List[int]) -> int:
    """Maximum cyclic gap (semitones) between adjacent scale tones on the 12-class circle."""
    if len(pcs) <= 1:
        return 12
    gaps = []
    for a, b in zip(pcs, pcs[1:]):
        gaps.append(b - a)
    # wrap-around
    gaps.append((pcs[0] + N_PCS) - pcs[-1])
    return max(gaps)


def count_by_size(require_root: bool, max_gap: Optional[int]) -> Counter:
    """
    Count how many valid scales there are for each size (1..12),
    given constraints:
      - require_root: if True, pitch class 0 must be included
      - max_gap: if an int, require max_circular_gap(pcs) <= max_gap;
                 if None, no gap constraint.
    Returns a Counter keyed by size (int) with counts (int).
    """
    counts = Counter()
    total = 1 << N_PCS
    for mask in range(total):
        if require_root and not (mask & 1):
            continue
        pcs = pcs_from_mask(mask)
        if max_gap is not None and max_circular_gap(pcs) > max_gap:
            continue
        counts[len(pcs)] += 1
    return counts


def counts_to_df(counts: Counter) -> pd.DataFrame:
    sizes = list(range(1, N_PCS + 1))
    return pd.DataFrame({
        "notes_in_scale": sizes,
        "count": [counts.get(k, 0) for k in sizes]
    })


def plot_histogram(df: pd.DataFrame, title: str, out_png: str, out_svg: str):
    plt.figure(figsize=(8, 5))
    plt.bar(df["notes_in_scale"], df["count"])
    plt.xlabel("Number of notes in the scale")
    plt.ylabel("Number of scales")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_svg)
    plt.close()


def plot_comparison(series_map: Dict[str, pd.Series], out_png: str, out_svg: str):
    """
    Overlay comparison plot for multiple series.
    series_map: label -> pd.Series indexed by notes_in_scale with counts
    """
    plt.figure(figsize=(9, 5.5))
    # Use lines with markers for clarity (no explicit colors)
    for label, ser in series_map.items():
        x = ser.index.values.tolist()
        y = ser.values.tolist()
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Number of notes in the scale")
    plt.ylabel("Number of scales")
    plt.title("12-TET Scales by Size: Comparison Across Max Circular Step Constraints")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_svg)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare 12-TET scale-size distributions for multiple max-gap constraints.")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Directory to save ALL images and data files.")
    ap.add_argument("--max-gaps", type=int, nargs="+", default=[4],
                    help="List of max circular step constraints (semitones). Example: --max-gaps 3 4 5")
    ap.add_argument("--include-no-gap", action="store_true",
                    help="Also include a baseline with NO max-gap constraint.")
    ap.add_argument("--no-require-root", action="store_true",
                    help="Do NOT require pitch class 0 to be included.")
    args = ap.parse_args()

    require_root = not args.no_require_root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Validate max-gaps
    clean_gaps: List[int] = []
    for k in args.max_gaps:
        if k < 1 or k > 12:
            raise ValueError(f"--max-gaps value {k} is out of range (1..12)")
        clean_gaps.append(int(k))

    # Compute and save per-setting artifacts
    combined = OrderedDict()  # label -> pd.Series indexed by notes_in_scale
    for k in clean_gaps:
        counts = count_by_size(require_root=require_root, max_gap=k)
        df = counts_to_df(counts)
        label = f"gap≤{k}"
        df.to_csv(os.path.join(out_dir, f"counts_gap{k}.csv"), index=False)

        title = f"12-TET Scales by Size (require_root={require_root}, max circular step ≤ {k})"
        plot_histogram(
            df,
            title=title,
            out_png=os.path.join(out_dir, f"hist_gap{k}.png"),
            out_svg=os.path.join(out_dir, f"hist_gap{k}.svg"),
        )
        ser = df.set_index("notes_in_scale")["count"]
        combined[label] = ser

    # Optional no-gap baseline
    if args.include_no_gap:
        counts_ng = count_by_size(require_root=require_root, max_gap=None)
        df_ng = counts_to_df(counts_ng)
        df_ng.to_csv(os.path.join(out_dir, "counts_nogap.csv"), index=False)

        title = f"12-TET Scales by Size (require_root={require_root}, NO max-gap)"
        plot_histogram(
            df_ng, title=title,
            out_png=os.path.join(out_dir, "hist_nogap.png"),
            out_svg=os.path.join(out_dir, "hist_nogap.svg"),
        )
        combined["no-gap"] = df_ng.set_index("notes_in_scale")["count"]

    # Build combined CSV
    # Union of indices (but they should all be 1..12)
    index = sorted(set().union(*[s.index for s in combined.values()]))
    combined_df = pd.DataFrame({"notes_in_scale": index})
    for label, ser in combined.items():
        combined_df[label] = ser.reindex(index, fill_value=0).values
    combined_df.to_csv(os.path.join(out_dir, "counts_combined.csv"), index=False)

    # Comparison plot overlay
    plot_comparison(
        {label: combined_df.set_index("notes_in_scale")[label] for label in combined},
        out_png=os.path.join(out_dir, "hist_comparison.png"),
        out_svg=os.path.join(out_dir, "hist_comparison.svg"),
    )

    # Small human-readable summary
    totals = {label: int(combined_df[label].sum()) for label in combined}
    summary_lines = [
        f"Output directory: {out_dir}",
        f"Require root: {require_root}",
        f"Series totals (sum over sizes): {totals}",
        "Files written:",
        "  - counts_gapK.csv & hist_gapK.(png|svg) for each K in --max-gaps",
        "  - counts_nogap.csv & hist_nogap.(png|svg) if --include-no-gap",
        "  - counts_combined.csv",
        "  - hist_comparison.(png|svg)",
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

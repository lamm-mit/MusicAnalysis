import math
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, pstdev
import numpy as np

NUM_STEPS = 12
MAX_LEAP = 4
MIN_NOTES = 3

def int_to_pcs(mask):
    return [i for i in range(NUM_STEPS) if mask & (1 << i)]

def pcs_to_intervals(pcs):
    pcs = sorted(pcs)
    diffs = []
    for i, p in enumerate(pcs):
        q = pcs[(i + 1) % len(pcs)]
        if i == len(pcs) - 1:
            q += NUM_STEPS  # wrap to octave
        diffs.append(q - p)
    return diffs

def is_scale(mask, pcs=None):
    # root (0) must be present
    if not (mask & 1):
        return False
    pcs = pcs if pcs is not None else int_to_pcs(mask)
    if len(pcs) < MIN_NOTES:
        return False
    diffs = pcs_to_intervals(pcs)
    # no step larger than a major 3rd
    return max(diffs) <= MAX_LEAP

def count_imperfections(pcs):
    # imperfect degree = fifth above is not in the scale
    s = set(pcs)
    imperfections = 0
    for p in pcs:
        if (p + 7) % NUM_STEPS not in s:
            imperfections += 1
    return imperfections

def shannon_entropy(intervals):
    """Shannon entropy (bits) of the interval-size distribution."""
    counts = defaultdict(int)
    for step in intervals:
        counts[step] += 1
    total = len(intervals)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def build_table_and_entropy():
    """Return (table, entropy_data). table[n_notes][n_imperfections] = count."""
    table = defaultdict(lambda: defaultdict(int))
    entropy_data = []
    for mask in range(1, 1 << NUM_STEPS):
        if mask % 2 == 0:  # root must be present
            continue
        pcs = int_to_pcs(mask)
        if not is_scale(mask, pcs):
            continue
        n_notes = len(pcs)
        n_imperf = count_imperfections(pcs)
        table[n_notes][n_imperf] += 1
        intervals = pcs_to_intervals(pcs)
        entropy = shannon_entropy(intervals)
        entropy_data.append(
            {"imperfections": n_imperf, "entropy": entropy, "n_notes": n_notes}
        )
    return table, entropy_data

table, entropy_data = build_table_and_entropy()

# -------------------------
# Plot 1: 7-note scales only
# -------------------------
n = 7
imperf_counts = table[n]
ks = list(range(7))  # imperfections 0–6
counts = [imperf_counts.get(k, 0) for k in ks]

fig1, ax1 = plt.subplots(figsize=(5, 3.5))
ax1.bar(ks, counts, color="#1f77b4")  # blue bars
ax1.set_xlabel("# of imperfections")
ax1.set_ylabel("# of 7-note scales")
ax1.set_xticks(ks)

# annotate counts above bars
for x, y in zip(ks, counts):
    if y > 0:
        ax1.text(x, y + 1, str(y), ha='center', va='bottom', fontsize=8)

# add extra headroom so labels don't hit the top boundary
max_count = max(counts) if counts else 0
ax1.set_ylim(0, max_count * 1.2 if max_count > 0 else 1)

fig1.tight_layout()
fig1.savefig("7tone_imperfections_hist.svg", format="svg")
plt.close(fig1)

# ---------------------------------------------
# Plot 2: All scale sizes (1–12) as a heatmap
# ---------------------------------------------
max_k = 6
note_counts = list(range(1, NUM_STEPS + 1))
imperf_vals = list(range(max_k + 1))

# Build matrix: rows = n_notes (1–12), cols = imperfections (0–6)
matrix = []
for n in note_counts:
    row = [table[n].get(k, 0) for k in imperf_vals]
    matrix.append(row)

fig2, ax2 = plt.subplots(figsize=(7, 4))
im = ax2.imshow(matrix, aspect='auto', cmap="Blues")  # blue colormap

ax2.set_xlabel("# of imperfections")
ax2.set_ylabel("# of notes in scale")
ax2.set_xticks(range(len(imperf_vals)))
ax2.set_xticklabels(imperf_vals)
ax2.set_yticks(range(len(note_counts)))
ax2.set_yticklabels(note_counts)

# annotate cells with counts
for i, n in enumerate(note_counts):
    for j, k in enumerate(imperf_vals):
        val = matrix[i][j]
        if val > 0:
            ax2.text(j, i, str(val), ha='center', va='center', fontsize=6, color="black")

fig2.tight_layout()
fig2.savefig("all_scales_imperfections_heatmap.svg", format="svg")
plt.close(fig2)

# --------------------------------------------------------------
# Plot 3: Shannon entropy of interval pattern vs imperfections
# --------------------------------------------------------------
entropy_by_imperf = defaultdict(list)
for entry in entropy_data:
    entropy_by_imperf[entry["imperfections"]].append(entry["entropy"])

imperf_bins = sorted(entropy_by_imperf.keys())
entropy_means = [mean(entropy_by_imperf[k]) for k in imperf_bins]
entropy_stds = [
    pstdev(entropy_by_imperf[k]) if len(entropy_by_imperf[k]) > 1 else 0
    for k in imperf_bins
]

fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.errorbar(
    imperf_bins,
    entropy_means,
    yerr=entropy_stds,
    fmt='o-',
    capsize=4,
    color="#1f77b4",  # blue line + markers
)
ax3.set_xlabel("# of imperfections")
ax3.set_ylabel("Shannon entropy (bits)")
ax3.set_xticks(imperf_bins)
ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

fig3.tight_layout()
fig3.savefig("entropy_vs_imperfections.svg", format="svg")
plt.close(fig3)

# ------------------------------------------------------
# Plot 4: Scales by size, stacked by # imperfections
# ------------------------------------------------------
note_sizes = list(range(MIN_NOTES, NUM_STEPS + 1))
imperf_vals_plot = sorted({k for n in note_sizes for k in table.get(n, {})})

fig4, ax4 = plt.subplots(figsize=(7, 4))
bottom = [0] * len(note_sizes)

# blue shades for imperfection bins
colors_imperf = plt.cm.Blues(np.linspace(0.3, 0.9, len(imperf_vals_plot)))

for idx, k in enumerate(imperf_vals_plot):
    counts_k = [table.get(n, {}).get(k, 0) for n in note_sizes]
    ax4.bar(
        note_sizes,
        counts_k,
        bottom=bottom,
        color=colors_imperf[idx],
        label=f"{k} imperf.",
    )
    bottom = [b + c for b, c in zip(bottom, counts_k)]

ax4.set_xlabel("# of notes in scale")
ax4.set_ylabel("# of scales")
ax4.set_xticks(note_sizes)
ax4.legend(title="# imperfections", fontsize=8, title_fontsize=9, ncol=2)
ax4.grid(True, axis='y', linestyle='--', alpha=0.3)

# annotate total # of scales per note size
for x, total in zip(note_sizes, bottom):
    if total > 0:
        ax4.text(x, total + 5, str(total), ha='center', va='bottom', fontsize=8)

# extra headroom for labels
max_total = max(bottom) if bottom else 0
ax4.set_ylim(0, max_total * 1.2 if max_total > 0 else 1)

fig4.tight_layout()
fig4.savefig("scale_counts_by_size.svg", format="svg")
plt.close(fig4)

# ----------------------------------------------------------------
# Plot 5: # of scales vs. # of imperfections (stacked by scale size)
# ----------------------------------------------------------------
imperf_bins_all = sorted({k for n in note_sizes for k in table.get(n, {})})
size_vals_plot = note_sizes  # 3..12

fig5, ax5 = plt.subplots(figsize=(7, 4))
bottom_imperf = [0] * len(imperf_bins_all)

# blue shades for different scale sizes
colors_sizes = plt.cm.Blues(np.linspace(0.3, 0.9, len(size_vals_plot)))

for idx, n in enumerate(size_vals_plot):
    counts_n = [table.get(n, {}).get(k, 0) for k in imperf_bins_all]
    ax5.bar(
        imperf_bins_all,
        counts_n,
        bottom=bottom_imperf,
        color=colors_sizes[idx],
        label=f"{n} notes",
    )
    bottom_imperf = [b + c for b, c in zip(bottom_imperf, counts_n)]

ax5.set_xlabel("# of imperfections")
ax5.set_ylabel("# of scales")
ax5.set_xticks(imperf_bins_all)
ax5.legend(title="# notes in scale", fontsize=8, title_fontsize=9, ncol=2)
ax5.grid(True, axis='y', linestyle='--', alpha=0.3)

# annotate total # of scales per imperfection count
for x, total in zip(imperf_bins_all, bottom_imperf):
    if total > 0:
        ax5.text(x, total + 5, str(total), ha='center', va='bottom', fontsize=8)

max_total_imperf = max(bottom_imperf) if bottom_imperf else 0
ax5.set_ylim(0, max_total_imperf * 1.2 if max_total_imperf > 0 else 1)

fig5.tight_layout()
fig5.savefig("scales_by_imperfections_stacked.svg", format="svg")
plt.close(fig5)

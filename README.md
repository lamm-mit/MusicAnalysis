# MusicAnalysis

Music analysis tools especially around scales, defects and related topics.

Dependencies are declared in `pyproject.toml` (PEP 621). No `requirements.txt` or `environment.yml` needed.

**Conda (recommended workflow)**

- Create env: `conda create -n music-analysis python>=3.9`
- Activate: `conda activate music-analysis`
- Install this project (reads deps from `pyproject.toml`): `python -m pip install -e .`
- Verify: `music-cli --version`

**Clone**

- HTTPS:
  - `git clone https://github.com/lamm-mit/MusicAnalysis.git`
  - `cd MusicAnalysis`
- SSH:
  - `git clone git@github.com:lamm-mit/MusicAnalysis.git`
  - `cd MusicAnalysis`

**Install (editable for development)**

- Ensure Python 3.9+.
- From the repo root:
  - `pip install -e .`

If you're offline or want to avoid downloading build tools, use:

- `pip install -e . --no-build-isolation`

This installs a console command `music-cli` in your environment.

**Usage**

- Print version: `music-cli --version`
- Show environment info: `music-cli --info`

If you prefer not to install, you can run the wrapper script with the package on `PYTHONPATH`:

- `PYTHONPATH=src python scripts/music_cli.py --version`

### Compare sccales

1) Only K=4, save into ./out
```python ./scripts/scales_compare.py --out-dir out --max-gaps 4```

2) Compare K=3,4,5 and include baseline with no max-gap
```python ./scripts/scales_compare.py --out-dir out --max-gaps 3 4 5 --include-no-gap```

3) Explore without requiring the root
```python ./scripts/scales_compare.py --out-dir out --max-gaps 4 --no-require-root```

### Compute scale defects

1) No gap filter (baseline)
```python ./scripts/scales_defects_cli.py --out-dir out/no_gap --no-gap-filter```

2) Classic filter (e.g., K=4)
```python ./scripts/scales_defects_cli.py --out-dir out/maxgap4 --max-gap 4```

3) Customize composite weights
```python ./scripts/scales_defects_cli.py --out-dir out/maxgap4 --max-gap 4 --w1 0.6 --w2 0.2 --w3 0.2```

4) Without requiring root (optional)
```python ./scripts/scales_defects_cli.py --out-dir out/maxgap4_no_root --max-gap 4 --no-require-root```

5) All subsets (no gap filter), 20 bins in [0,1] for heatmaps
```python scripts/scales_defects_cli.py --out-dir out/no_gap --no-gap-filter --heatmap-bins 20```

6) Ian Ring–style gap (≤4) with defaults
```python scripts/scales_defects_cli.py --out-dir out/maxgap4 --max-gap 4```

## Additional analysis: entropy, symmetry, and graphs

Entropy & sequence complexity (`scripts/scales_entropy_cli.py`)
- Purpose: quantify order/disorder of a scale’s step vector and relate it to geometric symmetry (arrangement defect).
- Inputs: `--n` (N‑TET, default 12), `--min-k/--max-k` or `--k`, optional `--max-gap`, and `--no-require-root`.
- Step vector: for sorted pitch classes including the root, take consecutive differences on the circle plus wrap‑around; entries are positive integers that sum to `n`.
- Metrics
  - Shannon entropy (bits): `H = -Σ_s p(s)·log2 p(s)` where `p(s)` is the frequency of step size `s` in the vector.
  - Normalized entropy: divide by `log2(min(k, n))` to get `[0,1]`, enabling comparisons across `k`.
  - LZ76 complexity (normalized): parse the step sequence into novel phrases; report `c(n)*log2(n)/n` in `[0,1]`. Sensitive to sequential structure beyond histograms.
  - Arrangement defect: `1 − max_τ cosine(g, rotate(reverse(g), τ))`; equals `0` for palindromic arrangements under rotation, increases with asymmetry.
- Outputs
  - `entropy_metrics.csv`: one row per scale with `k`, `entropy_bits`, `entropy_norm`, `lz_norm`, `arrangement_defect`.
  - Figures: `hist_entropy.svg`, `scatter_entropy_vs_k.svg`, `scatter_entropy_vs_arrangement.svg`, `scatter_lz_vs_entropy.svg`, `mean_entropy_vs_k.svg` (raw scatter + mean-by-k line with ±1 SD shaded band), `heatmap_entropy_vs_evenness.svg` (2D density of entropy vs evenness).
- Cultural overlays (12‑TET approximations): add labeled examples to the distribution and scatter plots using `--overlay-cultural`. Included: Western major/minor, major/minor pentatonic, raga Bhairav, raga Kalyani (Lydian), maqam Bayati, maqam Hijaz. These use simple 12‑TET step approximations for illustration.
  - Also included: diminished (octatonic, both W–H and H–W forms), harmonic minor (7), bebop dominant (8), bebop major (8), and bebop harmonic minor (8). Names and step patterns use common 12‑TET approximations.
  - Example: `python scripts/scales_entropy_cli.py --out-dir out/entropy --max-gap 4 --overlay-cultural`
  - Extra outputs: `cultural_scales_metrics.csv`, `hist_entropy_with_cultural.svg`, `scatter_entropy_vs_k_cultural.svg`, `scatter_entropy_vs_arrangement_cultural.svg`, `mean_entropy_vs_k_cultural.svg` (raw scatter + mean line with ±1 SD band + cultural overlays), `heatmap_entropy_vs_evenness_cultural.svg`.
  - Plotting details: cultural examples use distinct colors and markers with a legend placed outside the axes to avoid overlap. Small, non-inferential jitter is applied when multiple examples share identical coordinates so each marker remains visible while still indicating co-location.

Mode equivalence & symmetry (`scripts/scales_mode_equivalence_cli.py`)
- Purpose: collapse rotation‑equivalent (and optionally reflection‑equivalent) modes and measure symmetry.
- Canonical representative: lexicographically smallest rotation (and reflection if `--dihedral`) of the step pattern.
- Symmetry order: number of group elements (C_k or D_k) that fix the pattern; higher means more regularity.
- Outputs
  - `mode_classes.csv`: `canonical` (e.g., `2-2-1-2-2-2-1`), `k`, `class_size` (distinct masks hitting this pattern under filters), `symmetry_order`.
  - `classes_by_k.csv`: count of canonical classes per `k`.
  - Figures: `bar_classes_by_k.svg`, `hist_symmetry_order.svg`.
- Example: `python scripts/scales_mode_equivalence_cli.py --out-dir out/modes --max-gap 4 --dihedral`

Graphs & distances (`scripts/scales_graph_cli.py`)
- Purpose: view scales as nodes, connect them by minimal edits, and analyze neighborhoods, components, and morph paths.
- Node set: all masks satisfying filters (`--n`, `--k`/`--min-k/--max-k`, `--max-gap`, root requirement).
- Edge types
  - `flip`: toggle one pitch class (Hamming distance 1) — add/remove.
  - `swap`: remove one and add one (Hamming distance 2) — preserves `k`.
- Distances & paths
  - Nearest neighbors to `--target` by Hamming distance (accepts `major`, `minor`, `mask:<int>`, `steps:…`).
  - Shortest path between `--path-src` and `--path-dst` on the constructed graph (BFS).
- Outputs
  - `graph_nodes.csv` (`mask,k,degree,component_id`), `graph_edges.csv` (`u,v`).
  - Figures: `degree_histogram.svg`, `component_sizes.svg`.
  - Optional: `nearest_neighbors.csv`, `shortest_path.csv` when requested.
- Examples
  - Flip-edge graph: `python scripts/scales_graph_cli.py --out-dir out/graph --max-gap 4`
  - Swap-edge graph k=7: `python scripts/scales_graph_cli.py --out-dir out/graph_swap --edge-type swap --k 7`
  - NN to major: `python scripts/scales_graph_cli.py --out-dir out/graph_nn --k 7 --target major`
  - Path major→minor: `python scripts/scales_graph_cli.py --out-dir out/graph_path --edge-type swap --k 7 --path-src major --path-dst minor`

Notes on scale enumeration
- Runtime grows with `n` and relaxed filters. For `n=12` it’s fast; for larger `n`, constrain `--k` and/or `--max-gap`.
- Figures are saved as SVG with text preserved and a clean grid suitable for publications.

### Example results

<img width="615" height="770" alt="image" src="https://github.com/user-attachments/assets/c77f6e43-fd02-4060-b5b8-6c3fb3f3290b" />

<img width="1041" height="626" alt="image" src="https://github.com/user-attachments/assets/0a3aa668-c18f-4be3-849f-43456220e586" />

Entropy versus evenness defect across the space of valid 12-TET scales, with cultural exemplars overlaid. The dashed ellipse highlights the “cultural corridor” of moderate entropy and defect, where nearly all salient systems (major, minor, pentatonic, raga, maqam, bebop) cluster. The analysis provides a direct quantitative link between defect-mediated resilience in matter and imperfection-driven expressivity in music. The result leads also towards a unified view in which both material toughness and musical expressivity  arise most robustly at intermediate levels of imperfection.

**Uninstall**

- `pip uninstall music-analysis`

**Dependencies**

Python dependencies are declared in `pyproject.toml` and installed by pip:
- numpy, pandas, matplotlib

On Conda, you can also preinstall them: `conda install -n music-analysis numpy pandas matplotlib`, then run `python -m pip install -e .`.


## Reference and Citation

```bibtex
@article{buehler2025musicanalysis,
  title={Materiomusic as a Framework for Creativity and Discovery},
  author={Buehler, Markus J.},
  journal={arXiv preprint arXiv:2509.xxxxx},
  year={2025}
}
```

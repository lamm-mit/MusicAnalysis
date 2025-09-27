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

### Example results

<img width="615" height="770" alt="image" src="https://github.com/user-attachments/assets/c77f6e43-fd02-4060-b5b8-6c3fb3f3290b" />

**Uninstall**

- `pip uninstall music-analysis`

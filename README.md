# MusicAnalysis

Music analysis tools especially around scales, defects and related topics.

**Conda**

- Create from `environment.yml` (installs package editable via pip):
  - `conda env create -f environment.yml`
  - `conda activate music-analysis`
  - Verify: `music-cli --version`
- Update an existing env to match `environment.yml`:
  - `conda env update -f environment.yml -n music-analysis`
- Manual Conda setup (alternative):
  - `conda create -n music-analysis python>=3.9`
  - `conda activate music-analysis`
  - `python -m pip install -e .` (add `--no-build-isolation` if desired)

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

**Uninstall**

- `pip uninstall music-analysis`

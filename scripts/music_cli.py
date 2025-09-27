#!/usr/bin/env python3
"""Thin wrapper to run the package CLI without installation.

Examples:
  PYTHONPATH=src python scripts/music_cli.py --version
  PYTHONPATH=src python scripts/music_cli.py --info
"""

from __future__ import annotations

from music_analysis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

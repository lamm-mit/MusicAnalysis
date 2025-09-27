from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="music-cli", description="Music Analysis command-line tools"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print package version and exit"
    )
    parser.add_argument(
        "--info", action="store_true", help="Print basic environment information"
    )

    args = parser.parse_args(argv)

    if args.version:
        try:
            from . import __version__  # type: ignore
        except Exception:
            __version__ = "0.0.0"
        print(__version__)
        return 0

    if args.info:
        print(f"Python {sys.version.split()[0]} | cwd={Path.cwd()}")
        return 0

    parser.print_help()
    return 0


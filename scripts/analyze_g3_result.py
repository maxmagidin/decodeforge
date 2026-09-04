#!/usr/bin/env python3
"""Build one closed G3 result bundle from three accepted session files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decodeforge.g3_results import G3ResultError, analyze_and_write_g3_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs=3,
        required=True,
        type=Path,
        metavar=("SESSION_1", "SESSION_2", "SESSION_3"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        analyze_and_write_g3_result(arguments.sessions, arguments.output_dir)
    except G3ResultError as error:
        print(f"g3-analysis: error: {error}", file=sys.stderr)
        return 2
    print(f"g3-analysis: wrote {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

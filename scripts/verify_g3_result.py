#!/usr/bin/env python3
"""Verify one closed G3 result bundle without trusting retained summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decodeforge.g3_results import G3ResultError, verify_g3_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        verify_g3_result(arguments.bundle)
    except G3ResultError as error:
        print(f"g3-verification: error: {error}", file=sys.stderr)
        return 2
    print("g3-verification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

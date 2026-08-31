#!/usr/bin/env python3
"""Verify a G0 bundle's portable content and explicit Git provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decodeforge.g0_repository import verify_g0_repository_bundle


def main(argv: list[str] | None = None) -> int:
    """Run the explicit-checkout G0 repository gate without auto-discovery."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    arguments = parser.parse_args(argv)
    diagnostics = verify_g0_repository_bundle(arguments.bundle, arguments.checkout)
    if diagnostics is None:
        print(
            json.dumps(
                {
                    "code": "DFE-BUNDLE-009",
                    "component": "bundle",
                    "context": {"artifact": "run-manifest.json", "field": "milestone"},
                    "schema_version": 1,
                    "severity": "error",
                    "summary": "The repository provenance gate requires milestone g0.",
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    if diagnostics:
        for diagnostic in diagnostics:
            print(
                json.dumps(
                    diagnostic,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        return 1
    print("g0-repository-check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

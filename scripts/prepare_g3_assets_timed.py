#!/usr/bin/env python3
"""Prepare canonical G3 q_proj assets and publish a separate timing receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from decodeforge.g3_preparation import (
    G3PreparationError,
    capture_preparation_receipt,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--prepare-tool", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        projection = capture_preparation_receipt(
            checkout=arguments.checkout,
            source=arguments.source,
            output=arguments.output,
            receipt=arguments.receipt,
            prepare_tool=arguments.prepare_tool,
        )
    except G3PreparationError as error:
        print(f"g3-preparation: rejected: {error}")
        return 2
    print(
        "g3-preparation: ok "
        f"aggregate={projection['asset_inventory_identity']} "
        f"elapsed_ns={projection['elapsed_ns']} "
        f"receipt={projection['receipt_identity']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

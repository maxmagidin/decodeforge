#!/usr/bin/env python3
"""Run the bounded, non-benchmark DecodeForge presentation demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decodeforge.presentation_demo import (
    PresentationDemoError,
    PresentationRequest,
    run_presentation_demo,
    write_demo_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        evidence = run_presentation_demo(
            PresentationRequest(
                model_directory=arguments.model_dir,
                asset_directory=arguments.assets,
                bridge_library=arguments.library,
                bridge_sha256=arguments.library_sha256,
                prompt=arguments.prompt,
                max_new_tokens=arguments.max_new_tokens,
            )
        )
        if arguments.output is not None:
            write_demo_json(arguments.output, evidence)
        print("Generated text:", evidence["comparison"]["text"])
        print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except PresentationDemoError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

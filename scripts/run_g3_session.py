#!/usr/bin/env python3
"""Run one strict-offline frozen G3 generation session."""

from __future__ import annotations

import time

PROCESS_START_NS = time.perf_counter_ns()

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from decodeforge.g3_session import (  # noqa: E402
    SessionRequest,
    publish_new_json,
    run_session,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--session-index", required=True, type=int, choices=range(3))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--preparation-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/g3/spec.json"))
    arguments = parser.parse_args()
    result = run_session(
        SessionRequest(
            session_id=arguments.session_id,
            session_index=arguments.session_index,
            spec_path=arguments.spec,
            model_directory=arguments.model_dir,
            asset_directory=arguments.assets,
            bridge_library=arguments.library,
            bridge_sha256=arguments.library_sha256,
            preparation_receipt=arguments.preparation_receipt,
            session_output=arguments.output,
            process_start_ns=PROCESS_START_NS,
        )
    )
    publish_new_json(arguments.output, result)
    print(f"g3-session: accepted {arguments.session_id} -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one bounded DecodeForge evaluation-v1 correctness/performance session."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROCESS_START_NS = time.perf_counter_ns()

from decodeforge.evaluation import (  # noqa: E402
    EvaluationError,
    EvaluationRequest,
    run_evaluation,
    write_evaluation_json,
)
from decodeforge.presentation_demo import PresentationDemoError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "performance"), required=True)
    parser.add_argument(
        "--model-dir", "--model", dest="model_directory", required=True, type=Path
    )
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--library-sha256",
        "--hash",
        "--bridge-sha256",
        dest="bridge_sha256",
        required=True,
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-index", type=int, choices=range(3), default=0)
    arguments = parser.parse_args()
    try:
        evidence = run_evaluation(
            EvaluationRequest(
                spec_path=arguments.spec.resolve(),
                model_directory=arguments.model_directory,
                asset_directory=arguments.assets,
                bridge_library=arguments.library,
                bridge_sha256=arguments.bridge_sha256,
                mode=arguments.mode,
                session_index=arguments.session_index,
                process_start_ns=PROCESS_START_NS,
            ),
        )
        write_evaluation_json(arguments.output, evidence)
    except (EvaluationError, PresentationDemoError) as error:
        print(f"evaluation: error: {error}", file=sys.stderr)
        failure = {
            "format": "decodeforge_evaluation_v1",
            "g3_evidence": False,
            "mode": arguments.mode,
            "session_index": arguments.session_index,
            "accepted": False,
            "error": str(error),
        }
        try:
            write_evaluation_json(arguments.output, failure)
        except Exception as write_error:
            print(
                f"evaluation: unable to write rejection: {write_error}", file=sys.stderr
            )
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0 if evidence.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())

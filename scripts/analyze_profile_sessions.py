#!/usr/bin/env python3
"""Validate three diagnostic captures and compare their decode cost rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, NoReturn

from decodeforge.profile_analysis import ProfileAnalysisError, analyze_profile_sessions
from decodeforge.profile_capture import (
    ProfileCaptureError,
    check_profile_output,
    write_profile_capture,
)

MAX_CAPTURE_BYTES = 32 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def read_capture(path: Path) -> tuple[dict[str, Any], str]:
    """Read bounded, unambiguous JSON and hash the exact input bytes."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CAPTURE_BYTES:
            raise ValueError("capture must be a regular file of at most 32 MiB")
        raw = stream.read(MAX_CAPTURE_BYTES + 1)
    if len(raw) > MAX_CAPTURE_BYTES:
        raise ValueError("capture exceeds 32 MiB")
    document = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    if not isinstance(document, dict):
        raise ValueError("capture must be a JSON object")
    return document, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True, nargs=3, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        check_profile_output(arguments.output)
        inputs = [read_capture(path) for path in arguments.sessions]
        report = analyze_profile_sessions([document for document, _ in inputs])
        report["input_captures"] = sorted(
            [
                {"session_index": document["session_index"], "sha256": digest}
                for document, digest in inputs
            ],
            key=lambda item: item["session_index"],
        )
        write_profile_capture(arguments.output, report)
    except (
        OSError,
        ValueError,
        RecursionError,
        ProfileAnalysisError,
        ProfileCaptureError,
    ) as error:
        parser.exit(2, f"profile-analysis: rejected: {error}\n")
    print(f"profile-analysis: wrote {arguments.output} (diagnostic only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

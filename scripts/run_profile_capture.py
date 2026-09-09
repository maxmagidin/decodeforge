#!/usr/bin/env python3
"""Capture one fresh-process DecodeForge diagnostic profile session."""

from __future__ import annotations

import argparse
from pathlib import Path

from decodeforge.profile_capture import (
    ProfileCaptureError,
    ProfileCaptureRequest,
    check_profile_output,
    rejected_profile_capture,
    run_profile_capture,
    write_profile_capture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--session-index", required=True, type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = ProfileCaptureRequest(
        model_directory=arguments.model_dir,
        asset_directory=arguments.assets,
        bridge_library=arguments.library,
        bridge_sha256=arguments.library_sha256,
        prompt=arguments.prompt,
        session_index=arguments.session_index,
        max_new_tokens=arguments.max_new_tokens,
    )
    try:
        check_profile_output(arguments.output)
        document = run_profile_capture(request)
        write_profile_capture(arguments.output, document)
        print(f"profile-capture: wrote {arguments.output}")
        return 0
    except ProfileCaptureError as error:
        try:
            write_profile_capture(
                arguments.output, rejected_profile_capture(request, error)
            )
        except ProfileCaptureError as publication_error:
            parser.error(
                f"{error}; rejected output was not published: {publication_error}"
            )
        parser.exit(
            2,
            f"profile-capture: rejected at {error.stage}; wrote {arguments.output}\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())

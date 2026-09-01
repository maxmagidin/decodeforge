#!/usr/bin/env python3
"""Analyze three raw DecodeForge G1 benchmark sessions.

The command validates every input before creating either output.  The JSON
report and Markdown summary contain no timestamps or local paths, so rerunning
the command with the same sessions produces byte-identical output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from decodeforge.g1_evidence import (
    G1AnalysisError,
    analyze_sessions,
    load_sessions,
    write_analysis_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help="three raw run-session JSON files (or use --session repeatedly)",
    )
    parser.add_argument(
        "--session",
        dest="session_options",
        action="append",
        type=Path,
        default=[],
        help="one raw run-session JSON file; may be repeated three times",
    )
    parser.add_argument(
        "--session-id",
        dest="session_ids",
        action="append",
        default=[],
        help=(
            "explicit session identity; repeat three times to assert the file "
            "identities (IDs must be unique)"
        ),
    )
    parser.add_argument(
        "--sessions",
        dest="sessions_option",
        nargs=3,
        type=Path,
        help="exactly three raw run-session JSON files",
    )
    parser.add_argument(
        "--output",
        "--report",
        dest="report",
        type=Path,
        help="machine-readable JSON report path",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Markdown summary path (defaults to report path with .md suffix)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write report.json and report.md under this directory",
    )
    return parser


def _session_paths(args: argparse.Namespace) -> list[Path]:
    choices = [
        paths
        for paths in (
            args.sessions_option,
            args.session_options if args.session_options else None,
            args.sessions if args.sessions else None,
        )
        if paths is not None
    ]
    if len(choices) != 1:
        raise G1AnalysisError(
            "provide exactly one of positional sessions, --sessions, or --session"
        )
    paths = list(choices[0])
    if len(paths) != 3:
        raise G1AnalysisError("exactly three session files are required")
    return paths


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    report = args.report
    if args.output_dir is not None:
        report = args.output_dir / "report.json"
        summary = args.output_dir / "report.md"
    else:
        if report is None:
            raise G1AnalysisError("--output is required unless --output-dir is used")
        summary = args.summary or report.with_suffix(".md")
    if report.resolve() == summary.resolve():
        raise G1AnalysisError(
            "JSON report and Markdown summary must be different files"
        )
    return report, summary


def _check_explicit_session_ids(
    args: argparse.Namespace, sessions: Sequence[dict[str, object]]
) -> None:
    if not args.session_ids:
        return
    if len(args.session_ids) != 3 or len(set(args.session_ids)) != 3:
        raise G1AnalysisError(
            "--session-id must be supplied exactly three times with unique IDs"
        )
    actual = [session.get("session_id") for session in sessions]
    if set(args.session_ids) != set(actual):
        raise G1AnalysisError("explicit --session-id values do not match session files")


def _print_error(error: G1AnalysisError) -> None:
    print(f"g1-analysis: error: {error}", file=sys.stderr)
    for diagnostic in error.diagnostics:
        print(
            json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = _session_paths(args)
        report_path, summary_path = _output_paths(args)
        sessions = load_sessions(paths)
        _check_explicit_session_ids(args, sessions)
        report = analyze_sessions(sessions)
        write_analysis_outputs(report, report_path, summary_path)
    except G1AnalysisError as error:
        _print_error(error)
        return 1
    except (OSError, OverflowError, ValueError, json.JSONDecodeError) as error:
        print(f"g1-analysis: error: {error}", file=sys.stderr)
        return 1
    print(f"g1-analysis: wrote {report_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

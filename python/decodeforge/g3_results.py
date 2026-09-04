"""Deterministic analysis and closed-bundle verification for G3 evidence."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

JsonObject: TypeAlias = dict[str, Any]

PROTOCOL_ID: Final = "g3-tinyllama-qproj-generation-v1"
SESSION_FORMAT: Final = "decodeforge_g3_generation_session_v1"
ANALYSIS_FORMAT: Final = "decodeforge_g3_analysis_v1"
MANIFEST_FORMAT: Final = "decodeforge_g3_result_manifest_v1"
CANONICAL_IDENTITY_ALGORITHM: Final = (
    "sha256_utf8_json_sort_keys_compact_unicode_no_nan_v1"
)
SESSION_COUNT: Final = 3
MAX_SESSION_BYTES: Final = 64 * 1024 * 1024
MAX_BUNDLE_BYTES: Final = 192 * 1024 * 1024
_DARWIN_RENAME_EXCL: Final = 0x00000004
_LINUX_RENAME_NOREPLACE: Final = 1
_AT_FDCWD: Final = -100

_BUNDLE_INVENTORY: Final[tuple[tuple[str, str], ...]] = (
    ("README.md", "human_summary"),
    ("analysis.json", "derived_analysis"),
    ("asset-inventory.json", "asset_provenance"),
    ("correctness.json", "correctness_evidence"),
    ("coverage.json", "dispatch_coverage"),
    ("environment.txt", "environment_capture"),
    ("generated.txt", "demonstration_text"),
    ("manifest.json", "closed_bundle_manifest"),
    ("prompt.json", "prompt_and_token_ids"),
    ("timings.csv", "raw_timing_samples"),
)
_BUNDLE_NAMES: Final = frozenset(path for path, _ in _BUNDLE_INVENTORY)
_FILE_LIMITS: Final[dict[str, int]] = {
    "README.md": 64 * 1024,
    "analysis.json": 96 * 1024 * 1024,
    "asset-inventory.json": 2 * 1024 * 1024,
    "correctness.json": 32 * 1024 * 1024,
    "coverage.json": 48 * 1024 * 1024,
    "environment.txt": 2 * 1024 * 1024,
    "generated.txt": 4 * 1024 * 1024,
    "manifest.json": 256 * 1024,
    "prompt.json": 256 * 1024,
    "timings.csv": 16 * 1024 * 1024,
}


class G3ResultError(ValueError):
    """A G3 input or result bundle violated the closed contract."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value under the session-declared identity algorithm."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def session_identity(session: Mapping[str, Any]) -> str:
    """Return the canonical semantic identity of one session object."""

    return _identity(canonical_json_bytes(session))


def _parse_json(content: bytes, label: str) -> JsonObject:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as error:
        raise G3ResultError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise G3ResultError(f"{label} must contain one JSON object")
    return cast(JsonObject, value)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_path(path: Path, limit: int) -> tuple[bytes, tuple[int, int]]:
    """Read one bounded stable regular-file snapshot without following a symlink."""

    descriptor = -1
    try:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | nofollow
        )
        if nofollow == 0 and stat.S_ISLNK(os.lstat(path).st_mode):
            raise G3ResultError("session inputs must be non-symlink regular files")
        descriptor = os.open(path, flags)
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise G3ResultError("session inputs must be non-symlink regular files")
        if initial.st_size > limit:
            raise G3ResultError(f"session input exceeds the {limit}-byte bound")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > limit:
            raise G3ResultError(f"session input exceeds the {limit}-byte bound")
        final = os.fstat(descriptor)
        path_final = os.stat(path, follow_symlinks=False)
        if _file_identity(initial) != _file_identity(final) or _file_identity(
            initial
        ) != _file_identity(path_final):
            raise G3ResultError("session input changed while it was read")
        return content, (initial.st_dev, initial.st_ino)
    except G3ResultError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise G3ResultError(
                "session inputs must be non-symlink regular files"
            ) from error
        raise G3ResultError("session input could not be read safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_g3_sessions(paths: Sequence[Path]) -> list[JsonObject]:
    """Read exactly three distinct bounded session snapshots."""

    if len(paths) != SESSION_COUNT:
        raise G3ResultError("G3 analysis requires exactly three session files")
    sessions: list[JsonObject] = []
    file_identities: set[tuple[int, int]] = set()
    for path in paths:
        content, file_identity = _read_regular_path(path, MAX_SESSION_BYTES)
        if file_identity in file_identities:
            raise G3ResultError("G3 session inputs must be distinct files")
        file_identities.add(file_identity)
        sessions.append(_parse_json(content, "session input"))
    return sessions


def _without_process_id(environment: Mapping[str, Any]) -> JsonObject:
    return {
        key: value for key, value in environment.items() if key != "runner_process_id"
    }


def analyze_g3_sessions(sessions: Sequence[JsonObject]) -> list[JsonObject]:
    """Validate and normalize the three independent accepted G3 sessions."""

    from decodeforge.contracts import validate_data

    if len(sessions) != SESSION_COUNT:
        raise G3ResultError("G3 analysis requires exactly three sessions")
    normalized = sorted(sessions, key=lambda item: item.get("session_index", -1))
    for index, session in enumerate(normalized):
        diagnostics = validate_data(session, "g3-generation-session")
        if diagnostics:
            raise G3ResultError(
                f"session {index} violates the G3 generation-session contract"
            )
        if session["state"] != {
            "status": "accepted",
            "stage": "completed",
            "accepted": True,
            "rejection_reasons": [],
        }:
            raise G3ResultError("all analyzed G3 sessions must be accepted")
    if [session["session_index"] for session in normalized] != [0, 1, 2]:
        raise G3ResultError("G3 session_index values must be exactly 0, 1, and 2")
    session_ids = [cast(str, session["session_id"]) for session in normalized]
    process_ids = [
        cast(int, cast(JsonObject, session["environment"])["runner_process_id"])
        for session in normalized
    ]
    if len(set(session_ids)) != SESSION_COUNT:
        raise G3ResultError("G3 session IDs must be distinct")
    if len(set(process_ids)) != SESSION_COUNT:
        raise G3ResultError("G3 runner process IDs must be distinct")

    first = normalized[0]
    shared_fields = (
        "protocol_id",
        "format",
        "canonical_identity_algorithm",
        "experiment_spec",
        "bundle_inventory",
        "prompt",
        "provenance",
        "assets",
        "offline_preparation",
    )
    for session in normalized[1:]:
        if any(session[field] != first[field] for field in shared_fields):
            raise G3ResultError(
                "G3 sessions must share spec, prompt, asset, and preparation provenance"
            )
        if _without_process_id(cast(JsonObject, session["environment"])) != (
            _without_process_id(cast(JsonObject, first["environment"]))
        ):
            raise G3ResultError(
                "G3 sessions must share one environment apart from process ID"
            )
    return normalized


def _median_fraction(values: Sequence[int]) -> JsonObject:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise G3ResultError("cannot summarize an empty timing sample")
    if count % 2:
        return {"numerator": ordered[count // 2], "denominator": 1}
    return {
        "numerator": ordered[count // 2 - 1] + ordered[count // 2],
        "denominator": 2,
    }


def _analysis_document(sessions: list[JsonObject]) -> JsonObject:
    timing_summary = []
    for path in ("same_q8_reference", "hybrid_native"):
        samples = [
            cast(int, cast(JsonObject, run["timing"])["total_ns"])
            for session in sessions
            for run in cast(list[JsonObject], session["runs"])
            if run["phase"] == "measured" and run["path"] == path
        ]
        timing_summary.append(
            {
                "path": path,
                "sample_count": len(samples),
                "minimum_ns": min(samples),
                "maximum_ns": max(samples),
                "median_ns": _median_fraction(samples),
            }
        )
    return {
        "schema_version": 1,
        "format": ANALYSIS_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "canonical_identity_algorithm": CANONICAL_IDENTITY_ALGORITHM,
        "sessions": sessions,
        "summary": {
            "accepted_session_count": len(sessions),
            "session_indices": [session["session_index"] for session in sessions],
            "session_ids": [session["session_id"] for session in sessions],
            "runner_process_ids": [
                cast(JsonObject, session["environment"])["runner_process_id"]
                for session in sessions
            ],
            "all_correctness_pass": all(
                cast(JsonObject, session["correctness"])["overall_pass"]
                for session in sessions
            ),
            "all_reconciliation_pass": all(
                cast(JsonObject, session["reconciliation"])["pass"]
                for session in sessions
            ),
            "all_drift_pass": all(
                cast(JsonObject, session["drift"])["pass"] for session in sessions
            ),
            "measured_total_generation": timing_summary,
        },
    }


def _asset_inventory_document(session: JsonObject) -> JsonObject:
    assets = cast(JsonObject, session["assets"])
    entries = [
        {key: value for key, value in entry.items() if key != "layer_path"}
        for entry in cast(list[JsonObject], assets["entries"])
    ]
    return {
        "schema_version": 1,
        "format": assets["format"],
        "source": assets["source"],
        "layer_count": assets["layer_count"],
        "entries": entries,
        "total_packed_bytes": assets["total_packed_bytes"],
        "total_fallback_bytes": assets["total_fallback_bytes"],
        "aggregate_identity": assets["aggregate_identity"],
    }


def _correctness_document(sessions: list[JsonObject]) -> JsonObject:
    records = []
    for session in sessions:
        run_evidence = [
            {
                "order_index": run["order_index"],
                "phase": run["phase"],
                "repetition": run["repetition"],
                "path": run["path"],
                "output_ids": run["output_ids"],
                "steps": run["steps"],
                "native_output_checks": run["native_output_checks"],
            }
            for run in cast(list[JsonObject], session["runs"])
        ]
        records.append(
            {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "correctness": session["correctness"],
                "run_evidence": run_evidence,
            }
        )
    return {
        "schema_version": 1,
        "format": "decodeforge_g3_correctness_v1",
        "protocol_id": PROTOCOL_ID,
        "sessions": records,
    }


def _coverage_document(sessions: list[JsonObject]) -> JsonObject:
    records = []
    for session in sessions:
        run_counters = [
            {
                "order_index": run["order_index"],
                "phase": run["phase"],
                "repetition": run["repetition"],
                "path": run["path"],
                "counters": run["counters"],
            }
            for run in cast(list[JsonObject], session["runs"])
        ]
        records.append(
            {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "run_counters": run_counters,
                "reconciliation": session["reconciliation"],
            }
        )
    return {
        "schema_version": 1,
        "format": "decodeforge_g3_coverage_v1",
        "protocol_id": PROTOCOL_ID,
        "sessions": records,
    }


def _prompt_document(sessions: list[JsonObject]) -> JsonObject:
    return {
        "schema_version": 1,
        "format": "decodeforge_g3_prompt_v1",
        "protocol_id": PROTOCOL_ID,
        "prompt": sessions[0]["prompt"],
        "sessions": [
            {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "outputs": [
                    {
                        "order_index": run["order_index"],
                        "path": run["path"],
                        "output_ids": run["output_ids"],
                    }
                    for run in cast(list[JsonObject], session["runs"])
                ],
            }
            for session in sessions
        ],
    }


def _environment_text(sessions: list[JsonObject]) -> bytes:
    records = [
        {
            "session_index": session["session_index"],
            "session_id": session["session_id"],
            "environment": session["environment"],
            "provenance": session["provenance"],
            "offline_preparation": session["offline_preparation"],
            "startup_timings": session["startup_timings"],
        }
        for session in sessions
    ]
    lines = ["DecodeForge G3 environment capture v1"]
    lines.extend(canonical_json_bytes(record).decode("utf-8") for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _generated_text(sessions: list[JsonObject]) -> bytes:
    lines = ["DecodeForge G3 generated text v1"]
    for session in sessions:
        for run in cast(list[JsonObject], session["runs"]):
            record = {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "order_index": run["order_index"],
                "phase": run["phase"],
                "repetition": run["repetition"],
                "path": run["path"],
                "output_ids": run["output_ids"],
                "decoded_text": run["decoded_text"],
                "text_role": run["text_role"],
            }
            lines.append(canonical_json_bytes(record).decode("utf-8"))
    return ("\n".join(lines) + "\n").encode("utf-8")


_TIMING_COLUMNS: Final = (
    "session_index",
    "session_id",
    "record_type",
    "run_order_index",
    "phase",
    "repetition",
    "path",
    "step_index",
    "layer",
    "layer_path",
    "dispatch",
    "metric",
    "value",
    "unit",
    "detail",
)


def _timing_csv(sessions: list[JsonObject]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_TIMING_COLUMNS, lineterminator="\n")
    writer.writeheader()

    def emit(**values: object) -> None:
        row: dict[str, object] = dict.fromkeys(_TIMING_COLUMNS, "")
        row.update(values)
        writer.writerow(row)

    for session in sessions:
        common = {
            "session_index": session["session_index"],
            "session_id": session["session_id"],
        }
        preparation = cast(JsonObject, session["offline_preparation"])
        emit(
            **common,
            record_type="offline_preparation",
            metric="elapsed_ns",
            value=preparation["elapsed_ns"],
            unit="ns",
            detail=preparation["receipt_identity"],
        )
        startup = cast(JsonObject, session["startup_timings"])
        for metric in (
            "cold_startup_ns",
            "tokenizer_load_ns",
            "model_load_ns",
            "install_ns",
        ):
            emit(
                **common,
                record_type="startup",
                metric=metric,
                value=startup[metric],
                unit="ns",
            )
        for metric in ("baseline_rss_bytes", "peak_rss_bytes"):
            emit(
                **common,
                record_type="memory",
                metric=metric,
                value=startup[metric],
                unit="bytes",
            )
        for run in cast(list[JsonObject], session["runs"]):
            run_fields = {
                **common,
                "run_order_index": run["order_index"],
                "phase": run["phase"],
                "repetition": run["repetition"],
                "path": run["path"],
            }
            timing = cast(JsonObject, run["timing"])
            for metric in (
                "prefill_ns",
                "time_to_first_token_ns",
                "q_projection_ns",
                "native_work_ns",
                "total_ns",
            ):
                value = timing[metric]
                emit(
                    **run_fields,
                    record_type="run",
                    metric=metric,
                    value="" if value is None else value,
                    unit="ns",
                    detail=(
                        timing["native_work_unavailable_reason"]
                        if metric == "native_work_ns" and value is None
                        else ""
                    ),
                )
            for step_index, elapsed in enumerate(
                cast(list[int], timing["cached_step_ns"]), start=1
            ):
                emit(
                    **run_fields,
                    record_type="cached_step",
                    step_index=step_index,
                    metric="cached_step_ns",
                    value=elapsed,
                    unit="ns",
                )
            for dispatch in cast(list[JsonObject], timing["q_projection_dispatches"]):
                emit(
                    **run_fields,
                    record_type="q_projection_dispatch",
                    step_index=dispatch["step_index"],
                    layer=dispatch["layer"],
                    layer_path=dispatch["layer_path"],
                    dispatch=dispatch["dispatch"],
                    metric="dispatch_ns",
                    value=dispatch["dispatch_ns"],
                    unit="ns",
                )
    return output.getvalue().encode("utf-8")


def _readme(sessions: list[JsonObject], analysis: JsonObject) -> bytes:
    summary = cast(JsonObject, analysis["summary"])
    lines = [
        "# DecodeForge G3 TinyLlama generation evidence",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        "This closed bundle contains three independently captured, schema-valid ",
        "accepted sessions. Generated text is demonstration output only; direct ",
        "operator, model-logit, token, and dispatch evidence determine correctness.",
        "",
        f"Accepted sessions: {summary['accepted_session_count']}",
        "",
        "| Path | Measured samples | Minimum ns | Median fraction ns | Maximum ns |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in cast(list[JsonObject], summary["measured_total_generation"]):
        median = cast(JsonObject, record["median_ns"])
        lines.append(
            f"| `{record['path']}` | {record['sample_count']} | "
            f"{record['minimum_ns']} | {median['numerator']}/{median['denominator']} | "
            f"{record['maximum_ns']} |"
        )
    lines.extend(
        [
            "",
            "`analysis.json` retains the canonical session objects needed to ",
            "reconstruct and independently regenerate every bundle member.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _manifest_document(
    sessions: list[JsonObject], contents: Mapping[str, bytes]
) -> JsonObject:
    roles = dict(_BUNDLE_INVENTORY)
    manifest: JsonObject = {
        "schema_version": 1,
        "format": MANIFEST_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "canonical_identity_algorithm": CANONICAL_IDENTITY_ALGORITHM,
        "inventory": [{"path": path, "role": role} for path, role in _BUNDLE_INVENTORY],
        "sessions": [
            {
                "session_index": session["session_index"],
                "session_id": session["session_id"],
                "canonical_identity": session_identity(session),
            }
            for session in sessions
        ],
        "artifacts": [
            {
                "path": path,
                "role": roles[path],
                "bytes": len(contents[path]),
                "sha256": hashlib.sha256(contents[path]).hexdigest(),
            }
            for path, _ in _BUNDLE_INVENTORY
            if path != "manifest.json"
        ],
    }
    manifest["bundle_identity"] = _identity(
        b"DecodeForge/G3/result-bundle/v1\0" + canonical_json_bytes(manifest)
    )
    return manifest


def build_g3_bundle(sessions: Sequence[JsonObject]) -> dict[str, bytes]:
    """Return all ten deterministic bundle members from validated raw sessions."""

    normalized = analyze_g3_sessions(sessions)
    analysis = _analysis_document(normalized)
    contents = {
        "README.md": _readme(normalized, analysis),
        "analysis.json": canonical_json_bytes(analysis),
        "asset-inventory.json": canonical_json_bytes(
            _asset_inventory_document(normalized[0])
        ),
        "correctness.json": canonical_json_bytes(_correctness_document(normalized)),
        "coverage.json": canonical_json_bytes(_coverage_document(normalized)),
        "environment.txt": _environment_text(normalized),
        "generated.txt": _generated_text(normalized),
        "prompt.json": canonical_json_bytes(_prompt_document(normalized)),
        "timings.csv": _timing_csv(normalized),
    }
    contents["manifest.json"] = canonical_json_bytes(
        _manifest_document(normalized, contents)
    )
    if set(contents) != _BUNDLE_NAMES:
        raise AssertionError("internal G3 bundle inventory is not closed")
    if any(len(content) > _FILE_LIMITS[name] for name, content in contents.items()):
        raise G3ResultError("generated G3 bundle exceeds a closed member size bound")
    if sum(map(len, contents.values())) > MAX_BUNDLE_BYTES:
        raise G3ResultError("generated G3 bundle exceeds its aggregate size bound")
    return contents


def _read_bundle(bundle: Path) -> dict[str, bytes]:
    root_fd = -1
    try:
        initial_path = os.lstat(bundle)
        if not stat.S_ISDIR(initial_path.st_mode):
            raise G3ResultError("G3 bundle root must be a non-symlink directory")
        root_fd = os.open(
            bundle,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        initial_root = os.fstat(root_fd)
        names = set(os.listdir(root_fd))
        if names != _BUNDLE_NAMES:
            missing = sorted(_BUNDLE_NAMES - names)
            extra = sorted(names - _BUNDLE_NAMES)
            raise G3ResultError(
                f"G3 bundle inventory mismatch (missing={missing}, extra={extra})"
            )
        contents: dict[str, bytes] = {}
        total = 0
        for name, _ in _BUNDLE_INVENTORY:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise G3ResultError(f"G3 member {name} is not a regular file")
                limit = _FILE_LIMITS[name]
                if before.st_size > limit:
                    raise G3ResultError(f"G3 member {name} exceeds its size bound")
                chunks: list[bytes] = []
                remaining = limit + 1
                while remaining:
                    chunk = os.read(descriptor, min(64 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                content = b"".join(chunks)
                after = os.fstat(descriptor)
                if len(content) > limit or _file_identity(before) != _file_identity(
                    after
                ):
                    raise G3ResultError(f"G3 member {name} changed while read")
                contents[name] = content
                total += len(content)
                if total > MAX_BUNDLE_BYTES:
                    raise G3ResultError("G3 bundle exceeds its aggregate size bound")
            finally:
                os.close(descriptor)
        if set(os.listdir(root_fd)) != names or _file_identity(
            os.fstat(root_fd)
        ) != _file_identity(initial_root):
            raise G3ResultError("G3 bundle inventory changed while read")
        final_path = os.stat(bundle, follow_symlinks=False)
        if _file_identity(initial_path) != _file_identity(final_path):
            raise G3ResultError("G3 bundle root changed while read")
        return contents
    except G3ResultError:
        raise
    except OSError as error:
        raise G3ResultError("G3 bundle could not be read safely") from error
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def verify_g3_result(bundle: Path) -> None:
    """Verify one closed bundle and independently regenerate every member."""

    contents = _read_bundle(bundle)
    analysis = _parse_json(contents["analysis.json"], "analysis.json")
    if set(analysis) != {
        "schema_version",
        "format",
        "protocol_id",
        "canonical_identity_algorithm",
        "sessions",
        "summary",
    }:
        raise G3ResultError("analysis.json has an open or incomplete root")
    if (
        analysis["schema_version"] != 1
        or analysis["format"] != ANALYSIS_FORMAT
        or analysis["protocol_id"] != PROTOCOL_ID
        or analysis["canonical_identity_algorithm"] != CANONICAL_IDENTITY_ALGORITHM
        or not isinstance(analysis["sessions"], list)
    ):
        raise G3ResultError("analysis.json has the wrong closed contract identity")
    sessions = cast(list[JsonObject], analysis["sessions"])
    expected = build_g3_bundle(sessions)
    for name, _ in _BUNDLE_INVENTORY:
        if contents[name] != expected[name]:
            raise G3ResultError(f"G3 member {name} does not recompute exactly")


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_no_overwrite(staging: Path, target: Path) -> None:
    system = platform.system()
    libc = ctypes.CDLL(None, use_errno=True)
    if system == "Darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise G3ResultError("atomic no-overwrite rename is unavailable") from error
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staging), os.fsencode(target), _DARWIN_RENAME_EXCL)
    elif system == "Linux":
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise G3ResultError("atomic no-overwrite rename is unavailable") from error
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            _AT_FDCWD,
            os.fsencode(staging),
            _AT_FDCWD,
            os.fsencode(target),
            _LINUX_RENAME_NOREPLACE,
        )
    else:
        raise G3ResultError("atomic no-overwrite rename is unsupported on this host")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise G3ResultError("G3 output target already exists")
    raise G3ResultError("G3 bundle could not be atomically installed")


def write_g3_bundle(sessions: Sequence[JsonObject], output: Path) -> None:
    """Durably stage, self-verify, and atomically install one new G3 bundle."""

    if output.name in {"", ".", ".."}:
        raise G3ResultError("G3 output must name an explicit directory")
    try:
        parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise G3ResultError("G3 output parent is unavailable") from error
    if not parent.is_dir():
        raise G3ResultError("G3 output parent is unavailable")
    target = parent / output.name
    if target.exists() or target.is_symlink():
        raise G3ResultError("G3 output target already exists")
    contents = build_g3_bundle(sessions)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    installed = False
    try:
        for name, _ in _BUNDLE_INVENTORY:
            _write_new(staging / name, contents[name])
        _fsync_directory(staging)
        verify_g3_result(staging)
        _install_no_overwrite(staging, target)
        installed = True
        _fsync_directory(parent)
    except G3ResultError:
        raise
    except OSError as error:
        raise G3ResultError("G3 bundle publication failed") from error
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)


def analyze_and_write_g3_result(paths: Sequence[Path], output: Path) -> None:
    """Load three session paths and publish their closed result bundle."""

    write_g3_bundle(load_g3_sessions(paths), output)

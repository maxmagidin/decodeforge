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
import secrets
import shlex
import stat
from collections.abc import Mapping, Sequence
from contextlib import suppress
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


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, left.st_mode) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
    )


def _require_descriptor_features() -> tuple[int, int]:
    """Fail closed when the host cannot provide no-follow directory walks."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if nofollow == 0 or directory == 0:
        raise G3ResultError(
            "secure G3 filesystem access requires O_NOFOLLOW and O_DIRECTORY"
        )
    return nofollow, directory


def _open_parent_nofollow(path: Path) -> tuple[int, str, os.stat_result]:
    """Open every existing parent component without following symlinks."""

    if path.name in {"", ".", ".."} or ".." in path.parent.parts:
        raise G3ResultError("path must have one explicit leaf without traversal")
    nofollow, directory = _require_descriptor_features()
    flags = os.O_RDONLY | directory | getattr(os, "O_CLOEXEC", 0) | nofollow
    descriptor = os.open("/" if path.is_absolute() else ".", flags)
    components = path.parent.parts[1:] if path.is_absolute() else path.parent.parts
    try:
        for component in components:
            if component in {"", "."}:
                continue
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, path.name, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def _parent_path_unchanged(path: Path, expected: os.stat_result) -> bool:
    descriptor = -1
    try:
        descriptor, _, observed = _open_parent_nofollow(path)
        return (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino)
    except (G3ResultError, OSError):
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stat_at(parent_fd: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)


def _read_regular_path(path: Path, limit: int) -> tuple[bytes, tuple[int, int]]:
    """Read one bounded stable regular-file snapshot without following a symlink."""

    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, name, parent_metadata = _open_parent_nofollow(path)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        initial = os.fstat(descriptor)
        named_initial = _stat_at(parent_fd, name)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_nlink != 1
            or _file_identity(initial) != _file_identity(named_initial)
        ):
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
        path_final = _stat_at(parent_fd, name)
        if _file_identity(initial) != _file_identity(final) or _file_identity(
            initial
        ) != _file_identity(path_final):
            raise G3ResultError("session input changed while it was read")
        if not _parent_path_unchanged(path, parent_metadata):
            raise G3ResultError("session input ancestor changed while it was read")
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
        if parent_fd >= 0:
            os.close(parent_fd)


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


def _invariant_provenance(provenance: Mapping[str, Any]) -> JsonObject:
    commands = cast(JsonObject, provenance["rebuild_commands"])
    return {
        "checkout": provenance["checkout"],
        "model": provenance["model"],
        "bridge_library": provenance["bridge_library"],
        "asset_inventory_identity": provenance["asset_inventory_identity"],
        "rebuild_commands": {
            "build_bridge": commands["build_bridge"],
            "prepare_assets": commands["prepare_assets"],
        },
    }


def _bind_preparation_receipt(
    sessions: Sequence[JsonObject], receipt: Mapping[str, Any]
) -> JsonObject:
    """Validate and bind the portable full receipt to all session evidence."""

    from decodeforge.g3_preparation import (
        G3PreparationError,
        validate_preparation_receipt_document,
    )

    try:
        document = validate_preparation_receipt_document(receipt)
    except G3PreparationError as error:
        raise G3ResultError("preparation receipt is invalid") from error
    checkout = cast(JsonObject, document["checkout"])
    command = cast(JsonObject, document["command"])
    source = cast(JsonObject, document["source"])
    output = cast(JsonObject, document["output"])
    timing = cast(JsonObject, document["timing"])
    projection = {
        "source": "separately_captured_prepare_command",
        "receipt_identity": document["receipt_identity"],
        "elapsed_ns": timing["elapsed_ns"],
        "asset_inventory_identity": output["asset_inventory_identity"],
    }
    prepare_command = shlex.join(cast(list[str], command["argv"]))
    for session in sessions:
        assets = cast(JsonObject, session["assets"])
        provenance = cast(JsonObject, session["provenance"])
        rebuild = cast(JsonObject, provenance["rebuild_commands"])
        bound_output = {
            "asset_inventory_identity": assets["aggregate_identity"],
            "layer_count": assets["layer_count"],
            "total_packed_bytes": assets["total_packed_bytes"],
            "total_fallback_bytes": assets["total_fallback_bytes"],
        }
        if (
            session["offline_preparation"] != projection
            or provenance["checkout"] != checkout
            or provenance["model"] != source
            or assets["source"] != source
            or bound_output != output
            or provenance["asset_inventory_identity"]
            != output["asset_inventory_identity"]
            or rebuild["prepare_assets"] != prepare_command
        ):
            raise G3ResultError(
                "preparation receipt does not bind exactly to every G3 session"
            )
    return document


def analyze_g3_sessions(sessions: Sequence[JsonObject]) -> list[JsonObject]:
    """Validate and normalize the three independent accepted G3 sessions."""

    from decodeforge.contracts import validate_data

    if len(sessions) != SESSION_COUNT:
        raise G3ResultError("G3 analysis requires exactly three sessions")
    for index, session in enumerate(sessions):
        if not isinstance(session, dict):
            raise G3ResultError(f"session {index} must be one JSON object")
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
    normalized = sorted(sessions, key=lambda item: cast(int, item["session_index"]))
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
        if _invariant_provenance(cast(JsonObject, session["provenance"])) != (
            _invariant_provenance(cast(JsonObject, first["provenance"]))
        ):
            raise G3ResultError(
                "G3 sessions must share checkout, model, bridge, asset, build, "
                "and preparation provenance"
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


def _analysis_document(
    sessions: list[JsonObject], preparation_receipt: JsonObject
) -> JsonObject:
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
        "preparation_receipt": preparation_receipt,
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


def build_g3_bundle(
    sessions: Sequence[JsonObject], preparation_receipt: Mapping[str, Any]
) -> dict[str, bytes]:
    """Return all ten deterministic bundle members from validated raw sessions."""

    normalized = analyze_g3_sessions(sessions)
    receipt = _bind_preparation_receipt(normalized, preparation_receipt)
    analysis = _analysis_document(normalized, receipt)
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


def _read_bundle_fd(root_fd: int) -> dict[str, bytes]:
    """Snapshot a bundle through one already anchored directory descriptor."""

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
            named_before = _stat_at(root_fd, name)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or _file_identity(before) != _file_identity(named_before)
            ):
                raise G3ResultError(
                    f"G3 member {name} is not a singly linked regular file"
                )
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
            named_after = _stat_at(root_fd, name)
            if (
                len(content) > limit
                or _file_identity(before) != _file_identity(after)
                or _file_identity(before) != _file_identity(named_after)
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
    return contents


def _read_bundle(bundle: Path) -> dict[str, bytes]:
    parent_fd = -1
    root_fd = -1
    try:
        parent_fd, name, parent_metadata = _open_parent_nofollow(bundle)
        root_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        initial_root = os.fstat(root_fd)
        named_root = _stat_at(parent_fd, name)
        if not stat.S_ISDIR(initial_root.st_mode) or _file_identity(
            initial_root
        ) != _file_identity(named_root):
            raise G3ResultError("G3 bundle root must be a non-symlink directory")
        contents = _read_bundle_fd(root_fd)
        final_root = _stat_at(parent_fd, name)
        if _file_identity(initial_root) != _file_identity(final_root):
            raise G3ResultError("G3 bundle root changed while read")
        if not _parent_path_unchanged(bundle, parent_metadata):
            raise G3ResultError("G3 bundle ancestor changed while read")
        return contents
    except G3ResultError:
        raise
    except OSError as error:
        raise G3ResultError("G3 bundle could not be read safely") from error
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _verify_contents(contents: Mapping[str, bytes]) -> None:
    analysis = _parse_json(contents["analysis.json"], "analysis.json")
    if set(analysis) != {
        "schema_version",
        "format",
        "protocol_id",
        "canonical_identity_algorithm",
        "preparation_receipt",
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
    receipt = cast(JsonObject, analysis["preparation_receipt"])
    expected = build_g3_bundle(sessions, receipt)
    for name, _ in _BUNDLE_INVENTORY:
        if contents[name] != expected[name]:
            raise G3ResultError(f"G3 member {name} does not recompute exactly")


def verify_g3_result(bundle: Path) -> None:
    """Verify one closed bundle and independently regenerate every member."""

    _verify_contents(_read_bundle(bundle))


def _write_new_at(parent_fd: int, name: str, content: bytes) -> os.stat_result:
    descriptor = -1
    created: os.stat_result | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = os.fstat(descriptor)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        named = _stat_at(parent_fd, name)
        if metadata.st_nlink != 1 or _file_identity(metadata) != _file_identity(named):
            raise G3ResultError("staged G3 member changed while written")
        return metadata
    except BaseException:
        if descriptor >= 0 and created is not None:
            try:
                held = os.fstat(descriptor)
                named = _stat_at(parent_fd, name)
                if _same_object(held, created) and _same_object(named, created):
                    os.unlink(name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory_fd(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise G3ResultError("G3 output directory could not be synchronized") from error


def _new_staging(parent_fd: int, target_name: str) -> tuple[str, int, os.stat_result]:
    nofollow, directory = _require_descriptor_features()
    flags = os.O_RDONLY | directory | getattr(os, "O_CLOEXEC", 0) | nofollow
    for _ in range(128):
        name = f".{target_name}.staging-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        created: os.stat_result | None = None
        descriptor = -1
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
            created = os.fstat(descriptor)
            named = _stat_at(parent_fd, name)
            if not _same_object(created, named):
                raise G3ResultError("G3 staging directory changed while created")
            return name, descriptor, created
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(OSError):
                named = _stat_at(parent_fd, name)
                if created is not None and _same_object(named, created):
                    os.rmdir(name, dir_fd=parent_fd)
            raise
    raise G3ResultError("G3 staging directory could not be allocated")


def _install_no_overwrite(parent_fd: int, staging: str, target: str) -> None:
    system = platform.system()
    libc = ctypes.CDLL(None, use_errno=True)
    if system == "Darwin":
        try:
            rename = libc.renameatx_np
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
            parent_fd,
            os.fsencode(staging),
            parent_fd,
            os.fsencode(target),
            _DARWIN_RENAME_EXCL,
        )
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
            parent_fd,
            os.fsencode(staging),
            parent_fd,
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


def _cleanup_staging(
    parent_fd: int,
    staging_name: str,
    staging_fd: int,
    staging_metadata: os.stat_result,
    members: Mapping[str, os.stat_result],
) -> bool:
    """Remove only the still-anchored staging objects created by this call."""

    try:
        named_staging = _stat_at(parent_fd, staging_name)
        if not _same_object(named_staging, staging_metadata):
            return False
        for name, expected in members.items():
            observed = _stat_at(staging_fd, name)
            if not _same_object(observed, expected):
                return False
        for name in members:
            os.unlink(name, dir_fd=staging_fd)
        if os.listdir(staging_fd):
            return False
        os.rmdir(staging_name, dir_fd=parent_fd)
        return True
    except OSError:
        return False


def write_g3_bundle(
    sessions: Sequence[JsonObject], preparation_receipt: Mapping[str, Any], output: Path
) -> None:
    """Durably stage, self-verify, and atomically install one new G3 bundle."""

    contents = build_g3_bundle(sessions, preparation_receipt)
    parent_fd = -1
    staging_fd = -1
    staging_name = ""
    staging_metadata: os.stat_result | None = None
    members: dict[str, os.stat_result] = {}
    installed_name: str | None = None
    published = False
    try:
        parent_fd, target_name, parent_metadata = _open_parent_nofollow(output)
        try:
            _stat_at(parent_fd, target_name)
        except FileNotFoundError:
            pass
        else:
            raise G3ResultError("G3 output target already exists")
        staging_name, staging_fd, staging_metadata = _new_staging(
            parent_fd, target_name
        )
        for name, _ in _BUNDLE_INVENTORY:
            members[name] = _write_new_at(staging_fd, name, contents[name])
        _fsync_directory_fd(staging_fd)
        staging_metadata = os.fstat(staging_fd)
        _verify_contents(_read_bundle_fd(staging_fd))
        if not _parent_path_unchanged(output, parent_metadata):
            raise G3ResultError("G3 output parent changed during publication")
        _install_no_overwrite(parent_fd, staging_name, target_name)
        installed_name = target_name
        installed_metadata = _stat_at(parent_fd, target_name)
        if not _same_object(installed_metadata, staging_metadata):
            raise G3ResultError("installed G3 bundle does not match staging")
        _verify_contents(_read_bundle_fd(staging_fd))
        try:
            _fsync_directory_fd(parent_fd)
        except G3ResultError as error:
            raise G3ResultError(
                "G3 bundle was installed but its parent sync failed"
            ) from error
        if not _parent_path_unchanged(output, parent_metadata):
            raise G3ResultError(
                "G3 bundle was installed in an anchored directory whose visible "
                "parent path changed"
            )
        published = True
    except G3ResultError:
        raise
    except OSError as error:
        raise G3ResultError("G3 bundle publication failed") from error
    finally:
        if (
            not published
            and parent_fd >= 0
            and staging_fd >= 0
            and staging_metadata is not None
        ):
            cleaned = _cleanup_staging(
                parent_fd,
                installed_name or staging_name,
                staging_fd,
                staging_metadata,
                members,
            )
            if cleaned:
                with suppress(OSError):
                    os.fsync(parent_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def analyze_and_write_g3_result(
    paths: Sequence[Path], preparation_receipt: Path, output: Path
) -> None:
    """Load three session paths and publish their closed result bundle."""

    from decodeforge.g3_preparation import (
        G3PreparationError,
        load_preparation_receipt_document,
        verify_preparation_receipt,
    )

    try:
        verified = verify_preparation_receipt(preparation_receipt)
        document = load_preparation_receipt_document(preparation_receipt)
    except G3PreparationError as error:
        raise G3ResultError("preparation receipt could not be verified") from error
    tool = cast(JsonObject, document["tool"])
    checkout = cast(JsonObject, document["checkout"])
    command = cast(JsonObject, document["command"])
    if (
        verified.session_projection()
        != {
            "source": "separately_captured_prepare_command",
            "receipt_identity": document["receipt_identity"],
            "elapsed_ns": cast(JsonObject, document["timing"])["elapsed_ns"],
            "asset_inventory_identity": cast(JsonObject, document["output"])[
                "asset_inventory_identity"
            ],
        }
        or verified.checkout_revision != checkout["revision"]
        or verified.command_argv != tuple(cast(list[str], command["argv"]))
        or verified.tool_executable_identity != tool["executable_identity"]
    ):
        raise G3ResultError("preparation receipt changed while it was verified")
    write_g3_bundle(load_g3_sessions(paths), document, output)

"""Canonical data contracts shared by G0 evidence producers and verifiers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias

JsonObject: TypeAlias = dict[str, Any]

RUN_MANIFEST_NAME: Final = "run-manifest.json"
BUNDLE_ID_PREFIX: Final = b"DecodeForge/run-bundle/v1\0"
G0_ARTIFACTS: Final[tuple[tuple[str, str], ...]] = (
    ("fixture-manifest.json", "fixture-manifest"),
    ("host.json", "host-manifest"),
    ("report.md", "report"),
)
G0_REQUIRED_CHECKS: Final[dict[str, str]] = {
    "schema": "pass",
    "correctness": "pass",
    "assembly": "not-applicable",
    "certified_performance": "not-applicable",
}
G0_NOT_APPLICABLE: Final[tuple[str, ...]] = (
    "assembly",
    "certified_performance",
)
G0_REPRODUCTION_COMMANDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "schema-contracts-v1",
        (
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/validate_schemas.py",
            "--all",
        ),
    ),
    (
        "q8-python-fixtures-v1",
        (
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/generate_q8_fixtures.py",
            "--check",
        ),
    ),
    ("q8-rust-fixtures-v1", ("make", "rust-fixture-check")),
)
G0_REPRODUCTION_ENV_KEYS: Final[tuple[str, ...]] = (
    "CARGO_NET_OFFLINE",
    "DECODEFORGE_SOURCE_REVISION",
    "UV_OFFLINE",
)
G0_ALLOWED_PATHS: Final[frozenset[str]] = frozenset(
    {RUN_MANIFEST_NAME, *(path for path, _ in G0_ARTIFACTS)}
)
G0_INVENTORY_ENTRY_CAP: Final = len(G0_ALLOWED_PATHS) + 1
_CANONICAL_JSON_MAX_DEPTH: Final = 64
G0_FILE_LIMITS: Final[dict[str, int]] = {
    RUN_MANIFEST_NAME: 64 * 1024,
    "fixture-manifest.json": 64 * 1024,
    "host.json": 32 * 1024,
    "report.md": 256 * 1024,
}


class _SnapshotError(Exception):
    """A path did not yield one bounded regular-file snapshot."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Snapshot:
    """One byte snapshot and the descriptor identity that produced it."""

    content: bytes
    signature: tuple[int, int, int, int, int, int]


def _diagnostic(
    code: str, summary: str, context: JsonObject, *, component: str = "bundle"
) -> JsonObject:
    return {
        "schema_version": 1,
        "code": code,
        "severity": "error",
        "component": component,
        "summary": summary,
        "context": context,
    }


def _snapshot_signature(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_flags(*, directory: bool = False) -> int:
    """Return the non-following descriptor flags used for G0 snapshots."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    return flags


def _open_bundle_root(bundle: Path) -> int:
    """Open the bundle once, rejecting a symlink at its final path component."""

    if getattr(os, "O_NOFOLLOW", 0) == 0:
        try:
            if stat.S_ISLNK(os.lstat(bundle).st_mode):
                raise _SnapshotError("symbolic link")
        except OSError as error:
            raise _SnapshotError("unreadable") from error
    try:
        descriptor = os.open(bundle, _open_flags(directory=True))
    except OSError as error:
        raise _SnapshotError("unreadable") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise _SnapshotError("not a directory")
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise _SnapshotError("unreadable") from error
    except _SnapshotError:
        os.close(descriptor)
        raise


def _read_regular_snapshot(
    root_fd: int, filename: str, maximum_bytes: int
) -> _Snapshot:
    """Read one flat G0 artifact relative to the opened bundle descriptor."""

    if getattr(os, "O_NOFOLLOW", 0) == 0:
        try:
            metadata = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise _SnapshotError("symbolic link")
        except OSError as error:
            raise _SnapshotError("unreadable") from error
    try:
        descriptor = os.open(filename, _open_flags(), dir_fd=root_fd)
    except OSError as error:
        raise _SnapshotError("unreadable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _SnapshotError("not a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise _SnapshotError("size limit")

        content = bytearray()
        while True:
            remaining = maximum_bytes + 1 - len(content)
            if remaining <= 0:
                raise _SnapshotError("size limit")
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > maximum_bytes:
            raise _SnapshotError("size limit")
        after = os.fstat(descriptor)
        if _snapshot_signature(before) != _snapshot_signature(after):
            raise _SnapshotError("changed while reading")
        return _Snapshot(bytes(content), _snapshot_signature(after))
    except OSError as error:
        raise _SnapshotError("unreadable") from error
    finally:
        os.close(descriptor)


def _inventory_names(root_fd: int) -> set[str] | None:
    """List at most the closed inventory plus one fd-relative entry.

    ``None`` means the directory exceeded the closed four-entry inventory.
    """

    try:
        names: set[str] = set()
        with os.scandir(root_fd) as entries:
            for entry in entries:
                if len(names) == G0_INVENTORY_ENTRY_CAP - 1:
                    return None
                names.add(entry.name)
        return names
    except OSError:
        raise _SnapshotError("unreadable") from None


def _snapshot_recheck_diagnostics(
    root_fd: int, snapshots: dict[str, _Snapshot]
) -> list[JsonObject]:
    """Reject a replacement after the snapshot's hash and parsing work."""

    for filename, snapshot in snapshots.items():
        try:
            current = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
        except OSError:
            return [_snapshot_diagnostic(filename, "changed after validation")]
        if _snapshot_signature(current) != snapshot.signature:
            return [_snapshot_diagnostic(filename, "changed after validation")]
    return []


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> JsonObject:
    from decodeforge.contracts import DuplicateKeyError

    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def _decode_json_snapshot(
    content: bytes, filename: str
) -> tuple[JsonObject | None, list[JsonObject]]:
    """Parse one bounded JSON object with duplicate-key rejection.

    Routing accepts ordinary UTF-8 foundation manifests.  G0's stricter ASCII
    and integer-only representation is checked later against the same bytes.
    """

    from decodeforge.contracts import DuplicateKeyError

    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except DuplicateKeyError:
        return None, [
            _diagnostic(
                "DFE-SCHEMA-008",
                "A JSON object contains a duplicate key.",
                {"path": [filename]},
                component="schema",
            )
        ]
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError):
        return None, [
            _diagnostic(
                "DFE-SCHEMA-001",
                "The JSON document could not be parsed.",
                {"path": [filename]},
                component="schema",
            )
        ]
    if not isinstance(value, dict):
        return None, [
            _diagnostic(
                "DFE-SCHEMA-001",
                "The JSON document could not be parsed.",
                {"path": [filename]},
                component="schema",
            )
        ]
    return value, []


def _canonical_json_snapshot(
    content: bytes, filename: str
) -> tuple[JsonObject | None, list[JsonObject]]:
    """Parse and canonicalize one JSON snapshot without rereading its path."""

    instance, diagnostics = _decode_json_snapshot(content, filename)
    if diagnostics or instance is None:
        return None, diagnostics
    try:
        expected = canonical_json_bytes(instance)
    except ValueError:
        return None, [
            _diagnostic(
                "DFE-BUNDLE-010",
                "A G0 JSON artifact is not canonical ASCII integer-only JSON.",
                {"reason": f"canonical-json:{filename}"},
            )
        ]
    if filename in {"fixture-manifest.json", "host.json"}:
        expected += b"\n"
    if content != expected:
        return None, [
            _diagnostic(
                "DFE-BUNDLE-010",
                "A G0 JSON artifact is not canonical ASCII integer-only JSON.",
                {"reason": f"canonical-json:{filename}"},
            )
        ]
    return instance, []


def _root_snapshot_diagnostic() -> JsonObject:
    return _diagnostic(
        "DFE-BUNDLE-005",
        "The G0 bundle root is not a real directory.",
        {"artifact": "."},
    )


def _inventory_diagnostics(root_fd: int) -> list[JsonObject]:
    try:
        names = _inventory_names(root_fd)
    except _SnapshotError:
        return [
            _diagnostic(
                "DFE-BUNDLE-005",
                "The G0 bundle inventory could not be listed.",
                {"artifact": "."},
            )
        ]
    if names is None:
        return [
            _diagnostic(
                "DFE-BUNDLE-002",
                "A G0 bundle exceeds its closed inventory entry cap.",
                {"artifact": "<inventory-entry-cap>"},
            )
        ]
    diagnostics: list[JsonObject] = []
    for filename in sorted(G0_ALLOWED_PATHS - names):
        diagnostics.append(
            _diagnostic(
                "DFE-BUNDLE-001",
                "A G0 bundle artifact is missing.",
                {"artifact": filename},
            )
        )
    for filename in sorted(names - G0_ALLOWED_PATHS):
        diagnostics.append(
            _diagnostic(
                "DFE-BUNDLE-002",
                "A G0 bundle contains an undeclared path.",
                {"artifact": filename},
            )
        )
    return diagnostics


def _snapshot_diagnostic(filename: str, reason: str) -> JsonObject:
    if reason == "size limit":
        return _diagnostic(
            "DFE-BUNDLE-010",
            "A G0 artifact exceeds its closed size limit.",
            {"reason": f"size-limit:{filename}"},
        )
    return _diagnostic(
        "DFE-BUNDLE-005",
        "A G0 artifact could not be read as one regular-file snapshot.",
        {"artifact": filename},
    )


def _record_diagnostics(manifest: JsonObject) -> list[JsonObject]:
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        return []
    records = [record for record in raw_records if isinstance(record, dict)]
    paths = [record.get("path") for record in records]
    string_paths = [path for path in paths if isinstance(path, str)]
    duplicates = sorted({path for path in string_paths if string_paths.count(path) > 1})
    if duplicates:
        return [
            _diagnostic(
                "DFE-BUNDLE-004",
                "An artifact path appears more than once.",
                {"artifact": path},
            )
            for path in duplicates
        ]
    if RUN_MANIFEST_NAME in string_paths:
        return [
            _diagnostic(
                "DFE-BUNDLE-002",
                "The root run manifest must not list itself as an artifact.",
                {"artifact": RUN_MANIFEST_NAME},
            )
        ]
    expected_pairs = list(G0_ARTIFACTS)
    actual_pairs = [(record.get("path"), record.get("role")) for record in records]
    if string_paths != sorted(string_paths) or actual_pairs != expected_pairs:
        return [
            _diagnostic(
                "DFE-BUNDLE-010",
                "G0 artifact records must be the exact sorted path-role inventory.",
                {"reason": "closed-path-role-records"},
            )
        ]
    return []


def _artifact_hash_diagnostics(
    manifest: JsonObject, snapshots: dict[str, _Snapshot]
) -> list[JsonObject]:
    records = {
        record["path"]: record
        for record in manifest["artifacts"]
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    diagnostics: list[JsonObject] = []
    for filename, _ in G0_ARTIFACTS:
        content = snapshots[filename].content
        record = records[filename]
        expected_size = record["bytes"]
        observed_size = len(content)
        if observed_size != expected_size:
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-006",
                    "An artifact byte length differs from the manifest.",
                    {
                        "artifact": filename,
                        "size": observed_size,
                        "expected_size": expected_size,
                    },
                )
            )
            continue
        observed_hash = hashlib.sha256(content).hexdigest()
        expected_hash = record["sha256"]
        if observed_hash != expected_hash:
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-007",
                    "An artifact digest differs from the manifest.",
                    {
                        "artifact": filename,
                        "sha256": observed_hash,
                        "expected_sha256": expected_hash,
                    },
                )
            )
    return diagnostics


def verify_g0_bundle(bundle: Path) -> list[JsonObject] | None:
    """Verify portable G0 snapshots, returning ``None`` for foundation bundles.

    One directory descriptor anchors every G0 read and inventory operation.
    This function has no subprocess calls and never evaluates a recorded argv.
    Only a safely parsed manifest with a non-G0 milestone falls through to the
    pre-existing foundation verifier.
    """

    try:
        root_fd = _open_bundle_root(bundle)
    except _SnapshotError:
        return [_root_snapshot_diagnostic()]
    try:
        try:
            manifest_snapshot = _read_regular_snapshot(
                root_fd, RUN_MANIFEST_NAME, G0_FILE_LIMITS[RUN_MANIFEST_NAME]
            )
        except _SnapshotError as error:
            return [_snapshot_diagnostic(RUN_MANIFEST_NAME, error.reason)]
        manifest_bytes = manifest_snapshot.content
        manifest, parse_diagnostics = _decode_json_snapshot(
            manifest_bytes, RUN_MANIFEST_NAME
        )
        if parse_diagnostics or manifest is None:
            return parse_diagnostics
        if manifest.get("milestone") != "g0":
            return None

        from decodeforge.contracts import validate_data

        schema_diagnostics = validate_data(
            manifest, "run-manifest", document_name=RUN_MANIFEST_NAME
        )
        if schema_diagnostics:
            return schema_diagnostics
        try:
            canonical_manifest = canonical_json_bytes(manifest)
            expected_bundle_id = g0_bundle_id(manifest)
        except ValueError:
            return [
                _diagnostic(
                    "DFE-SCHEMA-006",
                    "The run manifest must use ASCII strings and integer-only numbers.",
                    {"path": [RUN_MANIFEST_NAME]},
                    component="schema",
                )
            ]
        if manifest_bytes != canonical_manifest:
            return [
                _diagnostic(
                    "DFE-BUNDLE-010",
                    "The G0 run manifest is not canonical ASCII JSON.",
                    {"reason": "canonical-run-manifest"},
                )
            ]
        if manifest.get("bundle_id") != expected_bundle_id:
            return [
                _diagnostic(
                    "DFE-BUNDLE-008",
                    "The run manifest bundle identifier does not match its canonical "
                    "preimage.",
                    {"bundle": RUN_MANIFEST_NAME},
                )
            ]
        diagnostics = _inventory_diagnostics(root_fd)
        if diagnostics:
            return diagnostics
        diagnostics = _record_diagnostics(manifest)
        if diagnostics:
            return diagnostics

        snapshots: dict[str, _Snapshot] = {RUN_MANIFEST_NAME: manifest_snapshot}
        for filename, _ in G0_ARTIFACTS:
            try:
                snapshots[filename] = _read_regular_snapshot(
                    root_fd, filename, G0_FILE_LIMITS[filename]
                )
            except _SnapshotError as error:
                return [_snapshot_diagnostic(filename, error.reason)]
        diagnostics = _inventory_diagnostics(root_fd)
        if diagnostics:
            return diagnostics
        diagnostics = _artifact_hash_diagnostics(manifest, snapshots)
        if diagnostics:
            return diagnostics
        for filename, schema_name in (
            ("fixture-manifest.json", "fixture-manifest"),
            ("host.json", "host-manifest"),
        ):
            instance, diagnostics = _canonical_json_snapshot(
                snapshots[filename].content, filename
            )
            if diagnostics or instance is None:
                return diagnostics
            schema_diagnostics = validate_data(
                instance, schema_name, document_name=filename
            )
            if schema_diagnostics:
                return schema_diagnostics
        diagnostics = _snapshot_recheck_diagnostics(root_fd, snapshots)
        if diagnostics:
            return diagnostics
        diagnostics = _inventory_diagnostics(root_fd)
        if diagnostics:
            return diagnostics
        return []
    finally:
        os.close(root_fd)


def _assert_ascii_integer_json(value: Any) -> None:
    """Iteratively reject non-canonical JSON values without unbounded recursion."""

    pending: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    while pending:
        current, depth = pending.pop()
        if current is None or type(current) is bool or type(current) is int:
            continue
        if isinstance(current, str):
            if not current.isascii():
                raise ValueError("non-ASCII string")
            continue
        if not isinstance(current, list | dict):
            raise ValueError(
                "JSON values must use strings, booleans, null, or integers"
            )
        if depth >= _CANONICAL_JSON_MAX_DEPTH:
            raise ValueError("JSON nesting exceeds the canonical depth limit")
        container_id = id(current)
        if container_id in seen_containers:
            raise ValueError("JSON containers must not be cyclic or reused")
        seen_containers.add(container_id)
        if isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
            continue
        for key, item in current.items():
            if not isinstance(key, str) or not key.isascii():
                raise ValueError("non-ASCII object key")
            pending.append((item, depth + 1))


def canonical_json_bytes(value: JsonObject) -> bytes:
    """Render sorted, compact ASCII JSON after enforcing integer-only numbers."""

    _assert_ascii_integer_json(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("JSON value cannot be canonicalized") from error


def canonical_run_manifest_bytes(manifest: JsonObject) -> bytes:
    """Render the unsigned G0 bundle-ID preimage with ``bundle_id`` omitted."""

    unsigned = {key: value for key, value in manifest.items() if key != "bundle_id"}
    return canonical_json_bytes(unsigned)


def g0_bundle_id(manifest: JsonObject) -> str:
    """Return the versioned SHA-256 identifier for an unsigned run manifest."""

    digest = hashlib.sha256(BUNDLE_ID_PREFIX + canonical_run_manifest_bytes(manifest))
    return f"sha256:{digest.hexdigest()}"


def g0_reproduction(revision: str) -> JsonObject:
    """Return the closed ordered G0 reproduction metadata record."""

    return {
        "cwd": ".",
        "policy": "g0-correctness-v1",
        "source_revision": revision,
        "environment": {
            "CARGO_NET_OFFLINE": "true",
            "DECODEFORGE_SOURCE_REVISION": revision,
            "UV_OFFLINE": "true",
        },
        "commands": [
            {"id": check_id, "argv": list(argv), "expected_exit_code": 0}
            for check_id, argv in G0_REPRODUCTION_COMMANDS
        ],
    }

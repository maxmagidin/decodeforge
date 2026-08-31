"""Canonical data contracts shared by G0 evidence producers and verifiers."""

from __future__ import annotations

import hashlib
import json
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


def _assert_ascii_integer_json(value: Any) -> None:
    """Reject non-ASCII strings and every non-integer JSON number."""

    if value is None or type(value) is bool or type(value) is int:
        return
    if isinstance(value, str):
        if not value.isascii():
            raise ValueError("non-ASCII string")
        return
    if isinstance(value, list):
        for item in value:
            _assert_ascii_integer_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise ValueError("non-ASCII object key")
            _assert_ascii_integer_json(item)
        return
    raise ValueError("JSON values must use strings, booleans, null, or integers")


def canonical_json_bytes(value: JsonObject) -> bytes:
    """Render sorted, compact ASCII JSON after enforcing integer-only numbers."""

    _assert_ascii_integer_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


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

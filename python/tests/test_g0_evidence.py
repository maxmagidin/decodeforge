"""Canonical G0 evidence-contract tests."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from decodeforge.contracts import ROOT, load_json, validate_data
from decodeforge.g0_evidence import (
    BUNDLE_ID_PREFIX,
    G0_ARTIFACTS,
    G0_REPRODUCTION_ENV_KEYS,
    canonical_json_bytes,
    canonical_run_manifest_bytes,
    g0_bundle_id,
    g0_reproduction,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _vector_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "milestone": "g0",
        "bundle_class": "correctness",
        "created_utc": "2026-08-30T00:00:00Z",
        "project": {
            "revision": REVISION,
            "dirty": False,
            "format": "DFQ8_B32_V1",
            "numeric_mode": "strict_f32_v1",
            "compiler_version": "0.1.0",
            "generated_abi": 1,
            "runtime_abi": 1,
        },
        "target": {"triple": "aarch64-apple-darwin", "features": ["neon"]},
        "reproduction": g0_reproduction(REVISION),
        "artifacts": [
            {
                "path": path,
                "role": role,
                "bytes": 0,
                "sha256": "0" * 64,
            }
            for path, role in G0_ARTIFACTS
        ],
        "checks": {
            "schema": "pass",
            "correctness": "pass",
            "assembly": "not-applicable",
            "certified_performance": "not-applicable",
        },
        "not_applicable": ["assembly", "certified_performance"],
    }


def test_g0_bundle_id_has_a_hardcoded_canonical_vector() -> None:
    manifest = _vector_manifest()
    manifest["bundle_id"] = (
        "sha256:7e03a8172fc90e28a4b97c25e4ee7b4bb357ad3b0170dfe31f0186e41b15dd8e"
    )
    expected_preimage = canonical_run_manifest_bytes(manifest)
    unsigned = dict(manifest)
    del unsigned["bundle_id"]
    assert expected_preimage == canonical_json_bytes(unsigned)
    assert BUNDLE_ID_PREFIX + expected_preimage == (
        b"DecodeForge/run-bundle/v1\0" + expected_preimage
    )
    assert g0_bundle_id(manifest) == manifest["bundle_id"]

    changed_identifier = deepcopy(manifest)
    changed_identifier["bundle_id"] = "sha256:" + "f" * 64
    assert canonical_run_manifest_bytes(changed_identifier) == expected_preimage
    assert g0_bundle_id(changed_identifier) == manifest["bundle_id"]

    reordered = json.loads(json.dumps(manifest, indent=2, sort_keys=True))
    assert g0_bundle_id(reordered) == manifest["bundle_id"]

    changed_artifact = deepcopy(manifest)
    artifacts = changed_artifact["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[0]["sha256"] = "1" * 64
    assert g0_bundle_id(changed_artifact) != manifest["bundle_id"]


def test_canonical_g0_json_rejects_floats_and_non_ascii() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"integer": 1.0})
    with pytest.raises(ValueError):
        canonical_json_bytes({"text": "caf\u00e9"})
    assert canonical_json_bytes({"flag": False, "integer": 1}) == (
        b'{"flag":false,"integer":1}'
    )


def test_g0_reproduction_is_closed_and_ordered() -> None:
    reproduction = g0_reproduction(REVISION)
    assert reproduction["cwd"] == "."
    assert reproduction["policy"] == "g0-correctness-v1"
    assert reproduction["source_revision"] == REVISION
    environment = reproduction["environment"]
    assert isinstance(environment, dict)
    assert tuple(environment) == G0_REPRODUCTION_ENV_KEYS
    assert environment["DECODEFORGE_SOURCE_REVISION"] == REVISION
    assert reproduction["commands"] == [
        {
            "id": "schema-contracts-v1",
            "argv": [
                "uv",
                "run",
                "--frozen",
                "python",
                "scripts/validate_schemas.py",
                "--all",
            ],
            "expected_exit_code": 0,
        },
        {
            "id": "q8-python-fixtures-v1",
            "argv": [
                "uv",
                "run",
                "--frozen",
                "python",
                "scripts/generate_q8_fixtures.py",
                "--check",
            ],
            "expected_exit_code": 0,
        },
        {
            "id": "q8-rust-fixtures-v1",
            "argv": ["make", "rust-fixture-check"],
            "expected_exit_code": 0,
        },
    ]


def test_correctness_schema_requires_the_g0_reproduction_fields() -> None:
    example = load_json(
        ROOT / "schemas" / "examples" / "run-manifest" / "valid-g0-correctness.json"
    )
    assert validate_data(example, "run-manifest") == []
    assert g0_bundle_id(example) == example["bundle_id"]
    assert example["reproduction"] == g0_reproduction(REVISION)

    missing_created = deepcopy(example)
    del missing_created["created_utc"]
    diagnostics = validate_data(missing_created, "run-manifest")
    assert "DFE-SCHEMA-004" in {item["code"] for item in diagnostics}

    missing_runtime_abi = deepcopy(example)
    project = missing_runtime_abi["project"]
    assert isinstance(project, dict)
    del project["runtime_abi"]
    diagnostics = validate_data(missing_runtime_abi, "run-manifest")
    assert "DFE-SCHEMA-004" in {item["code"] for item in diagnostics}

    wrong_class = deepcopy(example)
    wrong_class["bundle_class"] = "fixture"
    diagnostics = validate_data(wrong_class, "run-manifest")
    assert "DFE-SCHEMA-007" in {item["code"] for item in diagnostics}

    wrong_g0_policy = deepcopy(example)
    reproduction = wrong_g0_policy["reproduction"]
    assert isinstance(reproduction, dict)
    reproduction["policy"] = "g1-provenance-v1"
    diagnostics = validate_data(wrong_g0_policy, "run-manifest")
    assert "DFE-SCHEMA-007" in {item["code"] for item in diagnostics}

    future_correctness = deepcopy(example)
    future_correctness["milestone"] = "g1"
    del future_correctness["created_utc"]
    del future_correctness["reproduction"]
    assert validate_data(future_correctness, "run-manifest") == []

    future_correctness["reproduction"] = {
        "cwd": "capture",
        "policy": "g1-provenance-v1",
        "source_revision": "unknown",
        "environment": {"G1_CAPTURE": "enabled"},
        "commands": [
            {
                "id": "g1-capture-v1",
                "argv": ["python", "scripts/capture.py"],
                "expected_exit_code": 17,
            }
        ],
    }
    assert validate_data(future_correctness, "run-manifest") == []

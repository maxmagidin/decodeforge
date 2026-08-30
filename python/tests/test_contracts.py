"""Schema, diagnostic, and non-executing bundle validation tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from decodeforge.contracts import (
    ROOT,
    check_all,
    load_json,
    validate_data,
    validate_path,
    verify_bundle,
)

EXAMPLES = ROOT / "schemas" / "examples"
BUNDLES = ROOT / "tests" / "fixtures" / "bundles"


def test_catalog_and_directed_examples_are_consistent() -> None:
    assert check_all() == []


def test_schema_error_codes_are_stable() -> None:
    request = load_json(EXAMPLES / "compiler-request" / "valid-minimal.json")

    wrong_version = dict(request, schema_version=2)
    assert [
        item["code"] for item in validate_data(wrong_version, "compiler-request")
    ] == ["DFE-SCHEMA-002"]

    unknown_field = dict(request, weights_path="forbidden")
    assert [
        item["code"] for item in validate_data(unknown_field, "compiler-request")
    ] == ["DFE-SCHEMA-005"]

    wrong_type = dict(request, n="four")
    assert [item["code"] for item in validate_data(wrong_type, "compiler-request")] == [
        "DFE-SCHEMA-006"
    ]


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    diagnostics = validate_path(duplicate, "diagnostic")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-008"]


def test_empty_foundation_bundle_has_exact_missing_artifacts() -> None:
    diagnostics = verify_bundle(BUNDLES / "foundation-empty")
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-001"] * 3
    assert [item["context"]["artifact"] for item in diagnostics] == [
        "host.json",
        "report.md",
        "request.json",
    ]
    for item in diagnostics:
        assert validate_data(item, "diagnostic") == []


def test_minimal_foundation_bundle_is_accepted() -> None:
    assert verify_bundle(BUNDLES / "foundation-valid") == []


def test_bundle_hash_mutation_is_rejected(tmp_path: Path) -> None:
    mutated = tmp_path / "bundle"
    shutil.copytree(BUNDLES / "foundation-valid", mutated)
    report = mutated / "report.md"
    report.write_text(
        report.read_text(encoding="utf-8") + "mutated\n", encoding="utf-8"
    )
    diagnostics = verify_bundle(mutated)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-006"]
    assert diagnostics[0]["context"]["artifact"] == "report.md"


def test_bundle_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    mutated = tmp_path / "bundle"
    shutil.copytree(BUNDLES / "foundation-valid", mutated)
    report = mutated / "report.md"
    report.unlink()
    report.symlink_to("request.json")
    diagnostics = verify_bundle(mutated)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-005"]
    assert diagnostics[0]["context"]["artifact"] == "report.md"


def test_bundle_symlink_manifest_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run-manifest.json").symlink_to(
        BUNDLES / "foundation-valid" / "run-manifest.json"
    )
    diagnostics = verify_bundle(bundle)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-005"]
    assert diagnostics[0]["context"]["artifact"] == "run-manifest.json"


def test_diagnostic_registry_codes_are_unique() -> None:
    registry = load_json(ROOT / "schemas" / "diagnostic-codes.json")
    raw_codes = registry["codes"]
    assert isinstance(raw_codes, list)
    codes = [entry["code"] for entry in raw_codes]
    assert len(codes) == len(set(codes))
    json.dumps(registry, allow_nan=False)

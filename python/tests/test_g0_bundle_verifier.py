"""Descriptor-backed portable G0 bundle verification tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, NoReturn

import pytest
from decodeforge.contracts import ROOT, validate_data, verify_bundle
from decodeforge.g0_evidence import (
    G0_ARTIFACTS,
    G0_FILE_LIMITS,
    G0_INVENTORY_ENTRY_CAP,
    G0_NOT_APPLICABLE,
    G0_REQUIRED_CHECKS,
    _open_bundle_root,
    _read_regular_snapshot,
    _Snapshot,
    _SnapshotError,
    canonical_json_bytes,
    g0_bundle_id,
    g0_reproduction,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _host_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "host_id": "g0-fixture-apple",
        "role": "mac-primary",
        "architecture": "aarch64",
        "cpu": {
            "model": "fixture Apple CPU",
            "physical_cores": 1,
            "logical_cores": 1,
            "features": ["neon"],
        },
        "os": {"name": "Darwin", "version": "fixture", "kernel": "fixture"},
        "toolchains": {
            "clang": "fixture",
            "python": "3.12.14",
            "rust": "1.98.0",
            "uv": "0.12.5",
        },
        "source": {"revision": REVISION, "dirty": False},
    }


def _run_manifest(contents: dict[str, bytes]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
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
                "bytes": len(contents[path]),
                "sha256": hashlib.sha256(contents[path]).hexdigest(),
            }
            for path, role in G0_ARTIFACTS
        ],
        "checks": dict(G0_REQUIRED_CHECKS),
        "not_applicable": list(G0_NOT_APPLICABLE),
    }
    manifest["bundle_id"] = g0_bundle_id(manifest)
    return manifest


def _write_run(bundle: Path, manifest: dict[str, Any]) -> None:
    manifest["bundle_id"] = g0_bundle_id(manifest)
    (bundle / "run-manifest.json").write_bytes(canonical_json_bytes(manifest))


def _make_bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    bundle = tmp_path / f"g0-{len(list(tmp_path.iterdir()))}"
    bundle.mkdir()
    contents = {
        "fixture-manifest.json": (
            ROOT / "tests" / "fixtures" / "v1" / "manifest.json"
        ).read_bytes(),
        "host.json": canonical_json_bytes(_host_manifest()) + b"\n",
        "report.md": b"# G0 verifier fixture\n",
    }
    for path, content in contents.items():
        (bundle / path).write_bytes(content)
    manifest = _run_manifest(contents)
    _write_run(bundle, manifest)
    return bundle, manifest


def _record(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    records = manifest["artifacts"]
    assert isinstance(records, list)
    return next(record for record in records if record["path"] == path)


def _refresh_record(bundle: Path, manifest: dict[str, Any], path: str) -> None:
    content = (bundle / path).read_bytes()
    record = _record(manifest, path)
    record["bytes"] = len(content)
    record["sha256"] = hashlib.sha256(content).hexdigest()
    _write_run(bundle, manifest)


def _codes(bundle: Path) -> list[str]:
    diagnostics = verify_bundle(bundle)
    assert all(validate_data(item, "diagnostic") == [] for item in diagnostics)
    return [item["code"] for item in diagnostics]


def test_portable_g0_bundle_and_foundation_compatibility(tmp_path: Path) -> None:
    bundle, _ = _make_bundle(tmp_path)
    assert verify_bundle(bundle) == []
    assert (
        verify_bundle(ROOT / "tests" / "fixtures" / "bundles" / "foundation-valid")
        == []
    )
    empty = verify_bundle(ROOT / "tests" / "fixtures" / "bundles" / "foundation-empty")
    assert [item["code"] for item in empty] == ["DFE-BUNDLE-001"] * 3


def test_g0_allows_a_symlinked_ancestor_but_not_a_symlinked_root(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "ancestor-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    bundle, _ = _make_bundle(real_parent)

    assert verify_bundle(alias / bundle.name) == []

    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(bundle, target_is_directory=True)
    assert _codes(root_alias) == ["DFE-BUNDLE-005"]


def test_open_bundle_descriptor_survives_a_root_namespace_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decodeforge.g0_evidence as evidence

    bundle, _ = _make_bundle(tmp_path)
    moved = tmp_path / "moved-root"
    original = evidence._read_regular_snapshot
    swapped = False

    def _swap_root(root_fd: int, filename: str, maximum_bytes: int) -> _Snapshot:
        nonlocal swapped
        if not swapped:
            swapped = True
            bundle.rename(moved)
            bundle.mkdir()
        return original(root_fd, filename, maximum_bytes)

    monkeypatch.setattr(evidence, "_read_regular_snapshot", _swap_root)
    assert verify_bundle(bundle) == []
    assert swapped


def test_verifier_never_executes_recorded_command_argv(tmp_path: Path) -> None:
    bundle, manifest = _make_bundle(tmp_path)
    marker = tmp_path / "argv-was-executed"
    reproduction = manifest["reproduction"]
    assert isinstance(reproduction, dict)
    commands = reproduction["commands"]
    assert isinstance(commands, list)
    commands[0]["argv"] = [
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'w').write('unexpected')",
    ]
    _write_run(bundle, manifest)

    assert verify_bundle(bundle) == []
    assert not marker.exists()


def test_closed_inventory_and_path_role_records_are_enforced(tmp_path: Path) -> None:
    bundle, _ = _make_bundle(tmp_path)
    (bundle / "extra.txt").write_text("extra", encoding="ascii")
    assert _codes(bundle) == ["DFE-BUNDLE-002"]

    bundle, _ = _make_bundle(tmp_path)
    (bundle / "report.md").unlink()
    assert _codes(bundle) == ["DFE-BUNDLE-001"]

    bundle, manifest = _make_bundle(tmp_path)
    records = manifest["artifacts"]
    assert isinstance(records, list)
    records.append(deepcopy(records[0]))
    _write_run(bundle, manifest)
    assert _codes(bundle) == ["DFE-BUNDLE-004"]

    bundle, manifest = _make_bundle(tmp_path)
    _record(manifest, "report.md")["role"] = "wrong"
    _write_run(bundle, manifest)
    assert _codes(bundle) == ["DFE-BUNDLE-010"]

    bundle, manifest = _make_bundle(tmp_path)
    records = manifest["artifacts"]
    assert isinstance(records, list)
    records.reverse()
    _write_run(bundle, manifest)
    assert _codes(bundle) == ["DFE-BUNDLE-010"]

    bundle, manifest = _make_bundle(tmp_path)
    records = manifest["artifacts"]
    assert isinstance(records, list)
    records.append(
        {
            "path": "run-manifest.json",
            "role": "forbidden",
            "bytes": 0,
            "sha256": "0" * 64,
        }
    )
    _write_run(bundle, manifest)
    assert _codes(bundle) == ["DFE-BUNDLE-002"]


def test_many_extra_names_hit_the_bounded_inventory_entry_cap(tmp_path: Path) -> None:
    bundle, _ = _make_bundle(tmp_path)
    for index in range(G0_INVENTORY_ENTRY_CAP * 32):
        (bundle / f"extra-{index:03d}").write_text("extra", encoding="ascii")

    diagnostics = verify_bundle(bundle)
    assert all(validate_data(item, "diagnostic") == [] for item in diagnostics)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-002"]
    assert diagnostics[0]["context"] == {"artifact": "<inventory-entry-cap>"}


def test_special_artifacts_and_oversized_g0_roots_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _make_bundle(tmp_path)
    report = bundle / "report.md"
    report.unlink()
    report.mkdir()
    assert _codes(bundle) == ["DFE-BUNDLE-005"]

    bundle, _ = _make_bundle(tmp_path)
    report = bundle / "report.md"
    report.unlink()
    report.symlink_to("host.json")
    assert _codes(bundle) == ["DFE-BUNDLE-005"]

    if hasattr(os, "mkfifo"):
        bundle, _ = _make_bundle(tmp_path)
        report = bundle / "report.md"
        report.unlink()
        os.mkfifo(report)
        assert _codes(bundle) == ["DFE-BUNDLE-005"]

    if os.path.exists(os.devnull):
        dev_root = _open_bundle_root(Path(os.devnull).parent)
        try:
            with pytest.raises(_SnapshotError):
                _read_regular_snapshot(dev_root, Path(os.devnull).name, 1024)
        finally:
            os.close(dev_root)

    if hasattr(socket, "AF_UNIX"):
        bundle, _ = _make_bundle(tmp_path)
        report = bundle / "report.md"
        report.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            monkeypatch.chdir(bundle)
            listener.bind(report.name)
            assert _codes(bundle) == ["DFE-BUNDLE-005"]
        finally:
            listener.close()

    bundle, _ = _make_bundle(tmp_path)
    run_manifest = bundle / "run-manifest.json"
    run_manifest.write_bytes(
        run_manifest.read_bytes() + b" " * (G0_FILE_LIMITS["run-manifest.json"] + 1)
    )
    assert _codes(bundle) == ["DFE-BUNDLE-010"]

    bundle, _ = _make_bundle(tmp_path)
    (bundle / "report.md").write_bytes(b"x" * (G0_FILE_LIMITS["report.md"] + 1))
    assert _codes(bundle) == ["DFE-BUNDLE-010"]


def test_root_failures_never_fall_back_to_the_foundation_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decodeforge.contracts as contracts

    def _unexpected_foundation(_: Path) -> list[dict[str, Any]]:
        raise AssertionError("the bounded root failure reached the foundation reader")

    monkeypatch.setattr(contracts, "_verify_foundation_bundle", _unexpected_foundation)

    missing = tmp_path / "missing-root"
    missing.mkdir()
    assert _codes(missing) == ["DFE-BUNDLE-005"]

    malformed = tmp_path / "malformed-root"
    malformed.mkdir()
    (malformed / "run-manifest.json").write_bytes(b'{"milestone":')
    assert _codes(malformed) == ["DFE-SCHEMA-001"]

    oversized = tmp_path / "oversized-root"
    oversized.mkdir()
    (oversized / "run-manifest.json").write_bytes(
        b"{" + b" " * G0_FILE_LIMITS["run-manifest.json"]
    )
    assert _codes(oversized) == ["DFE-BUNDLE-010"]


def test_deep_root_and_artifact_json_are_rejected_without_recursion(
    tmp_path: Path,
) -> None:
    deep_root = tmp_path / "deep-root"
    deep_root.mkdir()
    (deep_root / "run-manifest.json").write_bytes(b"[" * 20_000)
    assert _codes(deep_root) == ["DFE-SCHEMA-001"]

    bundle, manifest = _make_bundle(tmp_path)
    host = bundle / "host.json"
    host.write_bytes(b'{"a":' * 256 + b"0" + b"}" * 256 + b"\n")
    _refresh_record(bundle, manifest, "host.json")
    assert _codes(bundle) == ["DFE-BUNDLE-010"]


def test_deep_g0_manifest_is_a_stable_parse_diagnostic(tmp_path: Path) -> None:
    bundle = tmp_path / "g0-deep"
    bundle.mkdir()
    nested_values: list[object] = []
    for _ in range(2):
        value: object = "assembly"
        for _ in range(900):
            value = [value]
        nested_values.append(value)
    payload = json.dumps(
        {"milestone": "g0", "not_applicable": nested_values},
        separators=(",", ":"),
    )
    assert len(payload.encode("ascii")) == 3_659
    (bundle / "run-manifest.json").write_text(payload, encoding="ascii")

    diagnostics = verify_bundle(bundle)
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-001"]
    assert diagnostics[0]["context"] == {"path": ["run-manifest.json"]}


def test_inventory_recursion_error_is_not_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decodeforge.g0_evidence as evidence

    bundle, _ = _make_bundle(tmp_path)

    def _trusted_inventory_failure(_: int) -> NoReturn:
        raise RecursionError("trusted inventory")

    monkeypatch.setattr(evidence, "_inventory_names", _trusted_inventory_failure)
    with pytest.raises(RecursionError, match="trusted inventory"):
        verify_bundle(bundle)


def test_hashes_and_canonical_json_use_the_same_snapshot(tmp_path: Path) -> None:
    bundle, _ = _make_bundle(tmp_path)
    report = bundle / "report.md"
    original = report.read_bytes()
    report.write_bytes(b"!" + original[1:])
    assert _codes(bundle) == ["DFE-BUNDLE-007"]

    bundle, manifest = _make_bundle(tmp_path)
    host = bundle / "host.json"
    host.write_bytes(host.read_bytes() + b" ")
    _refresh_record(bundle, manifest, "host.json")
    assert _codes(bundle) == ["DFE-BUNDLE-010"]

    bundle, manifest = _make_bundle(tmp_path)
    host = bundle / "host.json"
    document = json.loads(host.read_text(encoding="ascii"))
    host.write_text(
        json.dumps(document, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )
    _refresh_record(bundle, manifest, "host.json")
    assert _codes(bundle) == []

    bundle, manifest = _make_bundle(tmp_path)
    host = bundle / "host.json"
    host.write_bytes(b'{"number":1.0}\n')
    _refresh_record(bundle, manifest, "host.json")
    assert _codes(bundle) == ["DFE-BUNDLE-010"]

    bundle, manifest = _make_bundle(tmp_path)
    host = bundle / "host.json"
    host.write_bytes(b'{"text":"caf\xc3\xa9"}\n')
    _refresh_record(bundle, manifest, "host.json")
    assert _codes(bundle) == ["DFE-BUNDLE-010"]

    bundle, manifest = _make_bundle(tmp_path)
    host = bundle / "host.json"
    host.write_bytes(b'{"duplicate":1,"duplicate":2}\n')
    _refresh_record(bundle, manifest, "host.json")
    assert _codes(bundle) == ["DFE-SCHEMA-008"]


@pytest.mark.parametrize(
    ("filename", "field", "value", "expected"),
    [
        ("host.json", "unexpected", "value", "DFE-SCHEMA-005"),
        ("host.json", "cpu", 7, "DFE-SCHEMA-006"),
        ("fixture-manifest.json", "unexpected", "value", "DFE-SCHEMA-005"),
        ("fixture-manifest.json", "artifacts", 7, "DFE-SCHEMA-006"),
    ],
)
def test_copied_json_artifacts_validate_their_single_file_schemas(
    tmp_path: Path, filename: str, field: str, value: Any, expected: str
) -> None:
    bundle, manifest = _make_bundle(tmp_path)
    artifact = bundle / filename
    document = json.loads(artifact.read_text(encoding="ascii"))
    assert isinstance(document, dict)
    document[field] = value
    artifact.write_bytes(canonical_json_bytes(document) + b"\n")
    _refresh_record(bundle, manifest, filename)
    assert _codes(bundle) == [expected]


@pytest.mark.parametrize("filename", ["host.json", "run-manifest.json"])
def test_snapshot_names_are_rechecked_after_hash_and_canonical_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    import decodeforge.g0_evidence as evidence

    bundle, _ = _make_bundle(tmp_path)
    original = evidence._artifact_hash_diagnostics

    def _replace_after_hash(
        manifest: dict[str, Any], snapshots: dict[str, _Snapshot]
    ) -> list[dict[str, Any]]:
        diagnostics = original(manifest, snapshots)
        target = bundle / filename
        replacement = bundle / f"replacement-{filename}"
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
        return diagnostics

    monkeypatch.setattr(evidence, "_artifact_hash_diagnostics", _replace_after_hash)
    assert _codes(bundle) == ["DFE-BUNDLE-005"]


def test_inventory_is_checked_again_after_artifact_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decodeforge.g0_evidence as evidence

    bundle, _ = _make_bundle(tmp_path)
    original = evidence._read_regular_snapshot

    def _late_extra(root_fd: int, filename: str, maximum_bytes: int) -> _Snapshot:
        content = original(root_fd, filename, maximum_bytes)
        if filename == "report.md":
            (bundle / "late-extra").write_text("late", encoding="ascii")
        return content

    monkeypatch.setattr(evidence, "_read_regular_snapshot", _late_extra)
    assert _codes(bundle) == ["DFE-BUNDLE-002"]


def test_inventory_is_the_final_g0_validation_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import decodeforge.g0_evidence as evidence

    bundle, _ = _make_bundle(tmp_path)
    original = evidence._snapshot_recheck_diagnostics

    def _late_extra(
        root_fd: int, snapshots: dict[str, _Snapshot]
    ) -> list[dict[str, Any]]:
        diagnostics = original(root_fd, snapshots)
        (bundle / "late-extra").write_text("late", encoding="ascii")
        return diagnostics

    monkeypatch.setattr(evidence, "_snapshot_recheck_diagnostics", _late_extra)
    assert _codes(bundle) == ["DFE-BUNDLE-002"]


def test_wheel_carries_schema_corpus_and_verifies_outside_checkout(
    tmp_path: Path,
) -> None:
    """An extracted offline wheel must not consult this source checkout."""

    distribution = tmp_path / "distribution"
    build_environment = os.environ.copy()
    build_environment["UV_OFFLINE"] = "true"
    build = subprocess.run(
        [
            "uv",
            "build",
            "--sdist",
            "--wheel",
            "--out-dir",
            str(distribution),
            ".",
        ],
        cwd=ROOT,
        env=build_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    wheel = next(distribution.glob("*.whl"))
    sdist = next(distribution.glob("*.tar.gz"))
    expected_wheel_members = {
        "decodeforge/_schemas/common.schema.json",
        "decodeforge/_schemas/diagnostic-codes.json",
        "decodeforge/_schemas/examples/run-manifest/valid-g0-correctness.json",
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = set(archive.namelist())
    assert expected_wheel_members <= wheel_members
    assert not any(member.startswith("schemas/") for member in wheel_members)

    with tarfile.open(sdist) as archive:
        sdist_members = archive.getnames()
    for expected in (
        "schemas/common.schema.json",
        "schemas/diagnostic-codes.json",
        "schemas/examples/run-manifest/valid-g0-correctness.json",
    ):
        assert any(member.endswith(expected) for member in sdist_members)

    wheel_root = tmp_path / "extracted-wheel"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(wheel_root)
    source_bundle, _ = _make_bundle(tmp_path)
    outside_checkout = tmp_path / "outside-checkout"
    outside_checkout.mkdir()
    copied_bundle = outside_checkout / "bundle"
    shutil.copytree(source_bundle, copied_bundle)

    smoke = """
from pathlib import Path
import sys

import decodeforge.contracts as contracts

wheel_root = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
bundle = Path(sys.argv[3])
assert Path(contracts.__file__).resolve().is_relative_to(wheel_root)
assert contracts.SCHEMA_DIR.resolve().is_relative_to(wheel_root)
assert not contracts.SCHEMA_DIR.resolve().is_relative_to(source_root)
assert contracts.check_all() == []
assert contracts.verify_bundle(bundle) == []
"""
    smoke_environment = os.environ.copy()
    smoke_environment["PYTHONPATH"] = str(wheel_root)
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            smoke,
            str(wheel_root),
            str(ROOT),
            str(copied_bundle),
        ],
        cwd=outside_checkout,
        env=smoke_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

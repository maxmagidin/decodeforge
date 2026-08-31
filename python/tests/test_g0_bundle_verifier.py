"""Descriptor-backed portable G0 bundle verification tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import decodeforge.g0_evidence as evidence
import decodeforge.g0_repository as repository
import pytest
from decodeforge.contracts import ROOT, validate_data, verify_bundle
from decodeforge.g0_evidence import (
    G0_ARTIFACTS,
    G0_COMPILER_VERSION,
    G0_FILE_LIMITS,
    G0_HOST_CPU_MODEL,
    G0_HOST_ID,
    G0_HOST_LOGICAL_CORES,
    G0_HOST_PHYSICAL_CORES,
    G0_INVENTORY_ENTRY_CAP,
    G0_NOT_APPLICABLE,
    G0_PROVENANCE_BASELINE,
    G0_REQUIRED_CHECKS,
    G0_TOOLCHAINS,
    _open_bundle_root,
    _read_regular_snapshot,
    _Snapshot,
    _SnapshotError,
    canonical_json_bytes,
    g0_bundle_id,
    g0_reproduction,
)
from decodeforge.g0_repository import verify_g0_repository_bundle

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _host_manifest(revision: str = REVISION) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "host_id": G0_HOST_ID,
        "role": "mac-primary",
        "architecture": "aarch64",
        "cpu": {
            "model": G0_HOST_CPU_MODEL,
            "physical_cores": G0_HOST_PHYSICAL_CORES,
            "logical_cores": G0_HOST_LOGICAL_CORES,
            "features": ["neon"],
        },
        "os": {"name": "Darwin", "version": "fixture", "kernel": "fixture"},
        "toolchains": dict(G0_TOOLCHAINS),
        "source": {"revision": revision, "dirty": False},
    }


def _run_manifest(
    contents: dict[str, bytes], revision: str = REVISION
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "g0",
        "bundle_class": "correctness",
        "created_utc": "2026-08-30T00:00:00Z",
        "project": {
            "revision": revision,
            "dirty": False,
            "format": "DFQ8_B32_V1",
            "numeric_mode": "strict_f32_v1",
            "compiler_version": G0_COMPILER_VERSION,
            "generated_abi": 1,
            "runtime_abi": 1,
        },
        "target": {"triple": "aarch64-apple-darwin", "features": ["neon"]},
        "reproduction": g0_reproduction(revision),
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


def _make_bundle(
    tmp_path: Path, revision: str = REVISION
) -> tuple[Path, dict[str, Any]]:
    bundle = tmp_path / f"g0-{len(list(tmp_path.iterdir()))}"
    bundle.mkdir()
    contents = {
        "fixture-manifest.json": (
            ROOT / "tests" / "fixtures" / "v1" / "manifest.json"
        ).read_bytes(),
        "host.json": canonical_json_bytes(_host_manifest(revision)) + b"\n",
        "report.md": b"# G0 verifier fixture\n",
    }
    for path, content in contents.items():
        (bundle / path).write_bytes(content)
    manifest = _run_manifest(contents, revision)
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


def _profile_fields(bundle: Path) -> set[tuple[str, str]]:
    """Return the stable artifact/field pairs from G0 profile diagnostics."""

    diagnostics = verify_bundle(bundle)
    assert all(validate_data(item, "diagnostic") == [] for item in diagnostics)
    return {
        (item["context"]["artifact"], item["context"]["field"])
        for item in diagnostics
        if item["code"] == "DFE-BUNDLE-009"
    }


def _repository_diagnostics(bundle: Path, checkout: Path) -> list[dict[str, Any]]:
    """Run the explicit repository gate and retain schema-valid diagnostics."""

    diagnostics = verify_g0_repository_bundle(bundle, checkout)
    assert diagnostics is not None
    assert all(validate_data(item, "diagnostic") == [] for item in diagnostics)
    return diagnostics


def _repository_fields(bundle: Path, checkout: Path) -> set[str]:
    """Return the repository fields from deterministic provenance diagnostics."""

    return {
        item["context"]["field"]
        for item in _repository_diagnostics(bundle, checkout)
        if item["code"] == "DFE-BUNDLE-009"
    }


def _write_canonical_json_artifact(
    bundle: Path, filename: str, document: dict[str, Any]
) -> None:
    (bundle / filename).write_bytes(canonical_json_bytes(document) + b"\n")


def _host_document(bundle: Path) -> dict[str, Any]:
    """Load the test-only host JSON before refreshing its artifact record."""

    document = json.loads((bundle / "host.json").read_text(encoding="ascii"))
    assert isinstance(document, dict)
    return document


def _refresh_host(
    bundle: Path, manifest: dict[str, Any], document: dict[str, Any]
) -> None:
    """Write one canonical host document and bind it into the run manifest."""

    _write_canonical_json_artifact(bundle, "host.json", document)
    _refresh_record(bundle, manifest, "host.json")


def _set_bundle_revision(bundle: Path, manifest: dict[str, Any], revision: str) -> None:
    project = manifest["project"]
    assert isinstance(project, dict)
    project["revision"] = revision
    manifest["reproduction"] = g0_reproduction(revision)
    host = _host_document(bundle)
    source = host["source"]
    assert isinstance(source, dict)
    source["revision"] = revision
    _refresh_host(bundle, manifest, host)


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _clone_checkout(tmp_path: Path, name: str = "checkout") -> Path:
    checkout = tmp_path / name
    result = subprocess.run(
        ["git", "clone", "--quiet", "--no-local", str(ROOT), str(checkout)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return checkout


def _commit_checked_in_bundle(tmp_path: Path, checkout: Path) -> Path:
    """Commit a valid fixture bundle after its recorded source revision."""

    source_bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    bundle = checkout / "results" / "g0" / "checked-in-fixture"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_bundle, bundle)
    _git(checkout, "add", "results/g0/checked-in-fixture")
    _git(
        checkout,
        "-c",
        "user.name=DecodeForge Test",
        "-c",
        "user.email=decodeforge-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Add checked-in G0 fixture bundle",
    )
    return bundle


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

    assert _codes(bundle) == ["DFE-BUNDLE-009"]
    assert not marker.exists()


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (
            "project",
            {
                ("run-manifest.json", "reproduction.source_revision"),
                (
                    "run-manifest.json",
                    "reproduction.environment.DECODEFORGE_SOURCE_REVISION",
                ),
                ("host.json", "source.revision"),
            },
        ),
        (
            "reproduction-source",
            {("run-manifest.json", "reproduction.source_revision")},
        ),
        (
            "reproduction-environment",
            {
                (
                    "run-manifest.json",
                    "reproduction.environment.DECODEFORGE_SOURCE_REVISION",
                )
            },
        ),
        ("host", {("host.json", "source.revision")}),
    ],
)
def test_g0_revision_records_must_all_agree(
    tmp_path: Path, record: str, expected: set[tuple[str, str]]
) -> None:
    """Every full revision fan-out is independently tied to project.revision."""

    bundle, manifest = _make_bundle(tmp_path)
    replacement = "f" * 40
    if record == "project":
        project = manifest["project"]
        assert isinstance(project, dict)
        project["revision"] = replacement
        _write_run(bundle, manifest)
    elif record == "reproduction-source":
        reproduction = manifest["reproduction"]
        assert isinstance(reproduction, dict)
        reproduction["source_revision"] = replacement
        _write_run(bundle, manifest)
    elif record == "reproduction-environment":
        reproduction = manifest["reproduction"]
        assert isinstance(reproduction, dict)
        environment = reproduction["environment"]
        assert isinstance(environment, dict)
        environment["DECODEFORGE_SOURCE_REVISION"] = replacement
        _write_run(bundle, manifest)
    else:
        host = _host_document(bundle)
        source = host["source"]
        assert isinstance(source, dict)
        source["revision"] = replacement
        _refresh_host(bundle, manifest, host)

    assert expected <= _profile_fields(bundle)


def test_g0_requires_clean_records_and_snapshot_fixture_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-file semantics retain snapshots rather than reopening bundle paths."""

    bundle, manifest = _make_bundle(tmp_path)
    project = manifest["project"]
    assert isinstance(project, dict)
    project["dirty"] = True
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "project.dirty") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    source = host["source"]
    assert isinstance(source, dict)
    source["dirty"] = True
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "source.dirty") in _profile_fields(bundle)

    bundle, _ = _make_bundle(tmp_path)
    verified, diagnostics = evidence._verified_g0_bundle(bundle)
    assert diagnostics is None
    assert verified is not None
    changed_fixture = dict(verified.fixture_manifest)
    changed_fixture["format"] = "another-format"
    changed_fixture["numeric_mode"] = "another-numeric-mode"

    def _no_path_reopen(*_: object) -> NoReturn:
        raise AssertionError("cross-file semantics reopened a bundle path")

    monkeypatch.setattr(evidence, "_read_regular_snapshot", _no_path_reopen)
    fields = {
        item["context"]["field"]
        for item in evidence._g0_semantic_diagnostics(
            replace(verified, fixture_manifest=changed_fixture)
        )
    }
    assert fields == {"format", "numeric_mode"}


def test_g0_target_profile_permits_extra_host_features_but_not_drift(
    tmp_path: Path,
) -> None:
    """Target features are canonical, include NEON, and are a host subset."""

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    cpu = host["cpu"]
    assert isinstance(cpu, dict)
    cpu["features"] = ["neon", "sve"]
    _refresh_host(bundle, manifest, host)
    assert verify_bundle(bundle) == []

    bundle, manifest = _make_bundle(tmp_path)
    target = manifest["target"]
    assert isinstance(target, dict)
    target["features"] = ["neon", "sve"]
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "target.features") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    target = manifest["target"]
    assert isinstance(target, dict)
    target["features"] = ["neon", "asimd"]
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "target.features") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    cpu = host["cpu"]
    assert isinstance(cpu, dict)
    cpu["features"] = ["neon", "asimd"]
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "cpu.features") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    target = manifest["target"]
    assert isinstance(target, dict)
    target["features"] = []
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "target.features") in _profile_fields(bundle)


def test_g0_closes_host_target_toolchain_check_and_reproduction_profiles(
    tmp_path: Path,
) -> None:
    """The schema-valid G0 profile is narrower than its generic schemas."""

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    host["role"] = "development"
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "role") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    host["architecture"] = "x86_64"
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "architecture") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    operating_system = host["os"]
    assert isinstance(operating_system, dict)
    operating_system["name"] = "Linux"
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "os.name") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    target = manifest["target"]
    assert isinstance(target, dict)
    target["triple"] = "x86_64-unknown-linux-gnu"
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "target.triple") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    toolchains = host["toolchains"]
    assert isinstance(toolchains, dict)
    toolchains["uv"] = "0.0.0"
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "toolchains") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    checks = manifest["checks"]
    assert isinstance(checks, dict)
    checks["schema"] = "fail"
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "checks") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    not_applicable = manifest["not_applicable"]
    assert isinstance(not_applicable, list)
    not_applicable.reverse()
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "not_applicable") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    reproduction = manifest["reproduction"]
    assert isinstance(reproduction, dict)
    commands = reproduction["commands"]
    assert isinstance(commands, list)
    first_command = commands[0]
    assert isinstance(first_command, dict)
    first_command["id"] = "not-the-closed-command"
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "reproduction") in _profile_fields(bundle)


def test_g0_closes_the_emitted_m4_profile_and_optional_shapes(
    tmp_path: Path,
) -> None:
    """G0 admits only the documented non-identifying Apple-M4 record shape."""

    bundle, manifest = _make_bundle(tmp_path)
    project = manifest["project"]
    assert isinstance(project, dict)
    project["compiler_version"] = "9.9.9"
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "project.compiler_version") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    host["host_id"] = "m4-owner-hostname"
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "host_id") in _profile_fields(bundle)

    for field, value in (
        ("model", "Apple M3"),
        ("physical_cores", 8),
        ("logical_cores", 8),
    ):
        bundle, manifest = _make_bundle(tmp_path)
        host = _host_document(bundle)
        cpu = host["cpu"]
        assert isinstance(cpu, dict)
        cpu[field] = value
        _refresh_host(bundle, manifest, host)
        assert ("host.json", f"cpu.{field}") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    manifest["metadata"] = {"hostname": "not-allowed"}
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "root.keys") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    target = manifest["target"]
    assert isinstance(target, dict)
    target["metadata"] = {"absolute_path": "not-allowed"}
    _write_run(bundle, manifest)
    assert ("run-manifest.json", "target.keys") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    host["execution"] = {"thread_environment": {"USER": "not-allowed"}}
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "root.keys") in _profile_fields(bundle)

    bundle, manifest = _make_bundle(tmp_path)
    host = _host_document(bundle)
    toolchains = host["toolchains"]
    assert isinstance(toolchains, dict)
    toolchains["home"] = "not-allowed"
    _refresh_host(bundle, manifest, host)
    assert ("host.json", "toolchains.keys") in _profile_fields(bundle)


def test_repository_gate_accepts_clean_checked_in_bundle_and_explicit_cli(
    tmp_path: Path,
) -> None:
    """A results/g0 bundle is valid when HEAD descends from its revision."""

    checkout = _clone_checkout(tmp_path, "checkout-cleanliness")
    bundle = _commit_checked_in_bundle(tmp_path, checkout)
    assert verify_g0_repository_bundle(bundle, checkout) == []

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_g0_repository.py"),
            "--bundle",
            str(bundle),
            "--checkout",
            str(checkout),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "g0-repository-check: ok\n"
    assert result.stderr == ""


def test_repository_gate_accepts_outside_bundle_from_an_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repository gate consumes only its two explicit paths."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    assert verify_g0_repository_bundle(bundle, checkout) == []
    _git(checkout, "checkout", "--detach")
    assert verify_g0_repository_bundle(bundle, checkout) == []


def test_repository_cli_emits_canonical_path_free_diagnostics(tmp_path: Path) -> None:
    """The explicit command surface returns only canonical public diagnostics."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_g0_repository.py"),
            "--bundle",
            str(bundle),
            "--checkout",
            str(checkout),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert len(lines) == 1
    diagnostic = json.loads(lines[0])
    assert isinstance(diagnostic, dict)
    assert validate_data(diagnostic, "diagnostic") == []
    assert diagnostic["context"] == {
        "artifact": "run-manifest.json",
        "field": "project.revision",
    }
    assert lines[0] == json.dumps(
        diagnostic, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    assert str(bundle) not in result.stderr
    assert str(checkout) not in result.stderr


@pytest.mark.parametrize("state", ["staged", "unstaged", "untracked"])
def test_repository_gate_rejects_all_nonignored_checkout_changes(
    tmp_path: Path, state: str
) -> None:
    """Staged, unstaged, and untracked changes all invalidate provenance."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    if state == "staged":
        dirty = checkout / "staged-change"
        dirty.write_text("staged\n", encoding="ascii")
        _git(checkout, "add", dirty.name)
    elif state == "unstaged":
        readme = checkout / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        (checkout / "untracked-change").write_text("untracked\n", encoding="ascii")
    assert _repository_fields(bundle, checkout) == {"repository.cleanliness"}


def test_repository_gate_allows_ignored_changes_and_isolates_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignored worktree state and hostile caller Git variables do not leak in."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    exclude = checkout / ".git" / "info" / "exclude"
    exclude.write_text(
        exclude.read_text(encoding="utf-8") + "ignored-only\n", encoding="utf-8"
    )
    (checkout / "ignored-only").write_text("ignored\n", encoding="ascii")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "not-a-git-directory"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "not-a-worktree"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "hostile-config"))
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    assert verify_g0_repository_bundle(bundle, checkout) == []
    assert repository._git_environment()["GIT_NO_LAZY_FETCH"] == "1"


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_repository_gate_rejects_hidden_index_entries(
    tmp_path: Path, index_flag: str
) -> None:
    """Neither index bit may hide a modified tracked file from the gate."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    _git(checkout, "update-index", index_flag, "README.md")
    readme = checkout / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "hidden\n", encoding="utf-8")

    assert _repository_fields(bundle, checkout) == {"repository.index_flags"}


def test_repository_gate_requires_exact_worktree_and_no_local_worktree_override(
    tmp_path: Path,
) -> None:
    """Discovery from a nested path and local core.worktree are not accepted."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    assert _repository_fields(bundle, checkout / "docs") == {"repository.checkout"}

    shadow = tmp_path / "redirected-worktree"
    shadow.mkdir()
    _git(checkout, "config", "core.worktree", str(shadow))
    assert _repository_fields(bundle, checkout) == {"repository.checkout"}


def test_repository_gate_rejects_a_bare_checkout(tmp_path: Path) -> None:
    """A Git directory without an exact worktree cannot attest provenance."""

    bare = tmp_path / "bare.git"
    result = subprocess.run(
        ["git", "clone", "--quiet", "--bare", "--no-local", str(ROOT), str(bare)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)

    assert _repository_fields(bundle, bare) == {"repository.checkout"}


def test_repository_gate_detects_mode_changes_despite_local_filemode_override(
    tmp_path: Path,
) -> None:
    """The gate forces file-mode checking instead of trusting local config."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    readme = checkout / "README.md"
    _git(checkout, "config", "core.filemode", "false")
    readme.chmod(readme.stat().st_mode ^ stat.S_IXUSR)

    assert _repository_fields(bundle, checkout) == {"repository.cleanliness"}


def test_repository_gate_fails_closed_on_unreadable_untracked_status_warning(
    tmp_path: Path,
) -> None:
    """A status warning cannot be mistaken for an empty, clean checkout."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    bound_checkout = repository._checkout_binding(checkout)
    assert bound_checkout is not None
    unreadable = checkout / "unreadable-untracked"
    unreadable.mkdir()
    (unreadable / "entry").write_text("unreadable\n", encoding="ascii")
    mode = stat.S_IMODE(unreadable.stat().st_mode)
    unreadable.chmod(0)
    try:
        probe = subprocess.run(
            [
                "git",
                *repository._GIT_GLOBAL_OPTIONS,
                f"--git-dir={bound_checkout.git_dir}",
                f"--work-tree={bound_checkout.work_tree}",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=no",
                "--ignore-submodules=none",
            ],
            cwd=checkout,
            env=repository._git_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        if not (probe.returncode == 0 and probe.stdout == b"" and bool(probe.stderr)):
            pytest.skip("Git/permissions did not reproduce the empty-status warning")
        diagnostics = _repository_diagnostics(bundle, checkout)
        assert {
            item["context"]["field"]
            for item in diagnostics
            if item["code"] == "DFE-BUNDLE-009"
        } == {"repository.cleanliness"}
        assert "unreadable-untracked" not in json.dumps(diagnostics)
    finally:
        unreadable.chmod(mode)


def test_repository_gate_disables_local_fsmonitor_helpers(tmp_path: Path) -> None:
    """A hostile local status helper cannot execute during provenance checks."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    marker = tmp_path / "fsmonitor-was-executed"
    helper = tmp_path / "malicious-fsmonitor"
    helper.write_text(
        f"#!/bin/sh\n: > {str(marker)!r}\nprintf 'version=2\\n\\n'\n",
        encoding="ascii",
    )
    helper.chmod(0o755)
    _git(checkout, "config", "core.fsmonitor", str(helper))

    assert verify_g0_repository_bundle(bundle, checkout) == []
    assert not marker.exists()


def test_hardened_git_runner_enforces_stdout_stderr_and_time_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner failures are bounded before a hostile child can allocate output."""

    output_script = tmp_path / "git-output-flood"
    output_script.write_text("#!/bin/sh\nhead -c 65537 /dev/zero\n", encoding="ascii")
    output_script.chmod(0o755)
    monkeypatch.setattr(repository, "_GIT_EXECUTABLE", str(output_script))
    output_result = repository._run_git_process(tmp_path, (), ("ignored",))
    assert output_result.failure == "output-limit"
    assert output_result.stdout == b""

    stderr_script = tmp_path / "git-stderr-flood"
    stderr_script.write_text(
        "#!/bin/sh\nhead -c 65537 /dev/zero >&2\n", encoding="ascii"
    )
    stderr_script.chmod(0o755)
    monkeypatch.setattr(repository, "_GIT_EXECUTABLE", str(stderr_script))
    stderr_result = repository._run_git_process(tmp_path, (), ("ignored",))
    assert stderr_result.failure == "stderr"
    assert stderr_result.stdout == b""

    timeout_script = tmp_path / "git-timeout"
    timeout_script.write_text("#!/bin/sh\nsleep 5\n", encoding="ascii")
    timeout_script.chmod(0o755)
    monkeypatch.setattr(repository, "_GIT_EXECUTABLE", str(timeout_script))
    monkeypatch.setattr(repository, "_GIT_TIMEOUT_SECONDS", 0.1)
    timeout_result = repository._run_git_process(tmp_path, (), ("ignored",))
    assert timeout_result.failure == "timeout"
    assert timeout_result.stdout == b""


def test_hardened_git_runner_kills_a_descendant_after_wrapper_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrapper cannot escape its timeout by leaving a stdout-owning child."""

    marker = tmp_path / "descendant-survived"
    wrapper = tmp_path / "git-wrapper-with-child"
    wrapper.write_text(
        f"#!/bin/sh\n(sleep 0.3; : > {str(marker)!r}) &\nexit 0\n",
        encoding="ascii",
    )
    wrapper.chmod(0o755)
    monkeypatch.setattr(repository, "_GIT_EXECUTABLE", str(wrapper))
    monkeypatch.setattr(repository, "_GIT_TIMEOUT_SECONDS", 0.05)

    result = repository._run_git_process(tmp_path, (), ("ignored",))
    assert result.failure == "timeout"
    time.sleep(0.4)
    assert not marker.exists()


def test_repository_gate_fails_closed_for_a_missing_revision_object(
    tmp_path: Path,
) -> None:
    """A missing object cannot trigger a promisor fetch or masquerade as false."""

    checkout = _clone_checkout(tmp_path)
    bundle, manifest = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    _set_bundle_revision(bundle, manifest, "f" * 40)

    assert _repository_fields(bundle, checkout) == {"project.revision"}


def test_repository_gate_reports_git_uncertainty_without_false_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Object and ancestry runner failures retain their safe tri-state meaning."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)

    def _unknown_commit(*_: object) -> None:
        return None

    monkeypatch.setattr(repository, "_is_commit", _unknown_commit)
    diagnostics = _repository_diagnostics(bundle, checkout)
    assert diagnostics[0]["context"]["field"] == "repository.baseline"
    assert diagnostics[0]["summary"].endswith("could not be determined safely.")

    monkeypatch.undo()

    def _known_commit(*_: object) -> bool:
        return True

    def _unknown_ancestry(*_: object) -> None:
        return None

    monkeypatch.setattr(repository, "_is_commit", _known_commit)
    monkeypatch.setattr(repository, "_is_ancestor", _unknown_ancestry)
    diagnostics = _repository_diagnostics(bundle, checkout)
    assert diagnostics[0]["context"]["field"] == "repository.ancestry.baseline"
    assert diagnostics[0]["summary"].endswith("could not be determined safely.")


def test_repository_gate_rejects_noncommit_and_wrong_ancestry_revisions(
    tmp_path: Path,
) -> None:
    """The recorded revision must be a commit between baseline and HEAD."""

    checkout = _clone_checkout(tmp_path)
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")

    bundle, manifest = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    _set_bundle_revision(bundle, manifest, tree)
    assert _repository_fields(bundle, checkout) == {"project.revision"}

    unrelated = _git(
        checkout,
        "-c",
        "user.name=DecodeForge Test",
        "-c",
        "user.email=decodeforge-test@example.invalid",
        "commit-tree",
        tree,
    )
    bundle, manifest = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    _set_bundle_revision(bundle, manifest, unrelated)
    assert _repository_fields(bundle, checkout) == {"repository.ancestry.baseline"}

    head = _git(checkout, "rev-parse", "HEAD")
    future = _git(
        checkout,
        "-c",
        "user.name=DecodeForge Test",
        "-c",
        "user.email=decodeforge-test@example.invalid",
        "commit-tree",
        tree,
        "-p",
        head,
    )
    bundle, manifest = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    _set_bundle_revision(bundle, manifest, future)
    assert _repository_fields(bundle, checkout) == {"repository.ancestry.head"}


def test_repository_gate_rejects_a_noncommit_checkout_head(tmp_path: Path) -> None:
    """HEAD itself must resolve through Git's commit peeling syntax."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    tree = _git(checkout, "rev-parse", "HEAD^{tree}")
    _git(checkout, "checkout", "--detach")
    (checkout / ".git" / "HEAD").write_text(f"{tree}\n", encoding="ascii")
    assert _repository_fields(bundle, checkout) == {"repository.head"}


def test_repository_gate_compares_the_verified_fixture_snapshot_to_git(
    tmp_path: Path,
) -> None:
    """A hash/ID-recomputed copied fixture still must equal the revision blob."""

    checkout = _clone_checkout(tmp_path)
    bundle, manifest = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    fixture = json.loads((bundle / "fixture-manifest.json").read_text(encoding="ascii"))
    assert isinstance(fixture, dict)
    artifacts = fixture["artifacts"]
    assert isinstance(artifacts, list)
    first_artifact = artifacts[0]
    assert isinstance(first_artifact, dict)
    first_artifact["sha256"] = "0" * 64
    _write_canonical_json_artifact(bundle, "fixture-manifest.json", fixture)
    _refresh_record(bundle, manifest, "fixture-manifest.json")
    assert manifest["bundle_id"] == g0_bundle_id(manifest)

    assert _repository_fields(bundle, checkout) == {
        "fixture-manifest.json.repository_blob"
    }


def test_repository_gate_uses_only_the_portable_verified_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-verification bundle mutation cannot redirect the Git blob compare."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    original = evidence._verified_g0_bundle

    def _verify_then_mutate(path: Path) -> tuple[object, object]:
        verified, diagnostics = original(path)
        (bundle / "fixture-manifest.json").write_bytes(b"replaced after snapshot")
        return verified, diagnostics

    monkeypatch.setattr(repository, "_verified_g0_bundle", _verify_then_mutate)
    assert verify_g0_repository_bundle(bundle, checkout) == []


def test_repository_gate_rechecks_head_and_cleanliness_after_blob_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Late checkout movement and dirtiness cannot inherit early observations."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    original_fixture_blob = repository._fixture_blob

    def _move_head(checkout_path: Any, revision: str) -> bytes | None:
        blob = original_fixture_blob(checkout_path, revision)
        late_commit_file = checkout / "late-head-commit"
        late_commit_file.write_text("late\n", encoding="ascii")
        _git(checkout, "add", late_commit_file.name)
        _git(
            checkout,
            "-c",
            "user.name=DecodeForge Test",
            "-c",
            "user.email=decodeforge-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Move HEAD during gate",
        )
        return blob

    monkeypatch.setattr(repository, "_fixture_blob", _move_head)
    assert _repository_fields(bundle, checkout) == {"repository.head"}

    checkout = _clone_checkout(tmp_path, "checkout-late-dirty")
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    monkeypatch.setattr(repository, "_fixture_blob", original_fixture_blob)

    def _make_dirty(checkout_path: Any, revision: str) -> bytes | None:
        blob = original_fixture_blob(checkout_path, revision)
        (checkout / "late-untracked-change").write_text("late\n", encoding="ascii")
        return blob

    monkeypatch.setattr(repository, "_fixture_blob", _make_dirty)
    assert _repository_fields(bundle, checkout) == {"repository.cleanliness"}


def test_repository_gate_observes_head_after_final_clean_and_index_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch move after the final index observation cannot pass as stable."""

    checkout = _clone_checkout(tmp_path)
    bundle, _ = _make_bundle(tmp_path, G0_PROVENANCE_BASELINE)
    original_index_check = repository._checkout_index_unflagged
    observations = 0

    def _move_head_after_final_index(bound_checkout: Any) -> bool | None:
        nonlocal observations
        observed = original_index_check(bound_checkout)
        observations += 1
        if observations == 2:
            late_commit_file = checkout / "late-final-head-commit"
            late_commit_file.write_text("late\n", encoding="ascii")
            _git(checkout, "add", late_commit_file.name)
            _git(
                checkout,
                "-c",
                "user.name=DecodeForge Test",
                "-c",
                "user.email=decodeforge-test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "Move HEAD after final index check",
            )
        return observed

    monkeypatch.setattr(
        repository, "_checkout_index_unflagged", _move_head_after_final_index
    )
    assert _repository_fields(bundle, checkout) == {"repository.head"}
    assert observations == 2


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

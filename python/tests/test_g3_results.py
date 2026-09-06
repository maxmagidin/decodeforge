"""End-to-end and adversarial checks for the closed G3 result bundle."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import decodeforge.g3_results as g3_results
import pytest
from decodeforge.g3_preparation import receipt_identity
from decodeforge.g3_results import (
    G3ResultError,
    analyze_and_write_g3_result,
    analyze_g3_sessions,
    build_g3_bundle,
    canonical_json_bytes,
    load_g3_sessions,
    session_identity,
    verify_g3_result,
    write_g3_bundle,
)

from test_g3_evidence import _accepted_session

JsonObject = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]

EXPECTED_MEMBERS = {
    "README.md",
    "analysis.json",
    "asset-inventory.json",
    "correctness.json",
    "coverage.json",
    "environment.txt",
    "generated.txt",
    "manifest.json",
    "prompt.json",
    "timings.csv",
}


def _receipt(*, tool: Path | None = None) -> JsonObject:
    fixture_root = (ROOT / ".g3-test-fixture").absolute()
    tool_path = tool or fixture_root / "decodeforge-prepare-qproj"
    unsigned: JsonObject = {
        "schema_version": 1,
        "format": "decodeforge_g3_offline_preparation_receipt_v1",
        "protocol_id": "g3-tinyllama-qproj-generation-v1",
        "checkout": {"revision": "1" * 40, "dirty": False},
        "command": {
            "argv": [
                str(tool_path.absolute()),
                "--source",
                str(fixture_root / "model.safetensors"),
                "--output",
                str(fixture_root / "assets"),
            ]
        },
        "source": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "filename": "model.safetensors",
            "bytes": 2_200_119_864,
            "identity": (
                "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
            ),
        },
        "tool": {
            "name": "decodeforge-prepare-qproj",
            "version": "0.1.0",
            "executable_identity": "sha256:" + "9" * 64,
        },
        "output": {
            "asset_inventory_identity": (
                "sha256:f659b26572357af84a5e5b66138331a2e35c319c5c9b8300cf81f7ea217ae0de"
            ),
            "layer_count": 22,
            "total_packed_bytes": 103_809_024,
            "total_fallback_bytes": 369_098_752,
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "start_ns": 100,
            "stop_ns": 1100,
            "elapsed_ns": 1000,
        },
    }
    if tool is not None:
        unsigned["tool"]["executable_identity"] = (
            "sha256:" + hashlib.sha256(tool.read_bytes()).hexdigest()
        )
    return {**unsigned, "receipt_identity": receipt_identity(unsigned)}


def _sessions(receipt: JsonObject | None = None) -> list[JsonObject]:
    receipt = _receipt() if receipt is None else receipt
    projection = {
        "source": "separately_captured_prepare_command",
        "receipt_identity": receipt["receipt_identity"],
        "elapsed_ns": receipt["timing"]["elapsed_ns"],
        "asset_inventory_identity": receipt["output"]["asset_inventory_identity"],
    }
    prepare_command = shlex.join(receipt["command"]["argv"])
    sessions = []
    for index in range(3):
        session = _accepted_session()
        session["session_index"] = index
        session["session_id"] = f"synthetic-session-{index}"
        session["environment"]["runner_process_id"] = 2000 + index
        session["offline_preparation"] = deepcopy(projection)
        session["provenance"]["checkout"] = deepcopy(receipt["checkout"])
        session["provenance"]["rebuild_commands"]["prepare_assets"] = prepare_command
        session["provenance"]["rebuild_commands"]["run_session"] = (
            "python scripts/run_g3.py --session-index "
            f"{index} --session-id synthetic-session-{index} "
            f"--output .g3-test-fixture/session-{index}.json"
        )
        sessions.append(session)
    return sessions


def _write_sessions(tmp_path: Path, sessions: list[JsonObject]) -> list[Path]:
    paths = []
    for index, session in enumerate(sessions):
        path = tmp_path / f"session-{index}.json"
        path.write_bytes(canonical_json_bytes(session))
        paths.append(path)
    return paths


def _write_live_receipt(tmp_path: Path) -> tuple[Path, JsonObject]:
    tool = tmp_path / "decodeforge-prepare-qproj"
    tool.write_bytes(b"synthetic preparation tool\n")
    tool.chmod(0o700)
    receipt = _receipt(tool=tool)
    path = tmp_path / "preparation-receipt.json"
    path.write_bytes(canonical_json_bytes(receipt))
    return path, receipt


def _publish(tmp_path: Path) -> tuple[Path, list[JsonObject]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt = _receipt()
    sessions = _sessions(receipt)
    bundle = tmp_path / "bundle"
    write_g3_bundle(sessions, receipt, bundle)
    return bundle, sessions


def test_bundle_round_trip_is_lossless_and_closed(tmp_path: Path) -> None:
    bundle, sessions = _publish(tmp_path)
    assert {path.name for path in bundle.iterdir()} == EXPECTED_MEMBERS
    verify_g3_result(bundle)

    analysis = json.loads((bundle / "analysis.json").read_bytes())
    assert analysis["sessions"] == sessions
    assert analysis["preparation_receipt"] == _receipt()
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    assert [record["canonical_identity"] for record in manifest["sessions"]] == [
        session_identity(session) for session in sessions
    ]
    assert len(manifest["artifacts"]) == 9

    rows = list(
        csv.DictReader(io.StringIO((bundle / "timings.csv").read_text("utf-8")))
    )
    assert len(rows) == 3 * (7 + 24 * (5 + 1 + 44))
    assert sum(row["record_type"] == "q_projection_dispatch" for row in rows) == (
        3 * 24 * 44
    )


def test_bundle_bytes_are_stable_for_input_order() -> None:
    receipt = _receipt()
    sessions = _sessions(receipt)
    assert build_g3_bundle(sessions, receipt) == build_g3_bundle(
        list(reversed(sessions)), receipt
    )


def test_analyzer_live_verifies_receipt_tool_and_embeds_portable_receipt(
    tmp_path: Path,
) -> None:
    receipt_path, receipt = _write_live_receipt(tmp_path)
    sessions = _sessions(receipt)
    paths = _write_sessions(tmp_path, sessions)
    output = tmp_path / "bundle"
    analyze_and_write_g3_result(paths, receipt_path, output)
    assert (
        json.loads((output / "analysis.json").read_bytes())["preparation_receipt"]
        == receipt
    )
    verify_g3_result(output)

    Path(receipt["command"]["argv"][0]).write_bytes(b"changed tool\n")
    with pytest.raises(G3ResultError, match="could not be verified"):
        analyze_and_write_g3_result(paths, receipt_path, tmp_path / "rejected")


def test_bundle_rejects_unbound_preparation_receipt() -> None:
    receipt = _receipt()
    sessions = _sessions(receipt)
    changed = deepcopy(receipt)
    changed["timing"]["start_ns"] += 1
    changed["timing"]["stop_ns"] += 1
    unsigned = {
        key: value for key, value in changed.items() if key != "receipt_identity"
    }
    changed["receipt_identity"] = receipt_identity(unsigned)
    with pytest.raises(G3ResultError, match="does not bind exactly"):
        build_g3_bundle(sessions, changed)


def test_file_loader_requires_three_distinct_regular_inputs(tmp_path: Path) -> None:
    paths = _write_sessions(tmp_path, _sessions())
    assert load_g3_sessions(paths) == _sessions()
    with pytest.raises(G3ResultError, match="exactly three"):
        load_g3_sessions(paths[:2])
    with pytest.raises(G3ResultError, match="distinct files"):
        load_g3_sessions([paths[0], paths[0], paths[2]])
    link = tmp_path / "linked.json"
    link.symlink_to(paths[0])
    with pytest.raises(G3ResultError, match="non-symlink"):
        load_g3_sessions([link, paths[1], paths[2]])


def test_file_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    paths = _write_sessions(tmp_path, _sessions())
    paths[0].write_bytes(b'{"value":NaN}')
    with pytest.raises(G3ResultError, match="strict UTF-8 JSON"):
        load_g3_sessions(paths)


def test_file_loader_rejects_duplicate_keys_and_oversize(tmp_path: Path) -> None:
    paths = _write_sessions(tmp_path, _sessions())
    paths[0].write_bytes(b'{"value":1,"value":2}')
    with pytest.raises(G3ResultError, match="strict UTF-8 JSON"):
        load_g3_sessions(paths)


def test_file_loader_rejects_symlinked_ancestor_and_hardlink(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    paths = _write_sessions(real_parent, _sessions())
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(G3ResultError, match="safely"):
        load_g3_sessions([linked_parent / paths[0].name, paths[1], paths[2]])

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(paths[0])
    with pytest.raises(G3ResultError, match="non-symlink regular"):
        load_g3_sessions(paths)


@pytest.mark.parametrize("content", [b"[]", b'{"session_index":"wrong"}'])
def test_cli_turns_malformed_session_shapes_into_closed_errors(
    tmp_path: Path, content: bytes
) -> None:
    receipt_path, receipt = _write_live_receipt(tmp_path)
    paths = _write_sessions(tmp_path, _sessions(receipt))
    paths[0].write_bytes(content)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_g3_result.py"),
            "--sessions",
            *(str(path) for path in paths),
            "--preparation-receipt",
            str(receipt_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "g3-analysis: error:" in result.stderr
    with paths[0].open("wb") as oversized:
        oversized.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(G3ResultError, match="byte bound"):
        load_g3_sessions(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("index", "session_index"),
        ("session_id", "session IDs"),
        ("process_id", "process IDs"),
        ("spec", "generation-session contract"),
        ("host", "generation-session contract"),
        ("assets", "generation-session contract"),
        ("provenance", "share checkout"),
        ("bridge", "share checkout"),
        ("preparation", "share spec"),
        ("receipt", "share spec"),
    ],
)
def test_cross_session_invariants_are_closed(mutation: str, message: str) -> None:
    sessions = _sessions()
    if mutation == "index":
        sessions[2]["session_index"] = 1
    elif mutation == "session_id":
        sessions[2]["session_id"] = sessions[1]["session_id"]
    elif mutation == "process_id":
        sessions[2]["environment"]["runner_process_id"] = 2001
    elif mutation == "spec":
        sessions[2]["experiment_spec"]["path"] = "different.json"
    elif mutation == "host":
        sessions[2]["environment"]["runner_process_id"] = 9999
        sessions[2]["environment"]["host"]["cpu_model"] = "different"
    elif mutation == "assets":
        sessions[2]["assets"]["entries"][0]["tensor_identity"] = "sha256:" + "1" * 64
    elif mutation == "provenance":
        sessions[2]["provenance"]["checkout"]["revision"] = "2" * 40
    elif mutation == "bridge":
        sessions[2]["provenance"]["bridge_library"]["sha256"] = "8" * 64
    elif mutation == "receipt":
        sessions[2]["offline_preparation"]["receipt_identity"] = "sha256:" + "2" * 64
    else:
        sessions[2]["offline_preparation"]["elapsed_ns"] += 1
    with pytest.raises(G3ResultError, match=message):
        analyze_g3_sessions(sessions)


def test_session_replay_commands_may_vary_but_build_command_must_not() -> None:
    sessions = _sessions()
    assert [session["session_index"] for session in analyze_g3_sessions(sessions)] == [
        0,
        1,
        2,
    ]
    sessions[2]["provenance"]["rebuild_commands"]["build_bridge"] = (
        "env CARGO_TARGET_DIR=/opt/other-target make build-g3-bridge"
    )
    with pytest.raises(G3ResultError, match="build"):
        analyze_g3_sessions(sessions)


def test_publish_refuses_overwrite_and_preserves_existing_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "bundle"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(G3ResultError, match="already exists"):
        write_g3_bundle(_sessions(), _receipt(), target)
    assert marker.read_text(encoding="utf-8") == "keep"

    linked_target = tmp_path / "linked-bundle"
    linked_target.symlink_to(target, target_is_directory=True)
    with pytest.raises(G3ResultError, match="already exists"):
        write_g3_bundle(_sessions(), _receipt(), linked_target)


def test_publish_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-output"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(G3ResultError, match="publication failed"):
        write_g3_bundle(_sessions(), _receipt(), linked_parent / "bundle")
    assert list(real_parent.iterdir()) == []


def test_publish_rejects_visible_parent_swap_after_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "visible-parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"
    original_install = g3_results._install_no_overwrite

    def install_then_swap(parent_fd: int, staging: str, target: str) -> None:
        original_install(parent_fd, staging, target)
        parent.rename(moved)
        parent.mkdir()

    monkeypatch.setattr(g3_results, "_install_no_overwrite", install_then_swap)
    with pytest.raises(G3ResultError, match="visible parent path changed"):
        write_g3_bundle(_sessions(), _receipt(), parent / "bundle")
    assert not (parent / "bundle").exists()
    assert not (moved / "bundle").exists()


def test_publish_reverifies_content_after_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    original_install = g3_results._install_no_overwrite

    def install_then_mutate(parent_fd: int, staging: str, target: str) -> None:
        original_install(parent_fd, staging, target)
        root_fd = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            member_fd = os.open("README.md", os.O_WRONLY, dir_fd=root_fd)
            try:
                os.ftruncate(member_fd, 0)
                os.write(member_fd, b"mutated after staging verification\n")
            finally:
                os.close(member_fd)
        finally:
            os.close(root_fd)

    monkeypatch.setattr(g3_results, "_install_no_overwrite", install_then_mutate)
    with pytest.raises(G3ResultError, match="README.md"):
        write_g3_bundle(_sessions(), _receipt(), output)
    assert not output.exists()


def test_failed_install_cleans_only_unchanged_anchored_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_install(parent_fd: int, staging: str, target: str) -> None:
        del parent_fd, staging, target
        raise G3ResultError("injected install failure")

    monkeypatch.setattr(g3_results, "_install_no_overwrite", fail_install)
    with pytest.raises(G3ResultError, match="injected install failure"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


def test_mid_write_failure_cleans_created_members_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = g3_results._write_new_at
    calls = 0

    def fail_fourth_write(parent_fd: int, name: str, content: bytes) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected member write failure")
        return original_write(parent_fd, name, content)

    monkeypatch.setattr(g3_results, "_write_new_at", fail_fourth_write)
    with pytest.raises(G3ResultError, match="publication failed"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


def test_member_fsync_failure_removes_the_created_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fsync = os.fsync
    calls = 0

    def fail_first_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected member sync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_sync)
    with pytest.raises(G3ResultError, match="publication failed"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")
    assert list(tmp_path.iterdir()) == []


def test_failed_member_write_preserves_a_replacement_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    original_write = os.write
    swapped = False

    def swap_after_write(descriptor: int, content: bytes | bytearray) -> int:
        nonlocal swapped
        written = original_write(descriptor, content)
        if not swapped:
            swapped = True
            os.rename(
                "member.txt",
                "moved-owned-member.txt",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            replacement = os.open(
                "member.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                original_write(replacement, b"replacement\n")
            finally:
                os.close(replacement)
            raise OSError("injected member replacement")
        return written

    monkeypatch.setattr(os, "write", swap_after_write)
    try:
        with pytest.raises(OSError, match="injected member replacement"):
            g3_results._write_new_at(parent_fd, "member.txt", b"owned content\n")
    finally:
        os.close(parent_fd)
    assert (tmp_path / "member.txt").read_bytes() == b"replacement\n"
    assert (tmp_path / "moved-owned-member.txt").read_bytes() == b"owned content\n"


def test_staging_open_failure_preserves_unanchored_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = os.open

    def fail_staging_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if isinstance(path, str) and path.startswith(".bundle.staging-"):
            raise OSError("injected staging open failure")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_staging_open)
    with pytest.raises(G3ResultError, match="publication failed"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")
    staging = list(tmp_path.iterdir())
    assert len(staging) == 1
    assert staging[0].name.startswith(".bundle.staging-")
    assert staging[0].is_dir()
    assert list(staging[0].iterdir()) == []


def test_staging_named_stat_failure_removes_anchored_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_stat = g3_results._stat_at
    failed = False

    def fail_first_staging_stat(parent_fd: int, name: str) -> os.stat_result:
        nonlocal failed
        if not failed and name.startswith(".bundle.staging-"):
            failed = True
            raise OSError("injected staging stat failure")
        return original_stat(parent_fd, name)

    monkeypatch.setattr(g3_results, "_stat_at", fail_first_staging_stat)
    with pytest.raises(G3ResultError, match="publication failed"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")
    assert failed
    assert list(tmp_path.iterdir()) == []


def test_cleanup_does_not_remove_replacement_staging_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved_name = "moved-owned-staging"

    def swap_and_fail(parent_fd: int, staging: str, target: str) -> None:
        del target
        os.rename(
            staging,
            moved_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(staging, dir_fd=parent_fd)
        replacement_root_fd = os.open(
            staging, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd
        )
        try:
            replacement_fd = os.open(
                "unrelated.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_root_fd,
            )
            os.close(replacement_fd)
        finally:
            os.close(replacement_root_fd)
        raise G3ResultError("injected swapped staging")

    monkeypatch.setattr(g3_results, "_install_no_overwrite", swap_and_fail)
    with pytest.raises(G3ResultError, match="injected swapped staging"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")
    replacement = next(
        path for path in tmp_path.iterdir() if path.name.startswith(".bundle.staging-")
    )
    assert (replacement / "unrelated.txt").is_file()
    assert (tmp_path / moved_name / "analysis.json").is_file()


def test_post_install_target_replacement_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moved_name = "moved-installed-bundle"
    original_install = g3_results._install_no_overwrite

    def replace_target(parent_fd: int, staging: str, target: str) -> None:
        original_install(parent_fd, staging, target)
        os.rename(
            target,
            moved_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(target, dir_fd=parent_fd)
        replacement_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd)
        try:
            marker_fd = os.open(
                "unrelated.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=replacement_fd,
            )
            os.close(marker_fd)
        finally:
            os.close(replacement_fd)

    monkeypatch.setattr(g3_results, "_install_no_overwrite", replace_target)
    output = tmp_path / "bundle"
    with pytest.raises(G3ResultError, match="does not match staging"):
        write_g3_bundle(_sessions(), _receipt(), output)
    assert (output / "unrelated.txt").is_file()
    assert (tmp_path / moved_name / "analysis.json").is_file()


def test_parent_sync_failure_reports_installed_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original_sync = g3_results._fsync_directory_fd

    def fail_second_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise G3ResultError("injected sync failure")
        original_sync(descriptor)

    monkeypatch.setattr(g3_results, "_fsync_directory_fd", fail_second_sync)
    output = tmp_path / "bundle"
    with pytest.raises(G3ResultError, match="installed but its parent sync failed"):
        write_g3_bundle(_sessions(), _receipt(), output)
    assert not output.exists()


def test_descriptor_features_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "O_NOFOLLOW", 0)
    with pytest.raises(G3ResultError, match="requires O_NOFOLLOW"):
        write_g3_bundle(_sessions(), _receipt(), tmp_path / "bundle")


@pytest.mark.parametrize("member", sorted(EXPECTED_MEMBERS))
def test_verifier_rejects_every_tampered_member(tmp_path: Path, member: str) -> None:
    bundle, _ = _publish(tmp_path)
    path = bundle / member
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(G3ResultError):
        verify_g3_result(bundle)


def test_verifier_rejects_missing_extra_and_symlink_members(tmp_path: Path) -> None:
    bundle, _ = _publish(tmp_path)
    (bundle / "generated.txt").unlink()
    with pytest.raises(G3ResultError, match="missing"):
        verify_g3_result(bundle)

    bundle, _ = _publish(tmp_path / "extra-case")
    (bundle / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(G3ResultError, match="extra"):
        verify_g3_result(bundle)

    bundle, _ = _publish(tmp_path / "symlink-case")
    generated = bundle / "generated.txt"
    generated.unlink()
    generated.symlink_to(bundle / "README.md")
    with pytest.raises(G3ResultError, match="safely|regular"):
        verify_g3_result(bundle)

    root_link = tmp_path / "bundle-link"
    root_link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(G3ResultError, match="safely|root"):
        verify_g3_result(root_link)


def test_verifier_rejects_hardlinked_member(tmp_path: Path) -> None:
    bundle, _ = _publish(tmp_path)
    outside_link = tmp_path / "readme-hardlink"
    outside_link.hardlink_to(bundle / "README.md")
    with pytest.raises(G3ResultError, match="singly linked"):
        verify_g3_result(bundle)


def test_verifier_detects_member_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _publish(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    original_read = os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        content = original_read(descriptor, size)
        if not swapped:
            swapped = True
            os.replace(replacement, bundle / "README.md")
        return content

    monkeypatch.setattr(os, "read", swapping_read)
    with pytest.raises(G3ResultError, match="changed while read"):
        verify_g3_result(bundle)


def test_verifier_detects_bundle_root_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _publish(tmp_path)
    moved = tmp_path / "moved-bundle"
    original_listdir = os.listdir
    swapped = False

    def swapping_listdir(descriptor: int) -> list[str]:
        nonlocal swapped
        names = original_listdir(descriptor)
        if not swapped:
            swapped = True
            bundle.rename(moved)
            bundle.mkdir()
        return names

    monkeypatch.setattr(os, "listdir", swapping_listdir)
    with pytest.raises(G3ResultError, match="inventory changed|root changed"):
        verify_g3_result(bundle)


def test_loader_detects_session_ancestor_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_parent = tmp_path / "sessions"
    session_parent.mkdir()
    paths = _write_sessions(session_parent, _sessions())
    moved = tmp_path / "moved-sessions"
    original_read = os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        content = original_read(descriptor, size)
        if not swapped:
            swapped = True
            session_parent.rename(moved)
            session_parent.mkdir()
        return content

    monkeypatch.setattr(os, "read", swapping_read)
    with pytest.raises(G3ResultError, match="ancestor changed"):
        load_g3_sessions(paths)


def test_verifier_rejects_nonfinite_retained_evidence(tmp_path: Path) -> None:
    bundle, _ = _publish(tmp_path)
    (bundle / "analysis.json").write_bytes(b'{"summary":NaN}')
    with pytest.raises(G3ResultError, match="strict UTF-8 JSON"):
        verify_g3_result(bundle)


def test_summary_and_manifest_cannot_be_refreshed_to_hide_tampering(
    tmp_path: Path,
) -> None:
    bundle, _ = _publish(tmp_path)
    analysis = json.loads((bundle / "analysis.json").read_bytes())
    analysis["summary"]["accepted_session_count"] = 2
    (bundle / "analysis.json").write_bytes(canonical_json_bytes(analysis))
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    record = next(
        item for item in manifest["artifacts"] if item["path"] == "analysis.json"
    )
    content = (bundle / "analysis.json").read_bytes()
    record["bytes"] = len(content)
    record["sha256"] = hashlib.sha256(content).hexdigest()
    (bundle / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(G3ResultError, match="analysis.json"):
        verify_g3_result(bundle)


def test_session_canonical_hash_disagreement_is_rejected(tmp_path: Path) -> None:
    bundle, _ = _publish(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    manifest["sessions"][0]["canonical_identity"] = "sha256:" + "0" * 64
    (bundle / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(G3ResultError, match="manifest.json"):
        verify_g3_result(bundle)


def test_analyzer_does_not_mutate_caller_sessions() -> None:
    sessions = _sessions()
    before = deepcopy(sessions)
    receipt = _receipt()
    build_g3_bundle(sessions, receipt)
    assert sessions == before

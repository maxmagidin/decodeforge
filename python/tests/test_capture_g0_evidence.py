"""Minimal privacy-safe G0 evidence capture tests."""

from __future__ import annotations

import errno
import fcntl
import importlib.util
import json
import os
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from decodeforge.contracts import ROOT
from decodeforge.g0_evidence import (
    G0_COMPILER_VERSION,
    G0_HOST_CPU_MODEL,
    G0_HOST_ID,
    G0_HOST_LOGICAL_CORES,
    G0_HOST_PHYSICAL_CORES,
    G0_PROVENANCE_BASELINE,
    canonical_json_bytes,
    g0_reproduction,
    verify_g0_bundle,
)

REVISION = G0_PROVENANCE_BASELINE
LATE_REVISION = "f" * 40
_CAPTURE_SPEC = importlib.util.spec_from_file_location(
    "decodeforge_test_capture_g0_evidence",
    ROOT / "scripts" / "capture_g0_evidence.py",
)
assert _CAPTURE_SPEC is not None and _CAPTURE_SPEC.loader is not None
capture: Any = importlib.util.module_from_spec(_CAPTURE_SPEC)
sys.modules[_CAPTURE_SPEC.name] = capture
_CAPTURE_SPEC.loader.exec_module(capture)


@dataclass(frozen=True)
class _FakeCheckout:
    work_tree: Path


def _fake_tools(tmp_path: Path) -> Any:
    def pinned(name: str) -> Any:
        return capture._PinnedExecutable(
            path=(tmp_path / "pinned-tools" / name / name).resolve(),
            identity=capture._ExecutableIdentity(1, 1, 0o100755, 1, 1, 1),
        )

    return capture._CaptureTools(
        uv=pinned("uv"),
        rustup=pinned("rustup"),
        clang=pinned("clang"),
        sysctl=pinned("sysctl"),
        make=pinned("make"),
        outer_python=pinned("python"),
    )


def _host(revision: str = REVISION) -> dict[str, Any]:
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
        "os": {"name": "Darwin", "version": "15.5", "kernel": "24.5.0"},
        "toolchains": dict(capture.G0_TOOLCHAINS),
        "source": {"revision": revision, "dirty": False},
    }


def _prepare_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    fixture = checkout / "tests" / "fixtures" / "v1" / "manifest.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(
        (ROOT / "tests" / "fixtures" / "v1" / "manifest.json").read_bytes()
    )
    output_parent = tmp_path / "evidence"
    output_parent.mkdir()
    bound = _FakeCheckout(checkout)
    monkeypatch.setattr(capture, "_checkout_binding", lambda _: bound)
    monkeypatch.setattr(capture, "_producer_bound_to_checkout", lambda _: True)
    monkeypatch.setattr(
        capture, "_resolve_capture_tools", lambda: _fake_tools(tmp_path)
    )
    monkeypatch.setattr(capture, "_recheck_tools", lambda _: None)
    monkeypatch.setattr(capture, "_clean_revision", lambda _: REVISION)
    monkeypatch.setattr(capture, "_collect_host", lambda *_: _host())
    monkeypatch.setattr(capture, "_run_checks", lambda *_: None)
    monkeypatch.setattr(capture, "verify_g0_repository_bundle", lambda *_: [])
    return checkout, output_parent


def _assert_no_partial_target(parent: Path, target: Path) -> None:
    assert not target.exists()
    assert not list(parent.glob(f".{target.name}.capture-*"))


def test_capture_builds_a_canonical_private_verified_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "g0-capture"

    capture.capture(output, checkout)

    assert verify_g0_bundle(output) == []
    manifest_bytes = (output / "run-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == canonical_json_bytes(manifest)
    assert not manifest_bytes.endswith(b"\n")
    assert manifest["project"]["compiler_version"] == G0_COMPILER_VERSION
    assert manifest["reproduction"] == g0_reproduction(REVISION)
    assert (output / "fixture-manifest.json").read_bytes() == (
        ROOT / "tests" / "fixtures" / "v1" / "manifest.json"
    ).read_bytes()
    host_bytes = (output / "host.json").read_bytes()
    assert host_bytes == canonical_json_bytes(json.loads(host_bytes)) + b"\n"
    report = (output / "report.md").read_text(encoding="ascii")
    assert "no performance" in report
    evidence_bytes = b"".join(path.read_bytes() for path in output.iterdir())
    assert str(checkout).encode() not in evidence_bytes
    assert b"hostname" not in evidence_bytes
    assert b"/" + b"Users" + b"/" not in evidence_bytes
    assert b"HOME=" not in evidence_bytes


def test_capture_rejects_wrong_host_without_a_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "wrong-host"
    wrong_host = _host()
    cpu = wrong_host["cpu"]
    assert isinstance(cpu, dict)
    cpu["model"] = "Apple M3"
    monkeypatch.setattr(capture, "_collect_host", lambda *_: wrong_host)

    with pytest.raises(capture.CaptureError):
        capture.capture(output, checkout)
    _assert_no_partial_target(output_parent, output)


def test_capture_rejects_dirty_and_failed_checks_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_clean_revision = capture._clean_revision
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "dirty"
    bound = _FakeCheckout(checkout)
    monkeypatch.setattr(capture, "_checkout_binding", lambda _: bound)
    monkeypatch.setattr(capture, "_checkout_head", lambda _: REVISION)
    monkeypatch.setattr(capture, "_checkout_clean", lambda _: False)
    monkeypatch.setattr(capture, "_clean_revision", real_clean_revision)

    with pytest.raises(capture.CaptureError):
        capture.capture(output, checkout)
    _assert_no_partial_target(output_parent, output)

    monkeypatch.setattr(capture, "_clean_revision", lambda _: REVISION)
    monkeypatch.setattr(
        capture,
        "_run_checks",
        lambda *_: (_ for _ in ()).throw(capture.CaptureError("check failed")),
    )
    output = output_parent / "failed-check"
    with pytest.raises(capture.CaptureError):
        capture.capture(output, checkout)
    _assert_no_partial_target(output_parent, output)


def test_capture_rejects_an_existing_or_inside_checkout_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    existing = output_parent / "existing"
    existing.mkdir()

    with pytest.raises(capture.CaptureError):
        capture.capture(existing, checkout)
    assert existing.is_dir()
    assert list(existing.iterdir()) == []
    late_staging = output_parent / "late-staging"
    late_staging.mkdir()
    with pytest.raises(capture.CaptureError):
        capture._install_no_overwrite(late_staging, existing)
    assert late_staging.is_dir()

    inside = checkout / "results" / "g0"
    inside.parent.mkdir(parents=True)
    with pytest.raises(capture.CaptureError):
        capture.capture(inside, checkout)


def test_capture_rechecks_the_revision_and_cleans_its_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "late-revision"
    revisions = iter((REVISION, LATE_REVISION))
    monkeypatch.setattr(capture, "_clean_revision", lambda _: next(revisions))

    with pytest.raises(capture.CaptureError):
        capture.capture(output, checkout)
    _assert_no_partial_target(output_parent, output)


def test_capture_rechecks_tools_and_host_after_the_fixed_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "rechecked-tools"
    events: list[str] = []

    def _recheck(_: Any) -> None:
        events.append("tools")

    def _collect(*_: Any) -> dict[str, Any]:
        events.append("host")
        return _host()

    def _checks(*_: Any) -> None:
        events.append("checks")

    monkeypatch.setattr(capture, "_recheck_tools", _recheck)
    monkeypatch.setattr(capture, "_collect_host", _collect)
    monkeypatch.setattr(capture, "_run_checks", _checks)

    capture.capture(output, checkout)

    assert events == ["tools", "host", "checks", "tools", "host", "tools"]


def test_capture_never_executes_reproduction_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "inert-reproduction"
    marker = tmp_path / "recorded-argv-executed"

    def _malicious_reproduction(revision: str) -> dict[str, Any]:
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
                {
                    "id": "malicious",
                    "argv": [
                        sys.executable,
                        "-c",
                        f"open({str(marker)!r}, 'w').write('unexpected')",
                    ],
                    "expected_exit_code": 0,
                }
            ],
        }

    monkeypatch.setattr(capture, "g0_reproduction", _malicious_reproduction)
    monkeypatch.setattr(capture, "verify_g0_bundle", lambda _: [])
    capture.capture(output, checkout)

    assert output.is_dir()
    assert not marker.exists()
    assert capture._CAPTURE_CHECK_COMMANDS == (
        (
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/validate_schemas.py",
            "--all",
        ),
        (
            "uv",
            "run",
            "--frozen",
            "python",
            "scripts/generate_q8_fixtures.py",
            "--check",
        ),
        ("make", "rust-fixture-check"),
    )


def test_capture_environment_is_allowlisted_and_never_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/trusted/tools")
    monkeypatch.setenv("HOME", "/" + "private" + "/secret-home")
    tools = _fake_tools(tmp_path)
    environment = capture._capture_environment(REVISION, tools)

    assert environment == {
        "PATH": os.pathsep.join(
            (
                str(tools.rustup.path.parent),
                str(tools.uv.path.parent),
                str(tools.clang.path.parent),
                str(tools.sysctl.path.parent),
                str(tools.make.path.parent),
                os.defpath,
            )
        ),
        "LANG": "C",
        "LC_ALL": "C",
        "CARGO_NET_OFFLINE": "true",
        "DECODEFORGE_SOURCE_REVISION": REVISION,
        "UV_OFFLINE": "true",
    }


def test_capture_checks_use_fixed_argv_and_the_allowlisted_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    tools = _fake_tools(tmp_path)

    def _run(
        command: tuple[str, ...],
        _: Path,
        environment: dict[str, Any],
        **__: Any,
    ) -> Any:
        calls.append((command, environment))
        return capture._ProcessResult(0, b"", None)

    monkeypatch.setenv("PATH", "/trusted/tools")
    expected_environment = capture._capture_environment(REVISION, tools)
    monkeypatch.setenv("PATH", "/changed/after/resolution")
    monkeypatch.setattr(capture, "_run_bounded", _run)
    capture._run_checks(tmp_path, REVISION, tools)

    assert [command for command, _ in calls] == list(
        capture._capture_check_commands(tools)
    )
    assert all(environment == expected_environment for _, environment in calls)
    assert "/changed/after/resolution" not in expected_environment["PATH"]


def test_resolved_check_argv_survives_a_later_path_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = tmp_path / "initial-tools"
    later = tmp_path / "later-tools"
    for directory in (initial, later):
        directory.mkdir()
        for name in ("uv", "rustup", "clang", "sysctl", "make"):
            executable = directory / name
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(initial))
    tools = capture._resolve_capture_tools()
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def _run(
        command: tuple[str, ...],
        _: Path,
        environment: dict[str, Any],
        **__: Any,
    ) -> Any:
        calls.append((command, environment))
        return capture._ProcessResult(0, b"", None)

    monkeypatch.setenv("PATH", str(later))
    monkeypatch.setattr(capture, "_run_bounded", _run)
    capture._run_checks(tmp_path, REVISION, tools)

    assert [command[0] for command, _ in calls] == [
        str((initial / "uv").resolve()),
        str((initial / "uv").resolve()),
        str((initial / "make").resolve()),
    ]
    assert str(later) not in calls[0][1]["PATH"]
    assert calls[0][1]["PATH"].split(os.pathsep)[0] == str(
        (initial / "rustup").resolve().parent
    )


def test_capture_probes_the_exact_uv_nested_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _fake_tools(tmp_path)
    probes: list[tuple[str, ...]] = []
    responses = {
        (str(tools.sysctl.path), "-n", "machdep.cpu.brand_string"): "Apple M4\n",
        (str(tools.sysctl.path), "-n", "hw.physicalcpu"): "10\n",
        (str(tools.sysctl.path), "-n", "hw.logicalcpu"): "10\n",
        (str(tools.sysctl.path), "-n", "hw.optional.neon"): "1\n",
        (str(tools.clang.path), "--version"): f"{capture.G0_TOOLCHAINS['clang']}\n",
        (str(tools.uv.path), "--version"): f"uv {capture.G0_TOOLCHAINS['uv']}\n",
        (
            str(tools.rustup.path),
            "run",
            capture.G0_TOOLCHAINS["rust"],
            "rustc",
            "--version",
        ): f"rustc {capture.G0_TOOLCHAINS['rust']}\n",
        (str(tools.outer_python.path), "--version"): "Python 3.12.14\n",
        (
            str(tools.uv.path),
            "run",
            "--frozen",
            "python",
            "--version",
        ): "Python 3.12.14\n",
    }

    def _probe(command: tuple[str, ...], *_: Any) -> str:
        probes.append(command)
        return responses[command]

    monkeypatch.setattr(capture, "_probe_text", _probe)
    monkeypatch.setattr(capture.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(capture.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(capture.platform, "mac_ver", lambda: ("15.5", ("", "", ""), ""))
    monkeypatch.setattr(capture.platform, "release", lambda: "24.5.0")
    monkeypatch.setattr(
        capture.sys, "version_info", SimpleNamespace(major=3, minor=12, micro=14)
    )

    host = capture._collect_host(REVISION, tmp_path, tools)

    assert host == _host()
    assert (
        str(tools.uv.path),
        "run",
        "--frozen",
        "python",
        "--version",
    ) in probes
    assert all(command[0].startswith(str(tmp_path)) for command in probes)


def test_capture_rejects_a_producer_checkout_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer_bound_to_checkout = capture._producer_bound_to_checkout
    checkout, output_parent = _prepare_capture(tmp_path, monkeypatch)
    output = output_parent / "producer-mismatch"

    assert not producer_bound_to_checkout(checkout)
    monkeypatch.setattr(capture, "_producer_bound_to_checkout", lambda _: False)

    with pytest.raises(capture.CaptureError, match="producer"):
        capture.capture(output, checkout)
    _assert_no_partial_target(output_parent, output)


def test_pinned_executable_replacement_is_rejected(tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    executable.chmod(0o755)
    pinned = capture._pin_executable(executable)
    replacement = tmp_path / "replacement"
    replacement.write_text("#!/bin/sh\nexit 1\n", encoding="ascii")
    replacement.chmod(0o755)
    os.replace(replacement, executable)
    tools = capture._CaptureTools(
        uv=pinned,
        rustup=pinned,
        clang=pinned,
        sysctl=pinned,
        make=pinned,
        outer_python=pinned,
    )

    with pytest.raises(capture.CaptureError, match="executable changed"):
        capture._recheck_tools(tools)


def test_probe_stdout_is_incrementally_bounded(tmp_path: Path) -> None:
    environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    oversized = (
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 5000); sys.stdout.flush()",
    )
    endless = (
        sys.executable,
        "-c",
        (
            "import sys\nwhile True:\n"
            "    sys.stdout.write('x' * 512)\n"
            "    sys.stdout.flush()"
        ),
    )

    with pytest.raises(capture.CaptureError, match="output limit"):
        capture._probe_text(oversized, tmp_path, environment)
    with pytest.raises(capture.CaptureError, match="output limit"):
        capture._probe_text(endless, tmp_path, environment)


def test_bounded_runner_kills_timeout_and_orphaned_stdout_descendant(
    tmp_path: Path,
) -> None:
    environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    timed_out = capture._run_bounded(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        tmp_path,
        environment,
        timeout=0.05,
        output_limit=64,
        retain_stdout=False,
    )
    ready, lock_path, group_path, direct_parent = _orphan_stdout_commands(tmp_path)
    group_id: int | None = None
    death_proven = False
    try:
        orphaned = capture._run_bounded(
            direct_parent,
            tmp_path,
            environment,
            timeout=2.0,
            output_limit=64,
            retain_stdout=False,
        )

        assert timed_out.failure == "timeout"
        assert orphaned.failure == "timeout"
        assert ready.read_text(encoding="ascii") == "ready"
        group_id = _read_group_id(group_path)
        assert group_id is not None
        _wait_for_exclusive_lock(lock_path, timeout=5.0)
        death_proven = True
    finally:
        if not death_proven:
            if group_id is None:
                group_id = _read_group_id(group_path)
            _cleanup_process_group(group_id)


def _orphan_stdout_commands(
    tmp_path: Path,
) -> tuple[Path, Path, Path, tuple[str, ...]]:
    """Build a ready-gated descendant that retains stdout and an OS lock."""

    ready = tmp_path / "orphan-ready"
    lock_path = tmp_path / "orphan.lock"
    group_path = tmp_path / "orphan-group"
    child = (
        "import fcntl, os, pathlib, sys, time\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        f"lock_path = pathlib.Path({str(lock_path)!r})\n"
        "lock_file = lock_path.open('a+b')\n"
        "try:\n"
        "    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)\n"
        "    ready.write_text('ready', encoding='ascii')\n"
        "    sys.stdout.write('ready\\n')\n"
        "    sys.stdout.flush()\n"
        "    while True:\n"
        "        time.sleep(1.0)\n"
        "finally:\n"
        "    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)\n"
        "    lock_file.close()\n"
    )
    direct_parent = (
        sys.executable,
        "-c",
        "import os, pathlib, subprocess, sys, time\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        f"group = pathlib.Path({str(group_path)!r})\n"
        "group.write_text(str(os.getpgrp()), encoding='ascii')\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "deadline = time.monotonic() + 10.0\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "if not ready.exists():\n"
        "    child.kill()\n"
        "    child.wait()\n"
        "    raise SystemExit(3)\n",
    )
    return ready, lock_path, group_path, direct_parent


def _read_group_id(path: Path) -> int | None:
    try:
        group_id = int(path.read_text(encoding="ascii"), 10)
    except (OSError, ValueError):
        return None
    return group_id if group_id > 0 else None


def _cleanup_process_group(group_id: int | None) -> None:
    if group_id is None:
        return
    with suppress(ProcessLookupError):
        os.killpg(group_id, signal.SIGKILL)


def _try_exclusive_lock(path: Path) -> bool:
    with path.open("a+b") as descriptor:
        try:
            fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
        return True


def _wait_for_exclusive_lock(path: Path, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _try_exclusive_lock(path):
            return
        time.sleep(0.01)
    raise AssertionError("orphan descendant still owns its lock")


def test_bounded_runner_death_proof_rejects_sigstopped_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock proof cannot falsely pass when termination only SIGSTOPs."""

    environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    ready, lock_path, _, direct_parent = _orphan_stdout_commands(tmp_path)
    stopped_group_id: int | None = None

    def stop_process_group(process: Any) -> None:
        nonlocal stopped_group_id
        stopped_group_id = process.pid
        os.killpg(process.pid, signal.SIGSTOP)

    monkeypatch.setattr(capture, "_terminate_process_group", stop_process_group)
    try:
        result = capture._run_bounded(
            direct_parent,
            tmp_path,
            environment,
            timeout=2.0,
            output_limit=64,
            retain_stdout=False,
        )
        assert result.failure == "timeout"
        assert ready.read_text(encoding="ascii") == "ready"
        with pytest.raises(AssertionError, match="lock"):
            _wait_for_exclusive_lock(lock_path, timeout=0.5)
    finally:
        _cleanup_process_group(stopped_group_id)

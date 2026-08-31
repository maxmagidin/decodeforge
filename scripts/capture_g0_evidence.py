#!/usr/bin/env python3
"""Capture one privacy-safe, self-verified G0 correctness bundle."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import platform
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from decodeforge import g0_evidence as _g0_evidence_module
from decodeforge import g0_repository as _g0_repository_module
from decodeforge import q8 as _q8_module
from decodeforge.g0_evidence import (
    G0_ARTIFACTS,
    G0_COMPILER_VERSION,
    G0_HOST_ARCHITECTURE,
    G0_HOST_CPU_MODEL,
    G0_HOST_ID,
    G0_HOST_LOGICAL_CORES,
    G0_HOST_OS_NAME,
    G0_HOST_PHYSICAL_CORES,
    G0_HOST_ROLE,
    G0_NOT_APPLICABLE,
    G0_REQUIRED_CHECKS,
    G0_REQUIRED_TARGET_FEATURE,
    G0_TARGET_TRIPLE,
    G0_TOOLCHAINS,
    JsonObject,
    canonical_json_bytes,
    g0_bundle_id,
    g0_reproduction,
    verify_g0_bundle,
)
from decodeforge.g0_repository import (
    _checkout_binding,
    _checkout_clean,
    _checkout_head,
    _checkout_index_unflagged,
    _GitCheckout,
    verify_g0_repository_bundle,
)
from decodeforge.q8 import FORMAT, NUMERIC_MODE

_CAPTURE_CHECK_COMMANDS: Final[tuple[tuple[str, ...], ...]] = (
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
_CHECK_TIMEOUT_SECONDS: Final = 300.0
_CHECK_OUTPUT_LIMIT: Final = 64 * 1024
_PROBE_TIMEOUT_SECONDS: Final = 5.0
_PROBE_OUTPUT_LIMIT: Final = 4096
_FIXTURE_MANIFEST: Final = Path("tests/fixtures/v1/manifest.json")
_RENAME_EXCL: Final = 0x00000004
_REPORT_BYTES: Final = (
    b"# DecodeForge G0 correctness evidence\n\n"
    b"This bundle records correctness-check provenance for the closed Apple M4 "
    b"profile. It makes no performance, assembly, native-kernel, or completion "
    b"claim.\n"
)


class CaptureError(Exception):
    """The requested capture could not safely create a bundle."""


@dataclass(frozen=True)
class _ExecutableIdentity:
    """The replacement-sensitive identity of one resolved executable."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _PinnedExecutable:
    """One canonical executable path and the identity recorded at capture start."""

    path: Path
    identity: _ExecutableIdentity


@dataclass(frozen=True)
class _CaptureTools:
    """The fixed executable set used by the closed G0 capture profile."""

    uv: _PinnedExecutable
    rustup: _PinnedExecutable
    clang: _PinnedExecutable
    sysctl: _PinnedExecutable
    make: _PinnedExecutable
    outer_python: _PinnedExecutable


@dataclass(frozen=True)
class _ProcessResult:
    """One bounded command result without stderr content."""

    returncode: int | None
    stdout: bytes
    failure: str | None


def _pin_executable(path: Path) -> _PinnedExecutable:
    """Resolve one regular executable and retain its replacement-sensitive stat."""

    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except (OSError, RuntimeError) as error:
        raise CaptureError("A required G0 executable is unavailable.") from error
    if not stat.S_ISREG(metadata.st_mode) or not (
        metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    ):
        raise CaptureError("A required G0 executable is not a regular executable.")
    return _PinnedExecutable(
        path=resolved,
        identity=_ExecutableIdentity(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        ),
    )


def _resolve_executable(name: str) -> _PinnedExecutable:
    """Resolve exactly one command from the initial process path."""

    candidate = shutil.which(name)
    if candidate is None:
        raise CaptureError("A required G0 executable is unavailable.")
    return _pin_executable(Path(candidate))


def _resolve_capture_tools() -> _CaptureTools:
    """Resolve the complete fixed command set once before a capture begins."""

    return _CaptureTools(
        uv=_resolve_executable("uv"),
        rustup=_resolve_executable("rustup"),
        clang=_resolve_executable("clang"),
        sysctl=_resolve_executable("sysctl"),
        make=_resolve_executable("make"),
        outer_python=_pin_executable(Path(sys.executable)),
    )


def _recheck_tools(tools: _CaptureTools) -> None:
    """Reject an executable replacement or metadata change after resolution."""

    for executable in (
        tools.uv,
        tools.rustup,
        tools.clang,
        tools.sysctl,
        tools.make,
        tools.outer_python,
    ):
        try:
            current = _pin_executable(executable.path)
        except CaptureError as error:
            raise CaptureError(
                "A required G0 executable changed during capture."
            ) from error
        if current != executable:
            raise CaptureError("A required G0 executable changed during capture.")


def _producer_bound_to_checkout(checkout: Path) -> bool:
    """Require this script and its imported producer modules to come from checkout."""

    try:
        script_root = Path(__file__).resolve(strict=True).parent.parent
        package_root = (checkout / "python" / "decodeforge").resolve(strict=True)
        expected_modules = (
            (_g0_evidence_module, "g0_evidence.py"),
            (_g0_repository_module, "g0_repository.py"),
            (_q8_module, "q8.py"),
        )
        if script_root != checkout:
            return False
        for module, name in expected_modules:
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                return False
            if Path(module_file).resolve(strict=True) != package_root / name:
                return False
    except (OSError, RuntimeError):
        return False
    return True


def _capture_environment(revision: str, tools: _CaptureTools) -> dict[str, str]:
    """Return the fixed non-secret execution environment for capture checks."""

    directories: list[str] = []
    # ``make rust-fixture-check`` invokes ``rustup`` by name. Put the pinned
    # rustup directory first so no other resolved tool directory can shadow it.
    for executable in (tools.rustup, tools.uv, tools.clang, tools.sysctl, tools.make):
        directory = str(executable.path.parent)
        if directory not in directories:
            directories.append(directory)
    return {
        "PATH": os.pathsep.join((*directories, os.defpath)),
        "LANG": "C",
        "LC_ALL": "C",
        "CARGO_NET_OFFLINE": "true",
        "DECODEFORGE_SOURCE_REVISION": revision,
        "UV_OFFLINE": "true",
    }


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill and reap a command group even when its direct parent already exited."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
    with suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=1)


def _run_bounded(
    command: tuple[str, ...],
    checkout: Path,
    environment: dict[str, str],
    *,
    timeout: float,
    output_limit: int,
    retain_stdout: bool,
) -> _ProcessResult:
    """Run one literal argv with bounded incremental stdout and no stderr data."""

    try:
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        return _ProcessResult(None, b"", "execution")

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    chunks: list[bytes] = []
    output_size = 0
    deadline = time.monotonic() + timeout
    stdout_fd = process.stdout.fileno()
    try:
        selector.register(stdout_fd, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                return _ProcessResult(None, b"", "timeout")
            events = selector.select(remaining)
            if not events:
                # A direct parent can exit while a descendant retains stdout.
                # That descendant is part of this fixed command's session and
                # must not survive the timeout boundary.
                _terminate_process_group(process)
                return _ProcessResult(None, b"", "timeout")
            for _, _ in events:
                chunk = os.read(
                    stdout_fd,
                    min(8192, output_limit + 1 - output_size),
                )
                if not chunk:
                    selector.unregister(stdout_fd)
                    continue
                output_size += len(chunk)
                if output_size > output_limit:
                    _terminate_process_group(process)
                    return _ProcessResult(None, b"", "output-limit")
                if retain_stdout:
                    chunks.append(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            return _ProcessResult(None, b"", "timeout")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return _ProcessResult(None, b"", "timeout")
        # Even a successful direct parent may have left a descendant in its
        # session. Fixed capture commands have no background-process contract.
        _terminate_process_group(process)
        if returncode != 0:
            return _ProcessResult(returncode, b"", "returncode")
        return _ProcessResult(returncode, b"".join(chunks), None)
    except OSError:
        _terminate_process_group(process)
        return _ProcessResult(None, b"", "execution")
    finally:
        selector.close()
        process.stdout.close()
        if process.poll() is None:
            _terminate_process_group(process)


def _probe_text(
    command: tuple[str, ...], checkout: Path, environment: dict[str, str]
) -> str:
    """Read one fixed host/tool probe with a small stdout and time budget."""

    result = _run_bounded(
        command,
        checkout,
        environment,
        timeout=_PROBE_TIMEOUT_SECONDS,
        output_limit=_PROBE_OUTPUT_LIMIT,
        retain_stdout=True,
    )
    if result.failure == "output-limit":
        raise CaptureError("A required G0 host probe exceeded its output limit.")
    if result.failure is not None:
        raise CaptureError("A required G0 host probe did not finish safely.")
    if result.returncode != 0:
        raise CaptureError("A required G0 host probe did not complete successfully.")
    try:
        return result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise CaptureError("A required G0 host probe returned invalid text.") from error


def _first_line(value: str) -> str:
    """Return a nonempty first probe line without retaining later output."""

    first, _, _ = value.partition("\n")
    if not first:
        raise CaptureError("A required G0 host probe returned no value.")
    return first


def _version_from_probe(value: str, program: str) -> str:
    """Read the normalized version token from one fixed tool response."""

    words = _first_line(value).split()
    if len(words) < 2 or words[0] != program:
        raise CaptureError("A required G0 tool probe returned an unexpected value.")
    return words[1]


def _required_text(value: str, expected: str) -> None:
    """Require a closed profile value without reporting host-specific output."""

    if value != expected:
        raise CaptureError("The requested G0 Apple-M4 profile is unavailable.")


def _required_decimal(value: str, expected: int) -> None:
    """Require one closed numeric host property."""

    if not value.isdecimal() or int(value) != expected:
        raise CaptureError("The requested G0 Apple-M4 profile is unavailable.")


def _collect_host(revision: str, checkout: Path, tools: _CaptureTools) -> JsonObject:
    """Collect only the fixed, non-identifying Apple-M4 G0 host fields."""

    environment = _capture_environment(revision, tools)
    if platform.system() != G0_HOST_OS_NAME:
        raise CaptureError("The requested G0 Apple-M4 profile is unavailable.")
    machine = platform.machine().lower()
    if machine != "arm64":
        raise CaptureError("The requested G0 Apple-M4 profile is unavailable.")
    mac_version = platform.mac_ver()[0]
    kernel = platform.release()
    if not mac_version or not kernel:
        raise CaptureError("The requested G0 Apple-M4 profile is unavailable.")

    _required_text(
        _first_line(
            _probe_text(
                (str(tools.sysctl.path), "-n", "machdep.cpu.brand_string"),
                checkout,
                environment,
            )
        ),
        G0_HOST_CPU_MODEL,
    )
    _required_decimal(
        _first_line(
            _probe_text(
                (str(tools.sysctl.path), "-n", "hw.physicalcpu"),
                checkout,
                environment,
            )
        ),
        G0_HOST_PHYSICAL_CORES,
    )
    _required_decimal(
        _first_line(
            _probe_text(
                (str(tools.sysctl.path), "-n", "hw.logicalcpu"),
                checkout,
                environment,
            )
        ),
        G0_HOST_LOGICAL_CORES,
    )
    _required_text(
        _first_line(
            _probe_text(
                (str(tools.sysctl.path), "-n", "hw.optional.neon"),
                checkout,
                environment,
            )
        ),
        "1",
    )
    _required_text(
        _first_line(
            _probe_text((str(tools.clang.path), "--version"), checkout, environment)
        ),
        G0_TOOLCHAINS["clang"],
    )
    _required_text(
        _version_from_probe(
            _probe_text((str(tools.uv.path), "--version"), checkout, environment),
            "uv",
        ),
        G0_TOOLCHAINS["uv"],
    )
    _required_text(
        _version_from_probe(
            _probe_text(
                (
                    str(tools.rustup.path),
                    "run",
                    G0_TOOLCHAINS["rust"],
                    "rustc",
                    "--version",
                ),
                checkout,
                environment,
            ),
            "rustc",
        ),
        G0_TOOLCHAINS["rust"],
    )
    _required_text(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        G0_TOOLCHAINS["python"],
    )
    _required_text(
        _version_from_probe(
            _probe_text(
                (str(tools.outer_python.path), "--version"), checkout, environment
            ),
            "Python",
        ),
        G0_TOOLCHAINS["python"],
    )
    _required_text(
        _version_from_probe(
            _probe_text(
                (
                    str(tools.uv.path),
                    "run",
                    "--frozen",
                    "python",
                    "--version",
                ),
                checkout,
                environment,
            ),
            "Python",
        ),
        G0_TOOLCHAINS["python"],
    )

    return {
        "schema_version": 1,
        "host_id": G0_HOST_ID,
        "role": G0_HOST_ROLE,
        "architecture": G0_HOST_ARCHITECTURE,
        "cpu": {
            "model": G0_HOST_CPU_MODEL,
            "physical_cores": G0_HOST_PHYSICAL_CORES,
            "logical_cores": G0_HOST_LOGICAL_CORES,
            "features": [G0_REQUIRED_TARGET_FEATURE],
        },
        "os": {"name": G0_HOST_OS_NAME, "version": mac_version, "kernel": kernel},
        "toolchains": dict(G0_TOOLCHAINS),
        "source": {"revision": revision, "dirty": False},
    }


def _clean_revision(checkout: _GitCheckout) -> str:
    """Return one stable, full, clean checkout revision for capture."""

    revision = _checkout_head(checkout)
    if revision is None:
        raise CaptureError("The explicit checkout does not expose one full commit.")
    if _checkout_clean(checkout) is not True:
        raise CaptureError("The explicit checkout is not clean for G0 capture.")
    if _checkout_index_unflagged(checkout) is not True:
        raise CaptureError("The explicit checkout index is not safe for G0 capture.")
    if _checkout_head(checkout) != revision:
        raise CaptureError("The explicit checkout changed during G0 capture.")
    return revision


def _capture_check_commands(tools: _CaptureTools) -> tuple[tuple[str, ...], ...]:
    """Build the fixed check argv using only capture-start executable paths."""

    return (
        (str(tools.uv.path), *_CAPTURE_CHECK_COMMANDS[0][1:]),
        (str(tools.uv.path), *_CAPTURE_CHECK_COMMANDS[1][1:]),
        (str(tools.make.path), *_CAPTURE_CHECK_COMMANDS[2][1:]),
    )


def _run_checks(checkout: Path, revision: str, tools: _CaptureTools) -> None:
    """Run only the fixed G0 checks; this never reads recorded argv metadata."""

    environment = _capture_environment(revision, tools)
    for command in _capture_check_commands(tools):
        result = _run_bounded(
            command,
            checkout,
            environment,
            timeout=_CHECK_TIMEOUT_SECONDS,
            output_limit=_CHECK_OUTPUT_LIMIT,
            retain_stdout=False,
        )
        if result.failure is not None or result.returncode != 0:
            raise CaptureError("A required G0 correctness check did not pass.")


def _output_target(output: Path, checkout: Path) -> Path:
    """Resolve an explicit pre-commit output sibling outside the checkout."""

    if output.name in {"", ".", ".."}:
        raise CaptureError("The requested output target is not a bundle path.")
    try:
        parent = output.parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CaptureError("The requested output parent is unavailable.") from error
    if not parent.is_dir():
        raise CaptureError("The requested output parent is unavailable.")
    target = parent / output.name
    if target.is_relative_to(checkout):
        raise CaptureError("Pre-commit G0 output must be outside the checkout.")
    return target


def _target_exists(target: Path) -> bool:
    """Treat an existing or dangling final component as a no-overwrite target."""

    try:
        return target.exists() or target.is_symlink()
    except OSError as error:
        raise CaptureError(
            "The requested output target could not be inspected."
        ) from error


def _write_new(path: Path, content: bytes) -> None:
    """Create and durably write one staging artifact without overwrite."""

    with path.open("xb") as destination:
        destination.write(content)
        destination.flush()
        os.fsync(destination.fileno())


def _fsync_directory(path: Path) -> None:
    """Synchronize one created staging or output directory."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _staged_contents(
    revision: str, fixture_bytes: bytes, host: JsonObject
) -> dict[str, bytes]:
    """Build the exact four-file G0 inventory from bounded capture values."""

    contents = {
        "fixture-manifest.json": fixture_bytes,
        "host.json": canonical_json_bytes(host) + b"\n",
        "report.md": _REPORT_BYTES,
    }
    manifest: JsonObject = {
        "schema_version": 1,
        "milestone": "g0",
        "bundle_class": "correctness",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": {
            "revision": revision,
            "dirty": False,
            "format": FORMAT,
            "numeric_mode": NUMERIC_MODE,
            "compiler_version": G0_COMPILER_VERSION,
            "generated_abi": 1,
            "runtime_abi": 1,
        },
        "target": {
            "triple": G0_TARGET_TRIPLE,
            "features": [G0_REQUIRED_TARGET_FEATURE],
        },
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
    contents["run-manifest.json"] = canonical_json_bytes(manifest)
    return contents


def _write_staging(staging: Path, contents: dict[str, bytes]) -> None:
    """Create the closed G0 inventory in one private sibling staging directory."""

    for name in (
        "fixture-manifest.json",
        "host.json",
        "report.md",
        "run-manifest.json",
    ):
        _write_new(staging / name, contents[name])
    _fsync_directory(staging)


def _install_no_overwrite(staging: Path, target: Path) -> None:
    """Atomically install the sibling staging directory without replacement."""

    if platform.system() != "Darwin":
        # Real capture cannot reach this branch: the closed host probe already
        # rejects non-Darwin systems. Keep injected tests portable.
        if _target_exists(target):
            raise CaptureError("The requested output target already exists.")
        os.rename(staging, target)
        return
    try:
        renamex = ctypes.CDLL(None, use_errno=True).renamex_np
    except AttributeError as error:
        raise CaptureError("The G0 output could not be installed safely.") from error
    renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    renamex.restype = ctypes.c_int
    if renamex(os.fsencode(staging), os.fsencode(target), _RENAME_EXCL) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CaptureError("The requested output target already exists.")
    raise OSError(error_number, "renamex_np")


def capture(output: Path, checkout: Path) -> None:
    """Capture, self-verify, then install one new G0 evidence directory."""

    bound_checkout = _checkout_binding(checkout)
    if bound_checkout is None:
        raise CaptureError("The explicit checkout is not one safe Git worktree.")
    if not _producer_bound_to_checkout(bound_checkout.work_tree):
        raise CaptureError("The G0 capture producer is not bound to the checkout.")
    tools = _resolve_capture_tools()
    target = _output_target(output, bound_checkout.work_tree)
    if _target_exists(target):
        raise CaptureError("The requested output target already exists.")
    revision = _clean_revision(bound_checkout)
    _recheck_tools(tools)
    host = _collect_host(revision, bound_checkout.work_tree, tools)
    _run_checks(bound_checkout.work_tree, revision, tools)
    _recheck_tools(tools)
    if _collect_host(revision, bound_checkout.work_tree, tools) != host:
        raise CaptureError("The G0 host profile changed during capture.")
    _recheck_tools(tools)
    try:
        fixture_bytes = (bound_checkout.work_tree / _FIXTURE_MANIFEST).read_bytes()
    except OSError as error:
        raise CaptureError(
            "The checked-in fixture manifest could not be read."
        ) from error

    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.capture-", dir=target.parent
        ) as staging_name:
            staging = Path(staging_name)
            _write_staging(staging, _staged_contents(revision, fixture_bytes, host))
            if verify_g0_bundle(staging) != []:
                raise CaptureError(
                    "The staged G0 bundle did not pass portable verification."
                )
            if verify_g0_repository_bundle(staging, bound_checkout.work_tree) != []:
                raise CaptureError(
                    "The staged G0 bundle did not pass repository verification."
                )
            if _clean_revision(bound_checkout) != revision:
                raise CaptureError("The explicit checkout changed during G0 capture.")
            if _target_exists(target):
                raise CaptureError("The requested output target already exists.")
            _install_no_overwrite(staging, target)
            _fsync_directory(target.parent)
    except OSError as error:
        raise CaptureError("The G0 output could not be installed safely.") from error


def main(argv: list[str] | None = None) -> int:
    """Run the explicit-path G0 capture command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        capture(arguments.output, arguments.checkout)
    except CaptureError as error:
        print(f"g0-evidence-capture: {error}", file=sys.stderr)
        return 1
    print("g0-evidence-capture: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

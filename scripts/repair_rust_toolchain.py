#!/usr/bin/env python3
"""Repair one missing Rust 1.98 macOS LLVM loader link.

The command is deliberately conservative: it is a dry run unless ``--apply``
is supplied, and the only permitted mutation is creation of the exact missing
``libLLVM.dylib`` symlink inside the rustup-managed host sysroot.  It never
sets a ``DYLD_*`` variable, removes an existing path, rewrites a Mach-O file,
or installs another toolchain.

Exit status is 0 when the normal-environment preflight passes (or on a
non-Darwin host), 1 when the preflight is still failing or a dry run reports a
repairable missing link, and 2 when trusted toolchain identity/path checks or
the guarded repair checks fail.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

Run = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _load_checker() -> object:
    spec = importlib.util.spec_from_file_location(
        "decodeforge_rust_toolchain_checker",
        Path(__file__).with_name("check_rust_toolchain.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Rust toolchain preflight")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            env=None,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(arguments, 127, "", str(error))


def _single_line(output: str, label: str) -> str:
    lines = output.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise ValueError(f"{label} returned an invalid value")
    return lines[0].strip()


def _identity(rust_version: str, run: Run) -> tuple[Path, str, Path, Path, Path, Path]:
    rustc_result = run(["rustup", "which", "--toolchain", rust_version, "rustc"])
    if rustc_result.returncode != 0:
        raise ValueError("rustup cannot resolve the pinned rustc")
    rustc = Path(_single_line(rustc_result.stdout, "rustup which"))
    if not rustc.is_absolute() or rustc.is_symlink() or not rustc.is_file():
        raise ValueError("rustup resolved rustc is not a trusted regular file")
    if rustc.name != "rustc" or rustc.parent.name != "bin":
        raise ValueError("rustup resolved rustc has an unexpected path")

    toolchain = rustc.parent.parent
    if toolchain.is_symlink() or not toolchain.is_dir():
        raise ValueError("rustup resolved an invalid toolchain root")
    try:
        toolchain_root = toolchain.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"cannot resolve the trusted toolchain root: {error}"
        ) from error
    sysroot_result = run([os.fspath(rustc), "--print", "sysroot"])
    if sysroot_result.returncode != 0:
        raise ValueError("pinned rustc cannot report its sysroot")
    reported_sysroot = Path(_single_line(sysroot_result.stdout, "rustc sysroot"))
    if (
        not reported_sysroot.is_absolute()
        or reported_sysroot.resolve() != toolchain_root
    ):
        raise ValueError("rustc sysroot does not match rustup's resolved toolchain")

    version_result = run([os.fspath(rustc), "-vV"])
    if version_result.returncode != 0:
        raise ValueError("pinned rustc cannot report its identity")
    release = next(
        (
            line.removeprefix("release: ").strip()
            for line in version_result.stdout.splitlines()
            if line.startswith("release: ")
        ),
        "",
    )
    host = next(
        (
            line.removeprefix("host: ").strip()
            for line in version_result.stdout.splitlines()
            if line.startswith("host: ")
        ),
        "",
    )
    if release != rust_version:
        raise ValueError(
            f"rustc release {release or '<missing>'!r} does not match {rust_version!r}"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("rustc reported an invalid host triple")

    objcopy = toolchain / "lib" / "rustlib" / host / "bin" / "rust-objcopy"
    llvm = toolchain / "lib" / "libLLVM.dylib"
    link_dir = toolchain / "lib" / "rustlib" / host / "lib"
    link = link_dir / "libLLVM.dylib"
    directories = (
        (toolchain / "lib", "Rust sysroot lib directory"),
        (toolchain / "lib" / "rustlib", "Rust rustlib directory"),
        (toolchain / "lib" / "rustlib" / host, "Rust host directory"),
        (objcopy.parent, "Rust host bin directory"),
        (link_dir, "Rust host library directory"),
    )
    for directory, label in directories:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"trusted {label} is missing or unsafe")
        try:
            resolved = directory.resolve(strict=True)
            resolved.relative_to(toolchain_root)
        except (OSError, ValueError) as error:
            raise ValueError(f"trusted {label} escapes the toolchain root") from error
    for path, label in ((objcopy, "rust-objcopy"), (llvm, "libLLVM.dylib")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trusted {label} path is missing or not a regular file")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(toolchain_root)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"trusted {label} path escapes the toolchain root"
            ) from error
    return rustc, host, toolchain, objcopy, llvm, link


def _link_state(link: Path, llvm: Path) -> str:
    """Return ``missing``, ``correct``, or ``wrong`` without unlinking anything."""
    if not os.path.lexists(link):
        return "missing"
    if not link.is_symlink():
        return "wrong"
    try:
        target = os.readlink(link)
    except OSError:
        return "wrong"
    return "correct" if target == os.fspath(llvm) else "wrong"


def repair_toolchain(
    rust_version: str,
    system: str,
    apply: bool = False,
    run: Run = _run,
) -> int:
    """Check and, only with ``apply=True``, create the guarded missing link."""
    if system != "Darwin":
        print("rust-toolchain-repair: not applicable on this host")
        return 0

    try:
        _rustc, _host, _toolchain, _objcopy, llvm, link = _identity(rust_version, run)
    except (OSError, ValueError) as error:
        print(
            f"rust-toolchain-repair: unable to validate trusted toolchain: {error}",
            file=sys.stderr,
        )
        return 2

    checker = _load_checker()
    preflight = checker.check_toolchain(rust_version, system, run)  # type: ignore[attr-defined]
    if preflight == 0:
        print("rust-toolchain-repair: no repair needed")
        return 0
    if preflight != 1:
        print(
            "rust-toolchain-repair: strict preflight failed before a repairable "
            f"loader failure was established (status {preflight})",
            file=sys.stderr,
        )
        return 2

    state = _link_state(link, llvm)
    if state == "wrong":
        print(
            "rust-toolchain-repair: refusing to replace an existing non-matching "
            f"{link}",
            file=sys.stderr,
        )
        return 2
    if state == "missing" and not apply:
        print(
            "rust-toolchain-repair: dry run; would create the missing exact "
            f"symlink {link} -> {llvm}",
            file=sys.stderr,
        )
        return 1
    if state == "missing":
        try:
            os.symlink(os.fspath(llvm), os.fspath(link))
        except FileExistsError:
            print(
                f"rust-toolchain-repair: refusing a concurrent replacement of {link}",
                file=sys.stderr,
            )
            return 2
        except OSError as error:
            print(
                f"rust-toolchain-repair: cannot create guarded link: {error}",
                file=sys.stderr,
            )
            return 2
        if _link_state(link, llvm) != "correct":
            print(
                "rust-toolchain-repair: created link failed exact verification",
                file=sys.stderr,
            )
            return 2
        print(f"rust-toolchain-repair: created {link} -> {llvm}")

    after = checker.check_toolchain(rust_version, system, run)  # type: ignore[attr-defined]
    if after != 0:
        print(
            "rust-toolchain-repair: normal-environment preflight still fails; "
            "no loader environment workaround was used",
            file=sys.stderr,
        )
        return 1
    print("rust-toolchain-repair: repaired and normal-environment preflight passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-version", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create the one guarded missing sysroot symlink (default: dry run)",
    )
    arguments = parser.parse_args()
    return repair_toolchain(arguments.rust_version, platform.system(), arguments.apply)


if __name__ == "__main__":
    raise SystemExit(main())

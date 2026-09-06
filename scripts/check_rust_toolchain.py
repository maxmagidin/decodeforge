#!/usr/bin/env python3
"""Preflight the pinned macOS Rust objcopy/LLVM dynamic-library pairing."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

Run = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(
    arguments: list[str], environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return subprocess.CompletedProcess(arguments, 127, "", str(error))


def check_toolchain(
    rust_version: str,
    system: str,
    run: Run = _run,
) -> int:
    """Check the toolchain without changing the caller's loader environment."""
    if system != "Darwin":
        print("rust-toolchain: strip preflight not applicable on this host")
        return 0

    rustc = run(["rustup", "which", "--toolchain", rust_version, "rustc"])
    if rustc.returncode != 0:
        print("rust-toolchain: pinned rustc is unavailable", file=sys.stderr)
        return 2
    rustc_path = Path(rustc.stdout.strip())
    if not rustc_path.is_file():
        print("rust-toolchain: pinned rustc path is missing", file=sys.stderr)
        return 2
    toolchain = rustc_path.parent.parent
    host = run([os.fspath(rustc_path), "-vV"])
    if host.returncode != 0:
        print("rust-toolchain: unable to query pinned rustc", file=sys.stderr)
        return 2
    host_triple = next(
        (
            line.removeprefix("host: ").strip()
            for line in host.stdout.splitlines()
            if line.startswith("host: ")
        ),
        "",
    )
    if not host_triple:
        print("rust-toolchain: rustc did not report a host triple", file=sys.stderr)
        return 2
    objcopy = toolchain / "lib" / "rustlib" / host_triple / "bin" / "rust-objcopy"
    llvm = toolchain / "lib" / "libLLVM.dylib"
    if not objcopy.is_file() or not llvm.is_file():
        print(
            "rust-toolchain: pinned rust-objcopy or libLLVM.dylib is missing",
            file=sys.stderr,
        )
        return 2

    probe = run([os.fspath(objcopy), "--version"])
    if probe.returncode == 0:
        print("rust-toolchain: rust-objcopy/libLLVM load check passed")
        return 0
    print(
        "rust-toolchain: rust-objcopy cannot load pinned libLLVM in the normal "
        "environment; repair the toolchain rpath or installation",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-version", required=True)
    arguments = parser.parse_args()
    return check_toolchain(arguments.rust_version, platform.system())


if __name__ == "__main__":
    raise SystemExit(main())

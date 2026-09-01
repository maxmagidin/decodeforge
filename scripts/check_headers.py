#!/usr/bin/env python3
"""Compile the foundation C11/C++17 ABI smoke fixtures with Clang."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ROOT / "include"
ABI_HEADER = INCLUDE / "decodeforge" / "abi_v1.h"
RUNTIME_HEADER = INCLUDE / "decodeforge" / "runtime_v1.h"
SMOKE_SOURCES = (
    (ROOT / "tests" / "native" / "abi_v1_c11.c", "clang", "c11"),
    (ROOT / "tests" / "native" / "abi_v1_cpp17.cpp", "clang++", "c++17"),
)
UNIMPLEMENTED_RUNTIME_NAMES = ("df_runtime_preload_v1",)


def _path_markers() -> tuple[str, ...]:
    return ("/" + "Users" + "/", "/" + "home" + "/", "/" + "private" + "/")


def _run_compile(
    source: Path, compiler_name: str, standard: str, output: Path
) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise FileNotFoundError(compiler_name)
    command = [
        compiler,
        f"-std={standard}",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-I",
        str(INCLUDE),
        "-c",
        str(source),
        "-o",
        str(output),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def check() -> list[str]:
    errors: list[str] = []
    if not ABI_HEADER.is_file():
        errors.append("missing include/decodeforge/abi_v1.h")
        return errors
    if not RUNTIME_HEADER.is_file():
        errors.append("missing include/decodeforge/runtime_v1.h")
        return errors

    abi_text = ABI_HEADER.read_text(encoding="utf-8")
    runtime_text = RUNTIME_HEADER.read_text(encoding="utf-8")
    if not re.search(r"typedef\s+struct\s+df_call_v1\s*\{", abi_text):
        errors.append("abi_v1.h does not define df_call_v1")
    if "sizeof(df_call_v1) == 48" not in abi_text:
        errors.append("abi_v1.h does not assert the 48-byte df_call_v1 layout")
    for symbol in ("df_abi_version", "df_artifact_id", "df_run_v1"):
        if symbol not in abi_text:
            errors.append(f"abi_v1.h is missing declaration {symbol}")
    if "DF_RUNTIME_ABI_VERSION_V1" not in runtime_text or not re.search(
        r"typedef\s+uint64_t\s+df_runtime_handle_v1\s*;", runtime_text
    ):
        errors.append("runtime_v1.h must reserve its version and handle type")
    for name in UNIMPLEMENTED_RUNTIME_NAMES:
        if name in runtime_text:
            errors.append(f"runtime_v1.h freezes an unimplemented operation: {name}")
    for symbol in (
        "df_runtime_bridge_abi_version_v1",
        "df_runtime_create_neon_v1",
        "df_runtime_run_v1",
        "df_runtime_get_descriptor_v1",
        "df_runtime_destroy_v1",
        "df_runtime_last_error_v1",
    ):
        if symbol not in runtime_text:
            errors.append(f"runtime_v1.h is missing declaration {symbol}")
    for path in (ABI_HEADER, RUNTIME_HEADER):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in _path_markers()):
            errors.append(f"absolute developer path in {path.relative_to(ROOT)}")

    missing_compilers = sorted(
        {compiler for _, compiler, _ in SMOKE_SOURCES if shutil.which(compiler) is None}
    )
    if missing_compilers:
        errors.append("missing Clang compiler(s): " + ", ".join(missing_compilers))
        return errors

    with tempfile.TemporaryDirectory(prefix="decodeforge-header-check-") as temporary:
        temporary_path = Path(temporary)
        for source, compiler, standard in SMOKE_SOURCES:
            if not source.is_file():
                errors.append(
                    f"missing native smoke source: {source.relative_to(ROOT)}"
                )
                continue
            output = temporary_path / (source.stem + ".o")
            try:
                result = _run_compile(source, compiler, standard, output)
            except OSError as error:
                errors.append(f"could not invoke {compiler}: {error}")
                continue
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                errors.append(
                    f"{source.relative_to(ROOT)} failed Clang {standard}: {detail}"
                )
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"headers-check: {error}", file=sys.stderr)
        return 1
    print("headers-check: ok (C11 and C++17 with Clang -Werror)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

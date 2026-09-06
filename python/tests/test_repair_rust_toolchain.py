"""Tests for the opt-in Rust toolchain link repair."""

from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "decodeforge_test_repair_rust_toolchain",
    Path(__file__).parents[2] / "scripts" / "repair_rust_toolchain.py",
)
assert _SPEC is not None and _SPEC.loader is not None
tooling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tooling)


def _toolchain(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "toolchain"
    rustc = root / "bin" / "rustc"
    objcopy = root / "lib" / "rustlib" / "aarch64-apple-darwin" / "bin" / "rust-objcopy"
    link_dir = objcopy.parent.parent / "lib"
    rustc.parent.mkdir(parents=True)
    objcopy.parent.mkdir(parents=True)
    link_dir.mkdir(parents=True)
    rustc.touch()
    objcopy.touch()
    llvm = root / "lib" / "libLLVM.dylib"
    llvm.touch()
    return rustc, objcopy, llvm


def _runner(
    rustc: Path,
    objcopy: Path,
    llvm: Path,
    probe_status: int = 0,
    fail_after_link: bool = False,
) -> Callable[[list[str]], subprocess.CompletedProcess[str]]:
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if arguments[:2] == ["rustup", "which"]:
            return subprocess.CompletedProcess(arguments, 0, f"{rustc}\n", "")
        if arguments == [str(rustc), "--print", "sysroot"]:
            return subprocess.CompletedProcess(
                arguments, 0, f"{rustc.parent.parent}\n", ""
            )
        if arguments == [str(rustc), "-vV"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                "rustc 1.98.0 (test)\nrelease: 1.98.0\nhost: aarch64-apple-darwin\n",
                "",
            )
        assert arguments == [str(objcopy), "--version"]
        status = (
            probe_status
            if fail_after_link and link.is_symlink()
            else (
                0
                if link.is_symlink() and link.resolve() == llvm.resolve()
                else probe_status
            )
        )
        return subprocess.CompletedProcess(
            arguments, status, "llvm-objcopy\n", "loader error"
        )

    return run


def test_dry_run_does_not_mutate_missing_link(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", run=_runner(rustc, objcopy, llvm, 1)
        )
        == 1
    )
    assert not link.exists() and not link.is_symlink()
    assert "dry run" in capsys.readouterr().err


def test_apply_creates_only_exact_missing_link(tmp_path: Path) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", apply=True, run=_runner(rustc, objcopy, llvm, 1)
        )
        == 0
    )
    assert link.is_symlink()
    assert link.readlink() == llvm


def test_existing_correct_link_is_idempotent(tmp_path: Path) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"
    link.symlink_to(llvm)
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", apply=True, run=_runner(rustc, objcopy, llvm, 1)
        )
        == 0
    )
    assert link.readlink() == llvm


@pytest.mark.parametrize("entry", ["wrong-link", "regular-file"])
def test_refuses_wrong_existing_entry(tmp_path: Path, entry: str) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"
    if entry == "wrong-link":
        link.symlink_to(tmp_path / "other.dylib")
    else:
        link.touch()
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", apply=True, run=_runner(rustc, objcopy, llvm, 1)
        )
        == 2
    )
    assert link.is_symlink() == (entry == "wrong-link")


def test_does_not_remove_link_when_probe_still_fails(tmp_path: Path) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"
    run = _runner(rustc, objcopy, llvm, 134, fail_after_link=True)
    assert tooling.repair_toolchain("1.98.0", "Darwin", apply=True, run=run) == 1
    assert link.is_symlink()


def test_refuses_release_mismatch(tmp_path: Path) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    base = _runner(rustc, objcopy, llvm, 1)

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        result = base(arguments)
        if arguments == [str(rustc), "-vV"]:
            return subprocess.CompletedProcess(
                arguments, 0, "release: 1.97.0\nhost: aarch64-apple-darwin\n", ""
            )
        return result

    assert tooling.repair_toolchain("1.98.0", "Darwin", apply=True, run=run) == 2


def test_refuses_symlinked_llvm_source(tmp_path: Path) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    real_llvm = tmp_path / "real-libLLVM.dylib"
    real_llvm.touch()
    llvm.unlink()
    llvm.symlink_to(real_llvm)
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", apply=True, run=_runner(rustc, objcopy, llvm, 1)
        )
        == 2
    )


def test_refuses_symlinked_host_library_directory(tmp_path: Path) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    link_dir = objcopy.parent.parent / "lib"
    real_dir = tmp_path / "real-host-lib"
    real_dir.mkdir()
    link_dir.rmdir()
    link_dir.symlink_to(real_dir)
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", apply=True, run=_runner(rustc, objcopy, llvm, 1)
        )
        == 2
    )


@pytest.mark.parametrize("intermediate", ["lib", "rustlib", "host"])
def test_refuses_symlinked_intermediate_escape(
    tmp_path: Path, intermediate: str
) -> None:
    rustc, objcopy, llvm = _toolchain(tmp_path)
    root = rustc.parent.parent
    if intermediate == "lib":
        path = root / "lib"
    elif intermediate == "rustlib":
        path = root / "lib" / "rustlib"
    else:
        path = root / "lib" / "rustlib" / "aarch64-apple-darwin"
    external = tmp_path / f"external-{intermediate}"
    path.rename(external)
    path.symlink_to(external)
    link = objcopy.parent.parent / "lib" / "libLLVM.dylib"
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Darwin", apply=True, run=_runner(rustc, objcopy, llvm, 1)
        )
        == 2
    )
    assert not link.exists() and not link.is_symlink()


def test_filesystem_error_is_a_clear_failure() -> None:
    def run(_arguments: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError("simulated rustup filesystem failure")

    assert tooling.repair_toolchain("1.98.0", "Darwin", apply=True, run=run) == 2


def test_skips_unsupported_host(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        tooling.repair_toolchain(
            "1.98.0", "Linux", apply=True, run=lambda _args: pytest.fail()
        )
        == 0
    )
    assert "not applicable" in capsys.readouterr().out

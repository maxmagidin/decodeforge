"""Tests for the pinned Rust objcopy/LLVM preflight."""

from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "decodeforge_test_rust_toolchain",
    Path(__file__).parents[2] / "scripts" / "check_rust_toolchain.py",
)
assert _SPEC is not None and _SPEC.loader is not None
tooling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tooling)


def _toolchain(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "toolchain"
    rustc = root / "bin" / "rustc"
    objcopy = root / "lib" / "rustlib" / "aarch64-apple-darwin" / "bin" / "rust-objcopy"
    rustc.parent.mkdir(parents=True)
    objcopy.parent.mkdir(parents=True)
    rustc.touch()
    objcopy.touch()
    (root / "lib" / "libLLVM.dylib").touch()
    return rustc, objcopy


def _runner(
    rustc: Path, objcopy: Path, objcopy_status: int = 0
) -> tuple[Callable[[list[str]], subprocess.CompletedProcess[str]], list[list[str]]]:
    calls: list[list[str]] = []

    def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[:2] == ["rustup", "which"]:
            return subprocess.CompletedProcess(arguments, 0, f"{rustc}\n", "")
        if arguments == [str(rustc), "-vV"]:
            return subprocess.CompletedProcess(
                arguments, 0, "host: aarch64-apple-darwin\n", ""
            )
        assert arguments == [str(objcopy), "--version"]
        return subprocess.CompletedProcess(
            arguments, objcopy_status, "", "loader error"
        )

    return run, calls


def test_preflight_passes_without_loader_environment_injection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rustc, objcopy = _toolchain(tmp_path)
    run, calls = _runner(rustc, objcopy)
    assert tooling.check_toolchain("1.98.0", "Darwin", run) == 0
    assert calls[-1] == [str(objcopy), "--version"]
    assert "passed" in capsys.readouterr().out


def test_preflight_fails_on_normal_loader_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rustc, objcopy = _toolchain(tmp_path)
    run, _calls = _runner(rustc, objcopy, objcopy_status=134)
    assert tooling.check_toolchain("1.98.0", "Darwin", run) == 1
    assert "normal environment" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["rustc", "llvm"])
def test_preflight_fails_closed_for_missing_components(
    tmp_path: Path, missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rustc, objcopy = _toolchain(tmp_path)
    if missing == "rustc":
        rustc.unlink()
    else:
        (rustc.parent.parent / "lib" / "libLLVM.dylib").unlink()
    run, _calls = _runner(rustc, objcopy)
    assert tooling.check_toolchain("1.98.0", "Darwin", run) == 2
    assert "missing" in capsys.readouterr().err


def test_preflight_skips_portable_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    assert tooling.check_toolchain("1.98.0", "Linux", lambda _args: pytest.fail()) == 0
    assert "not applicable" in capsys.readouterr().out

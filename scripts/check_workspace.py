#!/usr/bin/env python3
"""Check the Rust workspace shape and portable tracked artifacts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEMBERS = (
    "compiler/decodeforge-core",
    "compiler/decodeforge-runtime",
    "compiler/decodeforge-cli",
)
EXPECTED_PACKAGES = {
    "decodeforge-core": ROOT / "compiler" / "decodeforge-core" / "Cargo.toml",
    "decodeforge-runtime": ROOT / "compiler" / "decodeforge-runtime" / "Cargo.toml",
    "decodeforge": ROOT / "compiler" / "decodeforge-cli" / "Cargo.toml",
}

# Construct the user-directory markers so this checker does not report its own
# source as a path violation.  These are portability checks, not a ban on
# ordinary absolute system paths used by an external tool at runtime.
ABSOLUTE_USER_MARKERS = (
    "/" + "Users" + "/",
    "/" + "home" + "/",
    "/" + "private" + "/",
)
WINDOWS_ABSOLUTE_RE = re.compile(r"(?:^|[\s\"'=(])(?:[A-Za-z]:[\\/])")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _workspace_members(text: str) -> list[str]:
    match = re.search(r"(?ms)^members\s*=\s*\[(.*?)\]", text)
    if match is None:
        return []
    return re.findall(r'"([^\"]+)"', match.group(1))


def _package_name(text: str) -> str | None:
    package = re.search(r"(?ms)^\[package\]\s*(.*?)(?:^\[|\Z)", text)
    if package is None:
        return None
    name = re.search(r'^name\s*=\s*"([^\"]+)"\s*$', package.group(1), re.MULTILINE)
    return None if name is None else name.group(1)


def _package_edition(text: str) -> str | None:
    package = re.search(r"(?ms)^\[package\]\s*(.*?)(?:^\[|\Z)", text)
    if package is None:
        return None
    edition = re.search(
        r'^edition\s*=\s*"([^\"]+)"\s*$', package.group(1), re.MULTILINE
    )
    return None if edition is None else edition.group(1)


def _tracked_or_present_files() -> Iterable[Path]:
    git = shutil.which("git")
    if git is not None:
        result = subprocess.run(
            [git, "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for value in result.stdout.splitlines():
                path = (ROOT / value).resolve()
                if path.is_file() and ROOT in path.parents:
                    yield path
            return

    # Keep the check useful in a source archive without Git metadata.
    ignored_parts = {
        ".git",
        ".lavish",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "target",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        yield path


def _portable_path_errors() -> list[str]:
    errors: list[str] = []
    ignored_checker = (ROOT / "scripts" / "check_docs.py").resolve()
    for path in _tracked_or_present_files():
        # This pre-existing documentation checker contains the literal marker
        # it looks for; scanning it would be a false positive.
        if path == ignored_checker:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(
            marker in text for marker in ABSOLUTE_USER_MARKERS
        ) or WINDOWS_ABSOLUTE_RE.search(text):
            errors.append(f"absolute developer path: {path.relative_to(ROOT)}")
    return errors


def check() -> list[str]:
    errors: list[str] = []
    root_manifest = ROOT / "Cargo.toml"
    if not root_manifest.is_file():
        errors.append("missing root Cargo.toml")
        return errors

    root_text = _read(root_manifest)
    if not re.search(r'^resolver\s*=\s*"3"\s*$', root_text, re.MULTILINE):
        errors.append('workspace resolver must be "3"')
    members = _workspace_members(root_text)
    if tuple(members) != EXPECTED_MEMBERS:
        errors.append(
            f"workspace members are {members!r}; expected {list(EXPECTED_MEMBERS)!r}"
        )

    manifests = sorted((ROOT / "compiler").rglob("Cargo.toml"))
    expected_manifest_paths = sorted(EXPECTED_PACKAGES.values())
    if manifests != expected_manifest_paths:
        errors.append(
            "compiler workspace must contain exactly the three expected manifests: "
            f"{[path.relative_to(ROOT).as_posix() for path in manifests]!r}"
        )

    package_text: dict[str, str] = {}
    for package, manifest in EXPECTED_PACKAGES.items():
        if not manifest.is_file():
            errors.append(
                f"missing manifest for {package}: {manifest.relative_to(ROOT)}"
            )
            continue
        text = _read(manifest)
        package_text[package] = text
        if _package_name(text) != package:
            errors.append(
                f"manifest {manifest.relative_to(ROOT)} has the wrong package name"
            )
        if _package_edition(text) != "2024":
            errors.append(
                f"manifest {manifest.relative_to(ROOT)} must use edition 2024"
            )

    core = package_text.get("decodeforge-core", "")
    runtime = package_text.get("decodeforge-runtime", "")
    cli = package_text.get("decodeforge", "")
    if "decodeforge-core" in runtime:
        errors.append("decodeforge-runtime must not depend on decodeforge-core")
    if "decodeforge-runtime" in core:
        errors.append("decodeforge-core must not depend on decodeforge-runtime")
    for dependency in ("decodeforge-core", "decodeforge-runtime"):
        if dependency not in cli:
            errors.append(f"decodeforge CLI is missing dependency {dependency}")

    errors.extend(_portable_path_errors())
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"workspace-check: {error}", file=sys.stderr)
        return 1
    print("workspace-check: ok (three crates, resolver 3, edition 2024)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

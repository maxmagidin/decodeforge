#!/usr/bin/env python3
"""Run portable structural and local-link checks over repository Markdown."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "PRIMER.md",
    ROOT / "docs" / "DESIGN.md",
    ROOT / "docs" / "BENCHMARKS.md",
    ROOT / "docs" / "Q8_FORMAT_V1.md",
    ROOT / "docs" / "EVALUATION_V1.md",
    ROOT / "results" / "README.md",
}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXCLUDED_DIRS = {
    ".git",
    ".lavish",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "target",
}


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if EXCLUDED_DIRS.isdisjoint(path.relative_to(ROOT).parts)
    )


def local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = unquote(target.split("#", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def check() -> list[str]:
    errors: list[str] = []
    for path in sorted(REQUIRED):
        if not path.is_file():
            errors.append(f"missing required document: {path.relative_to(ROOT)}")

    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if sum(line.startswith("```") for line in text.splitlines()) % 2:
            errors.append(f"unbalanced fenced code block: {relative}")
        for match in LINK_RE.finditer(text):
            target = local_link_target(path, match.group(1))
            if target is not None and not target.exists():
                errors.append(f"broken local link in {relative}: {match.group(1)}")
        if "file://" in text:
            errors.append(f"nonportable file URI in {relative}")
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"docs-check: {error}", file=sys.stderr)
        return 1
    print(f"docs-check: ok ({len(markdown_files())} Markdown files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

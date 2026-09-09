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
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
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


def local_link_target(source: Path, raw_target: str) -> tuple[Path, str | None] | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith(("http://", "https://", "mailto:")):
        return None
    path_text, separator, fragment = target.partition("#")
    path_text = unquote(path_text)
    path = source if not path_text else (source.parent / path_text).resolve()
    return path, unquote(fragment) if separator else None


def heading_slug(heading: str) -> str:
    """Approximate GitHub's stable Markdown heading IDs for local link checks."""
    heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = heading.replace("`", "").lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return re.sub(r"-+", "-", heading.replace(" ", "-")).strip("-")


def markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if match is None:
            continue
        base = heading_slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


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
            local = local_link_target(path, match.group(1))
            if local is None:
                continue
            target, fragment = local
            if not target.exists():
                errors.append(f"broken local link in {relative}: {match.group(1)}")
            elif (
                fragment
                and target.is_file()
                and target.suffix.lower() == ".md"
                and fragment not in markdown_anchors(target)
            ):
                errors.append(f"broken local anchor in {relative}: {match.group(1)}")
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

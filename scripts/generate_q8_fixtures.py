#!/usr/bin/env python3
"""Generate or verify the committed deterministic DFQ8 fixture corpus.

``--check`` is intentionally read-only: it verifies the closed manifest,
recomputes every fixture in memory, validates the semantic schema, and fails on
any byte or hash drift.  Only ``--write`` updates fixture files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from decodeforge.contracts import load_json, validate_data
from decodeforge.q8 import (
    FORMAT,
    NUMERIC_MODE,
    canonical_linear_f32_bits,
    fixture_identity,
    float_to_f32_bits,
    logical_weight_identity,
    quantize_f32_bits,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "v1"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"


def _f32(value: float) -> int:
    return float_to_f32_bits(value)


def _counter_words(domain: bytes, count: int) -> list[int]:
    """Produce finite words from a SHA-256 counter stream."""

    words: list[int] = []
    counter = 0
    while len(words) < count:
        digest = hashlib.sha256(
            b"DecodeForge/DFQ8_B32_V1/corpus/v1\0" + domain + struct.pack("<Q", counter)
        ).digest()
        for offset in range(0, len(digest), 4):
            word = int.from_bytes(digest[offset : offset + 4], "little")
            # Preserve signs and broad exponent coverage while avoiding NaN or
            # infinity in a fixture that must be accepted by the quantizer.
            # Keep the counter stream's sign and fraction bits while placing
            # values in a moderate finite range so canonical evaluation cannot
            # overflow merely because a random exponent was sampled.
            word = (word & 0x807FFFFF) | (124 << 23)
            words.append(word)
            if len(words) == count:
                break
        counter += 1
    return words


def _case_sources() -> Iterable[tuple[str, int, int, list[int], list[int]]]:
    """Yield (name, N, K, source bits, input bits) in canonical order."""

    yield "zero-signed-zero", 1, 1, [0x80000000], [0x80000000]
    exhaustive_source: list[int] = []
    for q in range(-127, 128):
        exhaustive_source.extend([_f32(127.0), _f32(float(q))])
    yield "exhaustive-q8", 255, 2, exhaustive_source, [_f32(1.0), _f32(1.0)]

    # +127 and -127 force scale to exactly +1.0.  The midpoint lanes then
    # prove ties-to-even for both signs; the first two lanes also cover q's
    # representable extrema and the final two preserve signed zero inputs.
    yield (
        "ties-and-extrema",
        1,
        8,
        [
            _f32(127.0),
            _f32(-127.0),
            _f32(2.5),
            _f32(3.5),
            _f32(-2.5),
            _f32(-3.5),
            _f32(0.0),
            _f32(-0.0),
        ],
        [_f32(1.0)] * 8,
    )

    yield (
        "finite-extremes",
        1,
        6,
        [
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x00800000,
            0x80800000,
            0x00000001,
            0x80000001,
        ],
        [0, 0, 0, 0, 0, 0],
    )

    yield (
        "subnormal-clamp",
        1,
        2,
        [0x000000BE, 0x800000BE],
        [_f32(1.0), _f32(1.0)],
    )

    one = _f32(1.0)

    for k in (1, 31, 32, 33, 63, 64, 65):
        source = [_f32(1.0 if index % 2 == 0 else -0.5) for index in range(k)]
        inputs = [_f32(-1.0 if index % 3 == 0 else 0.25) for index in range(k)]
        yield f"k-{k:02d}", 1, k, source, inputs

    yield "subnormal-scale-zero", 1, 1, [0x0000003F], [one]
    yield "subnormal-scale-min", 1, 2, [0x00000040, 0x80000020], [one, _f32(-1.0)]

    mixed_source: list[int] = []
    for row in range(2):
        for index in range(65):
            value = (index + 1) / 17.0
            if (row + index) % 2:
                value = -value
            mixed_source.append(_f32(value))
    mixed_input = [_f32(-0.75 if index % 2 else 1.25) for index in range(65)]
    yield "mixed-signs-tail", 2, 65, mixed_source, mixed_input

    n, k = 3, 33
    yield (
        "random-sha256-counter",
        n,
        k,
        _counter_words(b"source", n * k),
        _counter_words(b"input", k),
    )


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fixture_document(
    name: str, n: int, k: int, source: list[int], inputs: list[int]
) -> dict[str, Any]:
    weights = quantize_f32_bits(n, k, source)
    outputs = canonical_linear_f32_bits(inputs, weights)
    return {
        "schema_version": 1,
        "operator": "q8_linear",
        "operator_version": "q8_linear_v1",
        "format": FORMAT,
        "numeric_mode": NUMERIC_MODE,
        "case_id": name,
        "n": n,
        "k": k,
        "blocks": weights.blocks,
        "source_fp32_bits": source,
        "expected_scale_bits": list(weights.scale_bits),
        "expected_q_bytes": list(weights.q_values),
        "input_fp32_bits": inputs,
        "expected_output_fp32_bits": list(outputs),
        "error_bound": {
            "policy": "strict_f32_v1",
            "comparator": "dfq8_forward_v1",
        },
        "logical_weight_identity": logical_weight_identity(weights),
        "fixture_identity": fixture_identity(n, k, source, weights, inputs, outputs),
    }


def _generated_documents() -> dict[str, bytes]:
    documents: dict[str, bytes] = {}
    for name, n, k, source, inputs in _case_sources():
        relative = f"fixtures/{name}.json"
        documents[relative] = _canonical_json(
            _fixture_document(name, n, k, source, inputs)
        )
    return dict(sorted(documents.items()))


def _manifest(documents: dict[str, bytes]) -> bytes:
    artifacts = [
        {
            "path": path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content in sorted(documents.items())
    ]
    return _canonical_json(
        {
            "schema_version": 1,
            "format": FORMAT,
            "numeric_mode": NUMERIC_MODE,
            "artifacts": artifacts,
        }
    )


def _safe_path(raw: str) -> Path | None:
    if not raw or "\\" in raw or raw.startswith("/"):
        return None
    parsed = PurePosixPath(raw)
    if parsed.as_posix() != raw or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        return None
    candidate = FIXTURE_ROOT.joinpath(*parsed.parts)
    if not candidate.resolve().is_relative_to(FIXTURE_ROOT.resolve()):
        return None
    return candidate


def _contains_symlink(path: Path) -> bool:
    """Return whether *path* or any ancestor is a symbolic link."""

    try:
        path.relative_to(FIXTURE_ROOT)
    except ValueError:
        return True
    current = path
    while True:
        if current.is_symlink():
            return True
        if current.parent == current:
            break
        current = current.parent
    return False


def _inventory() -> tuple[set[str], list[str]]:
    """Return regular files and every symlink below the fixture root."""

    actual: set[str] = set()
    symlinks: list[str] = []
    for path in FIXTURE_ROOT.rglob("*"):
        if path == MANIFEST_PATH:
            continue
        relative = path.relative_to(FIXTURE_ROOT).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file():
            actual.add(relative)
    return actual, sorted(symlinks)


def _diagnostic(message: str) -> None:
    print(f"fixture-check: {message}", file=sys.stderr)


def _check(documents: dict[str, bytes]) -> list[str]:
    errors: list[str] = []
    expected_manifest = _manifest(documents)
    if not MANIFEST_PATH.is_file() or MANIFEST_PATH.is_symlink():
        errors.append("manifest.json is missing or is a symlink")
        return errors
    try:
        manifest = load_json(MANIFEST_PATH)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return ["manifest.json is not valid JSON"]
    diagnostics = validate_data(manifest, "fixture-manifest")
    if diagnostics:
        errors.extend(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in diagnostics
        )
    expected_records = load_json_bytes(expected_manifest)["artifacts"]
    if manifest.get("artifacts") != expected_records:
        errors.append("manifest contents differ from deterministic regeneration")

    declared: set[str] = set()
    raw_records = manifest.get("artifacts", [])
    if isinstance(raw_records, list):
        for item in raw_records:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                declared.add(item["path"])
    actual, symlinks = _inventory()
    for relative in symlinks:
        errors.append(f"symlink is not allowed in fixture inventory: {relative}")
    if actual != declared:
        errors.append(
            "closed manifest file set differs "
            f"(extra-or-missing={sorted(actual ^ declared)})"
        )
    for relative, expected in documents.items():
        path = _safe_path(relative)
        if path is None or _contains_symlink(path) or not path.is_file():
            errors.append(f"unsafe or missing fixture artifact: {relative}")
            continue
        observed = path.read_bytes()
        if observed != expected:
            errors.append(
                f"fixture bytes differ from deterministic regeneration: {relative}"
            )
        if len(observed) != len(expected):
            errors.append(f"fixture byte count differs: {relative}")
        if hashlib.sha256(observed).hexdigest() != hashlib.sha256(expected).hexdigest():
            errors.append(f"fixture hash differs: {relative}")
        diagnostics = validate_data(load_json(path), "quant-fixture")
        if diagnostics:
            errors.extend(
                f"{relative}: {json.dumps(item, sort_keys=True, separators=(',', ':'))}"
                for item in diagnostics
            )
    return errors


def load_json_bytes(content: bytes) -> dict[str, Any]:
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    return value


def _write(documents: dict[str, bytes]) -> None:
    # Resolve and inspect every destination first.  This preflight is
    # deliberately before mkdir or any write, including the manifest, so a
    # symlink cannot redirect one artifact while another has already changed.
    pending: list[tuple[Path, bytes]] = []
    for relative, content in documents.items():
        path = _safe_path(relative)
        if path is None:
            raise RuntimeError(f"unsafe generated path: {relative}")
        if _contains_symlink(path):
            raise RuntimeError(f"refusing to write through symlink: {relative}")
        pending.append((path, content))
    if _contains_symlink(MANIFEST_PATH):
        raise RuntimeError("refusing to write manifest through symlink")

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    for path, content in pending:
        # Recheck immediately before the write as a defense against a path
        # being replaced between preflight and the mkdir of its parent.
        if _contains_symlink(path):
            raise RuntimeError(f"refusing to write through symlink: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if _contains_symlink(MANIFEST_PATH):
        raise RuntimeError("refusing to write manifest through symlink")
    MANIFEST_PATH.write_bytes(_manifest(documents))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write", action="store_true", help="write deterministic fixtures"
    )
    mode.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)
    documents = _generated_documents()
    if args.write:
        _write(documents)
        errors = _check(documents)
        if errors:
            for error in errors:
                _diagnostic(error)
            return 1
        print(f"fixture-check: wrote and verified {len(documents)} fixtures")
        return 0
    errors = _check(documents)
    if errors:
        for error in errors:
            _diagnostic(error)
        return 1
    print(f"fixture-check: ok ({len(documents)} deterministic fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

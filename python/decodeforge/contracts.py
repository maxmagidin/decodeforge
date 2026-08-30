"""Offline V1 schema validation and non-executing bundle inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable

from decodeforge.q8 import (
    Q8Error,
    Q8Weights,
    canonical_linear_f32_bits,
    fixture_identity,
    is_finite_f32_bits,
    logical_weight_identity,
    quantize_f32_bits,
)

ROOT: Final = Path(__file__).resolve().parents[2]
SCHEMA_DIR: Final = ROOT / "schemas"
SCHEMA_FILES: Final = {
    "compiler-request": SCHEMA_DIR / "compiler-request.schema.json",
    "quant-fixture": SCHEMA_DIR / "quant-fixture.schema.json",
    "fixture-manifest": SCHEMA_DIR / "fixture-manifest.schema.json",
    "schedule": SCHEMA_DIR / "schedule.schema.json",
    "diagnostic": SCHEMA_DIR / "diagnostic.schema.json",
    "host-manifest": SCHEMA_DIR / "host-manifest.schema.json",
    "run-manifest": SCHEMA_DIR / "run-manifest.schema.json",
}
CATALOG_FILES: Final = (SCHEMA_DIR / "common.schema.json", *SCHEMA_FILES.values())
FOUNDATION_REQUIRED_ARTIFACTS: Final = ("host.json", "report.md", "request.json")

JsonObject = dict[str, Any]
Diagnostic = dict[str, Any]


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains the same key more than once."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def load_json(path: Path) -> JsonObject:
    """Load one JSON object while rejecting duplicate keys."""

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("root JSON value must be an object")
    return cast(JsonObject, value)


def _diagnostic(
    code: str,
    component: str,
    summary: str,
    context: JsonObject,
) -> Diagnostic:
    return {
        "schema_version": 1,
        "code": code,
        "severity": "error",
        "component": component,
        "summary": summary,
        "context": context,
    }


def _schema_registry() -> tuple[Registry[Any], dict[str, JsonObject]]:
    registry: Registry[Any] = Registry()
    schemas: dict[str, JsonObject] = {}
    identifiers: set[str] = set()
    for path in CATALOG_FILES:
        schema = load_json(path)
        identifier = schema.get("$id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"schema has no $id: {path.name}")
        if identifier in identifiers:
            raise ValueError(f"duplicate schema $id: {identifier}")
        identifiers.add(identifier)
        Draft202012Validator.check_schema(schema)
        registry = registry.with_resource(identifier, Resource.from_contents(schema))
        schemas[path.name] = schema
    return registry, schemas


def _schema_error_code(error: Any) -> str:
    path = list(error.absolute_path)
    if error.validator == "const" and path == ["schema_version"]:
        return "DFE-SCHEMA-002"
    return {
        "required": "DFE-SCHEMA-004",
        "additionalProperties": "DFE-SCHEMA-005",
        "type": "DFE-SCHEMA-006",
    }.get(str(error.validator), "DFE-SCHEMA-007")


def _validate_quant_fixture(instance: JsonObject) -> list[Diagnostic]:
    """Apply shape, bit-pattern, and identity checks after JSON Schema."""

    n = instance.get("n")
    k = instance.get("k")
    blocks = instance.get("blocks")
    source = instance.get("source_fp32_bits")
    scales = instance.get("expected_scale_bits")
    q_values = instance.get("expected_q_bytes")
    inputs = instance.get("input_fp32_bits")
    outputs = instance.get("expected_output_fp32_bits")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (n, k, blocks)
    ):
        return [
            _diagnostic("DFE-QUANT-001", "quant", "Fixture dimensions are invalid.", {})
        ]
    if not all(
        isinstance(value, list) for value in (source, scales, q_values, inputs, outputs)
    ):
        return [
            _diagnostic("DFE-QUANT-003", "quant", "Fixture arrays are invalid.", {})
        ]

    n_value = cast(int, n)
    k_value = cast(int, k)
    blocks_value = cast(int, blocks)
    source_values = cast(list[int], source)
    scale_values = cast(list[int], scales)
    q_value_list = cast(list[int], q_values)
    input_values = cast(list[int], inputs)
    output_values = cast(list[int], outputs)

    try:
        computed = quantize_f32_bits(n_value, k_value, source_values)
    except Q8Error as error:
        return [cast(Diagnostic, error.diagnostic)]
    expected_blocks = computed.blocks
    diagnostics: list[Diagnostic] = []
    expected_lengths = {
        "source_fp32_bits": n_value * k_value,
        "expected_scale_bits": n_value * expected_blocks,
        "expected_q_bytes": n_value * expected_blocks * 32,
        "input_fp32_bits": k_value,
        "expected_output_fp32_bits": n_value,
    }
    for field, expected in expected_lengths.items():
        actual = len(cast(list[Any], instance[field]))
        if actual != expected:
            diagnostics.append(
                _diagnostic(
                    "DFE-QUANT-003",
                    "quant",
                    "A fixture array length does not match its shape.",
                    {"field": field, "expected": expected, "actual": actual},
                )
            )
    if blocks_value != expected_blocks:
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-001",
                "quant",
                "The fixture block count does not equal ceil(K/32).",
                {"expected": expected_blocks, "actual": blocks_value},
            )
        )
    if diagnostics:
        return diagnostics

    source_words = tuple(source_values)
    input_words = tuple(input_values)
    output_words = tuple(output_values)
    if any(not is_finite_f32_bits(word) for word in source_words):
        return [
            _diagnostic(
                "DFE-QUANT-004",
                "quant",
                "A source weight is not finite.",
                {"field": "source_fp32_bits"},
            )
        ]
    if any(not is_finite_f32_bits(word) for word in input_words):
        return [
            _diagnostic(
                "DFE-QUANT-005",
                "quant",
                "An input activation is not finite.",
                {"field": "input_fp32_bits"},
            )
        ]

    if tuple(scale_values) != computed.scale_bits:
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-011",
                "quant",
                "Fixture scale bits do not match the canonical quantizer.",
                {"field": "expected_scale_bits"},
            )
        )
    if tuple(q_value_list) != computed.q_values:
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-011",
                "quant",
                "Fixture q bytes do not match the canonical quantizer.",
                {"field": "expected_q_bytes"},
            )
        )
    try:
        expected_weights = Q8Weights(
            n_value,
            k_value,
            expected_blocks,
            bytes(value & 0xFF for value in q_value_list),
            tuple(scale_values),
        )
    except Q8Error as error:
        diagnostics.append(cast(Diagnostic, error.diagnostic))
        return diagnostics
    if any(not is_finite_f32_bits(word) for word in output_words):
        first = next(
            index
            for index, word in enumerate(output_words)
            if not is_finite_f32_bits(word)
        )
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-010",
                "quant",
                "An expected output word is not finite.",
                {
                    "field": "expected_output_fp32_bits",
                    "index": first,
                    "bits": output_words[first],
                },
            )
        )
        return diagnostics
    try:
        canonical = canonical_linear_f32_bits(input_words, expected_weights)
    except Q8Error as error:
        diagnostics.append(cast(Diagnostic, error.diagnostic))
        return diagnostics
    if canonical != output_words:
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-011",
                "quant",
                "Fixture output bits do not match canonical scalar evaluation.",
                {"field": "expected_output_fp32_bits"},
            )
        )

    expected_logical = cast(str, instance["logical_weight_identity"])
    try:
        observed_logical = logical_weight_identity(expected_weights)
    except (Q8Error, TypeError, ValueError, OverflowError):
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-008",
                "quant",
                "The logical-weight identity could not be recomputed.",
                {"field": "logical_weight_identity"},
            )
        )
    else:
        if expected_logical != observed_logical:
            diagnostics.append(
                _diagnostic(
                    "DFE-QUANT-008",
                    "quant",
                    "The logical-weight identity does not match the fixture bytes.",
                    {"field": "logical_weight_identity"},
                )
            )
    expected_fixture = cast(str, instance["fixture_identity"])
    try:
        observed_fixture = fixture_identity(
            n_value,
            k_value,
            source_words,
            expected_weights,
            input_words,
            output_words,
        )
    except (Q8Error, TypeError, ValueError, OverflowError):
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-008",
                "quant",
                "The fixture identity could not be recomputed.",
                {"field": "fixture_identity"},
            )
        )
    else:
        if expected_fixture != observed_fixture:
            diagnostics.append(
                _diagnostic(
                    "DFE-QUANT-008",
                    "quant",
                    "The fixture identity does not match the canonical preimage.",
                    {"field": "fixture_identity"},
                )
            )
    return diagnostics


def _validate_fixture_manifest(instance: JsonObject) -> list[Diagnostic]:
    """Validate the closed, sorted path/hash manifest used by fixture checks."""

    raw_artifacts = instance.get("artifacts")
    if not isinstance(raw_artifacts, list):
        return []
    paths: list[str] = []
    diagnostics: list[Diagnostic] = []
    for item in raw_artifacts:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str):
            continue
        parsed = PurePosixPath(raw_path)
        unsafe = (
            not raw_path
            or "\\" in raw_path
            or raw_path.startswith("/")
            or parsed.as_posix() != raw_path
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or raw_path == "manifest.json"
        )
        if unsafe:
            diagnostics.append(
                _diagnostic(
                    "DFE-QUANT-012",
                    "quant",
                    "A fixture manifest path is unsafe or self-referential.",
                    {"artifact": raw_path},
                )
            )
        paths.append(raw_path)
    if len(paths) != len(set(paths)):
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-012",
                "quant",
                "A fixture manifest path appears more than once.",
                {"reason": "duplicate-path"},
            )
        )
    if paths != sorted(paths):
        diagnostics.append(
            _diagnostic(
                "DFE-QUANT-012",
                "quant",
                "Fixture manifest paths are not in canonical sorted order.",
                {"reason": "unsorted-paths"},
            )
        )
    return diagnostics


def validate_data(instance: JsonObject, schema_name: str) -> list[Diagnostic]:
    """Validate parsed data against one named local schema."""

    registry, schemas = _schema_registry()
    path = SCHEMA_FILES[schema_name]
    validator = Draft202012Validator(schemas[path.name], registry=registry)
    try:
        errors = list(validator.iter_errors(instance))
    except Unresolvable as error:
        return [
            _diagnostic(
                "DFE-SCHEMA-003",
                "schema",
                "A schema reference was not available in the local catalog.",
                {"ref": str(error)},
            )
        ]

    diagnostics = [
        _diagnostic(
            _schema_error_code(error),
            "schema",
            "The document does not satisfy its pinned schema.",
            {
                "path": list(error.absolute_path),
                "keyword": str(error.validator),
                "schema": schema_name,
            },
        )
        for error in errors
    ]
    diagnostics = sorted(
        diagnostics,
        key=lambda item: (
            json.dumps(item["context"].get("path", [])),
            item["code"],
        ),
    )
    if diagnostics:
        return diagnostics
    if schema_name == "quant-fixture":
        return _validate_quant_fixture(instance)
    if schema_name == "fixture-manifest":
        return _validate_fixture_manifest(instance)
    return diagnostics


def validate_path(path: Path, schema_name: str) -> list[Diagnostic]:
    """Parse and validate one JSON file with stable parse diagnostics."""

    try:
        instance = load_json(path)
    except DuplicateKeyError:
        return [
            _diagnostic(
                "DFE-SCHEMA-008",
                "schema",
                "A JSON object contains a duplicate key.",
                {"path": [path.name]},
            )
        ]
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return [
            _diagnostic(
                "DFE-SCHEMA-001",
                "schema",
                "The JSON document could not be parsed.",
                {"path": [path.name]},
            )
        ]
    diagnostics = validate_data(instance, schema_name)
    return diagnostics


def _check_code_registry() -> list[str]:
    errors: list[str] = []
    registry = load_json(SCHEMA_DIR / "diagnostic-codes.json")
    if registry.get("schema_version") != 1:
        errors.append("diagnostic-code registry schema_version must be 1")
    codes = registry.get("codes")
    if not isinstance(codes, list):
        return ["diagnostic-code registry codes must be an array"]

    seen_codes: set[str] = set()
    seen_names: set[tuple[str, str]] = set()
    for index, raw in enumerate(codes):
        if not isinstance(raw, dict):
            errors.append(f"diagnostic-code entry {index} is not an object")
            continue
        code = raw.get("code")
        name = raw.get("name")
        component = raw.get("component")
        if (
            not isinstance(code, str)
            or re.fullmatch(r"DFE-[A-Z0-9]+-[0-9]{3}", code) is None
        ):
            errors.append(f"diagnostic-code entry {index} has invalid code")
        elif code in seen_codes:
            errors.append(f"duplicate diagnostic code: {code}")
        else:
            seen_codes.add(code)
        if not isinstance(name, str) or not isinstance(component, str):
            errors.append(f"diagnostic-code entry {index} needs name/component")
        elif (component, name) in seen_names:
            errors.append(f"duplicate diagnostic name: {component}/{name}")
        else:
            seen_names.add((component, name))
        if not isinstance(raw.get("required_context"), list):
            errors.append(f"diagnostic-code entry {index} needs required_context")
    return errors


def check_all() -> list[str]:
    """Validate the catalog, registry, and every directed example offline."""

    errors: list[str] = []
    try:
        _schema_registry()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"schema catalog: {error}")
        return errors

    errors.extend(_check_code_registry())
    for schema_name in sorted(SCHEMA_FILES):
        directory = SCHEMA_DIR / "examples" / schema_name
        examples = sorted(directory.glob("*.json"))
        if not examples:
            errors.append(f"no examples for {schema_name}")
            continue
        for path in examples:
            diagnostics = validate_path(path, schema_name)
            if path.name.startswith("valid-") and diagnostics:
                errors.append(f"valid example rejected: {path.relative_to(ROOT)}")
            if path.name.startswith("invalid-") and not diagnostics:
                errors.append(f"invalid example accepted: {path.relative_to(ROOT)}")
            if not path.name.startswith(("valid-", "invalid-")):
                relative = path.relative_to(ROOT)
                errors.append(f"example name needs valid-/invalid- prefix: {relative}")
    return errors


def _safe_artifact_path(bundle: Path, raw: str) -> Path | None:
    if not raw or "\\" in raw or raw.startswith("/"):
        return None
    parsed = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in parsed.parts):
        return None
    if parsed.as_posix() != raw:
        return None
    root = bundle.resolve()
    candidate = root.joinpath(*parsed.parts)
    if not candidate.resolve().is_relative_to(root):
        return None
    return candidate


def _contains_symlink(bundle: Path, path: Path) -> bool:
    root = bundle.resolve()
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _artifact_identity(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _artifact_diagnostics(bundle: Path, manifest: JsonObject) -> list[Diagnostic]:
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        return []
    records = [item for item in raw_artifacts if isinstance(item, dict)]
    paths: list[str] = []
    for item in records:
        path = item.get("path")
        if isinstance(path, str):
            paths.append(path)

    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    if duplicates:
        return [
            _diagnostic(
                "DFE-BUNDLE-004",
                "bundle",
                "An artifact path appears more than once.",
                {"artifact": path},
            )
            for path in duplicates
        ]

    if manifest.get("milestone") != "foundation":
        return [
            _diagnostic(
                "DFE-BUNDLE-010",
                "bundle",
                "The foundation validator accepts only foundation bundles.",
                {"reason": "later milestone requirements are not installed"},
            )
        ]

    missing = sorted(set(FOUNDATION_REQUIRED_ARTIFACTS) - set(paths))
    if missing:
        return [
            _diagnostic(
                "DFE-BUNDLE-001",
                "bundle",
                "A foundation artifact is missing from the manifest.",
                {"artifact": path},
            )
            for path in missing
        ]

    diagnostics: list[Diagnostic] = []
    for record in sorted(records, key=lambda item: str(item.get("path", ""))):
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            continue
        path = _safe_artifact_path(bundle, raw_path)
        if path is None:
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-003",
                    "bundle",
                    "An artifact path is not a normalized bundle-relative path.",
                    {"artifact": raw_path},
                )
            )
            continue
        if _contains_symlink(bundle, path) or not path.is_file():
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-005",
                    "bundle",
                    "A listed artifact does not exist as a regular file.",
                    {"artifact": raw_path},
                )
            )
            continue
        try:
            observed_size, observed_hash = _artifact_identity(path)
        except OSError:
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-005",
                    "bundle",
                    "A listed artifact could not be read as a regular file.",
                    {"artifact": raw_path},
                )
            )
            continue
        expected_size = record.get("bytes")
        if expected_size != observed_size:
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-006",
                    "bundle",
                    "An artifact byte length differs from the manifest.",
                    {
                        "artifact": raw_path,
                        "size": observed_size,
                        "expected_size": expected_size,
                    },
                )
            )
            continue
        expected_hash = record.get("sha256")
        if expected_hash != observed_hash:
            diagnostics.append(
                _diagnostic(
                    "DFE-BUNDLE-007",
                    "bundle",
                    "An artifact digest differs from the manifest.",
                    {
                        "artifact": raw_path,
                        "sha256": observed_hash,
                        "expected_sha256": expected_hash,
                    },
                )
            )
    return diagnostics


def verify_bundle(bundle: Path) -> list[Diagnostic]:
    """Validate a foundation bundle without loading or executing artifacts."""

    manifest_path = bundle / "run-manifest.json"
    if manifest_path.is_symlink():
        return [
            _diagnostic(
                "DFE-BUNDLE-005",
                "bundle",
                "The root run manifest must be a regular file, not a symlink.",
                {"artifact": "run-manifest.json"},
            )
        ]
    if not manifest_path.is_file():
        return [
            _diagnostic(
                "DFE-BUNDLE-001",
                "bundle",
                "The root run manifest is missing.",
                {"artifact": "run-manifest.json"},
            )
        ]

    diagnostics = validate_path(manifest_path, "run-manifest")
    if diagnostics:
        return diagnostics
    manifest = load_json(manifest_path)
    diagnostics = _artifact_diagnostics(bundle, manifest)
    if diagnostics:
        return diagnostics

    host_diagnostics = validate_path(bundle / "host.json", "host-manifest")
    request_diagnostics = validate_path(bundle / "request.json", "compiler-request")
    diagnostics = host_diagnostics + request_diagnostics
    if diagnostics:
        return sorted(
            diagnostics, key=lambda item: (item["code"], json.dumps(item["context"]))
        )

    request = load_json(bundle / "request.json")
    project = manifest.get("project")
    if isinstance(project, dict):
        for field in ("format", "numeric_mode"):
            if request.get(field) != project.get(field):
                diagnostics.append(
                    _diagnostic(
                        "DFE-BUNDLE-009",
                        "bundle",
                        "The request and run manifest disagree.",
                        {"artifact": "request.json", "field": field},
                    )
                )
    return diagnostics


def _print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    for diagnostic in diagnostics:
        print(
            json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all", action="store_true", help="validate all schemas/examples"
    )
    mode.add_argument(
        "--bundle", type=Path, help="validate one foundation fixture bundle"
    )
    args = parser.parse_args(argv)

    if args.all:
        errors = check_all()
        if errors:
            for error in errors:
                print(f"schema-check: {error}", file=sys.stderr)
            return 1
        example_count = len(list((SCHEMA_DIR / "examples").glob("*/*.json")))
        print(
            f"schema-check: ok ({len(SCHEMA_FILES)} semantic schemas, "
            f"{example_count} directed examples)"
        )
        return 0

    bundle = cast(Path, args.bundle)
    diagnostics = verify_bundle(bundle)
    if diagnostics:
        _print_diagnostics(diagnostics)
        return 1
    print(f"bundle-check: ok ({bundle})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

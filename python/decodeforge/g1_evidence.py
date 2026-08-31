"""Validation and deterministic analysis for the G1 paired benchmark.

The Rust runner deliberately emits raw observations.  This module is the
portable evidence boundary: it validates the runner's closed session shape,
keeps scalar/NEON observations paired, rejects a drifting session, and emits
the only performance statistics that the G1 protocol permits.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import statistics
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

JsonObject: TypeAlias = dict[str, Any]
Diagnostic: TypeAlias = dict[str, Any]
SessionInput: TypeAlias = Mapping[str, Any]

PROTOCOL_ID: Final = "g1-prepared-call-paired-v1"
REAL_CASE_ID: Final = "tinyllama-q-proj-2048x2048"
REAL_N: Final = 2048
REAL_K: Final = 2048
REAL_INPUT_IDENTITY: Final = (
    "sha256:03263339062e7a0839f45a28b256b1e1585e5a35e4281853283049017859c590"
)
REAL_EXPECTED_IDENTITY: Final = (
    "sha256:96d06e866b38c28e2f08acdfb6055515b95dc63aede37dddf3b4315b0e5e2f4a"
)
REAL_LOGICAL_WEIGHT_IDENTITY: Final = (
    "sha256:07c6e1c13a280960451fae4698d09dd48a0d0af2b24f29eebc062d86da3253e2"
)
REAL_PACKED_WEIGHT_IDENTITY: Final = (
    "sha256:75641573aa3deae8fe3754919ab2af644546ee79121a3ee6d06f1cecaa872efc"
)
REAL_SOURCE_INPUT_SEMANTIC_IDENTITY: Final = (
    "sha256:5abf98c51f903941a1592f3df83e2e56ca7149252f5d6665c7662927c83008ac"
)
REAL_SCALAR_SOURCE_IDENTITY: Final = (
    "sha256:99aca3f9ed5177ce4abc7d1990bfb28374fb14cfa411081a2bc6cf767e28b58c"
)
REAL_NEON_SOURCE_IDENTITY: Final = (
    "sha256:d803253668b9568a9a4c0100ada42d05a9045a5ef30d56ba6099b9253c3a26de"
)
REAL_SCALAR_DISASSEMBLY_IDENTITY: Final = (
    "sha256:6f5aae5a5f3623336892b38c298ea7f00236adb5f177567f22f9bc136169c3cd"
)
REAL_NEON_DISASSEMBLY_IDENTITY: Final = (
    "sha256:05cdb5e3727da34d842e9f2903745bc714e144356c8405a9e9867714e8f173cb"
)
CASE_BUNDLE_IDENTITY: Final = (
    "sha256:797c19ddb60a5d3ec819f46650fbbe030d61b273790fe281784aa6047392beee"
)
SPEC_IDENTITY: Final = (
    "sha256:6650d0e413b403e7f81111cca75a77f931801c44f2a35cd8f275256be54848dd"
)
SESSION_FORMAT: Final = "decodeforge_g1_benchmark_v1"
PACK_FORMAT: Final = "DFQ8_B32_OI4_V1"
NUMERIC_MODE: Final = "strict_f32_v1"
MAX_SESSION_JSON_BYTES: Final = 16 * 1024 * 1024
U64_MAX: Final = (1 << 64) - 1
PAIRED_ROUNDS: Final = 40
RAW_OBSERVATIONS: Final = PAIRED_ROUNDS * 2
SCALAR_FIRST_PAIRS: Final = 20
NEON_FIRST_PAIRS: Final = 20
WARMUP_MIN_CALLS: Final = 16
WARMUP_MIN_NS: Final = 500_000_000
CALIBRATION_TARGET_NS: Final = 25_000_000
CALIBRATION_MAX_REPETITIONS: Final = 1_048_576
DRIFT_REJECTION_FRACTION: Final = 0.10
BOOTSTRAP_METHOD: Final = "paired_bca"
BOOTSTRAP_REPLICATES: Final = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL: Final = 0.95
BOOTSTRAP_SEED: Final = "sha256-counter-v1/bootstrap"
DEGENERATE_BCA_POLICY: Final = "reject"
DRIFT_WINDOW_PAIRS: Final = 10
EFFECT_ESTIMATOR: Final = "exp_median_log_paired_latency_ratio"
CLAIM_RULE: Final = (
    "real_tinyllama_2048x2048_and_all_three_session_lower_ci_bounds_gt_1"
)
AGGREGATE_POLICY: Final = "pooled_point_descriptive_only_no_ci_no_claim"
DESCRIPTIVE_LATENCY_METHODS: Final = (
    "median",
    "median_absolute_deviation",
    "nearest_rank_p95",
)
RUNNER_HOST_OS: Final = "macos"
RUNNER_OS_VERSION: Final = "15.5"
RUNNER_OS_BUILD: Final = "24F74"
RUNNER_KERNEL_RELEASE: Final = "24.5.0"
RUNNER_HOST_ARCH: Final = "aarch64"
RUNNER_POINTER_WIDTH: Final = 64
RUNNER_CPU_MODEL: Final = "Apple M4"
RUNNER_HARDWARE_MODEL: Final = "Mac16,13"
RUNNER_PHYSICAL_CORES: Final = 10
RUNNER_LOGICAL_CORES: Final = 10
RUNNER_FEATURES: Final = ("neon",)
RUNNER_THREAD_POLICY: Final = (
    "single calling thread; generated kernels create no workers"
)
RUNNER_AFFINITY_POLICY: Final = "macOS default scheduler; no hard affinity requested"
RUNNER_TIMING_BOUNDARY: Final = (
    "PreparedCall::invoke plus fixed Rust repetition loop and timer: "
    "sentinel fill + df_run_v1 + status decode + finite scan; no allocation"
)
ABI_HEADER_IDENTITY: Final = (
    "sha256:7e455301d59979be04e79b83a47e3b1bd7a143e681db01227b86557d989966cc"
)
EXPECTED_DYNAMIC_EXPORTS: Final = (
    "df_abi_version",
    "df_artifact_id",
    "df_run_v1",
)
RUNNER_COMPILER: Final = "clang"
RUNNER_COMPILER_VERSION: Final = (
    "Apple clang version 17.0.0 (clang-1700.0.13.5) "
    "Target: arm64-apple-darwin24.5.0 Thread model: posix "
    "InstalledDir: /Library/Developer/CommandLineTools/usr/bin"
)
RUNNER_TARGET: Final = "arm64-apple-darwin24.5.0"
RUNNER_SDK_VERSION: Final = "15.5"
RUNNER_OBJDUMP_VERSION: Final = (
    "Apple LLVM version 17.0.0 Optimized build. Registered Targets: "
    "aarch64 - AArch64 (little endian) aarch64_32 - AArch64 (little endian ILP32) "
    "aarch64_be - AArch64 (big endian) arm - ARM arm64 - ARM64 (little endian) "
    "arm64_32 - ARM64 (little endian ILP32) armeb - ARM (big endian) thumb - Thumb "
    "thumbeb - Thumb (big endian) x86 - 32-bit X86: Pentium-Pro and above x86-64 - "
    "64-bit X86: EM64T and AMD64"
)
_APPLE_CLANG_VERSION_RE: Final = re.compile(
    r"^Apple clang version \d+(?:\.\d+)+(?: \([^)]*\))? "
    r"Target: (?P<target>\S+) Thread model: posix(?: .*)?$"
)
_APPLE_SDK_VERSION_RE: Final = re.compile(r"^\d+\.\d+(?:\.\d+)?$")
_APPLE_LLVM_OBJDUMP_RE: Final = re.compile(
    r"^Apple LLVM version \d+(?:\.\d+)+.*Registered Targets:.*"
    r"\baarch64\b\s*-\s*AArch64\b.*$"
)
SCALAR_CLANG_FLAGS: Final = (
    "-std=c11",
    "--no-default-config",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-fno-fast-math",
    "-ffp-model=strict",
    "-ffp-contract=off",
    "-fdenormal-fp-math=ieee",
    "-fno-vectorize",
    "-fno-slp-vectorize",
    "-fno-unroll-loops",
    "-fvisibility=hidden",
    "-fPIC",
    "-dynamiclib",
    "-arch",
    "arm64",
    "-mmacosx-version-min=15.0",
    "-Wl,-install_name,@rpath/decodeforge_scalar_v1.dylib",
    "-Wl,-fatal_warnings",
    "-Wl,-exported_symbol,_df_abi_version",
    "-Wl,-exported_symbol,_df_artifact_id",
    "-Wl,-exported_symbol,_df_run_v1",
)
NEON_CLANG_FLAGS: Final = (
    *SCALAR_CLANG_FLAGS[:-5],
    "-Wl,-install_name,@rpath/decodeforge_neon_v1.dylib",
    *SCALAR_CLANG_FLAGS[-4:],
)
BACKEND_ARTIFACT_STABLE_FIELDS: Final = (
    "module_id",
    "source_hash",
    "abi_header_hash",
    "flags",
    "compiler",
    "compiler_version",
    "target",
    "sdk_version",
    "objdump_version",
    "dynamic_exports",
    "source",
    "disassembly",
    "audit",
)
_DIAGNOSTIC_CODE: Final = "DFE-SCHEMA-007"


class G1AnalysisError(ValueError):
    """A session or analysis failed the closed G1 evidence contract."""

    def __init__(self, message: str, diagnostics: Sequence[Diagnostic] = ()) -> None:
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


@dataclass(frozen=True, slots=True)
class _Pair:
    """One paired scalar/NEON observation normalized to ns per invocation."""

    pair_index: int
    scalar_ns: float
    neon_ns: float
    first_backend: str


@dataclass(frozen=True, slots=True)
class _SessionCase:
    """Validated case observations from one runner session."""

    session_id: str
    case_id: str
    kind: str
    n: int
    k: int
    expected_identity: str
    input_identity: str
    logical_weight_identity: str
    packed_weight_identity: str
    region_ir: str
    scalar_loop_ir: str
    neon_loop_ir: str
    pack_manifest: Mapping[str, Any]
    scalar_artifact: Mapping[str, Any]
    neon_artifact: Mapping[str, Any]
    pairs: tuple[_Pair, ...]
    scalar_first_pairs: int
    neon_first_pairs: int


def _diagnostic(path: Sequence[str | int], reason: str) -> Diagnostic:
    return {
        "schema_version": 1,
        "code": _DIAGNOSTIC_CODE,
        "severity": "error",
        "component": "schema",
        "summary": "The G1 benchmark session violates a cross-field invariant.",
        "context": {"path": list(path), "reason": reason},
    }


def _validate_pinned_real_case(
    case: Mapping[str, Any], path: tuple[str | int, ...]
) -> list[Diagnostic]:
    if case.get("case_id") != REAL_CASE_ID:
        return []
    expected = {
        "kind": "real",
        "n": REAL_N,
        "k": REAL_K,
        "input_identity": REAL_INPUT_IDENTITY,
        "expected_identity": REAL_EXPECTED_IDENTITY,
        "logical_weight_identity": REAL_LOGICAL_WEIGHT_IDENTITY,
        "packed_weight_identity": REAL_PACKED_WEIGHT_IDENTITY,
    }
    return [
        _diagnostic((*path, field), "real-case-identity-is-not-pinned")
        for field, value in expected.items()
        if case.get(field) != value
    ]


def _is_integer(value: object) -> bool:
    """Return true only for a canonical JSON integer, excluding booleans."""

    return type(value) is int


def _is_positive_integer(value: object) -> bool:
    return _is_integer(value) and cast(int, value) > 0


def _is_sha256_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_nonzero_sha256_identity(value: object) -> bool:
    """Return true for a content identity with an actual digest value."""

    return _is_sha256_identity(value) and cast(str, value)[7:] != "0" * 64


def _activation_digest(session_id: str, case_id: str, pair_index: int) -> bytes:
    """Reproduce the runner's length-framed activation digest exactly."""

    try:
        session_bytes = session_id.encode("utf-8")
        case_bytes = case_id.encode("utf-8")
    except UnicodeError as error:
        raise G1AnalysisError(
            "session and case identities must be valid UTF-8"
        ) from error
    hasher = hashlib.sha256()
    hasher.update(b"DecodeForge/G1/sha256-counter/v1/activation\0")
    hasher.update(len(session_bytes).to_bytes(8, "little", signed=False))
    hasher.update(session_bytes)
    hasher.update(len(case_bytes).to_bytes(8, "little", signed=False))
    hasher.update(case_bytes)
    hasher.update(pair_index.to_bytes(8, "little", signed=False))
    return hasher.digest()


def _activation_expectations(
    session_id: str, case_id: str
) -> dict[int, tuple[str, str]]:
    """Return the deterministic backend/rank assignment for every pair."""

    digests = {
        pair_index: _activation_digest(session_id, case_id, pair_index)
        for pair_index in range(PAIRED_ROUNDS)
    }
    ranked = sorted(digests.items(), key=lambda item: (item[1], item[0]))
    scalar_first = {pair_index for pair_index, _ in ranked[:SCALAR_FIRST_PAIRS]}
    return {
        pair_index: (
            "scalar" if pair_index in scalar_first else "neon",
            hashlib.sha256(digest).hexdigest(),
        )
        for pair_index, digest in digests.items()
    }


def _walk_nonfinite(
    value: object, path: tuple[str | int, ...] = ()
) -> list[Diagnostic]:
    """Find non-finite values even when callers bypass JSON parsing."""

    if isinstance(value, float):
        if not math.isfinite(value):
            return [_diagnostic(path, "nonfinite-number")]
        return [_diagnostic(path, "canonical-integer-required")]
    if _is_integer(value) and not 0 <= cast(int, value) <= U64_MAX:
        return [_diagnostic(path, "integer-out-of-u64-range")]
    if isinstance(value, Mapping):
        diagnostics: list[Diagnostic] = []
        for key, child in value.items():
            if isinstance(key, str):
                diagnostics.extend(_walk_nonfinite(child, (*path, key)))
        return diagnostics
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        diagnostics = []
        for index, child in enumerate(value):
            diagnostics.extend(_walk_nonfinite(child, (*path, index)))
        return diagnostics
    return []


def _field(
    value: object,
    path: Sequence[str | int],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    expected: int | None = None,
) -> list[Diagnostic]:
    if not _is_integer(value):
        return [_diagnostic(path, "canonical-integer-required")]
    integer = cast(int, value)
    if integer < 0 or integer > U64_MAX:
        return [_diagnostic(path, "integer-out-of-u64-range")]
    if expected is not None and integer != expected:
        return [_diagnostic(path, f"expected-{expected}")]
    if minimum is not None and integer < minimum:
        return [_diagnostic(path, f"minimum-{minimum}")]
    if maximum is not None and integer > maximum:
        return [_diagnostic(path, f"maximum-{maximum}")]
    return []


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int and int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            return False
        return all(_strict_json_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _validate_warmup(
    backend: Mapping[str, Any], path: tuple[str | int, ...]
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(
        _field(backend.get("calls"), (*path, "calls"), minimum=WARMUP_MIN_CALLS)
    )
    diagnostics.extend(
        _field(
            backend.get("elapsed_ns"),
            (*path, "elapsed_ns"),
            minimum=WARMUP_MIN_NS,
        )
    )
    return diagnostics


def _validate_calibration(
    backend: Mapping[str, Any], path: tuple[str | int, ...]
) -> list[Diagnostic]:
    calibration = backend.get("calibration")
    if not isinstance(calibration, Mapping):
        return [_diagnostic((*path, "calibration"), "object-required")]
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(
        _field(
            calibration.get("target_ns"),
            (*path, "calibration", "target_ns"),
            expected=CALIBRATION_TARGET_NS,
        )
    )
    diagnostics.extend(
        _field(
            calibration.get("max_repetitions"),
            (*path, "calibration", "max_repetitions"),
            expected=CALIBRATION_MAX_REPETITIONS,
        )
    )
    selected = calibration.get("selected_repetitions")
    diagnostics.extend(
        _field(
            selected,
            (*path, "calibration", "selected_repetitions"),
            minimum=1,
            maximum=CALIBRATION_MAX_REPETITIONS,
        )
    )
    attempts = calibration.get("attempts")
    if not isinstance(attempts, Sequence) or isinstance(
        attempts, str | bytes | bytearray
    ):
        return [
            *_dedupe_diagnostics(diagnostics),
            _diagnostic((*path, "calibration", "attempts"), "array-required"),
        ]
    if not attempts:
        diagnostics.append(
            _diagnostic((*path, "calibration", "attempts"), "nonempty-array-required")
        )
        return _dedupe_diagnostics(diagnostics)
    for index, attempt in enumerate(attempts):
        attempt_path = (*path, "calibration", "attempts", index)
        if not isinstance(attempt, Mapping):
            diagnostics.append(_diagnostic(attempt_path, "object-required"))
            continue
        diagnostics.extend(
            _field(attempt.get("elapsed_ns"), (*attempt_path, "elapsed_ns"), minimum=1)
        )
        diagnostics.extend(
            _field(
                attempt.get("repetitions"),
                (*attempt_path, "repetitions"),
                minimum=1,
                maximum=CALIBRATION_MAX_REPETITIONS,
            )
        )
    first_repetitions = (
        attempts[0].get("repetitions") if isinstance(attempts[0], Mapping) else None
    )
    if first_repetitions != 1:
        diagnostics.append(
            _diagnostic(
                (*path, "calibration", "attempts", 0, "repetitions"),
                "calibration-must-start-at-one-repetition",
            )
        )
    for index in range(1, len(attempts)):
        previous = attempts[index - 1]
        current = attempts[index]
        if isinstance(previous, Mapping) and isinstance(current, Mapping):
            previous_repetitions = previous.get("repetitions")
            current_repetitions = current.get("repetitions")
            if _is_integer(previous_repetitions) and _is_integer(current_repetitions):
                expected_repetitions = cast(int, previous_repetitions) * 2
                if cast(int, current_repetitions) != expected_repetitions:
                    diagnostics.append(
                        _diagnostic(
                            (*path, "calibration", "attempts", index, "repetitions"),
                            "calibration-repetitions-must-double",
                        )
                    )
    for index, attempt in enumerate(attempts[:-1]):
        if isinstance(attempt, Mapping):
            elapsed = attempt.get("elapsed_ns")
            if _is_integer(elapsed) and cast(int, elapsed) >= CALIBRATION_TARGET_NS:
                diagnostics.append(
                    _diagnostic(
                        (*path, "calibration", "attempts", index, "elapsed_ns"),
                        "pre-final-calibration-attempt-reached-target",
                    )
                )
    final = attempts[-1]
    if isinstance(final, Mapping):
        final_elapsed = final.get("elapsed_ns")
        final_repetitions = final.get("repetitions")
        if (
            _is_integer(selected)
            and _is_integer(final_repetitions)
            and cast(int, selected) != cast(int, final_repetitions)
        ):
            diagnostics.append(
                _diagnostic(
                    (*path, "calibration", "selected_repetitions"),
                    "selected-repetitions-must-match-final-attempt",
                )
            )
        if (
            _is_integer(final_elapsed)
            and cast(int, final_elapsed) < CALIBRATION_TARGET_NS
        ):
            diagnostics.append(
                _diagnostic(
                    (*path, "calibration", "attempts", len(attempts) - 1, "elapsed_ns"),
                    "calibration-target-not-reached",
                )
            )
    return _dedupe_diagnostics(diagnostics)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_canonical_ir(
    value: object, path: tuple[str | int, ...]
) -> tuple[Mapping[str, Any] | None, list[Diagnostic]]:
    if not isinstance(value, str):
        return None, [_diagnostic(path, "canonical-ir-string-required")]
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        return None, [_diagnostic(path, "canonical-ir-json-required")]
    if not isinstance(parsed, Mapping):
        return None, [_diagnostic(path, "canonical-ir-object-required")]
    try:
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError, UnicodeError):
        return None, [_diagnostic(path, "canonical-ir-json-required")]
    if canonical != value:
        return None, [_diagnostic(path, "ir-must-be-canonical-compact-json")]
    return cast(Mapping[str, Any], parsed), []


def _rust_schedule_json(loop: Mapping[str, Any]) -> str | None:
    """Render the exact schedule projection used by the Rust module hasher."""

    shape = loop.get("shape")
    pack = loop.get("pack")
    if not isinstance(shape, Mapping) or not isinstance(pack, Mapping):
        return None
    integer_fields = (
        "schema_version",
        "vector_lanes",
        "n_tile",
        "k_block",
        "k_unroll",
        "accumulators",
    )
    if any(not _is_integer(loop.get(field)) for field in integer_fields):
        return None
    if any(not _is_integer(shape.get(field)) for field in ("m", "n", "k")):
        return None
    if not _is_integer(pack.get("alignment")):
        return None
    string_fields = (
        "variant",
        "vector_axis",
        "reduction_order",
        "arithmetic",
        "k_padding",
        "n_tail",
        "numeric_mode",
    )
    if any(not isinstance(loop.get(field), str) for field in string_fields):
        return None
    if not isinstance(pack.get("layout"), str):
        return None
    variant = cast(str, loop["variant"])
    if variant == "scalar":
        target = {"triple": "portable", "features": []}
    elif variant == "neon":
        target = {"triple": "aarch64-apple-darwin", "features": ["neon"]}
    else:
        return None
    schedule: JsonObject = {
        "schema_version": loop["schema_version"],
        "shape": {
            "m": shape["m"],
            "n": shape["n"],
            "k": shape["k"],
        },
        "variant": loop["variant"],
        "vector_axis": loop["vector_axis"],
        "vector_lanes": loop["vector_lanes"],
        "n_tile": loop["n_tile"],
        "k_block": loop["k_block"],
        "k_unroll": loop["k_unroll"],
        "accumulators": loop["accumulators"],
        "reduction_order": loop["reduction_order"],
        "arithmetic": loop["arithmetic"],
        "k_padding": loop["k_padding"],
        "pack": {"layout": pack["layout"], "alignment": pack["alignment"]},
        "n_tail": loop["n_tail"],
        "target": target,
        "numeric_mode": loop["numeric_mode"],
    }
    try:
        return json.dumps(
            schedule,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError, UnicodeError):
        return None


def _module_identity_from_ir(case: Mapping[str, Any], backend_name: str) -> str | None:
    """Reproduce scalar_c/neon_c's length-framed module identity preimage."""

    region, region_diagnostics = _parse_canonical_ir(case.get("region_ir"), ())
    loop_field = f"{backend_name}_loop_ir"
    loop, loop_diagnostics = _parse_canonical_ir(case.get(loop_field), ())
    if region is None or loop is None or region_diagnostics or loop_diagnostics:
        return None
    operator = region.get("operator")
    region_numeric_mode = region.get("numeric_mode")
    loop_numeric_mode = loop.get("numeric_mode")
    schedule_json = _rust_schedule_json(loop)
    if (
        not isinstance(operator, str)
        or not isinstance(region_numeric_mode, str)
        or not isinstance(loop_numeric_mode, str)
        or schedule_json is None
    ):
        return None
    source_format = (
        "decodeforge_scalar_c_v1"
        if backend_name == "scalar"
        else "decodeforge_neon_c_v1"
        if backend_name == "neon"
        else None
    )
    domain = (
        b"DecodeForge/generated-module/scalar-c/v1\0"
        if backend_name == "scalar"
        else b"DecodeForge/generated-module/neon-c/v1\0"
        if backend_name == "neon"
        else None
    )
    if source_format is None or domain is None:
        return None
    try:
        hasher = hashlib.sha256()
        hasher.update(domain)
        for value in (
            operator.encode("utf-8"),
            schedule_json.encode("utf-8"),
            region_numeric_mode.encode("utf-8"),
            (1).to_bytes(4, "little", signed=False),
            source_format.encode("utf-8"),
        ):
            hasher.update(len(value).to_bytes(8, "little", signed=False))
            hasher.update(value)
        return "sha256:" + hasher.hexdigest()
    except (UnicodeError, RecursionError):
        return None


def _validate_ir_evidence(
    case: Mapping[str, Any], path: tuple[str | int, ...]
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    n = case.get("n")
    k = case.get("k")
    logical_identity = case.get("logical_weight_identity")
    numeric_mode = NUMERIC_MODE
    expected_region = {
        "schema_version": 1,
        "operator": "q8_linear",
        "shape": {"m": 1, "n": n, "k": k},
        "logical_weight_identity": logical_identity,
        "numeric_mode": numeric_mode,
    }
    region, region_diagnostics = _parse_canonical_ir(
        case.get("region_ir"), (*path, "region_ir")
    )
    diagnostics.extend(region_diagnostics)
    if region is not None and not _strict_json_equal(region, expected_region):
        diagnostics.append(
            _diagnostic((*path, "region_ir"), "region-ir-does-not-match-case")
        )
    pack_spec = {
        "schema_version": 1,
        "format": PACK_FORMAT,
        "layout": "output-interleaved",
        "tile": 4,
        "block_size": 32,
        "record_bytes": 144,
        "alignment": 16,
    }
    for backend_name, vector_lanes in (("scalar", 1), ("neon", 4)):
        expected_loop = {
            "schema_version": 1,
            "shape": {"m": 1, "n": n, "k": k},
            "variant": backend_name,
            "vector_axis": "output",
            "vector_lanes": vector_lanes,
            "n_tile": 4,
            "k_block": 32,
            "k_unroll": 1,
            "accumulators": 1,
            "reduction_order": "ascending-block-lane",
            "arithmetic": "separate-mul-add",
            "k_padding": "logical-only",
            "pack": pack_spec,
            "n_tail": "scalar-cleanup",
            "numeric_mode": numeric_mode,
            "logical_weight_identity": logical_identity,
        }
        field = f"{backend_name}_loop_ir"
        loop, loop_diagnostics = _parse_canonical_ir(case.get(field), (*path, field))
        diagnostics.extend(loop_diagnostics)
        if loop is not None and not _strict_json_equal(loop, expected_loop):
            diagnostics.append(
                _diagnostic((*path, field), "loop-ir-does-not-match-fixed-contract")
            )

    pack_manifest = case.get("pack_manifest")
    if not isinstance(pack_manifest, Mapping):
        return diagnostics
    expected_shape = {"m": 1, "n": n, "k": k}
    if not _strict_json_equal(pack_manifest.get("shape"), expected_shape):
        diagnostics.append(
            _diagnostic((*path, "pack_manifest", "shape"), "pack-shape-must-match-case")
        )
    if pack_manifest.get("logical_weight_identity") != logical_identity:
        diagnostics.append(
            _diagnostic(
                (*path, "pack_manifest", "logical_weight_identity"),
                "pack-logical-identity-must-match-case",
            )
        )
    if pack_manifest.get("packed_identity") != case.get("packed_weight_identity"):
        diagnostics.append(
            _diagnostic(
                (*path, "pack_manifest", "packed_identity"),
                "pack-identity-must-match-case",
            )
        )
    if not _strict_json_equal(pack_manifest.get("spec"), pack_spec):
        diagnostics.append(
            _diagnostic(
                (*path, "pack_manifest", "spec"),
                "pack-spec-does-not-match-fixed-contract",
            )
        )
    if _is_integer(n) and _is_integer(k):
        expected_payload = ((cast(int, n) + 3) // 4) * ((cast(int, k) + 31) // 32) * 144
        if not _strict_json_equal(pack_manifest.get("payload_bytes"), expected_payload):
            diagnostics.append(
                _diagnostic(
                    (*path, "pack_manifest", "payload_bytes"),
                    "pack-payload-size-does-not-match-case",
                )
            )
    return diagnostics


def _dedupe_diagnostics(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    seen: set[tuple[str, str]] = set()
    result: list[Diagnostic] = []
    for diagnostic in diagnostics:
        context = diagnostic.get("context")
        if not isinstance(context, Mapping):
            result.append(diagnostic)
            continue
        path = json.dumps(context.get("path", []), separators=(",", ":"))
        reason = str(context.get("reason", ""))
        key = (path, reason)
        if key not in seen:
            seen.add(key)
            result.append(diagnostic)
    return result


def _validate_runner_host(
    host: Mapping[str, Any], path: tuple[str | int, ...]
) -> list[Diagnostic]:
    """Close the claim to the runner's declared primary M4 host."""

    diagnostics: list[Diagnostic] = []
    expected_values: tuple[tuple[str, object], ...] = (
        ("os", RUNNER_HOST_OS),
        ("os_version", RUNNER_OS_VERSION),
        ("os_build", RUNNER_OS_BUILD),
        ("kernel_release", RUNNER_KERNEL_RELEASE),
        ("arch", RUNNER_HOST_ARCH),
        ("pointer_width", RUNNER_POINTER_WIDTH),
        ("native_supported", True),
        ("cpu_model", RUNNER_CPU_MODEL),
        ("hardware_model", RUNNER_HARDWARE_MODEL),
        ("physical_cores", RUNNER_PHYSICAL_CORES),
        ("logical_cores", RUNNER_LOGICAL_CORES),
        ("thread_policy", RUNNER_THREAD_POLICY),
        ("affinity_policy", RUNNER_AFFINITY_POLICY),
    )
    for field, expected in expected_values:
        if not _strict_json_equal(host.get(field), expected):
            diagnostics.append(
                _diagnostic((*path, field), "host-field-does-not-match-runner-contract")
            )
    features = host.get("features")
    if not isinstance(features, list) or not _strict_json_equal(
        features, list(RUNNER_FEATURES)
    ):
        diagnostics.append(
            _diagnostic((*path, "features"), "host-features-must-be-exactly-neon")
        )
    return diagnostics


def _validate_artifact(
    backend_name: str,
    artifact: Mapping[str, Any],
    case: Mapping[str, Any],
    path: tuple[str | int, ...],
    expected_target: str,
) -> list[Diagnostic]:
    """Validate the artifact and its backend-specific audit provenance."""

    diagnostics: list[Diagnostic] = []
    expected_flags = (
        SCALAR_CLANG_FLAGS if backend_name == "scalar" else NEON_CLANG_FLAGS
    )
    if not _strict_json_equal(artifact.get("flags"), list(expected_flags)):
        diagnostics.append(
            _diagnostic((*path, "flags"), "clang-flags-do-not-match-fixed-contract")
        )
    if not _strict_json_equal(
        artifact.get("dynamic_exports"), list(EXPECTED_DYNAMIC_EXPORTS)
    ):
        diagnostics.append(
            _diagnostic(
                (*path, "dynamic_exports"),
                "dynamic-exports-do-not-match-fixed-contract",
            )
        )
    if artifact.get("compiler") != RUNNER_COMPILER:
        diagnostics.append(
            _diagnostic((*path, "compiler"), "compiler-must-be-runner-clang")
        )
    if artifact.get("target") != expected_target:
        diagnostics.append(
            _diagnostic(
                (*path, "target"),
                "artifact-target-must-match-host-kernel-release",
            )
        )
    if artifact.get("target") != RUNNER_TARGET:
        diagnostics.append(
            _diagnostic(
                (*path, "target"), "artifact-target-is-not-the-captured-runner-target"
            )
        )
    compiler_version = artifact.get("compiler_version")
    compiler_match = (
        _APPLE_CLANG_VERSION_RE.fullmatch(compiler_version)
        if isinstance(compiler_version, str)
        else None
    )
    if compiler_match is None:
        diagnostics.append(
            _diagnostic(
                (*path, "compiler_version"),
                "compiler-version-must-be-apple-clang-with-posix-target",
            )
        )
    elif compiler_match.group("target") != expected_target:
        diagnostics.append(
            _diagnostic(
                (*path, "compiler_version"),
                "compiler-version-target-does-not-match-host-kernel-release",
            )
        )
    if compiler_version != RUNNER_COMPILER_VERSION:
        diagnostics.append(
            _diagnostic(
                (*path, "compiler_version"),
                "compiler-version-is-not-the-captured-runner-toolchain",
            )
        )
    sdk_version = artifact.get("sdk_version")
    if (
        not isinstance(sdk_version, str)
        or _APPLE_SDK_VERSION_RE.fullmatch(sdk_version) is None
    ):
        diagnostics.append(
            _diagnostic((*path, "sdk_version"), "sdk-version-must-be-numeric-apple-sdk")
        )
    if sdk_version != RUNNER_SDK_VERSION:
        diagnostics.append(
            _diagnostic(
                (*path, "sdk_version"), "sdk-version-is-not-the-captured-runner-sdk"
            )
        )
    objdump_version = artifact.get("objdump_version")
    if (
        not isinstance(objdump_version, str)
        or _APPLE_LLVM_OBJDUMP_RE.fullmatch(objdump_version) is None
    ):
        diagnostics.append(
            _diagnostic(
                (*path, "objdump_version"),
                "objdump-version-must-register-aarch64-apple-llvm-target",
            )
        )
    if objdump_version != RUNNER_OBJDUMP_VERSION:
        diagnostics.append(
            _diagnostic(
                (*path, "objdump_version"),
                "objdump-version-is-not-the-captured-runner-toolchain",
            )
        )
    if artifact.get("abi_header_hash") != ABI_HEADER_IDENTITY:
        diagnostics.append(
            _diagnostic((*path, "abi_header_hash"), "abi-header-identity-mismatch")
        )
    source = artifact.get("source")
    disassembly = artifact.get("disassembly")
    if not isinstance(source, str) or not source:
        diagnostics.append(_diagnostic((*path, "source"), "nonempty-source-required"))
    else:
        try:
            source_hash = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
        except UnicodeError:
            diagnostics.append(
                _diagnostic((*path, "source"), "source-must-be-valid-utf8")
            )
        else:
            if artifact.get("source_hash") != source_hash:
                diagnostics.append(
                    _diagnostic(
                        (*path, "source_hash"), "source-hash-does-not-match-source"
                    )
                )
    if not isinstance(disassembly, str) or not disassembly:
        diagnostics.append(
            _diagnostic((*path, "disassembly"), "nonempty-disassembly-required")
        )
        disassembly_bytes: bytes | None = None
    else:
        try:
            disassembly_bytes = disassembly.encode("utf-8")
        except UnicodeError:
            disassembly_bytes = None
            diagnostics.append(
                _diagnostic((*path, "disassembly"), "disassembly-must-be-valid-utf8")
            )
    module_id = artifact.get("module_id")
    helper_symbol: str | None = None
    if _is_nonzero_sha256_identity(module_id):
        module_id = cast(str, module_id)
        helper_symbol = f"df_kernel_{backend_name}_v1_{module_id[7:]}"
        if not isinstance(source, str) or helper_symbol not in source:
            diagnostics.append(
                _diagnostic((*path, "source"), "source-must-name-audited-helper")
            )
        if not isinstance(disassembly, str) or helper_symbol not in disassembly:
            diagnostics.append(
                _diagnostic(
                    (*path, "disassembly"), "disassembly-must-name-audited-helper"
                )
            )
    else:
        diagnostics.append(
            _diagnostic(
                (*path, "module_id"), "module-identity-required-for-helper-binding"
            )
        )
    expected_module_id = _module_identity_from_ir(case, backend_name)
    if expected_module_id is not None and module_id != expected_module_id:
        diagnostics.append(
            _diagnostic(
                (*path, "module_id"),
                "module-identity-does-not-match-rust-ir-preimage",
            )
        )
    for field in ("module_id", "source_hash", "abi_header_hash", "dylib_hash"):
        if not _is_nonzero_sha256_identity(artifact.get(field)):
            diagnostics.append(
                _diagnostic((*path, field), "sha256-artifact-identity-required")
            )
    if case.get("case_id") == REAL_CASE_ID:
        expected_source_identity = (
            REAL_SCALAR_SOURCE_IDENTITY
            if backend_name == "scalar"
            else REAL_NEON_SOURCE_IDENTITY
        )
        expected_disassembly_identity = (
            REAL_SCALAR_DISASSEMBLY_IDENTITY
            if backend_name == "scalar"
            else REAL_NEON_DISASSEMBLY_IDENTITY
        )
        if artifact.get("source_hash") != expected_source_identity:
            diagnostics.append(
                _diagnostic(
                    (*path, "source_hash"),
                    "real-artifact-source-identity-is-not-pinned",
                )
            )
        if disassembly_bytes is not None:
            disassembly_identity = (
                "sha256:" + hashlib.sha256(disassembly_bytes).hexdigest()
            )
            if disassembly_identity != expected_disassembly_identity:
                diagnostics.append(
                    _diagnostic(
                        (*path, "disassembly"),
                        "real-artifact-disassembly-identity-is-not-pinned",
                    )
                )
    audit = artifact.get("audit")
    if not isinstance(audit, Mapping):
        diagnostics.append(_diagnostic((*path, "audit"), "audit-object-required"))
        return diagnostics
    if audit.get("backend") != backend_name:
        diagnostics.append(
            _diagnostic(
                (*path, "audit", "backend"), "audit-backend-does-not-match-artifact"
            )
        )
    if helper_symbol is not None and audit.get("helper_symbol") != helper_symbol:
        diagnostics.append(
            _diagnostic(
                (*path, "audit", "helper_symbol"),
                "audit-helper-does-not-match-artifact-module",
            )
        )
    diagnostics.extend(
        _validate_audit_shape(backend_name, audit, case, (*path, "audit"))
    )
    return diagnostics


def _validate_audit_shape(
    backend_name: str,
    audit: Mapping[str, Any],
    case: Mapping[str, Any],
    path: tuple[str | int, ...],
) -> list[Diagnostic]:
    """Check audit booleans/count evidence implied by the generated shape."""

    n = case.get("n")
    k = case.get("k")
    diagnostics: list[Diagnostic] = []

    def count(field: str, minimum: int = 1) -> None:
        diagnostics.extend(_field(audit.get(field), (*path, field), minimum=minimum))

    if backend_name == "scalar":
        for field in ("scalar_scvtf_count", "scalar_fmul_count", "scalar_fadd_count"):
            count(field)
        diagnostics.extend(
            _field(audit.get("return_count"), (*path, "return_count"), expected=1)
        )
        count("conditional_branch_count", minimum=0)
        count("comparison_count", minimum=0)
        expected_loop = _is_integer(k) and cast(int, k) > 1
        if (
            type(audit.get("logical_lane_loop_observed")) is not bool
            or audit.get("logical_lane_loop_observed") is not expected_loop
        ):
            diagnostics.append(
                _diagnostic(
                    (*path, "logical_lane_loop_observed"),
                    "scalar-loop-evidence-does-not-match-shape",
                )
            )
        if n == REAL_N and k == REAL_K:
            expected_counts = {
                "scalar_scvtf_count": 1,
                "scalar_fmul_count": 2,
                "scalar_fadd_count": 2,
                "conditional_branch_count": 4,
                "comparison_count": 5,
            }
            for field, expected in expected_counts.items():
                diagnostics.extend(
                    _field(audit.get(field), (*path, field), expected=expected)
                )
        return diagnostics

    vector_expected = _is_integer(n) and cast(int, n) >= 4
    tail_expected = _is_integer(n) and cast(int, n) % 4 != 0
    expected_loop = vector_expected and _is_integer(k) and cast(int, k) > 1
    for field in (
        "signed_widen_8_to_16_count",
        "signed_widen_16_to_32_count",
        "signed_q8_to_i32_count",
        "vector_scvtf_count",
        "vector_fmul_count",
        "vector_fadd_count",
        "vector_broadcast_count",
        "vector_store_count",
    ):
        count(field, minimum=1 if vector_expected else 0)
    diagnostics.extend(
        _field(audit.get("return_count"), (*path, "return_count"), expected=1)
    )
    count("conditional_branch_count", minimum=0)
    for field, expected in (
        ("vector_path_observed", vector_expected),
        ("scalar_tail_observed", tail_expected),
        ("logical_vector_lane_loop_observed", expected_loop),
    ):
        if type(audit.get(field)) is not bool or audit.get(field) is not expected:
            diagnostics.append(
                _diagnostic((*path, field), "neon-audit-evidence-does-not-match-shape")
            )
    if n == REAL_N and k == REAL_K:
        expected_counts = {
            "signed_widen_8_to_16_count": 1,
            "signed_widen_16_to_32_count": 1,
            "signed_q8_to_i32_count": 1,
            "vector_scvtf_count": 1,
            "vector_fmul_count": 2,
            "vector_fadd_count": 2,
            "vector_broadcast_count": 1,
            "vector_store_count": 1,
            "conditional_branch_count": 3,
        }
        for field, expected in expected_counts.items():
            diagnostics.extend(
                _field(audit.get(field), (*path, field), expected=expected)
            )
    return diagnostics


def _validate_session_semantics(instance: JsonObject) -> list[Diagnostic]:
    """Validate G1 constraints JSON Schema cannot express by itself.

    This function is intentionally defensive because it is called by the
    offline schema registry after structural validation, but is also useful to
    callers that construct mutation dictionaries directly in tests.
    """

    diagnostics = _walk_nonfinite(instance)
    if diagnostics:
        return diagnostics
    if not isinstance(instance, Mapping):
        return [_diagnostic((), "object-required")]

    for field, root_expected in (
        ("format", SESSION_FORMAT),
        ("spec_identity", SPEC_IDENTITY),
        ("numeric_mode", NUMERIC_MODE),
        ("pack_format", PACK_FORMAT),
    ):
        if instance.get(field) != root_expected:
            diagnostics.append(
                _diagnostic((field,), "session-field-does-not-match-fixed-contract")
            )
    if instance.get("case_bundle_identity") != CASE_BUNDLE_IDENTITY:
        diagnostics.append(
            _diagnostic(
                ("case_bundle_identity",),
                "case-bundle-identity-does-not-match-pinned-manifest",
            )
        )
    source = instance.get("source")
    if (
        isinstance(source, Mapping)
        and source.get("input_semantic_identity") != REAL_SOURCE_INPUT_SEMANTIC_IDENTITY
    ):
        diagnostics.append(
            _diagnostic(
                ("source", "input_semantic_identity"),
                "source-input-identity-is-not-pinned",
            )
        )

    host = instance.get("host")
    expected_target = ""
    if isinstance(host, Mapping):
        diagnostics.extend(_validate_runner_host(host, ("host",)))
        kernel_release = host.get("kernel_release")
        if isinstance(kernel_release, str) and kernel_release:
            expected_target = f"arm64-apple-darwin{kernel_release}"

    checkout = instance.get("checkout")
    if isinstance(checkout, Mapping):
        revision = checkout.get("revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or not all(character in "0123456789abcdef" for character in revision)
            or not any(character != "0" for character in revision)
        ):
            diagnostics.append(
                _diagnostic(
                    ("checkout", "revision"),
                    "checkout-revision-must-be-nonzero-lowercase-object-id",
                )
            )

    timing = instance.get("timing")
    if isinstance(timing, Mapping):
        if timing.get("boundary") != RUNNER_TIMING_BOUNDARY:
            diagnostics.append(
                _diagnostic(
                    ("timing", "boundary"), "timing-boundary-does-not-match-runner"
                )
            )
        checks = (
            ("warmup_min_calls", WARMUP_MIN_CALLS),
            ("warmup_min_ns", WARMUP_MIN_NS),
            ("calibration_target_ns", CALIBRATION_TARGET_NS),
            ("calibration_max_repetitions", CALIBRATION_MAX_REPETITIONS),
            ("paired_rounds", PAIRED_ROUNDS),
            ("scalar_first_pairs", SCALAR_FIRST_PAIRS),
            ("neon_first_pairs", NEON_FIRST_PAIRS),
        )
        for name, expected in checks:
            diagnostics.extend(
                _field(timing.get(name), ("timing", name), expected=expected)
            )

    activation = instance.get("activation")
    activation_map: dict[int, str] = {}
    expected_activation: dict[int, tuple[str, str]] = {}
    session_id = instance.get("session_id")
    if isinstance(session_id, str):
        first_case = instance.get("cases")
        if (
            isinstance(first_case, Sequence)
            and not isinstance(first_case, str | bytes | bytearray)
            and first_case
            and isinstance(first_case[0], Mapping)
        ):
            first_case_id = first_case[0].get("case_id")
            if isinstance(first_case_id, str):
                try:
                    expected_activation = _activation_expectations(
                        session_id, first_case_id
                    )
                except (G1AnalysisError, UnicodeError, RecursionError):
                    diagnostics.append(
                        _diagnostic(
                            ("session_id",),
                            "session-and-case-identities-must-be-valid-utf8",
                        )
                    )
    if isinstance(activation, Sequence) and not isinstance(
        activation, str | bytes | bytearray
    ):
        for index, record in enumerate(activation):
            if not isinstance(record, Mapping):
                diagnostics.append(
                    _diagnostic(("activation", index), "object-required")
                )
                continue
            pair_index = record.get("pair_index")
            if (
                not _is_integer(pair_index)
                or not 0 <= cast(int, pair_index) < PAIRED_ROUNDS
            ):
                diagnostics.append(
                    _diagnostic(
                        ("activation", index, "pair_index"), "pair-index-out-of-range"
                    )
                )
                continue
            pair = cast(int, pair_index)
            first_backend = record.get("first_backend")
            if first_backend not in {"scalar", "neon"}:
                diagnostics.append(
                    _diagnostic(
                        ("activation", index, "first_backend"), "backend-required"
                    )
                )
                continue
            if pair in activation_map:
                diagnostics.append(
                    _diagnostic(
                        ("activation", index, "pair_index"), "duplicate-pair-index"
                    )
                )
            else:
                activation_map[pair] = cast(str, first_backend)
                activation_expected = expected_activation.get(pair)
                if activation_expected is not None:
                    if first_backend != activation_expected[0]:
                        diagnostics.append(
                            _diagnostic(
                                ("activation", index, "first_backend"),
                                "first-backend-does-not-match-deterministic-rank",
                            )
                        )
                    if record.get("digest") != activation_expected[1]:
                        diagnostics.append(
                            _diagnostic(
                                ("activation", index, "digest"),
                                "activation-digest-mismatch",
                            )
                        )
        if set(activation_map) != set(range(PAIRED_ROUNDS)):
            diagnostics.append(
                _diagnostic(("activation",), "pair-indices-must-be-0-through-39")
            )
        scalar_first = sum(value == "scalar" for value in activation_map.values())
        neon_first = sum(value == "neon" for value in activation_map.values())
        if scalar_first != SCALAR_FIRST_PAIRS:
            diagnostics.append(
                _diagnostic(("activation",), "scalar-first-count-must-be-20")
            )
        if neon_first != NEON_FIRST_PAIRS:
            diagnostics.append(
                _diagnostic(("activation",), "neon-first-count-must-be-20")
            )

    cases = instance.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, str | bytes | bytearray):
        diagnostics.append(_diagnostic(("cases",), "array-required"))
        return _dedupe_diagnostics(diagnostics)
    seen_cases: set[str] = set()
    for case_index, case in enumerate(cases):
        case_path = ("cases", case_index)
        if not isinstance(case, Mapping):
            diagnostics.append(_diagnostic(case_path, "object-required"))
            continue
        case_id = case.get("case_id")
        if isinstance(case_id, str):
            if case_id in seen_cases:
                diagnostics.append(
                    _diagnostic((*case_path, "case_id"), "duplicate-case-id")
                )
            seen_cases.add(case_id)
        diagnostics.extend(_validate_pinned_real_case(case, case_path))
        diagnostics.extend(_validate_ir_evidence(case, case_path))
        expected_identity = case.get("expected_identity")
        pack_manifest = case.get("pack_manifest")
        if isinstance(pack_manifest, Mapping):
            pack_shape = pack_manifest.get("shape")
            if isinstance(pack_shape, Mapping):
                for field in ("m", "n", "k"):
                    expected_shape = 1 if field == "m" else case.get(field)
                    if not _strict_json_equal(pack_shape.get(field), expected_shape):
                        diagnostics.append(
                            _diagnostic(
                                (*case_path, "pack_manifest", "shape", field),
                                "pack-shape-must-match-case",
                            )
                        )
            if pack_manifest.get("logical_weight_identity") != case.get(
                "logical_weight_identity"
            ):
                diagnostics.append(
                    _diagnostic(
                        (*case_path, "pack_manifest", "logical_weight_identity"),
                        "pack-logical-identity-must-match-case",
                    )
                )
            if pack_manifest.get("packed_identity") != case.get(
                "packed_weight_identity"
            ):
                diagnostics.append(
                    _diagnostic(
                        (*case_path, "pack_manifest", "packed_identity"),
                        "pack-identity-must-match-case",
                    )
                )
        for backend_name in ("scalar", "neon"):
            backend = case.get(backend_name)
            backend_path = (*case_path, backend_name)
            if not isinstance(backend, Mapping):
                diagnostics.append(_diagnostic(backend_path, "object-required"))
                continue
            artifact = backend.get("artifact")
            if isinstance(artifact, Mapping):
                diagnostics.extend(
                    _validate_artifact(
                        backend_name,
                        artifact,
                        case,
                        (*backend_path, "artifact"),
                        expected_target,
                    )
                )
            else:
                diagnostics.append(
                    _diagnostic((*backend_path, "artifact"), "object-required")
                )
            diagnostics.extend(
                _validate_warmup(backend.get("warmup", {}), (*backend_path, "warmup"))
                if isinstance(backend.get("warmup", {}), Mapping)
                else [_diagnostic((*backend_path, "warmup"), "object-required")]
            )
            diagnostics.extend(_validate_calibration(backend, backend_path))
            correctness = backend.get("correctness")
            if not isinstance(correctness, Mapping):
                diagnostics.append(
                    _diagnostic((*backend_path, "correctness"), "object-required")
                )
            else:
                for field in ("pre_timing_bit_exact", "post_timing_bit_exact"):
                    if correctness.get(field) is not True:
                        diagnostics.append(
                            _diagnostic(
                                (*backend_path, "correctness", field), "must-be-true"
                            )
                        )
                if correctness.get("expected_identity") != expected_identity:
                    diagnostics.append(
                        _diagnostic(
                            (*backend_path, "correctness", "expected_identity"),
                            "identity-must-match-case",
                        )
                    )

        scalar_backend = case.get("scalar")
        neon_backend = case.get("neon")
        if isinstance(scalar_backend, Mapping) and isinstance(neon_backend, Mapping):
            scalar_artifact = scalar_backend.get("artifact")
            neon_artifact = neon_backend.get("artifact")
            if isinstance(scalar_artifact, Mapping) and isinstance(
                neon_artifact, Mapping
            ):
                for field in (
                    "abi_header_hash",
                    "compiler",
                    "compiler_version",
                    "target",
                    "sdk_version",
                    "objdump_version",
                ):
                    if not _strict_json_equal(
                        scalar_artifact.get(field), neon_artifact.get(field)
                    ):
                        diagnostics.append(
                            _diagnostic(
                                (*case_path, field),
                                "scalar-and-neon-toolchain-evidence-differs",
                            )
                        )

        selected_repetitions: dict[str, int] = {}
        for backend_name in ("scalar", "neon"):
            backend = case.get(backend_name)
            if isinstance(backend, Mapping):
                calibration = backend.get("calibration")
                if isinstance(calibration, Mapping) and _is_integer(
                    calibration.get("selected_repetitions")
                ):
                    selected_repetitions[backend_name] = cast(
                        int, calibration["selected_repetitions"]
                    )

        samples = case.get("samples")
        sample_map: dict[tuple[int, str], Mapping[str, Any]] = {}
        if not isinstance(samples, Sequence) or isinstance(
            samples, str | bytes | bytearray
        ):
            diagnostics.append(_diagnostic((*case_path, "samples"), "array-required"))
            continue
        for sample_index, sample in enumerate(samples):
            sample_path = (*case_path, "samples", sample_index)
            if not isinstance(sample, Mapping):
                diagnostics.append(_diagnostic(sample_path, "object-required"))
                continue
            pair_index = sample.get("pair_index")
            backend = sample.get("backend")
            if (
                not _is_integer(pair_index)
                or not 0 <= cast(int, pair_index) < PAIRED_ROUNDS
            ):
                diagnostics.append(
                    _diagnostic((*sample_path, "pair_index"), "pair-index-out-of-range")
                )
                continue
            if backend not in {"scalar", "neon"}:
                diagnostics.append(
                    _diagnostic((*sample_path, "backend"), "backend-required")
                )
                continue
            diagnostics.extend(
                _field(
                    sample.get("elapsed_ns"), (*sample_path, "elapsed_ns"), minimum=1
                )
            )
            selected = selected_repetitions.get(cast(str, backend))
            repetitions = sample.get("repetitions")
            if (
                selected is not None
                and _is_integer(repetitions)
                and cast(int, repetitions) != selected
            ):
                diagnostics.append(
                    _diagnostic(
                        (*sample_path, "repetitions"),
                        "sample-repetitions-must-match-backend-calibration",
                    )
                )
            diagnostics.extend(
                _field(
                    sample.get("repetitions"),
                    (*sample_path, "repetitions"),
                    minimum=1,
                    maximum=CALIBRATION_MAX_REPETITIONS,
                )
            )
            key = (cast(int, pair_index), cast(str, backend))
            if key in sample_map:
                diagnostics.append(
                    _diagnostic(sample_path, "duplicate-paired-observation")
                )
            else:
                sample_map[key] = sample
            position = sample.get("position")
            if position not in {"first", "second"}:
                diagnostics.append(
                    _diagnostic((*sample_path, "position"), "position-required")
                )
            elif key[0] in activation_map:
                expected_position = (
                    "first" if activation_map[key[0]] == key[1] else "second"
                )
                if position != expected_position:
                    diagnostics.append(
                        _diagnostic(
                            (*sample_path, "position"),
                            "position-does-not-match-activation",
                        )
                    )
        if set(sample_map) != {
            (pair, backend)
            for pair in range(PAIRED_ROUNDS)
            for backend in ("scalar", "neon")
        }:
            diagnostics.append(
                _diagnostic(
                    (*case_path, "samples"),
                    "must-contain-one-scalar-and-one-neon-observation-per-pair",
                )
            )
    return _dedupe_diagnostics(diagnostics)


def validate_session(instance: JsonObject) -> list[Diagnostic]:
    """Validate one raw session against the schema and semantic invariants."""

    from decodeforge.contracts import validate_data

    return validate_data(instance, "g1-benchmark-session")


def _session_case(instance: SessionInput, case: Mapping[str, Any]) -> _SessionCase:
    session_id = cast(str, instance["session_id"])
    case_id = cast(str, case["case_id"])
    samples = cast(Sequence[Mapping[str, Any]], case["samples"])
    activation = cast(Sequence[Mapping[str, Any]], instance["activation"])
    activation_map = {
        cast(int, record["pair_index"]): cast(str, record["first_backend"])
        for record in activation
    }
    observations = {
        (cast(int, sample["pair_index"]), cast(str, sample["backend"])): sample
        for sample in samples
    }
    pairs: list[_Pair] = []
    for pair_index in range(PAIRED_ROUNDS):
        scalar = observations[(pair_index, "scalar")]
        neon = observations[(pair_index, "neon")]
        try:
            scalar_ns = float(cast(int, scalar["elapsed_ns"])) / float(
                cast(int, scalar["repetitions"])
            )
            neon_ns = float(cast(int, neon["elapsed_ns"])) / float(
                cast(int, neon["repetitions"])
            )
        except OverflowError as error:
            raise G1AnalysisError(
                "normalized latency integer overflows float"
            ) from error
        if not math.isfinite(scalar_ns) or not math.isfinite(neon_ns):
            raise G1AnalysisError("normalized latency is non-finite")
        pairs.append(_Pair(pair_index, scalar_ns, neon_ns, activation_map[pair_index]))
    return _SessionCase(
        session_id=session_id,
        case_id=case_id,
        kind=cast(str, case["kind"]),
        n=cast(int, case["n"]),
        k=cast(int, case["k"]),
        expected_identity=cast(str, case["expected_identity"]),
        input_identity=cast(str, case["input_identity"]),
        logical_weight_identity=cast(str, case["logical_weight_identity"]),
        packed_weight_identity=cast(str, case["packed_weight_identity"]),
        region_ir=cast(str, case["region_ir"]),
        scalar_loop_ir=cast(str, case["scalar_loop_ir"]),
        neon_loop_ir=cast(str, case["neon_loop_ir"]),
        pack_manifest=cast(Mapping[str, Any], case["pack_manifest"]),
        scalar_artifact=cast(
            Mapping[str, Any], cast(Mapping[str, Any], case["scalar"])["artifact"]
        ),
        neon_artifact=cast(
            Mapping[str, Any], cast(Mapping[str, Any], case["neon"])["artifact"]
        ),
        pairs=tuple(pairs),
        scalar_first_pairs=sum(pair.first_backend == "scalar" for pair in pairs),
        neon_first_pairs=sum(pair.first_backend == "neon" for pair in pairs),
    )


def _stable_artifact_evidence(artifact: Mapping[str, Any]) -> JsonObject:
    """Return artifact evidence excluding the per-build dylib hash."""

    return {field: artifact.get(field) for field in BACKEND_ARTIFACT_STABLE_FIELDS}


def _drift(pairs: Sequence[_Pair]) -> tuple[float, float, float]:
    centers = [math.sqrt(pair.scalar_ns * pair.neon_ns) for pair in pairs]
    first = math.exp(
        statistics.fmean(math.log(value) for value in centers[:DRIFT_WINDOW_PAIRS])
    )
    last = math.exp(
        statistics.fmean(math.log(value) for value in centers[-DRIFT_WINDOW_PAIRS:])
    )
    ratio = last / first
    if not all(math.isfinite(value) and value > 0 for value in (first, last, ratio)):
        raise G1AnalysisError("session drift statistic is non-finite")
    return first, last, ratio


def _speedup(pairs: Sequence[_Pair]) -> float:
    log_ratios = [math.log(pair.scalar_ns / pair.neon_ns) for pair in pairs]
    result = math.exp(statistics.median(log_ratios))
    if not math.isfinite(result) or result <= 0:
        raise G1AnalysisError("session speedup is non-finite")
    return result


def _bootstrap_seed() -> int:
    """Map the human-readable spec seed to a stable NumPy generator seed."""

    digest = hashlib.sha256(BOOTSTRAP_SEED.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _paired_log_speedup(scalar: Any, neon: Any) -> float:
    # Imported lazily so the base package and schema checker do not require
    # SciPy; the g1-benchmark optional extra owns this numerical dependency.
    import numpy as np

    scalar_array = np.asarray(scalar, dtype=np.float64)
    neon_array = np.asarray(neon, dtype=np.float64)
    result = np.median(np.log(scalar_array / neon_array))
    return float(result)


def _paired_bca(pairs: Sequence[_Pair]) -> tuple[float, float, float]:
    """Return geometric-median speedup and a paired SciPy BCa interval."""

    import numpy as np
    from scipy import stats  # type: ignore[import-untyped]

    scalar = np.asarray([pair.scalar_ns for pair in pairs], dtype=np.float64)
    neon = np.asarray([pair.neon_ns for pair in pairs], dtype=np.float64)
    observed_log = _paired_log_speedup(scalar, neon)
    if not math.isfinite(observed_log):
        raise G1AnalysisError("observed speedup is non-finite")
    log_ratios = np.log(scalar / neon)
    if log_ratios.size < 2 or not np.all(np.isfinite(log_ratios)):
        raise G1AnalysisError("paired BCa input is non-finite or too small")
    if bool(np.all(log_ratios == log_ratios[0])):
        raise G1AnalysisError("paired BCa interval is undefined for degenerate ratios")
    try:
        with warnings.catch_warnings(record=True) as bootstrap_warnings:
            warnings.simplefilter("always")
            result = stats.bootstrap(
                (scalar, neon),
                statistic=_paired_log_speedup,
                paired=True,
                vectorized=False,
                n_resamples=BOOTSTRAP_REPLICATES,
                confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                method="BCa",
                random_state=np.random.default_rng(_bootstrap_seed()),
            )
        if bootstrap_warnings:
            raise G1AnalysisError("paired BCa interval is undefined")
    except (ValueError, OverflowError, FloatingPointError) as error:
        raise G1AnalysisError("paired BCa interval is undefined") from error
    interval = result.confidence_interval
    lower = float(interval.low)
    upper = float(interval.high)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise G1AnalysisError("BCa confidence interval is non-finite or undefined")
    speedup = math.exp(observed_log)
    lower_speedup = math.exp(lower)
    upper_speedup = math.exp(upper)
    if not all(
        math.isfinite(value) and value > 0
        for value in (speedup, lower_speedup, upper_speedup)
    ):
        raise G1AnalysisError("BCa speedup interval is non-finite")
    return speedup, lower_speedup, upper_speedup


def _ci_report(pairs: Sequence[_Pair]) -> tuple[float, dict[str, Any]]:
    speedup, lower, upper = _paired_bca(pairs)
    return speedup, {
        "method": BOOTSTRAP_METHOD,
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "lower": lower,
        "upper": upper,
        "pairs": len(pairs),
    }


def _latency_summary(values: Sequence[float]) -> JsonObject:
    """Summarize batch-normalized latency with explicit reproducible methods."""

    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise G1AnalysisError("backend latency summary is non-finite")
    ordered = sorted(values)
    median = statistics.median(ordered)
    mad = statistics.median(abs(value - median) for value in ordered)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    p95 = ordered[rank - 1]
    return {
        "units": "ns/invocation",
        "normalization": "elapsed_ns / repetitions",
        "sample_count": len(ordered),
        "method_ids": list(DESCRIPTIVE_LATENCY_METHODS),
        "methods": {
            "median": "arithmetic median",
            "median_absolute_deviation": "median(abs(x - median(x)))",
            "nearest_rank_p95": "sorted[ceil(0.95 * n) - 1] (one-indexed rank)",
        },
        "median": median,
        "median_absolute_deviation": mad,
        "nearest_rank_p95": p95,
    }


def _case_report(
    case_key: tuple[str, str, int, int, str],
    sessions: Sequence[_SessionCase],
) -> JsonObject:
    case_id, kind, n, k, expected_identity = case_key
    session_reports: list[JsonObject] = []
    all_pairs: list[_Pair] = []
    for session in sessions:
        first_center, last_center, drift_ratio = _drift(session.pairs)
        if (
            drift_ratio > 1.0 + DRIFT_REJECTION_FRACTION
            or drift_ratio < 1.0 - DRIFT_REJECTION_FRACTION
        ):
            raise G1AnalysisError(
                f"session {session.session_id!r} exceeds the 10% drift limit",
                [
                    _diagnostic(
                        ["cases", case_id, "sessions", session.session_id, "drift"],
                        "drift-limit-exceeded",
                    )
                ],
            )
        speedup, ci = _ci_report(session.pairs)
        scalar_latencies = [pair.scalar_ns for pair in session.pairs]
        neon_latencies = [pair.neon_ns for pair in session.pairs]
        session_reports.append(
            {
                "session_id": session.session_id,
                "paired_rounds": PAIRED_ROUNDS,
                "raw_observations": RAW_OBSERVATIONS,
                "order_balance": {
                    "scalar_first": session.scalar_first_pairs,
                    "neon_first": session.neon_first_pairs,
                },
                "drift": {
                    "first_10_geometric_center_ns": first_center,
                    "last_10_geometric_center_ns": last_center,
                    "window_pairs": DRIFT_WINDOW_PAIRS,
                    "ratio": drift_ratio,
                    "rejection_fraction": DRIFT_REJECTION_FRACTION,
                    "accepted": True,
                },
                "speedup": speedup,
                "confidence_interval": ci,
                "latency_summary": {
                    "units": "ns/invocation",
                    "normalization": "elapsed_ns / repetitions",
                    "method_ids": list(DESCRIPTIVE_LATENCY_METHODS),
                    "methods": {
                        "median": "arithmetic median",
                        "median_absolute_deviation": "median(abs(x - median(x)))",
                        "nearest_rank_p95": (
                            "sorted[ceil(0.95 * n) - 1] (one-indexed rank)"
                        ),
                    },
                    "scalar": _latency_summary(scalar_latencies),
                    "neon": _latency_summary(neon_latencies),
                },
                "artifact_hashes": {
                    "scalar_dylib": session.scalar_artifact.get("dylib_hash"),
                    "neon_dylib": session.neon_artifact.get("dylib_hash"),
                },
            }
        )
        all_pairs.extend(session.pairs)
    aggregate_speedup = _speedup(all_pairs)
    lower_bounds = [
        cast(float, report["confidence_interval"]["lower"])
        for report in session_reports
    ]
    is_real_tinyllama = (
        kind == "real" and case_id == REAL_CASE_ID and n == REAL_N and k == REAL_K
    )
    if not is_real_tinyllama:
        claim_allowed = False
        claim_reason = "claim-restricted-to-real-tinyllama-2048x2048"
    elif not all(value > 1.0 for value in lower_bounds):
        claim_allowed = False
        claim_reason = "one-or-more-session-lower-bounds-not-greater-than-one"
    else:
        claim_allowed = True
        claim_reason = "all-three-session-lower-bounds-exceed-one"
    return {
        "case_id": case_id,
        "kind": kind,
        "n": n,
        "k": k,
        "expected_identity": expected_identity,
        "input_identity": sessions[0].input_identity,
        "logical_weight_identity": sessions[0].logical_weight_identity,
        "packed_weight_identity": sessions[0].packed_weight_identity,
        "region_ir": sessions[0].region_ir,
        "scalar_loop_ir": sessions[0].scalar_loop_ir,
        "neon_loop_ir": sessions[0].neon_loop_ir,
        "pack_manifest": dict(sessions[0].pack_manifest),
        "sessions": session_reports,
        "aggregate": {
            "inference": AGGREGATE_POLICY,
            "paired_rounds": len(all_pairs),
            "raw_observations": len(all_pairs) * 2,
            "speedup": aggregate_speedup,
        },
        "claim": {
            "allowed": claim_allowed,
            "reason": claim_reason,
            "session_lower_bounds": lower_bounds,
        },
    }


def _validate_input_sessions(sessions: Sequence[SessionInput]) -> list[SessionInput]:
    if len(sessions) != 3:
        raise G1AnalysisError("G1 analysis requires exactly three sessions")
    ordered = sorted(sessions, key=lambda session: str(session.get("session_id", "")))
    session_ids: list[str] = []
    process_ids: set[int] = set()
    for session in ordered:
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise G1AnalysisError("every session must have a non-empty session_id")
        if session_id in session_ids:
            raise G1AnalysisError(
                "the three sessions must have distinct session_id values"
            )
        session_ids.append(session_id)
        diagnostics = validate_session(cast(JsonObject, dict(session)))
        if diagnostics:
            if any(
                isinstance(item.get("context"), Mapping)
                and cast(Mapping[str, Any], item["context"]).get("path", [])[:1]
                == ["host"]
                for item in diagnostics
            ):
                raise G1AnalysisError(
                    f"session {session_id!r} host fingerprint is invalid",
                    diagnostics,
                )
            raise G1AnalysisError(
                f"session {session_id!r} failed schema validation", diagnostics
            )
        checkout = session.get("checkout")
        if not isinstance(checkout, Mapping) or checkout.get("dirty") is not False:
            raise G1AnalysisError("G1 analysis requires a clean checkout")
        host = session.get("host")
        if not isinstance(host, Mapping):
            raise G1AnalysisError(f"session {session_id!r} has no host fingerprint")
        process_id = host.get("process_id")
        if not _is_positive_integer(process_id):
            raise G1AnalysisError(f"session {session_id!r} has an invalid process_id")
        process_id_int = cast(int, process_id)
        if process_id_int in process_ids:
            raise G1AnalysisError(
                "the three sessions must have distinct process_id values"
            )
        process_ids.add(process_id_int)
    reference = ordered[0]
    for session in ordered[1:]:
        for field in (
            "format",
            "spec_identity",
            "case_bundle_identity",
            "source",
            "numeric_mode",
            "pack_format",
            "checkout",
            "timing",
        ):
            if session.get(field) != reference.get(field):
                raise G1AnalysisError(f"session metadata field {field!r} differs")
        reference_host = reference.get("host")
        current_host = session.get("host")
        if isinstance(reference_host, Mapping) and isinstance(current_host, Mapping):
            reference_fingerprint = {
                key: value
                for key, value in reference_host.items()
                if key != "process_id"
            }
            current_fingerprint = {
                key: value for key, value in current_host.items() if key != "process_id"
            }
            if current_fingerprint != reference_fingerprint:
                raise G1AnalysisError("session host fingerprint differs")
    return ordered


def analyze_sessions(sessions: Sequence[SessionInput]) -> JsonObject:
    """Analyze exactly three validated G1 session objects."""

    ordered = _validate_input_sessions(sessions)
    by_case: dict[str, list[_SessionCase]] = {}
    case_keys: dict[str, tuple[str, str, int, int, str]] = {}
    case_signatures: dict[str, str] = {}
    for session in ordered:
        raw_cases = cast(Sequence[Mapping[str, Any]], session["cases"])
        for raw_case in raw_cases:
            case = _session_case(session, raw_case)
            key = (case.case_id, case.kind, case.n, case.k, case.expected_identity)
            prior = case_keys.get(case.case_id)
            if prior is not None and prior != key:
                raise G1AnalysisError(f"case {case.case_id!r} differs across sessions")
            signature = json.dumps(
                {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "n": case.n,
                    "k": case.k,
                    "input_identity": case.input_identity,
                    "expected_identity": case.expected_identity,
                    "logical_weight_identity": case.logical_weight_identity,
                    "packed_weight_identity": case.packed_weight_identity,
                    "region_ir": case.region_ir,
                    "scalar_loop_ir": case.scalar_loop_ir,
                    "neon_loop_ir": case.neon_loop_ir,
                    "pack_manifest": case.pack_manifest,
                    "scalar_artifact": _stable_artifact_evidence(case.scalar_artifact),
                    "neon_artifact": _stable_artifact_evidence(case.neon_artifact),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            prior_signature = case_signatures.get(case.case_id)
            if prior_signature is not None and prior_signature != signature:
                raise G1AnalysisError(
                    f"case {case.case_id!r} identity metadata differs across sessions"
                )
            case_signatures[case.case_id] = signature
            case_keys[case.case_id] = key
            by_case.setdefault(case.case_id, []).append(case)
    expected_sessions = {cast(str, session["session_id"]) for session in ordered}
    for case_id, case_sessions in by_case.items():
        observed = {session.session_id for session in case_sessions}
        if observed != expected_sessions:
            raise G1AnalysisError(
                f"case {case_id!r} must be present in all three sessions"
            )
    reports = [
        _case_report(
            case_keys[case_id],
            sorted(by_case[case_id], key=lambda value: value.session_id),
        )
        for case_id in sorted(by_case)
    ]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "format": cast(str, ordered[0]["format"]),
        "spec_identity": cast(str, ordered[0]["spec_identity"]),
        "case_bundle_identity": cast(str, ordered[0]["case_bundle_identity"]),
        "source": cast(JsonObject, dict(cast(Mapping[str, Any], ordered[0]["source"]))),
        "numeric_mode": cast(str, ordered[0]["numeric_mode"]),
        "pack_format": cast(str, ordered[0]["pack_format"]),
        "checkout": cast(
            JsonObject, dict(cast(Mapping[str, Any], ordered[0]["checkout"]))
        ),
        "host_fingerprint": {
            key: value
            for key, value in cast(Mapping[str, Any], ordered[0]["host"]).items()
            if key != "process_id"
        },
        "process_ids": [
            cast(int, cast(Mapping[str, Any], session["host"])["process_id"])
            for session in ordered
        ],
        "session_ids": [cast(str, session["session_id"]) for session in ordered],
        "timing": {
            "boundary": RUNNER_TIMING_BOUNDARY,
            "paired_rounds": PAIRED_ROUNDS,
            "raw_observations_per_case": RAW_OBSERVATIONS,
            "warmup_min_calls": WARMUP_MIN_CALLS,
            "warmup_min_ns": WARMUP_MIN_NS,
            "calibration_target_ns": CALIBRATION_TARGET_NS,
            "drift_rejection_fraction": DRIFT_REJECTION_FRACTION,
            "drift_window_pairs": DRIFT_WINDOW_PAIRS,
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "degenerate_policy": DEGENERATE_BCA_POLICY,
        },
        "inference_policy": {
            "effect_estimator": EFFECT_ESTIMATOR,
            "claim_rule": CLAIM_RULE,
            "aggregate_policy": AGGREGATE_POLICY,
            "descriptive_latency_methods": list(DESCRIPTIVE_LATENCY_METHODS),
        },
        "cases": reports,
    }


def load_sessions(paths: Sequence[Path]) -> list[JsonObject]:
    """Load and parse raw session JSON files without accepting duplicates."""

    if len(paths) != 3:
        raise G1AnalysisError("G1 analysis requires exactly three session files")
    sessions: list[JsonObject] = []
    for path in paths:
        descriptor = -1
        try:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | nofollow
            )
            flags |= getattr(os, "O_NONBLOCK", 0)
            if nofollow == 0:
                initial_path_metadata = os.lstat(path)
                if stat.S_ISLNK(initial_path_metadata.st_mode):
                    raise G1AnalysisError(
                        f"session input {path} must be a regular non-symlink file"
                    )
            try:
                descriptor = os.open(path, flags)
            except OSError as error:
                if nofollow and error.errno == errno.ELOOP:
                    raise G1AnalysisError(
                        f"session input {path} must be a regular non-symlink file"
                    ) from error
                raise
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise G1AnalysisError(
                    f"session input {path} must be a regular non-symlink file"
                )
            if metadata.st_size > MAX_SESSION_JSON_BYTES:
                raise G1AnalysisError(
                    f"session input {path} exceeds the "
                    f"{MAX_SESSION_JSON_BYTES}-byte bound"
                )
            initial_identity = _file_identity(metadata)
            chunks: list[bytes] = []
            total_bytes = 0
            remaining = MAX_SESSION_JSON_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
                total_bytes += len(chunk)
                if total_bytes > MAX_SESSION_JSON_BYTES:
                    raise G1AnalysisError(
                        f"session input {path} exceeds the "
                        f"{MAX_SESSION_JSON_BYTES}-byte bound"
                    )
            data = b"".join(chunks)
            final_descriptor_metadata = os.fstat(descriptor)
            if _file_identity(final_descriptor_metadata) != initial_identity:
                raise G1AnalysisError(f"session input {path} changed while it was read")
            final_path_metadata = os.stat(path, follow_symlinks=False)
            if _file_identity(final_path_metadata) != initial_identity:
                raise G1AnalysisError(f"session input {path} changed while it was read")
            value = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=_no_duplicate_object,
                parse_constant=_reject_json_constant,
            )
            if not isinstance(value, dict):
                raise ValueError("root JSON value must be an object")
            sessions.append(cast(JsonObject, value))
        except G1AnalysisError:
            raise
        except (
            OSError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
            RecursionError,
        ) as error:
            raise G1AnalysisError(f"could not parse session {path}") from error
        finally:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
    return sessions


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the descriptor/path identity fields checked around each read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def canonical_json_bytes(value: JsonObject) -> bytes:
    """Serialize a report deterministically and reject non-finite numbers."""

    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path: Path, content: bytes) -> None:
    if not path.name or path.name in {".", ".."}:
        raise G1AnalysisError("output path must name an explicit file")
    parent = path.parent if path.parent != Path("") else Path(".")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
            except OSError:
                directory_fd = -1
            if directory_fd >= 0:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            with suppress(OSError):
                temporary.unlink()
            raise
    except OSError as error:
        raise G1AnalysisError(f"atomic output write failed for {path}") from error


def markdown_summary(report: JsonObject) -> str:
    """Render a concise deterministic summary from a machine report."""

    lines = [
        "# DecodeForge G1 paired benchmark",
        "",
        "Protocol: `g1-prepared-call-paired-v1`",
        "",
        "| Case | Session speedups (95% BCa CI) | Pooled descriptive speedup | Claim |",
        "| --- | --- | --- | --- |",
    ]
    for case in cast(list[JsonObject], report["cases"]):
        sessions = cast(list[JsonObject], case["sessions"])
        session_values = []
        for session in sessions:
            ci = cast(JsonObject, session["confidence_interval"])
            session_values.append(
                f"{float(session['speedup']):.6g} "
                f"[{float(ci['lower']):.6g}, {float(ci['upper']):.6g}]"
            )
        aggregate = cast(JsonObject, case["aggregate"])
        aggregate_text = f"{float(aggregate['speedup']):.6g} (point only)"
        claim = cast(JsonObject, case["claim"])
        lines.append(
            f"| `{case['case_id']}` | "
            f"{'<br>'.join(session_values)} | {aggregate_text} | "
            f"{'allowed' if claim['allowed'] else 'not allowed'} |"
        )
    lines.extend(
        [
            "",
            "The aggregate is pooled-descriptive (not used for the claim).",
            "BCa intervals use exactly 10,000 deterministic paired resamples; "
            "each pair remains intact.",
            "Latency summaries normalize every batch as elapsed_ns / repetitions "
            "and report median, median absolute deviation, and nearest-rank p95 "
            "in ns/invocation.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_analysis_outputs(
    report: JsonObject, report_path: Path, summary_path: Path
) -> None:
    """Atomically write the deterministic JSON report and Markdown summary."""

    if report_path.resolve() == summary_path.resolve():
        raise G1AnalysisError(
            "JSON report and Markdown summary must be different files"
        )
    _atomic_write(report_path, canonical_json_bytes(report))
    _atomic_write(summary_path, markdown_summary(report).encode("utf-8"))


# Compatibility aliases make the intended public boundary discoverable to
# callers that use the noun from the protocol rather than ``analyze_sessions``.
analyze_g1_sessions = analyze_sessions
write_report_atomic = write_analysis_outputs

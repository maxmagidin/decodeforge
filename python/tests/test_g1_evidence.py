"""G1 evidence schema, mutation, statistics, and determinism tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from decodeforge.contracts import load_json, validate_data, validate_path
from decodeforge.g1_evidence import (
    ABI_HEADER_IDENTITY,
    AGGREGATE_POLICY,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CALIBRATION_TARGET_NS,
    CASE_BUNDLE_IDENTITY,
    CLAIM_RULE,
    DEGENERATE_BCA_POLICY,
    DESCRIPTIVE_LATENCY_METHODS,
    DRIFT_REJECTION_FRACTION,
    DRIFT_WINDOW_PAIRS,
    EFFECT_ESTIMATOR,
    EXPECTED_DYNAMIC_EXPORTS,
    MAX_SESSION_JSON_BYTES,
    NEON_CLANG_FLAGS,
    REAL_CASE_ID,
    REAL_NEON_DISASSEMBLY_IDENTITY,
    REAL_NEON_SOURCE_IDENTITY,
    REAL_SCALAR_DISASSEMBLY_IDENTITY,
    REAL_SCALAR_SOURCE_IDENTITY,
    RUNNER_AFFINITY_POLICY,
    RUNNER_COMPILER,
    RUNNER_COMPILER_VERSION,
    RUNNER_CPU_MODEL,
    RUNNER_FEATURES,
    RUNNER_HARDWARE_MODEL,
    RUNNER_HOST_ARCH,
    RUNNER_HOST_OS,
    RUNNER_KERNEL_RELEASE,
    RUNNER_LOGICAL_CORES,
    RUNNER_OBJDUMP_VERSION,
    RUNNER_OS_BUILD,
    RUNNER_OS_VERSION,
    RUNNER_PHYSICAL_CORES,
    RUNNER_POINTER_WIDTH,
    RUNNER_SDK_VERSION,
    RUNNER_TARGET,
    RUNNER_THREAD_POLICY,
    RUNNER_TIMING_BOUNDARY,
    SCALAR_CLANG_FLAGS,
    SPEC_IDENTITY,
    G1AnalysisError,
    _activation_expectations,
    analyze_sessions,
    canonical_json_bytes,
    markdown_summary,
    write_analysis_outputs,
)

EXAMPLE = (
    Path(__file__).parents[2]
    / "schemas"
    / "examples"
    / "g1-benchmark-session"
    / "valid-minimal.json"
)
SPEC = Path(__file__).parents[2] / "benchmarks" / "g1" / "spec.json"


def _session_set() -> list[dict[str, Any]]:
    baseline = load_json(EXAMPLE)
    sessions: list[dict[str, Any]] = []
    for index in range(3):
        session = deepcopy(baseline)
        session["session_id"] = f"test-session-{index}"
        session["host"]["process_id"] = 1001 + index
        expected = _activation_expectations(
            session["session_id"], session["cases"][0]["case_id"]
        )
        for record in session["activation"]:
            backend, digest = expected[record["pair_index"]]
            record["first_backend"] = backend
            record["digest"] = digest
        for sample in session["cases"][0]["samples"]:
            pair_index = sample["pair_index"]
            if sample["backend"] == "scalar":
                sample["elapsed_ns"] = (
                    100000 + pair_index * 100 + (pair_index * pair_index * 17) % 997
                )
            else:
                sample["elapsed_ns"] = (
                    50000 + pair_index * 50 + (pair_index * pair_index * 31) % 499
                )
            first_backend = expected[sample["pair_index"]][0]
            sample["position"] = (
                "first" if sample["backend"] == first_backend else "second"
            )
        sessions.append(session)
    return sessions


def test_schema_example_and_semantic_validator_accept_valid_session() -> None:
    assert validate_path(EXAMPLE, "g1-benchmark-session") == []


def test_analysis_constants_match_the_content_addressed_protocol_spec() -> None:
    spec_bytes = SPEC.read_bytes()
    spec = json.loads(spec_bytes)
    timing = spec["timing"]
    host = spec["primary_host"]
    artifact_policy = spec["artifact_policy"]
    assert f"sha256:{hashlib.sha256(spec_bytes).hexdigest()}" == SPEC_IDENTITY
    assert load_json(EXAMPLE)["case_bundle_identity"] == CASE_BUNDLE_IDENTITY
    assert host == {
        "os": RUNNER_HOST_OS,
        "os_version": RUNNER_OS_VERSION,
        "os_build": RUNNER_OS_BUILD,
        "kernel_release": RUNNER_KERNEL_RELEASE,
        "arch": RUNNER_HOST_ARCH,
        "pointer_width": RUNNER_POINTER_WIDTH,
        "native_supported": True,
        "cpu_model": RUNNER_CPU_MODEL,
        "hardware_model": RUNNER_HARDWARE_MODEL,
        "physical_cores": RUNNER_PHYSICAL_CORES,
        "logical_cores": RUNNER_LOGICAL_CORES,
        "features": list(RUNNER_FEATURES),
        "thread_policy": RUNNER_THREAD_POLICY,
        "affinity_policy": RUNNER_AFFINITY_POLICY,
    }
    assert timing["boundary"] == RUNNER_TIMING_BOUNDARY
    assert timing["degenerate_bca_policy"] == DEGENERATE_BCA_POLICY
    assert timing["confidence_level"] == BOOTSTRAP_CONFIDENCE_LEVEL
    assert timing["drift_window_pairs"] == DRIFT_WINDOW_PAIRS
    assert timing["drift_rejection_fraction"] == DRIFT_REJECTION_FRACTION
    assert timing["effect_estimator"] == EFFECT_ESTIMATOR
    assert timing["claim_rule"] == CLAIM_RULE
    assert timing["aggregate_policy"] == AGGREGATE_POLICY
    assert timing["descriptive_latency_methods"] == list(DESCRIPTIVE_LATENCY_METHODS)
    assert timing["bootstrap_method"] == BOOTSTRAP_METHOD
    assert timing["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert timing["bootstrap_seed"] == BOOTSTRAP_SEED
    assert artifact_policy == {
        "real_case_id": REAL_CASE_ID,
        "scalar_source_identity": REAL_SCALAR_SOURCE_IDENTITY,
        "neon_source_identity": REAL_NEON_SOURCE_IDENTITY,
        "scalar_disassembly_identity": REAL_SCALAR_DISASSEMBLY_IDENTITY,
        "neon_disassembly_identity": REAL_NEON_DISASSEMBLY_IDENTITY,
        "scalar": {
            "compiler": RUNNER_COMPILER,
            "compiler_version": RUNNER_COMPILER_VERSION,
            "target": RUNNER_TARGET,
            "sdk_version": RUNNER_SDK_VERSION,
            "objdump_version": RUNNER_OBJDUMP_VERSION,
            "abi_header_identity": ABI_HEADER_IDENTITY,
            "dynamic_exports": list(EXPECTED_DYNAMIC_EXPORTS),
            "flags": list(SCALAR_CLANG_FLAGS),
        },
        "neon": {
            "compiler": RUNNER_COMPILER,
            "compiler_version": RUNNER_COMPILER_VERSION,
            "target": RUNNER_TARGET,
            "sdk_version": RUNNER_SDK_VERSION,
            "objdump_version": RUNNER_OBJDUMP_VERSION,
            "abi_header_identity": ABI_HEADER_IDENTITY,
            "dynamic_exports": list(EXPECTED_DYNAMIC_EXPORTS),
            "flags": list(NEON_CLANG_FLAGS),
        },
    }


def test_schema_rejects_unknown_field_and_noncanonical_numbers() -> None:
    session = load_json(EXAMPLE)
    unknown = deepcopy(session)
    unknown["unexpected"] = True
    assert validate_data(unknown, "g1-benchmark-session")

    noncanonical = deepcopy(session)
    noncanonical["cases"][0]["samples"][0]["elapsed_ns"] = 1.0
    diagnostics = validate_data(noncanonical, "g1-benchmark-session")
    assert any(
        item["context"]["reason"] == "canonical-integer-required"
        for item in diagnostics
    )

    oversized_session_id = deepcopy(session)
    oversized_session_id["session_id"] = "s" * 65
    assert validate_data(oversized_session_id, "g1-benchmark-session")


def test_mutations_reject_pairing_order_warmup_calibration_and_correctness() -> None:
    mutations: list[tuple[str, Any]] = []

    session = load_json(EXAMPLE)
    activation = deepcopy(session)
    activation["activation"][1]["pair_index"] = activation["activation"][0][
        "pair_index"
    ]
    mutations.append(("activation", activation))

    samples = deepcopy(session)
    samples["cases"][0]["samples"][0]["position"] = "first"
    mutations.append(("position", samples))

    warmup = deepcopy(session)
    warmup["cases"][0]["scalar"]["warmup"]["elapsed_ns"] = 499_999_999
    mutations.append(("warmup", warmup))

    calibration = deepcopy(session)
    calibration["cases"][0]["neon"]["calibration"]["attempts"][0]["elapsed_ns"] = (
        CALIBRATION_TARGET_NS - 1
    )
    mutations.append(("calibration", calibration))

    correctness = deepcopy(session)
    correctness["cases"][0]["scalar"]["correctness"]["post_timing_bit_exact"] = False
    mutations.append(("correctness", correctness))

    for name, mutated in mutations:
        assert validate_data(mutated, "g1-benchmark-session"), name


def test_analysis_requires_three_distinct_sessions_and_common_cases() -> None:
    sessions = _session_set()
    with pytest.raises(G1AnalysisError):
        analyze_sessions(sessions[:2])
    duplicate = deepcopy(sessions)
    duplicate[2]["session_id"] = duplicate[1]["session_id"]
    with pytest.raises(G1AnalysisError):
        analyze_sessions(duplicate)
    missing_case = deepcopy(sessions)
    missing_case[2]["cases"] = []
    with pytest.raises(G1AnalysisError):
        analyze_sessions(missing_case)


def test_analysis_reports_speedup_ci_aggregate_and_real_case_claim() -> None:
    report = analyze_sessions(_session_set())
    case = report["cases"][0]
    assert case["sessions"][0]["speedup"] == pytest.approx(2.0, rel=0.01)
    assert case["sessions"][0]["confidence_interval"]["replicates"] == 10_000
    assert case["aggregate"]["paired_rounds"] == 120
    assert case["aggregate"]["raw_observations"] == 240
    assert (
        case["aggregate"]["inference"] == "pooled_point_descriptive_only_no_ci_no_claim"
    )
    assert "confidence_interval" not in case["aggregate"]
    assert case["sessions"][0]["latency_summary"]["scalar"]["units"] == "ns/invocation"
    assert case["sessions"][0]["latency_summary"]["scalar"]["sample_count"] == 40
    assert case["claim"]["allowed"] is True

    synthetic = _session_set()
    for session in synthetic:
        case = session["cases"][0]
        case["kind"] = "synthetic"
        case["case_id"] = "synthetic-tail"
        expected = _activation_expectations(session["session_id"], case["case_id"])
        for record in session["activation"]:
            backend, digest = expected[record["pair_index"]]
            record["first_backend"] = backend
            record["digest"] = digest
        for sample in case["samples"]:
            sample["position"] = (
                "first"
                if sample["backend"] == expected[sample["pair_index"]][0]
                else "second"
            )
    report = analyze_sessions(synthetic)
    assert report["cases"][0]["claim"]["allowed"] is False


def test_drift_rejection_and_paired_determinism() -> None:
    sessions = _session_set()
    for session in sessions:
        for sample in session["cases"][0]["samples"]:
            if sample["pair_index"] >= 30:
                sample["elapsed_ns"] *= 2
    with pytest.raises(G1AnalysisError, match="drift"):
        analyze_sessions(sessions)

    stable = _session_set()
    first = analyze_sessions(stable)
    second = analyze_sessions(list(reversed(stable)))
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert markdown_summary(first) == markdown_summary(second)


def test_outputs_are_atomic_and_byte_stable(tmp_path: Path) -> None:
    report = analyze_sessions(_session_set())
    report_path = tmp_path / "analysis" / "report.json"
    summary_path = tmp_path / "analysis" / "report.md"
    write_analysis_outputs(report, report_path, summary_path)
    first_report = report_path.read_bytes()
    first_summary = summary_path.read_bytes()
    write_analysis_outputs(report, report_path, summary_path)
    assert report_path.read_bytes() == first_report
    assert summary_path.read_bytes() == first_summary
    assert not list(report_path.parent.glob(".*.tmp-*"))
    with pytest.raises(G1AnalysisError, match="different files"):
        write_analysis_outputs(report, report_path, report_path)


def test_nonfinite_programmatic_value_is_rejected() -> None:
    session = load_json(EXAMPLE)
    session["cases"][0]["samples"][0]["elapsed_ns"] = float("nan")
    diagnostics = validate_data(session, "g1-benchmark-session")
    assert diagnostics


def test_runner_integer_bound_and_strict_ir_number_types() -> None:
    session = load_json(EXAMPLE)
    huge = deepcopy(session)
    huge["cases"][0]["samples"][0]["elapsed_ns"] = 1 << 64
    diagnostics = validate_data(huge, "g1-benchmark-session")
    assert diagnostics

    ir_bool = deepcopy(session)
    region = json.loads(ir_bool["cases"][0]["region_ir"])
    region["shape"]["m"] = True
    ir_bool["cases"][0]["region_ir"] = json.dumps(region, separators=(",", ":"))
    assert validate_data(ir_bool, "g1-benchmark-session")

    ir_float = deepcopy(session)
    loop = json.loads(ir_float["cases"][0]["scalar_loop_ir"])
    loop["vector_lanes"] = 1.0
    ir_float["cases"][0]["scalar_loop_ir"] = json.dumps(loop, separators=(",", ":"))
    assert validate_data(ir_float, "g1-benchmark-session")


def test_malformed_unicode_and_deep_ir_return_diagnostics() -> None:
    ir_surrogate = load_json(EXAMPLE)
    region = json.loads(ir_surrogate["cases"][0]["region_ir"])
    region["operator"] = "\ud800"
    ir_surrogate["cases"][0]["region_ir"] = json.dumps(
        region, ensure_ascii=False, separators=(",", ":")
    )
    assert validate_data(ir_surrogate, "g1-benchmark-session")

    disassembly_surrogate = load_json(EXAMPLE)
    disassembly_surrogate["cases"][0]["scalar"]["artifact"]["disassembly"] += "\ud800"
    assert validate_data(disassembly_surrogate, "g1-benchmark-session")

    case_id_surrogate = load_json(EXAMPLE)
    case_id_surrogate["cases"][0]["case_id"] = "\ud800"
    assert validate_data(case_id_surrogate, "g1-benchmark-session")

    deep = load_json(EXAMPLE)
    deep["cases"][0]["region_ir"] = "[" * 10000 + "0" + "]" * 10000
    assert validate_data(deep, "g1-benchmark-session")


def test_degenerate_bca_is_rejected_without_a_point_interval() -> None:
    sessions = _session_set()
    for session in sessions:
        samples = session["cases"][0]["samples"]
        neon = {
            sample["pair_index"]: sample["elapsed_ns"]
            for sample in samples
            if sample["backend"] == "neon"
        }
        for sample in samples:
            if sample["backend"] == "scalar":
                sample["elapsed_ns"] = 2 * neon[sample["pair_index"]]
    with pytest.raises(G1AnalysisError, match="undefined"):
        analyze_sessions(sessions)


def test_artifact_provenance_mutations_and_cross_session_mismatch_reject() -> None:
    session = load_json(EXAMPLE)
    swapped_audit = deepcopy(session)
    swapped_audit["cases"][0]["scalar"]["artifact"]["audit"] = deepcopy(
        swapped_audit["cases"][0]["neon"]["artifact"]["audit"]
    )
    assert validate_data(swapped_audit, "g1-benchmark-session")

    bogus_flags = deepcopy(session)
    bogus_flags["cases"][0]["scalar"]["artifact"]["flags"][0] = "-O3"
    assert validate_data(bogus_flags, "g1-benchmark-session")

    bogus_source = deepcopy(session)
    bogus_source["cases"][0]["scalar"]["artifact"]["source"] = "changed"
    assert validate_data(bogus_source, "g1-benchmark-session")

    bogus_exports = deepcopy(session)
    bogus_exports["cases"][0]["neon"]["artifact"]["dynamic_exports"] = ["df_run_v1"]
    assert validate_data(bogus_exports, "g1-benchmark-session")

    ineffective_audit = deepcopy(session)
    ineffective_audit["cases"][0]["neon"]["artifact"]["audit"][
        "vector_path_observed"
    ] = False
    assert validate_data(ineffective_audit, "g1-benchmark-session")

    bogus_host = deepcopy(session)
    bogus_host["host"]["thread_policy"] = "workers"
    assert validate_data(bogus_host, "g1-benchmark-session")

    bogus_boundary = deepcopy(session)
    bogus_boundary["timing"]["boundary"] = "whole-program wall time"
    assert validate_data(bogus_boundary, "g1-benchmark-session")

    sessions = _session_set()
    sessions[1]["cases"][0]["scalar"]["artifact"]["compiler_version"] = "other"
    with pytest.raises(G1AnalysisError, match="schema|artifact|toolchain"):
        analyze_sessions(sessions)


def test_coherently_falsified_artifact_toolchain_and_content_reject() -> None:
    sessions = _session_set()
    fake_target = "arm64-apple-darwin99.9.9"
    fake_compiler = (
        "Apple clang version 99.0.0 (clang-9900.0.0) Target: "
        f"{fake_target} Thread model: posix InstalledDir: /usr/bin"
    )
    fake_objdump = (
        "Apple LLVM version 99.0.0 Optimized build. Registered Targets: "
        "aarch64 - AArch64 (little endian)"
    )
    for session in sessions:
        host = session["host"]
        host.update(
            {
                "os_version": "16.0",
                "os_build": "99Z99",
                "kernel_release": "99.9.9",
                "cpu_model": "Apple M5",
                "hardware_model": "Mac99,99",
                "physical_cores": 12,
                "logical_cores": 12,
            }
        )
        for backend in ("scalar", "neon"):
            artifact = session["cases"][0][backend]["artifact"]
            artifact["target"] = fake_target
            artifact["compiler_version"] = fake_compiler
            artifact["sdk_version"] = "99.9"
            artifact["objdump_version"] = fake_objdump
    with pytest.raises(G1AnalysisError, match="schema|artifact|target|host"):
        analyze_sessions(sessions)

    zero_digest = load_json(EXAMPLE)
    zero_digest["cases"][0]["scalar"]["artifact"]["dylib_hash"] = "sha256:" + "0" * 64
    assert validate_data(zero_digest, "g1-benchmark-session")

    forged_bundle = _session_set()
    for session in forged_bundle:
        session["case_bundle_identity"] = "sha256:" + "a" * 64
    with pytest.raises(G1AnalysisError, match="schema|bundle|manifest"):
        analyze_sessions(forged_bundle)


def test_real_artifact_source_disassembly_and_module_bindings_reject_forgery() -> None:
    sessions = _session_set()
    for session in sessions:
        artifact = session["cases"][0]["scalar"]["artifact"]
        old_module = artifact["module_id"]
        fake_module = "sha256:" + "a" * 64
        old_helper = f"df_kernel_scalar_v1_{old_module[7:]}"
        new_helper = f"df_kernel_scalar_v1_{fake_module[7:]}"
        artifact["module_id"] = fake_module
        artifact["source"] = artifact["source"].replace(old_helper, new_helper)
        artifact["disassembly"] = artifact["disassembly"].replace(
            old_helper, new_helper
        )
        artifact["source_hash"] = (
            "sha256:" + hashlib.sha256(artifact["source"].encode("utf-8")).hexdigest()
        )
        artifact["audit"]["helper_symbol"] = new_helper
    with pytest.raises(G1AnalysisError, match="schema|module|source"):
        analyze_sessions(sessions)

    sessions = _session_set()
    for session in sessions:
        artifact = session["cases"][0]["neon"]["artifact"]
        artifact["source"] += "\n/* forged provenance */\n"
        artifact["source_hash"] = (
            "sha256:" + hashlib.sha256(artifact["source"].encode("utf-8")).hexdigest()
        )
        artifact["disassembly"] += "\nforged instruction\n"
    with pytest.raises(G1AnalysisError, match="schema|source|disassembly"):
        analyze_sessions(sessions)


def test_mutations_reject_provenance_host_process_calibration_and_ir() -> None:
    session = load_json(EXAMPLE)
    ir = deepcopy(session)
    ir["cases"][0]["region_ir"] = '{"schema_version":1}'
    assert validate_data(ir, "g1-benchmark-session")

    noncanonical_ir = deepcopy(session)
    noncanonical_ir["cases"][0]["region_ir"] = (
        ' {"schema_version":1,"operator":"q8_linear"}'
    )
    assert validate_data(noncanonical_ir, "g1-benchmark-session")

    repetitions = deepcopy(session)
    repetitions["cases"][0]["scalar"]["calibration"]["attempts"] = [
        {"elapsed_ns": 1, "repetitions": 1},
        {"elapsed_ns": CALIBRATION_TARGET_NS, "repetitions": 4},
    ]
    repetitions["cases"][0]["scalar"]["calibration"]["selected_repetitions"] = 4
    assert validate_data(repetitions, "g1-benchmark-session")

    sessions = _session_set()
    sessions[1]["host"]["cpu_model"] = "different-host"
    with pytest.raises(G1AnalysisError, match="host fingerprint"):
        analyze_sessions(sessions)

    sessions = _session_set()
    sessions[2]["host"]["process_id"] = sessions[1]["host"]["process_id"]
    with pytest.raises(G1AnalysisError, match="distinct process_id"):
        analyze_sessions(sessions)

    sessions = _session_set()
    sessions[0]["checkout"]["dirty"] = True
    with pytest.raises(G1AnalysisError, match="clean checkout"):
        analyze_sessions(sessions)


def test_load_sessions_rejects_symlink_and_oversize_inputs(tmp_path: Path) -> None:
    content = EXAMPLE.read_bytes()
    regular = [tmp_path / f"session-{index}.json" for index in range(3)]
    for path in regular:
        path.write_bytes(content)
    from decodeforge.g1_evidence import load_sessions

    assert len(load_sessions(regular)) == 3
    link = tmp_path / "session-link.json"
    link.symlink_to(regular[0])
    with pytest.raises(G1AnalysisError, match="non-symlink"):
        load_sessions([link, regular[1], regular[2]])

    oversized = tmp_path / "session-oversized.json"
    oversized.write_bytes(b"{" + b" " * MAX_SESSION_JSON_BYTES + b"}")
    with pytest.raises(G1AnalysisError, match="bound"):
        load_sessions([oversized, regular[1], regular[2]])


def test_duplicate_json_key_is_rejected_by_schema_path(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    diagnostics = validate_path(path, "g1-benchmark-session")
    assert diagnostics[0]["code"] == "DFE-SCHEMA-008"

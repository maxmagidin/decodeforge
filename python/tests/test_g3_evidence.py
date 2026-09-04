"""Directed semantic checks for the G3 generation-session contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from decodeforge.contracts import load_json, validate_data, validate_path

ROOT = Path(__file__).resolve().parents[2]
SESSION_TEMPLATE = ROOT / "benchmarks" / "g3" / "session-template.json"
ASSET_INVENTORY = ROOT / "tests" / "fixtures" / "g3" / "asset-inventory.json"

JsonObject = dict[str, Any]

_NUMERIC_FIELDS = (
    "forward",
    "native_attempt",
    "native_success",
    "native_error",
    "fallback_attempt",
    "fallback_success",
    "fallback_error",
    "predispatch_error",
    "rejected_closed",
    "in_flight",
)


def _identity(value: int) -> str:
    return f"sha256:{value:064x}"


def _layer_path(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.q_proj"


def _counter_values() -> JsonObject:
    return {
        **dict.fromkeys(_NUMERIC_FIELDS, 0),
        "closed": False,
    }


def _snapshot(layer: int, values: JsonObject) -> JsonObject:
    return {
        "layer": layer,
        "layer_path": _layer_path(layer),
        "values": copy.deepcopy(values),
    }


def _run(
    *,
    phase: str,
    repetition: int,
    order_index: int,
    path: str,
    states: list[JsonObject],
) -> JsonObject:
    output_ids = [100, 101]
    before = [_snapshot(layer, values) for layer, values in enumerate(states)]
    native = 1 if path == "hybrid_native" else 0
    fallback = 1 if path == "hybrid_native" else 2
    delta_values = {
        "forward": 2,
        "native_attempt": native,
        "native_success": native,
        "native_error": 0,
        "fallback_attempt": fallback,
        "fallback_success": fallback,
        "fallback_error": 0,
        "predispatch_error": 0,
        "rejected_closed": 0,
        "in_flight": 0,
        "closed_changed": False,
    }
    for values in states:
        for field in _NUMERIC_FIELDS:
            values[field] += delta_values[field]
    after = [_snapshot(layer, values) for layer, values in enumerate(states)]
    delta = [
        {
            "layer": layer,
            "layer_path": _layer_path(layer),
            "values": copy.deepcopy(delta_values),
        }
        for layer in range(22)
    ]
    steps = [
        {
            "step_index": step_index,
            "token_id": token_id,
            "logits_sha256": f"{order_index * 8 + step_index + 1:064x}",
            "logit_count": 32000,
            "finite": True,
            "hybrid_comparison": (
                {
                    "max_abs": 0.0,
                    "max_allowed": 0.001,
                    "max_excess": -0.001,
                    "pass": True,
                }
                if path == "hybrid_native"
                else None
            ),
        }
        for step_index, token_id in enumerate(output_ids)
    ]
    native_output_checks = (
        [
            {
                "layer": layer,
                "layer_path": _layer_path(layer),
                "step_index": 1,
                "result": {
                    "max_abs": 0.0,
                    "max_allowed": 0.0001,
                    "max_excess": -0.0001,
                    "finite": True,
                    "pass": True,
                },
            }
            for layer in range(22)
        ]
        if path == "hybrid_native"
        else []
    )
    return {
        "phase": phase,
        "repetition": repetition,
        "order_index": order_index,
        "path": path,
        "output_ids": output_ids,
        "decoded_text": " synthetic",
        "text_role": "demonstration_only_not_correctness_evidence",
        "steps": steps,
        "native_output_checks": native_output_checks,
        "timing": {
            "prefill_ns": 100,
            "time_to_first_token_ns": 110,
            "cached_step_ns": [50],
            "q_projection_dispatches": [
                {
                    "step_index": step_index,
                    "layer": layer,
                    "layer_path": _layer_path(layer),
                    "dispatch": (
                        "native"
                        if path == "hybrid_native" and step_index > 0
                        else "fallback"
                    ),
                    "dispatch_ns": 1,
                }
                for step_index in range(len(output_ids))
                for layer in range(22)
            ],
            "q_projection_ns": 44,
            "native_work_ns": None,
            "native_work_unavailable_reason": "bridge exposes no kernel-only timer",
            "total_ns": 200,
        },
        "counters": {"before": before, "after": after, "delta": delta},
    }


def _accepted_session() -> JsonObject:
    session = copy.deepcopy(load_json(SESSION_TEMPLATE))
    inventory = load_json(ASSET_INVENTORY)
    entries = [
        {**entry, "layer_path": _layer_path(layer)}
        for layer, entry in enumerate(inventory["entries"])
    ]
    aggregate_identity = inventory["aggregate_identity"]
    states = [_counter_values() for _ in range(22)]
    for values in states:
        values["forward"] = 1
        values["native_attempt"] = 1
        values["native_success"] = 1
    runs: list[JsonObject] = []
    order_index = 0
    for phase, repetitions in (("warmup", 2), ("measured", 10)):
        for repetition in range(repetitions):
            paths = (
                ("same_q8_reference", "hybrid_native")
                if repetition % 2 == 0
                else ("hybrid_native", "same_q8_reference")
            )
            for path in paths:
                runs.append(
                    _run(
                        phase=phase,
                        repetition=repetition,
                        order_index=order_index,
                        path=path,
                        states=states,
                    )
                )
                order_index += 1
    post_before = [_snapshot(layer, values) for layer, values in enumerate(states)]
    post_delta_values = {
        "forward": 12,
        "native_attempt": 0,
        "native_success": 0,
        "native_error": 0,
        "fallback_attempt": 12,
        "fallback_success": 12,
        "fallback_error": 0,
        "predispatch_error": 0,
        "rejected_closed": 0,
        "in_flight": 0,
        "closed_changed": False,
    }
    for values in states:
        for field in _NUMERIC_FIELDS:
            values[field] += post_delta_values[field]
    post_after = [_snapshot(layer, values) for layer, values in enumerate(states)]
    post_delta = [
        {
            "layer": layer,
            "layer_path": _layer_path(layer),
            "values": copy.deepcopy(post_delta_values),
        }
        for layer in range(22)
    ]
    direct_layers = [
        {
            "layer": layer,
            "layer_path": _layer_path(layer),
            "result": {
                "max_abs": 0.0,
                "max_allowed": 0.0001,
                "max_excess": -0.0001,
                "finite": True,
                "pass": True,
            },
        }
        for layer in range(22)
    ]
    session.update(
        {
            "session_id": "synthetic-accepted-contract-test",
            "state": {
                "status": "accepted",
                "stage": "completed",
                "accepted": True,
                "rejection_reasons": [],
            },
            "provenance": {
                "checkout": {"revision": "1" * 40, "dirty": False},
                "model": {
                    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
                    "filename": "model.safetensors",
                    "bytes": 2200119864,
                    "identity": (
                        "sha256:"
                        "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
                    ),
                },
                "bridge_library": {
                    "path": "target/release/libdecodeforge_bridge.dylib",
                    "size_bytes": 1048576,
                    "sha256": "9" * 64,
                },
                "asset_inventory_identity": aggregate_identity,
                "rebuild_commands": {
                    "build_bridge": (
                        "cargo build --release --locked -p decodeforge-bridge"
                    ),
                    "prepare_assets": (
                        "make prepare-g3-assets WEIGHTS=<pinned> OUTPUT=<assets>"
                    ),
                    "run_session": "make run-g3-demo ASSETS=<assets> OUTPUT=<session>",
                },
            },
            "environment": {
                "software": {
                    "python": "3.12.14",
                    "numpy": "2.4.4",
                    "torch": "2.13.0",
                    "transformers": "5.12.1",
                    "tokenizers": "0.22.2",
                    "safetensors": "0.6.2",
                    "rust": "1.98.0",
                    "clang": "Apple clang version 17.0.0 (clang-1700.0.13.5)",
                },
                "tokenizer": {
                    "class": "LlamaTokenizer",
                    "is_fast": True,
                    "method": "AutoTokenizer.from_pretrained-local-files-only",
                },
                "model": {
                    "class": "LlamaForCausalLM",
                    "device": "cpu",
                    "dtype": "float32",
                    "eval_mode": True,
                },
                "host": {
                    "host_id": "apple-m4-primary",
                    "os": "macos",
                    "os_version": "15.5",
                    "os_build": "24F74",
                    "kernel_release": "24.5.0",
                    "arch": "aarch64",
                    "cpu_model": "Apple M4",
                    "hardware_model": "Mac16,13",
                    "physical_cores": 10,
                    "logical_cores": 10,
                    "features": ["neon"],
                    "affinity_policy": (
                        "macOS default scheduler; no hard affinity requested"
                    ),
                },
                "runner_process_id": 1001,
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
                "hf_hub_offline": True,
                "transformers_offline": True,
                "local_files_only": True,
            },
            "assets": {
                "format": inventory["format"],
                "source": inventory["source"],
                "aggregate_identity": aggregate_identity,
                "layer_count": 22,
                "total_packed_bytes": 103809024,
                "total_fallback_bytes": 369098752,
                "entries": entries,
            },
            "offline_preparation": {
                "source": "separately_captured_prepare_command",
                "receipt_identity": _identity(8000),
                "elapsed_ns": 1000,
                "asset_inventory_identity": aggregate_identity,
            },
            "startup_timings": {
                "cold_startup_ns": 300,
                "tokenizer_load_ns": 50,
                "model_load_ns": 100,
                "install_ns": 100,
                "baseline_rss_bytes": 1000,
                "peak_rss_bytes": 2000,
            },
            "runs": runs,
            "correctness": {
                "direct_operator": {
                    "absolute_tolerance": 0.0001,
                    "relative_tolerance": 0.0001,
                    "preinstallation_layers": direct_layers,
                    "native_output_check_count": 264,
                    "pass": True,
                },
                "model_logits": {
                    "absolute_tolerance": 0.001,
                    "relative_tolerance": 0.001,
                    "comparison_count": 24,
                    "pass": True,
                },
                "tokens": {
                    "policy": "exact_token_id_match",
                    "comparison_count": 12,
                    "pass": True,
                },
                "overall_pass": True,
            },
            "reconciliation": {
                "post_run_native_validation_counters": {
                    "before": post_before,
                    "after": post_after,
                    "delta": post_delta,
                },
                "installation": {
                    "installed_modules": 22,
                    "restored_modules": 22,
                    "live_adapters": 0,
                    "in_flight": 0,
                    "closed": True,
                },
                "paths": [
                    {
                        "path": path,
                        "warmup_runs": 2,
                        "measured_runs": 10,
                        "prefill_calls_per_layer": 12,
                        "cached_decode_calls_per_layer": 12,
                        "total_calls_per_layer": 24,
                        "pass": True,
                    }
                    for path in ("same_q8_reference", "hybrid_native")
                ],
                "pass": True,
            },
            "drift": {
                "metric": (
                    "ratio_of_last_window_median_to_first_window_median_"
                    "total_generation_ns_per_path"
                ),
                "window_generations": 3,
                "ratio_lower": 0.9,
                "ratio_upper": 1.1,
                "paths": [
                    {
                        "path": path,
                        "first_window_median_ns": 200.0,
                        "last_window_median_ns": 200.0,
                        "ratio": 1.0,
                        "pass": True,
                    }
                    for path in ("same_q8_reference", "hybrid_native")
                ],
                "pass": True,
            },
        }
    )
    return session


def _semantic_path(instance: JsonObject) -> list[str | int]:
    diagnostics = validate_data(instance, "g3-generation-session")
    assert len(diagnostics) == 1
    return diagnostics[0]["context"]["path"]  # type: ignore[no-any-return]


def test_canonical_session_template_is_valid_and_explicitly_not_run() -> None:
    assert validate_path(SESSION_TEMPLATE, "g3-generation-session") == []
    template = load_json(SESSION_TEMPLATE)
    assert template["state"] == {
        "status": "not_run",
        "stage": "not_run",
        "accepted": False,
        "rejection_reasons": ["not_run"],
    }


def test_synthetic_accepted_session_reconciles() -> None:
    assert validate_data(_accepted_session(), "g3-generation-session") == []


def test_rejected_preflight_session_can_retain_no_measurements() -> None:
    session = copy.deepcopy(load_json(SESSION_TEMPLATE))
    session["state"] = {
        "status": "rejected",
        "stage": "preflight",
        "accepted": False,
        "rejection_reasons": ["environment_mismatch"],
    }
    assert validate_data(session, "g3-generation-session") == []


def test_asset_order_is_semantically_closed() -> None:
    session = _accepted_session()
    session["assets"]["entries"][0], session["assets"]["entries"][1] = (
        session["assets"]["entries"][1],
        session["assets"]["entries"][0],
    )
    assert _semantic_path(session) == ["assets", "entries", 0]


def test_asset_aggregate_is_recomputed_from_canonical_preimage() -> None:
    session = _accepted_session()
    session["assets"]["entries"][0]["tensor_identity"] = _identity(9999)
    assert _semantic_path(session) == ["assets", "aggregate_identity"]


def test_asset_preimage_does_not_depend_on_json_object_key_order() -> None:
    session = _accepted_session()
    source = session["assets"]["source"]
    session["assets"]["source"] = dict(reversed(tuple(source.items())))
    assert validate_data(session, "g3-generation-session") == []


def test_offline_receipt_is_bound_to_the_asset_inventory() -> None:
    session = _accepted_session()
    session["offline_preparation"]["asset_inventory_identity"] = _identity(9999)
    paths = [
        diagnostic["context"]["path"]
        for diagnostic in validate_data(session, "g3-generation-session")
    ]
    assert ["offline_preparation", "asset_inventory_identity"] in paths


def test_predispatch_error_cannot_hide_in_an_accepted_delta() -> None:
    session = _accepted_session()
    delta = session["runs"][0]["counters"]["delta"][0]["values"]
    delta["predispatch_error"] = 1
    assert _semantic_path(session) == [
        "runs",
        0,
        "counters",
        "delta",
        0,
        "values",
        "predispatch_error",
    ]


def test_logit_pass_is_derived_from_retained_maxima() -> None:
    session = _accepted_session()
    comparison = session["runs"][1]["steps"][0]["hybrid_comparison"]
    comparison["pass"] = False
    assert _semantic_path(session) == [
        "runs",
        1,
        "steps",
        0,
        "hybrid_comparison",
        "pass",
    ]


def test_native_output_checks_are_step_then_layer_ordered() -> None:
    session = _accepted_session()
    check = session["runs"][1]["native_output_checks"][0]
    check["layer"] = 1
    check["layer_path"] = _layer_path(1)
    assert _semantic_path(session) == ["runs", 1, "native_output_checks", 0]


def test_native_output_check_count_is_derived_from_all_hybrid_runs() -> None:
    session = _accepted_session()
    session["correctness"]["direct_operator"]["native_output_check_count"] = 265
    assert _semantic_path(session) == ["correctness", "direct_operator", "pass"]


def test_paired_output_token_mismatch_is_rejected() -> None:
    session = _accepted_session()
    hybrid = session["runs"][1]
    hybrid["output_ids"][1] = 102
    hybrid["steps"][1]["token_id"] = 102
    assert _semantic_path(session) == ["runs"]


def test_inclusive_total_timing_contains_raw_steps() -> None:
    session = _accepted_session()
    session["runs"][0]["timing"]["total_ns"] = 149
    assert _semantic_path(session) == ["runs", 0, "timing"]


def test_accepted_q_projection_timing_is_not_nullable() -> None:
    session = _accepted_session()
    session["runs"][0]["timing"]["q_projection_ns"] = None
    assert _semantic_path(session) == ["runs", 0, "timing"]


def test_q_projection_dispatches_are_step_then_layer_ordered() -> None:
    session = _accepted_session()
    dispatches = session["runs"][0]["timing"]["q_projection_dispatches"]
    dispatches[0], dispatches[1] = dispatches[1], dispatches[0]
    assert _semantic_path(session) == [
        "runs",
        0,
        "timing",
        "q_projection_dispatches",
        0,
    ]


def test_q_projection_aggregate_equals_raw_dispatch_samples() -> None:
    session = _accepted_session()
    session["runs"][0]["timing"]["q_projection_ns"] = 45
    assert _semantic_path(session) == ["runs", 0, "timing"]


def test_q_projection_aggregate_is_contained_by_total_timing() -> None:
    session = _accepted_session()
    timing = session["runs"][0]["timing"]
    for dispatch in timing["q_projection_dispatches"]:
        dispatch["dispatch_ns"] = 5
    timing["q_projection_ns"] = 220
    assert _semantic_path(session) == ["runs", 0, "timing"]


def test_counter_continuity_includes_warmup_to_measured_boundary() -> None:
    session = _accepted_session()
    boundary_run = session["runs"][4]
    for side in ("before", "after"):
        values = boundary_run["counters"][side][0]["values"]
        values["forward"] += 1
        values["fallback_attempt"] += 1
        values["fallback_success"] += 1
    assert _semantic_path(session) == ["runs", 4, "counters", "before"]


def test_cold_startup_contains_tokenizer_model_and_installation() -> None:
    session = _accepted_session()
    session["startup_timings"]["cold_startup_ns"] = 249
    assert _semantic_path(session) == ["startup_timings"]


def test_post_run_validation_counter_errors_are_rejected() -> None:
    session = _accepted_session()
    delta = session["reconciliation"]["post_run_native_validation_counters"]["delta"][
        0
    ]["values"]
    delta["predispatch_error"] = 1
    assert _semantic_path(session) == [
        "reconciliation",
        "post_run_native_validation_counters",
        "delta",
        0,
        "values",
        "predispatch_error",
    ]


def test_drift_summary_is_recomputed_from_measured_runs() -> None:
    session = _accepted_session()
    session["drift"]["paths"][0]["ratio"] = 1.01
    assert _semantic_path(session) == ["drift", "paths", 0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_comparison_evidence_is_rejected(value: float) -> None:
    session = _accepted_session()
    session["correctness"]["direct_operator"]["preinstallation_layers"][0]["result"][
        "max_excess"
    ] = value
    diagnostics = validate_data(session, "g3-generation-session")
    assert diagnostics
    assert diagnostics[0]["context"]["path"] == [
        "correctness",
        "direct_operator",
        "preinstallation_layers",
        0,
        "result",
        "max_excess",
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_drift_evidence_is_rejected(value: float) -> None:
    session = _accepted_session()
    session["drift"]["paths"][0]["ratio"] = value
    diagnostics = validate_data(session, "g3-generation-session")
    assert diagnostics
    assert diagnostics[0]["context"]["path"] == ["drift", "paths", 0, "ratio"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timing_evidence_is_rejected(value: float) -> None:
    session = _accepted_session()
    session["startup_timings"]["cold_startup_ns"] = value
    diagnostics = validate_data(session, "g3-generation-session")
    assert diagnostics
    assert diagnostics[0]["context"]["path"] == [
        "startup_timings",
        "cold_startup_ns",
    ]

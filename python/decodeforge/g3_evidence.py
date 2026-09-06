"""Semantic validation for the closed G3 generation-session contract."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeAlias, cast

JsonObject: TypeAlias = dict[str, Any]
Diagnostic: TypeAlias = dict[str, Any]

_DIAGNOSTIC_CODE: Final = "DFE-SCHEMA-007"
_CANONICAL_ASSET_INVENTORY_IDENTITY: Final = (
    "sha256:f659b26572357af84a5e5b66138331a2e35c319c5c9b8300cf81f7ea217ae0de"
)
_CANONICAL_MODULE_IDENTITY: Final = (
    "sha256:564dbd74857d3fe00b25bf4acbe6cf06d6ff47ab603ae9eb1ba3a530edc8ea44"
)
_PATHS: Final = ("same_q8_reference", "hybrid_native")
_NUMERIC_COUNTER_FIELDS: Final = (
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
_IDENTITY_FIELDS: Final = (
    "manifest_identity",
    "tensor_identity",
    "logical_weight_identity",
    "packed_weight_identity",
    "fallback_identity",
)


def _diagnostic(path: Sequence[str | int], reason: str) -> Diagnostic:
    return {
        "schema_version": 1,
        "code": _DIAGNOSTIC_CODE,
        "severity": "error",
        "component": "schema",
        "summary": "The G3 generation session is not semantically consistent.",
        "context": {"path": list(path), "reason": reason},
    }


def _layer_path(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.q_proj"


def _ordered_run_keys() -> list[tuple[str, int, str]]:
    result: list[tuple[str, int, str]] = []
    for phase, repetitions in (("warmup", 2), ("measured", 10)):
        for repetition in range(repetitions):
            order = _PATHS if repetition % 2 == 0 else tuple(reversed(_PATHS))
            result.extend((phase, repetition, path) for path in order)
    return result


def _nonfinite_path(instance: Mapping[str, Any]) -> list[str | int] | None:
    stack: list[tuple[Any, list[str | int]]] = [(instance, [])]
    while stack:
        value, path = stack.pop()
        if isinstance(value, float) and not math.isfinite(value):
            return path
        if isinstance(value, dict):
            stack.extend(
                (item, [*path, key]) for key, item in reversed(tuple(value.items()))
            )
        elif isinstance(value, list):
            stack.extend(
                (item, [*path, index])
                for index, item in reversed(tuple(enumerate(value)))
            )
    return None


def _validate_assets(
    assets: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
) -> Diagnostic | None:
    entries = cast(list[JsonObject], assets["entries"])
    identity_sets: dict[str, set[str]] = {field: set() for field in _IDENTITY_FIELDS}
    packed_bytes = 0
    fallback_bytes = 0
    for layer, entry in enumerate(entries):
        expected_path = _layer_path(layer)
        expected_tensor = f"{expected_path}.weight"
        if (
            entry["layer"] != layer
            or entry["directory"] != f"layers/{layer:02}"
            or entry["layer_path"] != expected_path
            or entry["tensor_name"] != expected_tensor
        ):
            return _diagnostic(
                ["assets", "entries", layer],
                "asset entries must be the exact ordered TinyLlama q_proj layers",
            )
        for field in _IDENTITY_FIELDS:
            identity = cast(str, entry[field])
            if identity in identity_sets[field]:
                return _diagnostic(
                    ["assets", "entries", layer, field],
                    "each layer must retain a distinct asset identity",
                )
            identity_sets[field].add(identity)
        if entry["module_identity"] != _CANONICAL_MODULE_IDENTITY:
            return _diagnostic(
                ["assets", "entries", layer, "module_identity"],
                "layer does not use the canonical shared generated module",
            )
        packed_bytes += cast(int, entry["packed_bytes"])
        fallback_bytes += cast(int, entry["fallback_bytes"])
    if packed_bytes != assets["total_packed_bytes"]:
        return _diagnostic(
            ["assets", "total_packed_bytes"],
            "aggregate packed bytes do not equal the ordered entries",
        )
    if fallback_bytes != assets["total_fallback_bytes"]:
        return _diagnostic(
            ["assets", "total_fallback_bytes"],
            "aggregate fallback bytes do not equal the ordered entries",
        )
    source = cast(JsonObject, assets["source"])
    inventory_source = {
        "model_id": source["model_id"],
        "revision": source["revision"],
        "filename": source["filename"],
        "bytes": source["bytes"],
        "identity": source["identity"],
    }
    inventory_entries = [
        {
            "layer": entry["layer"],
            "directory": entry["directory"],
            "manifest_identity": entry["manifest_identity"],
            "tensor_name": entry["tensor_name"],
            "tensor_identity": entry["tensor_identity"],
            "logical_weight_identity": entry["logical_weight_identity"],
            "packed_weight_identity": entry["packed_weight_identity"],
            "packed_bytes": entry["packed_bytes"],
            "module_identity": entry["module_identity"],
            "fallback_identity": entry["fallback_identity"],
            "fallback_bytes": entry["fallback_bytes"],
        }
        for entry in entries
    ]
    preimage = {
        "schema_version": 1,
        "format": assets["format"],
        "source": inventory_source,
        "layer_count": assets["layer_count"],
        "entries": inventory_entries,
        "total_packed_bytes": assets["total_packed_bytes"],
        "total_fallback_bytes": assets["total_fallback_bytes"],
    }
    encoded = json.dumps(preimage, separators=(",", ":"), ensure_ascii=True).encode()
    observed_identity = (
        "sha256:"
        + hashlib.sha256(b"DecodeForge/q-proj-inventory/v1\0" + encoded).hexdigest()
    )
    if (
        assets["aggregate_identity"] != observed_identity
        or assets["aggregate_identity"] != _CANONICAL_ASSET_INVENTORY_IDENTITY
    ):
        return _diagnostic(
            ["assets", "aggregate_identity"],
            "asset aggregate identity is not the canonical recomputed preimage",
        )
    if (
        provenance is not None
        and provenance["asset_inventory_identity"] != assets["aggregate_identity"]
    ):
        return _diagnostic(
            ["provenance", "asset_inventory_identity"],
            "provenance does not name the retained asset inventory",
        )
    return None


def _counter_invariants(values: Mapping[str, Any]) -> bool:
    return bool(
        values["forward"]
        == values["native_attempt"]
        + values["fallback_attempt"]
        + values["predispatch_error"]
        and values["native_attempt"]
        == values["native_success"] + values["native_error"]
        and values["fallback_attempt"]
        == values["fallback_success"] + values["fallback_error"]
    )


def _validate_run_counters(
    run: Mapping[str, Any],
    run_index: int,
    *,
    require_acceptance: bool,
) -> Diagnostic | None:
    counters = cast(JsonObject, run["counters"])
    before = cast(list[JsonObject], counters["before"])
    after = cast(list[JsonObject], counters["after"])
    delta = cast(list[JsonObject], counters["delta"])
    output_count = len(cast(list[Any], run["output_ids"]))
    path = cast(str, run["path"])
    for layer in range(22):
        records = (before[layer], after[layer], delta[layer])
        expected_path = _layer_path(layer)
        if any(
            record["layer"] != layer or record["layer_path"] != expected_path
            for record in records
        ):
            return _diagnostic(
                ["runs", run_index, "counters"],
                "counter snapshots must retain exact ordered layer identities",
            )
        before_values = cast(JsonObject, before[layer]["values"])
        after_values = cast(JsonObject, after[layer]["values"])
        delta_values = cast(JsonObject, delta[layer]["values"])
        if not _counter_invariants(before_values) or not _counter_invariants(
            after_values
        ):
            return _diagnostic(
                ["runs", run_index, "counters", "after", layer, "values"],
                "adapter counter partition invariants do not reconcile",
            )
        for field in _NUMERIC_COUNTER_FIELDS:
            observed = cast(int, after_values[field]) - cast(int, before_values[field])
            if delta_values[field] != observed or observed < 0:
                return _diagnostic(
                    ["runs", run_index, "counters", "delta", layer, "values", field],
                    "counter delta must equal the nonnegative after-minus-before value",
                )
        if delta_values["closed_changed"] != (
            after_values["closed"] != before_values["closed"]
        ):
            return _diagnostic(
                [
                    "runs",
                    run_index,
                    "counters",
                    "delta",
                    layer,
                    "values",
                    "closed_changed",
                ],
                "closed_changed does not match the snapshots",
            )
        if not require_acceptance:
            continue
        expected_native = output_count - 1 if path == "hybrid_native" else 0
        expected_fallback = 1 if path == "hybrid_native" else output_count
        expected = {
            "forward": output_count,
            "native_attempt": expected_native,
            "native_success": expected_native,
            "native_error": 0,
            "fallback_attempt": expected_fallback,
            "fallback_success": expected_fallback,
            "fallback_error": 0,
            "predispatch_error": 0,
            "rejected_closed": 0,
            "in_flight": 0,
        }
        if any(delta_values[field] != value for field, value in expected.items()):
            return _diagnostic(
                ["runs", run_index, "counters", "delta", layer, "values"],
                "accepted coverage does not match prefill/decode dispatch policy",
            )
        if (
            before_values["closed"]
            or after_values["closed"]
            or delta_values["closed_changed"]
            or after_values["in_flight"] != 0
        ):
            return _diagnostic(
                ["runs", run_index, "counters", "after", layer, "values"],
                "accepted run boundaries require open adapters and zero "
                "in-flight calls",
            )
    return None


def _validate_runs(
    runs: list[JsonObject],
    *,
    require_acceptance: bool,
) -> tuple[Diagnostic | None, dict[tuple[str, int, str], JsonObject]]:
    expected_keys = _ordered_run_keys()
    by_key: dict[tuple[str, int, str], JsonObject] = {}
    previous_after: list[JsonObject] | None = None
    for index, run in enumerate(runs):
        key = (
            cast(str, run["phase"]),
            cast(int, run["repetition"]),
            cast(str, run["path"]),
        )
        if run["order_index"] != index or key != expected_keys[index]:
            return (
                _diagnostic(
                    ["runs", index],
                    "runs must follow the frozen alternating warmup/measured order",
                ),
                {},
            )
        by_key[key] = run
        output_ids = cast(list[int], run["output_ids"])
        steps = cast(list[JsonObject], run["steps"])
        cached = cast(list[int], cast(JsonObject, run["timing"])["cached_step_ns"])
        if len(steps) != len(output_ids) or len(cached) != len(output_ids) - 1:
            return (
                _diagnostic(
                    ["runs", index, "steps"],
                    "step and cached-timing counts must follow generated token count",
                ),
                {},
            )
        for step_index, (token_id, step) in enumerate(
            zip(output_ids, steps, strict=True)
        ):
            if step["step_index"] != step_index or step["token_id"] != token_id:
                return (
                    _diagnostic(
                        ["runs", index, "steps", step_index],
                        "step index/token must reproduce output_ids",
                    ),
                    {},
                )
            comparison = step["hybrid_comparison"]
            if run["path"] == "same_q8_reference":
                if comparison is not None:
                    return (
                        _diagnostic(
                            ["runs", index, "steps", step_index, "hybrid_comparison"],
                            "reference steps cannot contain a hybrid comparison",
                        ),
                        {},
                    )
            elif not isinstance(comparison, dict):
                return (
                    _diagnostic(
                        ["runs", index, "steps", step_index, "hybrid_comparison"],
                        "hybrid steps require the paired logit comparison",
                    ),
                    {},
                )
            elif comparison["pass"] != (comparison["max_excess"] <= 0):
                return (
                    _diagnostic(
                        [
                            "runs",
                            index,
                            "steps",
                            step_index,
                            "hybrid_comparison",
                            "pass",
                        ],
                        "logit pass must be derived from max_excess",
                    ),
                    {},
                )
            if require_acceptance and (
                not step["finite"]
                or (isinstance(comparison, dict) and not comparison["pass"])
            ):
                return (
                    _diagnostic(
                        ["runs", index, "steps", step_index],
                        "accepted sessions require finite, passing step evidence",
                    ),
                    {},
                )
        native_checks = cast(list[JsonObject], run["native_output_checks"])
        if run["path"] == "same_q8_reference":
            if native_checks:
                return (
                    _diagnostic(
                        ["runs", index, "native_output_checks"],
                        "reference runs cannot contain native output checks",
                    ),
                    {},
                )
        else:
            expected_checks = [
                (step_index, layer)
                for step_index in range(1, len(output_ids))
                for layer in range(22)
            ]
            if len(native_checks) != len(expected_checks):
                return (
                    _diagnostic(
                        ["runs", index, "native_output_checks"],
                        "every hybrid native layer output requires a direct check",
                    ),
                    {},
                )
            for check_index, (check, expected) in enumerate(
                zip(native_checks, expected_checks, strict=True)
            ):
                step_index, layer = expected
                result = cast(JsonObject, check["result"])
                derived = result["finite"] and result["max_excess"] <= 0
                if (
                    check["step_index"] != step_index
                    or check["layer"] != layer
                    or check["layer_path"] != _layer_path(layer)
                    or result["pass"] != derived
                    or (require_acceptance and not derived)
                ):
                    return (
                        _diagnostic(
                            ["runs", index, "native_output_checks", check_index],
                            "native output check is missing, unordered, "
                            "or inconsistent",
                        ),
                        {},
                    )
        timing = cast(JsonObject, run["timing"])
        dispatches = cast(list[JsonObject], timing["q_projection_dispatches"])
        expected_dispatches = [
            (
                step_index,
                layer,
                "native"
                if run["path"] == "hybrid_native" and step_index > 0
                else "fallback",
            )
            for step_index in range(len(output_ids))
            for layer in range(22)
        ]
        if len(dispatches) != len(expected_dispatches):
            return (
                _diagnostic(
                    ["runs", index, "timing", "q_projection_dispatches"],
                    "every q_proj call requires one ordered dispatch timing sample",
                ),
                {},
            )
        for dispatch_index, (dispatch, expected_dispatch) in enumerate(
            zip(dispatches, expected_dispatches, strict=True)
        ):
            step_index, layer, mode = expected_dispatch
            if (
                dispatch["step_index"] != step_index
                or dispatch["layer"] != layer
                or dispatch["layer_path"] != _layer_path(layer)
                or dispatch["dispatch"] != mode
            ):
                return (
                    _diagnostic(
                        [
                            "runs",
                            index,
                            "timing",
                            "q_projection_dispatches",
                            dispatch_index,
                        ],
                        "q_proj timing samples must be ordered step then layer and "
                        "name the executed dispatch",
                    ),
                    {},
                )
        q_projection_ns = timing["q_projection_ns"]
        if (
            timing["time_to_first_token_ns"] < timing["prefill_ns"]
            or timing["total_ns"] < timing["time_to_first_token_ns"] + sum(cached)
            or (require_acceptance and q_projection_ns is None)
            or (
                q_projection_ns is not None
                and (
                    q_projection_ns
                    != sum(
                        cast(int, dispatch["dispatch_ns"]) for dispatch in dispatches
                    )
                    or q_projection_ns > timing["total_ns"]
                )
            )
            or (
                (timing["native_work_ns"] is None)
                == (timing["native_work_unavailable_reason"] is None)
            )
        ):
            return (
                _diagnostic(
                    ["runs", index, "timing"],
                    "inclusive timings do not contain or reconcile their raw samples",
                ),
                {},
            )
        counter_error = _validate_run_counters(
            run, index, require_acceptance=require_acceptance
        )
        if counter_error is not None:
            return counter_error, {}
        current_before = cast(
            list[JsonObject], cast(JsonObject, run["counters"])["before"]
        )
        if require_acceptance and index == 0:
            for layer, snapshot in enumerate(current_before):
                values = cast(JsonObject, snapshot["values"])
                expected_probe = {
                    "forward": 1,
                    "native_attempt": 1,
                    "native_success": 1,
                    "native_error": 0,
                    "fallback_attempt": 0,
                    "fallback_success": 0,
                    "fallback_error": 0,
                    "predispatch_error": 0,
                    "rejected_closed": 0,
                    "in_flight": 0,
                }
                if (
                    any(
                        values[field] != value
                        for field, value in expected_probe.items()
                    )
                    or values["closed"]
                ):
                    return (
                        _diagnostic(
                            ["runs", 0, "counters", "before", layer, "values"],
                            "first baseline must account for one native "
                            "preinstall probe",
                        ),
                        {},
                    )
        if previous_after is not None and current_before != previous_after:
            return (
                _diagnostic(
                    ["runs", index, "counters", "before"],
                    "counter snapshots are not continuous across consecutive runs",
                ),
                {},
            )
        previous_after = cast(
            list[JsonObject], cast(JsonObject, run["counters"])["after"]
        )
    for phase, repetitions in (("warmup", 2), ("measured", 10)):
        for repetition in range(repetitions):
            reference = by_key[(phase, repetition, "same_q8_reference")]
            hybrid = by_key[(phase, repetition, "hybrid_native")]
            if reference["output_ids"] != hybrid["output_ids"]:
                return (
                    _diagnostic(
                        ["runs"],
                        "paired reference and hybrid runs must emit exact token IDs",
                    ),
                    {},
                )
    return None, by_key


def _validate_correctness(
    correctness: Mapping[str, Any],
    by_key: Mapping[tuple[str, int, str], JsonObject],
    *,
    require_acceptance: bool,
) -> Diagnostic | None:
    direct = cast(JsonObject, correctness["direct_operator"])
    direct_layers = cast(list[JsonObject], direct["preinstallation_layers"])
    direct_pass = True
    for layer, record in enumerate(direct_layers):
        if record["layer"] != layer or record["layer_path"] != _layer_path(layer):
            return _diagnostic(
                ["correctness", "direct_operator", "preinstallation_layers", layer],
                "direct results must retain exact ordered layer paths",
            )
        result = cast(JsonObject, record["result"])
        derived = (
            cast(bool, result["finite"]) and cast(float, result["max_excess"]) <= 0
        )
        if result["pass"] != derived:
            return _diagnostic(
                [
                    "correctness",
                    "direct_operator",
                    "preinstallation_layers",
                    layer,
                    "result",
                    "pass",
                ],
                "direct pass must be derived from finite/max_excess evidence",
            )
        direct_pass = direct_pass and derived
    native_checks = [
        check
        for run in by_key.values()
        for check in cast(list[JsonObject], run["native_output_checks"])
    ]
    native_pass = all(
        cast(JsonObject, check["result"])["pass"] for check in native_checks
    )
    if direct["native_output_check_count"] != len(native_checks) or direct["pass"] != (
        direct_pass and native_pass
    ):
        return _diagnostic(
            ["correctness", "direct_operator", "pass"],
            "direct summary does not equal preinstall and native-output checks",
        )
    hybrid_steps = [
        step
        for key, run in by_key.items()
        if key[2] == "hybrid_native"
        for step in cast(list[JsonObject], run["steps"])
    ]
    logit_pass = all(
        step["finite"] and cast(JsonObject, step["hybrid_comparison"])["pass"]
        for step in hybrid_steps
    )
    logits = cast(JsonObject, correctness["model_logits"])
    if logits["comparison_count"] != len(hybrid_steps) or logits["pass"] != logit_pass:
        return _diagnostic(
            ["correctness", "model_logits"],
            "logit summary does not equal paired hybrid step evidence",
        )
    token_pass = all(
        by_key[(phase, repetition, "same_q8_reference")]["output_ids"]
        == by_key[(phase, repetition, "hybrid_native")]["output_ids"]
        for phase, repetitions in (("warmup", 2), ("measured", 10))
        for repetition in range(repetitions)
    )
    tokens = cast(JsonObject, correctness["tokens"])
    if tokens["comparison_count"] != 12 or tokens["pass"] != token_pass:
        return _diagnostic(
            ["correctness", "tokens"],
            "token summary does not equal the twelve paired runs",
        )
    overall = direct_pass and native_pass and logit_pass and token_pass
    if correctness["overall_pass"] != overall or (require_acceptance and not overall):
        return _diagnostic(
            ["correctness", "overall_pass"],
            "overall correctness does not equal the frozen component policies",
        )
    return None


def _validate_reconciliation(
    reconciliation: Mapping[str, Any],
    by_key: Mapping[tuple[str, int, str], JsonObject],
    *,
    require_acceptance: bool,
) -> Diagnostic | None:
    post = cast(JsonObject, reconciliation["post_run_native_validation_counters"])
    before = cast(list[JsonObject], post["before"])
    after = cast(list[JsonObject], post["after"])
    delta = cast(list[JsonObject], post["delta"])
    last_run = list(by_key.values())[-1]
    last_after = cast(list[JsonObject], cast(JsonObject, last_run["counters"])["after"])
    if before != last_after:
        return _diagnostic(
            ["reconciliation", "post_run_native_validation_counters", "before"],
            "post-run validation must start at the final timed-run snapshot",
        )
    expected_checks_per_layer = sum(
        len(cast(list[Any], run["output_ids"])) - 1
        for key, run in by_key.items()
        if key[2] == "hybrid_native"
    )
    for layer in range(22):
        records = (before[layer], after[layer], delta[layer])
        if any(
            record["layer"] != layer or record["layer_path"] != _layer_path(layer)
            for record in records
        ):
            return _diagnostic(
                ["reconciliation", "post_run_native_validation_counters"],
                "post-run validation counters must retain ordered layer paths",
            )
        before_values = cast(JsonObject, before[layer]["values"])
        after_values = cast(JsonObject, after[layer]["values"])
        delta_values = cast(JsonObject, delta[layer]["values"])
        if not _counter_invariants(before_values) or not _counter_invariants(
            after_values
        ):
            return _diagnostic(
                [
                    "reconciliation",
                    "post_run_native_validation_counters",
                    "after",
                    layer,
                    "values",
                ],
                "post-run adapter counter partitions do not reconcile",
            )
        for field in _NUMERIC_COUNTER_FIELDS:
            observed = cast(int, after_values[field]) - cast(int, before_values[field])
            if delta_values[field] != observed or observed < 0:
                return _diagnostic(
                    [
                        "reconciliation",
                        "post_run_native_validation_counters",
                        "delta",
                        layer,
                        "values",
                        field,
                    ],
                    "post-run validation delta does not match its snapshots",
                )
        if delta_values["closed_changed"] != (
            after_values["closed"] != before_values["closed"]
        ):
            return _diagnostic(
                [
                    "reconciliation",
                    "post_run_native_validation_counters",
                    "delta",
                    layer,
                    "values",
                    "closed_changed",
                ],
                "post-run closed_changed does not match its snapshots",
            )
        if require_acceptance:
            expected = {
                "forward": expected_checks_per_layer,
                "native_attempt": 0,
                "native_success": 0,
                "native_error": 0,
                "fallback_attempt": expected_checks_per_layer,
                "fallback_success": expected_checks_per_layer,
                "fallback_error": 0,
                "predispatch_error": 0,
                "rejected_closed": 0,
                "in_flight": 0,
            }
            if any(
                delta_values[field] != value for field, value in expected.items()
            ) or any(
                (
                    before_values["closed"],
                    after_values["closed"],
                    delta_values["closed_changed"],
                    after_values["in_flight"] != 0,
                )
            ):
                return _diagnostic(
                    [
                        "reconciliation",
                        "post_run_native_validation_counters",
                        "delta",
                        layer,
                    ],
                    "accepted post-run shadow checks must use error-free fallback",
                )
    path_records = cast(list[JsonObject], reconciliation["paths"])
    all_pass = True
    for path_index, path in enumerate(_PATHS):
        record = path_records[path_index]
        path_runs = [run for key, run in by_key.items() if key[2] == path]
        cached_calls = sum(
            len(cast(list[Any], run["output_ids"])) - 1 for run in path_runs
        )
        total_calls = cached_calls + len(path_runs)
        passed = (
            record["path"] == path
            and record["cached_decode_calls_per_layer"] == cached_calls
            and record["total_calls_per_layer"] == total_calls
        )
        if not passed or record["pass"] != passed:
            return _diagnostic(
                ["reconciliation", "paths", path_index],
                "path call totals do not reconcile with raw generation runs",
            )
        all_pass = all_pass and passed
    installation = cast(JsonObject, reconciliation["installation"])
    installation_pass = (
        installation["installed_modules"] == 22
        and installation["restored_modules"] == 22
        and installation["live_adapters"] == 0
        and installation["in_flight"] == 0
        and installation["closed"] is True
    )
    all_pass = all_pass and installation_pass
    if reconciliation["pass"] != all_pass or (require_acceptance and not all_pass):
        return _diagnostic(
            ["reconciliation", "pass"],
            "session reconciliation does not match lifecycle/call evidence",
        )
    return None


def _validate_drift(
    drift: Mapping[str, Any],
    by_key: Mapping[tuple[str, int, str], JsonObject],
    *,
    require_acceptance: bool,
) -> Diagnostic | None:
    records = cast(list[JsonObject], drift["paths"])
    lower = cast(float, drift["ratio_lower"])
    upper = cast(float, drift["ratio_upper"])
    all_pass = True
    for path_index, path in enumerate(_PATHS):
        values = [
            cast(
                int,
                cast(JsonObject, by_key[("measured", repetition, path)]["timing"])[
                    "total_ns"
                ],
            )
            for repetition in range(10)
        ]
        try:
            first = float(statistics.median(values[:3]))
            last = float(statistics.median(values[-3:]))
            ratio = last / first
        except (OverflowError, ZeroDivisionError):
            return _diagnostic(
                ["drift", "paths", path_index],
                "drift timing values must fit finite floating-point arithmetic",
            )
        record = records[path_index]
        passed = lower <= ratio <= upper
        try:
            recorded_values_match = (
                math.isclose(record["first_window_median_ns"], first)
                and math.isclose(record["last_window_median_ns"], last)
                and math.isclose(record["ratio"], ratio)
            )
        except OverflowError:
            return _diagnostic(
                ["drift", "paths", path_index],
                "drift summary values must fit finite floating-point arithmetic",
            )
        if (
            record["path"] != path
            or not recorded_values_match
            or record["pass"] != passed
        ):
            return _diagnostic(
                ["drift", "paths", path_index],
                "drift summary is not derived from retained measured total_ns samples",
            )
        all_pass = all_pass and passed
    if drift["pass"] != all_pass or (require_acceptance and not all_pass):
        return _diagnostic(
            ["drift", "pass"],
            "drift summary does not equal its path results",
        )
    return None


def _validate_session_semantics(instance: Mapping[str, Any]) -> list[Diagnostic]:
    """Reconcile G3 evidence that JSON Schema cannot express across fields."""

    nonfinite_path = _nonfinite_path(instance)
    if nonfinite_path is not None:
        return [
            _diagnostic(
                nonfinite_path,
                "all numeric evidence must be finite in every session state",
            )
        ]
    state = cast(JsonObject, instance["state"])
    if state["status"] == "not_run":
        return []
    require_acceptance = state["status"] == "accepted"
    provenance = (
        cast(JsonObject, instance["provenance"])
        if isinstance(instance["provenance"], dict)
        else None
    )
    assets = (
        cast(JsonObject, instance["assets"])
        if isinstance(instance["assets"], dict)
        else None
    )
    runs = (
        cast(list[JsonObject], instance["runs"])
        if isinstance(instance["runs"], list)
        else None
    )
    correctness = (
        cast(JsonObject, instance["correctness"])
        if isinstance(instance["correctness"], dict)
        else None
    )
    reconciliation = (
        cast(JsonObject, instance["reconciliation"])
        if isinstance(instance["reconciliation"], dict)
        else None
    )
    drift = (
        cast(JsonObject, instance["drift"])
        if isinstance(instance["drift"], dict)
        else None
    )
    if assets is not None:
        error = _validate_assets(assets, provenance)
        if error is not None:
            return [error]
    offline_preparation = instance["offline_preparation"]
    if (
        isinstance(offline_preparation, dict)
        and assets is not None
        and offline_preparation["asset_inventory_identity"]
        != assets["aggregate_identity"]
    ):
        return [
            _diagnostic(
                ["offline_preparation", "asset_inventory_identity"],
                "offline preparation receipt does not name the session assets",
            )
        ]
    if runs is None:
        if any(value is not None for value in (correctness, reconciliation, drift)):
            return [
                _diagnostic(
                    ["runs"],
                    "derived evidence cannot be retained without its raw runs",
                )
            ]
        return []
    error, by_key = _validate_runs(runs, require_acceptance=require_acceptance)
    if error is not None:
        return [error]
    if correctness is not None:
        error = _validate_correctness(
            correctness, by_key, require_acceptance=require_acceptance
        )
        if error is not None:
            return [error]
    if reconciliation is not None:
        error = _validate_reconciliation(
            reconciliation, by_key, require_acceptance=require_acceptance
        )
        if error is not None:
            return [error]
    if drift is not None:
        error = _validate_drift(drift, by_key, require_acceptance=require_acceptance)
        if error is not None:
            return [error]
    startup = instance["startup_timings"]
    if isinstance(startup, dict) and (
        startup["peak_rss_bytes"] < startup["baseline_rss_bytes"]
        or startup["cold_startup_ns"]
        < startup["tokenizer_load_ns"]
        + startup["model_load_ns"]
        + startup["install_ns"]
    ):
        return [
            _diagnostic(
                ["startup_timings"],
                "cold startup and peak-memory boundaries do not contain "
                "their components",
            )
        ]
    return []

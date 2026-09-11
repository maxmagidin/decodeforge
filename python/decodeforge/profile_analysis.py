"""Strict consistency analysis of three saved diagnostic profile sessions.

This validates declared observations, not their authenticity or independence.
Exclusive event durations partition each phase without double-counting nested
adapter spans. Instrumented observations are never performance evidence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, cast

from . import presentation_demo as presentation
from .decode_profile import MAX_PROFILE_EVENTS, _summaries
from .evaluation import CachedGeneration, _reconcile
from .profile_capture import _check_profile_trace
from .qproj_model import tinyllama_qproj_paths
from .torch_bridge import BRIDGE_ABI_VERSION

MAX_ANALYSIS_NS: Final = 2**63 - 1
_PATHS: Final = ("same_q8_reference", "hybrid_native")
_COUNTS: Final = (
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
_AGGREGATES: Final = (
    "forward",
    "native_attempt",
    "native_success",
    "fallback_attempt",
    "fallback_success",
)
_SOURCE_FILES: Final = {
    f"python/decodeforge/{name}.py"
    for name in (
        "evaluation",
        "g3_session",
        "presentation_demo",
        "decode_profile",
        "profile_capture",
        "qproj_adapter",
        "qproj_model",
        "qproj_profile",
        "torch_bridge",
    )
} | {"scripts/run_profile_capture.py"}

BUCKET_DESCRIPTIONS: Final = {
    "input_preparation": "Input tensor creation and attention-mask extension.",
    "output_validation": "Model output extraction and finite-logit validation.",
    "token_selection": "Greedy token selection and scalar extraction.",
    "bookkeeping": "Generated-token bookkeeping.",
    "step_remainder": "Step exclusive remainder, including observer work.",
    "model_remainder": "Unselected model work and observer overhead.",
    "qproj_remainder": (
        "Query-projection exclusive remainder, including the outer native "
        "eligibility guard."
    ),
    "adapter_storage_guard": "Both existing adapter fallback-storage validation sites.",
    "fallback_remainder": "Fallback exclusive remainder after measured children.",
    "fallback_clone": "Production fallback weight clone.",
    "fallback_hash": "Production fallback identity hashing and comparison.",
    "fallback_linear": "Same-Q8 fallback linear operation.",
    "native_operator_remainder": (
        "Guarded operator exclusive remainder outside binding execution; "
        "not kernel-only time."
    ),
    "guarded_binding_run": (
        "Binding lock, pointer checks, native ABI execution and status handling; "
        "not kernel-only time."
    ),
    "token_embedding": "Token embedding module.",
    "layer_norm": "Transformer layer norms and final model norm.",
    "attention_k_proj": "Attention key projections.",
    "attention_v_proj": "Attention value projections.",
    "attention_o_proj": "Attention output projections.",
    "mlp": "Whole transformer MLP modules.",
    "lm_head": "Language-model output head.",
}
COST_BUCKETS: Final = tuple(BUCKET_DESCRIPTIONS)


class ProfileAnalysisError(ValueError):
    """Saved sessions are incomplete, inconsistent, or outside this contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileAnalysisError(message)


def _object(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return dict(value)


def _integer(value: Any, label: str, low: int = 0, high: int = MAX_ANALYSIS_NS) -> int:
    _require(type(value) is int and low <= value <= high, f"{label} is invalid")
    return int(value)


def _text(value: Any, label: str, limit: int = 4096) -> str:
    _require(isinstance(value, str) and 0 < len(value) <= limit, f"{label} is invalid")
    return str(value)


def _hex(value: Any, label: str, length: int = 64) -> str:
    result = _text(value, label, length)
    _require(
        len(result) == length and all(c in "0123456789abcdef" for c in result),
        f"{label} must be lowercase hexadecimal",
    )
    return result


def _fields(value: Mapping[str, Any], names: set[str], label: str) -> None:
    _require(set(value) == names, f"{label} fields are invalid")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(left: Any, right: Any, label: str) -> None:
    # JSON encoding distinguishes bools from integers, unlike Python equality.
    _require(_canonical(left) == _canonical(right), f"{label} do not match")


def _json_values(document: Any) -> None:
    """Bound API inputs and reject non-JSON/nonfinite values even in unused fields."""
    pending = [(document, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        _require(
            nodes <= 2_000_000 and depth <= 32, "capture JSON exceeds analysis bounds"
        )
        if isinstance(value, dict):
            _require(
                len(value) <= 1024 and all(isinstance(k, str) for k in value),
                "capture JSON object is invalid",
            )
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            _require(
                len(value) <= MAX_PROFILE_EVENTS, "capture JSON array is too large"
            )
            pending.extend((item, depth + 1) for item in value)
        elif isinstance(value, str):
            _require(len(value) <= 65_536, "capture JSON string is too large")
        elif type(value) is int:
            _require(
                -MAX_ANALYSIS_NS <= value <= MAX_ANALYSIS_NS,
                "capture integer exceeds bounds",
            )
        elif type(value) is float:
            _require(math.isfinite(value), "capture JSON contains a nonfinite number")
        else:
            _require(
                value is None or type(value) is bool, "capture contains non-JSON values"
            )


def _identity(document: dict[str, Any]) -> dict[str, Any]:
    source = _object(document.get("source"), "source")
    _fields(source, {"git_revision", "git_dirty", "files"}, "source")
    _hex(source.get("git_revision"), "source revision", 40)
    _require(
        source.get("git_dirty") is False, "analysis requires clean producer source"
    )
    files = _object(source.get("files"), "source files")
    _fields(files, _SOURCE_FILES, "source files")
    for name, digest in files.items():
        _hex(digest, f"source file {name}")
    model = _object(document.get("model_identity"), "model identity")
    expected_files = {
        name: {"bytes": size, "sha256": digest}
        for name, (size, digest) in presentation._PINNED_MODEL_FILES.items()
    }
    _same(
        model,
        {
            "model_id": presentation._PINNED_MODEL_ID,
            "revision": presentation._PINNED_MODEL_REVISION,
            "files": expected_files,
        },
        "pinned model identities",
    )
    asset = _text(document.get("asset_inventory_identity"), "asset identity", 71)
    _require(asset.startswith("sha256:"), "asset identity must use sha256")
    _hex(asset[7:], "asset identity digest")
    bridge = _hex(document.get("bridge_library_sha256"), "bridge digest")
    _require(
        _integer(document.get("bridge_abi_version"), "bridge ABI")
        == BRIDGE_ABI_VERSION,
        "bridge ABI is unsupported",
    )
    environment = _object(document.get("environment"), "environment")
    _fields(
        environment,
        {
            "python",
            "platform",
            "machine",
            "packages",
            "model_class",
            "tokenizer_class",
            "torch_num_threads",
            "torch_num_interop_threads",
            "offline_environment",
            "pid",
            "platform_system",
        },
        "environment",
    )
    for field in (
        "python",
        "platform",
        "machine",
        "model_class",
        "tokenizer_class",
        "platform_system",
    ):
        _text(environment.get(field), f"environment.{field}")
    for field in ("torch_num_threads", "torch_num_interop_threads"):
        _require(
            _integer(environment.get(field), field) == 1,
            "capture thread settings must be one",
        )
    _same(
        environment.get("offline_environment"),
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "offline environments",
    )
    packages = _object(environment.get("packages"), "environment packages")
    _fields(
        packages,
        {"torch", "transformers", "tokenizers", "safetensors"},
        "environment packages",
    )
    for name, version in packages.items():
        _text(version, f"package {name}", 128)
    pid = _integer(document.get("process_id"), "process ID", 1)
    _require(
        _integer(environment.get("pid"), "environment PID", 1) == pid,
        "process IDs disagree within capture",
    )
    environment.pop("pid")
    workload = _object(document.get("workload"), "workload")
    _fields(
        workload,
        {
            "prompt",
            "rendered_prompt_sha256",
            "input_token_ids",
            "max_new_tokens",
            "greedy",
            "use_cache",
        },
        "workload",
    )
    _require(
        bool(_text(workload.get("prompt"), "prompt").strip()),
        "prompt must not be blank",
    )
    _hex(workload.get("rendered_prompt_sha256"), "rendered prompt digest")
    _tokens(workload.get("input_token_ids"), "input token IDs", 2, 512)
    _integer(workload.get("max_new_tokens"), "max_new_tokens", 2, 64)
    _require(
        workload.get("greedy") is True and workload.get("use_cache") is True,
        "workload must use greedy cached generation",
    )
    return {
        "workload": workload,
        "environment_without_pid": environment,
        "source": source,
        "model_identity": model,
        "asset_inventory_identity": asset,
        "bridge_library_sha256": bridge,
        "bridge_abi_version": BRIDGE_ABI_VERSION,
    }


def _tokens(value: Any, label: str, low: int, high: int) -> list[int]:
    _require(
        isinstance(value, list) and low <= len(value) <= high,
        f"{label} length is invalid",
    )
    return [_integer(token, label) for token in value]


def _generation(value: Any, workload: dict[str, Any]) -> CachedGeneration:
    record = _object(value, "generation")
    _fields(
        record,
        {
            "token_ids",
            "generated_token_ids",
            "generated_token_count",
            "stop_reason",
            "stopped_by_eos",
            "text",
        },
        "generation",
    )
    generated = _tokens(
        record.get("generated_token_ids"),
        "generated token IDs",
        2,
        workload["max_new_tokens"],
    )
    ids = _tokens(record.get("token_ids"), "all token IDs", 4, 576)
    _require(
        ids == workload["input_token_ids"] + generated,
        "generation token prefix is inconsistent",
    )
    _require(
        _integer(record.get("generated_token_count"), "generated token count")
        == len(generated),
        "generated token count is inconsistent",
    )
    stop = record.get("stop_reason")
    _require(stop in ("eos", "max_new_tokens"), "generation stop reason is invalid")
    _require(
        record.get("stopped_by_eos") is (stop == "eos"),
        "generation EOS flag is inconsistent",
    )
    if stop == "max_new_tokens":
        _require(
            len(generated) == workload["max_new_tokens"], "max-token stop ended early"
        )
    _require(isinstance(record.get("text"), str), "generation text must be a string")
    return CachedGeneration(ids, generated, str(stop), stop == "eos", [], [])


def _snapshot(value: Any, *, closed: bool = False) -> dict[str, Any]:
    record = _object(value, "counter snapshot")
    _fields(
        record,
        {
            "installed_modules",
            "restored_modules",
            "live_adapters",
            "in_flight",
            "closed",
            "layers",
            *_AGGREGATES,
        },
        "counter snapshot",
    )
    for field, expected in (
        ("installed_modules", 0 if closed else 22),
        ("restored_modules", 22 if closed else 0),
        ("live_adapters", 0 if closed else 22),
        ("in_flight", 0),
    ):
        _require(
            _integer(record.get(field), field) == expected,
            "installation lifecycle is inconsistent",
        )
    _require(record.get("closed") is closed, "installation closed flag is inconsistent")
    layers = record.get("layers")
    _require(
        isinstance(layers, list) and len(layers) == 22,
        "counter coverage must contain 22 layers",
    )
    layers = cast(list[Any], layers)
    for index, (value, path) in enumerate(
        zip(layers, tinyllama_qproj_paths(), strict=True)
    ):
        layer = _object(value, "layer counters")
        _fields(layer, {"layer", "layer_path", "closed", *_COUNTS}, "layer counters")
        _require(
            _integer(layer.get("layer"), "layer index") == index
            and layer.get("layer_path") == path,
            "counter layer ordering is inconsistent",
        )
        _require(layer.get("closed") is closed, "layer closed flag is inconsistent")
        counts = {field: _integer(layer.get(field), field, 0, 384) for field in _COUNTS}
        _require(
            all(
                counts[field] == 0
                for field in (
                    "native_error",
                    "fallback_error",
                    "predispatch_error",
                    "rejected_closed",
                    "in_flight",
                )
            ),
            "capture counters contain errors or in-flight work",
        )
        _require(
            counts["native_attempt"] == counts["native_success"]
            and counts["fallback_attempt"] == counts["fallback_success"]
            and counts["forward"]
            == counts["native_attempt"] + counts["fallback_attempt"],
            "counter dispatch partition is inconsistent",
        )
    for field in _AGGREGATES:
        _require(
            _integer(record.get(field), field) == sum(layer[field] for layer in layers),
            "aggregate counters disagree with layers",
        )
    return record


def _bucket(event: dict[str, Any]) -> str:
    boundary = str(event["boundary"])
    renamed = {
        "prefill": "step_remainder",
        "cached_step": "step_remainder",
        "model_forward": "model_remainder",
        "fallback": "fallback_remainder",
        "guarded_native_operator": "native_operator_remainder",
    }
    if boundary in renamed:
        return renamed[boundary]
    if boundary != "module_forward":
        _require(boundary in COST_BUCKETS, "unrecognized diagnostic cost boundary")
        return boundary
    path = str(event["module_path"])
    if path.endswith(".q_proj"):
        return "qproj_remainder"
    if path == "model.embed_tokens":
        return "token_embedding"
    if path == "model.norm" or path.endswith("layernorm"):
        return "layer_norm"
    for suffix, bucket in (
        (".k_proj", "attention_k_proj"),
        (".v_proj", "attention_v_proj"),
        (".o_proj", "attention_o_proj"),
        (".mlp", "mlp"),
        ("lm_head", "lm_head"),
    ):
        if path.endswith(suffix):
            return bucket
    raise ProfileAnalysisError("unrecognized component cost boundary")


def _ranking(buckets: Mapping[str, int]) -> list[list[str]]:
    levels: dict[int, list[str]] = {}
    for name, value in buckets.items():
        levels.setdefault(value, []).append(name)
    return [sorted(levels[value]) for value in sorted(levels, reverse=True)]


def _phase(trace: dict[str, Any], phase: str) -> dict[str, Any]:
    events = [event for event in trace["events"] if event["phase"] == phase]
    boundaries = [
        event for event in events if event["boundary"] in {"prefill", "cached_step"}
    ]
    total = sum(event["inclusive_ns"] for event in boundaries)
    _integer(total, f"{phase} total", 1)
    buckets = dict.fromkeys(COST_BUCKETS, 0)
    for event in events:
        buckets[_bucket(event)] += event["exclusive_ns"]
    _require(
        sum(buckets.values()) == total, f"{phase} cost partition does not conserve time"
    )
    return {
        "total_ns": total,
        "steps": len(boundaries),
        "buckets_ns": buckets,
        "bucket_shares": {name: value / total for name, value in buckets.items()},
        "ranking_groups": _ranking(buckets),
    }


def _session(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], CachedGeneration]:
    _fields(
        document,
        {
            "format",
            "schema_version",
            "claim_class",
            "benchmark",
            "g3_evidence",
            "performance_claim_allowed",
            "capture_status",
            "session_index",
            "process_id",
            "command_line",
            "source",
            "environment",
            "model_identity",
            "asset_inventory_identity",
            "bridge_library_sha256",
            "bridge_abi_version",
            "workload",
            "mode_order",
            "runs",
            "comparison",
            "installation",
            "restoration",
            "interpretation",
        },
        "capture",
    )
    command = document.get("command_line")
    _require(
        isinstance(command, list) and 1 <= len(command) <= 256,
        "capture command line is invalid",
    )
    for argument in cast(list[Any], command):
        _require(
            isinstance(argument, str) and len(argument) <= 4096,
            "capture command argument is invalid",
        )
    _text(document.get("interpretation"), "capture interpretation")
    for field, expected in (
        ("format", "decodeforge_profile_capture_v1"),
        ("claim_class", "diagnostic_profile"),
        ("capture_status", "complete"),
    ):
        _require(document.get(field) == expected, f"capture {field} is unsupported")
    _require(
        _integer(document.get("schema_version"), "capture schema version") == 1,
        "capture schema version is unsupported",
    )
    for field in ("benchmark", "g3_evidence", "performance_claim_allowed"):
        _require(document.get(field) is False, "capture must be diagnostic-only")
    index = _integer(document.get("session_index"), "session index", 0, 2)
    identity = _identity(document)
    mode_order = list(_PATHS if index % 2 == 0 else reversed(_PATHS))
    measurement_order = (
        ["control", "profile"] if index % 2 == 0 else ["profile", "control"]
    )
    _same(document.get("mode_order"), mode_order, "mode order")
    runs = _object(document.get("runs"), "runs")
    _fields(runs, set(_PATHS), "runs")
    _same(
        document.get("comparison"),
        {
            "control_profile_exact": True,
            "same_q8_hybrid_exact": True,
            "cached_decode_covered": True,
        },
        "comparison declarations",
    )
    previous = _snapshot(document.get("installation"))
    _require(previous["forward"] == 0, "installation counters must start at zero")
    generation: CachedGeneration | None = None
    clock_identity: dict[str, Any] | None = None
    paths: dict[str, Any] = {}
    for name in mode_order:
        run = _object(runs[name], f"{name} run")
        _fields(
            run,
            {
                "measurement_order",
                "warmup",
                "control",
                "profile",
                "observer_cost_ratio",
            },
            "mode run",
        )
        _same(run.get("measurement_order"), measurement_order, "measurement order")
        for kind in ["warmup", *measurement_order]:
            record = _object(run.get(kind), f"{name}.{kind}")
            _fields(
                record,
                {
                    "kind",
                    "measured",
                    "outer_elapsed_ns",
                    "generation",
                    "counters_before",
                    "counters_after",
                    "counter_delta",
                    "trace",
                },
                "generation run",
            )
            _require(
                record.get("kind") == kind
                and record.get("measured") is (kind != "warmup"),
                "run kind/measured flags are inconsistent",
            )
            outer = _integer(record.get("outer_elapsed_ns"), "outer elapsed time", 1)
            actual = _generation(record.get("generation"), identity["workload"])
            if generation is not None:
                _require(
                    actual == generation,
                    "generation parity failed between paths or repetitions",
                )
            generation = actual
            before, after = (
                _snapshot(record.get("counters_before")),
                _snapshot(record.get("counters_after")),
            )
            _same(before, previous, "counter snapshot continuity")
            delta = presentation._counter_delta(before, after)
            _same(record.get("counter_delta"), delta, "saved/recomputed counter deltas")
            _reconcile(name, len(actual.generated_ids), delta)
            previous = after
            if kind != "profile":
                _require(
                    record.get("trace") is None,
                    "control/warmup must not contain profile traces",
                )
                continue
            trace = _object(record.get("trace"), "profile trace")
            _fields(
                trace,
                {
                    "format",
                    "schema_version",
                    "claim_class",
                    "performance_claim_allowed",
                    "boundary_contract",
                    "qproj_details",
                    "clock",
                    "generated_token_ids",
                    "generated_token_count",
                    "stop_reason",
                    "module_paths",
                    "events",
                    "summary",
                    "interpretation",
                },
                "profile trace",
            )
            _text(trace.get("interpretation"), "trace interpretation")
            _check_profile_trace(name, actual, trace)
            clock = _object(trace.get("clock"), "profile clock")
            _fields(
                clock, {"name", "resolution_ns", "overhead_subtracted"}, "profile clock"
            )
            _require(
                clock.get("name") == "time.perf_counter_ns",
                "analysis requires the production monotonic clock",
            )
            _integer(clock.get("resolution_ns"), "clock resolution", 1)
            if clock_identity is not None:
                _same(clock, clock_identity, "profile clocks")
            clock_identity = clock
            events = trace["events"]
            for event in events:
                _fields(
                    event,
                    {
                        "event_id",
                        "parent_id",
                        "boundary",
                        "phase",
                        "step_index",
                        "module_path",
                        "start_ns",
                        "end_ns",
                        "inclusive_ns",
                        "exclusive_ns",
                        "dispatch",
                        "guard_reason",
                    },
                    "trace event",
                )
            _require(
                [event["event_id"] for event in events] == list(range(len(events))),
                "trace events must be in original ID order",
            )
            _require(
                all(
                    left["start_ns"] <= right["start_ns"]
                    for left, right in pairwise(events)
                ),
                "trace timestamps do not follow event order",
            )
            _same(
                trace.get("summary"),
                _summaries(events),
                "saved/recomputed trace summaries",
            )
            root = events[0]
            _require(
                outer >= root["inclusive_ns"], "profile root exceeds outer elapsed time"
            )
            prefill, cached = _phase(trace, "prefill"), _phase(trace, "cached_decode")
            _require(
                prefill["total_ns"] + cached["total_ns"] + root["exclusive_ns"]
                == root["inclusive_ns"],
                "generation cost partition does not conserve time",
            )
            paths[name] = {
                "prefill": prefill,
                "cached_decode": cached,
                "generation_remainder_ns": root["exclusive_ns"],
                "generation_total_ns": root["inclusive_ns"],
            }
        control_ns = run["control"]["outer_elapsed_ns"]
        profile_ns = run["profile"]["outer_elapsed_ns"]
        ratio = profile_ns / control_ns
        saved_ratio = run.get("observer_cost_ratio")
        _require(
            isinstance(saved_ratio, int | float)
            and not isinstance(saved_ratio, bool)
            and math.isfinite(saved_ratio)
            and saved_ratio == ratio,
            "saved observer ratio disagrees with elapsed times",
        )
        paths[name]["observer"] = {
            "control_ns": control_ns,
            "profile_ns": profile_ns,
            "ratio": ratio,
        }
    restoration = _object(document.get("restoration"), "restoration")
    _fields(
        restoration, {"closed", "original_modules_restored", "counters"}, "restoration"
    )
    _require(
        restoration.get("closed") is True
        and restoration.get("original_modules_restored") is True,
        "model restoration is incomplete",
    )
    final = _snapshot(restoration.get("counters"), closed=True)
    for before, after in zip(previous["layers"], final["layers"], strict=True):
        _same(
            {key: before[key] for key in _COUNTS},
            {key: after[key] for key in _COUNTS},
            "teardown counter continuity",
        )
    _require(generation is not None, "capture has no generations")
    assert generation is not None
    identity["clock"] = clock_identity
    return (
        {"session_index": index, "process_id": document["process_id"], "paths": paths},
        identity,
        generation,
    )


def analyze_profile_sessions(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate and partition exactly three comparable detailed captures.

    PIDs and clean-source declarations are consistency checks only. This
    function does not authenticate files or establish process independence.
    """
    _require(len(documents) == 3, "analysis requires exactly three sessions")
    sessions: list[dict[str, Any]] = []
    matched: dict[str, Any] | None = None
    generated: CachedGeneration | None = None
    try:
        for document in documents:
            _json_values(document)
            session, identity, generation = _session(_object(document, "capture"))
            if matched is not None:
                _same(
                    identity,
                    matched,
                    "session workload/environment/source/artifact identities",
                )
                _require(
                    generation == generated, "generation parity failed across sessions"
                )
            matched, generated = identity, generation
            sessions.append(session)
    except ProfileAnalysisError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ProfileAnalysisError(f"invalid profile session: {error}") from error
    _require(
        {session["session_index"] for session in sessions} == {0, 1, 2},
        "session indexes must be exactly 0, 1, and 2",
    )
    _require(
        len({session["process_id"] for session in sessions}) == 3,
        "sessions must declare three distinct process IDs",
    )
    sessions.sort(key=lambda session: session["session_index"])
    stability: dict[str, Any] = {}
    for path in _PATHS:
        rankings = [
            session["paths"][path]["cached_decode"]["ranking_groups"]
            for session in sessions
        ]
        stable = all(value == rankings[0] for value in rankings[1:])
        top_stable = all(value[0] == rankings[0][0] for value in rankings[1:])
        stability[path] = {
            "ordering_stable": stable,
            "stable_ranking_groups": rankings[0] if stable else None,
            "top_group_stable": top_stable,
            "stable_top_group": rankings[0][0] if top_stable else None,
        }
    return {
        "format": "decodeforge_profile_analysis_v1",
        "schema_version": 1,
        "claim_class": "diagnostic_profile_analysis",
        "benchmark": False,
        "g3_evidence": False,
        "performance_claim_allowed": False,
        "matched_inputs": matched,
        "sessions": sessions,
        "stability": stability,
        "bucket_descriptions": dict(BUCKET_DESCRIPTIONS),
        "interpretation": (
            "Exclusive durations partition each prefill/cached phase including all "
            "remainders. Exact ties are grouped; ordering and top-group stability "
            "are separate observations. Observer ratios compare whole calls, not "
            "cached-only overhead. Distinct declared PIDs and matching identities "
            "do not authenticate captures or prove independence. Instrumented "
            "timings establish no speedup or completed optimization."
        ),
    }


__all__ = [
    "COST_BUCKETS",
    "MAX_ANALYSIS_NS",
    "ProfileAnalysisError",
    "analyze_profile_sessions",
]

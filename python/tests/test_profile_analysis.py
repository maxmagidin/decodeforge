"""Adversarial offline analysis of complete synthetic capture sessions."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from typing import Any

import pytest
from decodeforge import presentation_demo as presentation
from decodeforge import profile_analysis as analysis
from decodeforge.decode_profile import _summaries

from test_profile_capture import _run

PATHS = ("same_q8_reference", "hybrid_native")


def _retime(trace: dict[str, Any], costs: dict[str, int]) -> int:
    """Assign independent exclusive costs while preserving the captured topology."""
    children: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for event in trace["events"]:
        children[event["parent_id"]].append(event)
    for siblings in children.values():
        siblings.sort(key=lambda event: event["event_id"])

    def visit(event: dict[str, Any], now: int) -> int:
        exclusive = costs.get(event["boundary"], 1)
        event["start_ns"] = now
        ended = now + exclusive
        for child in children[event["event_id"]]:
            ended = visit(child, ended)
        event["end_ns"] = ended
        event["inclusive_ns"] = ended - now
        event["exclusive_ns"] = exclusive
        return ended

    root = children[None][0]
    visit(root, 1_000)
    trace["summary"] = _summaries(trace["events"])
    return int(root["inclusive_ns"])


def _retime_session(document: dict[str, Any], costs: dict[str, int]) -> None:
    for name in PATHS:
        runs = document["runs"][name]
        total = _retime(runs["profile"]["trace"], costs)
        runs["warmup"]["outer_elapsed_ns"] = 2 * total
        runs["control"]["outer_elapsed_ns"] = 2 * total
        runs["profile"]["outer_elapsed_ns"] = 3 * total
        runs["observer_cost_ratio"] = 1.5


@pytest.fixture(scope="module")
def captured_sessions(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Real owning-adapter counters and hooks, synthetic identities and durations."""
    result = []
    for index in range(3):
        directory = tmp_path_factory.mktemp(f"profile-analysis-{index}")
        document, _model, _installation = _run(directory, session_index=index)
        document["process_id"] = 10_000 + index
        document["environment"]["pid"] = 10_000 + index
        document["command_line"] = ["synthetic-profile-capture", str(index)]
        document["source"]["git_dirty"] = False
        document["source"]["git_revision"] = "1" * 40
        document["model_identity"]["files"] = {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in presentation._PINNED_MODEL_FILES.items()
        }
        for name in PATHS:
            document["runs"][name]["profile"]["trace"]["clock"] = {
                "name": "time.perf_counter_ns",
                "resolution_ns": 1,
                "overhead_subtracted": False,
            }
        _retime_session(document, {"fallback_hash": 1_000})
        result.append(document)
    return result


@pytest.fixture
def sessions(captured_sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(captured_sessions)


def test_analysis_is_deterministic_and_does_not_mutate_inputs(
    sessions: list[dict[str, Any]],
) -> None:
    before = copy.deepcopy(sessions)
    expected = analysis.analyze_profile_sessions(sessions)
    assert analysis.analyze_profile_sessions(list(reversed(sessions))) == expected
    assert (
        analysis.analyze_profile_sessions(
            json.loads(json.dumps(sessions, sort_keys=True))
        )
        == expected
    )
    assert sessions == before
    assert [item["session_index"] for item in expected["sessions"]] == [0, 1, 2]
    assert json.loads(json.dumps(expected, allow_nan=False)) == expected


def test_exclusive_attribution_conserves_each_phase_and_generation(
    sessions: list[dict[str, Any]],
) -> None:
    result = analysis.analyze_profile_sessions(sessions)
    for source, session in zip(sessions, result["sessions"], strict=True):
        for name in PATHS:
            path = session["paths"][name]
            events = source["runs"][name]["profile"]["trace"]["events"]
            for phase, boundary in (
                ("prefill", "prefill"),
                ("cached_decode", "cached_step"),
            ):
                expected = sum(
                    event["inclusive_ns"]
                    for event in events
                    if event["boundary"] == boundary
                )
                assert path[phase]["total_ns"] == expected
                assert sum(path[phase]["buckets_ns"].values()) == expected
                assert path[phase]["steps"] == 1
            assert path["generation_total_ns"] == (
                path["prefill"]["total_ns"]
                + path["cached_decode"]["total_ns"]
                + path["generation_remainder_ns"]
            )
            assert path["generation_remainder_ns"] == 1
            assert path["observer"]["ratio"] == 1.5
    reference = result["sessions"][0]["paths"]["same_q8_reference"]
    assert reference["cached_decode"]["buckets_ns"]["fallback_hash"] == 22_000
    assert reference["cached_decode"]["buckets_ns"]["fallback_remainder"] == 22
    hybrid = result["sessions"][0]["paths"]["hybrid_native"]
    assert hybrid["cached_decode"]["buckets_ns"]["guarded_binding_run"] == 22
    assert hybrid["cached_decode"]["buckets_ns"]["native_operator_remainder"] == 22


def test_stable_ranking_uses_exact_tie_groups(sessions: list[dict[str, Any]]) -> None:
    for document in sessions:
        _retime_session(document, {"fallback_hash": 1_000, "fallback_linear": 1_000})
    result = analysis.analyze_profile_sessions(sessions)
    stable = result["stability"]["same_q8_reference"]
    assert stable["ordering_stable"] is True
    assert stable["top_group_stable"] is True
    assert stable["stable_top_group"] == ["fallback_hash", "fallback_linear"]
    assert stable["stable_ranking_groups"][0] == ["fallback_hash", "fallback_linear"]


def test_changed_winner_is_reported_as_unstable(sessions: list[dict[str, Any]]) -> None:
    _retime_session(sessions[2], {"fallback_linear": 2_000})
    result = analysis.analyze_profile_sessions(sessions)
    stable = result["stability"]["same_q8_reference"]
    assert stable["ordering_stable"] is False
    assert stable["stable_ranking_groups"] is None
    assert stable["top_group_stable"] is False
    assert stable["stable_top_group"] is None


def test_stable_winner_does_not_imply_stable_full_order(
    sessions: list[dict[str, Any]],
) -> None:
    _retime_session(sessions[2], {"fallback_hash": 1_000, "fallback_clone": 3})
    stable = analysis.analyze_profile_sessions(sessions)["stability"][
        "same_q8_reference"
    ]
    assert stable["top_group_stable"] is True
    assert stable["stable_top_group"] == ["fallback_hash"]
    assert stable["ordering_stable"] is False
    assert stable["stable_ranking_groups"] is None


def test_zero_cost_buckets_form_a_deterministic_tie_group(
    sessions: list[dict[str, Any]],
) -> None:
    boundaries = {
        event["boundary"]
        for name in PATHS
        for event in sessions[0]["runs"][name]["profile"]["trace"]["events"]
    }
    for document in sessions:
        _retime_session(
            document, {**dict.fromkeys(boundaries, 0), "input_preparation": 1}
        )
    result = analysis.analyze_profile_sessions(sessions)
    for name in PATHS:
        phase = result["sessions"][0]["paths"][name]["cached_decode"]
        assert phase["total_ns"] == 1
        assert phase["ranking_groups"][0] == ["input_preparation"]
        zeros = sorted(key for key, value in phase["buckets_ns"].items() if value == 0)
        assert phase["ranking_groups"][1] == zeros
        assert len(phase["ranking_groups"]) == 2


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_exactly_three_sessions_are_required(
    sessions: list[dict[str, Any]], count: int
) -> None:
    documents = [*sessions, sessions[0]][:count]
    with pytest.raises(analysis.ProfileAnalysisError):
        analysis.analyze_profile_sessions(documents)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_session",
        "duplicate_process",
        "environment_pid",
        "dirty_source",
        "revision",
        "source_hash",
        "missing_source_file",
        "model_file",
        "asset_identity",
        "bridge_identity",
        "workload",
        "threads",
        "clock",
        "rejected",
        "claim",
        "comparison",
        "mode_order",
        "measurement_order",
        "cleanup_live",
        "cleanup_module",
        "counter_delta",
        "counter_continuity",
        "control_tokens",
        "cross_session_tokens",
        "boolean_counter",
        "boolean_version",
        "outer_timing",
        "observer_ratio",
        "summary",
        "unexpected_field",
        "unexpected_trace_field",
        "command_line_type",
        "command_line_item",
        "command_line_empty",
        "interpretation_type",
    ],
)
def test_analysis_rejects_tampered_capture(
    sessions: list[dict[str, Any]], mutation: str
) -> None:
    document = sessions[1]
    runs = document["runs"]["same_q8_reference"]
    if mutation == "duplicate_session":
        document["session_index"] = 0
    elif mutation == "duplicate_process":
        document["process_id"] = document["environment"]["pid"] = sessions[0][
            "process_id"
        ]
    elif mutation == "environment_pid":
        document["environment"]["pid"] += 1
    elif mutation == "dirty_source":
        document["source"]["git_dirty"] = True
    elif mutation == "revision":
        document["source"]["git_revision"] = "2" * 40
    elif mutation == "source_hash":
        document["source"]["files"]["python/decodeforge/qproj_adapter.py"] = "2" * 64
    elif mutation == "missing_source_file":
        document["source"]["files"].pop("python/decodeforge/qproj_adapter.py")
    elif mutation == "model_file":
        document["model_identity"]["files"]["config.json"]["sha256"] = "2" * 64
    elif mutation == "asset_identity":
        document["asset_inventory_identity"] = "sha256:" + "2" * 64
    elif mutation == "bridge_identity":
        document["bridge_library_sha256"] = "2" * 64
    elif mutation == "workload":
        document["workload"]["input_token_ids"][0] += 1
    elif mutation == "threads":
        document["environment"]["torch_num_threads"] = 2
    elif mutation == "clock":
        runs["profile"]["trace"]["clock"]["resolution_ns"] = 2
    elif mutation == "rejected":
        document["capture_status"] = "rejected"
    elif mutation == "claim":
        document["performance_claim_allowed"] = True
    elif mutation == "comparison":
        document["comparison"]["control_profile_exact"] = False
    elif mutation == "mode_order":
        document["mode_order"].reverse()
    elif mutation == "measurement_order":
        runs["measurement_order"].reverse()
    elif mutation == "cleanup_live":
        document["restoration"]["counters"]["live_adapters"] = 1
    elif mutation == "cleanup_module":
        document["restoration"]["original_modules_restored"] = False
    elif mutation == "counter_delta":
        runs["profile"]["counter_delta"][0]["fallback_success"] -= 1
    elif mutation == "counter_continuity":
        # Both snapshots and their delta remain valid in isolation. Only the
        # continuity with the preceding/following run exposes this corruption.
        for snapshot in ("counters_before", "counters_after"):
            for field in ("forward", "fallback_attempt", "fallback_success"):
                runs["control"][snapshot]["layers"][0][field] += 1
                runs["control"][snapshot][field] += 1
    elif mutation == "control_tokens":
        runs["control"]["generation"]["generated_token_ids"][0] = 77
        runs["control"]["generation"]["token_ids"][-2] = 77
    elif mutation == "cross_session_tokens":
        for name in PATHS:
            for kind in ("warmup", "control", "profile"):
                record = document["runs"][name][kind]
                record["generation"]["generated_token_ids"][0] = 77
                record["generation"]["token_ids"][-2] = 77
                record["generation"]["text"] = "clean:[77, 99]"
                if record["trace"] is not None:
                    record["trace"]["generated_token_ids"][0] = 77
    elif mutation == "boolean_counter":
        runs["profile"]["counter_delta"][0]["in_flight"] = False
    elif mutation == "boolean_version":
        document["schema_version"] = True
    elif mutation == "outer_timing":
        runs["profile"]["outer_elapsed_ns"] = 1
    elif mutation == "observer_ratio":
        runs["observer_cost_ratio"] = 2.0
    elif mutation == "summary":
        runs["profile"]["trace"]["summary"][0]["inclusive_total_ns"] += 1
    elif mutation == "unexpected_field":
        document["trusted"] = True
    elif mutation == "unexpected_trace_field":
        runs["profile"]["trace"]["trusted"] = True
    elif mutation == "command_line_type":
        document["command_line"] = "synthetic-profile-capture"
    elif mutation == "command_line_item":
        document["command_line"] = ["synthetic-profile-capture", 12]
    elif mutation == "command_line_empty":
        document["command_line"] = []
    elif mutation == "interpretation_type":
        document["interpretation"] = True
    else:
        raise AssertionError(mutation)
    with pytest.raises(analysis.ProfileAnalysisError):
        analysis.analyze_profile_sessions(sessions)


@pytest.mark.parametrize(
    "mutation", ["model_pin", "source_inventory", "threads", "boolean_version"]
)
def test_consistent_cross_session_corruption_is_still_rejected(
    sessions: list[dict[str, Any]], mutation: str
) -> None:
    for document in sessions:
        if mutation == "model_pin":
            document["model_identity"]["files"]["config.json"]["sha256"] = "2" * 64
        elif mutation == "source_inventory":
            document["source"]["files"].pop("python/decodeforge/qproj_profile.py")
        elif mutation == "threads":
            document["environment"]["torch_num_threads"] = 2
        elif mutation == "boolean_version":
            document["schema_version"] = True
        else:
            raise AssertionError(mutation)
    with pytest.raises(analysis.ProfileAnalysisError):
        analysis.analyze_profile_sessions(sessions)


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_time",
        "nan_time",
        "negative_time",
        "duplicate_id",
        "self_parent",
        "boolean_step",
        "bad_phase",
        "wrong_dispatch",
        "extra_child",
        "missing_event",
        "exclusive_total",
        "unexpected_field",
    ],
)
def test_analysis_rejects_malformed_event_tree(
    sessions: list[dict[str, Any]], mutation: str
) -> None:
    trace = sessions[0]["runs"]["hybrid_native"]["profile"]["trace"]
    events = trace["events"]
    event = events[-1]
    if mutation == "boolean_time":
        event["start_ns"] = True
    elif mutation == "nan_time":
        event["end_ns"] = float("nan")
    elif mutation == "negative_time":
        event["inclusive_ns"] = -1
    elif mutation == "duplicate_id":
        event["event_id"] = events[-2]["event_id"]
    elif mutation == "self_parent":
        event["parent_id"] = event["event_id"]
    elif mutation == "boolean_step":
        event["step_index"] = True
    elif mutation == "bad_phase":
        event["phase"] = "warmup"
    elif mutation == "wrong_dispatch":
        next(item for item in events if item["dispatch"] == "native")["dispatch"] = (
            "fallback"
        )
    elif mutation == "extra_child":
        extra = copy.deepcopy(event)
        extra["event_id"] += 1
        events.append(extra)
    elif mutation == "missing_event":
        events.pop()
    elif mutation == "exclusive_total":
        events[0]["exclusive_ns"] += 1
    elif mutation == "unexpected_field":
        event["trusted"] = True
    else:
        raise AssertionError(mutation)
    if mutation in {"bad_phase", "wrong_dispatch", "extra_child", "missing_event"}:
        trace["summary"] = _summaries(events)
    with pytest.raises(analysis.ProfileAnalysisError):
        analysis.analyze_profile_sessions(sessions)

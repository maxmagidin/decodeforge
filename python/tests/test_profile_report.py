"""Readable diagnostics preserve arithmetic, uncertainty, and metadata privacy."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, cast

import pytest
from decodeforge.profile_analysis import analyze_profile_sessions
from decodeforge.profile_report import render_profile_report

import test_profile_capture
from test_profile_analysis import PATHS, _retime_session
from test_profile_analysis import captured_sessions as captured_sessions


@pytest.fixture
def sessions(request: pytest.FixtureRequest) -> list[dict[str, Any]]:
    captures = cast(list[dict[str, Any]], request.getfixturevalue("captured_sessions"))
    return copy.deepcopy(captures)


def _section(markdown: str, heading: str) -> str:
    return markdown.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


def _hash_row(section: str) -> list[str]:
    line = next(
        line
        for line in section.splitlines()
        if line.startswith("|")
        and "fallback" in line.lower()
        and "hash" in line.lower()
    )
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_report_preserves_exclusive_nanosecond_arithmetic(
    sessions: list[dict[str, Any]],
) -> None:
    report = analyze_profile_sessions(sessions)
    text = render_profile_report(report)
    reference = _section(text, "Same-Q8 reference")
    # Each reference cached step contains 22 hashes at 1,000 ns apiece,
    # plus 273 ns in all other exclusive categories.
    row = _hash_row(reference)
    assert row[1:] == ["0.022", "98.77%", "98.77%", "98.77%"]
    assert "Median ms/cached step" in reference
    assert "1.500" in _section(text, "Prefill and observer context")
    assert "diagnostic" in text.lower()
    assert "neither native category is kernel-only time" in text.lower()
    assert not re.search(r"\d+(?:\.\d+)?[x\u00d7]\s+(?:speedup|faster)", text.lower())
    assert "p0 complete" not in text.lower()


def test_report_is_deterministic_and_leaves_analysis_unchanged(
    sessions: list[dict[str, Any]],
) -> None:
    report = analyze_profile_sessions(sessions)
    original = copy.deepcopy(report)
    expected = render_profile_report(report)
    assert render_profile_report(report) == expected
    reordered = json.loads(json.dumps(list(reversed(sessions)), sort_keys=True))
    assert render_profile_report(analyze_profile_sessions(reordered)) == expected
    assert report == original
    assert expected.endswith("\n")


def test_report_does_not_publish_prompts_commands_or_private_paths(
    sessions: list[dict[str, Any]],
) -> None:
    secret = "PRIVATE_PROMPT_<script>alert(1)</script>_DO_NOT_PUBLISH"
    private_command = "/workspace/private-capture-command"
    for document in sessions:
        document["workload"]["prompt"] = secret
        document["command_line"] = [private_command, secret]
    report = analyze_profile_sessions(sessions)
    report["input_captures"] = [
        {"session_index": index, "sha256": str(index + 2) * 64} for index in range(3)
    ]
    text = render_profile_report(report)
    assert secret not in text
    assert private_command not in text
    assert "<script>" not in text
    assert "1" * 40 in text
    assert all(str(index + 2) * 64 in text for index in range(3))


def test_per_step_cost_divides_by_observed_cached_steps(
    sessions: list[dict[str, Any]],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_model = test_profile_capture.FakeModel
    monkeypatch.setattr(
        test_profile_capture,
        "FakeModel",
        lambda: original_model(stop_after_cached=False),
    )
    documents = []
    for index, template in enumerate(sessions):
        document, _model, _installation = test_profile_capture._run(
            tmp_path_factory.mktemp(f"report-multistep-{index}"), session_index=index
        )
        for field in (
            "model_identity",
            "source",
            "environment",
            "process_id",
            "command_line",
        ):
            document[field] = copy.deepcopy(template[field])
        for name in PATHS:
            document["runs"][name]["profile"]["trace"]["clock"] = copy.deepcopy(
                template["runs"][name]["profile"]["trace"]["clock"]
            )
        _retime_session(document, {"fallback_hash": 1_000})
        documents.append(document)
    report = analyze_profile_sessions(documents)
    reference = report["sessions"][0]["paths"]["same_q8_reference"]["cached_decode"]
    assert reference["steps"] == 3
    assert reference["buckets_ns"]["fallback_hash"] == 66_000
    text = render_profile_report(report)
    assert _hash_row(_section(text, "Same-Q8 reference"))[1] == "0.022"


def test_tied_top_categories_are_reported_as_a_group(
    sessions: list[dict[str, Any]],
) -> None:
    for document in sessions:
        _retime_session(document, {"fallback_hash": 1_000, "fallback_linear": 1_000})
    report = analyze_profile_sessions(sessions)
    reference = _section(render_profile_report(report), "Same-Q8 reference")
    assert "Full category ordering: stable across all three sessions." in reference
    assert "Largest observed cost: exact tie between" in reference
    assert "Fallback weight hashing = Fallback linear operation" in reference
    assert "no single leader" in reference


def test_stable_top_and_unstable_lower_order_are_distinguished(
    sessions: list[dict[str, Any]],
) -> None:
    _retime_session(sessions[2], {"fallback_hash": 1_000, "fallback_clone": 3})
    reference = _section(
        render_profile_report(analyze_profile_sessions(sessions)), "Same-Q8 reference"
    )
    assert "Full category ordering: unstable across all three sessions." in reference
    assert (
        "Largest observed cost: Fallback weight hashing in all three sessions."
        in reference
    )


def test_changed_top_category_does_not_become_a_stable_recommendation(
    sessions: list[dict[str, Any]],
) -> None:
    _retime_session(sessions[2], {"fallback_linear": 2_000})
    reference = _section(
        render_profile_report(analyze_profile_sessions(sessions)), "Same-Q8 reference"
    )
    assert "Full category ordering: unstable across all three sessions." in reference
    assert any(
        "top" in line.lower()
        and any(word in line.lower() for word in ("changes", "changed", "unstable"))
        for line in reference.splitlines()
    )
    assert "hash" in reference.lower()
    assert "linear" in reference.lower()


@pytest.mark.parametrize("one_session_nonzero", [False, True])
def test_zero_categories_are_omitted_only_when_zero_in_every_session(
    sessions: list[dict[str, Any]],
    one_session_nonzero: bool,
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
    if one_session_nonzero:
        _retime_session(
            sessions[2],
            {
                **dict.fromkeys(boundaries, 0),
                "input_preparation": 1,
                "fallback_hash": 1,
            },
        )
    text = render_profile_report(analyze_profile_sessions(sessions))
    reference = _section(text, "Same-Q8 reference")
    if one_session_nonzero:
        row = _hash_row(reference)
        assert row[1:] == ["0.000", "0.00%", "0.00%", "95.65%"]
    else:
        rows = [line for line in reference.splitlines() if line.startswith("|")]
        assert any("Input preparation" in line for line in rows)
        assert not any("Fallback weight hashing" in line for line in rows)
        assert "20" in reference and "zero" in reference.lower()
        preparation = next(line for line in rows if "Input preparation" in line)
        assert preparation.split("|")[2].strip() == "<0.001"
        assert "Median cached-step total: <0.001 ms." in reference
    assert not re.search(r"\bnan\b", text.lower())
    assert "infinity" not in text.lower()

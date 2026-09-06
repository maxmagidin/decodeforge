"""Keep the separately declared evaluation workload bounded and reproducible."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _spec() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    value: dict[str, Any] = json.loads(
        (root / "benchmarks/evaluation-v1/spec.json").read_text(encoding="utf-8")
    )
    return value


def test_evaluation_has_thirty_unique_predeclared_cases() -> None:
    cases = _spec()["cases"]
    assert len(cases) == 30
    assert len({case["id"] for case in cases}) == 30
    assert len({case["prompt"] for case in cases}) == 30
    assert {case["max_new_tokens"] for case in cases} == {16, 32, 64}
    for case in cases:
        assert case["category"].strip()
        assert 0 < len(case["prompt"]) <= 4096
        assert 0 < len(case["reference_text"]) <= 4096


def test_performance_cases_are_a_fixed_distinct_subset() -> None:
    spec = _spec()
    selected = spec["performance_case_ids"]
    assert len(selected) == len(set(selected)) == 3
    assert set(selected) <= {case["id"] for case in spec["cases"]}
    cases = {case["id"]: case for case in spec["cases"]}
    assert {cases[case_id]["max_new_tokens"] for case_id in selected} == {16, 32, 64}

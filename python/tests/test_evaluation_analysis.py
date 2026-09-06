from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "analyze_evaluation.py"
_LOADER = importlib.util.spec_from_file_location("analyze_evaluation", _SCRIPT)
assert _LOADER and _LOADER.loader
analyze = importlib.util.module_from_spec(_LOADER)
_LOADER.loader.exec_module(analyze)


def _layers(path: str, count: int = 2) -> list[dict[str, int]]:
    native = count - 1 if path == "hybrid_native" else 0
    fallback = 1 if path == "hybrid_native" else count
    return [
        {
            "layer": i,
            "forward": count,
            "native_attempt": native,
            "native_success": native,
            "fallback_attempt": fallback,
            "fallback_success": fallback,
            "native_error": 0,
            "fallback_error": 0,
            "predispatch_error": 0,
            "rejected_closed": 0,
            "in_flight": 0,
        }
        for i in range(22)
    ]


def _common(mode: str, digest: str, index: int = 0) -> dict[str, Any]:
    return {
        "mode": mode,
        "session_index": index,
        "spec_sha256": digest,
        "accepted": True,
        "failures": [],
        "source": {"git_revision": "clean-rev", "git_dirty": False},
        "bridge_library_sha256": "bridge",
        "model_files": {"model": "pinned"},
        "asset_inventory_identity": "assets",
        "restoration": {
            "original_modules_restored": True,
            "counters": {
                "installed_modules": 0,
                "restored_modules": 22,
                "live_adapters": 0,
                "in_flight": 0,
            },
        },
    }


def _sample(
    tokens: list[int], counter: list[dict[str, int]] | None = None
) -> dict[str, Any]:
    decode = [10, 11]
    return {
        "generated_token_ids": tokens,
        "decode_ns": decode,
        "prefill_ns": 10,
        "total_generation_ns": 40,
        "counter_delta": counter or [],
    }


def _fixture() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str]:
    spec = {
        "cases": [
            {"id": "c0", "max_new_tokens": 4},
            {"id": "c1", "max_new_tokens": 4},
        ],
        "performance_case_ids": ["c0", "c1"],
    }
    digest = hashlib.sha256(b"spec").hexdigest()
    cases = []
    for cid, fp_nll, q8_nll, targets in (
        ("c0", [0.0, 4.0], [0.0, 6.0], [7, 8]),
        ("c1", [1.0] * 4, [2.0] * 4, [9, 10, 11, 12]),
    ):
        cases.append(
            {
                "id": cid,
                "status": "passed",
                "max_new_tokens": 4,
                "fp32": {"generated_token_ids": [20, 21]},
                "same_q8_reference": {"token_ids": [1, 2, 20, 21]},
                "hybrid_native": {
                    "token_ids": [1, 2, 20, 21],
                    "generated_token_ids": [20, 21],
                    "stop_reason": "eos",
                },
                "counter_deltas": {
                    "same_q8_reference": _layers("same_q8_reference"),
                    "hybrid_native": _layers("hybrid_native"),
                },
                "comparison": {
                    "passed": True,
                    "compared_steps": 2,
                    "logit_checks": [
                        {"finite": True, "within_tolerance": True, "max_abs_diff": 0.1},
                        {"finite": True, "within_tolerance": True, "max_abs_diff": 0.2},
                    ],
                },
                "quality_fp32": {
                    "target_token_ids": targets,
                    "per_token_nll": fp_nll,
                    "argmax_token_ids": targets,
                },
                "quality_same_q8": {
                    "target_token_ids": targets,
                    "per_token_nll": q8_nll,
                    "argmax_token_ids": targets,
                },
            }
        )
    correctness = _common("correctness", digest)
    correctness["cases"] = cases
    performance = []
    for index in range(3):
        session = _common("performance", digest, index)
        rows = []
        for cid in ("c0", "c1"):
            rows.append(
                {
                    "case": {"id": cid},
                    "paths": {
                        "fp32": {
                            "warmup_generations": 1,
                            "measured_generations": [
                                _sample([20, 21, 22]) for _ in range(3)
                            ],
                        },
                        "same_q8_reference": {
                            "warmup_generations": 1,
                            "measured_generations": [
                                _sample([20, 21, 22], _layers("same_q8_reference", 3))
                                for _ in range(3)
                            ],
                        },
                        "hybrid_native": {
                            "warmup_generations": 1,
                            "measured_generations": [
                                _sample([20, 21, 22], _layers("hybrid_native", 3))
                                for _ in range(3)
                            ],
                        },
                    },
                }
            )
        session["performance"] = rows
        performance.append(session)
    return correctness, performance, spec, digest


def test_summary_uses_token_weighted_nll_and_fp32_agreement() -> None:
    correctness, performance, spec, digest = _fixture()
    result = analyze.summarize(correctness, performance, spec, digest)
    sensitivity = result["quantization_sensitivity"]
    assert sensitivity["reference_tokens"] == 6
    assert sensitivity["fp32_mean_nll"] == pytest.approx(8 / 6)
    assert sensitivity["same_q8_mean_nll"] == pytest.approx(14 / 6)
    assert sensitivity["q8_minus_fp32_mean_nll"] == pytest.approx(14 / 6 - 8 / 6)
    assert sensitivity["argmax_agreement_with_fp32"] == 1.0
    assert sensitivity["exact_greedy_sequences_with_fp32"] == 2
    assert len(result["performance_by_session_case_path"]) == 18


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("parity", "token mismatch"),
        ("session", "three performance sessions"),
        ("digest", "spec digest mismatch"),
        ("counter", "layer dispatch mismatch"),
        ("dirty", "dirty producer"),
    ],
)
def test_summary_rejects_incomplete_or_tampered_evidence(
    mutation: str, message: str
) -> None:
    correctness, performance, spec, digest = _fixture()
    correctness = copy.deepcopy(correctness)
    performance = copy.deepcopy(performance)
    if mutation == "parity":
        correctness["cases"][0]["hybrid_native"]["token_ids"][-1] = 99
    elif mutation == "session":
        performance.pop()
    elif mutation == "digest":
        digest = "0" * 64
    elif mutation == "counter":
        correctness["cases"][0]["counter_deltas"]["same_q8_reference"][0]["forward"] = 1
    else:
        correctness["source"]["git_dirty"] = True
    with pytest.raises(ValueError, match=message):
        analyze.summarize(correctness, performance, spec, digest)


def test_cli_verifies_summary_without_writing_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    correctness, performance, spec, _ = _fixture()
    spec_text = json.dumps(spec)
    digest = hashlib.sha256(spec_text.encode()).hexdigest()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec_text)
    paths = [
        tmp_path / "correctness.json",
        *[tmp_path / f"perf-{i}.json" for i in range(3)],
    ]
    for path, data in zip(paths, [correctness, *performance], strict=True):
        data["spec_sha256"] = digest
        path.write_text(json.dumps(data))
    summary_path = tmp_path / "summary.json"
    base = [
        "analyze_evaluation.py",
        "--correctness",
        str(paths[0]),
        "--performance",
        *[str(path) for path in paths[1:]],
        "--spec",
        str(spec_path),
    ]
    monkeypatch.setattr(sys, "argv", [*base, "--output", str(summary_path)])
    assert analyze.main() == 0
    original = summary_path.read_bytes()
    monkeypatch.setattr(sys, "argv", [*base, "--verify-summary", str(summary_path)])
    assert analyze.main() == 0
    assert summary_path.read_bytes() == original
    summary = json.loads(original)
    summary["correctness"]["cases"] = 999
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(SystemExit) as caught:
        analyze.main()
    assert caught.value.code == 2

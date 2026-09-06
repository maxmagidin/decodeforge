#!/usr/bin/env python3
"""Recompute a bounded evaluation summary from retained non-G3 observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values) and all(math.isfinite(x) for x in values), "invalid samples")
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def _counters(values: Sequence[Mapping[str, Any]], path: str, count: int) -> None:
    _require(len(values) == 22, "missing layer counters")
    native = count - 1 if path == "hybrid_native" else 0
    fallback = 1 if path == "hybrid_native" else count
    for index, layer in enumerate(values):
        expected = {
            "layer": index,
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
        _require(dict(layer) == expected, "layer dispatch mismatch")


def summarize(
    correctness: Mapping[str, Any],
    performance: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    spec_digest: str,
) -> dict[str, Any]:
    """Validate basic completeness and derive descriptive, not causal, metrics."""

    _require(len(performance) == 3, "three performance sessions required")
    _require(
        {item["session_index"] for item in performance} == {0, 1, 2}, "session IDs"
    )
    for item in [correctness, *performance]:
        _require(item["spec_sha256"] == spec_digest, "spec digest mismatch")
        _require(item["accepted"] is True and not item["failures"], "rejected evidence")
        _require(item["source"]["git_dirty"] is False, "dirty producer")
        _require(
            item["source"]["git_revision"] == correctness["source"]["git_revision"],
            "producer revisions differ",
        )
        final = item["restoration"]
        _require(final["original_modules_restored"] is True, "modules not restored")
        for key, expected in (
            ("installed_modules", 0),
            ("restored_modules", 22),
            ("live_adapters", 0),
            ("in_flight", 0),
        ):
            _require(final["counters"][key] == expected, f"cleanup {key}")
        _require(
            item["bridge_library_sha256"] == correctness["bridge_library_sha256"],
            "bridge differs across sessions",
        )
        _require(item["model_files"] == correctness["model_files"], "model differs")
        _require(
            item["asset_inventory_identity"] == correctness["asset_inventory_identity"],
            "Q8 assets differ",
        )
    _require(correctness["mode"] == "correctness", "wrong correctness mode")
    _require(
        all(item["mode"] == "performance" for item in performance),
        "wrong performance mode",
    )
    cases = correctness["cases"]
    expected_ids = [case["id"] for case in spec["cases"]]
    _require([case["id"] for case in cases] == expected_ids, "case coverage mismatch")
    fp32_nll: list[float] = []
    q8_nll: list[float] = []
    argmax_matches = 0
    exact_fp32 = 0
    generation_tokens = 0
    stop_reasons: dict[str, int] = {}
    max_logit_error = 0.0
    for case in cases:
        _require(case["status"] == "passed", f"failed case {case['id']}")
        ref, native = case["same_q8_reference"], case["hybrid_native"]
        _require(ref["token_ids"] == native["token_ids"], "token mismatch")
        count = len(native["generated_token_ids"])
        _require(
            count >= 2 and count <= case["max_new_tokens"], "native coverage bound"
        )
        comparison = case["comparison"]
        for path in ("same_q8_reference", "hybrid_native"):
            _counters(case["counter_deltas"][path], path, count)
        _require(
            comparison["passed"] is True and comparison["compared_steps"] == count,
            "incomplete logit evidence",
        )
        for check in comparison["logit_checks"]:
            _require(
                check["finite"] is True and check["within_tolerance"] is True,
                "logit tolerance failure",
            )
            max_logit_error = max(max_logit_error, check["max_abs_diff"])
        exact_fp32 += (
            case["fp32"]["generated_token_ids"] == native["generated_token_ids"]
        )
        generation_tokens += count
        stop_reasons[native["stop_reason"]] = (
            stop_reasons.get(native["stop_reason"], 0) + 1
        )
        original, quantized = case["quality_fp32"], case["quality_same_q8"]
        _require(
            original["target_token_ids"] == quantized["target_token_ids"],
            "quality targets differ",
        )
        targets = original["target_token_ids"]
        for result in (original, quantized):
            _require(
                len(result["per_token_nll"])
                == len(result["argmax_token_ids"])
                == len(targets)
                > 0,
                "incomplete quality metrics",
            )
            _require(
                all(math.isfinite(x) and x >= 0 for x in result["per_token_nll"]),
                "invalid NLL",
            )
        fp32_nll.extend(original["per_token_nll"])
        q8_nll.extend(quantized["per_token_nll"])
        argmax_matches += sum(
            a == b
            for a, b in zip(
                original["argmax_token_ids"], quantized["argmax_token_ids"], strict=True
            )
        )
    timing: list[dict[str, Any]] = []
    for session in performance:
        rows = session["performance"]
        _require(
            [row["case"]["id"] for row in rows] == spec["performance_case_ids"],
            "performance cases",
        )
        for row in rows:
            paths = row["paths"]
            _require(
                set(paths) == {"fp32", "same_q8_reference", "hybrid_native"},
                "performance paths",
            )
            reference_samples = paths["same_q8_reference"]["measured_generations"]
            for name, path in paths.items():
                samples = path["measured_generations"]
                _require(
                    path["warmup_generations"] == 1 and len(samples) == 3,
                    "sample count",
                )
                decode_rates: list[float] = []
                for index, sample in enumerate(samples):
                    tokens = sample["generated_token_ids"]
                    decode = sample["decode_ns"]
                    _require(len(decode) == len(tokens) - 1 > 0, "decode sample count")
                    _require(
                        all(type(n) is int and n > 0 for n in decode),
                        "invalid decode timing",
                    )
                    _require(
                        sample["total_generation_ns"]
                        >= sample["prefill_ns"] + sum(decode),
                        "invalid total timing",
                    )
                    if name == "hybrid_native":
                        _require(
                            tokens == reference_samples[index]["generated_token_ids"],
                            "performance token mismatch",
                        )
                    if name != "fp32":
                        _counters(sample["counter_delta"], name, len(tokens))
                    decode_rates.append(len(decode) * 1e9 / sum(decode))
                timing.append(
                    {
                        "session_index": session["session_index"],
                        "case_id": row["case"]["id"],
                        "path": name,
                        "prefill_ms": _stats([s["prefill_ns"] / 1e6 for s in samples]),
                        "decode_tokens_per_second": _stats(decode_rates),
                        "generation_seconds": _stats(
                            [s["total_generation_ns"] / 1e9 for s in samples]
                        ),
                        "generated_token_counts": [
                            len(s["generated_token_ids"]) for s in samples
                        ],
                    }
                )
    return {
        "format": "decodeforge_evaluation_summary_v1",
        "g3_evidence": False,
        "spec_sha256": spec_digest,
        "source": correctness["source"],
        "setup_and_memory_by_session": [
            {
                "session_index": session["session_index"],
                "timing": session.get("timing", {}),
            }
            for session in performance
        ],
        "correctness": {
            "cases": len(cases),
            "native_reference_exact": len(cases),
            "compared_generation_steps": generation_tokens,
            "max_logit_abs_error": max_logit_error,
            "stop_reasons": stop_reasons,
        },
        "quantization_sensitivity": {
            "reference_tokens": len(fp32_nll),
            "fp32_mean_nll": statistics.mean(fp32_nll),
            "same_q8_mean_nll": statistics.mean(q8_nll),
            "q8_minus_fp32_mean_nll": statistics.mean(q8_nll)
            - statistics.mean(fp32_nll),
            "argmax_agreement_with_fp32": argmax_matches / len(fp32_nll),
            "exact_greedy_sequences_with_fp32": exact_fp32,
            "interpretation": (
                "synthetic fixed-continuation sensitivity, not broad task accuracy"
            ),
        },
        "performance_by_session_case_path": timing,
        "limits": [
            "one physical host",
            "production guards and fallback clone/hash included",
            "FP32 output lengths can differ; total latency is not equal-token speedup",
            "three process clusters; no confidence interval or general engine claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--performance", type=Path, nargs=3, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    publication = parser.add_mutually_exclusive_group(required=True)
    publication.add_argument("--output", type=Path)
    publication.add_argument("--verify-summary", type=Path)
    args = parser.parse_args()
    try:
        spec_bytes = args.spec.read_bytes()
        summary = summarize(
            json.loads(args.correctness.read_text()),
            [json.loads(path.read_text()) for path in args.performance],
            json.loads(spec_bytes),
            hashlib.sha256(spec_bytes).hexdigest(),
        )
        if args.verify_summary is not None:
            _require(
                summary == json.loads(args.verify_summary.read_text()),
                "retained summary differs from recomputed observations",
            )
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.verify_summary is not None:
        print("evaluation-summary-verification: ok")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

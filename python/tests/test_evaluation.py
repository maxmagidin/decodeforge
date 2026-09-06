from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from decodeforge import evaluation as ev
from torch import nn


class TinyCachedModel(nn.Module):
    def __init__(self, tokens: list[int]) -> None:
        super().__init__()
        self.tokens = tokens
        self.calls: list[dict[str, Any]] = []
        self.cache = object()

    def forward(self, **kwargs: Any) -> Any:
        index = len(self.calls)
        self.calls.append(kwargs)
        assert not torch.is_grad_enabled()
        assert kwargs["use_cache"] is True
        assert kwargs["return_dict"] is True
        if index:
            assert kwargs["past_key_values"] is self.cache
        logits = torch.full((1, kwargs["input_ids"].shape[1], 8), -2.0)
        logits[0, -1, self.tokens[index]] = 2
        return SimpleNamespace(logits=logits, past_key_values=self.cache)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[3, 4, 5]])
    return ids, torch.ones_like(ids)


def _run(ids: list[int], logits: list[torch.Tensor]) -> ev.CachedGeneration:
    return ev.CachedGeneration([3, 4, *ids], ids, "max_new_tokens", False, logits, [])


def test_cached_loop_preserves_prefill_cache_mask_and_eos() -> None:
    model = TinyCachedModel([1, 2, 7])
    inp, mask = _inputs()
    result = ev.generate_cached(
        model, SimpleNamespace(eos_token_id=7), inp, mask, 8, retain_logits=True
    )
    assert result.generated_ids == [1, 2, 7]
    assert result.all_ids == [3, 4, 5, 1, 2, 7]
    assert result.stop_reason == "eos"
    assert result.stopped_by_eos
    assert len(result.logits) == 3
    assert [call["input_ids"].tolist() for call in model.calls] == [
        [[3, 4, 5]],
        [[1]],
        [[2]],
    ]
    assert [call["attention_mask"].shape[1] for call in model.calls] == [3, 4, 5]
    assert inp.tolist() == [[3, 4, 5]]
    assert mask.tolist() == [[1, 1, 1]]


def test_first_token_eos_is_not_suppressed_for_coverage() -> None:
    model = TinyCachedModel([7])
    result = ev.generate_cached(model, SimpleNamespace(eos_token_id=7), *_inputs(), 64)
    assert result.generated_ids == [7]
    assert len(model.calls) == 1


def test_token_limit_has_no_extra_forward_or_sentence_stopping() -> None:
    model = TinyCachedModel([1, 2, 3])
    result = ev.generate_cached(model, SimpleNamespace(eos_token_id=7), *_inputs(), 2)
    assert result.generated_ids == [1, 2]
    assert result.stop_reason == "max_new_tokens"
    assert result.logits == []
    assert len(model.calls) == 2


@pytest.mark.parametrize("cap", [0, 65, True, 1.5])
def test_bad_generation_cap_is_rejected_before_forward(cap: Any) -> None:
    model = TinyCachedModel([1])
    with pytest.raises(ev.EvaluationError):
        ev.generate_cached(model, SimpleNamespace(eos_token_id=7), *_inputs(), cap)
    assert model.calls == []


def test_identical_tokens_do_not_hide_logit_failure() -> None:
    result = ev.compare_runs(
        _run([1], [torch.zeros(3)]), _run([1], [torch.tensor([0.0, 0.1, 0.0])])
    )
    assert result["token_ids_exact"]
    assert not result["passed"]


def test_divergence_compares_the_last_shared_input_prefix_only() -> None:
    result = ev.compare_runs(
        _run([1, 2, 3], [torch.zeros(4)] * 3),
        _run([1, 3, 2], [torch.zeros(4), torch.ones(4), torch.full((4,), 99.0)]),
    )
    assert not result["passed"]
    assert result["compared_steps"] == 2
    assert len(result["logit_checks"]) == 2


def test_missing_logits_cannot_pass_an_exact_token_run() -> None:
    result = ev.compare_runs(
        _run([1, 2], [torch.zeros(3)]), _run([1, 2], [torch.zeros(3)])
    )
    assert not result["passed"]


def test_timed_loop_has_exactly_one_decode_sample_per_cached_forward() -> None:
    model = TinyCachedModel([1, 2, 7])
    result, timing = ev._timed_generation(
        model, SimpleNamespace(eos_token_id=7), *_inputs(), 8
    )
    assert result.generated_ids == [1, 2, 7]
    assert len(timing["decode_ns"]) == 2
    assert timing["prefill_ns"] > 0
    assert all(value > 0 for value in timing["decode_ns"])
    assert timing["total_generation_ns"] >= timing["prefill_ns"] + sum(
        timing["decode_ns"]
    )
    assert result.logits == []
    assert result.chosen_logprobs == []


def test_timed_first_eos_has_no_decode_throughput_sample() -> None:
    _, timing = ev._timed_generation(
        TinyCachedModel([7]), SimpleNamespace(eos_token_id=7), *_inputs(), 64
    )
    assert timing["decode_ns"] == []


def test_teacher_forcing_uses_causal_shift_and_no_extra_eos() -> None:
    class TeacherModel(nn.Module):
        def forward(self, **kwargs: Any) -> Any:
            assert kwargs["input_ids"].tolist() == [[3, 4, 1, 2]]
            assert kwargs["use_cache"] is False
            logits = torch.zeros(1, 4, 8)
            logits[0, 1, 1] = 3
            logits[0, 2, 2] = 4
            return SimpleNamespace(logits=logits)

    result = ev._teacher_force(TeacherModel(), [3, 4], [1, 2])
    assert result["target_token_ids"] == [1, 2]
    assert result["argmax_token_ids"] == [1, 2]
    assert len(result["per_token_nll"]) == 2
    assert result["per_token_nll"][1] < result["per_token_nll"][0]


def _spec() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "benchmarks/evaluation-v1/spec.json").read_text())  # type: ignore[no-any-return]


def test_real_evaluation_spec_validates() -> None:
    assert len(ev.validate_spec(_spec())["cases"]) == 30


@pytest.mark.parametrize(
    "mutation", ["duplicate_id", "duplicate_prompt", "cap", "reference", "perf"]
)
def test_malformed_spec_rejected(mutation: str) -> None:
    spec = copy.deepcopy(_spec())
    if mutation == "duplicate_id":
        spec["cases"][1]["id"] = spec["cases"][0]["id"]
    elif mutation == "duplicate_prompt":
        spec["cases"][1]["prompt"] = spec["cases"][0]["prompt"]
    elif mutation == "cap":
        spec["cases"][0]["max_new_tokens"] = True
    elif mutation == "reference":
        spec["cases"][0]["reference_text"] = " "
    else:
        spec["performance_case_ids"] = ["missing"] * 3
    with pytest.raises(ev.EvaluationError):
        ev.validate_spec(spec)


def test_result_writer_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    ev.write_evaluation_json(path, {"accepted": False})
    before = path.read_bytes()
    with pytest.raises(Exception, match="exists"):
        ev.write_evaluation_json(path, {"accepted": True})
    assert path.read_bytes() == before


def test_noncanonical_spec_is_rejected_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    spec["unrecognized_policy"] = "must not silently ignore this"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(
        ev,
        "_source_identity",
        lambda _path: {"git_revision": "a" * 40, "git_dirty": False},
    )
    request = ev.EvaluationRequest(path, tmp_path, tmp_path, tmp_path, "0" * 64)
    with pytest.raises(ev.EvaluationError, match="committed evaluation-v1"):
        ev.run_evaluation(request)


def test_finite_extreme_comparison_is_json_serializable() -> None:
    largest = torch.finfo(torch.float32).max
    result = ev.compare_runs(
        _run([1], [torch.tensor([largest])]),
        _run([1], [torch.tensor([-largest])]),
    )
    assert not result["passed"]
    json.dumps(result, allow_nan=False)

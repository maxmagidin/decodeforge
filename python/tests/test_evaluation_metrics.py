from __future__ import annotations

import math

import pytest
import torch
from decodeforge.evaluation_metrics import compare_logits, token_metrics


def test_uniform_logits_have_known_nll() -> None:
    result = token_metrics(torch.zeros(2, 4), torch.tensor([0, 3]))
    assert result["per_token_nll"] == pytest.approx([math.log(4)] * 2)
    assert result["mean_nll"] == pytest.approx(math.log(4))
    assert result["target_token_ids"] == [0, 3]
    assert result["argmax_token_ids"] == [0, 0]


def test_likelihood_uses_declared_targets_not_argmax() -> None:
    logits = torch.tensor([[0.0, math.log(3)], [math.log(3), 0.0]])
    result = token_metrics(logits, torch.tensor([0, 0]))
    assert result["per_token_nll"] == pytest.approx([math.log(4), -math.log(0.75)])
    assert result["argmax_token_ids"] == [1, 0]


@pytest.mark.parametrize("target", [-1, 2])
def test_out_of_vocabulary_targets_rejected(target: int) -> None:
    with pytest.raises(ValueError, match="align"):
        token_metrics(torch.zeros(1, 2), torch.tensor([target]))


def test_metric_shape_mismatch_and_noninteger_targets_rejected() -> None:
    with pytest.raises(ValueError, match="align"):
        token_metrics(torch.zeros(2, 3), torch.tensor([0]))
    with pytest.raises(ValueError, match="align"):
        token_metrics(torch.zeros(1, 3), torch.tensor([0.0]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        token_metrics(torch.tensor([[value, 0.0]]), torch.tensor([0]))
    with pytest.raises(ValueError, match="finite"):
        compare_logits([torch.zeros(2)], [torch.tensor([value, 0.0])])


def test_tolerance_is_elementwise_and_reference_relative() -> None:
    reference = [torch.tensor([0.0, 100.0])]
    assert compare_logits(reference, [torch.tensor([0.0009, 100.09])])["passed"]
    failure = compare_logits(reference, [torch.tensor([0.002, 100.0])])
    assert not failure["passed"]
    assert failure["steps"] == 1
    assert failure["max_tolerance_excess"] == pytest.approx(0.001)


def test_exact_tolerance_boundary_passes() -> None:
    result = compare_logits(
        [torch.tensor([0.0])], [torch.tensor([0.5])], atol=0.5, rtol=0
    )
    assert result["passed"]
    assert result["max_tolerance_excess"] == 0


def test_logit_shape_mismatch_cannot_broadcast() -> None:
    with pytest.raises(ValueError, match="shape"):
        compare_logits([torch.zeros(3)], [torch.zeros(1)])
    with pytest.raises(ValueError, match="equally long"):
        compare_logits([torch.zeros(3)], [])
    with pytest.raises(ValueError, match="equally long"):
        compare_logits([], [])


@pytest.mark.parametrize("tolerance", [-0.1, float("nan"), float("inf")])
def test_invalid_tolerances_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerances"):
        compare_logits([torch.zeros(1)], [torch.zeros(1)], atol=tolerance)


def test_metrics_reject_wrong_model_dtype() -> None:
    with pytest.raises(ValueError, match="FP32"):
        token_metrics(torch.zeros(1, 3, dtype=torch.float64), torch.tensor([0]))


def test_finite_extreme_logits_do_not_overflow_comparison() -> None:
    largest = torch.finfo(torch.float32).max
    result = compare_logits([torch.tensor([largest])], [torch.tensor([-largest])])
    assert not result["passed"]
    assert math.isfinite(result["max_abs_error"])

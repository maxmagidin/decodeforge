"""Small, independently tested metrics for the non-G3 evaluation protocol."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


def _finite_logits(value: torch.Tensor) -> None:
    if (
        value.device.type != "cpu"
        or value.dtype is not torch.float32
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("evaluation logits must be nonempty finite CPU FP32")


def token_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Score already aligned next-token logits against fixed target tokens.

    This measures likelihood of one declared continuation, not task accuracy.
    Callers own the causal shift and must retain the identical target sequence
    for both paths. Float64 scoring reduces metric-rounding error without
    changing the model's FP32 computation.
    """

    _finite_logits(logits)
    if (
        logits.ndim != 2
        or targets.ndim != 1
        or targets.device.type != "cpu"
        or targets.dtype is not torch.int64
        or logits.shape[0] != targets.numel()
        or bool((targets < 0).any())
        or bool((targets >= logits.shape[1]).any())
    ):
        raise ValueError("target tokens must align with [tokens, vocabulary] logits")
    log_probs = logits.to(torch.float64).log_softmax(dim=-1)
    nll = -log_probs.gather(1, targets[:, None]).squeeze(1)
    if not bool(torch.isfinite(nll).all()):
        raise ValueError("nonfinite token likelihood")
    return {
        "target_token_ids": targets.tolist(),
        "per_token_nll": nll.tolist(),
        "mean_nll": float(nll.mean()),
        "argmax_token_ids": logits.argmax(dim=-1).tolist(),
    }


def compare_logits(
    reference: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    atol: float = 0.001,
    rtol: float = 0.001,
) -> dict[str, Any]:
    """Compare logits only after callers establish identical token prefixes."""

    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    if not reference or len(reference) != len(actual):
        raise ValueError("logit sequences must be nonempty and equally long")
    maximum_error = 0.0
    maximum_excess = 0.0
    for expected, observed in zip(reference, actual, strict=True):
        _finite_logits(expected)
        _finite_logits(observed)
        if expected.ndim != 1 or expected.shape != observed.shape:
            raise ValueError("each logit pair must have the same vocabulary shape")
        # Compute differences in float64 to avoid overflow for finite FP32
        # extremes and to apply the declared reference-relative tolerance.
        baseline = expected.to(torch.float64)
        difference = (observed.to(torch.float64) - baseline).abs()
        excess = difference - (atol + rtol * baseline.abs())
        maximum_error = max(maximum_error, float(difference.max()))
        maximum_excess = max(maximum_excess, float(excess.max()))
    return {
        "passed": maximum_excess <= 0,
        "steps": len(reference),
        "max_abs_error": maximum_error,
        "max_tolerance_excess": maximum_excess,
    }

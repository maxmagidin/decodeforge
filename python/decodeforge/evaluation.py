"""Bounded correctness and performance evaluation for the TinyLlama adapter.

This is intentionally a small evaluation harness, separate from the frozen G3
experiment runner.  It uses the presentation demo's pinned local component
loaders and installation boundary, but performs greedy decoding explicitly so
the measured region is auditable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

import torch
from torch import nn

from . import presentation_demo as _presentation
from .evaluation_metrics import compare_logits, token_metrics
from .qproj_adapter import QProjExecutionMode
from .qproj_model import tinyllama_qproj_paths

MAX_CASES: Final = 30
MAX_NEW_TOKENS: Final = 64
PERFORMANCE_CASES: Final = 3
LOGIT_ATOL: Final = 1e-3
LOGIT_RTOL: Final = 1e-3


class EvaluationError(RuntimeError):
    """The evaluation could not safely produce an evidence bundle."""


class InstallationLike(Protocol):
    @property
    def inventory(self) -> Any: ...

    @property
    def counters(self) -> Any: ...

    @property
    def closed(self) -> bool: ...

    def set_execution_mode(self, mode: QProjExecutionMode) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EvaluationRequest:
    spec_path: Path
    model_directory: Path
    asset_directory: Path
    bridge_library: Path
    bridge_sha256: str
    mode: str = "correctness"
    session_index: int = 0
    process_start_ns: int | None = None


@dataclass
class CachedGeneration:
    """One greedy generation, with optional transient per-step logits."""

    all_ids: list[int]
    generated_ids: list[int]
    stop_reason: str
    stopped_by_eos: bool
    logits: list[torch.Tensor]
    chosen_logprobs: list[float]

    @property
    def token_ids(self) -> list[int]:
        return self.all_ids


def validate_spec(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return an evaluation-v1 specification.

    The protocol deliberately validates the bounded shape here rather than
    trusting an external schema: malformed cases must be rejected before a
    model or bridge is loaded.
    """

    if not isinstance(document, Mapping):
        raise EvaluationError("spec must be an object")
    protocol = document.get("protocol_id", document.get("protocol"))
    if protocol is not None and not isinstance(protocol, str):
        raise EvaluationError("spec protocol_id must be a string")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != MAX_CASES:
        raise EvaluationError("spec must contain exactly 30 cases")
    seen: set[str] = set()
    seen_prompts: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise EvaluationError(f"case {index} must be an object")
        case_id = case.get("id")
        category = case.get("category")
        prompt = case.get("prompt")
        maximum = case.get("max_new_tokens")
        reference = case.get("reference_text")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise EvaluationError(f"case {index} has an invalid or duplicate id")
        if not isinstance(category, str) or not category:
            raise EvaluationError(f"case {case_id} has an invalid category")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > 4096
            or prompt in seen_prompts
        ):
            raise EvaluationError(f"case {case_id} has an invalid prompt")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= MAX_NEW_TOKENS
        ):
            raise EvaluationError(f"case {case_id} max_new_tokens must be in 1..64")
        if not isinstance(reference, str) or not reference.strip():
            raise EvaluationError(
                f"case {case_id} reference_text must be nonempty text"
            )
        seen.add(case_id)
        seen_prompts.add(prompt)
    perf = document.get("performance_case_ids")
    if not isinstance(perf, list) or len(perf) != PERFORMANCE_CASES:
        raise EvaluationError("performance_case_ids must contain exactly 3 ids")
    if any(not isinstance(value, str) or value not in seen for value in perf):
        raise EvaluationError("performance_case_ids must refer to cases")
    if len(set(perf)) != PERFORMANCE_CASES:
        raise EvaluationError("performance_case_ids must be distinct")
    # When the canonical protocol sections are present, do not silently run
    # with a contradictory setting.  Keeping these checks conditional also
    # leaves the pure helper useful with a compact test specification.
    generation = document.get("generation")
    if isinstance(generation, Mapping):
        expected_generation = {
            "device": "cpu",
            "model_compute_dtype": "float32",
            "eval_mode": True,
            "torch_inference_mode": True,
            "decoding": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "min_new_tokens": 0,
            "stop_after_sentence": False,
            "seed": 0,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        }
        for key, expected in expected_generation.items():
            if key in generation and generation[key] != expected:
                raise EvaluationError(f"generation.{key} conflicts with protocol")
    model = document.get("model")
    if isinstance(model, Mapping) and model.get("local_files_only") is False:
        raise EvaluationError("model must use local files only")
    return dict(document)


def _output_parts(value: Any) -> tuple[torch.Tensor, Any]:
    logits = (
        value.get("logits")
        if isinstance(value, Mapping)
        else getattr(value, "logits", None)
    )
    past = (
        value.get("past_key_values")
        if isinstance(value, Mapping)
        else getattr(value, "past_key_values", None)
    )
    if not isinstance(logits, torch.Tensor) or past is None:
        raise EvaluationError("model output lacks logits or past_key_values")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] < 1:
        raise EvaluationError("model logits have an invalid shape")
    if logits.device.type != "cpu" or logits.dtype is not torch.float32:
        raise EvaluationError("model logits must be CPU floating-point")
    if not bool(torch.isfinite(logits).all().item()):
        raise EvaluationError("model produced nonfinite logits")
    return logits, past


def _eos_ids(tokenizer: Any) -> set[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, bool):
        raise EvaluationError("tokenizer eos_token_id is invalid")
    if isinstance(value, int):
        return {value}
    try:
        result = {int(item) for item in value}
    except (TypeError, ValueError) as error:
        raise EvaluationError("tokenizer eos_token_id is invalid") from error
    if any(item < 0 for item in result):
        raise EvaluationError("tokenizer eos_token_id is invalid")
    return result


def generate_cached(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    *,
    retain_logits: bool = False,
) -> CachedGeneration:
    """Greedily decode using one prefill and explicit cached decode forwards."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= MAX_NEW_TOKENS
    ):
        raise EvaluationError("max_new_tokens must be in 1..64")
    if (
        input_ids.shape != attention_mask.shape
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise EvaluationError("generation inputs must be one matching CPU sequence")
    if input_ids.device.type != "cpu" or input_ids.dtype is not torch.int64:
        raise EvaluationError("generation input_ids must be CPU int64")
    eos = _eos_ids(tokenizer)
    ids = input_ids.detach().clone().contiguous()
    mask = attention_mask.detach().clone().contiguous()
    prompt_ids = [int(value) for value in ids[0].tolist()]
    generated: list[int] = []
    retained: list[torch.Tensor] = []
    chosen_logprobs: list[float] = []
    with torch.inference_mode():
        output = model(
            input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True
        )
        logits, past = _output_parts(output)
        for step in range(max_new_tokens):
            row = logits[0, -1, :]
            token = int(torch.argmax(row).item())
            probability = torch.log_softmax(row, dim=-1)[token]
            chosen_logprobs.append(float(probability.item()))
            if retain_logits:
                retained.append(
                    row.detach().to(device="cpu", dtype=torch.float32).clone()
                )
            generated.append(token)
            if token in eos:
                return CachedGeneration(
                    prompt_ids + generated,
                    generated,
                    "eos",
                    True,
                    retained,
                    chosen_logprobs,
                )
            if step + 1 == max_new_tokens:
                break
            ids = torch.tensor([[token]], dtype=torch.int64, device="cpu")
            mask = torch.cat((mask, torch.ones((1, 1), dtype=torch.int64)), dim=1)
            output = model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits, past = _output_parts(output)
    return CachedGeneration(
        prompt_ids + generated,
        generated,
        "max_new_tokens",
        False,
        retained,
        chosen_logprobs,
    )


def _relative_diff(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference = reference.to(torch.float64)
    candidate = candidate.to(torch.float64)
    denominator = torch.maximum(
        reference.abs(), torch.tensor(1e-30, dtype=reference.dtype)
    )
    return float(((candidate - reference).abs() / denominator).max().item())


def compare_runs(
    reference: CachedGeneration | Mapping[str, Any],
    candidate: CachedGeneration | Mapping[str, Any],
    *,
    atol: float = LOGIT_ATOL,
    rtol: float = LOGIT_RTOL,
) -> dict[str, Any]:
    """Compare native/reference runs on the prefixes they actually share."""

    ref_ids = (
        reference.generated_ids
        if isinstance(reference, CachedGeneration)
        else list(reference["generated_ids"])
    )
    cand_ids = (
        candidate.generated_ids
        if isinstance(candidate, CachedGeneration)
        else list(candidate["generated_ids"])
    )
    ref_logits = (
        reference.logits
        if isinstance(reference, CachedGeneration)
        else list(reference.get("logits", ()))
    )
    cand_logits = (
        candidate.logits
        if isinstance(candidate, CachedGeneration)
        else list(candidate.get("logits", ()))
    )
    shared = 0
    while (
        shared < min(len(ref_ids), len(cand_ids))
        and ref_ids[shared] == cand_ids[shared]
    ):
        shared += 1
    checks: list[dict[str, Any]] = []
    # The logits at the first differing token still use the same prefix; all
    # later logits do not and must not be compared.
    comparable_steps = min(shared + 1, len(ref_logits), len(cand_logits))
    for index in range(comparable_steps):
        left, right = ref_logits[index], cand_logits[index]
        finite = bool(
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.device.type == "cpu"
            and right.device.type == "cpu"
            and left.dtype is torch.float32
            and right.dtype is torch.float32
            and left.ndim == 1
            and left.numel() > 0
            and left.shape == right.shape
            and bool(torch.isfinite(left).all().item())
            and bool(torch.isfinite(right).all().item())
        )
        max_abs: float | None = (
            float((right.double() - left.double()).abs().max().item())
            if finite
            else None
        )
        max_rel: float | None = _relative_diff(left, right) if finite else None
        allowed = atol + rtol * left.double().abs() if finite else None
        within = (
            finite
            and allowed is not None
            and bool(((right.double() - left.double()).abs() <= allowed).all().item())
        )
        checks.append(
            {
                "step_index": index,
                "finite": finite,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "within_tolerance": within,
            }
        )
    complete = len(ref_logits) == len(ref_ids) and len(cand_logits) == len(cand_ids)
    comparable = bool(checks) and len(checks) == comparable_steps
    metric_summary: dict[str, Any] | None = None
    if comparable:
        try:
            metric_summary = compare_logits(
                ref_logits[:comparable_steps],
                cand_logits[:comparable_steps],
                atol=atol,
                rtol=rtol,
            )
        except ValueError:
            comparable = False
    logit_pass = (
        comparable
        and metric_summary is not None
        and bool(metric_summary["passed"])
        and all(bool(item["within_tolerance"]) for item in checks)
    )
    return {
        "token_ids_exact": ref_ids == cand_ids,
        "shared_prefix_tokens": shared,
        "compared_steps": len(checks),
        "logit_checks": checks,
        "logit_summary": metric_summary,
        "logits_within_tolerance": logit_pass if comparable and complete else None,
        "logits_status": (
            "compared"
            if comparable and complete
            else "unavailable_incomplete_or_diverged_prefix"
        ),
        "passed": ref_ids == cand_ids and complete and logit_pass,
    }


def _counter_delta(before: Any, after: Any) -> list[dict[str, Any]]:
    return _presentation._counter_delta(
        _presentation._counter_summary(before), _presentation._counter_summary(after)
    )


def _reconcile(name: str, token_count: int, delta: Sequence[Mapping[str, Any]]) -> None:
    if len(delta) != 22:
        raise EvaluationError(f"{name} counter coverage is not 22 layers")
    if name == "hybrid_native" and token_count < 2:
        raise EvaluationError(
            "hybrid_native coverage incomplete: generation ended before cached decode"
        )
    native = max(token_count - 1, 0) if name == "hybrid_native" else 0
    fallback = 1 if name == "hybrid_native" else token_count
    for layer, values in enumerate(delta):
        expected = {
            "layer": layer,
            "forward": token_count,
            "native_attempt": native,
            "native_success": native,
            "native_error": 0,
            "fallback_attempt": fallback,
            "fallback_success": fallback,
            "fallback_error": 0,
            "predispatch_error": 0,
            "rejected_closed": 0,
            "in_flight": 0,
        }
        if dict(values) != expected:
            raise EvaluationError(
                f"{name} counter reconciliation failed at layer {layer}"
            )


def _check_installation_initial(installation: InstallationLike) -> dict[str, Any]:
    summary = _presentation._counter_summary(installation.counters)
    if (
        summary["installed_modules"] != 22
        or summary["restored_modules"] != 0
        or summary["live_adapters"] != 22
        or summary["in_flight"] != 0
        or summary["closed"]
        or len(summary["layers"]) != 22
        or any(
            layer["layer"] != index
            or layer["layer_path"] != tinyllama_qproj_paths()[index]
            or layer["closed"]
            or layer["in_flight"]
            for index, layer in enumerate(summary["layers"])
        )
    ):
        raise EvaluationError("initial installation must own all 22 layers")
    return summary


def _encode_reference(tokenizer: Any, text: str) -> list[int]:
    try:
        value = (
            tokenizer(text, add_special_tokens=False) if callable(tokenizer) else None
        )
        ids = (
            value.get("input_ids")
            if isinstance(value, Mapping)
            else getattr(value, "input_ids", None)
        )
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        if ids is not None:
            return [int(item) for item in ids]
    except Exception:
        pass
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise EvaluationError("tokenizer cannot encode reference_text")
    return [int(item) for item in encode(text, add_special_tokens=False)]


def _teacher_force(
    model: nn.Module, prompt_ids: list[int], target_ids: list[int]
) -> dict[str, Any]:
    if not target_ids:
        raise EvaluationError("reference must contain at least one target token")
    joined = torch.tensor([prompt_ids + target_ids], dtype=torch.int64)
    with torch.inference_mode():
        output = model(
            input_ids=joined,
            attention_mask=torch.ones_like(joined),
            use_cache=False,
            return_dict=True,
        )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        logits = output.get("logits") if isinstance(output, Mapping) else None
    if not isinstance(logits, torch.Tensor):
        raise EvaluationError("teacher-forced model output lacks logits")
    if (
        logits.device.type != "cpu"
        or logits.dtype is not torch.float32
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != len(prompt_ids) + len(target_ids)
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise EvaluationError("teacher-forced logits are not finite CPU values")
    selected = logits[0, len(prompt_ids) - 1 : len(prompt_ids) + len(target_ids) - 1, :]
    labels = torch.tensor(target_ids, dtype=torch.int64)
    try:
        metrics = token_metrics(selected, labels)
    except ValueError as error:
        raise EvaluationError(str(error)) from error
    predictions = torch.tensor(metrics["argmax_token_ids"], dtype=torch.int64)
    metrics["target_token_count"] = len(target_ids)
    metrics["argmax_agreement"] = float((predictions == labels).float().mean().item())
    return metrics


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _environment(model: nn.Module, tokenizer: Any) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "tokenizers", "safetensors"):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "offline_environment": {
            name: os.environ.get(name)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        },
    }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _source_identity(spec_path: Path) -> dict[str, Any]:
    del spec_path
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty": dirty}


def _case_record(
    case: Mapping[str, Any], prompt_ids: list[int], rendered: str
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "reference_text": case["reference_text"],
        "max_new_tokens": case["max_new_tokens"],
        "prompt_token_ids": prompt_ids,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }


def _generation_record(result: CachedGeneration, tokenizer: Any) -> dict[str, Any]:
    return {
        "token_ids": result.all_ids,
        "generated_token_ids": result.generated_ids,
        "generated_token_count": len(result.generated_ids),
        "stop_reason": result.stop_reason,
        "stopped_by_eos": result.stopped_by_eos,
        "text": str(tokenizer.decode(result.generated_ids, skip_special_tokens=True)),
        "chosen_token_logprobs": result.chosen_logprobs,
    }


def _timed_generation(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    maximum: int,
) -> tuple[CachedGeneration, dict[str, Any]]:
    eos = _eos_ids(tokenizer)
    ids = input_ids.detach().clone()
    mask = attention_mask.detach().clone()
    prompt = [int(item) for item in ids[0].tolist()]
    generated: list[int] = []
    total_start = time.perf_counter_ns()
    prefill_start = total_start
    with torch.inference_mode():
        output = model(
            input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True
        )
        prefill_end = time.perf_counter_ns()
        logits, past = _output_parts(output)
        decode_ns: list[int] = []
        row = logits[0, -1, :]
        token = int(torch.argmax(row).item())
        first_select_end = time.perf_counter_ns()
        generated.append(token)
        for _step in range(1, maximum):
            if generated[-1] in eos:
                break
            started = time.perf_counter_ns()
            ids = torch.tensor([[generated[-1]]], dtype=torch.int64)
            mask = torch.cat((mask, torch.ones((1, 1), dtype=torch.int64)), dim=1)
            output = model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            ended = time.perf_counter_ns()
            logits, past = _output_parts(output)
            row = logits[0, -1, :]
            token = int(torch.argmax(row).item())
            ended = time.perf_counter_ns()
            decode_ns.append(ended - started)
            generated.append(token)
    total_end = time.perf_counter_ns()
    stop = "eos" if generated and generated[-1] in eos else "max_new_tokens"
    result = CachedGeneration(
        prompt + generated, generated, stop, stop == "eos", [], []
    )
    return result, {
        "prefill_ns": prefill_end - prefill_start,
        "time_to_first_token_ns": first_select_end - total_start,
        "decode_ns": decode_ns,
        "total_generation_ns": total_end - total_start,
        "generated_tokens": len(generated),
        "stop_reason": stop,
    }


def run_evaluation(
    request: EvaluationRequest,
    *,
    model_loader: Callable[[Path], nn.Module] = _presentation._load_model,
    tokenizer_loader: Callable[[Path], Any] = _presentation._load_tokenizer,
    runtime_loader: Callable[[Path, str], Any] = _presentation._load_runtime,
    installer: Callable[
        [nn.Module, Path, Any], InstallationLike
    ] = _presentation._install,
    model_verifier: Callable[
        [Path], dict[str, Any]
    ] = _presentation._verify_model_directory,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one correctness or performance evaluation and return JSON evidence."""
    if request.mode not in {"correctness", "performance"}:
        raise EvaluationError("mode must be correctness or performance")
    if type(request.session_index) is not int or request.session_index not in range(3):
        raise EvaluationError("session_index must be 0, 1, or 2")
    try:
        spec_path = _presentation._require_path(request.spec_path, "spec")
        spec_text = spec_path.read_text(encoding="utf-8")
        spec = validate_spec(json.loads(spec_text))
    except (OSError, json.JSONDecodeError, EvaluationError) as error:
        raise EvaluationError(f"unable to load evaluation spec: {error}") from error
    spec_digest = hashlib.sha256(spec_text.encode()).hexdigest()
    source = _source_identity(spec_path)
    if source["git_revision"] is None or source["git_dirty"] is not False:
        raise EvaluationError("evaluation requires a clean git checkout")
    canonical_path = (
        Path(__file__).resolve().parents[2] / "benchmarks/evaluation-v1/spec.json"
    )
    if spec_path.read_bytes() != canonical_path.read_bytes():
        raise EvaluationError("spec must match the committed evaluation-v1 protocol")
    model_dir = _presentation._require_path(
        request.model_directory, "model directory", directory=True
    )
    assets = _presentation._require_path(
        request.asset_directory, "asset directory", directory=True
    )
    library_path = _presentation._require_path(request.bridge_library, "bridge library")
    digest = _presentation._require_sha256(request.bridge_sha256)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise EvaluationError(
                "unable to configure one Torch inter-op thread"
            ) from None
    torch.manual_seed(0)
    process_start_ns = request.process_start_ns or time.perf_counter_ns()
    model_files = model_verifier(model_dir)
    model_load_start_ns = time.perf_counter_ns()
    model = model_loader(model_dir)
    tokenizer = tokenizer_loader(model_dir)
    model_load_end_ns = time.perf_counter_ns()
    if model_verifier(model_dir) != model_files:
        raise EvaluationError("model files changed during loading")
    cases = cast(list[Mapping[str, Any]], spec["cases"])
    tokenized: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    reference_tokens: dict[str, list[int]] = {}
    for case in cases:
        cid = str(case["id"])
        tokenized[cid] = _presentation._chat_inputs(tokenizer, str(case["prompt"]))
        if tokenized[cid][0].shape[1] > 512:
            raise EvaluationError(f"case {cid} prompt exceeds the 512-token limit")
        reference_tokens[cid] = _encode_reference(
            tokenizer, str(case["reference_text"])
        )
        if len(tokenized[cid][0][0]) + len(reference_tokens[cid]) > 640:
            raise EvaluationError(f"case {cid} reference sequence exceeds 640 tokens")
    evidence: dict[str, Any] = {
        "format": "decodeforge_evaluation_v1",
        "g3_evidence": False,
        "protocol_id": spec.get("protocol_id", "evaluation-v1"),
        "mode": request.mode,
        "session_index": request.session_index,
        "spec_sha256": spec_digest,
        "source": source,
        "command_line": list(sys.argv),
        "model_files": model_files,
        "bridge_library_sha256": digest,
        "model_identity": {
            "model_id": _presentation._PINNED_MODEL_ID,
            "revision": _presentation._PINNED_MODEL_REVISION,
        },
        "environment": _environment(model, tokenizer),
        "cases": [],
        "accepted": True,
        "failures": [],
    }
    evidence["timing"] = {
        "process_to_model_load_ns": model_load_end_ns - process_start_ns,
        "model_and_tokenizer_load_ns": model_load_end_ns - model_load_start_ns,
        "model_preparation_ns": time.perf_counter_ns() - process_start_ns,
        "timing_scope_note": (
            "CLI starts after interpreter/stdlib startup but before evaluation-module "
            "imports; direct API calls start at model preparation. External toolchain "
            "preflight is excluded. Setup components exclude intervening FP32 runs."
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    original_modules = tuple(
        model.get_submodule(path) for path in tinyllama_qproj_paths()
    )
    installation: InstallationLike | None = None

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    try:
        if request.mode == "correctness":
            # The baseline is generated completely before adapter installation.
            baseline: dict[str, CachedGeneration] = {}
            for case in cases:
                cid = str(case["id"])
                announce(f"case {cid}: original_fp32")
                inp, mask, rendered = tokenized[cid]
                record = _case_record(case, [int(x) for x in inp[0]], rendered)
                try:
                    result = generate_cached(
                        model,
                        tokenizer,
                        inp,
                        mask,
                        int(case["max_new_tokens"]),
                        retain_logits=False,
                    )
                    baseline[cid] = result
                    record["fp32"] = _generation_record(result, tokenizer)
                    record["quality_fp32"] = _teacher_force(
                        model, record["prompt_token_ids"], reference_tokens[cid]
                    )
                except Exception as error:
                    record["status"] = "failed"
                    record["error"] = f"FP32: {error}"
                    evidence["failures"].append({"id": cid, "error": str(error)})
                evidence["cases"].append(record)
            install_start_ns = time.perf_counter_ns()
            runtime = runtime_loader(library_path, digest)
            installation = installer(model, assets, runtime)
            install_end_ns = time.perf_counter_ns()
            evidence["timing"]["runtime_and_install_ns"] = (
                install_end_ns - install_start_ns
            )
            evidence["timing"]["setup_components_ns"] = (
                evidence["timing"]["model_preparation_ns"]
                + evidence["timing"]["runtime_and_install_ns"]
            )
            initial_installation = _check_installation_initial(installation)
            evidence["installation"] = initial_installation
            evidence["asset_inventory_identity"] = str(
                installation.inventory.aggregate_identity
            )
            # Keep one case's logits transiently; no all-case vocabulary tensor
            # collection is retained in the evidence bundle.
            for record, case in zip(
                cast(list[dict[str, Any]], evidence["cases"]), cases, strict=True
            ):
                if "fp32" not in record:
                    continue
                cid = str(case["id"])
                inp, mask, _ = tokenized[cid]
                path_results: dict[str, CachedGeneration] = {}
                case_failed = False
                for name, mode in (
                    ("same_q8_reference", QProjExecutionMode.SAME_Q8_REFERENCE),
                    ("hybrid_native", QProjExecutionMode.HYBRID_NATIVE),
                ):
                    announce(f"case {cid}: {name}")
                    installation.set_execution_mode(mode)
                    before = installation.counters
                    try:
                        result = generate_cached(
                            model,
                            tokenizer,
                            inp,
                            mask,
                            int(case["max_new_tokens"]),
                            retain_logits=True,
                        )
                        after = installation.counters
                        delta = _counter_delta(before, after)
                        path_results[name] = result
                        record[name] = _generation_record(result, tokenizer)
                        record.setdefault("counter_deltas", {})[name] = delta
                        _reconcile(name, len(result.generated_ids), delta)
                        if name == "same_q8_reference":
                            quality_before = installation.counters
                            quality = _teacher_force(
                                model,
                                record["prompt_token_ids"],
                                reference_tokens[cid],
                            )
                            quality_delta = _counter_delta(
                                quality_before, installation.counters
                            )
                            _reconcile(name, 1, quality_delta)
                            fp32_argmax = record["quality_fp32"]["argmax_token_ids"]
                            q8_argmax = quality["argmax_token_ids"]
                            quality["argmax_agreement_with_original_fp32"] = (
                                sum(
                                    left == right
                                    for left, right in zip(
                                        fp32_argmax, q8_argmax, strict=True
                                    )
                                )
                                / len(fp32_argmax)
                                if fp32_argmax
                                else None
                            )
                            record["quality_same_q8"] = quality
                            record["counter_deltas"]["quality_same_q8"] = quality_delta
                    except Exception as error:
                        case_failed = True
                        record["status"] = "failed"
                        record.setdefault("errors", []).append(f"{name}: {error}")
                        evidence["failures"].append(
                            {"id": cid, "path": name, "error": str(error)}
                        )
                if len(path_results) == 2:
                    comparison = compare_runs(
                        path_results["same_q8_reference"], path_results["hybrid_native"]
                    )
                    record["comparison"] = comparison
                    comparison["fp32_token_ids_exact"] = (
                        baseline[cid].generated_ids
                        == path_results["hybrid_native"].generated_ids
                    )
                    if not comparison["passed"] or case_failed:
                        record["status"] = "failed"
                        evidence["failures"].append(
                            {"id": cid, "error": "native/reference mismatch"}
                        )
                    else:
                        record["status"] = "passed"
                elif "status" not in record:
                    record["status"] = "failed"
                # Explicitly release potentially large vocabulary tensors.
                path_results.clear()
        else:
            selected = [str(value) for value in spec["performance_case_ids"]]
            order = (
                ["same_q8_reference", "hybrid_native"]
                if request.session_index % 2 == 0
                else ["hybrid_native", "same_q8_reference"]
            )
            perf: list[dict[str, Any]] = []
            for cid in selected:
                case = next(item for item in cases if item["id"] == cid)
                inp, mask, rendered = tokenized[cid]
                runs: dict[str, Any] = {
                    "case": _case_record(case, [int(x) for x in inp[0]], rendered),
                    "paths": {},
                }
                # Original FP32 is necessarily run before installation.
                runs["paths"]["fp32"] = _performance_path(
                    model, tokenizer, inp, mask, int(case["max_new_tokens"]), None
                )
                perf.append(runs)
            install_start_ns = time.perf_counter_ns()
            runtime = runtime_loader(library_path, digest)
            installation = installer(model, assets, runtime)
            install_end_ns = time.perf_counter_ns()
            evidence["timing"]["runtime_and_install_ns"] = (
                install_end_ns - install_start_ns
            )
            evidence["timing"]["setup_components_ns"] = (
                evidence["timing"]["model_preparation_ns"]
                + evidence["timing"]["runtime_and_install_ns"]
            )
            evidence["installation"] = _check_installation_initial(installation)
            evidence["asset_inventory_identity"] = str(
                installation.inventory.aggregate_identity
            )
            for name in order:
                announce(f"performance path {name}")
                installation.set_execution_mode(
                    QProjExecutionMode.SAME_Q8_REFERENCE
                    if name == "same_q8_reference"
                    else QProjExecutionMode.HYBRID_NATIVE
                )
                for runs, cid in zip(perf, selected, strict=True):
                    case = next(item for item in cases if item["id"] == cid)
                    inp, mask, _ = tokenized[cid]
                    samples: list[Any] = []
                    for phase, repetitions in (("warmup", 1), ("measured", 3)):
                        for _repetition in range(repetitions):
                            timed_result: CachedGeneration | None = None
                            try:
                                before = installation.counters
                                timed_result, timing = _timed_generation(
                                    model,
                                    tokenizer,
                                    inp,
                                    mask,
                                    int(case["max_new_tokens"]),
                                )
                                after = installation.counters
                                delta = _counter_delta(before, after)
                                assert timed_result is not None
                                _reconcile(name, len(timed_result.generated_ids), delta)
                                if phase == "measured":
                                    samples.append(
                                        {
                                            **timing,
                                            "generated_token_ids": (
                                                timed_result.generated_ids
                                            ),
                                            "counter_delta": delta,
                                        }
                                    )
                            except Exception as error:
                                evidence["failures"].append(
                                    {
                                        "id": cid,
                                        "path": name,
                                        "phase": phase,
                                        "error": str(error),
                                    }
                                )
                                if phase == "measured":
                                    sample: dict[str, Any] = {"error": str(error)}
                                    if timed_result is not None:
                                        sample["generated_token_ids"] = (
                                            timed_result.generated_ids
                                        )
                                    samples.append(sample)
                    runs["paths"][name] = {
                        "warmup_generations": 1,
                        "measured_generations": samples,
                    }
            evidence["performance"] = perf
            evidence["actual_path_order"] = ["fp32", *order]
    finally:
        if installation is not None:
            installation.close()
            first_closed = bool(installation.closed)
            # The production owner promises idempotent cleanup.
            installation.close()
            restored = all(
                model.get_submodule(path) is original
                for path, original in zip(
                    tinyllama_qproj_paths(), original_modules, strict=True
                )
            )
            if not restored:
                raise EvaluationError(
                    "adapter cleanup did not restore original modules"
                )
            final = _presentation._counter_summary(installation.counters)
            if (
                not first_closed
                or not installation.closed
                or final["installed_modules"] != 0
                or final["restored_modules"] != 22
                or final["live_adapters"] != 0
                or final["in_flight"] != 0
                or not final["closed"]
                or len(final["layers"]) != 22
                or any(
                    not layer["closed"] or layer["in_flight"]
                    for layer in final["layers"]
                )
            ):
                raise EvaluationError("adapter cleanup counters did not reconcile")
            evidence["restoration"] = {
                "closed": bool(installation.closed),
                "second_close_idempotent": True,
                "original_modules_restored": restored,
                "counters": final,
            }
    evidence["accepted"] = not evidence["failures"]
    source_after = _source_identity(spec_path)
    evidence["source_after"] = source_after
    if source_after != evidence["source"]:
        evidence["accepted"] = False
        evidence["failures"].append(
            {"error": "source checkout changed during evaluation"}
        )
    evidence["timing"]["peak_rss_bytes"] = _peak_rss_bytes()
    return evidence


def _performance_path(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    maximum: int,
    _installation: Any,
) -> dict[str, Any]:
    _warmup, _ = _timed_generation(model, tokenizer, input_ids, attention_mask, maximum)
    samples: list[dict[str, Any]] = []
    for _ in range(3):
        result, timing = _timed_generation(
            model, tokenizer, input_ids, attention_mask, maximum
        )
        samples.append({**timing, "generated_token_ids": result.generated_ids})
    return {"warmup_generations": 1, "measured_generations": samples}


def write_evaluation_json(path: Path, evidence: Mapping[str, Any]) -> None:
    """Publish one result without replacing an existing file."""
    _presentation.write_demo_json(path, dict(evidence))


__all__ = [
    "CachedGeneration",
    "EvaluationError",
    "EvaluationRequest",
    "compare_runs",
    "generate_cached",
    "run_evaluation",
    "validate_spec",
    "write_evaluation_json",
]

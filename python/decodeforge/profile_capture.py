"""One-process diagnostic capture for whole-model decode attribution."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, TypeVar

import torch
from torch import nn

from . import evaluation
from . import presentation_demo as presentation
from .decode_profile import (
    MAX_PROFILE_EVENTS,
    DecodeProfile,
    generate_cached_unprofiled,
    profile_cached_generation,
    tinyllama_component_paths,
)
from .evaluation import CachedGeneration
from .g3_session import _open_directory, publish_new_json
from .qproj_adapter import QProjExecutionMode
from .qproj_model import tinyllama_qproj_paths
from .torch_bridge import BRIDGE_ABI_VERSION

MAX_PROMPT_CHARS: Final = 4096
ControlRunner = Callable[
    [nn.Module, Any, torch.Tensor, torch.Tensor, int], CachedGeneration
]
ProfileRunner = Callable[..., DecodeProfile]
T = TypeVar("T")


class ProfileCaptureError(RuntimeError):
    """A diagnostic capture could not complete without ambiguity."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "preflight",
        observations: dict[str, Any] | None = None,
        cleanup_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.observations = {} if observations is None else observations
        self.cleanup_error = cleanup_error


@dataclass(frozen=True)
class ProfileCaptureRequest:
    """Bounded local inputs for one process-isolated diagnostic session."""

    model_directory: Path
    asset_directory: Path
    bridge_library: Path
    bridge_sha256: str
    prompt: str
    session_index: int
    max_new_tokens: int = 16


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    files = (
        root / "python" / "decodeforge" / "evaluation.py",
        root / "python" / "decodeforge" / "g3_session.py",
        root / "python" / "decodeforge" / "presentation_demo.py",
        root / "python" / "decodeforge" / "decode_profile.py",
        root / "python" / "decodeforge" / "profile_capture.py",
        root / "python" / "decodeforge" / "qproj_adapter.py",
        root / "python" / "decodeforge" / "qproj_model.py",
        root / "python" / "decodeforge" / "qproj_profile.py",
        root / "python" / "decodeforge" / "torch_bridge.py",
        root / "scripts" / "run_profile_capture.py",
    )
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
        identities = {str(path.relative_to(root)): _file_sha256(path) for path in files}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ProfileCaptureError("unable to identify capture source") from error
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "files": identities,
    }


def _configure_torch() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch.manual_seed(0)
    except RuntimeError as error:
        raise ProfileCaptureError(
            "unable to configure deterministic single-thread CPU Torch"
        ) from error


def _at_stage(stage: str, call: Callable[[], T]) -> T:
    try:
        return call()
    except ProfileCaptureError as error:
        error.stage = stage
        raise
    except Exception as error:
        raise ProfileCaptureError(f"{stage} failed: {error}", stage=stage) from error


def _environment(model: nn.Module, tokenizer: Any) -> dict[str, Any]:
    result = evaluation._environment(model, tokenizer)
    return {
        **result,
        "pid": os.getpid(),
        "platform_system": platform.system(),
    }


def _generation_wire(result: CachedGeneration, tokenizer: Any) -> dict[str, Any]:
    return {
        "token_ids": list(result.all_ids),
        "generated_token_ids": list(result.generated_ids),
        "generated_token_count": len(result.generated_ids),
        "stop_reason": result.stop_reason,
        "stopped_by_eos": result.stopped_by_eos,
        "text": str(tokenizer.decode(result.generated_ids, skip_special_tokens=True)),
    }


def _snapshot_generation(
    value: Any, input_ids: torch.Tensor, max_new_tokens: int
) -> CachedGeneration:
    if not isinstance(value, CachedGeneration):
        raise ProfileCaptureError("generation runner returned an invalid result")
    all_ids = list(value.all_ids)
    generated_ids = list(value.generated_ids)
    prompt_ids = [int(item) for item in input_ids[0].tolist()]
    if (
        not generated_ids
        or len(generated_ids) > max_new_tokens
        or all_ids != prompt_ids + generated_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in all_ids
        )
        or value.stop_reason not in {"eos", "max_new_tokens"}
        or value.stopped_by_eos != (value.stop_reason == "eos")
        or value.logits
        or value.chosen_logprobs
    ):
        raise ProfileCaptureError("generation runner returned inconsistent output")
    return CachedGeneration(
        all_ids,
        generated_ids,
        value.stop_reason,
        value.stopped_by_eos,
        [],
        [],
    )


def _timed_call(clock: Callable[[], int], call: Callable[[], Any]) -> tuple[Any, int]:
    started = clock()
    result = call()
    ended = clock()
    if (
        isinstance(started, bool)
        or not isinstance(started, int)
        or isinstance(ended, bool)
        or not isinstance(ended, int)
        or started < 0
        or ended < started
    ):
        raise ProfileCaptureError("capture clock must be monotonic nanoseconds")
    return result, ended - started


def _run_once(
    *,
    name: str,
    kind: str,
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    installation: presentation.InstallationLike,
    clock: Callable[[], int],
    control_runner: ControlRunner,
    profile_runner: ProfileRunner,
) -> tuple[CachedGeneration, dict[str, Any]]:
    before = presentation._counter_summary(installation.counters)
    trace: dict[str, Any] | None = None

    if kind == "profile":
        captured, elapsed = _timed_call(
            clock,
            lambda: profile_runner(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens,
                module_paths=tinyllama_component_paths(),
                qproj_details=True,
            ),
        )
        if not isinstance(captured, DecodeProfile):
            raise ProfileCaptureError("profile runner returned an invalid result")
        generation = _snapshot_generation(
            captured.generation, input_ids, max_new_tokens
        )
        trace = captured.to_wire()
    elif kind in {"warmup", "control"}:
        raw_generation, elapsed = _timed_call(
            clock,
            lambda: control_runner(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens,
            ),
        )
        generation = _snapshot_generation(raw_generation, input_ids, max_new_tokens)
    else:
        raise ProfileCaptureError("capture run kind is invalid")

    after = presentation._counter_summary(installation.counters)
    delta = presentation._counter_delta(before, after)
    try:
        evaluation._reconcile(name, len(generation.generated_ids), delta)
    except evaluation.EvaluationError as error:
        raise ProfileCaptureError(str(error)) from error
    return generation, {
        "kind": kind,
        "measured": kind != "warmup",
        "outer_elapsed_ns": elapsed,
        "generation": _generation_wire(generation, tokenizer),
        "counters_before": before,
        "counters_after": after,
        "counter_delta": delta,
        "trace": trace,
    }


def _check_profile_trace(
    name: str, generation: CachedGeneration, trace: dict[str, Any]
) -> None:
    paths = list(tinyllama_component_paths())
    events = trace.get("events")
    trace_clock = trace.get("clock")
    generated_ids = trace.get("generated_token_ids")
    generated_count = trace.get("generated_token_count")
    if (
        trace.get("format") != "decodeforge_decode_profile_v1"
        or type(trace.get("schema_version")) is not int
        or trace.get("schema_version") != 1
        or trace.get("claim_class") != "diagnostic_profile"
        or trace.get("performance_claim_allowed") is not False
        or trace.get("boundary_contract") != "decodeforge_adapter_internal_v1"
        or trace.get("qproj_details") is not True
        or not isinstance(trace_clock, dict)
        or trace_clock.get("overhead_subtracted") is not False
        or not isinstance(generated_ids, list)
        or any(type(value) is not int or value < 0 for value in generated_ids)
        or generated_ids != generation.generated_ids
        or type(generated_count) is not int
        or generated_count != len(generation.generated_ids)
        or trace.get("stop_reason") != generation.stop_reason
        or trace.get("module_paths") != paths
        or not isinstance(events, list)
        or not 1 <= len(events) <= MAX_PROFILE_EVENTS
    ):
        raise ProfileCaptureError(f"{name} trace metadata is inconsistent")
    by_id: dict[int, dict[str, Any]] = {}
    children: dict[int | None, list[dict[str, Any]]] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ProfileCaptureError(f"{name} trace event is invalid")
        event_id = event.get("event_id")
        parent_id = event.get("parent_id")
        step_index = event.get("step_index")
        start_ns = event.get("start_ns")
        end_ns = event.get("end_ns")
        inclusive_ns = event.get("inclusive_ns")
        exclusive_ns = event.get("exclusive_ns")
        if (
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id < 0
            or event_id in by_id
            or (
                parent_id is not None
                and (isinstance(parent_id, bool) or not isinstance(parent_id, int))
            )
            or isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not -1 <= step_index < 64
            or isinstance(start_ns, bool)
            or not isinstance(start_ns, int)
            or start_ns < 0
            or isinstance(end_ns, bool)
            or not isinstance(end_ns, int)
            or end_ns < start_ns
            or isinstance(inclusive_ns, bool)
            or not isinstance(inclusive_ns, int)
            or inclusive_ns != end_ns - start_ns
            or isinstance(exclusive_ns, bool)
            or not isinstance(exclusive_ns, int)
            or not 0 <= exclusive_ns <= inclusive_ns
        ):
            raise ProfileCaptureError(f"{name} trace event identity is invalid")
        by_id[event_id] = event
        children.setdefault(parent_id, []).append(event)
    for values in children.values():
        values.sort(key=lambda event: int(event["event_id"]))
    for event in events:
        parent_id = event["parent_id"]
        if parent_id is not None:
            parent = by_id.get(parent_id)
            if (
                parent is None
                or parent_id >= event["event_id"]
                or event["start_ns"] < parent["start_ns"]
                or event["end_ns"] > parent["end_ns"]
            ):
                raise ProfileCaptureError(f"{name} trace nesting is invalid")
        direct = sorted(
            children.get(event["event_id"], []),
            key=lambda child: int(child["start_ns"]),
        )
        if any(left["end_ns"] > right["start_ns"] for left, right in pairwise(direct)):
            raise ProfileCaptureError(f"{name} trace siblings overlap")
        expected_exclusive = event["inclusive_ns"] - sum(
            child["inclusive_ns"] for child in direct
        )
        if event["exclusive_ns"] != expected_exclusive:
            raise ProfileCaptureError(f"{name} trace exclusive timing is invalid")

    roots = children.get(None, [])
    if (
        len(roots) != 1
        or roots[0].get("boundary") != "generation"
        or roots[0].get("phase") != "generation"
        or roots[0].get("step_index") != -1
        or roots[0].get("module_path") is not None
    ):
        raise ProfileCaptureError(f"{name} trace generation root is invalid")
    root_id = int(roots[0]["event_id"])
    visited = {root_id}
    qproj_paths = set(tinyllama_qproj_paths())
    steps = children.get(root_id, [])
    if len(steps) != len(generation.generated_ids):
        raise ProfileCaptureError(f"{name} trace step coverage is invalid")

    def exact_children(
        parent: dict[str, Any], expected: list[str]
    ) -> list[dict[str, Any]]:
        values = children.get(int(parent["event_id"]), [])
        if [value.get("boundary") for value in values] != expected:
            raise ProfileCaptureError(
                f"{name} trace {parent.get('boundary')} children are invalid"
            )
        visited.update(int(value["event_id"]) for value in values)
        return values

    def check_context(
        event: dict[str, Any], phase: str, step_index: int, module_path: str | None
    ) -> None:
        if (
            event.get("phase") != phase
            or event.get("step_index") != step_index
            or event.get("module_path") != module_path
        ):
            raise ProfileCaptureError(f"{name} trace event context is invalid")

    def check_unannotated(event: dict[str, Any]) -> None:
        if event.get("dispatch") is not None or event.get("guard_reason") is not None:
            raise ProfileCaptureError(f"{name} trace annotation is invalid")

    check_unannotated(roots[0])

    for step_index in range(len(generation.generated_ids)):
        boundary = "prefill" if step_index == 0 else "cached_step"
        phase = "prefill" if step_index == 0 else "cached_decode"
        step = steps[step_index]
        if step.get("boundary") != boundary:
            raise ProfileCaptureError(f"{name} trace step coverage is invalid")
        check_context(step, phase, step_index, None)
        check_unannotated(step)
        visited.add(int(step["event_id"]))
        step_boundaries = [
            "input_preparation",
            "model_forward",
            "output_validation",
            "token_selection",
            "bookkeeping",
        ]
        major = exact_children(step, step_boundaries)
        for event in major:
            check_context(event, phase, step_index, None)
            check_unannotated(event)
        major_by_name = {str(event["boundary"]): event for event in major}
        for child_boundary in (
            "input_preparation",
            "output_validation",
            "token_selection",
            "bookkeeping",
        ):
            exact_children(major_by_name[child_boundary], [])

        modules = exact_children(
            major_by_name["model_forward"], ["module_forward"] * len(paths)
        )
        if [event.get("module_path") for event in modules] != paths:
            raise ProfileCaptureError(f"{name} trace component coverage is invalid")
        for module in modules:
            module_path = str(module["module_path"])
            check_context(module, phase, step_index, module_path)
            if module_path not in qproj_paths:
                check_unannotated(module)
                exact_children(module, [])
                continue

            expected_dispatch = (
                "fallback"
                if name == "same_q8_reference" or step_index == 0
                else "native"
            )
            if module.get("dispatch") != expected_dispatch:
                raise ProfileCaptureError(
                    f"{name} trace dispatch mismatch at {module_path}"
                )
            expected_reason = (
                "forced_same_q8_reference"
                if name == "same_q8_reference"
                else "m_gt_one"
                if step_index == 0
                else None
            )
            if module.get("guard_reason") != expected_reason:
                raise ProfileCaptureError(
                    f"{name} trace guard reason mismatch at {module_path}"
                )
            if expected_dispatch == "fallback":
                internal = exact_children(module, ["adapter_storage_guard", "fallback"])
                fallback = internal[1]
                check_context(internal[0], phase, step_index, module_path)
                check_context(fallback, phase, step_index, module_path)
                fallback_children = exact_children(
                    fallback,
                    [
                        "adapter_storage_guard",
                        "fallback_clone",
                        "fallback_hash",
                        "fallback_linear",
                    ],
                )
                for event in fallback_children:
                    check_context(event, phase, step_index, module_path)
                    check_unannotated(event)
                    exact_children(event, [])
                for event in internal:
                    check_unannotated(event)
            else:
                internal = exact_children(
                    module, ["adapter_storage_guard", "guarded_native_operator"]
                )
                for event in internal:
                    check_context(event, phase, step_index, module_path)
                    check_unannotated(event)
                exact_children(internal[0], [])
                binding = exact_children(internal[1], ["guarded_binding_run"])
                check_context(binding[0], phase, step_index, module_path)
                check_unannotated(binding[0])
                exact_children(binding[0], [])

    if visited != set(by_id):
        raise ProfileCaptureError(f"{name} trace contains unexpected events")


def _check_restoration(
    model: nn.Module,
    installation: presentation.InstallationLike,
    original_modules: tuple[nn.Module, ...],
) -> dict[str, Any]:
    final = presentation._counter_summary(installation.counters)
    restored = all(
        model.get_submodule(path) is original
        for path, original in zip(
            tinyllama_qproj_paths(), original_modules, strict=True
        )
    )
    if (
        not installation.closed
        or not final["closed"]
        or not restored
        or final["installed_modules"] != 0
        or final["restored_modules"] != 22
        or final["live_adapters"] != 0
        or final["in_flight"] != 0
        or len(final["layers"]) != 22
        or any(not layer["closed"] or layer["in_flight"] for layer in final["layers"])
    ):
        raise ProfileCaptureError("capture cleanup did not fully restore the model")
    return {
        "closed": True,
        "original_modules_restored": True,
        "counters": final,
    }


def run_profile_capture(
    request: ProfileCaptureRequest,
    *,
    model_loader: presentation.ModelLoader = presentation._load_model,
    tokenizer_loader: presentation.TokenizerLoader = presentation._load_tokenizer,
    runtime_loader: presentation.RuntimeLoader = presentation._load_runtime,
    installer: presentation.Installer = presentation._install,
    model_verifier: Callable[
        [Path], dict[str, dict[str, Any]]
    ] = presentation._verify_model_directory,
    control_runner: ControlRunner = generate_cached_unprofiled,
    profile_runner: ProfileRunner = profile_cached_generation,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Capture warmup, matched control, and profiled runs in one process."""

    if (
        isinstance(request.session_index, bool)
        or not isinstance(request.session_index, int)
        or not 0 <= request.session_index <= 2
    ):
        raise ProfileCaptureError("session_index must be in 0..2")
    if (
        isinstance(request.max_new_tokens, bool)
        or not isinstance(request.max_new_tokens, int)
        or not 2 <= request.max_new_tokens <= 64
    ):
        raise ProfileCaptureError("max_new_tokens must be in 2..64")
    if (
        not isinstance(request.prompt, str)
        or not request.prompt.strip()
        or len(request.prompt) > MAX_PROMPT_CHARS
    ):
        raise ProfileCaptureError("prompt must be 1..4096 nonblank characters")
    try:
        model_dir = presentation._require_path(
            request.model_directory, "model directory", directory=True
        )
        assets = presentation._require_path(
            request.asset_directory, "asset directory", directory=True
        )
        library_path = presentation._require_path(
            request.bridge_library, "bridge library"
        )
        digest = presentation._require_sha256(request.bridge_sha256)
    except presentation.PresentationDemoError as error:
        raise ProfileCaptureError(str(error)) from error

    _at_stage("runtime_configuration", _configure_torch)
    source_before = _at_stage("source_identity", _source_identity)
    model_files = _at_stage("model_verification", lambda: model_verifier(model_dir))
    model = _at_stage("model_loading", lambda: model_loader(model_dir))
    tokenizer = _at_stage("tokenizer_loading", lambda: tokenizer_loader(model_dir))
    if (
        _at_stage("model_reverification", lambda: model_verifier(model_dir))
        != model_files
    ):
        raise ProfileCaptureError(
            "model files changed during loading", stage="model_reverification"
        )
    input_ids, attention_mask, rendered_prompt = _at_stage(
        "prompt_tokenization",
        lambda: presentation._chat_inputs(tokenizer, request.prompt),
    )
    if not 2 <= input_ids.shape[1] <= 512:
        raise ProfileCaptureError(
            "chat prompt must contain 2..512 tokens", stage="prompt_tokenization"
        )

    original_modules = _at_stage(
        "model_topology",
        lambda: tuple(model.get_submodule(path) for path in tinyllama_qproj_paths()),
    )
    runtime = _at_stage("runtime_loading", lambda: runtime_loader(library_path, digest))
    installation = _at_stage("installation", lambda: installer(model, assets, runtime))
    observations: dict[str, Any] = {}
    stage = "installation"
    try:
        try:
            initial = evaluation._check_installation_initial(installation)
            mode_order = ["same_q8_reference", "hybrid_native"]
            if request.session_index % 2:
                mode_order.reverse()
            measurement_order = ["control", "profile"]
            if request.session_index % 2:
                measurement_order.reverse()
            runs: dict[str, Any] = {}
            generations: dict[str, CachedGeneration] = {}
            for name in mode_order:
                observations[name] = {}
                mode = QProjExecutionMode(name)
                stage = f"{name}.mode_switch"
                installation.set_execution_mode(mode)
                stage = f"{name}.warmup"
                warmup, warmup_record = _run_once(
                    name=name,
                    kind="warmup",
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=request.max_new_tokens,
                    installation=installation,
                    clock=clock,
                    control_runner=control_runner,
                    profile_runner=profile_runner,
                )
                observations[name]["warmup"] = warmup_record
                records: dict[str, Any] = {"warmup": warmup_record}
                measured: dict[str, CachedGeneration] = {}
                for kind in measurement_order:
                    stage = f"{name}.{kind}"
                    generated, record = _run_once(
                        name=name,
                        kind=kind,
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_new_tokens,
                        installation=installation,
                        clock=clock,
                        control_runner=control_runner,
                        profile_runner=profile_runner,
                    )
                    measured[kind] = generated
                    records[kind] = record
                    observations[name][kind] = record
                if (
                    warmup != measured["control"]
                    or measured["control"] != measured["profile"]
                ):
                    raise ProfileCaptureError(
                        f"{name} warmup/control/profile outputs differ"
                    )
                trace = records["profile"]["trace"]
                if not isinstance(trace, dict):
                    raise ProfileCaptureError(f"{name} profile trace is missing")
                _check_profile_trace(name, measured["profile"], trace)
                control_ns = int(records["control"]["outer_elapsed_ns"])
                profile_ns = int(records["profile"]["outer_elapsed_ns"])
                runs[name] = {
                    "measurement_order": measurement_order,
                    "warmup": records["warmup"],
                    "control": records["control"],
                    "profile": records["profile"],
                    "observer_cost_ratio": (
                        profile_ns / control_ns if control_ns > 0 else None
                    ),
                }
                generations[name] = measured["profile"]
            if generations["same_q8_reference"] != generations["hybrid_native"]:
                raise ProfileCaptureError("same-Q8 and hybrid outputs differ")
            stage = "source_recheck"
            source_after = _source_identity()
            if source_after != source_before:
                raise ProfileCaptureError("capture source changed during the session")
        except BaseException as error:
            capture_error = (
                error
                if isinstance(error, ProfileCaptureError)
                else ProfileCaptureError("diagnostic capture failed")
            )
            capture_error.stage = stage
            capture_error.observations = observations
            try:
                installation.close()
            except BaseException as cleanup_error:
                raise ProfileCaptureError(
                    "capture and cleanup both failed",
                    stage="cleanup",
                    observations=observations,
                    cleanup_error=cleanup_error,
                ) from error
            raise capture_error from error
        else:
            try:
                installation.close()
            except BaseException as cleanup_error:
                raise ProfileCaptureError(
                    "diagnostic capture cleanup failed",
                    stage="cleanup",
                    observations=observations,
                    cleanup_error=cleanup_error,
                ) from cleanup_error
    except ProfileCaptureError:
        raise
    except BaseException as error:
        raise ProfileCaptureError(
            "diagnostic capture cleanup failed",
            stage="cleanup",
            observations=observations,
            cleanup_error=error,
        ) from error

    try:
        restoration = _check_restoration(model, installation, original_modules)
    except BaseException as error:
        raise ProfileCaptureError(
            "diagnostic capture restoration check failed",
            stage="restoration",
            observations=observations,
            cleanup_error=error,
        ) from error
    return {
        "format": "decodeforge_profile_capture_v1",
        "schema_version": 1,
        "claim_class": "diagnostic_profile",
        "benchmark": False,
        "g3_evidence": False,
        "performance_claim_allowed": False,
        "capture_status": "complete",
        "session_index": request.session_index,
        "process_id": os.getpid(),
        "command_line": list(sys.argv),
        "source": source_before,
        "environment": _environment(model, tokenizer),
        "model_identity": {
            "model_id": presentation._PINNED_MODEL_ID,
            "revision": presentation._PINNED_MODEL_REVISION,
            "files": model_files,
        },
        "asset_inventory_identity": str(installation.inventory.aggregate_identity),
        "bridge_library_sha256": digest,
        "bridge_abi_version": BRIDGE_ABI_VERSION,
        "workload": {
            "prompt": request.prompt,
            "rendered_prompt_sha256": hashlib.sha256(
                rendered_prompt.encode("utf-8")
            ).hexdigest(),
            "input_token_ids": [int(value) for value in input_ids[0].tolist()],
            "max_new_tokens": request.max_new_tokens,
            "greedy": True,
            "use_cache": True,
        },
        "mode_order": mode_order,
        "runs": runs,
        "comparison": {
            "control_profile_exact": True,
            "same_q8_hybrid_exact": True,
            "cached_decode_covered": True,
        },
        "installation": initial,
        "restoration": restoration,
        "interpretation": (
            "Diagnostic timings include instrumentation overhead and do not establish "
            "a model-level performance improvement."
        ),
    }


def rejected_profile_capture(
    request: ProfileCaptureRequest, error: ProfileCaptureError
) -> dict[str, Any]:
    """Return a small diagnostic record for a failed capture attempt."""

    return {
        "format": "decodeforge_profile_capture_v1",
        "schema_version": 1,
        "claim_class": "diagnostic_profile",
        "benchmark": False,
        "g3_evidence": False,
        "performance_claim_allowed": False,
        "capture_status": "rejected",
        "session_index": request.session_index,
        "process_id": os.getpid(),
        "command_line": list(sys.argv),
        "request": {
            "model_directory": str(request.model_directory),
            "asset_directory": str(request.asset_directory),
            "bridge_library": str(request.bridge_library),
            "bridge_sha256": request.bridge_sha256,
            "prompt": request.prompt,
            "max_new_tokens": request.max_new_tokens,
        },
        "failure": {
            "stage": error.stage,
            "reason": str(error),
            "cleanup_error": (
                None
                if error.cleanup_error is None
                else type(error.cleanup_error).__name__
            ),
        },
        "completed_observations": error.observations,
        "interpretation": (
            "This rejected diagnostic attempt is not benchmark or correctness evidence."
        ),
    }


def write_profile_capture(path: Path, document: dict[str, Any]) -> None:
    """Durably create one capture document without following or replacing paths."""

    try:
        publish_new_json(path, document)
    except Exception as error:
        raise ProfileCaptureError("unable to publish new capture output") from error


def check_profile_output(path: Path) -> None:
    """Fail early unless an absolute output can be created without replacement."""

    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ProfileCaptureError("capture output must be an absolute normalized path")
    descriptor = -1
    try:
        descriptor = _open_directory(path.parent)
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ProfileCaptureError("capture output already exists")
    except ProfileCaptureError:
        raise
    except Exception as error:
        raise ProfileCaptureError("capture output path is unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


__all__ = [
    "ProfileCaptureError",
    "ProfileCaptureRequest",
    "check_profile_output",
    "rejected_profile_capture",
    "run_profile_capture",
    "write_profile_capture",
]

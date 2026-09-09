"""Opt-in diagnostic profiling for cached CPU text generation.

This module is deliberately separate from the frozen G1, G3, and evaluation-v1
measurement paths. Its timings help locate work; they are not benchmark evidence
and must not be presented as a speedup claim.
"""

from __future__ import annotations

import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, Final

import torch
from torch import nn

from .evaluation import MAX_NEW_TOKENS, CachedGeneration, _eos_ids, _output_parts

Clock = Callable[[], int]

_COUNTER_FIELDS: Final = (
    "forward",
    "native_attempt",
    "native_success",
    "native_error",
    "fallback_attempt",
    "fallback_success",
    "fallback_error",
    "predispatch_error",
)
MAX_PROFILE_MODULES: Final = 512
MAX_PROFILE_PATH_CHARS: Final = 256
MAX_PROFILE_EVENTS: Final = 20_000
MAX_PROFILE_DEPTH: Final = 512
_ACTIVE_MODELS: weakref.WeakKeyDictionary[nn.Module, object] = (
    weakref.WeakKeyDictionary()
)
_ACTIVE_MODELS_LOCK = threading.Lock()


class DecodeProfileError(RuntimeError):
    """A diagnostic profile could not be captured without ambiguity."""


@dataclass(frozen=True)
class ProfileEvent:
    """One completed inclusive span in a nested diagnostic trace."""

    event_id: int
    parent_id: int | None
    boundary: str
    phase: str
    step_index: int
    module_path: str | None
    start_ns: int
    end_ns: int
    dispatch: str | None = None
    guard_reason: str | None = None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class _ActiveSpan:
    event_id: int
    parent_id: int | None
    boundary: str
    phase: str
    step_index: int
    module_path: str | None
    start_ns: int


class _ProfileCollector:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._owner_thread: int | None = None
        self._next_id = 0
        self._stack: list[int] = []
        self._active: dict[int, _ActiveSpan] = {}
        self._events: dict[int, ProfileEvent] = {}
        self._phase = "setup"
        self._step_index = -1
        self._last_clock_ns: int | None = None

    def _now(self) -> int:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DecodeProfileError("profile clock must return nonnegative integers")
        if self._last_clock_ns is not None and value < self._last_clock_ns:
            raise DecodeProfileError("profile clock moved backwards")
        self._last_clock_ns = value
        return value

    @contextmanager
    def step(self, phase: str, step_index: int) -> Iterator[None]:
        previous = self._phase, self._step_index
        self._phase, self._step_index = phase, step_index
        try:
            yield
        finally:
            self._phase, self._step_index = previous

    def begin(self, boundary: str, *, module_path: str | None = None) -> int:
        thread_id = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = thread_id
        elif self._owner_thread != thread_id:
            raise DecodeProfileError(
                "one profile cannot combine forwards from multiple Python threads"
            )
        if not boundary or not isinstance(boundary, str):
            raise DecodeProfileError("profile boundary must be nonempty text")
        if self._next_id >= MAX_PROFILE_EVENTS:
            raise DecodeProfileError("profile event limit exceeded")
        if len(self._stack) >= MAX_PROFILE_DEPTH:
            raise DecodeProfileError("profile nesting limit exceeded")
        event_id = self._next_id
        self._next_id += 1
        active = _ActiveSpan(
            event_id=event_id,
            parent_id=self._stack[-1] if self._stack else None,
            boundary=boundary,
            phase=self._phase,
            step_index=self._step_index,
            module_path=module_path,
            start_ns=self._now(),
        )
        self._active[event_id] = active
        self._stack.append(event_id)
        return event_id

    def finish(self, event_id: int) -> None:
        if self._owner_thread != threading.get_ident():
            raise DecodeProfileError(
                "profile spans must finish on the thread that started them"
            )
        if not self._stack or self._stack[-1] != event_id:
            raise DecodeProfileError("profile spans did not close in nested order")
        active = self._active.pop(event_id)
        self._stack.pop()
        end_ns = self._now()
        if end_ns < active.start_ns:
            raise DecodeProfileError("profile clock moved backwards")
        self._events[event_id] = ProfileEvent(
            event_id=active.event_id,
            parent_id=active.parent_id,
            boundary=active.boundary,
            phase=active.phase,
            step_index=active.step_index,
            module_path=active.module_path,
            start_ns=active.start_ns,
            end_ns=end_ns,
        )

    def annotate(
        self,
        event_id: int,
        *,
        dispatch: str | None,
        guard_reason: str | None,
    ) -> None:
        event = self._events.get(event_id)
        if event is None:
            raise DecodeProfileError("cannot annotate an unfinished profile span")
        self._events[event_id] = replace(
            event, dispatch=dispatch, guard_reason=guard_reason
        )

    @contextmanager
    def span(self, boundary: str, *, module_path: str | None = None) -> Iterator[None]:
        event_id = self.begin(boundary, module_path=module_path)
        try:
            yield
        except BaseException as error:
            try:
                self.finish(event_id)
            except BaseException as profile_error:
                error.add_note(f"profile cleanup also failed: {profile_error}")
            raise
        else:
            self.finish(event_id)

    def snapshot(self) -> tuple[ProfileEvent, ...]:
        if self._stack or self._active:
            raise DecodeProfileError("profile contains unfinished spans")
        return tuple(self._events[index] for index in sorted(self._events))


def _counter_snapshot(module: nn.Module) -> dict[str, int] | None:
    adapter = getattr(module, "adapter", None)
    if adapter is None:
        return None
    try:
        counters = adapter.counters
    except (AttributeError, RuntimeError):
        return None
    values: dict[str, int] = {}
    for field in _COUNTER_FIELDS:
        value = getattr(counters, field, None)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        values[field] = value
    return values


def _classify_dispatch(
    before: Mapping[str, int] | None,
    after: Mapping[str, int] | None,
) -> str | None:
    if before is None or after is None:
        return None
    delta = {field: after[field] - before[field] for field in _COUNTER_FIELDS}
    if delta["forward"] != 1 or any(value < 0 for value in delta.values()):
        return "ambiguous"
    outcomes = {
        "native": {"forward": 1, "native_attempt": 1, "native_success": 1},
        "native_error": {"forward": 1, "native_attempt": 1, "native_error": 1},
        "fallback": {
            "forward": 1,
            "fallback_attempt": 1,
            "fallback_success": 1,
        },
        "fallback_error": {
            "forward": 1,
            "fallback_attempt": 1,
            "fallback_error": 1,
        },
        "predispatch_error": {"forward": 1, "predispatch_error": 1},
    }
    for outcome, nonzero in outcomes.items():
        expected = {field: nonzero.get(field, 0) for field in _COUNTER_FIELDS}
        if delta == expected:
            return outcome
    return "ambiguous"


def _guard_reason(module: nn.Module) -> str | None:
    adapter = getattr(module, "adapter", None)
    value = getattr(adapter, "last_guard_reason", None)
    return value if isinstance(value, str) else None


def _resolve_modules(
    model: nn.Module, module_paths: Sequence[str]
) -> tuple[tuple[str, nn.Module], ...]:
    if len(module_paths) > MAX_PROFILE_MODULES:
        raise DecodeProfileError("profile module limit exceeded")
    if len(set(module_paths)) != len(module_paths):
        raise DecodeProfileError("profile module paths must be unique")
    resolved: list[tuple[str, nn.Module]] = []
    seen_modules: set[int] = set()
    for path in module_paths:
        if not isinstance(path, str) or not path or len(path) > MAX_PROFILE_PATH_CHARS:
            raise DecodeProfileError("profile module paths must be nonempty text")
        try:
            module = model.get_submodule(path)
        except (AttributeError, KeyError) as error:
            raise DecodeProfileError(f"profile module is missing: {path}") from error
        if id(module) in seen_modules:
            raise DecodeProfileError(
                "profile paths cannot name one shared module twice"
            )
        seen_modules.add(id(module))
        resolved.append((path, module))
    return tuple(resolved)


@contextmanager
def _profile_module_forwards(
    model: nn.Module,
    module_paths: Sequence[str],
    collector: _ProfileCollector,
) -> Iterator[None]:
    resolved = _resolve_modules(model, module_paths)
    token = object()
    with _ACTIVE_MODELS_LOCK:
        if model in _ACTIVE_MODELS:
            raise DecodeProfileError("a profile is already active for this model")
        _ACTIVE_MODELS[model] = token

    handles: list[Any] = []
    pending: dict[tuple[str, int], list[tuple[int, dict[str, int] | None]]] = (
        defaultdict(list)
    )
    failed_pre_hooks: dict[tuple[str, int], int] = defaultdict(int)

    def make_pre(path: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def pre(module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            before = _counter_snapshot(module)
            key = (path, threading.get_ident())
            try:
                event_id = collector.begin("module_forward", module_path=path)
            except BaseException:
                failed_pre_hooks[key] += 1
                raise
            pending[key].append((event_id, before))

        return pre

    def make_post(
        path: str,
    ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def post(module: nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            thread_id = threading.get_ident()
            key = (path, thread_id)
            if failed_pre_hooks[key]:
                failed_pre_hooks[key] -= 1
                return
            stack = pending.get(key)
            if not stack:
                if (
                    collector._owner_thread is not None
                    and collector._owner_thread != thread_id
                ):
                    return
                raise DecodeProfileError("module profile hook has no matching start")
            event_id, before = stack.pop()
            collector.finish(event_id)
            collector.annotate(
                event_id,
                dispatch=_classify_dispatch(before, _counter_snapshot(module)),
                guard_reason=_guard_reason(module),
            )

        return post

    try:
        for path, module in resolved:
            handles.append(module.register_forward_pre_hook(make_pre(path)))
            handles.append(
                module.register_forward_hook(make_post(path), always_call=True)
            )
        try:
            yield
        except BaseException as error:
            if any(pending.values()):
                error.add_note("one or more module profile hooks did not close")
            raise
        else:
            if any(pending.values()):
                raise DecodeProfileError(
                    "one or more module profile hooks did not close"
                )
    finally:
        for handle in reversed(handles):
            with suppress(Exception):
                handle.remove()
        with _ACTIVE_MODELS_LOCK:
            if _ACTIVE_MODELS.get(model) is token:
                del _ACTIVE_MODELS[model]


def tinyllama_component_paths() -> tuple[str, ...]:
    """Return nonoverlapping component paths for the pinned TinyLlama topology."""

    paths = ["model.embed_tokens"]
    for layer in range(22):
        prefix = f"model.layers.{layer}"
        paths.extend(
            (
                f"{prefix}.input_layernorm",
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.post_attention_layernorm",
                f"{prefix}.mlp",
            )
        )
    paths.extend(("model.norm", "lm_head"))
    return tuple(paths)


def _event_records(events: Sequence[ProfileEvent]) -> list[dict[str, Any]]:
    if not events or len(events) > MAX_PROFILE_EVENTS:
        raise DecodeProfileError("profile must contain a bounded event tree")
    by_id = {event.event_id: event for event in events}
    if len(by_id) != len(events):
        raise DecodeProfileError("profile event IDs must be unique")
    roots = [event for event in events if event.parent_id is None]
    if len(roots) != 1 or roots[0].boundary != "generation":
        raise DecodeProfileError("profile must contain one generation root")
    children: dict[int, list[ProfileEvent]] = defaultdict(list)
    for event in events:
        if (
            isinstance(event.event_id, bool)
            or not isinstance(event.event_id, int)
            or event.event_id < 0
            or isinstance(event.start_ns, bool)
            or not isinstance(event.start_ns, int)
            or event.start_ns < 0
            or isinstance(event.end_ns, bool)
            or not isinstance(event.end_ns, int)
            or event.end_ns < event.start_ns
        ):
            raise DecodeProfileError("profile event has invalid identity or timing")
        if event.parent_id is not None:
            parent = by_id.get(event.parent_id)
            if (
                parent is None
                or parent.event_id >= event.event_id
                or event.start_ns < parent.start_ns
                or event.end_ns > parent.end_ns
            ):
                raise DecodeProfileError("profile child lies outside its parent")
            children[event.parent_id].append(event)

    records: list[dict[str, Any]] = []
    for event in events:
        direct = sorted(children[event.event_id], key=lambda child: child.start_ns)
        if any(left.end_ns > right.start_ns for left, right in pairwise(direct)):
            raise DecodeProfileError("profile sibling spans overlap")
        child_ns = sum(child.duration_ns for child in direct)
        exclusive_ns = event.duration_ns - child_ns
        if exclusive_ns < 0:
            raise DecodeProfileError("profile exclusive duration is negative")
        records.append(
            {
                "event_id": event.event_id,
                "parent_id": event.parent_id,
                "boundary": event.boundary,
                "phase": event.phase,
                "step_index": event.step_index,
                "module_path": event.module_path,
                "start_ns": event.start_ns,
                "end_ns": event.end_ns,
                "inclusive_ns": event.duration_ns,
                "exclusive_ns": exclusive_ns,
                "dispatch": event.dispatch,
                "guard_reason": event.guard_reason,
            }
        )
    return records


def _summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str | None, str | None], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for record in records:
        key = (
            str(record["boundary"]),
            str(record["phase"]),
            record["module_path"] if isinstance(record["module_path"], str) else None,
            record["dispatch"] if isinstance(record["dispatch"], str) else None,
        )
        groups[key].append(record)
    result: list[dict[str, Any]] = []
    for (boundary, phase, module_path, dispatch), values in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        inclusive = sorted(int(value["inclusive_ns"]) for value in values)
        exclusive = [int(value["exclusive_ns"]) for value in values]
        result.append(
            {
                "boundary": boundary,
                "phase": phase,
                "module_path": module_path,
                "dispatch": dispatch,
                "calls": len(values),
                "inclusive_total_ns": sum(inclusive),
                "exclusive_total_ns": sum(exclusive),
                "inclusive_min_ns": inclusive[0],
                "inclusive_median_low_ns": inclusive[(len(inclusive) - 1) // 2],
                "inclusive_max_ns": inclusive[-1],
            }
        )
    return result


@dataclass(frozen=True)
class DecodeProfile:
    """One diagnostic generation trace and its unchanged generated output."""

    generation: CachedGeneration
    events: tuple[ProfileEvent, ...]
    module_paths: tuple[str, ...]
    clock_name: str
    clock_resolution_ns: int | None

    def to_wire(self) -> dict[str, Any]:
        records = _event_records(self.events)
        return {
            "format": "decodeforge_decode_profile_v1",
            "schema_version": 1,
            "claim_class": "diagnostic_profile",
            "performance_claim_allowed": False,
            "clock": {
                "name": self.clock_name,
                "resolution_ns": self.clock_resolution_ns,
                "overhead_subtracted": False,
            },
            "generated_token_ids": list(self.generation.generated_ids),
            "generated_token_count": len(self.generation.generated_ids),
            "stop_reason": self.generation.stop_reason,
            "module_paths": list(self.module_paths),
            "events": records,
            "summary": _summaries(records),
            "interpretation": (
                "Diagnostic inclusive/exclusive timings include profiler overhead. "
                "They do not establish a performance improvement."
            ),
        }


def _validate_generation_request(
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> set[int]:
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= MAX_NEW_TOKENS
    ):
        raise DecodeProfileError("max_new_tokens must be in 1..64")
    if (
        input_ids.shape != attention_mask.shape
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.device.type != "cpu"
        or attention_mask.device.type != "cpu"
        or input_ids.dtype is not torch.int64
        or attention_mask.dtype is not torch.int64
    ):
        raise DecodeProfileError(
            "profile inputs must be one matching CPU int64 sequence"
        )

    try:
        return _eos_ids(tokenizer)
    except Exception as error:
        raise DecodeProfileError(str(error)) from error


def _generation_result(
    prompt: list[int], generated: list[int], eos: set[int]
) -> CachedGeneration:
    stop = "eos" if generated and generated[-1] in eos else "max_new_tokens"
    return CachedGeneration(
        prompt + generated,
        generated,
        stop,
        stop == "eos",
        [],
        [],
    )


def generate_cached_unprofiled(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> CachedGeneration:
    """Run the diagnostic generation workload with no hooks or timer reads."""

    eos = _validate_generation_request(
        tokenizer, input_ids, attention_mask, max_new_tokens
    )
    generated: list[int] = []
    with torch.inference_mode():
        ids = input_ids.detach().clone().contiguous()
        mask = attention_mask.detach().clone().contiguous()
        prompt = [int(value) for value in ids[0].tolist()]
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
        logits, past = _output_parts(output)
        token = int(torch.argmax(logits[0, -1, :]).item())
        generated.append(token)

        for _step_index in range(1, max_new_tokens):
            if generated[-1] in eos:
                break
            ids = torch.tensor([[generated[-1]]], dtype=torch.int64)
            mask = torch.cat(
                (mask, torch.ones((1, 1), dtype=torch.int64)),
                dim=1,
            )
            output = model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits, past = _output_parts(output)
            token = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(token)
    return _generation_result(prompt, generated, eos)


def profile_cached_generation(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    *,
    module_paths: Sequence[str] = (),
    clock: Clock = time.perf_counter_ns,
) -> DecodeProfile:
    """Profile one greedy CPU generation without touching benchmark protocols."""

    eos = _validate_generation_request(
        tokenizer, input_ids, attention_mask, max_new_tokens
    )
    paths = tuple(module_paths)
    collector = _ProfileCollector(clock)
    generated: list[int] = []
    with (
        _profile_module_forwards(model, paths, collector),
        collector.step("generation", -1),
        collector.span("generation"),
        torch.inference_mode(),
    ):
        with collector.step("prefill", 0), collector.span("prefill"):
            with collector.span("input_preparation"):
                ids = input_ids.detach().clone().contiguous()
                mask = attention_mask.detach().clone().contiguous()
                prompt = [int(value) for value in ids[0].tolist()]
            with collector.span("model_forward"):
                output = model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=True,
                    return_dict=True,
                )
            with collector.span("output_validation"):
                logits, past = _output_parts(output)
            with collector.span("token_selection"):
                token = int(torch.argmax(logits[0, -1, :]).item())
            with collector.span("bookkeeping"):
                generated.append(token)

        for step_index in range(1, max_new_tokens):
            if generated[-1] in eos:
                break
            with (
                collector.step("cached_decode", step_index),
                collector.span("cached_step"),
            ):
                with collector.span("input_preparation"):
                    ids = torch.tensor([[generated[-1]]], dtype=torch.int64)
                    mask = torch.cat(
                        (mask, torch.ones((1, 1), dtype=torch.int64)),
                        dim=1,
                    )
                with collector.span("model_forward"):
                    output = model(
                        input_ids=ids,
                        attention_mask=mask,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                with collector.span("output_validation"):
                    logits, past = _output_parts(output)
                with collector.span("token_selection"):
                    token = int(torch.argmax(logits[0, -1, :]).item())
                with collector.span("bookkeeping"):
                    generated.append(token)

    generation = _generation_result(prompt, generated, eos)
    uses_default_clock = clock is time.perf_counter_ns
    resolution = (
        max(1, int(time.get_clock_info("perf_counter").resolution * 1_000_000_000))
        if uses_default_clock
        else None
    )
    return DecodeProfile(
        generation,
        collector.snapshot(),
        paths,
        "time.perf_counter_ns" if uses_default_clock else "injected",
        resolution,
    )


__all__ = [
    "MAX_PROFILE_DEPTH",
    "MAX_PROFILE_EVENTS",
    "MAX_PROFILE_MODULES",
    "MAX_PROFILE_PATH_CHARS",
    "DecodeProfile",
    "DecodeProfileError",
    "ProfileEvent",
    "generate_cached_unprofiled",
    "profile_cached_generation",
    "tinyllama_component_paths",
]

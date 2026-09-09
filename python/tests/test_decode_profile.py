from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from decodeforge import decode_profile as profile
from decodeforge import evaluation
from torch import nn


class TickClock:
    def __init__(self, step: int = 10) -> None:
        self.value = 0
        self.step = step
        self.calls = 0

    def __call__(self) -> int:
        result = self.value
        self.value += self.step
        self.calls += 1
        return result


class ProfileModel(nn.Module):
    def __init__(self, tokens: list[int], *, fail_at: int | None = None) -> None:
        super().__init__()
        self.component: nn.Module = nn.Identity()
        self.tokens = tokens
        self.fail_at = fail_at
        self.calls: list[dict[str, Any]] = []
        self.cache = object()

    def forward(self, **kwargs: Any) -> Any:
        index = len(self.calls)
        self.calls.append(kwargs)
        self.component(kwargs["input_ids"].to(torch.float32))
        if index == self.fail_at:
            raise ValueError("model exploded")
        logits = torch.full((1, kwargs["input_ids"].shape[1], 8), -2.0)
        logits[0, -1, self.tokens[index]] = 2.0
        return SimpleNamespace(logits=logits, past_key_values=self.cache)


class NativeCounterModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values = {field: 0 for field in profile._COUNTER_FIELDS}

    @property
    def adapter(self) -> NativeCounterModule:
        return self

    @property
    def counters(self) -> Any:
        return SimpleNamespace(**self.values)

    @property
    def last_guard_reason(self) -> None:
        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.values["forward"] += 1
        self.values["native_attempt"] += 1
        self.values["native_success"] += 1
        return inputs


class BlockingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("blocking test timed out")
        return inputs


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([[3, 4, 5]], dtype=torch.int64)
    return ids, torch.ones_like(ids)


def test_profile_preserves_generation_and_partitions_nested_spans() -> None:
    clock = TickClock()
    model = ProfileModel([1, 2, 7])
    captured = profile.profile_cached_generation(
        model,
        SimpleNamespace(eos_token_id=7),
        *_inputs(),
        8,
        module_paths=("component",),
        clock=clock,
    )
    reference = evaluation.generate_cached(
        ProfileModel([1, 2, 7]),
        SimpleNamespace(eos_token_id=7),
        *_inputs(),
        8,
    )

    assert captured.generation.generated_ids == reference.generated_ids == [1, 2, 7]
    assert captured.generation.stop_reason == "eos"
    assert len(model.calls) == 3
    assert [call["attention_mask"].shape[1] for call in model.calls] == [3, 4, 5]

    wire = captured.to_wire()
    assert wire["claim_class"] == "diagnostic_profile"
    assert wire["performance_claim_allowed"] is False
    events = wire["events"]
    roots = [event for event in events if event["parent_id"] is None]
    assert len(roots) == 1
    assert roots[0]["boundary"] == "generation"
    assert roots[0]["phase"] == "generation"
    prefill = [event for event in events if event["boundary"] == "prefill"]
    assert len(prefill) == 1
    assert prefill[0]["parent_id"] == roots[0]["event_id"]
    cached_steps = [event for event in events if event["boundary"] == "cached_step"]
    assert len(cached_steps) == 2
    assert all(event["parent_id"] == roots[0]["event_id"] for event in cached_steps)
    assert prefill[0]["end_ns"] <= cached_steps[0]["start_ns"]
    components = [
        event
        for event in events
        if event["boundary"] == "module_forward" and event["module_path"] == "component"
    ]
    assert [event["phase"] for event in components] == [
        "prefill",
        "cached_decode",
        "cached_decode",
    ]
    assert all(event["inclusive_ns"] >= event["exclusive_ns"] >= 0 for event in events)
    assert captured.clock_resolution_ns is None
    assert wire["clock"]["name"] == "injected"
    assert clock.calls == 2 * len(events)
    captured.generation.generated_ids[0] = 6
    assert wire["generated_token_ids"] == [1, 2, 7]


def test_first_token_eos_has_no_cached_step() -> None:
    captured = profile.profile_cached_generation(
        ProfileModel([7]),
        SimpleNamespace(eos_token_id=7),
        *_inputs(),
        64,
        clock=TickClock(),
    )
    assert captured.generation.generated_ids == [7]
    assert not any(event.boundary == "cached_step" for event in captured.events)


@pytest.mark.parametrize("tokens", [[7], [1, 2, 7], [1, 2, 3, 4]])
def test_unprofiled_control_matches_profiled_workload(tokens: list[int]) -> None:
    maximum = len(tokens)
    control_model = ProfileModel(tokens)
    profiled_model = ProfileModel(tokens)
    tokenizer = SimpleNamespace(eos_token_id=7)

    control = profile.generate_cached_unprofiled(
        control_model, tokenizer, *_inputs(), maximum
    )
    captured = profile.profile_cached_generation(
        profiled_model,
        tokenizer,
        *_inputs(),
        maximum,
        module_paths=("component",),
        clock=TickClock(),
    )

    assert control == captured.generation
    assert len(control_model.calls) == len(profiled_model.calls)
    assert [call["input_ids"].tolist() for call in control_model.calls] == [
        call["input_ids"].tolist() for call in profiled_model.calls
    ]
    assert [call["attention_mask"].tolist() for call in control_model.calls] == [
        call["attention_mask"].tolist() for call in profiled_model.calls
    ]
    assert not control_model.component._forward_pre_hooks
    assert not control_model.component._forward_hooks


def test_hooks_are_removed_and_original_model_error_is_preserved() -> None:
    model = ProfileModel([1, 2], fail_at=1)
    original_pre_hooks = dict(model.component._forward_pre_hooks)
    original_hooks = dict(model.component._forward_hooks)

    with pytest.raises(ValueError, match="model exploded"):
        profile.profile_cached_generation(
            model,
            SimpleNamespace(eos_token_id=7),
            *_inputs(),
            4,
            module_paths=("component",),
            clock=TickClock(),
        )

    assert model.component._forward_pre_hooks == original_pre_hooks
    assert model.component._forward_hooks == original_hooks


def test_missing_duplicate_and_shared_module_paths_fail_before_forward() -> None:
    model = ProfileModel([1])
    model.alias = model.component
    for paths in (("missing",), ("component", "component"), ("component", "alias")):
        with pytest.raises(profile.DecodeProfileError):
            profile.profile_cached_generation(
                model,
                SimpleNamespace(eos_token_id=7),
                *_inputs(),
                1,
                module_paths=paths,
                clock=TickClock(),
            )
    assert model.calls == []


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        ({"forward": 1, "native_attempt": 1, "native_success": 1}, "native"),
        (
            {"forward": 1, "fallback_attempt": 1, "fallback_success": 1},
            "fallback",
        ),
        ({"forward": 1, "predispatch_error": 1}, "predispatch_error"),
        ({"forward": 2, "native_attempt": 2, "native_success": 2}, "ambiguous"),
        (
            {
                "forward": 1,
                "native_attempt": 1,
                "native_success": 1,
                "fallback_attempt": 1,
            },
            "ambiguous",
        ),
    ],
)
def test_dispatch_classification_uses_completed_counter_deltas(
    after: dict[str, int], expected: str
) -> None:
    before = {field: 0 for field in profile._COUNTER_FIELDS}
    complete_after = {**before, **after}
    assert profile._classify_dispatch(before, complete_after) == expected


def test_invalid_inputs_do_not_install_hooks_or_call_model() -> None:
    model = ProfileModel([1])
    ids, mask = _inputs()
    with pytest.raises(profile.DecodeProfileError):
        profile.profile_cached_generation(
            model,
            SimpleNamespace(eos_token_id=7),
            ids.to(torch.float32),
            mask,
            1,
            module_paths=("component",),
        )
    assert model.calls == []
    assert not model.component._forward_pre_hooks
    assert not model.component._forward_hooks


def test_overlapping_profiles_for_one_model_are_rejected_and_cleaned_up() -> None:
    model = ProfileModel([1])
    first = profile._ProfileCollector(TickClock())
    second = profile._ProfileCollector(TickClock())

    with (
        profile._profile_module_forwards(model, ("component",), first),
        pytest.raises(profile.DecodeProfileError, match="already active"),
        profile._profile_module_forwards(model, (), second),
    ):
        pass

    with profile._profile_module_forwards(model, (), second):
        pass
    assert not model.component._forward_pre_hooks
    assert not model.component._forward_hooks


def test_profile_rejects_a_clock_that_moves_backwards() -> None:
    values = iter((10, 9))
    collector = profile._ProfileCollector(lambda: next(values))

    with (
        pytest.raises(profile.DecodeProfileError, match="moved backwards"),
        collector.span("boundary"),
    ):
        pass


def test_profile_rejects_a_clock_that_moves_backwards_between_siblings() -> None:
    values = iter((100, 120, 130, 110, 140))
    collector = profile._ProfileCollector(lambda: next(values))

    root = collector.begin("generation")
    with collector.span("first"):
        pass
    with pytest.raises(profile.DecodeProfileError, match="moved backwards"):
        collector.begin("second")
    collector.finish(root)


def test_module_hook_uses_real_adapter_counter_deltas() -> None:
    model = ProfileModel([7])
    model.component = NativeCounterModule()
    captured = profile.profile_cached_generation(
        model,
        SimpleNamespace(eos_token_id=7),
        *_inputs(),
        4,
        module_paths=("component",),
        clock=TickClock(),
    )

    component = next(
        event for event in captured.events if event.boundary == "module_forward"
    )
    assert component.dispatch == "native"


def test_foreign_thread_cannot_close_the_owner_module_span() -> None:
    model = nn.Module()
    component = BlockingModule()
    model.add_module("component", component)
    collector = profile._ProfileCollector(TickClock())
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            component(torch.ones(1))
        except BaseException as error:
            errors.append(error)

    with profile._profile_module_forwards(model, ("component",), collector):
        owner = threading.Thread(target=invoke)
        owner.start()
        assert component.entered.wait(timeout=5)
        foreign = threading.Thread(target=invoke)
        foreign.start()
        foreign.join(timeout=5)
        component.release.set()
        owner.join(timeout=5)

    assert not owner.is_alive()
    assert not foreign.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], profile.DecodeProfileError)
    assert "multiple Python threads" in str(errors[0])
    events = collector.snapshot()
    assert len(events) == 1
    assert events[0].module_path == "component"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda events: (events[0], events[0]),
        lambda events: (replace(events[0], parent_id=events[0].event_id),),
        lambda events: (replace(events[0], boundary="not_generation"),),
    ],
)
def test_public_profile_rejects_malformed_event_trees(mutate: Any) -> None:
    captured = profile.profile_cached_generation(
        ProfileModel([7]),
        SimpleNamespace(eos_token_id=7),
        *_inputs(),
        1,
        clock=TickClock(),
    )
    malformed = replace(captured, events=mutate(captured.events))

    with pytest.raises(profile.DecodeProfileError):
        malformed.to_wire()


def test_event_limit_failure_cleans_up_hooks_and_model_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ProfileModel([7])
    monkeypatch.setattr(profile, "MAX_PROFILE_EVENTS", 1)
    with pytest.raises(profile.DecodeProfileError, match="event limit"):
        profile.profile_cached_generation(
            model,
            SimpleNamespace(eos_token_id=7),
            *_inputs(),
            1,
            module_paths=("component",),
            clock=TickClock(),
        )
    assert not model.component._forward_pre_hooks
    assert not model.component._forward_hooks

    monkeypatch.setattr(profile, "MAX_PROFILE_EVENTS", 20_000)
    profile.profile_cached_generation(
        model,
        SimpleNamespace(eos_token_id=7),
        *_inputs(),
        1,
        module_paths=("component",),
        clock=TickClock(),
    )


def test_tinyllama_component_inventory_is_bounded_and_unique() -> None:
    paths = profile.tinyllama_component_paths()
    assert len(paths) == 157
    assert len(set(paths)) == len(paths)
    assert len(paths) <= profile.MAX_PROFILE_MODULES

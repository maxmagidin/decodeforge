"""Real owning-adapter dispatch with scoped diagnostic instance wrappers."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, cast

import pytest
import torch
from decodeforge import qproj_profile as profile
from decodeforge import torch_bridge as bridge
from decodeforge.qproj_adapter import (
    QProjAdapter,
    QProjAdapterError,
    QProjExecutionMode,
    fallback_weight_identity,
)
from decodeforge.qproj_model import _InstalledQProj
from torch import nn


class MemoryLibrary:
    """Real RuntimeBinding ownership with a deterministic in-memory ABI stand-in."""

    def __init__(self) -> None:
        self.weight = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 8
        self.binding: bridge.RuntimeBinding | None = None
        self.run_error: BaseException | None = None
        self.close_failures = 0
        self.runs = 0
        self.destroyed = 0

    def create_binding(self, _manifest: Any, _packed: Any) -> bridge.RuntimeBinding:
        descriptor = bridge.RuntimeDescriptor(
            n=4,
            k=8,
            packed_weight_bytes=144,
            module_id="sha256:" + "a" * 64,
            packed_weight_id="sha256:" + "b" * 64,
        )
        self.binding = bridge.RuntimeBinding(
            cast(bridge.RuntimeLibrary, self), 1, descriptor
        )
        return self.binding

    def run(
        self,
        _handle: int,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        self.runs += 1
        if self.run_error is not None:
            raise self.run_error
        source = (ctypes.c_float * input_length).from_address(input_address)
        destination = (ctypes.c_float * output_length).from_address(output_address)
        result = torch.nn.functional.linear(torch.tensor(list(source)), self.weight)
        for index, value in enumerate(result.tolist()):
            destination[index] = value

    def destroy(self, _handle: int) -> None:
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected destroy failure")
        self.destroyed += 1


class Recorder:
    def __init__(self, fail: tuple[str, str] | None = None) -> None:
        self.stack: list[str] = []
        self.records: list[tuple[str, str | None, str]] = []
        self.fail = fail

    @contextmanager
    def span(self, boundary: str, path: str) -> Iterator[None]:
        if self.fail == (boundary, "enter"):
            raise RuntimeError("observer entry failed")
        self.records.append((boundary, self.stack[-1] if self.stack else None, path))
        self.stack.append(boundary)
        try:
            yield
        finally:
            assert self.stack.pop() == boundary
            if self.fail == (boundary, "exit"):
                raise RuntimeError("observer exit failed")


@pytest.fixture
def owning_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[nn.Module, QProjAdapter, MemoryLibrary, bridge.BindingRegistry]]:
    registry = bridge.BindingRegistry()
    monkeypatch.setattr(bridge, "_BINDINGS", registry)
    library = MemoryLibrary()
    adapter = QProjAdapter(
        layer_name="q_proj",
        library=cast(bridge.RuntimeLibrary, library),
        pack_manifest_json=b"{}",
        packed_weight=b"packed",
        fallback_weight=library.weight,
        fallback_weight_id=fallback_weight_identity(library.weight),
        fallback_parent_packed_weight_id="sha256:" + "b" * 64,
        native_operator=bridge._native_q8_linear,
    )
    model = nn.Module()
    model.add_module("q_proj", _InstalledQProj(adapter))
    try:
        yield model, adapter, library, registry
    finally:
        adapter.close()
        registry.clear()


def _input(tokens: int) -> torch.Tensor:
    return torch.arange(tokens * 8, dtype=torch.float32).reshape(1, tokens, 8) / 8


def _state(adapter: QProjAdapter, registry: bridge.BindingRegistry) -> tuple[Any, ...]:
    return (
        dict(adapter.__dict__),
        dict(adapter._callable.__dict__),
        registry.get(cast(int, adapter._binding_id)),
    )


def _assert_restored(
    adapter: QProjAdapter, registry: bridge.BindingRegistry, state: tuple[Any, ...]
) -> None:
    # Forward counters legitimately advance; instrumentation attributes must
    # have exactly their original presence and original callable objects.
    for owner, original, names in (
        (adapter, state[0], ("_validate_fallback_storage", "_fallback")),
        (adapter._callable, state[1], ("fallback", "_native_operator")),
    ):
        for name in names:
            assert (name in owner.__dict__) == (name in original)
            if name in original:
                assert owner.__dict__[name] is original[name]
    assert registry.get(cast(int, adapter._binding_id)) is state[2]


@pytest.mark.parametrize(
    ("mode", "tokens", "native"),
    [
        (QProjExecutionMode.HYBRID_NATIVE, 1, True),
        (QProjExecutionMode.HYBRID_NATIVE, 3, False),
        (QProjExecutionMode.SAME_Q8_REFERENCE, 1, False),
    ],
)
def test_real_adapter_paths_have_exact_spans_and_restore(
    owning_adapter: Any, mode: QProjExecutionMode, tokens: int, native: bool
) -> None:
    model, adapter, library, registry = owning_adapter
    adapter.set_execution_mode(mode)
    inputs = _input(tokens)
    with torch.inference_mode():
        expected = adapter(inputs)
    before = adapter.counters
    original = _state(adapter, registry)
    recorder = Recorder()
    with profile.profile_qproj_internals(model, ("q_proj",), recorder.span):
        assert registry.get(adapter._binding_id) is not library.binding
        with torch.inference_mode():
            actual = adapter(inputs)
    assert torch.equal(actual, expected)
    after = adapter.counters
    assert after.forward - before.forward == 1
    assert after.native_success - before.native_success == int(native)
    assert after.fallback_success - before.fallback_success == int(not native)
    assert after.native_error == after.fallback_error == 0
    assert after.in_flight == 0
    _assert_restored(adapter, registry, original)
    if native:
        assert recorder.records == [
            ("adapter_storage_guard", None, "q_proj"),
            ("guarded_native_operator", None, "q_proj"),
            ("guarded_binding_run", "guarded_native_operator", "q_proj"),
        ]
    else:
        assert recorder.records == [
            ("adapter_storage_guard", None, "q_proj"),
            ("fallback", None, "q_proj"),
            ("adapter_storage_guard", "fallback", "q_proj"),
            ("fallback_clone", "fallback", "q_proj"),
            ("fallback_hash", "fallback", "q_proj"),
            ("fallback_linear", "fallback", "q_proj"),
        ]
    count = len(recorder.records)
    with torch.inference_mode():
        adapter(inputs)
    assert len(recorder.records) == count


@pytest.mark.parametrize(
    ("boundary", "when", "tokens"),
    [
        ("adapter_storage_guard", "enter", 1),
        ("guarded_native_operator", "exit", 1),
        ("guarded_binding_run", "enter", 1),
        ("guarded_binding_run", "exit", 1),
        ("fallback_clone", "enter", 2),
        ("fallback_hash", "exit", 2),
    ],
)
def test_observer_errors_are_deferred_without_corrupting_dispatch(
    owning_adapter: Any, boundary: str, when: str, tokens: int
) -> None:
    model, adapter, library, registry = owning_adapter
    original = _state(adapter, registry)
    recorder = Recorder((boundary, when))
    with (
        pytest.raises(profile.QProjProfileError, match="profiling or restoration"),
        profile.profile_qproj_internals(model, ("q_proj",), recorder.span),
    ):
        with torch.inference_mode():
            actual = adapter(_input(tokens))
        assert torch.equal(
            actual, torch.nn.functional.linear(_input(tokens), library.weight)
        )
        assert adapter.counters.forward == 1
        assert adapter.counters.native_error == adapter.counters.fallback_error == 0
    assert adapter.counters.native_success == int(tokens == 1)
    assert adapter.counters.fallback_success == int(tokens != 1)
    _assert_restored(adapter, registry, original)


def test_native_error_stays_primary_and_never_retries_fallback(
    owning_adapter: Any,
) -> None:
    model, adapter, library, registry = owning_adapter
    error = RuntimeError("native failed")
    library.run_error = error
    original = _state(adapter, registry)
    recorder = Recorder(("guarded_binding_run", "exit"))
    with (
        pytest.raises(RuntimeError, match="native failed") as observed,
        profile.profile_qproj_internals(model, ("q_proj",), recorder.span),
        torch.inference_mode(),
    ):
        adapter(_input(1))
    assert observed.value is error
    assert adapter.counters.native_error == 1
    assert adapter.counters.fallback_attempt == 0
    assert adapter.counters.in_flight == 0
    _assert_restored(adapter, registry, original)


def test_mutated_fallback_hash_is_still_rejected(owning_adapter: Any) -> None:
    model, adapter, _library, registry = owning_adapter
    adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
    original = _state(adapter, registry)
    # A NumPy write bypasses the Torch version counter, exercising content hash.
    adapter.same_q8_weight.numpy()[0, 0] += 1
    with (
        pytest.raises(QProjAdapterError, match="content no longer matches"),
        profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
        torch.inference_mode(),
    ):
        adapter(_input(1))
    assert adapter.counters.fallback_error == 1
    assert adapter.counters.native_attempt == 0
    _assert_restored(adapter, registry, original)


def test_partial_install_failure_restores_registry_and_attributes(
    owning_adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, adapter, _library, registry = owning_adapter
    original = _state(adapter, registry)
    replace = profile._replace_attribute

    def fail_late(owner: Any, name: str, value: Any, undo: Any) -> None:
        if name == "_native_operator":
            raise RuntimeError("partial setup failed")
        replace(owner, name, value, undo)

    with monkeypatch.context() as scoped:
        scoped.setattr(profile, "_replace_attribute", fail_late)
        with (
            pytest.raises(RuntimeError, match="partial setup failed"),
            profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
        ):
            pytest.fail("partial setup must not enter the body")
    _assert_restored(adapter, registry, original)
    with profile.profile_qproj_internals(model, ("q_proj",), Recorder().span):
        pass


def test_second_adapter_install_failure_rolls_back_first_adapter(
    owning_adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, adapter, _library, registry = owning_adapter
    second_library = MemoryLibrary()
    second = QProjAdapter(
        layer_name="other",
        library=cast(bridge.RuntimeLibrary, second_library),
        pack_manifest_json=b"{}",
        packed_weight=b"packed",
        fallback_weight=second_library.weight,
        fallback_weight_id=fallback_weight_identity(second_library.weight),
        fallback_parent_packed_weight_id="sha256:" + "b" * 64,
        native_operator=bridge._native_q8_linear,
    )
    model.add_module("other", _InstalledQProj(second))
    first_state, second_state = _state(adapter, registry), _state(second, registry)
    install_target = profile._install_target

    def fail_second(target: Any, observer: Any, bindings: Any, undo: Any) -> None:
        install_target(target, observer, bindings, undo)
        if target.adapter is second:
            raise RuntimeError("second adapter setup failed")

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(profile, "_install_target", fail_second)
            with (
                pytest.raises(RuntimeError, match="second adapter setup"),
                profile.profile_qproj_internals(
                    model, ("q_proj", "other"), Recorder().span
                ),
            ):
                pytest.fail("partial setup must not enter the body")
        _assert_restored(adapter, registry, first_state)
        _assert_restored(second, registry, second_state)
        with profile.profile_qproj_internals(
            model, ("q_proj", "other"), Recorder().span
        ):
            pass
    finally:
        second.close()


def test_observer_cannot_suppress_a_production_storage_error(
    owning_adapter: Any,
) -> None:
    model, adapter, _library, registry = owning_adapter
    original = _state(adapter, registry)

    @contextmanager
    def suppressing_observer(_boundary: str, _path: str) -> Iterator[None]:
        with suppress(QProjAdapterError):
            yield

    with torch.no_grad():
        adapter.same_q8_weight.add_(1)
    with (
        pytest.raises(QProjAdapterError, match="replaced or mutated"),
        profile.profile_qproj_internals(model, ("q_proj",), suppressing_observer),
        torch.inference_mode(),
    ):
        adapter(_input(1))
    assert adapter.counters.predispatch_error == 1
    assert adapter.counters.native_attempt == adapter.counters.fallback_attempt == 0
    _assert_restored(adapter, registry, original)


def test_overridden_fallback_is_rejected_before_mutation(owning_adapter: Any) -> None:
    model, adapter, _library, registry = owning_adapter
    original_fallback = adapter._callable.fallback
    try:
        adapter._callable.fallback = lambda value: value
        original = _state(adapter, registry)
        with (
            pytest.raises(profile.QProjProfileError, match="production fallback"),
            profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
        ):
            pass
        _assert_restored(adapter, registry, original)
    finally:
        adapter._callable.fallback = original_fallback


def test_overlapping_profiles_and_duplicate_adapters_are_rejected(
    owning_adapter: Any,
) -> None:
    model, adapter, _library, registry = owning_adapter
    original = _state(adapter, registry)
    with (
        profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
        pytest.raises(profile.QProjProfileError, match="already active"),
        profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
    ):
        pass
    with (
        pytest.raises(profile.QProjProfileError, match="two paths"),
        profile.profile_qproj_internals(model, ("q_proj", "q_proj"), Recorder().span),
    ):
        pass
    _assert_restored(adapter, registry, original)


@pytest.mark.parametrize("via_original", [True, False])
def test_proxy_preserves_real_binding_close_ownership(
    owning_adapter: Any, via_original: bool
) -> None:
    model, adapter, library, registry = owning_adapter
    binding_id = adapter._binding_id
    with profile.profile_qproj_internals(model, ("q_proj",), Recorder().span):
        if via_original:
            library.binding.close()
        else:
            adapter.close()
    assert library.binding.closed
    assert library.binding._registry_owner is None
    assert registry.get(binding_id) is None
    assert library.destroyed == 1
    # Direct original close bypasses the adapter's own lifecycle. Detach that
    # already-closed ID solely for fixture cleanup.
    if via_original:
        adapter._binding_id = None


def test_proxy_close_failure_retains_original_owner_for_retry(
    owning_adapter: Any,
) -> None:
    model, adapter, library, registry = owning_adapter
    original = _state(adapter, registry)
    library.close_failures = 1
    with (
        pytest.raises(RuntimeError, match="destroy failure"),
        profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
    ):
        adapter.close()
    _assert_restored(adapter, registry, original)
    adapter.close()
    assert library.binding.closed and library.destroyed == 1


def test_nonowning_modules_do_not_silently_claim_internal_coverage() -> None:
    model = nn.Module()
    model.add_module("q_proj", nn.Linear(8, 4))
    with (
        pytest.raises(profile.QProjProfileError, match="owning QProjAdapter"),
        profile.profile_qproj_internals(model, ("q_proj",), Recorder().span),
    ):
        pass

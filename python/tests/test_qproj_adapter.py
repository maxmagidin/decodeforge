"""Focused one-layer checks for the owning query-projection adapter."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from decodeforge import torch_bridge as bridge  # noqa: E402
from decodeforge.qproj_adapter import (  # noqa: E402
    SAME_Q8_REFERENCE_REASON,
    QProjAdapter,
    QProjAdapterError,
    QProjAdapterInitializationError,
    QProjCounters,
    QProjExecutionMode,
    fallback_weight_identity,
)

MODULE_ID = "sha256:" + "a" * 64
PACK_ID = "sha256:" + "b" * 64


def _call_bounded(callback: Callable[[], Any]) -> Any:
    """Run a potentially reentrant callback without allowing a deadlock."""

    outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue()

    def worker() -> None:
        try:
            outcomes.put((True, callback()))
        except BaseException as error:
            outcomes.put((False, error))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "reentrant adapter call deadlocked"
    succeeded, value = outcomes.get_nowait()
    if not succeeded:
        raise cast(BaseException, value)
    return value


class FakeBinding:
    def __init__(
        self,
        n: int = 4,
        k: int = 8,
        *,
        pack_id: str = PACK_ID,
        close_failures: int = 0,
    ) -> None:
        self.descriptor = bridge.RuntimeDescriptor(
            n=n,
            k=k,
            packed_weight_bytes=((n + 3) // 4) * ((k + 31) // 32) * 144,
            module_id=MODULE_ID,
            packed_weight_id=pack_id,
        )
        self.closed = False
        self.close_calls = 0
        self.close_failures = close_failures

    def run(
        self,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        del input_address, input_length, output_address, output_length

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("injected close failure")
        self.closed = True


class FakeLibrary:
    def __init__(self, binding: FakeBinding) -> None:
        self.binding = binding
        self.create_calls = 0

    def create_binding(
        self,
        manifest_json: bytes | bytearray | memoryview,
        packed_weight: bytes | bytearray | memoryview,
    ) -> FakeBinding:
        assert bytes(manifest_json) == b"{}"
        assert bytes(packed_weight) == b"packed"
        self.create_calls += 1
        return self.binding


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> bridge.BindingRegistry:
    value = bridge.BindingRegistry()
    monkeypatch.setattr(bridge, "_BINDINGS", value)
    return value


def _weight() -> Any:
    return torch.tensor(
        [
            [0.25, -0.5, 1.0, 0.0, 0.75, -1.25, 2.0, 0.5],
            [-0.75, 0.5, 0.25, 1.5, -1.0, 0.0, 0.5, 0.25],
            [1.0, 1.25, -0.25, 0.5, 0.0, -0.5, 0.75, -1.0],
            [0.0, 0.5, -1.5, 0.25, 1.0, 0.75, -0.25, 0.5],
        ],
        dtype=torch.float32,
    )


def _adapter(
    registry: bridge.BindingRegistry,
    *,
    binding: FakeBinding | None = None,
    weight: Any | None = None,
    pack_id: str = PACK_ID,
    native: Callable[[Any, int, int, int], Any] | None = None,
) -> tuple[QProjAdapter, FakeBinding, Any]:
    del registry
    selected_binding = FakeBinding() if binding is None else binding
    selected_weight = _weight() if weight is None else weight

    def default_native(x: Any, _binding_id: int, _n: int, _k: int) -> Any:
        return torch.nn.functional.linear(x, selected_weight)

    adapter = QProjAdapter(
        layer_name="model.layers.0.self_attn.q_proj",
        library=FakeLibrary(selected_binding),  # type: ignore[arg-type]
        pack_manifest_json=b"{}",
        packed_weight=b"packed",
        fallback_weight=selected_weight,
        fallback_weight_id=fallback_weight_identity(selected_weight),
        fallback_parent_packed_weight_id=pack_id,
        expected_module_id=MODULE_ID,
        native_operator=default_native if native is None else native,
    )
    return adapter, selected_binding, selected_weight


def test_base_import_remains_torch_free() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import decodeforge; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_adapter_owns_identity_bound_nonpersistent_fallback(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, binding, source_weight = _adapter(registry)
    assert adapter.metadata.layer_name == "model.layers.0.self_attn.q_proj"
    assert (adapter.metadata.n, adapter.metadata.k) == (4, 8)
    assert adapter.metadata.module_id == MODULE_ID
    assert adapter.metadata.packed_weight_id == PACK_ID
    assert adapter.metadata.fallback_weight_id == fallback_weight_identity(
        source_weight
    )
    assert adapter.same_q8_weight.data_ptr() != source_weight.data_ptr()
    assert not adapter.same_q8_weight.requires_grad
    assert "same_q8_weight" not in adapter.state_dict()
    assert not binding.closed
    adapter.close()
    assert binding.closed and binding.close_calls == 1


def test_m1_native_matches_exact_fallback_and_m_gt_one_falls_back(
    registry: bridge.BindingRegistry,
) -> None:
    native_calls: list[Any] = []
    weight = _weight()

    def native(x: Any, _binding_id: int, _n: int, _k: int) -> Any:
        native_calls.append(x)
        return torch.nn.functional.linear(x, weight)

    adapter, _binding, _weight_source = _adapter(registry, weight=weight, native=native)
    decode = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    expected_decode = torch.nn.functional.linear(decode, weight)
    assert torch.equal(adapter(decode), expected_decode)
    assert native_calls == [decode]

    prefill = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)
    expected_prefill = torch.nn.functional.linear(prefill, weight)
    assert torch.equal(adapter(prefill), expected_prefill)
    assert native_calls == [decode]
    assert adapter.last_guard_reason == "m_gt_one"
    assert adapter.counters == QProjCounters(
        forward=2,
        native_attempt=1,
        native_success=1,
        native_error=0,
        fallback_attempt=1,
        fallback_success=1,
        fallback_error=0,
        predispatch_error=0,
        rejected_closed=0,
        in_flight=0,
        closed=False,
    )
    adapter.close()


def test_noncontiguous_and_grad_inputs_use_fallback(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, _binding, weight = _adapter(
        registry,
        native=lambda *_args: pytest.fail("guard miss entered native"),
    )
    base = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
    noncontiguous = base[..., ::2]
    assert not noncontiguous.is_contiguous()
    assert torch.equal(
        adapter(noncontiguous), torch.nn.functional.linear(noncontiguous, weight)
    )
    assert adapter.last_guard_reason == "non_contiguous"

    requires_grad = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    requires_grad.requires_grad_(True)
    result = adapter(requires_grad)
    assert torch.equal(result, torch.nn.functional.linear(requires_grad, weight))
    assert result.requires_grad
    assert adapter.last_guard_reason == "requires_grad"
    assert adapter.counters.fallback_success == 2
    adapter.close()


@pytest.mark.parametrize(
    ("value", "guard_reason"),
    [
        (lambda: torch.ones((1, 1, 8), dtype=torch.float64), "dtype"),
        (lambda: torch.ones((1, 1, 7), dtype=torch.float32), "shape"),
    ],
)
def test_wrong_dtype_and_shape_are_observable_fallback_errors(
    registry: bridge.BindingRegistry,
    value: Callable[[], Any],
    guard_reason: str,
) -> None:
    adapter, _binding, _weight_source = _adapter(
        registry,
        native=lambda *_args: pytest.fail("guard miss entered native"),
    )
    with pytest.raises(RuntimeError):
        adapter(value())
    assert adapter.last_guard_reason == guard_reason
    assert adapter.counters.fallback_error == 1
    assert adapter.counters.fallback_success == 0
    adapter.close()


def test_injected_native_error_is_hard_and_never_reruns_fallback(
    registry: bridge.BindingRegistry,
) -> None:
    def fail_native(*_args: Any) -> Any:
        raise bridge.TorchBridgeError(bridge.BridgeStatus.EXECUTION_FAILED, "injected")

    adapter, _binding, _weight_source = _adapter(registry, native=fail_native)
    with pytest.raises(bridge.TorchBridgeError, match="injected"):
        adapter(torch.ones((1, 1, 8), dtype=torch.float32))
    assert adapter.last_guard_reason is None
    assert adapter.counters == QProjCounters(
        forward=1,
        native_attempt=1,
        native_success=0,
        native_error=1,
        fallback_attempt=0,
        fallback_success=0,
        fallback_error=0,
        predispatch_error=0,
        rejected_closed=0,
        in_flight=0,
        closed=False,
    )
    adapter.close()


def test_close_from_admitted_native_callback_fails_fast(
    registry: bridge.BindingRegistry,
) -> None:
    adapter: QProjAdapter

    def close_native(*_args: Any) -> Any:
        adapter.close()
        return pytest.fail("close unexpectedly returned")

    adapter, binding, _weight_source = _adapter(registry, native=close_native)
    with pytest.raises(QProjAdapterError, match="admitted forward"):
        _call_bounded(lambda: adapter(torch.ones((1, 1, 8), dtype=torch.float32)))
    assert adapter.counters.native_error == 1
    assert adapter.counters.in_flight == 0
    adapter.close()
    assert binding.closed


def test_recursive_native_forward_fails_fast(
    registry: bridge.BindingRegistry,
) -> None:
    adapter: QProjAdapter

    def recursive_native(*args: Any) -> Any:
        return adapter(args[0])

    adapter, binding, _weight_source = _adapter(registry, native=recursive_native)
    with pytest.raises(QProjAdapterError, match="recursive forward"):
        _call_bounded(lambda: adapter(torch.ones((1, 1, 8), dtype=torch.float32)))
    assert adapter.counters.native_error == 1
    assert adapter.counters.in_flight == 0
    adapter.close()
    assert binding.closed


def test_identity_shape_and_module_mismatches_close_new_binding(
    registry: bridge.BindingRegistry,
) -> None:
    cases = [
        (FakeBinding(pack_id="sha256:" + "c" * 64), _weight(), PACK_ID, MODULE_ID),
        (FakeBinding(n=3, k=8), _weight(), PACK_ID, MODULE_ID),
        (FakeBinding(), _weight(), PACK_ID, "sha256:" + "d" * 64),
    ]
    for binding, weight, parent_id, module_id in cases:
        with pytest.raises(QProjAdapterError):
            QProjAdapter(
                layer_name="model.layers.0.self_attn.q_proj",
                library=FakeLibrary(binding),  # type: ignore[arg-type]
                pack_manifest_json=b"{}",
                packed_weight=b"packed",
                fallback_weight=weight,
                fallback_weight_id=fallback_weight_identity(weight),
                fallback_parent_packed_weight_id=parent_id,
                expected_module_id=module_id,
                native_operator=lambda *_args: None,
            )
        assert binding.closed and binding.close_calls == 1


def test_declared_fallback_hash_is_checked_before_loading(
    registry: bridge.BindingRegistry,
) -> None:
    del registry
    binding = FakeBinding()
    library = FakeLibrary(binding)
    with pytest.raises(QProjAdapterError, match="does not equal declared"):
        QProjAdapter(
            layer_name="model.layers.0.self_attn.q_proj",
            library=library,  # type: ignore[arg-type]
            pack_manifest_json=b"{}",
            packed_weight=b"packed",
            fallback_weight=_weight(),
            fallback_weight_id="sha256:" + "0" * 64,
            fallback_parent_packed_weight_id=PACK_ID,
        )
    assert library.create_calls == 0
    assert binding.close_calls == 0


def test_close_is_idempotent_and_closed_forward_is_observable(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, binding, _weight_source = _adapter(registry)
    adapter.close()
    adapter.close()
    assert binding.close_calls == 1
    with pytest.raises(QProjAdapterError, match="closed"):
        adapter(torch.ones((1, 1, 8), dtype=torch.float32))
    assert adapter.counters.rejected_closed == 1
    assert adapter.counters.closed


def test_close_waits_for_admitted_forward_then_destroys_once(
    registry: bridge.BindingRegistry,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_native(x: Any, _binding_id: int, _n: int, _k: int) -> Any:
        entered.set()
        assert release.wait(timeout=5)
        return torch.zeros((*x.shape[:-1], 4), dtype=torch.float32)

    adapter, binding, _weight_source = _adapter(registry, native=blocking_native)
    result: list[Any] = []
    worker = threading.Thread(
        target=lambda: result.append(
            adapter(torch.ones((1, 1, 8), dtype=torch.float32))
        )
    )
    worker.start()
    assert entered.wait(timeout=5)
    assert adapter.counters.in_flight == 1

    closer = threading.Thread(target=adapter.close)
    closer.start()
    closer.join(timeout=0.05)
    assert closer.is_alive()
    assert binding.close_calls == 0
    release.set()
    worker.join(timeout=5)
    closer.join(timeout=5)
    assert not worker.is_alive() and not closer.is_alive()
    assert len(result) == 1
    assert binding.close_calls == 1
    assert adapter.counters.closed and adapter.counters.in_flight == 0


def test_standard_in_place_fallback_mutation_fails_closed(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, _binding, _weight_source = _adapter(registry)
    adapter.same_q8_weight.add_(1.0)
    with pytest.raises(QProjAdapterError, match="replaced or mutated"):
        adapter(torch.ones((1, 1, 8), dtype=torch.float32))
    assert adapter.counters.predispatch_error == 1
    assert adapter.counters.forward == 1
    adapter.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda weight: weight.data.add_(1.0),
        lambda weight: weight.numpy().__setitem__((0, 0), 12345.0),
        lambda weight: weight.untyped_storage().__setitem__(
            0, (weight.untyped_storage()[0] + 1) % 256
        ),
    ],
)
def test_alias_fallback_mutation_cannot_produce_a_wrong_result(
    registry: bridge.BindingRegistry,
    mutate: Callable[[Any], None],
) -> None:
    adapter, _binding, _weight_source = _adapter(registry)
    original_version = adapter.same_q8_weight._version
    mutate(adapter.same_q8_weight)
    assert adapter.same_q8_weight._version == original_version

    with pytest.raises(QProjAdapterError, match="no longer matches its identity"):
        adapter(torch.ones((1, 2, 8), dtype=torch.float32))
    counters = adapter.counters
    assert counters.forward == 1
    assert counters.fallback_attempt == 1
    assert counters.fallback_error == 1
    assert counters.fallback_success == 0
    assert counters.predispatch_error == 0
    adapter.close()


def test_close_failure_retains_retryable_ownership_and_rejects_forward(
    registry: bridge.BindingRegistry,
) -> None:
    binding = FakeBinding(close_failures=1)
    adapter, _binding, _weight_source = _adapter(registry, binding=binding)

    with pytest.raises(RuntimeError, match="injected close failure"):
        adapter.close()
    assert not adapter.closed
    assert registry.get(1) is binding
    with pytest.raises(QProjAdapterError, match="closed"):
        adapter(torch.ones((1, 1, 8), dtype=torch.float32))

    adapter.close()
    adapter.close()
    assert adapter.closed
    assert binding.close_calls == 2
    assert registry.get(1) is None
    assert adapter.counters.rejected_closed == 1


def test_constructor_cleanup_failure_returns_a_retryable_owner(
    registry: bridge.BindingRegistry,
) -> None:
    binding = FakeBinding(pack_id="sha256:" + "c" * 64, close_failures=1)

    with pytest.raises(QProjAdapterInitializationError) as caught:
        QProjAdapter(
            layer_name="model.layers.0.self_attn.q_proj",
            library=FakeLibrary(binding),  # type: ignore[arg-type]
            pack_manifest_json=b"{}",
            packed_weight=b"packed",
            fallback_weight=_weight(),
            fallback_weight_id=fallback_weight_identity(_weight()),
            fallback_parent_packed_weight_id=PACK_ID,
        )

    owner = caught.value
    assert isinstance(owner.initialization_error, QProjAdapterError)
    assert isinstance(owner.cleanup_error, RuntimeError)
    assert owner.binding_id == 1
    assert not owner.closed
    assert registry.get(owner.binding_id) is binding
    owner.close()
    owner.close()
    assert owner.closed
    assert binding.close_calls == 2
    assert registry.get(owner.binding_id) is None


def test_adapter_is_permanently_eval_and_rejects_apply_transformations(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, _binding, _weight_source = _adapter(registry)
    original = adapter.same_q8_weight
    assert not adapter.training
    assert adapter.eval() is adapter
    assert adapter.train(False) is adapter
    with pytest.raises(QProjAdapterError, match="inference-only"):
        adapter.train()

    for transform in (
        adapter.cpu,
        adapter.float,
        adapter.double,
        lambda: adapter.to(dtype=torch.float64),
    ):
        with pytest.raises(QProjAdapterError, match="does not support"):
            transform()
        assert adapter.same_q8_weight is original
        assert adapter.same_q8_weight.dtype is torch.float32
        assert adapter.same_q8_weight.device.type == "cpu"
    adapter.close()


def test_state_dict_is_empty_and_loading_is_explicitly_rejected(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, _binding, _weight_source = _adapter(registry)
    parent = torch.nn.Module()
    parent.projection = adapter
    assert adapter.state_dict() == {}
    assert parent.state_dict() == {}

    with pytest.raises(QProjAdapterError, match="rebuilt from prepared assets"):
        adapter.load_state_dict({})
    with pytest.raises(QProjAdapterError, match="rebuilt from prepared assets"):
        parent.load_state_dict({})
    adapter.close()


def test_execution_modes_are_closed_state_and_visible_in_repr(
    registry: bridge.BindingRegistry,
) -> None:
    adapter, _binding, _weight_source = _adapter(registry)
    assert [mode.value for mode in QProjExecutionMode] == [
        "hybrid_native",
        "same_q8_reference",
    ]
    assert adapter.execution_mode is QProjExecutionMode.HYBRID_NATIVE
    assert "execution_mode='hybrid_native'" in repr(adapter)
    assert adapter.state_dict() == {}

    previous = adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
    assert previous is QProjExecutionMode.HYBRID_NATIVE
    assert adapter.execution_mode.value == "same_q8_reference"
    assert "execution_mode='same_q8_reference'" in repr(adapter)
    assert (
        adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
        is QProjExecutionMode.SAME_Q8_REFERENCE
    )
    with pytest.raises(TypeError, match="QProjExecutionMode"):
        adapter.set_execution_mode(cast(Any, "hybrid_native"))
    assert adapter.execution_mode.value == "same_q8_reference"
    adapter.close()


def test_same_q8_reference_forces_all_shapes_through_checked_fallback(
    registry: bridge.BindingRegistry,
) -> None:
    native_calls: list[Any] = []

    def native(*args: Any) -> Any:
        native_calls.append(args)
        return pytest.fail("same-Q8 reference mode entered native")

    adapter, _binding, weight = _adapter(registry, native=native)
    adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
    inputs = (
        torch.arange(8, dtype=torch.float32).reshape(1, 1, 8),
        torch.arange(40, dtype=torch.float32).reshape(1, 5, 8),
    )
    for value in inputs:
        assert torch.equal(adapter(value), torch.nn.functional.linear(value, weight))
        assert adapter.last_guard_reason == SAME_Q8_REFERENCE_REASON

    assert native_calls == []
    counters = adapter.counters
    assert counters.forward == 2
    assert counters.native_attempt == 0
    assert counters.fallback_attempt == 2
    assert counters.fallback_success == 2
    assert counters.fallback_error == 0

    adapter.same_q8_weight.data.add_(1.0)
    with pytest.raises(QProjAdapterError, match="no longer matches its identity"):
        adapter(inputs[0])
    assert adapter.last_guard_reason == SAME_Q8_REFERENCE_REASON
    assert adapter.counters.fallback_error == 1
    assert native_calls == []
    adapter.close()


def test_mode_transition_waits_for_old_calls_and_blocks_new_admission(
    registry: bridge.BindingRegistry,
) -> None:
    native_entered = threading.Event()
    release_native = threading.Event()
    native_calls: list[Any] = []

    def native(x: Any, _binding_id: int, _n: int, _k: int) -> Any:
        native_calls.append(x)
        native_entered.set()
        assert release_native.wait(timeout=5)
        return torch.nn.functional.linear(x, _weight())

    adapter, _binding, weight = _adapter(registry, native=native)
    decode = torch.ones((1, 1, 8), dtype=torch.float32)
    results: list[Any] = []
    first = threading.Thread(target=lambda: results.append(adapter(decode)))
    first.start()
    assert native_entered.wait(timeout=5)

    previous_modes: list[QProjExecutionMode] = []
    transition = threading.Thread(
        target=lambda: previous_modes.append(
            adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
        )
    )
    transition.start()
    for _ in range(5_000):
        with adapter._lifecycle:
            if adapter._transitioning:
                break
            adapter._lifecycle.wait(timeout=0.001)
    else:
        pytest.fail("mode transition did not begin")

    second = threading.Thread(target=lambda: results.append(adapter(decode)))
    second.start()
    second.join(timeout=0.05)
    assert second.is_alive()
    assert len(native_calls) == 1

    release_native.set()
    first.join(timeout=5)
    transition.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not transition.is_alive() and not second.is_alive()
    assert previous_modes == [QProjExecutionMode.HYBRID_NATIVE]
    assert adapter.execution_mode is QProjExecutionMode.SAME_Q8_REFERENCE
    assert len(native_calls) == 1
    assert len(results) == 2
    assert all(
        torch.equal(result, torch.nn.functional.linear(decode, weight))
        for result in results
    )
    counters = adapter.counters
    assert counters.forward == 2
    assert counters.native_success == 1
    assert counters.fallback_success == 1
    assert adapter.last_guard_reason == SAME_Q8_REFERENCE_REASON
    adapter.close()


def test_mode_transition_from_forward_is_rejected_without_deadlock(
    registry: bridge.BindingRegistry,
) -> None:
    adapter: QProjAdapter
    transition_errors: list[QProjAdapterError] = []

    def native(x: Any, _binding_id: int, _n: int, _k: int) -> Any:
        try:
            adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
        except QProjAdapterError as error:
            transition_errors.append(error)
        return torch.nn.functional.linear(x, _weight())

    adapter, _binding, _weight_source = _adapter(registry, native=native)
    result = adapter(torch.ones((1, 1, 8), dtype=torch.float32))
    assert result.shape == (1, 1, 4)
    assert len(transition_errors) == 1
    assert "admitted forward" in str(transition_errors[0])
    assert adapter.execution_mode is QProjExecutionMode.HYBRID_NATIVE
    adapter.close()


def test_mode_transition_rejects_closing_closed_and_failed_close(
    registry: bridge.BindingRegistry,
) -> None:
    close_entered = threading.Event()
    release_close = threading.Event()

    class BlockingCloseBinding(FakeBinding):
        def close(self) -> None:
            close_entered.set()
            assert release_close.wait(timeout=5)
            super().close()

    closing_adapter, _binding, _weight_source = _adapter(
        registry, binding=BlockingCloseBinding()
    )
    closer = threading.Thread(target=closing_adapter.close)
    closer.start()
    assert close_entered.wait(timeout=5)
    with pytest.raises(QProjAdapterError, match="execution mode"):
        closing_adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
    release_close.set()
    closer.join(timeout=5)
    assert not closer.is_alive() and closing_adapter.closed
    with pytest.raises(QProjAdapterError, match="execution mode"):
        closing_adapter.set_execution_mode(QProjExecutionMode.HYBRID_NATIVE)

    failed_adapter, _binding, _weight_source = _adapter(
        registry, binding=FakeBinding(close_failures=1)
    )
    with pytest.raises(RuntimeError, match="injected close failure"):
        failed_adapter.close()
    with pytest.raises(QProjAdapterError, match="execution mode"):
        failed_adapter.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
    failed_adapter.close()

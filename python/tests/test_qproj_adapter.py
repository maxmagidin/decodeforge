"""Focused one-layer checks for the owning query-projection adapter."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from decodeforge import torch_bridge as bridge  # noqa: E402
from decodeforge.qproj_adapter import (  # noqa: E402
    QProjAdapter,
    QProjAdapterError,
    QProjCounters,
    fallback_weight_identity,
)

MODULE_ID = "sha256:" + "a" * 64
PACK_ID = "sha256:" + "b" * 64


class FakeBinding:
    def __init__(self, n: int = 4, k: int = 8, *, pack_id: str = PACK_ID) -> None:
        self.descriptor = bridge.RuntimeDescriptor(
            n=n,
            k=k,
            packed_weight_bytes=((n + 3) // 4) * ((k + 31) // 32) * 144,
            module_id=MODULE_ID,
            packed_weight_id=pack_id,
        )
        self.closed = False
        self.close_calls = 0

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
    assert adapter.counters == QProjCounters(2, 1, 1, 0, 1, 1, 0, 0, 0, False)
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
    assert adapter.counters == QProjCounters(1, 1, 0, 1, 0, 0, 0, 0, 0, False)
    adapter.close()


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
    adapter.close()

"""Contract tests for the optional eager PyTorch bridge.

Most tests use a tiny tensor/runtime double.  This keeps pointer identity and
failure accounting observable without pretending that a fake native module
proves numerical correctness.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest
from decodeforge import torch_bridge as bridge


class FakeDevice:
    def __init__(self, device_type: str = "cpu") -> None:
        self.type = device_type


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: object,
        layout: object,
        pointer: int = 0x1000,
        requires_grad: bool = False,
        conjugate: bool = False,
        negative: bool = False,
        contiguous: bool = True,
        device_type: str = "cpu",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self._pointer = pointer
        self.requires_grad = requires_grad
        self._conjugate = conjugate
        self._negative = negative
        self._contiguous = contiguous
        self.device = FakeDevice(device_type)
        self.calls: list[str] = []

    def is_conj(self) -> bool:
        self.calls.append("is_conj")
        return self._conjugate

    def is_neg(self) -> bool:
        self.calls.append("is_neg")
        return self._negative

    def is_contiguous(self) -> bool:
        self.calls.append("is_contiguous")
        return self._contiguous

    def numel(self) -> int:
        self.calls.append("numel")
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def data_ptr(self) -> int:
        self.calls.append("data_ptr")
        return self._pointer


class FakeTorch:
    float32 = object()
    strided = object()

    def __init__(self) -> None:
        self.outputs: list[FakeTensor] = []

    def empty(
        self, shape: tuple[int, ...], *, dtype: object, device: str
    ) -> FakeTensor:
        assert dtype is self.float32
        assert device == "cpu"
        output = FakeTensor(
            shape,
            dtype=self.float32,
            layout=self.strided,
            pointer=0x9000 + len(self.outputs) * 0x100,
        )
        self.outputs.append(output)
        return output


class FakeBinding:
    def __init__(self, n: int = 4, k: int = 8) -> None:
        self.descriptor = bridge.RuntimeDescriptor(
            n=n,
            k=k,
            packed_weight_bytes=((n + 3) // 4) * ((k + 31) // 32) * 144,
            module_id="sha256:" + "a" * 64,
            packed_weight_id="sha256:" + "b" * 64,
        )
        self.closed = False
        self.calls: list[tuple[int, int, int, int]] = []

    def run(
        self,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        self.calls.append((input_address, input_length, output_address, output_length))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeTorch, FakeBinding, int]:
    torch = FakeTorch()
    binding = FakeBinding()
    registry = bridge.BindingRegistry()
    binding_id = registry.register(binding)
    monkeypatch.setattr(bridge, "_BINDINGS", registry)
    monkeypatch.setattr(bridge, "_torch_module", lambda: torch)
    return torch, binding, binding_id


def _valid_tensor(torch: FakeTorch, k: int = 8) -> FakeTensor:
    return FakeTensor((1, k), dtype=torch.float32, layout=torch.strided)


def test_import_boundaries_do_not_import_torch() -> None:
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import decodeforge; "
                "assert decodeforge.__version__ == '0.1.0'; "
                "assert 'torch' not in sys.modules; "
                "import decodeforge.torch_bridge; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_native_pointer_identity_and_single_output_shape(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, binding, binding_id = fake_environment
    x = _valid_tensor(torch)
    output = bridge._native_q8_linear(x, binding_id, 4, 8, torch_module=torch)
    assert output.shape == (1, 4)
    assert len(torch.outputs) == 1
    assert binding.calls == [(0x1000, 8, 0x9000, 4)]
    assert x.calls.count("data_ptr") == 1
    assert "clone" not in x.calls
    assert "contiguous" not in x.calls
    assert "numpy" not in x.calls
    assert "cpu" not in x.calls
    assert "float" not in x.calls
    assert "to" not in x.calls


def test_real_torch_registration_preserves_decode_shape_and_prefill_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    binding = FakeBinding(n=2048, k=2048)
    registry = bridge.BindingRegistry()
    binding_id = registry.register(binding)
    monkeypatch.setattr(bridge, "_BINDINGS", registry)
    monkeypatch.setattr(bridge, "_TORCH_LIBRARIES", None)

    bridge.register_torch_ops()
    decode = torch.zeros((1, 1, 2048), dtype=torch.float32, device="cpu")
    output = torch.ops.decodeforge.q8_linear_v1(decode, binding_id, 2048, 2048)
    assert tuple(output.shape) == (1, 1, 2048)
    assert output.is_contiguous()
    assert binding.calls == [(decode.data_ptr(), 2048, output.data_ptr(), 2048)]

    prefill = torch.zeros((1, 23, 2048), dtype=torch.float32, device="cpu")
    dispatcher = bridge.NativeQ8Linear(
        binding_id,
        2048,
        2048,
        lambda original: original,
        native_operator=lambda *_args: pytest.fail("prefill reached native"),
    )
    assert dispatcher(prefill) is prefill
    assert dispatcher.last_guard_reason == "m_gt_one"
    assert dispatcher.counters == bridge.DispatchCounters(1, 0, 0, 0, 1)


def test_native_preserves_all_singleton_leading_dimensions(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    x = FakeTensor((1, 1, 8), dtype=torch.float32, layout=torch.strided)
    output = bridge._native_q8_linear(x, binding_id, 4, 8, torch_module=torch)
    assert output.shape == (1, 1, 4)


def test_direct_native_operator_rechecks_guards_and_never_falls_back(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    bad = FakeTensor((2, 8), dtype=torch.float32, layout=torch.strided)
    with pytest.raises(bridge.TorchBridgeError, match="m_gt_one"):
        bridge._native_q8_linear(bad, binding_id, 4, 8, torch_module=torch)
    with pytest.raises(bridge.TorchBridgeError, match="dimensions"):
        bridge._native_q8_linear(
            _valid_tensor(torch),
            binding_id,
            True,
            8,
            torch_module=torch,
        )


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("requires_grad", True, "requires_grad"),
        ("_conjugate", True, "conjugate"),
        ("_negative", True, "negative"),
        ("_contiguous", False, "non_contiguous"),
    ],
)
def test_guard_misses_fallback_on_original_tensor(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
    attribute: str,
    value: bool,
    reason: str,
) -> None:
    torch, _binding, binding_id = fake_environment
    x = _valid_tensor(torch)
    setattr(x, attribute, value)
    seen: list[object] = []

    def fallback(original: object) -> str:
        seen.append(original)
        return "fallback"

    callable_ = bridge.NativeQ8Linear(
        binding_id,
        4,
        8,
        fallback,
        native_operator=lambda *_args: pytest.fail("guard miss reached native"),
    )
    assert callable_(x) == "fallback"
    assert seen == [x]
    assert callable_.last_guard_reason == reason
    assert callable_.counters == bridge.DispatchCounters(1, 0, 0, 0, 1)


def test_guard_reasons_cover_m_gt_one_device_dtype_layout_shape_and_numel(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    cases = [
        (FakeTensor((2, 8), dtype=torch.float32, layout=torch.strided), "m_gt_one"),
        (
            FakeTensor(
                (1, 8),
                dtype=torch.float32,
                layout=torch.strided,
                device_type="mps",
            ),
            "device",
        ),
        (FakeTensor((1, 8), dtype=object(), layout=torch.strided), "dtype"),
        (FakeTensor((1, 8), dtype=torch.float32, layout=object()), "layout"),
        (FakeTensor((1, 7), dtype=torch.float32, layout=torch.strided), "shape"),
        (FakeTensor((1, 8), dtype=torch.float32, layout=torch.strided), "numel"),
    ]
    cases[-1][0].numel = lambda: 7  # type: ignore[method-assign]
    callable_ = bridge.NativeQ8Linear(
        binding_id, 4, 8, lambda original: original, native_operator=lambda *_: None
    )
    for x, expected in cases:
        assert callable_.guard_reason(x) == expected


def test_native_errors_are_hard_and_counters_remain_partitioned(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    x = _valid_tensor(torch)
    fallback_calls: list[object] = []

    def failing_native(*_args: object) -> object:
        raise bridge.TorchBridgeError(bridge.BridgeStatus.NONFINITE_OUTPUT, "bad")

    def fallback(original: object) -> str:
        fallback_calls.append(original)
        return "fallback"

    callable_ = bridge.NativeQ8Linear(
        binding_id,
        4,
        8,
        fallback,
        native_operator=failing_native,
    )
    with pytest.raises(bridge.TorchBridgeError):
        callable_(x)
    assert fallback_calls == []
    assert callable_.counters == bridge.DispatchCounters(1, 1, 0, 1, 0)

    bad = FakeTensor((2, 8), dtype=torch.float32, layout=torch.strided)
    assert callable_(bad) == "fallback"
    counters = callable_.counters
    assert counters == bridge.DispatchCounters(2, 1, 0, 1, 1)
    counters.validate()


def test_counter_snapshots_remain_valid_during_a_native_call(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    entered = threading.Event()
    release = threading.Event()

    def blocking_native(*_args: object) -> str:
        entered.set()
        assert release.wait(timeout=5)
        return "native"

    callable_ = bridge.NativeQ8Linear(
        binding_id,
        4,
        8,
        lambda _original: pytest.fail("valid input reached fallback"),
        native_operator=blocking_native,
    )
    results: list[str] = []
    worker = threading.Thread(
        target=lambda: results.append(callable_(_valid_tensor(torch)))
    )
    worker.start()
    assert entered.wait(timeout=5)
    assert callable_.counters == bridge.DispatchCounters(0, 0, 0, 0, 0)
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results == ["native"]
    assert callable_.counters == bridge.DispatchCounters(1, 1, 1, 0, 0)


def test_counter_snapshots_remain_valid_during_a_fallback_call(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    entered = threading.Event()
    release = threading.Event()

    def blocking_fallback(_original: object) -> str:
        entered.set()
        assert release.wait(timeout=5)
        return "fallback"

    callable_ = bridge.NativeQ8Linear(
        binding_id,
        4,
        8,
        blocking_fallback,
        native_operator=lambda *_args: pytest.fail("guard miss reached native"),
    )
    results: list[str] = []
    bad = FakeTensor((2, 8), dtype=torch.float32, layout=torch.strided)
    worker = threading.Thread(target=lambda: results.append(callable_(bad)))
    worker.start()
    assert entered.wait(timeout=5)
    assert callable_.counters == bridge.DispatchCounters(0, 0, 0, 0, 0)
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results == ["fallback"]
    assert callable_.counters == bridge.DispatchCounters(1, 0, 0, 0, 1)


def test_raising_fallback_is_counted_only_after_it_finishes(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, _binding, binding_id = fake_environment
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def raising_fallback(_original: object) -> None:
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("fallback failed")

    callable_ = bridge.NativeQ8Linear(
        binding_id,
        4,
        8,
        raising_fallback,
        native_operator=lambda *_args: pytest.fail("guard miss reached native"),
    )
    bad = FakeTensor((2, 8), dtype=torch.float32, layout=torch.strided)

    def invoke() -> None:
        try:
            callable_(bad)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=invoke)
    worker.start()
    assert entered.wait(timeout=5)
    assert callable_.counters == bridge.DispatchCounters(0, 0, 0, 0, 0)
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert callable_.counters == bridge.DispatchCounters(1, 0, 0, 0, 1)


def test_binding_forgery_and_closed_lifetime_are_guard_misses(
    fake_environment: tuple[FakeTorch, FakeBinding, int],
) -> None:
    torch, binding, binding_id = fake_environment
    x = _valid_tensor(torch)
    seen: list[object] = []

    def fallback(original: object) -> str:
        seen.append(original)
        return "fallback"

    callable_ = bridge.NativeQ8Linear(
        binding_id + 100,
        4,
        8,
        fallback,
        native_operator=lambda *_: pytest.fail("forged ID reached native"),
    )
    assert callable_(x) == "fallback"
    bridge.close_binding(binding_id)
    assert binding.closed
    assert callable_(x) == "fallback"
    assert seen == [x, x]
    assert callable_.last_guard_reason == "binding_unavailable"


def test_registry_rejects_boolean_binding_ids() -> None:
    registry = bridge.BindingRegistry()
    with pytest.raises(bridge.TorchBridgeError, match="not registered"):
        registry.close(True)


def test_registry_ids_are_nonzero_synchronized_and_not_reused() -> None:
    registry = bridge.BindingRegistry()
    first = FakeBinding()
    second = FakeBinding()
    first_id = registry.register(first)
    second_id = registry.register(second)
    assert first_id > 0 and second_id > first_id
    registry.close(first_id)
    assert first.closed
    third_id = registry.register(FakeBinding())
    assert third_id > second_id
    with pytest.raises(bridge.TorchBridgeError):
        registry.close(first_id)


class FakeFunction:
    def __init__(self, function: Any) -> None:
        self._function = function
        self.argtypes: list[Any] | None = None
        self.restype: Any = None

    def __call__(self, *arguments: Any) -> Any:
        return self._function(*arguments)


class FakeCDLL:
    def __init__(self) -> None:
        self.error = b"fake bridge error"
        self.destroyed: list[int] = []
        self.loaded_paths: list[Path] = []
        self.runs: list[tuple[int, int, int, int, int]] = []
        self._next_handle = 41
        self.df_runtime_bridge_abi_version_v1 = FakeFunction(lambda: 1)
        self.df_runtime_create_neon_v1 = FakeFunction(self._create)
        self.df_runtime_get_descriptor_v1 = FakeFunction(self._descriptor)
        self.df_runtime_run_v1 = FakeFunction(self._run)
        self.df_runtime_destroy_v1 = FakeFunction(self._destroy)
        self.df_runtime_last_error_v1 = FakeFunction(self._last_error)

    def _create(self, *_arguments: Any) -> int:
        output = ctypes.cast(_arguments[-1], bridge._HANDLE_POINTER).contents
        output.value = self._next_handle
        self._next_handle += 1
        return int(bridge.BridgeStatus.OK)

    def _descriptor(self, _handle: Any, pointer: Any) -> int:
        descriptor = ctypes.cast(pointer, bridge._DESCRIPTOR_POINTER).contents
        descriptor.abi_version = 1
        descriptor.struct_size = ctypes.sizeof(bridge._CDescriptor)
        descriptor.n = 4
        descriptor.k = 8
        descriptor.packed_weight_bytes = 144
        descriptor.module_id = b"sha256:" + b"a" * 64
        descriptor.packed_weight_id = b"sha256:" + b"b" * 64
        return int(bridge.BridgeStatus.OK)

    def _run(
        self,
        handle: Any,
        input_pointer: Any,
        input_length: Any,
        output_pointer: Any,
        output_length: Any,
    ) -> int:
        self.runs.append(
            (
                int(handle.value),
                int(ctypes.cast(input_pointer, ctypes.c_void_p).value or 0),
                int(input_length),
                int(ctypes.cast(output_pointer, ctypes.c_void_p).value or 0),
                int(output_length),
            )
        )
        return int(bridge.BridgeStatus.OK)

    def _destroy(self, handle: Any) -> int:
        self.destroyed.append(int(handle.value))
        return int(bridge.BridgeStatus.OK)

    def _last_error(self, buffer: Any, length: int, required: Any) -> int:
        message = self.error + b"\0"
        ctypes.cast(required, bridge._SIZE_POINTER).contents.value = len(message)
        if buffer is not None and length:
            ctypes.memmove(buffer, message, min(len(message), int(length)))
        return int(bridge.BridgeStatus.OK)


def _fake_library(tmp_path: Path) -> tuple[bridge.RuntimeLibrary, FakeCDLL]:
    path = tmp_path / "decodeforge_bridge.dylib"
    data = b"frozen bridge bytes"
    path.write_bytes(data)
    fake = FakeCDLL()

    def load(candidate: str) -> FakeCDLL:
        fake.loaded_paths.append(Path(candidate))
        return fake

    library = bridge.RuntimeLibrary(
        path,
        "sha256:" + hashlib.sha256(data).hexdigest(),
        cdll_loader=load,
    )
    return library, fake


def test_runtime_library_verifies_path_hash_and_exact_ctypes_signatures(
    tmp_path: Path,
) -> None:
    library, fake = _fake_library(tmp_path)
    assert fake.loaded_paths == [library._snapshot.path]
    assert fake.loaded_paths[0].read_bytes() == b"frozen bridge bytes"
    assert fake.loaded_paths[0].stat().st_mode & 0o777 == 0o400
    assert library._create.argtypes == [
        bridge._U8_POINTER,
        ctypes.c_size_t,
        bridge._U8_POINTER,
        ctypes.c_size_t,
        bridge._HANDLE_POINTER,
    ]
    assert library._run.argtypes == [
        ctypes.c_uint64,
        bridge._F32_POINTER,
        ctypes.c_size_t,
        bridge._F32_POINTER,
        ctypes.c_size_t,
    ]
    binding = library.create_binding(b"{}", b"payload")
    assert binding.handle == 41
    assert binding.descriptor.n == 4
    binding.run(0x1000, 8, 0x2000, 4)
    assert fake.runs == [(41, 0x1000, 8, 0x2000, 4)]
    binding.close()
    assert fake.destroyed == [41]


def test_runtime_library_loads_the_verified_private_snapshot(
    tmp_path: Path,
) -> None:
    original = tmp_path / "bridge.dylib"
    verified = b"verified bridge image"
    replacement = b"replaced after hash"
    original.write_bytes(verified)
    fake = FakeCDLL()
    loaded: dict[str, object] = {}

    def replace_then_load(candidate: str) -> FakeCDLL:
        original.write_bytes(replacement)
        loaded["path"] = Path(candidate)
        loaded["image"] = Path(candidate).read_bytes()
        return fake

    library = bridge.RuntimeLibrary(
        original,
        "sha256:" + hashlib.sha256(verified).hexdigest(),
        cdll_loader=replace_then_load,
    )
    assert library.image == verified
    assert loaded["image"] == verified
    assert loaded["path"] == library._snapshot.path
    assert loaded["path"] != original
    assert original.read_bytes() == replacement


def test_runtime_library_rejects_hash_mismatch_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "bridge.dylib"
    path.write_bytes(b"bridge")
    with pytest.raises(bridge.TorchBridgeError, match="SHA-256"):
        bridge.RuntimeLibrary(path, "sha256:" + "0" * 64)
    link = tmp_path / "link.dylib"
    link.symlink_to(path)
    with pytest.raises(bridge.TorchBridgeError, match="regular"):
        bridge.RuntimeLibrary(link, "sha256:" + hashlib.sha256(b"bridge").hexdigest())


def test_runtime_library_maps_status_and_last_error(tmp_path: Path) -> None:
    library, fake = _fake_library(tmp_path)

    def reject(*_arguments: Any) -> int:
        fake.error = b"native status failure"
        return int(bridge.BridgeStatus.NONFINITE_INPUT)

    fake.df_runtime_run_v1 = FakeFunction(reject)
    library._run = fake.df_runtime_run_v1
    with pytest.raises(bridge.TorchBridgeError) as error:
        library.run(41, 0x1000, 8, 0x2000, 4)
    assert error.value.status is bridge.BridgeStatus.NONFINITE_INPUT
    assert "native status failure" in error.value.detail


def test_runtime_library_rejects_overlapping_borrowed_ranges(
    tmp_path: Path,
) -> None:
    library, fake = _fake_library(tmp_path)
    with pytest.raises(bridge.TorchBridgeError) as error:
        library.run(41, 0x1000, 8, 0x1010, 4)
    assert error.value.status is bridge.BridgeStatus.OVERLAP
    assert fake.runs == []


def test_torch_registration_is_idempotent_without_framework_import_at_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLibrary:
        created: ClassVar[list[tuple[str, str, str | None]]] = []

        def __init__(self, namespace: str, kind: str, dispatch: str | None = None):
            self.created.append((namespace, kind, dispatch))

        def define(self, _schema: str) -> None:
            return None

        def impl(self, _name: str, _function: Any) -> None:
            return None

    class FakeTorchForRegistration:
        library = type("LibraryNamespace", (), {"Library": FakeLibrary})

    monkeypatch.setattr(bridge, "_torch_module", lambda: FakeTorchForRegistration)
    monkeypatch.setattr(bridge, "_TORCH_LIBRARIES", None)
    bridge.register_torch_ops()
    first_count = len(FakeLibrary.created)
    bridge.register_torch_ops()
    assert first_count == 2
    assert len(FakeLibrary.created) == first_count

"""Bounded transactional checks for all-layer TinyLlama q_proj installation."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
from decodeforge import qproj_model as model_bridge  # noqa: E402
from decodeforge.qproj_adapter import (  # noqa: E402
    QProjCounters,
    QProjExecutionMode,
    QProjMetadata,
)
from torch import nn  # noqa: E402


def _identity(value: int) -> str:
    return f"sha256:{value:064x}"


class Attention(nn.Module):
    def __init__(self, projection: nn.Module) -> None:
        super().__init__()
        self.q_proj = projection
        self.fail_install = False

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name == "q_proj"
            and getattr(self, "fail_install", False)
            and isinstance(value, model_bridge._InstalledQProj)
        ):
            raise RuntimeError("injected module installation failure")
        super().__setattr__(name, value)


class Layer(nn.Module):
    def __init__(self, projection: nn.Module) -> None:
        super().__init__()
        self.self_attn = Attention(projection)


class Backbone(nn.Module):
    def __init__(self, projections: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer(value) for value in projections)


class TinyModel(nn.Module):
    def __init__(self, projections: list[nn.Module]) -> None:
        super().__init__()
        self.model = Backbone(projections)
        self.eval()


def _linear() -> nn.Linear:
    result = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(result)
    result.in_features = 2048
    result.out_features = 2048
    shared_storage = torch.zeros(1, dtype=torch.float32).expand(2048, 2048)
    result.weight = nn.Parameter(shared_storage, requires_grad=False)
    result.register_parameter("bias", None)
    result.eval()
    return result


def _model() -> TinyModel:
    return TinyModel([_linear() for _ in range(22)])


def _entry(layer: int) -> model_bridge.QProjAssetEntry:
    path = f"model.layers.{layer}.self_attn.q_proj"
    logical = _identity(100 + layer)
    packed = _identity(200 + layer)
    return model_bridge.QProjAssetEntry(
        layer=layer,
        directory=f"layers/{layer:02}",
        layer_path=path,
        manifest_identity=_identity(300 + layer),
        tensor_name=f"{path}.weight",
        tensor_identity=_identity(400 + layer),
        logical_weight_identity=logical,
        packed_weight_identity=packed,
        packed_bytes=model_bridge.QPROJ_PACKED_BYTES,
        module_identity=model_bridge.TINYLLAMA_QPROJ_MODULE_ID,
        fallback_weight_identity=_identity(600 + layer),
        fallback_parent_logical_weight_identity=logical,
        fallback_parent_packed_weight_identity=packed,
    )


def _loaded() -> tuple[
    model_bridge.QProjAssetInventory,
    tuple[model_bridge.VerifiedQProjAsset, ...],
]:
    entries = tuple(_entry(layer) for layer in range(22))
    inventory = model_bridge.QProjAssetInventory(
        source=model_bridge.QProjSource(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            filename="model.safetensors",
            bytes=2_200_119_864,
            identity="sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
        ),
        entries=entries,
        total_packed_bytes=model_bridge.QPROJ_TOTAL_PACKED_BYTES,
        total_fallback_bytes=22 * 2048 * 2048 * 4,
        aggregate_identity=_identity(0),
    )
    inventory = dataclasses.replace(
        inventory,
        aggregate_identity=model_bridge._inventory_identity(inventory),
    )
    assets = tuple(
        model_bridge.VerifiedQProjAsset(
            entry,
            b"verified pack manifest",
            b"verified payload",
            torch.zeros((1, 1), dtype=torch.float32),
        )
        for entry in entries
    )
    return inventory, assets


class FakeAdapter(nn.Module):
    def __init__(self, asset: model_bridge.VerifiedQProjAsset) -> None:
        super().__init__()
        entry = asset.entry
        self._metadata = QProjMetadata(
            layer_name=entry.layer_path,
            n=2048,
            k=2048,
            module_id=entry.module_identity,
            packed_weight_id=entry.packed_weight_identity,
            packed_weight_bytes=entry.packed_bytes,
            fallback_weight_id=entry.fallback_weight_identity,
        )
        self._closed = False
        self.close_calls = 0
        self.close_failures = 0
        self.mode_failure: QProjExecutionMode | None = None
        self._execution_mode = QProjExecutionMode.HYBRID_NATIVE
        self.eval()

    @property
    def metadata(self) -> QProjMetadata:
        return self._metadata

    @property
    def counters(self) -> QProjCounters:
        return QProjCounters(
            forward=0,
            native_attempt=0,
            native_success=0,
            native_error=0,
            fallback_attempt=0,
            fallback_success=0,
            fallback_error=0,
            predispatch_error=0,
            rejected_closed=0,
            in_flight=0,
            closed=self._closed,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def execution_mode(self) -> QProjExecutionMode:
        return self._execution_mode

    def set_execution_mode(self, mode: QProjExecutionMode) -> QProjExecutionMode:
        if mode is self.mode_failure:
            raise RuntimeError("injected execution-mode failure")
        previous = self._execution_mode
        self._execution_mode = mode
        return previous

    def forward(self, value: Any) -> Any:
        return value

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls <= self.close_failures:
            raise RuntimeError(f"injected close failure {self.metadata.layer_name}")
        self._closed = True


def _patch_loader(
    monkeypatch: pytest.MonkeyPatch,
    loaded: tuple[
        model_bridge.QProjAssetInventory,
        tuple[model_bridge.VerifiedQProjAsset, ...],
    ],
) -> None:
    monkeypatch.setattr(model_bridge, "_load_prepared_inventory", lambda _path: loaded)


def _collecting_factory(
    sink: list[FakeAdapter],
) -> model_bridge.AdapterFactory:
    def factory(asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        adapter = FakeAdapter(asset)
        sink.append(adapter)
        return adapter

    return factory


def test_exact_order_install_observability_guards_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    _patch_loader(monkeypatch, loaded)
    model = _model()
    originals = tuple(
        model.get_submodule(path) for path in model_bridge.tinyllama_qproj_paths()
    )
    original_state_keys = tuple(model.state_dict())
    order: list[str] = []
    adapters: list[FakeAdapter] = []

    def factory(asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        order.append(asset.entry.layer_path)
        adapter = FakeAdapter(asset)
        adapters.append(adapter)
        return adapter

    installation = model_bridge.install_tinyllama_qproj(
        model, Path("verified"), factory
    )

    assert tuple(order) == model_bridge.tinyllama_qproj_paths()
    assert installation.inventory is loaded[0]
    assert installation.counters.installed_modules == 22
    assert installation.counters.live_adapters == 22
    assert not installation.closed
    assert tuple(model.state_dict()) == tuple(
        key
        for key in original_state_keys
        if not key.endswith("self_attn.q_proj.weight")
    )
    model.eval()
    with pytest.raises(model_bridge.QProjModelError, match="training"):
        model.train()
    with pytest.raises(model_bridge.QProjModelError, match="migration"):
        model.to(dtype=torch.float64)
    with pytest.raises(model_bridge.QProjModelError, match="state loading"):
        model.load_state_dict({})

    installation.close()
    installation.close()
    assert installation.closed
    assert installation.counters.installed_modules == 0
    assert installation.counters.restored_modules == 22
    assert installation.counters.live_adapters == 0
    assert tuple(model.state_dict()) == original_state_keys
    assert all(
        model.get_submodule(path) is original
        for path, original in zip(
            model_bridge.tinyllama_qproj_paths(), originals, strict=True
        )
    )
    assert all(adapter.close_calls == 1 for adapter in adapters)
    model.train()
    assert model.training


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong"])
def test_model_preflight_rejects_missing_duplicate_and_wrong_module_without_creation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    loaded = _loaded()
    _patch_loader(monkeypatch, loaded)
    model = _model()
    if failure == "missing":
        del cast(Layer, model.model.layers[7]).self_attn.q_proj
    elif failure == "duplicate":
        seventh = cast(Layer, model.model.layers[7]).self_attn
        sixth = cast(Layer, model.model.layers[6]).self_attn
        seventh.q_proj = sixth.q_proj
    else:
        cast(Layer, model.model.layers[7]).self_attn.q_proj = nn.Identity()
        model.eval()
    calls = 0

    def factory(_asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        nonlocal calls
        calls += 1
        raise AssertionError("factory must not run")

    with pytest.raises(model_bridge.QProjModelError):
        model_bridge.install_tinyllama_qproj(model, "verified", factory)
    assert calls == 0
    assert not any(
        isinstance(module, model_bridge._InstalledQProj) for module in model.modules()
    )


@pytest.mark.parametrize("failure", ["parent", "module", "duplicate", "order"])
def test_asset_mismatch_fails_before_model_mutation_or_handle_creation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    inventory, assets = _loaded()
    entries = list(inventory.entries)
    if failure == "parent":
        entries[8] = dataclasses.replace(
            entries[8], fallback_parent_packed_weight_identity=_identity(999)
        )
    elif failure == "module":
        entries[8] = dataclasses.replace(entries[8], module_identity=_identity(999))
    elif failure == "duplicate":
        entries[8] = dataclasses.replace(
            entries[8], packed_weight_identity=entries[7].packed_weight_identity
        )
    else:
        entries[8] = dataclasses.replace(entries[8], layer_path="wrong.path")
    broken = dataclasses.replace(inventory, entries=tuple(entries))
    _patch_loader(monkeypatch, (broken, assets))
    model = _model()
    originals = tuple(
        model.get_submodule(path) for path in model_bridge.tinyllama_qproj_paths()
    )

    with pytest.raises(model_bridge.QProjModelError):
        model_bridge.install_tinyllama_qproj(
            model, "verified", lambda _asset: pytest.fail("factory must not run")
        )
    assert all(
        model.get_submodule(path) is original
        for path, original in zip(
            model_bridge.tinyllama_qproj_paths(), originals, strict=True
        )
    )


@pytest.mark.parametrize("failure", ["bias", "trainable", "shape", "nonfinite"])
def test_model_linear_contract_is_checked_before_creation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    projection = cast(Layer, model.model.layers[4]).self_attn.q_proj
    assert isinstance(projection, nn.Linear)
    if failure == "bias":
        projection.bias = nn.Parameter(torch.zeros(1), requires_grad=False)
    elif failure == "trainable":
        projection.weight.requires_grad_(True)
    elif failure == "nonfinite":
        with torch.no_grad():
            projection.weight.view(-1)[0] = float("nan")
    else:
        projection.in_features = 1024

    with pytest.raises(model_bridge.QProjModelError):
        model_bridge.install_tinyllama_qproj(
            model, "verified", lambda _asset: pytest.fail("factory must not run")
        )


def test_model_state_surface_is_checked_before_adapter_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    projection = cast(Layer, model.model.layers[4]).self_attn.q_proj
    assert isinstance(projection, nn.Linear)
    projection.register_buffer("unexpected", torch.zeros(1))
    calls = 0

    def factory(_asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        nonlocal calls
        calls += 1
        raise AssertionError("factory must not run")

    with pytest.raises(model_bridge.QProjModelError, match="state_dict"):
        model_bridge.install_tinyllama_qproj(model, "verified", factory)
    assert calls == 0


def test_partial_adapter_creation_failure_closes_every_created_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    originals = tuple(
        model.get_submodule(path) for path in model_bridge.tinyllama_qproj_paths()
    )
    adapters: list[FakeAdapter] = []

    def factory(asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        if asset.entry.layer == 9:
            raise RuntimeError("injected creation failure")
        adapter = FakeAdapter(asset)
        adapters.append(adapter)
        return adapter

    with pytest.raises(RuntimeError, match="creation failure"):
        model_bridge.install_tinyllama_qproj(model, "verified", factory)
    assert len(adapters) == 9 and all(adapter.closed for adapter in adapters)
    assert all(
        model.get_submodule(path) is original
        for path, original in zip(
            model_bridge.tinyllama_qproj_paths(), originals, strict=True
        )
    )


def test_interrupt_during_creation_still_closes_every_owned_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    originals = tuple(
        model.get_submodule(path) for path in model_bridge.tinyllama_qproj_paths()
    )
    adapters: list[FakeAdapter] = []

    def factory(asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        if asset.entry.layer == 9:
            raise KeyboardInterrupt("injected creation interruption")
        adapter = FakeAdapter(asset)
        adapters.append(adapter)
        return adapter

    with pytest.raises(KeyboardInterrupt, match="creation interruption"):
        model_bridge.install_tinyllama_qproj(model, "verified", factory)
    assert len(adapters) == 9 and all(adapter.closed for adapter in adapters)
    assert all(
        model.get_submodule(path) is original
        for path, original in zip(
            model_bridge.tinyllama_qproj_paths(), originals, strict=True
        )
    )


def test_failed_factory_exception_owned_binding_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    adapters: list[FakeAdapter] = []

    class InitializationFailure(RuntimeError):
        def __init__(self) -> None:
            super().__init__("injected initialization failure")
            self.closed = False
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            self.closed = True

    failure = InitializationFailure()

    def factory(asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        if asset.entry.layer == 4:
            raise failure
        adapter = FakeAdapter(asset)
        adapters.append(adapter)
        return adapter

    with pytest.raises(InitializationFailure):
        model_bridge.install_tinyllama_qproj(model, "verified", factory)
    assert failure.closed and failure.close_calls == 1
    assert len(adapters) == 4 and all(adapter.closed for adapter in adapters)


def test_partial_module_install_and_guard_setup_failures_fully_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    _patch_loader(monkeypatch, loaded)
    for guard_failure in (False, True):
        model = _model()
        originals = tuple(
            model.get_submodule(path) for path in model_bridge.tinyllama_qproj_paths()
        )
        adapters: list[FakeAdapter] = []
        if guard_failure:
            monkeypatch.setattr(
                model_bridge.QProjModelInstallation,
                "_install_model_guards",
                lambda _self: (_ for _ in ()).throw(
                    RuntimeError("injected guard failure")
                ),
            )
        else:
            cast(Layer, model.model.layers[11]).self_attn.fail_install = True

        with pytest.raises(RuntimeError, match="injected"):
            model_bridge.install_tinyllama_qproj(
                model, "verified", _collecting_factory(adapters)
            )
        assert len(adapters) == 22 and all(adapter.closed for adapter in adapters)
        assert all(
            model.get_submodule(path) is original
            for path, original in zip(
                model_bridge.tinyllama_qproj_paths(), originals, strict=True
            )
        )
        monkeypatch.undo()
        _patch_loader(monkeypatch, loaded)


def test_cleanup_attempts_every_adapter_aggregates_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    adapters: list[FakeAdapter] = []

    def factory(asset: model_bridge.VerifiedQProjAsset) -> FakeAdapter:
        adapter = FakeAdapter(asset)
        if asset.entry.layer in {3, 17}:
            adapter.close_failures = 1
        adapters.append(adapter)
        return adapter

    installation = model_bridge.install_tinyllama_qproj(model, "verified", factory)
    with pytest.raises(ExceptionGroup) as captured:
        installation.close()
    assert len(captured.value.exceptions) == 2
    assert all(adapter.close_calls == 1 for adapter in adapters)
    assert installation.counters.restored_modules == 22
    assert installation.counters.live_adapters == 2
    assert not installation.closed

    installation.close()
    assert installation.closed
    assert [adapter.close_calls for adapter in adapters].count(2) == 2


def test_cleanup_does_not_overwrite_external_model_guard_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    adapters: list[FakeAdapter] = []
    installation = model_bridge.install_tinyllama_qproj(
        model, "verified", _collecting_factory(adapters)
    )

    def external_train(_model: nn.Module, _mode: bool = True) -> nn.Module:
        return _model

    replacement = types.MethodType(external_train, model)
    model.train = replacement  # type: ignore[method-assign]
    with pytest.raises(ExceptionGroup, match="cleanup did not complete") as captured:
        installation.close()
    assert any(
        "guard changed outside" in str(error) for error in captured.value.exceptions
    )
    assert model.__dict__["train"] is replacement
    assert all(adapter.closed for adapter in adapters)
    assert not installation.closed
    assert not installation.counters.closed

    model.__dict__.pop("train")
    installation.close()
    assert installation.closed
    model.train()
    assert model.training


def test_execution_mode_switch_is_all_22_or_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    adapters: list[FakeAdapter] = []
    installation = model_bridge.install_tinyllama_qproj(
        model, "verified", _collecting_factory(adapters)
    )
    reference = QProjExecutionMode.SAME_Q8_REFERENCE
    hybrid = QProjExecutionMode.HYBRID_NATIVE

    assert installation.execution_mode is hybrid
    assert installation.set_execution_mode(reference) is hybrid
    assert all(adapter.execution_mode is reference for adapter in adapters)

    adapters[10].mode_failure = hybrid
    with pytest.raises(RuntimeError, match="execution-mode failure"):
        installation.set_execution_mode(hybrid)
    assert installation.execution_mode is reference
    assert all(adapter.execution_mode is reference for adapter in adapters)
    installation.close()


def test_repeat_setup_teardown_leaves_no_live_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, _loaded())
    model = _model()
    for _ in range(2):
        adapters: list[FakeAdapter] = []
        installation = model_bridge.install_tinyllama_qproj(
            model, "verified", _collecting_factory(adapters)
        )
        installation.close()
        assert installation.counters.in_flight == 0
        assert installation.counters.live_adapters == 0
        assert all(adapter.closed for adapter in adapters)


def test_trusted_loader_rejects_non_closed_directory(tmp_path: Path) -> None:
    (tmp_path / "inventory.json").write_text("{}", encoding="utf-8")
    (tmp_path / "layers").mkdir()
    (tmp_path / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(model_bridge.QProjModelError, match="unexpected"):
        model_bridge._load_prepared_inventory(tmp_path)


def test_trusted_loader_remains_anchored_when_root_path_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "prepared"
    replacement = tmp_path / "replacement"
    for root, inventory in ((original, b"{}"), (replacement, b"not-json")):
        root.mkdir()
        (root / "inventory.json").write_bytes(inventory)
        layers = root / "layers"
        layers.mkdir()
        for layer in range(model_bridge.TINYLLAMA_QPROJ_LAYERS):
            (layers / f"{layer:02}").mkdir()

    read_snapshot = model_bridge._read_snapshot
    swapped = False

    def swap_before_read(
        directory_fd: int,
        filename: str,
        maximum: int,
        label: str,
        exact: int | None = None,
    ) -> bytes:
        nonlocal swapped
        if filename == "inventory.json" and not swapped:
            original.rename(tmp_path / "parked")
            replacement.rename(original)
            swapped = True
        return read_snapshot(directory_fd, filename, maximum, label, exact)

    monkeypatch.setattr(model_bridge, "_read_snapshot", swap_before_read)
    with pytest.raises(model_bridge.QProjModelError, match="unsupported or missing"):
        model_bridge._load_prepared_inventory(original)
    assert swapped


def test_only_the_canonical_prepared_aggregate_is_installable() -> None:
    model_bridge._require_pinned_aggregate(model_bridge.TINYLLAMA_QPROJ_AGGREGATE_ID)
    with pytest.raises(model_bridge.QProjModelError, match="canonical pinned"):
        model_bridge._require_pinned_aggregate(_identity(999))

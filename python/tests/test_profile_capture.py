from __future__ import annotations

import copy
import ctypes
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from decodeforge import profile_capture as capture
from decodeforge import torch_bridge as bridge
from decodeforge.decode_profile import (
    DecodeProfile,
    generate_cached_unprofiled,
    profile_cached_generation,
    tinyllama_component_paths,
)
from decodeforge.evaluation import CachedGeneration
from decodeforge.qproj_adapter import (
    QProjAdapter,
    QProjExecutionMode,
    fallback_weight_identity,
)
from decodeforge.qproj_model import (
    QProjLayerCounters,
    QProjModelCounters,
    tinyllama_qproj_paths,
)
from decodeforge.torch_bridge import RuntimeDescriptor, RuntimeLibrary
from torch import nn


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, _messages: Any, **kwargs: Any) -> Any:
        if kwargs.get("tokenize") is False:
            return "<user>profile decode</user><assistant>"
        return torch.tensor([[10, 11, 12]], dtype=torch.int64)

    def decode(self, values: Any, *, skip_special_tokens: bool) -> str:
        prefix = "clean" if skip_special_tokens else "raw"
        return f"{prefix}:{list(values)}"


MODULE_ID = "sha256:" + "b" * 64
PACK_ID = "sha256:" + "c" * 64


class FakeBinding:
    def __init__(self) -> None:
        self.descriptor = RuntimeDescriptor(
            n=1,
            k=1,
            packed_weight_bytes=144,
            module_id=MODULE_ID,
            packed_weight_id=PACK_ID,
        )
        self.closed = False

    def run(
        self,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        assert input_length == output_length == 1
        source = (ctypes.c_float * input_length).from_address(input_address)
        destination = (ctypes.c_float * output_length).from_address(output_address)
        destination[0] = source[0]

    def close(self) -> None:
        self.closed = True


class FakeLibrary:
    def create_binding(self, _manifest: Any, _packed: Any) -> FakeBinding:
        return FakeBinding()


class FakeInstalledAdapter(nn.Module):
    def __init__(self, adapter: QProjAdapter) -> None:
        super().__init__()
        self.adapter = adapter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.adapter(inputs))


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(1, 1, bias=False)
        self.self_attn.k_proj = nn.Identity()
        self.self_attn.v_proj = nn.Identity()
        self.self_attn.o_proj = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Identity()


class FakeModel(nn.Module):
    def __init__(self, *, stop_after_cached: bool = True) -> None:
        super().__init__()
        self.stop_after_cached = stop_after_cached
        self.model = nn.Module()
        self.model.embed_tokens = nn.Identity()
        self.model.layers = nn.ModuleList(FakeLayer() for _ in range(22))
        self.model.norm = nn.Identity()
        self.lm_head = nn.Identity()
        self.eval()

    def forward(self, **kwargs: Any) -> Any:
        values = kwargs["input_ids"].to(torch.float32).unsqueeze(-1)
        values = self.model.get_submodule("embed_tokens")(values)
        layers = cast(nn.ModuleList, self.model.get_submodule("layers"))
        for layer in layers:
            for path in (
                "input_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "post_attention_layernorm",
                "mlp",
            ):
                values = layer.get_submodule(path)(values)
        values = self.model.get_submodule("norm")(values)
        self.lm_head(values)
        logits = torch.full((1, values.shape[1], 128), -2.0)
        token = (
            99
            if self.stop_after_cached and kwargs.get("past_key_values") is not None
            else 1
        )
        logits[0, -1, token] = 2.0
        return SimpleNamespace(logits=logits, past_key_values=object())


class FakeInstallation:
    def __init__(self, model: FakeModel, *, fail_close: bool = False) -> None:
        self.model = model
        self.mode = QProjExecutionMode.HYBRID_NATIVE
        self.originals = tuple(
            model.get_submodule(path) for path in tinyllama_qproj_paths()
        )
        weight = torch.ones((1, 1), dtype=torch.float32)
        fallback_id = fallback_weight_identity(weight)
        self.adapters = tuple(
            QProjAdapter(
                layer_name=path,
                library=FakeLibrary(),  # type: ignore[arg-type]
                pack_manifest_json=b"{}",
                packed_weight=b"packed",
                fallback_weight=weight,
                fallback_weight_id=fallback_id,
                fallback_parent_packed_weight_id=PACK_ID,
                expected_module_id=MODULE_ID,
                native_operator=lambda value,
                binding_id,
                n,
                k: bridge._native_q8_linear(
                    value, binding_id, n, k, torch_module=torch
                ),
            )
            for path in tinyllama_qproj_paths()
        )
        self.inventory = SimpleNamespace(aggregate_identity="sha256:" + "a" * 64)
        self._closed = False
        self.fail_close = fail_close
        for path, adapter in zip(tinyllama_qproj_paths(), self.adapters, strict=True):
            parent_path, _, name = path.rpartition(".")
            setattr(
                model.get_submodule(parent_path), name, FakeInstalledAdapter(adapter)
            )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def execution_mode(self) -> QProjExecutionMode:
        return self.mode

    @property
    def counters(self) -> QProjModelCounters:
        layers = tuple(
            QProjLayerCounters(
                layer=index,
                layer_path=tinyllama_qproj_paths()[index],
                forward=adapter.counters.forward,
                native_attempt=adapter.counters.native_attempt,
                native_success=adapter.counters.native_success,
                native_error=adapter.counters.native_error,
                fallback_attempt=adapter.counters.fallback_attempt,
                fallback_success=adapter.counters.fallback_success,
                fallback_error=adapter.counters.fallback_error,
                predispatch_error=adapter.counters.predispatch_error,
                rejected_closed=adapter.counters.rejected_closed,
                in_flight=adapter.counters.in_flight,
                closed=adapter.counters.closed,
            )
            for index, adapter in enumerate(self.adapters)
        )
        return QProjModelCounters(
            installed_modules=0 if self._closed else 22,
            restored_modules=22 if self._closed else 0,
            live_adapters=0 if self._closed else 22,
            in_flight=0,
            layers=layers,
            closed=self._closed,
        )

    def set_execution_mode(self, mode: QProjExecutionMode) -> QProjExecutionMode:
        previous = self.mode
        for adapter in self.adapters:
            adapter.set_execution_mode(mode)
        self.mode = mode
        return previous

    def close(self) -> None:
        if self._closed:
            return
        if self.fail_close:
            raise RuntimeError("injected close failure")
        for path, original in zip(tinyllama_qproj_paths(), self.originals, strict=True):
            parent_path, _, name = path.rpartition(".")
            setattr(self.model.get_submodule(parent_path), name, original)
        for adapter in self.adapters:
            adapter.close()
        self._closed = True


def _request(tmp_path: Path, session_index: int = 0) -> capture.ProfileCaptureRequest:
    model_dir = tmp_path / "model"
    assets = tmp_path / "assets"
    library = tmp_path / "bridge.dylib"
    model_dir.mkdir()
    assets.mkdir()
    library.write_bytes(b"bridge")
    return capture.ProfileCaptureRequest(
        model_dir,
        assets,
        library,
        "0" * 64,
        "Profile one compiler sentence.",
        session_index,
        4,
    )


def _run(
    tmp_path: Path,
    *,
    session_index: int = 0,
    control_runner: capture.ControlRunner | None = None,
    profile_runner: capture.ProfileRunner | None = None,
    fail_close: bool = False,
) -> tuple[dict[str, Any], FakeModel, FakeInstallation]:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    installation: FakeInstallation | None = None

    def install(
        current: nn.Module, _assets: Path, _runtime: RuntimeLibrary
    ) -> FakeInstallation:
        nonlocal installation
        assert current is model
        installation = FakeInstallation(model, fail_close=fail_close)
        return installation

    kwargs: dict[str, Any] = {}
    if control_runner is not None:
        kwargs["control_runner"] = control_runner
    if profile_runner is not None:
        kwargs["profile_runner"] = profile_runner
    document = capture.run_profile_capture(
        _request(tmp_path, session_index),
        model_loader=lambda _path: model,
        tokenizer_loader=lambda _path: tokenizer,
        runtime_loader=lambda _path, _digest: cast(RuntimeLibrary, object()),
        installer=install,
        model_verifier=lambda _path: {},
        **kwargs,
    )
    assert installation is not None
    return document, model, installation


@pytest.mark.parametrize(
    ("session_index", "mode_order", "measurement_order"),
    [
        (0, ["same_q8_reference", "hybrid_native"], ["control", "profile"]),
        (1, ["hybrid_native", "same_q8_reference"], ["profile", "control"]),
    ],
)
def test_capture_runs_matched_paths_and_restores_model(
    tmp_path: Path,
    session_index: int,
    mode_order: list[str],
    measurement_order: list[str],
) -> None:
    document, model, installation = _run(tmp_path, session_index=session_index)

    assert document["mode_order"] == mode_order
    assert document["performance_claim_allowed"] is False
    assert document["benchmark"] is False
    assert document["comparison"] == {
        "control_profile_exact": True,
        "same_q8_hybrid_exact": True,
        "cached_decode_covered": True,
    }
    assert installation.closed
    assert document["restoration"]["original_modules_restored"] is True
    assert all(
        model.get_submodule(path) is original
        for path, original in zip(
            tinyllama_qproj_paths(), installation.originals, strict=True
        )
    )
    for name in mode_order:
        run = document["runs"][name]
        assert run["measurement_order"] == measurement_order
        assert run["warmup"]["measured"] is False
        assert run["control"]["trace"] is None
        assert run["profile"]["trace"]["claim_class"] == "diagnostic_profile"
        assert (
            run["profile"]["trace"]["boundary_contract"]
            == "decodeforge_adapter_internal_v1"
        )
        assert run["profile"]["trace"]["qproj_details"] is True
        qproj = tinyllama_qproj_paths()[0]
        dispatch = [
            event["dispatch"]
            for event in run["profile"]["trace"]["events"]
            if event["module_path"] == qproj and event["boundary"] == "module_forward"
        ]
        assert dispatch == (
            ["fallback", "native"]
            if name == "hybrid_native"
            else ["fallback", "fallback"]
        )


def test_capture_closes_installation_after_generation_failure(tmp_path: Path) -> None:
    installation: FakeInstallation | None = None
    model = FakeModel()

    def install(
        _model: nn.Module, _assets: Path, _runtime: RuntimeLibrary
    ) -> FakeInstallation:
        nonlocal installation
        installation = FakeInstallation(model)
        return installation

    def fail_control(*_args: Any) -> CachedGeneration:
        raise RuntimeError("injected generation failure")

    request = _request(tmp_path)
    with pytest.raises(capture.ProfileCaptureError, match="capture failed"):
        capture.run_profile_capture(
            request,
            model_loader=lambda _path: model,
            tokenizer_loader=lambda _path: FakeTokenizer(),
            runtime_loader=lambda _path, _digest: cast(RuntimeLibrary, object()),
            installer=install,
            model_verifier=lambda _path: {},
            control_runner=fail_control,
        )
    assert installation is not None and installation.closed


def test_capture_preserves_observations_after_final_cleanup_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(capture.ProfileCaptureError, match="cleanup") as captured:
        _run(tmp_path, fail_close=True)

    error = captured.value
    assert error.stage == "cleanup"
    assert isinstance(error.cleanup_error, RuntimeError)
    assert set(error.observations) == {"same_q8_reference", "hybrid_native"}
    assert set(error.observations["hybrid_native"]) == {
        "warmup",
        "control",
        "profile",
    }


def test_capture_normalizes_model_loading_failures(tmp_path: Path) -> None:
    request = _request(tmp_path)

    def fail_load(_path: Path) -> nn.Module:
        raise RuntimeError("injected load failure")

    with pytest.raises(capture.ProfileCaptureError, match="model_loading") as captured:
        capture.run_profile_capture(
            request,
            model_loader=fail_load,
            model_verifier=lambda _path: {},
        )
    assert captured.value.stage == "model_loading"


def test_capture_rejects_control_profile_mismatch_and_cleans_up(
    tmp_path: Path,
) -> None:
    def mismatch_profile(*args: Any, **kwargs: Any) -> DecodeProfile:
        result = profile_cached_generation(*args, **kwargs)
        altered = replace(
            result.generation,
            generated_ids=[77, 99],
            all_ids=[10, 11, 12, 77, 99],
        )
        return replace(result, generation=altered)

    with pytest.raises(capture.ProfileCaptureError, match="outputs differ"):
        _run(tmp_path, profile_runner=mismatch_profile)


def test_capture_snapshots_mutable_generation_results_immediately(
    tmp_path: Path,
) -> None:
    shared: CachedGeneration | None = None

    def aliased_control(*args: Any) -> CachedGeneration:
        nonlocal shared
        current = generate_cached_unprofiled(*args)
        if shared is None:
            shared = current
        return shared

    def mutating_profile(*args: Any, **kwargs: Any) -> DecodeProfile:
        assert shared is not None
        current = profile_cached_generation(*args, **kwargs)
        shared.generated_ids[:] = [2, 99]
        shared.all_ids[:] = [10, 11, 12, 2, 99]
        return replace(current, generation=shared)

    with pytest.raises(capture.ProfileCaptureError, match="outputs differ"):
        _run(
            tmp_path,
            control_runner=aliased_control,
            profile_runner=mutating_profile,
        )


def test_capture_rejects_incomplete_profile_inventory(tmp_path: Path) -> None:
    def incomplete_profile(*args: Any, **kwargs: Any) -> DecodeProfile:
        current = profile_cached_generation(*args, **kwargs)
        return replace(current, module_paths=())

    with pytest.raises(capture.ProfileCaptureError, match="trace metadata"):
        _run(tmp_path, profile_runner=incomplete_profile)


def test_capture_rejects_component_only_profile_contract(tmp_path: Path) -> None:
    def component_only(*args: Any, **kwargs: Any) -> DecodeProfile:
        current = profile_cached_generation(*args, **kwargs)
        return replace(current, qproj_details=False)

    with pytest.raises(capture.ProfileCaptureError, match="trace metadata"):
        _run(tmp_path, profile_runner=component_only)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "wrong_context", "unexpected", "boolean_step", "invalid_timing"],
)
def test_capture_rejects_malformed_internal_trace(
    tmp_path: Path, mutation: str
) -> None:
    document, _model, _installation = _run(tmp_path)
    record = document["runs"]["same_q8_reference"]["profile"]
    trace = copy.deepcopy(record["trace"])
    generation_wire = record["generation"]
    generation = CachedGeneration(
        list(generation_wire["token_ids"]),
        list(generation_wire["generated_token_ids"]),
        str(generation_wire["stop_reason"]),
        generation_wire["stop_reason"] == "eos",
        [],
        [],
    )
    internal = next(
        event for event in trace["events"] if event["boundary"] == "fallback_hash"
    )
    if mutation == "missing":
        trace["events"].remove(internal)
    elif mutation == "wrong_context":
        internal["module_path"] = "model.layers.21.self_attn.q_proj"
    elif mutation == "unexpected":
        extra = dict(internal)
        extra["event_id"] = max(event["event_id"] for event in trace["events"]) + 1
        trace["events"].append(extra)
    elif mutation == "boolean_step":
        internal["step_index"] = False
    else:
        internal["end_ns"] = -1

    with pytest.raises(capture.ProfileCaptureError):
        capture._check_profile_trace("same_q8_reference", generation, trace)


def test_rejected_document_preserves_completed_observations(tmp_path: Path) -> None:
    def mismatch_profile(*args: Any, **kwargs: Any) -> DecodeProfile:
        current = profile_cached_generation(*args, **kwargs)
        altered = replace(
            current.generation,
            generated_ids=[77, 99],
            all_ids=[10, 11, 12, 77, 99],
        )
        return replace(current, generation=altered)

    with pytest.raises(capture.ProfileCaptureError) as captured:
        _run(tmp_path, profile_runner=mismatch_profile)
    request = capture.ProfileCaptureRequest(
        tmp_path / "model",
        tmp_path / "assets",
        tmp_path / "bridge.dylib",
        "0" * 64,
        "Profile one compiler sentence.",
        0,
        4,
    )
    rejected = capture.rejected_profile_capture(request, captured.value)

    assert rejected["capture_status"] == "rejected"
    assert rejected["performance_claim_allowed"] is False
    assert rejected["failure"]["stage"].endswith("profile")
    mode = rejected["completed_observations"]["same_q8_reference"]
    assert set(mode) == {"warmup", "control", "profile"}


def test_capture_publication_never_replaces_an_existing_object(tmp_path: Path) -> None:
    existing = tmp_path / "capture.json"
    existing.write_text("original", encoding="utf-8")
    with pytest.raises(capture.ProfileCaptureError, match="publish"):
        capture.write_profile_capture(existing, {"format": "diagnostic"})
    assert existing.read_text(encoding="utf-8") == "original"

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to("missing")
    with pytest.raises(capture.ProfileCaptureError, match="publish"):
        capture.write_profile_capture(dangling, {"format": "diagnostic"})
    assert dangling.is_symlink()


def test_output_preflight_rejects_existing_and_unsafe_paths(tmp_path: Path) -> None:
    available = tmp_path / "available.json"
    capture.check_profile_output(available)

    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(capture.ProfileCaptureError, match="already exists"):
        capture.check_profile_output(existing)

    linked_parent = tmp_path / "linked"
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(capture.ProfileCaptureError, match="unsafe"):
        capture.check_profile_output(linked_parent / "capture.json")

    with pytest.raises(capture.ProfileCaptureError, match="absolute normalized"):
        capture.check_profile_output(Path("relative.json"))


def test_source_identity_covers_every_executed_capture_module() -> None:
    files = capture._source_identity()["files"]
    assert set(files) == {
        "python/decodeforge/decode_profile.py",
        "python/decodeforge/evaluation.py",
        "python/decodeforge/g3_session.py",
        "python/decodeforge/presentation_demo.py",
        "python/decodeforge/profile_capture.py",
        "python/decodeforge/qproj_adapter.py",
        "python/decodeforge/qproj_model.py",
        "python/decodeforge/qproj_profile.py",
        "python/decodeforge/torch_bridge.py",
        "scripts/run_profile_capture.py",
    }


@pytest.mark.parametrize(
    ("mode", "expected_events"),
    [
        (QProjExecutionMode.SAME_Q8_REFERENCE, 18_881),
        (QProjExecutionMode.HYBRID_NATIVE, 14_723),
    ],
)
def test_detailed_maximum_length_trace_fits_event_budget(
    mode: QProjExecutionMode, expected_events: int
) -> None:
    model = FakeModel(stop_after_cached=False)
    installation = FakeInstallation(model)
    installation.set_execution_mode(mode)
    tokenizer = FakeTokenizer()
    input_ids = torch.tensor([[10, 11, 12]], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    try:
        traced = profile_cached_generation(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            64,
            module_paths=tinyllama_component_paths(),
            qproj_details=True,
        )
        assert len(traced.generation.generated_ids) == 64
        assert len(traced.events) == expected_events
    finally:
        installation.close()

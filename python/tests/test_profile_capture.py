from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from decodeforge import profile_capture as capture
from decodeforge.decode_profile import (
    DecodeProfile,
    generate_cached_unprofiled,
    profile_cached_generation,
)
from decodeforge.evaluation import CachedGeneration
from decodeforge.qproj_adapter import QProjExecutionMode
from decodeforge.qproj_model import (
    QProjLayerCounters,
    QProjModelCounters,
    tinyllama_qproj_paths,
)
from decodeforge.torch_bridge import RuntimeLibrary
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


class FakeAdapter(nn.Module):
    def __init__(self, layer: int, installation: FakeInstallation) -> None:
        super().__init__()
        self.layer = layer
        self.installation = installation
        self.forward_count = 0
        self.native = 0
        self.fallback = 0
        self._last_guard_reason: str | None = None

    @property
    def adapter(self) -> FakeAdapter:
        return self

    @property
    def counters(self) -> Any:
        return SimpleNamespace(
            forward=self.forward_count,
            native_attempt=self.native,
            native_success=self.native,
            native_error=0,
            fallback_attempt=self.fallback,
            fallback_success=self.fallback,
            fallback_error=0,
            predispatch_error=0,
        )

    @property
    def last_guard_reason(self) -> str | None:
        return self._last_guard_reason

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        cached = inputs.shape[1] == 1
        if self.installation.mode is QProjExecutionMode.HYBRID_NATIVE and cached:
            self.native += 1
            self._last_guard_reason = None
        else:
            self.fallback += 1
            self._last_guard_reason = "forced_or_prefill"
        return inputs


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
    def __init__(self) -> None:
        super().__init__()
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
        token = 1 if kwargs.get("past_key_values") is None else 99
        logits[0, -1, token] = 2.0
        return SimpleNamespace(logits=logits, past_key_values=object())


class FakeInstallation:
    def __init__(self, model: FakeModel, *, fail_close: bool = False) -> None:
        self.model = model
        self.mode = QProjExecutionMode.HYBRID_NATIVE
        self.originals = tuple(
            model.get_submodule(path) for path in tinyllama_qproj_paths()
        )
        self.adapters = tuple(FakeAdapter(layer, self) for layer in range(22))
        self.inventory = SimpleNamespace(aggregate_identity="sha256:" + "a" * 64)
        self._closed = False
        self.fail_close = fail_close
        for path, adapter in zip(tinyllama_qproj_paths(), self.adapters, strict=True):
            parent_path, _, name = path.rpartition(".")
            setattr(model.get_submodule(parent_path), name, adapter)

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
                forward=adapter.forward_count,
                native_attempt=adapter.native,
                native_success=adapter.native,
                native_error=0,
                fallback_attempt=adapter.fallback,
                fallback_success=adapter.fallback,
                fallback_error=0,
                predispatch_error=0,
                rejected_closed=0,
                in_flight=0,
                closed=self._closed,
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
        qproj = tinyllama_qproj_paths()[0]
        dispatch = [
            event["dispatch"]
            for event in run["profile"]["trace"]["events"]
            if event["module_path"] == qproj
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
        "python/decodeforge/torch_bridge.py",
        "scripts/run_profile_capture.py",
    }

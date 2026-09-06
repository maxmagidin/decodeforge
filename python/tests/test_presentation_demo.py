from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from decodeforge import presentation_demo as demo
from decodeforge.presentation_demo import (
    PresentationDemoError,
    PresentationRequest,
    run_presentation_demo,
)
from decodeforge.qproj_adapter import QProjExecutionMode
from decodeforge.qproj_model import QProjLayerCounters, tinyllama_qproj_paths
from decodeforge.torch_bridge import RuntimeLibrary
from torch import nn

ROOT = Path("/opt/decodeforge-presentation")


def test_presentation_model_pins_match_accepted_g3_model() -> None:
    repository = Path(__file__).resolve().parents[2]
    spec = json.loads((repository / "benchmarks/g3/spec.json").read_text())
    model = spec["model"]
    records = [
        model["weights_file"],
        *model["configuration_files"],
        *model["tokenizer_files"],
    ]
    assert model["model_id"] == demo._PINNED_MODEL_ID
    assert model["revision"] == demo._PINNED_MODEL_REVISION
    assert {
        record["filename"]: (record["size_bytes"], record["sha256"])
        for record in records
    } == demo._PINNED_MODEL_FILES


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = None

    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> torch.Tensor:
        self.template_calls.append({"messages": messages, **kwargs})
        if kwargs.get("tokenize") is False:
            return "<user>Explain a compiler in one sentence.<assistant>"  # type: ignore[return-value]
        return torch.tensor([[10, 11, 12]], dtype=torch.int64)

    def decode(self, values: Any, *, skip_special_tokens: bool) -> str:
        return ("clean text" if skip_special_tokens else "raw <eos>") + str(
            list(values)
        )


class FakeModel(nn.Module):
    def __init__(self, *, mismatch: bool = False) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        self.installation: FakeInstallation | None = None
        self.mismatch = mismatch
        self.generate_calls: list[dict[str, Any]] = []
        for _ in range(22):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(1, 1, bias=False)
            self.model.layers.append(layer)
        self.eval()

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        assert self.installation is not None
        increment = self.installation.mode
        # One prefill call plus one cached decode call reaches every q_proj.
        self.installation.calls += 2
        if increment is QProjExecutionMode.HYBRID_NATIVE:
            self.installation.native += 22
            self.installation.fallback += 22
        else:
            self.installation.fallback += 44
        token = (
            14
            if self.mismatch and increment is QProjExecutionMode.HYBRID_NATIVE
            else 13
        )
        return torch.cat(
            (input_ids, torch.tensor([[token, 99]], dtype=torch.int64)), dim=1
        )


@dataclass
class FakeInstallation:
    model: FakeModel
    mode: QProjExecutionMode = QProjExecutionMode.HYBRID_NATIVE
    calls: int = 0
    native: int = 0
    fallback: int = 0
    _closed: bool = False

    def __post_init__(self) -> None:
        self.inventory = SimpleNamespace(aggregate_identity="sha256:" + "a" * 64)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def execution_mode(self) -> QProjExecutionMode:
        return self.mode

    @property
    def counters(self) -> Any:
        layers = tuple(
            QProjLayerCounters(
                layer=layer,
                layer_path=tinyllama_qproj_paths()[layer],
                forward=self.calls,
                native_attempt=self.native // 22,
                native_success=self.native // 22,
                native_error=0,
                fallback_attempt=self.fallback // 22,
                fallback_success=self.fallback // 22,
                fallback_error=0,
                predispatch_error=0,
                rejected_closed=0,
                in_flight=0,
                closed=self._closed,
            )
            for layer in range(22)
        )
        return SimpleNamespace(
            installed_modules=0 if self._closed else 22,
            restored_modules=22 if self._closed else 0,
            live_adapters=0 if self._closed else 22,
            in_flight=0,
            closed=self._closed,
            layers=layers,
        )

    def set_execution_mode(self, mode: QProjExecutionMode) -> QProjExecutionMode:
        previous = self.mode
        self.mode = mode
        return previous

    def close(self) -> None:
        self._closed = True


def _request() -> PresentationRequest:
    return PresentationRequest(
        model_directory=ROOT / "model",
        asset_directory=ROOT / "assets",
        bridge_library=ROOT / "libdecodeforge_bridge.dylib",
        bridge_sha256="0" * 64,
        prompt="Explain a compiler in one sentence.",
        max_new_tokens=8,
    )


def _run(
    *, mismatch: bool = False
) -> tuple[dict[str, Any], FakeTokenizer, FakeModel, FakeInstallation]:
    tokenizer = FakeTokenizer()
    model = FakeModel(mismatch=mismatch)
    installation = FakeInstallation(model)

    def installer(current: nn.Module, _assets: Path, _runtime: Any) -> FakeInstallation:
        assert current is model
        model.installation = installation
        return installation

    result = run_presentation_demo(
        _request(),
        model_loader=lambda _path: model,
        tokenizer_loader=lambda _path: tokenizer,
        runtime_loader=lambda _path, _digest: cast(RuntimeLibrary, object()),
        installer=installer,
        model_verifier=lambda _path: {},
    )
    return result, tokenizer, model, installation


@pytest.fixture(autouse=True)
def local_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model"
    assets = tmp_path / "assets"
    library = tmp_path / "bridge.dylib"
    model.mkdir()
    assets.mkdir()
    library.write_bytes(b"bridge")
    monkeypatch.setattr(
        sys.modules[__name__],
        "_request",
        lambda: PresentationRequest(
            model, assets, library, "0" * 64, "Explain a compiler in one sentence.", 8
        ),
    )


def test_demo_applies_chat_template_and_proves_both_paths() -> None:
    result, tokenizer, model, installation = _run()
    assert tokenizer.template_calls == [
        {
            "messages": [
                {"role": "user", "content": "Explain a compiler in one sentence."}
            ],
            "add_generation_prompt": True,
            "tokenize": False,
        },
        {
            "messages": [
                {"role": "user", "content": "Explain a compiler in one sentence."}
            ],
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
        },
    ]
    assert len(model.generate_calls) == 2
    assert all(call["do_sample"] is False for call in model.generate_calls)
    assert all(call["use_cache"] is True for call in model.generate_calls)
    assert all(call["eos_token_id"] == 99 for call in model.generate_calls)
    assert all(call["pad_token_id"] == 99 for call in model.generate_calls)
    assert all(
        call["return_dict_in_generate"] is False for call in model.generate_calls
    )
    assert result["comparison"]["token_ids_exact"] is True
    assert result["runs"]["same_q8_reference"]["raw_text"].startswith("raw")
    assert result["runs"]["hybrid_native"]["text"].startswith("clean")
    assert result["runs"]["hybrid_native"]["stopped_by_eos"] is True
    assert result["runs"]["hybrid_native"]["stop_reason"] == "eos"
    assert result["runs"]["hybrid_native"]["generated_token_count"] == 2
    assert result["restoration"]["closed"] is True
    assert installation.closed


@pytest.mark.parametrize(
    ("closed", "field", "value"),
    [
        (False, "installed_modules", 21),
        (False, "live_adapters", 21),
        (False, "restored_modules", 1),
        (True, "installed_modules", 1),
        (True, "live_adapters", 1),
        (True, "restored_modules", 21),
    ],
)
def test_demo_rejects_inconsistent_lifecycle(
    monkeypatch: pytest.MonkeyPatch, closed: bool, field: str, value: int
) -> None:
    original = cast(property, vars(FakeInstallation)["counters"]).fget
    assert original is not None

    def counters(installation: FakeInstallation) -> Any:
        result = original(installation)
        if installation.closed == closed:
            setattr(result, field, value)
        return result

    monkeypatch.setattr(FakeInstallation, "counters", property(counters))
    with pytest.raises(PresentationDemoError, match="installation|cleanup"):
        _run()


def test_demo_rejects_incomplete_layer_coverage() -> None:
    result, *_ = _run()
    delta = result["runs"]["hybrid_native"]["counter_delta"]
    with pytest.raises(PresentationDemoError, match="all 22"):
        demo._check_counter_delta("hybrid_native", 2, delta[:21])
    delta[21]["native_success"] = 0
    with pytest.raises(PresentationDemoError, match="layer 21"):
        demo._check_counter_delta("hybrid_native", 2, delta)


def test_demo_rejects_no_cached_decode() -> None:
    result, *_ = _run()
    delta = result["runs"]["hybrid_native"]["counter_delta"]
    with pytest.raises(PresentationDemoError, match="without native"):
        demo._check_counter_delta("hybrid_native", 1, delta)


@pytest.mark.parametrize("tokens", [[99, 13], [13]])
def test_demo_rejects_incorrect_stopping(tokens: list[int]) -> None:
    class InvalidStopModel(nn.Module):
        def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return torch.cat((input_ids, torch.tensor([tokens])), dim=1)

    inputs = torch.tensor([[10, 11, 12]])
    with pytest.raises(PresentationDemoError, match="EOS"):
        demo._generate(InvalidStopModel(), FakeTokenizer(), inputs, inputs, 8)


def test_demo_does_not_overwrite_report(tmp_path: Path) -> None:
    path = tmp_path / "existing.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(PresentationDemoError, match="already exists"):
        demo.write_demo_json(path, {"benchmark": False})
    assert path.read_text(encoding="utf-8") == "original"


def test_demo_records_rendered_prompt_and_counters() -> None:
    result, _tokenizer, _model, _installation = _run()
    template = result["chat_template"]
    assert template["add_generation_prompt"] is True
    assert template["rendered"]
    reference = result["runs"]["same_q8_reference"]
    hybrid = result["runs"]["hybrid_native"]
    assert all(
        layer["fallback_success"] == 2 and layer["native_success"] == 0
        for layer in reference["counter_delta"]
    )
    assert all(
        layer["fallback_success"] == 1 and layer["native_success"] == 1
        for layer in hybrid["counter_delta"]
    )


def test_demo_accepts_batch_encoding_chat_template_result() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    installation = FakeInstallation(model)

    def batch_template(messages: Any, **kwargs: Any) -> Any:
        tokenizer.template_calls.append({"messages": messages, **kwargs})
        if kwargs.get("tokenize") is False:
            return "<user>Explain a compiler in one sentence.<assistant>"
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
        }

    tokenizer.apply_chat_template = batch_template  # type: ignore[method-assign]

    def installer(current: nn.Module, _assets: Path, _runtime: Any) -> FakeInstallation:
        model.installation = installation
        return installation

    run_presentation_demo(
        _request(),
        model_loader=lambda _path: model,
        tokenizer_loader=lambda _path: tokenizer,
        runtime_loader=lambda _path, _digest: cast(RuntimeLibrary, object()),
        installer=installer,
        model_verifier=lambda _path: {},
    )


def test_demo_rejects_token_bound_before_loading() -> None:
    request = _request()
    with pytest.raises(PresentationDemoError, match="1..64"):
        run_presentation_demo(
            PresentationRequest(
                request.model_directory,
                request.asset_directory,
                request.bridge_library,
                request.bridge_sha256,
                request.prompt,
                65,
            ),
            model_loader=lambda _path: pytest.fail("model loaded"),
        )


def test_demo_closes_installation_when_token_ids_differ() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(mismatch=True)
    installation = FakeInstallation(model)

    def installer(current: nn.Module, _assets: Path, _runtime: Any) -> FakeInstallation:
        model.installation = installation
        return installation

    with pytest.raises(PresentationDemoError, match="token IDs differ"):
        run_presentation_demo(
            _request(),
            model_loader=lambda _path: model,
            tokenizer_loader=lambda _path: tokenizer,
            runtime_loader=lambda _path, _digest: cast(RuntimeLibrary, object()),
            installer=installer,
            model_verifier=lambda _path: {},
        )
    assert installation.closed

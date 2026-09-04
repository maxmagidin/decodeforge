from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from decodeforge.g3_session import (
    G3SessionError,
    InstallResult,
    SessionDependencies,
    SessionRequest,
    VerifiedInputState,
    _load_preparation_receipt,
    _load_verified_components,
    _normalized_architecture,
    _open_directory,
    publish_new_json,
    run_session,
)
from decodeforge.qproj_adapter import QProjExecutionMode
from decodeforge.qproj_model import (
    TINYLLAMA_QPROJ_AGGREGATE_ID,
    TINYLLAMA_QPROJ_MODULE_ID,
    QProjAssetEntry,
    QProjAssetInventory,
    QProjLayerCounters,
    QProjModelCounters,
    QProjSource,
    tinyllama_qproj_paths,
)
from torch import nn

_SPEC = Path(__file__).parents[2] / "benchmarks/g3/spec.json"


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000

    def __call__(self) -> int:
        self.value += 100
        return self.value


class LlamaTokenizer:
    eos_token_id: int | list[int] | None = None
    is_fast = True

    def __init__(self, *, correct_prompt: bool = True) -> None:
        self.correct_prompt = correct_prompt

    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens
        if not self.correct_prompt:
            return [999]
        return [1, 14350, 697, 3273, 10541, 1048, 263, 6516, 29889]

    def decode(self, token_ids: Any, **_kwargs: object) -> str:
        return " ".join(str(value) for value in token_ids)


class _QProj(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.installation: _Installation | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        installation = self.installation
        assert installation is not None
        if not (installation.bad_counters and self.layer == 0):
            installation.forward[self.layer] += 1
            if (
                installation.mode is QProjExecutionMode.HYBRID_NATIVE
                and value.shape[-2] == 1
            ):
                installation.native[self.layer] += 1
            else:
                installation.fallback[self.layer] += 1
        return value


class _Layer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = _QProj(layer)


class LlamaForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sentinel = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(layer) for layer in range(22)])
        self.eval()

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> object:
        value = torch.zeros((1, input_ids.shape[1], 1), dtype=torch.float32)
        for path in tinyllama_qproj_paths():
            value = self.get_submodule(path)(value)
        logits = torch.zeros((1, input_ids.shape[1], 32000), dtype=torch.float32)
        logits[..., 3] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def _ident(seed: int) -> str:
    return f"sha256:{seed:064x}"


def _inventory() -> QProjAssetInventory:
    fixture = json.loads(
        (_SPEC.parents[2] / "tests/fixtures/g3/asset-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    entries = tuple(
        QProjAssetEntry(
            layer=entry["layer"],
            directory=entry["directory"],
            layer_path=tinyllama_qproj_paths()[entry["layer"]],
            manifest_identity=entry["manifest_identity"],
            tensor_name=entry["tensor_name"],
            tensor_identity=entry["tensor_identity"],
            logical_weight_identity=entry["logical_weight_identity"],
            packed_weight_identity=entry["packed_weight_identity"],
            packed_bytes=entry["packed_bytes"],
            module_identity=TINYLLAMA_QPROJ_MODULE_ID,
            fallback_weight_identity=entry["fallback_identity"],
            fallback_parent_logical_weight_identity=entry["logical_weight_identity"],
            fallback_parent_packed_weight_identity=entry["packed_weight_identity"],
        )
        for entry in fixture["entries"]
    )
    return QProjAssetInventory(
        source=QProjSource(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "model.safetensors",
            2_200_119_864,
            "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
        ),
        entries=entries,
        total_packed_bytes=103_809_024,
        total_fallback_bytes=369_098_752,
        aggregate_identity=fixture["aggregate_identity"],
    )


class _Installation:
    def __init__(self, model: LlamaForCausalLM, *, bad_counters: bool) -> None:
        self.mode = QProjExecutionMode.HYBRID_NATIVE
        self.forward = [1] * 22
        self.native = [1] * 22
        self.fallback = [0] * 22
        self._closed = False
        self.bad_counters = bad_counters
        self._inventory = _inventory()
        for path in tinyllama_qproj_paths():
            cast(_QProj, model.get_submodule(path)).installation = self

    @property
    def inventory(self) -> QProjAssetInventory:
        return self._inventory

    @property
    def execution_mode(self) -> QProjExecutionMode:
        return self.mode

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def counters(self) -> QProjModelCounters:
        layers = tuple(
            QProjLayerCounters(
                layer=layer,
                layer_path=tinyllama_qproj_paths()[layer],
                forward=self.forward[layer],
                native_attempt=self.native[layer],
                native_success=self.native[layer],
                native_error=0,
                fallback_attempt=self.fallback[layer],
                fallback_success=self.fallback[layer],
                fallback_error=0,
                predispatch_error=0,
                rejected_closed=0,
                in_flight=0,
                closed=self._closed,
            )
            for layer in range(22)
        )
        return QProjModelCounters(
            installed_modules=22,
            live_adapters=0 if self._closed else 22,
            restored_modules=22 if self._closed else 0,
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


@dataclass
class _Harness:
    correct_prompt: bool = True
    bad_counters: bool = False

    def dependencies(self) -> SessionDependencies:
        clock = _Clock()

        def verify(_request: SessionRequest, _spec: Any) -> VerifiedInputState:
            return VerifiedInputState(
                {
                    "environment": {
                        "software": {
                            "python": "3.12.14",
                            "numpy": "2.4.4",
                            "torch": "2.13.0",
                            "transformers": "5.12.1",
                            "tokenizers": "0.22.2",
                            "safetensors": "0.6.2",
                            "rust": "1.98.0",
                            "clang": "Apple clang version 17.0.0 (clang-1700.0.13.5)",
                        },
                        "host": {
                            "host_id": "apple-m4-primary",
                            "os": "macos",
                            "os_version": "15.5",
                            "os_build": "24F74",
                            "kernel_release": "24.5.0",
                            "arch": "aarch64",
                            "cpu_model": "Apple M4",
                            "hardware_model": "Mac16,13",
                            "physical_cores": 10,
                            "logical_cores": 10,
                            "features": ["neon"],
                            "affinity_policy": (
                                "macOS default scheduler; no hard affinity requested"
                            ),
                        },
                    },
                    "bridge_library": {
                        "path": "libdecodeforge_bridge.dylib",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    },
                },
                None,
                Path("fake-bridge"),
            )

        def install(model: nn.Module, *_arguments: object) -> InstallResult:
            installation = _Installation(
                cast(LlamaForCausalLM, model), bad_counters=self.bad_counters
            )
            checks = tuple(
                {
                    "layer": layer,
                    "layer_path": tinyllama_qproj_paths()[layer],
                    "result": {
                        "max_abs": 0.0,
                        "max_allowed": 0.0001,
                        "max_excess": -0.0001,
                        "finite": True,
                        "pass": True,
                    },
                }
                for layer in range(22)
            )
            return InstallResult(installation, checks)

        def load_tokenizer(_path: Path, _spec: Mapping[str, Any]) -> LlamaTokenizer:
            return LlamaTokenizer(correct_prompt=self.correct_prompt)

        return SessionDependencies(
            verify_inputs=verify,
            load_model=lambda _path, _spec: LlamaForCausalLM(),
            load_tokenizer=cast(Any, load_tokenizer),
            load_runtime=lambda _path, _sha: cast(Any, object()),
            install=install,
            configure_torch=lambda _spec: None,
            load_preparation_receipt=lambda _path: {
                "source": "separately_captured_prepare_command",
                "receipt_identity": _ident(700),
                "elapsed_ns": 1,
                "asset_inventory_identity": TINYLLAMA_QPROJ_AGGREGATE_ID,
                "_checkout_revision": "1" * 40,
                "_command_argv": ["decodeforge-prepare-qproj", "--offline"],
                "_tool_executable_identity": _ident(900),
            },
            checkout_evidence=lambda _path: {
                "revision": "1" * 40,
                "dirty": False,
            },
            verify_preparation_command=lambda _request, _argv, _identity: None,
            clock_ns=clock,
            peak_rss_bytes=lambda: 100,
        )


def _request() -> SessionRequest:
    return SessionRequest(
        session_id="fake-session",
        session_index=0,
        spec_path=_SPEC,
        model_directory=Path("unused-model"),
        asset_directory=Path("unused-assets"),
        bridge_library=Path("unused-library"),
        bridge_sha256="0" * 64,
        preparation_receipt=Path("unused-receipt"),
        process_start_ns=0,
    )


def test_fake_complete_session_is_fully_reconciled() -> None:
    result = run_session(_request(), dependencies=_Harness().dependencies())

    runs = result["runs"]
    assert len(runs) == 24
    assert [run["order_index"] for run in runs] == list(range(24))
    assert [run["path"] for run in runs[:4]] == [
        "same_q8_reference",
        "hybrid_native",
        "hybrid_native",
        "same_q8_reference",
    ]
    assert all(run["timing"]["q_projection_ns"] > 0 for run in runs)
    assert all(len(run["timing"]["q_projection_dispatches"]) == 176 for run in runs)
    hybrid = [run for run in runs if run["path"] == "hybrid_native"]
    reference = [run for run in runs if run["path"] == "same_q8_reference"]
    assert all(len(run["native_output_checks"]) == 154 for run in hybrid)
    assert all(run["native_output_checks"] == [] for run in reference)
    assert result["correctness"]["direct_operator"]["pass"] is True
    assert result["reconciliation"]["installation"] == {
        "installed_modules": 22,
        "restored_modules": 22,
        "live_adapters": 0,
        "in_flight": 0,
        "closed": True,
    }


def test_prompt_identity_mismatch_fails_closed() -> None:
    with pytest.raises(G3SessionError, match="prompt IDs"):
        run_session(
            _request(), dependencies=_Harness(correct_prompt=False).dependencies()
        )


def test_counter_mismatch_fails_closed() -> None:
    with pytest.raises(G3SessionError, match="counter reconciliation"):
        run_session(_request(), dependencies=_Harness(bad_counters=True).dependencies())


def test_retained_directory_descriptor_survives_path_swap(tmp_path: Path) -> None:
    requested = tmp_path / "model"
    requested.mkdir()
    (requested / "sentinel").write_text("verified", encoding="utf-8")
    model_fd = _open_directory(requested)
    parked = tmp_path / "parked"
    requested.rename(parked)
    requested.mkdir()
    (requested / "sentinel").write_text("replacement", encoding="utf-8")
    verified = VerifiedInputState({}, model_fd, Path("bridge"))
    observed: list[str] = []

    def load_model(path: Path, _spec: Any) -> nn.Module:
        observed.append((path / "sentinel").read_text(encoding="utf-8"))
        return LlamaForCausalLM()

    def load_tokenizer(path: Path, _spec: Mapping[str, Any]) -> LlamaTokenizer:
        observed.append((path / "sentinel").read_text(encoding="utf-8"))
        return LlamaTokenizer()

    dependencies = SessionDependencies(
        verify_inputs=lambda _request, _spec: verified,
        load_model=load_model,
        load_tokenizer=cast(Any, load_tokenizer),
        load_runtime=lambda _path, _sha: cast(Any, object()),
        install=cast(Any, None),
        configure_torch=lambda _spec: None,
        load_preparation_receipt=cast(Any, None),
        checkout_evidence=cast(Any, None),
        verify_preparation_command=cast(Any, None),
    )
    try:
        _load_verified_components(_request(), {}, verified, dependencies)
    finally:
        verified.close()
    assert observed == ["verified", "verified"]


def test_result_publication_is_no_replace_and_rejects_symlink_parent(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    output = output_directory / "session.json"
    publish_new_json(output, {"accepted": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"accepted": True}
    with pytest.raises(FileExistsError):
        publish_new_json(output, {"accepted": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {"accepted": True}

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(G3SessionError, match="without symlinks"):
        publish_new_json(linked / "session.json", {})


def _write_receipt(path: Path, *, tool_name: str = "decodeforge-prepare-qproj") -> None:
    value: dict[str, Any] = {
        "schema_version": 1,
        "format": "decodeforge_g3_offline_preparation_receipt_v1",
        "protocol_id": "g3-tinyllama-qproj-generation-v1",
        "checkout": {"revision": "1" * 40, "dirty": False},
        "command": {"argv": ["decodeforge-prepare-qproj", "--offline"]},
        "source": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "filename": "model.safetensors",
            "bytes": 2_200_119_864,
            "identity": (
                "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
            ),
        },
        "tool": {
            "name": tool_name,
            "version": "0.1.0",
            "executable_identity": _ident(900),
        },
        "output": {
            "asset_inventory_identity": TINYLLAMA_QPROJ_AGGREGATE_ID,
            "layer_count": 22,
            "total_packed_bytes": 103_809_024,
            "total_fallback_bytes": 369_098_752,
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "start_ns": 10,
            "stop_ns": 20,
            "elapsed_ns": 10,
        },
    }
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    value["receipt_identity"] = (
        "sha256:"
        + hashlib.sha256(
            b"DecodeForge/g3-offline-preparation-receipt/v1\0" + encoded
        ).hexdigest()
    )
    path.write_text(json.dumps(value), encoding="utf-8")


def test_preparation_receipt_is_identity_and_content_bound(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    _write_receipt(valid)
    assert _load_preparation_receipt(valid)["elapsed_ns"] == 10

    invalid = tmp_path / "invalid.json"
    _write_receipt(invalid, tool_name="other-tool")
    with pytest.raises(G3SessionError, match="tool"):
        _load_preparation_receipt(invalid)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"elapsed":NaN}', encoding="utf-8")
    with pytest.raises(G3SessionError, match="valid JSON"):
        _load_preparation_receipt(nonfinite)


def test_darwin_arm64_uses_the_contract_architecture_name() -> None:
    assert _normalized_architecture("arm64") == "aarch64"
    assert _normalized_architecture("x86_64") == "x86_64"

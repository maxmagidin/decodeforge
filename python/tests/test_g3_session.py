from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import decodeforge.g3_session as g3_session_module
import pytest
import torch
from decodeforge.g3_preparation import VerifiedPreparationReceipt
from decodeforge.g3_session import (
    G3SessionError,
    InstallResult,
    SessionDependencies,
    SessionRequest,
    VerifiedInputState,
    _bridge_rebuild_command,
    _checkout_evidence,
    _command_line,
    _load_verified_components,
    _normalized_architecture,
    _open_directory,
    _session_rebuild_command,
    publish_new_json,
    require_external_session_output,
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
_TEST_EXTERNAL = Path(tempfile.gettempdir()).resolve()


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
            installed_modules=0 if self._closed else 22,
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
    changing_checkout: bool = False

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
                    "bridge_rebuild_command": (
                        "env CARGO_TARGET_DIR=/opt/homebrew/var/"
                        "decodeforge-g3-evidence/cargo-target make build-g3-bridge"
                    ),
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

        checkout_calls = 0

        def checkout(_path: Path) -> dict[str, Any]:
            nonlocal checkout_calls
            checkout_calls += 1
            revision = (
                "2" * 40 if self.changing_checkout and checkout_calls > 1 else "1" * 40
            )
            return {"revision": revision, "dirty": False}

        return SessionDependencies(
            verify_inputs=verify,
            load_model=lambda _path, _spec: LlamaForCausalLM(),
            load_tokenizer=cast(Any, load_tokenizer),
            load_runtime=lambda _path, _sha: cast(Any, object()),
            install=install,
            configure_torch=lambda _spec: None,
            load_preparation_receipt=lambda _path: VerifiedPreparationReceipt(
                checkout_revision="1" * 40,
                command_argv=("decodeforge-prepare-qproj", "--offline"),
                tool_executable_identity=_ident(900),
                receipt_identity=_ident(700),
                elapsed_ns=1,
                asset_inventory_identity=TINYLLAMA_QPROJ_AGGREGATE_ID,
            ),
            checkout_evidence=checkout,
            verify_preparation_command=lambda _request, _argv, _identity: None,
            clock_ns=clock,
            peak_rss_bytes=lambda: 100,
        )


def _request() -> SessionRequest:
    return SessionRequest(
        session_id="fake-session",
        session_index=0,
        spec_path=_SPEC,
        model_directory=_TEST_EXTERNAL / "decodeforge-test-model",
        asset_directory=_TEST_EXTERNAL / "decodeforge-test-assets",
        bridge_library=Path(
            "/opt/homebrew/var/decodeforge-g3-evidence/cargo-target/"
            "release/libdecodeforge_bridge.dylib"
        ),
        bridge_sha256="0" * 64,
        preparation_receipt=_TEST_EXTERNAL / "decodeforge-test-receipt.json",
        session_output=_TEST_EXTERNAL / "decodeforge-test-session.json",
        process_start_ns=0,
    )


@pytest.mark.parametrize("phase", ["installation", "cleanup"])
def test_session_rejects_inconsistent_installed_module_counts(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    class InconsistentInstallation(_Installation):
        @property
        def counters(self) -> QProjModelCounters:
            snapshot = super().counters
            if phase == "installation" and not self.closed:
                return replace(snapshot, installed_modules=21)
            if phase == "cleanup" and self.closed:
                return replace(snapshot, installed_modules=1)
            return snapshot

    monkeypatch.setitem(globals(), "_Installation", InconsistentInstallation)
    expected = (
        "transactional installation is not fully live and idle"
        if phase == "installation"
        else "adapter cleanup left installed query projections"
    )
    with pytest.raises(G3SessionError, match=expected):
        run_session(_request(), dependencies=_Harness().dependencies())


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
    commands = result["provenance"]["rebuild_commands"]
    assert commands["build_bridge"] == (
        "env CARGO_TARGET_DIR=/opt/homebrew/var/"
        "decodeforge-g3-evidence/cargo-target make build-g3-bridge"
    )
    assert shlex.split(commands["run_session"]) == [
        "make",
        "run-g3-demo",
        "SESSION_ID=fake-session",
        "SESSION_INDEX=0",
        f"MODEL_DIR={_TEST_EXTERNAL / 'decodeforge-test-model'}",
        f"ASSETS={_TEST_EXTERNAL / 'decodeforge-test-assets'}",
        (
            "LIBRARY=/opt/homebrew/var/decodeforge-g3-evidence/cargo-target/"
            "release/libdecodeforge_bridge.dylib"
        ),
        f"LIBRARY_SHA256={'0' * 64}",
        f"PREPARATION_RECEIPT={_TEST_EXTERNAL / 'decodeforge-test-receipt.json'}",
        f"OUTPUT={_TEST_EXTERNAL / 'decodeforge-test-session.json'}",
    ]


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


def test_checkout_change_during_session_fails_closed() -> None:
    with pytest.raises(G3SessionError, match="checkout changed"):
        run_session(
            _request(),
            dependencies=_Harness(changing_checkout=True).dependencies(),
        )


def test_preparation_checkout_mismatch_fails_before_model_load_or_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _Harness().dependencies()
    calls: list[str] = []
    original_verify_inputs = dependencies.verify_inputs

    def verify_inputs(request: SessionRequest, spec: Any) -> VerifiedInputState:
        calls.append("verify_inputs")
        verified = original_verify_inputs(request, spec)
        original_close = verified.close

        def close() -> None:
            calls.append("close")
            original_close()

        monkeypatch.setattr(verified, "close", close)
        return verified

    def configure_torch(_spec: Mapping[str, Any]) -> None:
        calls.append("configure_torch")

    def verify_preparation(*_args: Any) -> None:
        calls.append("verify_preparation")

    receipt = VerifiedPreparationReceipt(
        checkout_revision="2" * 40,
        command_argv=("decodeforge-prepare-qproj", "--offline"),
        tool_executable_identity=_ident(900),
        receipt_identity=_ident(700),
        elapsed_ns=1,
        asset_inventory_identity=TINYLLAMA_QPROJ_AGGREGATE_ID,
    )

    def load_model(_path: Path, _spec: Any) -> nn.Module:
        calls.append("load_model")
        return LlamaForCausalLM()

    def install(*_args: Any) -> Any:
        calls.append("install")
        return None

    dependencies = replace(
        dependencies,
        verify_inputs=verify_inputs,
        load_preparation_receipt=lambda _path: receipt,
        load_model=load_model,
        install=install,
        configure_torch=configure_torch,
        verify_preparation_command=verify_preparation,
    )
    with pytest.raises(G3SessionError, match="preparation receipt checkout revision"):
        run_session(_request(), dependencies=dependencies)
    assert calls == ["verify_inputs", "close"]


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_result_publication_rejects_nonfinite_json_before_creating_output(
    tmp_path: Path, value: float
) -> None:
    output = tmp_path / "session.json"
    existing = tmp_path / "existing.txt"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float values"):
        publish_new_json(output, {"nested": {"value": value}})
    assert not output.exists()
    assert existing.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing.txt"]


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


def test_darwin_arm64_uses_the_contract_architecture_name() -> None:
    assert _normalized_architecture("arm64") == "aarch64"
    assert _normalized_architecture("x86_64") == "x86_64"


def test_bridge_rebuild_command_binds_exact_external_cargo_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cargo-target"
    (target / "release").mkdir(parents=True)
    library = target / "release/libdecodeforge_bridge.dylib"
    command = _bridge_rebuild_command(library, _SPEC)
    assert shlex.split(command) == [
        "env",
        f"CARGO_TARGET_DIR={target}",
        "make",
        "build-g3-bridge",
    ]


@pytest.mark.parametrize(
    "relative_library",
    (
        "debug/libdecodeforge_bridge.dylib",
        "release/renamed.dylib",
    ),
)
def test_bridge_rebuild_command_rejects_ambiguous_layouts(
    tmp_path: Path, relative_library: str
) -> None:
    target = tmp_path / "cargo-target"
    (target / Path(relative_library).parent).mkdir(parents=True)
    with pytest.raises(G3SessionError, match="release layout"):
        _bridge_rebuild_command(target / relative_library, _SPEC)


def test_bridge_rebuild_command_rejects_target_inside_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    spec = checkout / "benchmarks/g3/spec.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}", encoding="utf-8")
    target = checkout / "cargo-target"
    (target / "release").mkdir(parents=True)
    with pytest.raises(G3SessionError, match="outside"):
        _bridge_rebuild_command(target / "release/libdecodeforge_bridge.dylib", spec)


def test_session_rebuild_command_round_trips_through_make_dry_run() -> None:
    request = replace(
        _request(),
        session_id="session.with-portable_id",
        model_directory=Path("/opt/homebrew/var/model+tag@host:1"),
        asset_directory=Path("/opt/homebrew/var/assets-tag_1"),
        preparation_receipt=Path("/opt/homebrew/var/receipt.tag-1.json"),
        session_output=Path("/opt/homebrew/var/output.tag-1.json"),
    )
    replay = shlex.split(_session_rebuild_command(request))
    assert replay == [
        "make",
        "run-g3-demo",
        "SESSION_ID=session.with-portable_id",
        "SESSION_INDEX=0",
        "MODEL_DIR=/opt/homebrew/var/model+tag@host:1",
        "ASSETS=/opt/homebrew/var/assets-tag_1",
        (
            "LIBRARY=/opt/homebrew/var/decodeforge-g3-evidence/cargo-target/"
            "release/libdecodeforge_bridge.dylib"
        ),
        f"LIBRARY_SHA256={'0' * 64}",
        "PREPARATION_RECEIPT=/opt/homebrew/var/receipt.tag-1.json",
        "OUTPUT=/opt/homebrew/var/output.tag-1.json",
    ]
    dry_run = subprocess.run(
        [replay[0], "--no-print-directory", "--dry-run", *replay[1:]],
        cwd=_SPEC.parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for expected in (
        '--model-dir "${MODEL_DIR}"',
        '--assets "${ASSETS}"',
        '--library "${LIBRARY}"',
        '--preparation-receipt "${PREPARATION_RECEIPT}"',
        '--output "${OUTPUT}"',
        '--spec "${SPEC:-benchmarks/g3/spec.json}"',
    ):
        assert expected in dry_run
    for public_value in replay[4:]:
        assert public_value.split("=", maxsplit=1)[1] not in dry_run


@pytest.mark.parametrize(
    "unsafe_leaf",
    (
        "model$cash",
        "model`command`",
        'model"quote',
        "model\nnewline",
        "model space",
        "model\\escape",
        "model#comment",
        "model%pattern",
    ),
)
def test_session_rejects_make_replay_metacharacters(unsafe_leaf: str) -> None:
    with pytest.raises(G3SessionError, match="replay-safe"):
        run_session(
            replace(
                _request(),
                model_directory=Path("/opt/homebrew/var") / unsafe_leaf,
            ),
            dependencies=_Harness().dependencies(),
        )


def test_bridge_rebuild_command_rejects_make_expansion_in_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "cargo$(error-injected)"
    (target / "release").mkdir(parents=True)
    with pytest.raises(G3SessionError, match="replay-safe"):
        _bridge_rebuild_command(target / "release/libdecodeforge_bridge.dylib", _SPEC)


def test_session_rejects_overlong_replay_path() -> None:
    with pytest.raises(G3SessionError, match="replay-safe"):
        run_session(
            replace(
                _request(),
                model_directory=Path("/opt") / ("a" * 1024),
            ),
            dependencies=_Harness().dependencies(),
        )


def test_session_rebuild_command_rejects_overlong_aggregate() -> None:
    with pytest.raises(G3SessionError, match="exceeds its byte bound"):
        _session_rebuild_command(
            replace(
                _request(),
                model_directory=Path("/opt") / ("a" * 900),
                asset_directory=Path("/opt") / ("b" * 900),
                bridge_library=Path("/opt") / ("c" * 900),
                preparation_receipt=Path("/opt") / ("d" * 900),
                session_output=Path("/opt") / ("e" * 900),
            )
        )


def test_session_rejects_noncanonical_spec_path(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    canonical = checkout / "benchmarks/g3/spec.json"
    alternate = checkout / "alternate/g3/spec.json"
    canonical.parent.mkdir(parents=True)
    alternate.parent.mkdir(parents=True)
    canonical.write_text("{}", encoding="utf-8")
    alternate.write_text("{}", encoding="utf-8")
    with pytest.raises(G3SessionError, match="canonical experiment spec"):
        run_session(
            replace(_request(), spec_path=alternate),
            dependencies=_Harness().dependencies(),
        )


@pytest.mark.parametrize(
    ("field", "path"),
    (
        ("model_directory", Path("relative-model")),
        ("asset_directory", Path("/opt/homebrew/var/assets/../replacement")),
        ("bridge_library", Path("relative-library")),
        ("preparation_receipt", Path("relative-receipt")),
        ("session_output", Path("relative-output")),
    ),
)
def test_session_rejects_nonabsolute_or_unnormalized_replay_paths(
    field: str, path: Path
) -> None:
    with pytest.raises(G3SessionError, match="absolute normalized"):
        run_session(
            replace(_request(), **cast(Any, {field: path})),
            dependencies=_Harness().dependencies(),
        )


def test_empty_provenance_command_output_fails_closed() -> None:
    with pytest.raises(G3SessionError, match="no output"):
        _command_line(["/usr/bin/true"])


def test_publication_rolls_back_after_post_link_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "session.json"
    real_fsync = os.fsync
    calls = 0

    def fail_second(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_second)
    with pytest.raises(OSError, match="injected"):
        publish_new_json(output, {"accepted": True})
    assert not output.exists()


def test_publication_never_unlinks_a_swapped_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "session.json"
    real_fsync = os.fsync
    calls = 0

    def swap_then_fail(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            output.unlink()
            output.write_text("replacement", encoding="utf-8")
            raise OSError("injected")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", swap_then_fail)
    with pytest.raises(BaseExceptionGroup, match="rollback"):
        publish_new_json(output, {"accepted": True})
    assert output.read_text(encoding="utf-8") == "replacement"


def test_checkout_rejects_hidden_index_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def command(arguments: Any, *, allow_empty: bool = False) -> str:
        del allow_empty
        if "rev-parse" in arguments:
            return "1" * 40 + "\n"
        if "ls-files" in arguments:
            return "S python/decodeforge/g3_session.py\n"
        return ""

    monkeypatch.setattr(g3_session_module, "_command_output", command)
    with pytest.raises(G3SessionError, match="hidden tracked-file"):
        _checkout_evidence(_SPEC)


def test_snapshot_cleanup_failure_retains_retryable_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "one").write_bytes(b"x")
    state = VerifiedInputState(
        {},
        _open_directory(snapshot),
        Path("unused"),
        _model_snapshot_path=snapshot,
        _model_snapshot_files=("one",),
    )
    real_rmdir = os.rmdir
    failed = False

    def fail_once(path: os.PathLike[str] | str, *args: Any, **kwargs: Any) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected cleanup failure")
        real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "rmdir", fail_once)
    with pytest.raises(OSError, match="injected"):
        state.close()
    assert state.model_directory_fd is not None
    assert state._model_snapshot_path == snapshot
    state.close()
    assert state.model_directory_fd is None
    assert not snapshot.exists()


def test_session_output_must_be_outside_measured_checkout(tmp_path: Path) -> None:
    with pytest.raises(G3SessionError, match="outside"):
        require_external_session_output(
            _SPEC.parents[2] / "result.json",
            _SPEC,
        )
    external = tmp_path / "result.json"
    require_external_session_output(external, _SPEC)

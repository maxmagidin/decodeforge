"""Small, non-benchmark TinyLlama presentation demonstration.

This module deliberately does not use the frozen G3 experiment protocol.  It
loads a caller-supplied local checkpoint, applies its chat template, and runs
the same prompt through the owning Q-projection adapters in same-Q8 reference
and hybrid-native modes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, cast

import torch
from torch import nn

from .qproj_adapter import QProjAdapter, QProjExecutionMode
from .qproj_model import (
    QProjModelCounters,
    VerifiedQProjAsset,
    install_tinyllama_qproj,
    tinyllama_qproj_paths,
)
from .torch_bridge import RuntimeLibrary

MAX_NEW_TOKENS: Final = 64
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_PINNED_MODEL_ID: Final = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_PINNED_MODEL_REVISION: Final = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
_PINNED_MODEL_FILES: Final = {
    "model.safetensors": (
        2_200_119_864,
        "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
    ),
    "config.json": (
        608,
        "486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9",
    ),
    "generation_config.json": (
        124,
        "18046d04f5bd8b4998095ecabdd17a1bf0053d9acdccead4a05be4a3575f3c5c",
    ),
    "tokenizer.json": (
        1_842_767,
        "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
    ),
    "tokenizer.model": (
        499_723,
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    ),
    "tokenizer_config.json": (
        1_289,
        "7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a",
    ),
    "special_tokens_map.json": (
        551,
        "82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11aa8d60d9be857ce7db",
    ),
}


class PresentationDemoError(RuntimeError):
    """The standalone presentation demonstration could not complete."""


class InstallationLike(Protocol):
    @property
    def inventory(self) -> Any: ...

    @property
    def closed(self) -> bool: ...

    @property
    def execution_mode(self) -> QProjExecutionMode: ...

    @property
    def counters(self) -> QProjModelCounters: ...

    def set_execution_mode(self, mode: QProjExecutionMode) -> QProjExecutionMode: ...

    def close(self) -> None: ...


ModelLoader: TypeAlias = Callable[[Path], nn.Module]
TokenizerLoader: TypeAlias = Callable[[Path], Any]
RuntimeLoader: TypeAlias = Callable[[Path, str], RuntimeLibrary]
Installer: TypeAlias = Callable[[nn.Module, Path, RuntimeLibrary], InstallationLike]


@dataclass(frozen=True)
class PresentationRequest:
    """Bounded local inputs for one presentation-only run."""

    model_directory: Path
    asset_directory: Path
    bridge_library: Path
    bridge_sha256: str
    prompt: str
    max_new_tokens: int = MAX_NEW_TOKENS


def _require_path(path: Path, label: str, *, directory: bool = False) -> Path:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise PresentationDemoError(f"{label} must be an absolute normalized path")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise PresentationDemoError(f"{label} does not exist") from error
    if directory:
        if not path.is_dir() or path.is_symlink():
            raise PresentationDemoError(f"{label} must be a regular directory")
    elif path.is_symlink() or not path.is_file():
        raise PresentationDemoError(f"{label} must be a regular non-symlink file")
    del metadata
    return path


def _require_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PresentationDemoError("bridge_sha256 must be 64 lowercase hex digits")
    return value


def _verify_model_directory(directory: Path) -> dict[str, dict[str, Any]]:
    """Verify the exact pinned TinyLlama files before loading any weights."""

    try:
        names = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise PresentationDemoError("unable to enumerate model directory") from error
    if names != set(_PINNED_MODEL_FILES):
        raise PresentationDemoError("model directory does not contain the pinned files")
    records: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_digest) in _PINNED_MODEL_FILES.items():
        path = directory / name
        try:
            before = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not path.is_file()
                or before.st_size != expected_bytes
            ):
                raise PresentationDemoError(
                    f"model file {name} is not the pinned regular file"
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            after = path.stat(follow_symlinks=False)
        except OSError as error:
            raise PresentationDemoError(f"unable to read model file {name}") from error
        if (
            before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or digest.hexdigest() != expected_digest
        ):
            raise PresentationDemoError(f"model file {name} identity mismatch")
        records[name] = {"bytes": expected_bytes, "sha256": expected_digest}
    return records


def _load_model(directory: Path) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ),
    )
    model.requires_grad_(False)
    model.eval()
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or (
            parameter.is_floating_point() and parameter.dtype is not torch.float32
        ):
            raise PresentationDemoError("loaded model must be entirely CPU FP32")
    return model


def _load_tokenizer(directory: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
    )


def _load_runtime(path: Path, digest: str) -> RuntimeLibrary:
    return RuntimeLibrary(path, f"sha256:{_require_sha256(digest)}")


def _install(
    model: nn.Module, assets: Path, runtime: RuntimeLibrary
) -> InstallationLike:
    def factory(asset: VerifiedQProjAsset) -> QProjAdapter:
        entry = asset.entry
        return QProjAdapter(
            layer_name=entry.layer_path,
            library=runtime,
            pack_manifest_json=asset.pack_manifest_json,
            packed_weight=asset.packed_weight,
            fallback_weight=asset.fallback_weight,
            fallback_weight_id=entry.fallback_weight_identity,
            fallback_parent_packed_weight_id=(
                entry.fallback_parent_packed_weight_identity
            ),
            expected_module_id=entry.module_identity,
        )

    return install_tinyllama_qproj(model, assets, factory)


def _eos_ids(tokenizer: Any) -> tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, bool) or (
        not isinstance(value, int) and not isinstance(value, Sequence)
    ):
        raise PresentationDemoError("local tokenizer must define eos_token_id")
    if isinstance(value, int):
        if value < 0:
            raise PresentationDemoError("tokenizer eos_token_id must be nonnegative")
        return (value,)
    values = list(value)
    if not values or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in values
    ):
        raise PresentationDemoError("tokenizer eos_token_id must contain integers")
    return tuple(values)


def _chat_inputs(tokenizer: Any, prompt: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise PresentationDemoError("prompt must be nonempty text")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise PresentationDemoError("local tokenizer has no chat template")
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        values = apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as error:
        raise PresentationDemoError("local tokenizer chat template failed") from error
    if not isinstance(rendered, str) or not rendered:
        raise PresentationDemoError("chat template returned no rendered prompt")
    input_ids: Any
    attention_mask: Any
    if isinstance(values, torch.Tensor):
        input_ids = values
        attention_mask = torch.ones_like(values)
    elif isinstance(values, Mapping):
        input_ids = values.get("input_ids")
        attention_mask = values.get("attention_mask")
        if attention_mask is None and isinstance(input_ids, torch.Tensor):
            attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = None
        attention_mask = None
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        attention_mask, torch.Tensor
    ):
        raise PresentationDemoError(
            "chat template must return input_ids and attention_mask tensors"
        )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise PresentationDemoError("chat template returned an invalid token shape")
    if (
        input_ids.device.type != "cpu"
        or input_ids.dtype is not torch.int64
        or attention_mask.shape != input_ids.shape
        or attention_mask.device.type != "cpu"
        or attention_mask.dtype is not torch.int64
    ):
        raise PresentationDemoError("chat template tokens must be CPU int64")
    return input_ids.contiguous(), attention_mask.contiguous(), rendered


def _counter_summary(counters: QProjModelCounters) -> dict[str, Any]:
    layers = [asdict(layer) for layer in counters.layers]
    return {
        "installed_modules": counters.installed_modules,
        "restored_modules": counters.restored_modules,
        "live_adapters": counters.live_adapters,
        "in_flight": counters.in_flight,
        "closed": counters.closed,
        "forward": sum(layer["forward"] for layer in layers),
        "native_attempt": sum(layer["native_attempt"] for layer in layers),
        "native_success": sum(layer["native_success"] for layer in layers),
        "fallback_attempt": sum(layer["fallback_attempt"] for layer in layers),
        "fallback_success": sum(layer["fallback_success"] for layer in layers),
        "layers": layers,
    }


def _counter_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, Any]]:
    fields = (
        "forward",
        "native_attempt",
        "native_success",
        "native_error",
        "fallback_attempt",
        "fallback_success",
        "fallback_error",
        "predispatch_error",
        "rejected_closed",
        "in_flight",
    )
    return [
        {
            "layer": after_layer["layer"],
            **{field: after_layer[field] - before_layer[field] for field in fields},
        }
        for before_layer, after_layer in zip(
            before["layers"], after["layers"], strict=True
        )
    ]


def _check_counter_delta(
    name: str, token_count: int, delta: Sequence[Mapping[str, Any]]
) -> None:
    if len(delta) != 22:
        raise PresentationDemoError("q_proj coverage must contain all 22 layers")
    if name == "hybrid_native" and token_count < 2:
        raise PresentationDemoError("generation ended without native cached decode")
    expected_native = token_count - 1 if name == "hybrid_native" else 0
    expected_fallback = 1 if name == "hybrid_native" else token_count
    for layer, values in enumerate(delta):
        expected = {
            "layer": layer,
            "forward": token_count,
            "native_attempt": expected_native,
            "native_success": expected_native,
            "native_error": 0,
            "fallback_attempt": expected_fallback,
            "fallback_success": expected_fallback,
            "fallback_error": 0,
            "predispatch_error": 0,
            "rejected_closed": 0,
            "in_flight": 0,
        }
        if dict(values) != expected:
            raise PresentationDemoError(
                f"{name} q_proj counters do not reconcile at layer {layer}"
            )


def _generate(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], list[int], str, str, bool]:
    eos = _eos_ids(tokenizer)
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "min_new_tokens": 0,
        "use_cache": True,
        "return_dict_in_generate": False,
        # Transformers accepts an int or a list here.  Keep the validated
        # singleton as an int and normalize multi-EOS tokenizers to a list.
        "eos_token_id": eos[0] if len(eos) == 1 else list(eos),
        # Keep padding deterministic even when a local tokenizer advertises a
        # different padding convention; TinyLlama's EOS is the frozen pad ID.
        "pad_token_id": eos[0],
    }
    try:
        with torch.inference_mode():
            output = cast(Any, model).generate(input_ids, **kwargs)
    except Exception as error:
        raise PresentationDemoError("local model greedy generation failed") from error
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise PresentationDemoError("model generation returned an invalid token shape")
    all_ids = [int(value) for value in output[0].tolist()]
    prompt_ids = [int(value) for value in input_ids[0].tolist()]
    if all_ids[: len(prompt_ids)] != prompt_ids or len(all_ids) == len(prompt_ids):
        raise PresentationDemoError("model generation returned no valid continuation")
    generated_ids = all_ids[len(prompt_ids) :]
    if len(generated_ids) > max_new_tokens:
        raise PresentationDemoError("model generation exceeded its token bound")
    eos_positions = [index for index, value in enumerate(generated_ids) if value in eos]
    if eos_positions and eos_positions[0] != len(generated_ids) - 1:
        raise PresentationDemoError("model generation continued after EOS")
    stopped_by_eos = bool(generated_ids and generated_ids[-1] in eos)
    if not stopped_by_eos and len(generated_ids) != max_new_tokens:
        raise PresentationDemoError("generation stopped before EOS or the token limit")
    raw_text = str(tokenizer.decode(generated_ids, skip_special_tokens=False))
    text = str(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return all_ids, generated_ids, raw_text, text, stopped_by_eos


def run_presentation_demo(
    request: PresentationRequest,
    *,
    model_loader: ModelLoader = _load_model,
    tokenizer_loader: TokenizerLoader = _load_tokenizer,
    runtime_loader: RuntimeLoader = _load_runtime,
    installer: Installer = _install,
    model_verifier: Callable[
        [Path], dict[str, dict[str, Any]]
    ] = _verify_model_directory,
) -> dict[str, Any]:
    """Run the bounded presentation demo and return non-G3 JSON evidence."""

    if (
        type(request.max_new_tokens) is not int
        or not 1 <= request.max_new_tokens <= MAX_NEW_TOKENS
    ):
        raise PresentationDemoError("max_new_tokens must be in the range 1..64")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise PresentationDemoError("prompt must be nonempty text")
    if len(request.prompt) > 4096:
        raise PresentationDemoError("prompt exceeds the 4096-character limit")
    model_directory = _require_path(
        request.model_directory, "model directory", directory=True
    )
    asset_directory = _require_path(
        request.asset_directory, "asset directory", directory=True
    )
    library = _require_path(request.bridge_library, "bridge library")
    digest = _require_sha256(request.bridge_sha256)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        torch.manual_seed(0)
    except RuntimeError as error:
        raise PresentationDemoError(
            "unable to configure deterministic CPU Torch"
        ) from error
    model_files = model_verifier(model_directory)
    model = model_loader(model_directory)
    tokenizer = tokenizer_loader(model_directory)
    if model_verifier(model_directory) != model_files:
        raise PresentationDemoError("model files changed during loading")
    input_ids, attention_mask, rendered_prompt = _chat_inputs(tokenizer, request.prompt)
    if input_ids.shape[1] > 512:
        raise PresentationDemoError("chat prompt exceeds the 512-token limit")
    original_modules = tuple(
        model.get_submodule(path) for path in tinyllama_qproj_paths()
    )
    runtime = runtime_loader(library, digest)
    installation = installer(model, asset_directory, runtime)
    try:
        initial = _counter_summary(installation.counters)
        if (
            installation.closed
            or initial["closed"]
            or initial["installed_modules"] != 22
            or initial["restored_modules"] != 0
            or initial["live_adapters"] != 22
            or initial["in_flight"] != 0
            or len(initial["layers"]) != 22
            or any(
                layer["layer"] != index
                or layer["layer_path"] != path
                or layer["closed"]
                or layer["in_flight"] != 0
                for index, (layer, path) in enumerate(
                    zip(initial["layers"], tinyllama_qproj_paths(), strict=True)
                )
            )
        ):
            raise PresentationDemoError("initial installation must own all 22 layers")
        snapshots: dict[str, dict[str, Any]] = {}
        paths = (
            ("same_q8_reference", QProjExecutionMode.SAME_Q8_REFERENCE),
            ("hybrid_native", QProjExecutionMode.HYBRID_NATIVE),
        )
        token_runs: dict[str, list[int]] = {}
        generated_runs: dict[str, list[int]] = {}
        for name, mode in paths:
            installation.set_execution_mode(mode)
            before = _counter_summary(installation.counters)
            all_ids, generated_ids, raw_text, text, stopped_by_eos = _generate(
                model, tokenizer, input_ids, attention_mask, request.max_new_tokens
            )
            after = _counter_summary(installation.counters)
            delta = _counter_delta(before, after)
            _check_counter_delta(name, len(generated_ids), delta)
            token_runs[name] = all_ids
            generated_runs[name] = generated_ids
            snapshots[name] = {
                "token_ids": all_ids,
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "raw_text": raw_text,
                "text": text,
                "stopped_by_eos": stopped_by_eos,
                "stop_reason": "eos" if stopped_by_eos else "max_new_tokens",
                "counters_before": before,
                "counters_after": after,
                "counter_delta": delta,
            }
        if token_runs["same_q8_reference"] != token_runs["hybrid_native"]:
            raise PresentationDemoError("same-Q8 and hybrid token IDs differ")
        if generated_runs["same_q8_reference"] != generated_runs["hybrid_native"]:
            raise PresentationDemoError("same-Q8 and hybrid generated IDs differ")
    finally:
        installation.close()
    final = _counter_summary(installation.counters)
    restored = all(
        model.get_submodule(path) is original
        for path, original in zip(
            tinyllama_qproj_paths(), original_modules, strict=True
        )
    )
    if (
        not installation.closed
        or not final["closed"]
        or not restored
        or final["installed_modules"] != 0
        or final["restored_modules"] != 22
        or final["live_adapters"] != 0
        or final["in_flight"] != 0
        or len(final["layers"]) != 22
        or any(not layer["closed"] or layer["in_flight"] for layer in final["layers"])
    ):
        raise PresentationDemoError(
            "model adapter cleanup did not fully restore the model"
        )
    return {
        "format": "decodeforge_presentation_demo_v1",
        "benchmark": False,
        "g3_evidence": False,
        "model_directory": str(model_directory),
        "model_identity": {
            "model_id": _PINNED_MODEL_ID,
            "revision": _PINNED_MODEL_REVISION,
            "weights": {
                "bytes": _PINNED_MODEL_FILES["model.safetensors"][0],
                "sha256": _PINNED_MODEL_FILES["model.safetensors"][1],
            },
        },
        "model_files": model_files,
        "tokenizer_identity": {
            name: {
                "bytes": _PINNED_MODEL_FILES[name][0],
                "sha256": _PINNED_MODEL_FILES[name][1],
            }
            for name in (
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
        },
        "asset_inventory_identity": str(installation.inventory.aggregate_identity),
        "bridge_library_sha256": digest,
        "prompt": request.prompt,
        "chat_template": {
            "add_generation_prompt": True,
            "prompt_token_ids": [int(value) for value in input_ids[0].tolist()],
            "rendered": rendered_prompt,
            "rendered_sha256": hashlib.sha256(
                rendered_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "max_new_tokens": request.max_new_tokens,
        "runs": snapshots,
        "comparison": {
            "token_ids_exact": True,
            "generated_token_ids_exact": True,
            "text": snapshots["hybrid_native"]["text"],
        },
        "restoration": {
            "closed": True,
            "original_modules_restored": True,
            "counters": final,
        },
        "installation": initial,
    }


def write_demo_json(path: Path, evidence: dict[str, Any]) -> None:
    """Create one new JSON output without replacing an existing file."""

    path = _require_path(path, "output", directory=False) if path.exists() else path
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise PresentationDemoError("output must be an absolute normalized path")
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as error:
        raise PresentationDemoError("output already exists") from error
    except OSError as error:
        raise PresentationDemoError("unable to write output") from error


__all__ = [
    "MAX_NEW_TOKENS",
    "PresentationDemoError",
    "PresentationRequest",
    "run_presentation_demo",
    "write_demo_json",
]

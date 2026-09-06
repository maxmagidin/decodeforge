#!/usr/bin/env python3
"""Exercise one real prepared TinyLlama q_proj through the release bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Final, NoReturn, cast

import torch
import torch.nn.functional as functional
from decodeforge.qproj_adapter import QProjAdapter, QProjAdapterError
from decodeforge.torch_bridge import RuntimeLibrary

N: Final = 2048
K: Final = 2048
LAYER: Final = 0
LAYER_NAME: Final = "model.layers.0.self_attn.q_proj"


class RealAdapterCheckError(RuntimeError):
    """The real one-layer G3.2 checkpoint failed."""


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RealAdapterCheckError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_bytes()), str(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealAdapterCheckError(f"unable to load {path}: {error}") from error


def _bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise RealAdapterCheckError(f"unable to read {path}: {error}") from error


def _identity(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RealAdapterCheckError(
            f"{field} mismatch: expected {expected!r}, got {actual!r}"
        )


def _expected_error(call: Any, reason: str) -> None:
    try:
        call()
    except (QProjAdapterError, RuntimeError):
        return
    raise RealAdapterCheckError(f"{reason} unexpectedly succeeded")


def _fault(*_args: Any) -> NoReturn:
    raise RealAdapterCheckError("injected native failure")


def check(library_path: Path, assets: Path, spec_path: Path) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        raise RealAdapterCheckError("the real G3.2 checkpoint requires Darwin arm64")

    spec = _load_json(spec_path)
    correctness = _object(spec.get("correctness"), "spec.correctness")
    direct = _object(correctness.get("direct_operator"), "direct_operator")
    atol = direct.get("absolute_tolerance")
    rtol = direct.get("relative_tolerance")
    if not isinstance(atol, int | float) or not isinstance(rtol, int | float):
        raise RealAdapterCheckError("direct operator tolerances must be numeric")

    inventory = _load_json(assets / "inventory.json")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or len(entries) != 22:
        raise RealAdapterCheckError(
            "prepared inventory must contain exactly 22 entries"
        )
    entry = _object(entries[LAYER], "inventory.entries[0]")
    _require_equal(entry.get("layer"), LAYER, "inventory layer")
    _require_equal(entry.get("directory"), "layers/00", "inventory directory")
    _require_equal(
        entry.get("tensor_name"), f"{LAYER_NAME}.weight", "inventory tensor name"
    )

    layer_root = assets / "layers" / "00"
    manifest_bytes = _bytes(layer_root / "manifest.json")
    manifest = _object(json.loads(manifest_bytes), "asset manifest")
    pack_manifest = _bytes(layer_root / "pack-manifest.json")
    packed = _bytes(layer_root / "weights.oi4.bin")
    fallback_bytes = _bytes(layer_root / "fallback.f32.bin")
    _require_equal(
        _identity(manifest_bytes), entry.get("manifest_identity"), "manifest"
    )
    _require_equal(
        _identity(fallback_bytes), entry.get("fallback_identity"), "fallback"
    )
    _require_equal(len(fallback_bytes), N * K * 4, "fallback bytes")

    fallback = torch.frombuffer(bytearray(fallback_bytes), dtype=torch.float32).reshape(
        N, K
    )
    fallback_manifest = _object(manifest.get("fallback"), "manifest.fallback")
    module_manifest = _object(manifest.get("module"), "manifest.module")
    bridge_image = _bytes(library_path)
    runtime = RuntimeLibrary(library_path, _identity(bridge_image))

    adapter = QProjAdapter(
        layer_name=LAYER_NAME,
        library=runtime,
        pack_manifest_json=pack_manifest,
        packed_weight=packed,
        fallback_weight=fallback,
        fallback_weight_id=cast(str, fallback_manifest.get("identity")),
        fallback_parent_packed_weight_id=cast(
            str, fallback_manifest.get("parent_packed_weight_identity")
        ),
        expected_module_id=cast(str, module_manifest.get("identity")),
    )
    try:
        x1 = torch.linspace(-1.0, 1.0, K, dtype=torch.float32).reshape(1, 1, K)
        reference1 = functional.linear(x1, adapter.same_q8_weight)
        actual1 = adapter(x1)
        torch.testing.assert_close(
            actual1, reference1, atol=float(atol), rtol=float(rtol), equal_nan=False
        )
        maximum_absolute_error = float((actual1 - reference1).abs().max())

        x2 = torch.stack((x1.reshape(-1), x1.reshape(-1).flip(0))).reshape(1, 2, K)
        reference2 = functional.linear(x2, adapter.same_q8_weight)
        actual2 = adapter(x2)
        if not torch.equal(actual2, reference2):
            raise RealAdapterCheckError("M>1 same-Q8 fallback is not bitwise equal")

        noncontiguous = torch.empty((1, 1, K * 2), dtype=torch.float32)[:, :, ::2]
        noncontiguous.copy_(x1)
        if noncontiguous.is_contiguous():
            raise RealAdapterCheckError("noncontiguous checkpoint input is contiguous")
        expected_noncontiguous = functional.linear(
            noncontiguous, adapter.same_q8_weight
        )
        if not torch.equal(adapter(noncontiguous), expected_noncontiguous):
            raise RealAdapterCheckError("noncontiguous fallback is not bitwise equal")

        gradient_input = x1.clone().requires_grad_(True)
        gradient_output = adapter(gradient_input)
        if not gradient_output.requires_grad:
            raise RealAdapterCheckError("gradient-enabled fallback lost autograd")

        _expected_error(lambda: adapter(x1.to(torch.float64)), "wrong dtype")
        _expected_error(lambda: adapter(x1[:, :, :-1]), "wrong shape")
        counters = adapter.counters
        if (
            counters.native_attempt != 1
            or counters.native_success != 1
            or counters.native_error != 0
            or counters.fallback_attempt != 5
            or counters.fallback_success != 3
            or counters.fallback_error != 2
            or counters.in_flight != 0
        ):
            raise RealAdapterCheckError(f"unexpected real-adapter counters: {counters}")
    finally:
        adapter.close()
    if not adapter.closed or adapter.counters.in_flight != 0:
        raise RealAdapterCheckError("real adapter did not close cleanly")
    _expected_error(lambda: adapter(x1), "post-close call")

    injected = QProjAdapter(
        layer_name=LAYER_NAME,
        library=runtime,
        pack_manifest_json=pack_manifest,
        packed_weight=packed,
        fallback_weight=fallback,
        fallback_weight_id=cast(str, fallback_manifest.get("identity")),
        fallback_parent_packed_weight_id=cast(
            str, fallback_manifest.get("parent_packed_weight_identity")
        ),
        expected_module_id=cast(str, module_manifest.get("identity")),
        native_operator=_fault,
    )
    try:
        _expected_error(lambda: injected(x1), "injected native failure")
        if injected.counters.native_error != 1:
            raise RealAdapterCheckError("injected native error was not counted")
    finally:
        injected.close()

    return {
        "layer": LAYER,
        "shape": [N, K],
        "maximum_absolute_error": maximum_absolute_error,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "native_success": 1,
        "fallback_success": 3,
        "fallback_error": 2,
        "injected_native_error": 1,
        "closed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--spec", type=Path, default=Path("benchmarks/g3/spec.json"))
    arguments = parser.parse_args()
    result = check(
        arguments.library.resolve(), arguments.assets.resolve(), arguments.spec
    )
    print(f"qproj-adapter-real: ok {json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

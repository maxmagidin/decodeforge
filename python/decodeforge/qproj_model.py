"""Transactional TinyLlama query-projection model integration.

This module is deliberately opt-in: :mod:`decodeforge` does not import it, or
PyTorch, from its base package.  Installation requires exclusive ownership of
an evaluation-only CPU/FP32 model.  While installed, the original query
projection weights are intentionally absent from ``state_dict``; cleanup
restores the exact original modules and state-dict surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, cast

import torch
from torch import nn

from .qproj_adapter import QProjCounters, QProjMetadata

TINYLLAMA_QPROJ_LAYERS: Final = 22
TINYLLAMA_QPROJ_N: Final = 2048
TINYLLAMA_QPROJ_K: Final = 2048
QPROJ_PACKED_BYTES: Final = 4_718_592
QPROJ_TOTAL_PACKED_BYTES: Final = TINYLLAMA_QPROJ_LAYERS * QPROJ_PACKED_BYTES
BRIDGE_MAX_AGGREGATE_PACKED_BYTES: Final = 2 * 1024 * 1024 * 1024
TINYLLAMA_QPROJ_MODULE_ID: Final = (
    "sha256:564dbd74857d3fe00b25bf4acbe6cf06d6ff47ab603ae9eb1ba3a530edc8ea44"
)

_IDENTITY_PREFIX: Final = "sha256:"
_MAX_INVENTORY_BYTES: Final = 256 * 1024
_MAX_ASSET_MANIFEST_BYTES: Final = 64 * 1024
_MAX_PACK_MANIFEST_BYTES: Final = 16 * 1024
_FALLBACK_BYTES: Final = TINYLLAMA_QPROJ_N * TINYLLAMA_QPROJ_K * 4
_TINYLLAMA_SOURCE: Final = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "filename": "model.safetensors",
    "bytes": 2_200_119_864,
    "identity": (
        "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
    ),
}


class QProjModelError(RuntimeError):
    """A model inventory, installation, or lifecycle contract violation."""


@dataclass(frozen=True)
class QProjAssetEntry:
    """Immutable identity record for one canonically prepared layer asset."""

    layer: int
    directory: str
    layer_path: str
    manifest_identity: str
    tensor_name: str
    tensor_identity: str
    logical_weight_identity: str
    packed_weight_identity: str
    packed_bytes: int
    module_identity: str
    fallback_weight_identity: str
    fallback_parent_logical_weight_identity: str
    fallback_parent_packed_weight_identity: str


@dataclass(frozen=True)
class QProjSource:
    """Immutable model source provenance included in the canonical inventory."""

    model_id: str
    revision: str
    filename: str
    bytes: int
    identity: str


@dataclass(frozen=True)
class QProjAssetInventory:
    """Ordered G3.3 input inventory, validated again before handle creation."""

    source: QProjSource
    entries: tuple[QProjAssetEntry, ...]
    total_packed_bytes: int
    total_fallback_bytes: int
    aggregate_identity: str


@dataclass(frozen=True)
class VerifiedQProjAsset:
    """One descriptor-stable byte/tensor snapshot from a verified directory."""

    entry: QProjAssetEntry
    pack_manifest_json: bytes
    packed_weight: bytes
    fallback_weight: torch.Tensor


@dataclass(frozen=True)
class QProjLayerCounters:
    """Immutable per-layer adapter counter snapshot."""

    layer_path: str
    forward: int
    native_attempt: int
    native_success: int
    native_error: int
    fallback_attempt: int
    fallback_success: int
    fallback_error: int
    predispatch_error: int
    rejected_closed: int
    in_flight: int
    closed: bool


@dataclass(frozen=True)
class QProjModelCounters:
    """Immutable installation-wide lifecycle and dispatch snapshot."""

    installed_modules: int
    restored_modules: int
    live_adapters: int
    in_flight: int
    layers: tuple[QProjLayerCounters, ...]
    closed: bool


class QProjAdapterLike(Protocol):
    """The owning-adapter surface required by transactional installation."""

    training: bool

    @property
    def metadata(self) -> QProjMetadata: ...

    @property
    def counters(self) -> QProjCounters: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...

    def eval(self) -> QProjAdapterLike: ...


AdapterFactory: TypeAlias = Callable[[VerifiedQProjAsset], QProjAdapterLike]


def tinyllama_qproj_paths() -> tuple[str, ...]:
    """Return the sole accepted ordered TinyLlama query-projection paths."""

    return tuple(
        f"model.layers.{layer}.self_attn.q_proj"
        for layer in range(TINYLLAMA_QPROJ_LAYERS)
    )


def _identity(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith(_IDENTITY_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise QProjModelError(f"{field} must be sha256:<64 lowercase hex digits>")
    return value


def _validate_inventory(inventory: QProjAssetInventory) -> None:
    if not isinstance(inventory, QProjAssetInventory):
        raise TypeError("inventory must be a QProjAssetInventory")
    expected_paths = tinyllama_qproj_paths()
    if len(inventory.entries) != TINYLLAMA_QPROJ_LAYERS:
        raise QProjModelError("q_proj inventory must contain exactly 22 entries")
    _identity(inventory.source.identity, "source identity")
    if _source_wire(inventory.source) != _TINYLLAMA_SOURCE:
        raise QProjModelError("inventory source is not the pinned TinyLlama checkpoint")

    seen_tensor_ids: set[str] = set()
    seen_logical_ids: set[str] = set()
    seen_pack_ids: set[str] = set()
    seen_fallback_ids: set[str] = set()
    module_id: str | None = None
    total = 0
    for layer, entry in enumerate(inventory.entries):
        expected_path = expected_paths[layer]
        expected_tensor = f"{expected_path}.weight"
        if (
            entry.layer != layer
            or entry.directory != f"layers/{layer:02}"
            or entry.layer_path != expected_path
            or entry.tensor_name != expected_tensor
        ):
            raise QProjModelError(
                "q_proj inventory entries must be in exact TinyLlama layer order"
            )
        tensor_id = _identity(entry.tensor_identity, "tensor_identity")
        logical_id = _identity(entry.logical_weight_identity, "logical_weight_identity")
        pack_id = _identity(entry.packed_weight_identity, "packed_weight_identity")
        fallback_id = _identity(
            entry.fallback_weight_identity, "fallback_weight_identity"
        )
        current_module_id = _identity(entry.module_identity, "module_identity")
        fallback_logical_id = _identity(
            entry.fallback_parent_logical_weight_identity,
            "fallback_parent_logical_weight_identity",
        )
        fallback_pack_id = _identity(
            entry.fallback_parent_packed_weight_identity,
            "fallback_parent_packed_weight_identity",
        )
        if fallback_logical_id != logical_id or fallback_pack_id != pack_id:
            raise QProjModelError(
                f"{expected_path} fallback parent identities do not match its asset"
            )
        if module_id is None:
            module_id = current_module_id
        elif current_module_id != module_id:
            raise QProjModelError(
                "all [2048,2048] q_proj assets must share one module identity"
            )
        if current_module_id != TINYLLAMA_QPROJ_MODULE_ID:
            raise QProjModelError("q_proj asset has a noncanonical module identity")
        for identity, identities, label in (
            (tensor_id, seen_tensor_ids, "tensor"),
            (logical_id, seen_logical_ids, "logical weight"),
            (pack_id, seen_pack_ids, "packed weight"),
            (fallback_id, seen_fallback_ids, "fallback weight"),
        ):
            if identity in identities:
                raise QProjModelError(f"duplicate {label} identity in q_proj inventory")
            identities.add(identity)
        if entry.packed_bytes != QPROJ_PACKED_BYTES:
            raise QProjModelError(
                f"{expected_path} packed extent must be {QPROJ_PACKED_BYTES} bytes"
            )
        total += entry.packed_bytes

    if total != inventory.total_packed_bytes or total != QPROJ_TOTAL_PACKED_BYTES:
        raise QProjModelError("q_proj aggregate packed-byte accounting is inconsistent")
    if inventory.total_fallback_bytes != TINYLLAMA_QPROJ_LAYERS * _FALLBACK_BYTES:
        raise QProjModelError(
            "q_proj aggregate fallback-byte accounting is inconsistent"
        )
    if total > BRIDGE_MAX_AGGREGATE_PACKED_BYTES:
        raise QProjModelError("q_proj assets exceed the bridge aggregate packed quota")
    aggregate_id = _identity(inventory.aggregate_identity, "aggregate identity")
    if aggregate_id != _inventory_identity(inventory):
        raise QProjModelError("q_proj aggregate identity does not match its entries")


def _entry_wire(entry: QProjAssetEntry) -> dict[str, object]:
    return {
        "layer": entry.layer,
        "directory": entry.directory,
        "manifest_identity": entry.manifest_identity,
        "tensor_name": entry.tensor_name,
        "tensor_identity": entry.tensor_identity,
        "logical_weight_identity": entry.logical_weight_identity,
        "packed_weight_identity": entry.packed_weight_identity,
        "packed_bytes": entry.packed_bytes,
        "module_identity": entry.module_identity,
        "fallback_identity": entry.fallback_weight_identity,
        "fallback_bytes": _FALLBACK_BYTES,
    }


def _source_wire(source: QProjSource) -> dict[str, object]:
    return {
        "model_id": source.model_id,
        "revision": source.revision,
        "filename": source.filename,
        "bytes": source.bytes,
        "identity": source.identity,
    }


def _inventory_identity(inventory: QProjAssetInventory) -> str:
    preimage = {
        "schema_version": 1,
        "format": "decodeforge_q_proj_inventory_v1",
        "source": _source_wire(inventory.source),
        "layer_count": TINYLLAMA_QPROJ_LAYERS,
        "entries": [_entry_wire(entry) for entry in inventory.entries],
        "total_packed_bytes": inventory.total_packed_bytes,
        "total_fallback_bytes": inventory.total_fallback_bytes,
    }
    encoded = json.dumps(preimage, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(b"DecodeForge/q-proj-inventory/v1\0" + encoded).hexdigest()
    return f"sha256:{digest}"


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QProjModelError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_pairs_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QProjModelError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise QProjModelError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QProjModelError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise QProjModelError(f"{label} has unsupported or missing fields")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise QProjModelError(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QProjModelError(f"{label} must be an integer")
    return cast(int, value)


def _plain_directory(path: Path, expected: set[str], label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise QProjModelError(f"unable to inspect {label}") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise QProjModelError(f"{label} must be a plain directory")
    try:
        actual = {entry.name for entry in os.scandir(path)}
    except OSError as error:
        raise QProjModelError(f"unable to enumerate {label}") from error
    if actual != expected:
        raise QProjModelError(f"{label} has missing or unexpected entries")


def _read_snapshot(
    path: Path, maximum: int, label: str, exact: int | None = None
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise QProjModelError(
            f"unable to open {label} without following links"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QProjModelError(f"{label} must be a regular file")
        if before.st_size <= 0 or before.st_size > maximum:
            raise QProjModelError(f"{label} byte extent is outside its bound")
        if exact is not None and before.st_size != exact:
            raise QProjModelError(f"{label} byte extent does not match its manifest")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise QProjModelError(f"{label} changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise QProjModelError(f"{label} grew during read")
        after = os.fstat(descriptor)

        def fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if fingerprint(before) != fingerprint(after):
            raise QProjModelError(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_pack_manifest(
    raw: bytes, payload: bytes, logical_id: str, packed_id: str
) -> None:
    manifest = _json(raw, "pack manifest")
    _exact_fields(
        manifest,
        {
            "schema_version",
            "spec",
            "shape",
            "logical_weight_identity",
            "packed_identity",
            "payload_bytes",
        },
        "pack manifest",
    )
    spec = _object(manifest["spec"], "pack spec")
    shape = _object(manifest["shape"], "pack shape")
    _exact_fields(
        spec,
        {
            "schema_version",
            "format",
            "layout",
            "tile",
            "block_size",
            "record_bytes",
            "alignment",
        },
        "pack spec",
    )
    _exact_fields(shape, {"m", "n", "k"}, "pack shape")
    if manifest["schema_version"] != 1 or spec != {
        "schema_version": 1,
        "format": "DFQ8_B32_OI4_V1",
        "layout": "output-interleaved",
        "tile": 4,
        "block_size": 32,
        "record_bytes": 144,
        "alignment": 16,
    }:
        raise QProjModelError("pack manifest uses an unsupported format")
    if shape != {"m": 1, "n": TINYLLAMA_QPROJ_N, "k": TINYLLAMA_QPROJ_K}:
        raise QProjModelError("pack manifest shape is not [1,2048,2048]")
    if (
        manifest["logical_weight_identity"] != logical_id
        or manifest["packed_identity"] != packed_id
        or manifest["payload_bytes"] != QPROJ_PACKED_BYTES
    ):
        raise QProjModelError(
            "pack manifest identities or extent do not match the asset"
        )
    hasher = hashlib.sha256()
    hasher.update(b"DecodeForge/DFQ8_B32_OI4_V1/packed-weight/v1\0")
    hasher.update(logical_id.encode("ascii"))
    for value in (2048, 2048, 64, 512, 4, 32, 144, 16):
        hasher.update(value.to_bytes(4, "little"))
    hasher.update(b"output-interleaved\0")
    hasher.update(payload)
    if f"sha256:{hasher.hexdigest()}" != packed_id:
        raise QProjModelError("packed identity does not match the snapshotted payload")


def _verify_logical_and_fallback(
    payload: bytes, fallback: bytes, logical_id: str
) -> torch.Tensor:
    """Independently reorder OI4 and verify logical Q8 plus dequantization."""

    records = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
        512, 64, 144
    )
    scale_bytes = records[..., :16].contiguous()
    scales = scale_bytes.view(torch.float32).reshape(512, 64, 4)
    scale_bits = scale_bytes.view(torch.int32).reshape(512, 64, 4)
    sign_bit = torch.tensor(-2_147_483_648, dtype=torch.int32)
    if not bool(torch.isfinite(scales).all().item()) or bool(
        ((scale_bits & sign_bit) != 0).any().item()
    ):
        raise QProjModelError("packed scales must be finite and non-negative")

    interleaved_q = records[..., 16:].view(torch.int8).reshape(512, 64, 32, 4)
    if bool((interleaved_q == -128).any().item()):
        raise QProjModelError("packed q contains forbidden -128")
    if bool(((scales == 0).unsqueeze(2) & (interleaved_q != 0)).any().item()):
        raise QProjModelError("zero-scale packed lanes must contain only zero q")

    logical_q = interleaved_q.permute(0, 3, 1, 2).reshape(2048, 2048).contiguous()
    logical_scales = scales.permute(0, 2, 1).reshape(2048, 64).contiguous()
    hasher = hashlib.sha256()
    hasher.update(b"DecodeForge/DFQ8_B32_V1/logical-weight/v1\0")
    for value in (2048, 2048, 64):
        hasher.update(value.to_bytes(4, "little"))
    hasher.update(logical_q.numpy().tobytes(order="C"))
    hasher.update(logical_scales.numpy().tobytes(order="C"))
    if f"sha256:{hasher.hexdigest()}" != logical_id:
        raise QProjModelError("logical identity does not match packed q/scales")

    reconstructed = (
        logical_q.reshape(2048, 64, 32).to(torch.float32) * logical_scales.unsqueeze(2)
    ).reshape(2048, 2048)
    fallback_tensor = torch.frombuffer(
        bytearray(fallback), dtype=torch.float32
    ).reshape(2048, 2048)
    if not torch.equal(
        reconstructed.view(torch.uint8), fallback_tensor.view(torch.uint8)
    ):
        raise QProjModelError("fallback is not the exact dequantization of packed Q8")
    return fallback_tensor


def _load_prepared_inventory(
    prepared_directory: str | os.PathLike[str],
) -> tuple[QProjAssetInventory, tuple[VerifiedQProjAsset, ...]]:
    root = Path(prepared_directory)
    _plain_directory(root, {"inventory.json", "layers"}, "prepared inventory")
    layers = root / "layers"
    _plain_directory(
        layers,
        {f"{layer:02}" for layer in range(TINYLLAMA_QPROJ_LAYERS)},
        "prepared layers",
    )
    inventory_raw = _read_snapshot(
        root / "inventory.json", _MAX_INVENTORY_BYTES, "inventory.json"
    )
    wire = _json(inventory_raw, "q_proj inventory")
    _exact_fields(
        wire,
        {
            "schema_version",
            "format",
            "source",
            "layer_count",
            "entries",
            "total_packed_bytes",
            "total_fallback_bytes",
            "aggregate_identity",
        },
        "q_proj inventory",
    )
    if (
        wire["schema_version"] != 1
        or wire["format"] != "decodeforge_q_proj_inventory_v1"
        or wire["layer_count"] != TINYLLAMA_QPROJ_LAYERS
    ):
        raise QProjModelError("unsupported q_proj inventory contract")
    source_wire = _object(wire["source"], "inventory source")
    _exact_fields(
        source_wire,
        {"model_id", "revision", "filename", "bytes", "identity"},
        "inventory source",
    )
    source = QProjSource(
        model_id=_string(source_wire["model_id"], "source model_id"),
        revision=_string(source_wire["revision"], "source revision"),
        filename=_string(source_wire["filename"], "source filename"),
        bytes=_integer(source_wire["bytes"], "source bytes"),
        identity=_string(source_wire["identity"], "source identity"),
    )
    entries_wire = wire["entries"]
    if (
        not isinstance(entries_wire, list)
        or len(entries_wire) != TINYLLAMA_QPROJ_LAYERS
    ):
        raise QProjModelError("inventory must contain exactly 22 ordered entries")

    entries: list[QProjAssetEntry] = []
    assets: list[VerifiedQProjAsset] = []
    for layer, raw_entry in enumerate(entries_wire):
        inventory_entry = _object(raw_entry, "inventory entry")
        _exact_fields(
            inventory_entry,
            {
                "layer",
                "directory",
                "manifest_identity",
                "tensor_name",
                "tensor_identity",
                "logical_weight_identity",
                "packed_weight_identity",
                "packed_bytes",
                "module_identity",
                "fallback_identity",
                "fallback_bytes",
            },
            "inventory entry",
        )
        directory_name = _string(inventory_entry["directory"], "entry directory")
        if directory_name != f"layers/{layer:02}":
            raise QProjModelError("inventory directory order is not canonical")
        layer_directory = root / directory_name
        _plain_directory(
            layer_directory,
            {
                "manifest.json",
                "pack-manifest.json",
                "weights.oi4.bin",
                "fallback.f32.bin",
            },
            f"layer {layer} asset",
        )
        manifest_raw = _read_snapshot(
            layer_directory / "manifest.json",
            _MAX_ASSET_MANIFEST_BYTES,
            f"layer {layer} manifest",
        )
        if _sha256(manifest_raw) != inventory_entry["manifest_identity"]:
            raise QProjModelError(f"layer {layer} manifest identity mismatch")
        manifest = _json(manifest_raw, f"layer {layer} manifest")
        _exact_fields(
            manifest,
            {
                "schema_version",
                "format",
                "operator",
                "source",
                "tensor",
                "quantization",
                "pack",
                "module",
                "fallback",
                "tool",
            },
            "asset manifest",
        )
        tensor = _object(manifest["tensor"], "asset tensor")
        quantization = _object(manifest["quantization"], "asset quantization")
        pack = _object(manifest["pack"], "asset pack")
        module = _object(manifest["module"], "asset module")
        fallback = _object(manifest["fallback"], "asset fallback")
        tool = _object(manifest["tool"], "asset tool")
        _exact_fields(
            tensor,
            {"name", "dtype", "shape", "data_bytes", "data_identity"},
            "asset tensor",
        )
        _exact_fields(
            quantization,
            {"format", "numeric_mode", "logical_weight_identity"},
            "asset quantization",
        )
        _exact_fields(
            pack,
            {
                "format",
                "manifest_file",
                "manifest_bytes",
                "manifest_identity",
                "payload_file",
                "payload_bytes",
                "payload_identity",
                "packed_weight_identity",
            },
            "asset pack",
        )
        _exact_fields(module, {"variant", "source_format", "identity"}, "asset module")
        _exact_fields(
            fallback,
            {
                "format",
                "dtype",
                "file",
                "bytes",
                "identity",
                "parent_logical_weight_identity",
                "parent_packed_weight_identity",
            },
            "asset fallback",
        )
        _exact_fields(tool, {"name", "version"}, "asset tool")
        expected_path = tinyllama_qproj_paths()[layer]
        expected_tensor = f"{expected_path}.weight"
        dtype = tensor["dtype"]
        expected_tensor_bytes = 2048 * 2048 * (2 if dtype == "BF16" else 4)
        if (
            manifest["schema_version"] != 1
            or manifest["format"] != "decodeforge_q8_linear_asset_v1"
            or manifest["operator"] != "Q8Linear"
            or manifest["source"] != source_wire
            or tensor["name"] != expected_tensor
            or dtype not in {"BF16", "F32"}
            or tensor["shape"] != [2048, 2048]
            or tensor["data_bytes"] != expected_tensor_bytes
            or quantization["format"] != "DFQ8_B32_V1"
            or quantization["numeric_mode"] != "strict_f32_v1"
            or pack["format"] != "DFQ8_B32_OI4_V1"
            or pack["manifest_file"] != "pack-manifest.json"
            or pack["payload_file"] != "weights.oi4.bin"
            or pack["payload_bytes"] != QPROJ_PACKED_BYTES
            or module["variant"] != "neon"
            or module["source_format"] != "decodeforge_neon_c_v1"
            or fallback["format"] != "decodeforge_row_major_f32_le_v1"
            or fallback["dtype"] != "F32"
            or fallback["file"] != "fallback.f32.bin"
            or fallback["bytes"] != _FALLBACK_BYTES
            or tool != {"name": "decodeforge-compiler", "version": "0.1.0"}
        ):
            raise QProjModelError(f"layer {layer} asset manifest contract mismatch")
        pack_manifest = _read_snapshot(
            layer_directory / "pack-manifest.json",
            _MAX_PACK_MANIFEST_BYTES,
            f"layer {layer} pack manifest",
            _integer(pack["manifest_bytes"], "pack manifest bytes"),
        )
        payload = _read_snapshot(
            layer_directory / "weights.oi4.bin",
            QPROJ_PACKED_BYTES,
            f"layer {layer} packed payload",
            QPROJ_PACKED_BYTES,
        )
        fallback_bytes = _read_snapshot(
            layer_directory / "fallback.f32.bin",
            _FALLBACK_BYTES,
            f"layer {layer} fallback",
            _FALLBACK_BYTES,
        )
        for actual, declared, label in (
            (_sha256(pack_manifest), pack["manifest_identity"], "pack manifest"),
            (_sha256(payload), pack["payload_identity"], "pack payload"),
            (_sha256(fallback_bytes), fallback["identity"], "fallback"),
        ):
            if actual != declared:
                raise QProjModelError(f"layer {layer} {label} identity mismatch")
        logical_id = _string(
            quantization["logical_weight_identity"], "logical identity"
        )
        packed_id = _string(pack["packed_weight_identity"], "packed identity")
        _validate_pack_manifest(pack_manifest, payload, logical_id, packed_id)
        fallback_tensor = _verify_logical_and_fallback(
            payload, fallback_bytes, logical_id
        )
        entry = QProjAssetEntry(
            layer=layer,
            directory=directory_name,
            layer_path=expected_path,
            manifest_identity=_string(
                inventory_entry["manifest_identity"], "manifest identity"
            ),
            tensor_name=expected_tensor,
            tensor_identity=_string(tensor["data_identity"], "tensor identity"),
            logical_weight_identity=logical_id,
            packed_weight_identity=packed_id,
            packed_bytes=QPROJ_PACKED_BYTES,
            module_identity=_string(module["identity"], "module identity"),
            fallback_weight_identity=_string(fallback["identity"], "fallback identity"),
            fallback_parent_logical_weight_identity=_string(
                fallback["parent_logical_weight_identity"], "fallback logical parent"
            ),
            fallback_parent_packed_weight_identity=_string(
                fallback["parent_packed_weight_identity"], "fallback packed parent"
            ),
        )
        comparisons = {
            "layer": entry.layer,
            "directory": entry.directory,
            "manifest_identity": entry.manifest_identity,
            "tensor_name": entry.tensor_name,
            "tensor_identity": entry.tensor_identity,
            "logical_weight_identity": entry.logical_weight_identity,
            "packed_weight_identity": entry.packed_weight_identity,
            "packed_bytes": entry.packed_bytes,
            "module_identity": entry.module_identity,
            "fallback_identity": entry.fallback_weight_identity,
            "fallback_bytes": _FALLBACK_BYTES,
        }
        if comparisons != inventory_entry:
            raise QProjModelError(
                f"layer {layer} manifest does not match inventory entry"
            )
        entries.append(entry)
        assets.append(
            VerifiedQProjAsset(entry, pack_manifest, payload, fallback_tensor)
        )

    inventory = QProjAssetInventory(
        source=source,
        entries=tuple(entries),
        total_packed_bytes=_integer(wire["total_packed_bytes"], "total packed bytes"),
        total_fallback_bytes=_integer(
            wire["total_fallback_bytes"], "total fallback bytes"
        ),
        aggregate_identity=_string(wire["aggregate_identity"], "aggregate identity"),
    )
    _validate_inventory(inventory)
    return inventory, tuple(assets)


def _resolve_target(model: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    parent_path, _, child_name = path.rpartition(".")
    try:
        parent = model.get_submodule(parent_path)
        child = getattr(parent, child_name)
    except (AttributeError, IndexError, KeyError) as error:
        raise QProjModelError(f"missing required q_proj module {path}") from error
    if not isinstance(child, nn.Module):
        raise QProjModelError(f"required q_proj path {path} is not an nn.Module")
    return parent, child_name, child


def _validate_model(model: nn.Module) -> list[tuple[nn.Module, str, nn.Linear]]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if model.training:
        raise QProjModelError("q_proj installation requires model.eval()")

    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    seen_modules: set[int] = set()
    for path in tinyllama_qproj_paths():
        parent, child_name, module = _resolve_target(model, path)
        if type(module) is not nn.Linear:
            raise QProjModelError(f"{path} must be exactly torch.nn.Linear")
        linear = module
        if id(linear) in seen_modules:
            raise QProjModelError(
                "duplicate q_proj module object across required paths"
            )
        seen_modules.add(id(linear))
        if (
            linear.in_features != TINYLLAMA_QPROJ_K
            or linear.out_features != TINYLLAMA_QPROJ_N
        ):
            raise QProjModelError(f"{path} must declare [2048,2048]")
        if tuple(int(value) for value in linear.weight.shape) != (
            TINYLLAMA_QPROJ_N,
            TINYLLAMA_QPROJ_K,
        ):
            raise QProjModelError(f"{path} weight must have shape [2048,2048]")
        if linear.bias is not None:
            raise QProjModelError(f"{path} must be bias-free")
        if linear.weight.requires_grad:
            raise QProjModelError(f"{path} weight must be frozen")
        if (
            linear.weight.device.type != "cpu"
            or linear.weight.dtype is not torch.float32
        ):
            raise QProjModelError(f"{path} must be CPU FP32")
        if linear.training:
            raise QProjModelError(f"{path} must be in evaluation mode")
        targets.append((parent, child_name, linear))
    return targets


def _adapter_counter_snapshot(
    path: str, adapter: QProjAdapterLike
) -> QProjLayerCounters:
    counters = adapter.counters
    counters.validate()
    return QProjLayerCounters(
        layer_path=path,
        forward=counters.forward,
        native_attempt=counters.native_attempt,
        native_success=counters.native_success,
        native_error=counters.native_error,
        fallback_attempt=counters.fallback_attempt,
        fallback_success=counters.fallback_success,
        fallback_error=counters.fallback_error,
        predispatch_error=int(getattr(counters, "predispatch_error", 0)),
        rejected_closed=counters.rejected_closed,
        in_flight=counters.in_flight,
        closed=counters.closed,
    )


def _validate_adapter(adapter: QProjAdapterLike, entry: QProjAssetEntry) -> None:
    if not isinstance(adapter, nn.Module):
        raise QProjModelError("adapter factory must return a torch.nn.Module")
    if adapter.closed:
        raise QProjModelError("adapter factory returned a closed adapter")
    metadata = adapter.metadata
    expected = (
        entry.layer_path,
        TINYLLAMA_QPROJ_N,
        TINYLLAMA_QPROJ_K,
        entry.module_identity,
        entry.packed_weight_identity,
        entry.packed_bytes,
        entry.fallback_weight_identity,
    )
    actual = (
        metadata.layer_name,
        metadata.n,
        metadata.k,
        metadata.module_id,
        metadata.packed_weight_id,
        metadata.packed_weight_bytes,
        metadata.fallback_weight_id,
    )
    if actual != expected:
        raise QProjModelError(
            f"adapter metadata does not match ordered asset {entry.layer_path}"
        )
    if adapter.state_dict():
        raise QProjModelError(
            "q_proj adapters must not persist source or fallback weights in state_dict"
        )
    for value in (*adapter.parameters(), *adapter.buffers()):
        if value.device.type != "cpu" or (
            value.is_floating_point() and value.dtype is not torch.float32
        ):
            raise QProjModelError("q_proj adapter tensors must remain CPU FP32")
    adapter.eval()


class _InstalledQProj(nn.Module):
    """Reject stateful module operations while delegating inference calls."""

    def __init__(self, adapter: QProjAdapterLike) -> None:
        super().__init__()
        self.add_module("_decodeforge_adapter", cast(nn.Module, adapter))
        self.training = False

    @property
    def adapter(self) -> QProjAdapterLike:
        return cast(QProjAdapterLike, self._decodeforge_adapter)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.adapter(value))  # type: ignore[operator]

    def train(self, mode: bool = True) -> _InstalledQProj:
        if mode:
            raise QProjModelError("training is unsupported while q_proj is installed")
        super().train(False)
        return self

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> nn.Module:
        del fn, recurse
        raise QProjModelError(
            "device or dtype migration is unsupported while q_proj is installed"
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        del state_dict, local_metadata, strict, missing_keys, unexpected_keys
        error_msgs.append(f"{prefix}cannot load state while q_proj is installed")


@dataclass
class _InstalledRecord:
    entry: QProjAssetEntry
    parent: nn.Module
    child_name: str
    original: nn.Linear
    wrapper: _InstalledQProj
    restored: bool = False


@dataclass(frozen=True)
class _MethodState:
    name: str
    existed: bool
    value: object | None


class QProjModelInstallation:
    """Own all 22 replacements and restore/close them deterministically."""

    def __init__(
        self,
        model: nn.Module,
        inventory: QProjAssetInventory,
        records: Sequence[_InstalledRecord],
    ) -> None:
        self._model = model
        self._inventory = inventory
        self._records = tuple(records)
        self._lock = threading.RLock()
        self._method_states = self._install_model_guards()

    @property
    def inventory(self) -> QProjAssetInventory:
        return self._inventory

    @property
    def closed(self) -> bool:
        with self._lock:
            return all(
                record.restored and record.wrapper.adapter.closed
                for record in self._records
            )

    @property
    def counters(self) -> QProjModelCounters:
        with self._lock:
            layers = tuple(
                _adapter_counter_snapshot(
                    record.entry.layer_path, record.wrapper.adapter
                )
                for record in self._records
            )
            installed = sum(
                not record.restored
                and getattr(record.parent, record.child_name, None) is record.wrapper
                for record in self._records
            )
            restored = sum(record.restored for record in self._records)
            live = sum(not layer.closed for layer in layers)
            in_flight = sum(layer.in_flight for layer in layers)
            return QProjModelCounters(
                installed_modules=installed,
                restored_modules=restored,
                live_adapters=live,
                in_flight=in_flight,
                layers=layers,
                closed=restored == TINYLLAMA_QPROJ_LAYERS and live == 0,
            )

    def _install_model_guards(self) -> tuple[_MethodState, ...]:
        model = self._model
        states = tuple(
            _MethodState(name, name in model.__dict__, model.__dict__.get(name))
            for name in ("train", "_apply", "load_state_dict")
        )
        original_train = model.train

        def guarded_train(_model: nn.Module, mode: bool = True) -> nn.Module:
            if mode:
                raise QProjModelError(
                    "training is unsupported while q_proj is installed"
                )
            return original_train(False)

        def guarded_apply(
            _model: nn.Module, _fn: object, *args: object, **kwargs: object
        ) -> nn.Module:
            del args, kwargs
            raise QProjModelError(
                "device or dtype migration is unsupported while q_proj is installed"
            )

        def guarded_load(
            _model: nn.Module, *_args: object, **_kwargs: object
        ) -> object:
            raise QProjModelError(
                "state loading is unsupported while q_proj is installed"
            )

        try:
            model.train = types.MethodType(guarded_train, model)  # type: ignore[method-assign]
            model._apply = types.MethodType(guarded_apply, model)  # type: ignore[method-assign]
            model.load_state_dict = types.MethodType(guarded_load, model)  # type: ignore[method-assign]
        except Exception:
            for state in states:
                if state.existed:
                    setattr(model, state.name, state.value)
                else:
                    model.__dict__.pop(state.name, None)
            raise
        return states

    def _restore_model_guards(self) -> None:
        for state in self._method_states:
            if state.existed:
                setattr(self._model, state.name, state.value)
            else:
                self._model.__dict__.pop(state.name, None)

    def close(self) -> None:
        failures: list[Exception] = []
        with self._lock:
            for record in self._records:
                if record.restored:
                    continue
                current = getattr(record.parent, record.child_name, None)
                if current is record.original:
                    record.restored = True
                    continue
                if current is not record.wrapper:
                    failures.append(
                        QProjModelError(
                            f"{record.entry.layer_path} changed outside "
                            "its installation"
                        )
                    )
                    continue
                try:
                    setattr(record.parent, record.child_name, record.original)
                    record.restored = True
                except Exception as error:
                    error.add_note(f"while restoring {record.entry.layer_path}")
                    failures.append(error)

            for record in self._records:
                adapter = record.wrapper.adapter
                if adapter.closed:
                    continue
                try:
                    adapter.close()
                except Exception as error:
                    error.add_note(f"while closing {record.entry.layer_path}")
                    failures.append(error)

            if all(
                record.restored and record.wrapper.adapter.closed
                for record in self._records
            ):
                self._restore_model_guards()
        if failures:
            raise ExceptionGroup("q_proj cleanup did not complete", failures)

    def __enter__(self) -> QProjModelInstallation:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _close_created(
    created: Sequence[tuple[QProjAssetEntry, QProjAdapterLike]],
) -> list[Exception]:
    failures: list[Exception] = []
    for entry, adapter in created:
        if adapter.closed:
            continue
        try:
            adapter.close()
        except Exception as error:
            error.add_note(f"while rolling back {entry.layer_path}")
            failures.append(error)
    return failures


def _close_failed_factory_exception(error: Exception) -> list[Exception]:
    """Retry cleanup owned by QProjAdapterInitializationError without importing it."""

    close = getattr(error, "close", None)
    closed = getattr(error, "closed", True)
    if not callable(close) or bool(closed):
        return []
    try:
        close()
    except Exception as cleanup_error:
        cleanup_error.add_note("while retrying failed adapter initialization cleanup")
        return [cleanup_error]
    return []


def install_tinyllama_qproj(
    model: nn.Module,
    prepared_directory: str | os.PathLike[str],
    adapter_factory: AdapterFactory,
) -> QProjModelInstallation:
    """Validate, create, then atomically install all 22 owning adapters.

    Callers must not concurrently mutate or execute ``model`` during setup or
    cleanup.  Inventory and all original modules are checked before the first
    native handle is created; every adapter is created and checked before the
    first model mutation.
    """

    inventory, assets = _load_prepared_inventory(prepared_directory)
    _validate_inventory(inventory)
    if tuple(asset.entry for asset in assets) != inventory.entries:
        raise QProjModelError("verified asset snapshots do not match the inventory")
    targets = _validate_model(model)
    original_state_keys = tuple(model.state_dict().keys())

    created: list[tuple[QProjAssetEntry, QProjAdapterLike]] = []
    try:
        seen_adapters: set[int] = set()
        for asset in assets:
            entry = asset.entry
            adapter = adapter_factory(asset)
            if id(adapter) in seen_adapters:
                raise QProjModelError(
                    "adapter factory returned one adapter more than once"
                )
            seen_adapters.add(id(adapter))
            created.append((entry, adapter))
            _validate_adapter(adapter, entry)
    except Exception as error:
        failures = [
            error,
            *_close_failed_factory_exception(error),
            *_close_created(created),
        ]
        if len(failures) == 1:
            raise
        raise ExceptionGroup(
            "q_proj adapter creation rollback failed", failures
        ) from error

    try:
        records = [
            _InstalledRecord(
                entry=entry,
                parent=parent,
                child_name=child_name,
                original=original,
                wrapper=_InstalledQProj(adapter),
            )
            for (parent, child_name, original), (entry, adapter) in zip(
                targets, created, strict=True
            )
        ]
    except Exception as error:
        failures = [error, *_close_created(created)]
        if len(failures) == 1:
            raise
        raise ExceptionGroup(
            "q_proj adapter wrapping rollback failed", failures
        ) from error

    installed: list[_InstalledRecord] = []
    try:
        for record in records:
            if getattr(record.parent, record.child_name) is not record.original:
                raise QProjModelError(
                    f"{record.entry.layer_path} changed after model preflight"
                )
            setattr(record.parent, record.child_name, record.wrapper)
            installed.append(record)
        installed_state_keys = tuple(model.state_dict().keys())
        removed = {f"{path}.weight" for path in tinyllama_qproj_paths()}
        expected_state_keys = tuple(
            key for key in original_state_keys if key not in removed
        )
        if installed_state_keys != expected_state_keys:
            raise QProjModelError(
                "installed state_dict must differ only by the 22 q_proj weights"
            )
        return QProjModelInstallation(model, inventory, records)
    except Exception as error:
        rollback_failures: list[Exception] = [error]
        for record in reversed(installed):
            try:
                setattr(record.parent, record.child_name, record.original)
                record.restored = True
            except Exception as restore_error:
                restore_error.add_note(f"while rolling back {record.entry.layer_path}")
                rollback_failures.append(restore_error)
        rollback_failures.extend(_close_created(created))
        if len(rollback_failures) == 1:
            raise
        raise ExceptionGroup(
            "q_proj installation rollback failed", rollback_failures
        ) from error

    raise AssertionError("unreachable")


__all__ = [
    "BRIDGE_MAX_AGGREGATE_PACKED_BYTES",
    "QPROJ_PACKED_BYTES",
    "QPROJ_TOTAL_PACKED_BYTES",
    "TINYLLAMA_QPROJ_K",
    "TINYLLAMA_QPROJ_LAYERS",
    "TINYLLAMA_QPROJ_MODULE_ID",
    "TINYLLAMA_QPROJ_N",
    "AdapterFactory",
    "QProjAssetEntry",
    "QProjAssetInventory",
    "QProjLayerCounters",
    "QProjModelCounters",
    "QProjModelError",
    "QProjModelInstallation",
    "QProjSource",
    "VerifiedQProjAsset",
    "install_tinyllama_qproj",
    "tinyllama_qproj_paths",
]

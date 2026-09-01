#!/usr/bin/env python3
"""Exercise the actual release DecodeForge bridge through its public host path.

The Rust example on stdin produces a transport envelope from the checked-in
``exhaustive-q8`` fixture.  This checker intentionally reconstructs no pack:
it decodes the envelope and lends the exact manifest/payload to the release
cdylib.  Darwin arm64 must traverse the verified Python runtime, binding
registry, eager Torch operator, and guarded callable.  Linux checks the raw
unsupported-host ABI boundary without installing Torch.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import platform
import re
import struct
import sys
from pathlib import Path
from typing import Any, Final, NoReturn, cast

BRIDGE_ABI_VERSION: Final = 1
MAX_MANIFEST_BYTES: Final = 16 * 1024
EXPECTED_PAYLOAD_BYTES: Final = 9_216
STATUS_OK: Final = 0
STATUS_TRUNCATED: Final = 1
STATUS_UNSUPPORTED_HOST: Final = 10
MAX_ERROR_BYTES: Final = 4_096
IDENTITY_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")


class BridgeCheckError(RuntimeError):
    """A deterministic fixture or bridge-check failure."""


_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_CHAR_POINTER = ctypes.POINTER(ctypes.c_char)
_SIZE_POINTER = ctypes.POINTER(ctypes.c_size_t)
_HANDLE_POINTER = ctypes.POINTER(ctypes.c_uint64)


def _parse_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value}")


def _identity(value: Any, field: str) -> str:
    if not isinstance(value, str) or IDENTITY_PATTERN.fullmatch(value) is None:
        raise BridgeCheckError(f"{field} is not a lowercase SHA-256 identity")
    return value


def _integer_list(value: Any, expected_length: int, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise BridgeCheckError(f"{field} must contain exactly {expected_length} words")
    words: list[int] = []
    for index, word in enumerate(value):
        if (
            isinstance(word, bool)
            or not isinstance(word, int)
            or not 0 <= word <= 0xFFFFFFFF
        ):
            raise BridgeCheckError(f"{field}[{index}] is not a u32 bit word")
        words.append(word)
    return words


def _envelope_from_stdin() -> dict[str, Any]:
    raw = sys.stdin.buffer.read()
    try:
        value = json.loads(raw, parse_constant=_parse_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise BridgeCheckError(
            f"fixture envelope is not valid JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise BridgeCheckError("fixture envelope root must be an object")
    expected_keys = {
        "case_id",
        "n",
        "k",
        "module_id",
        "packed_weight_id",
        "pack_manifest_json",
        "packed_weight_hex",
        "input_fp32_bits",
        "expected_output_fp32_bits",
    }
    if set(value) != expected_keys:
        raise BridgeCheckError("fixture envelope keys are not the closed contract")
    if value["case_id"] != "exhaustive-q8":
        raise BridgeCheckError("fixture case_id is not exhaustive-q8")
    if value["n"] != 255 or value["k"] != 2:
        raise BridgeCheckError("fixture shape must be N=255,K=2")
    _identity(value["module_id"], "module_id")
    _identity(value["packed_weight_id"], "packed_weight_id")
    manifest_json = value["pack_manifest_json"]
    if not isinstance(manifest_json, str):
        raise BridgeCheckError("pack_manifest_json must be a string")
    manifest = manifest_json.encode("utf-8")
    if not 0 < len(manifest) <= MAX_MANIFEST_BYTES:
        raise BridgeCheckError("pack manifest is outside its ABI size bound")
    packed_hex = value["packed_weight_hex"]
    if (
        not isinstance(packed_hex, str)
        or len(packed_hex) != EXPECTED_PAYLOAD_BYTES * 2
        or re.fullmatch(r"[0-9a-f]+", packed_hex) is None
    ):
        raise BridgeCheckError("packed_weight_hex is not the exact lowercase payload")
    try:
        packed = bytes.fromhex(packed_hex)
    except ValueError as error:
        raise BridgeCheckError("packed_weight_hex is not valid hexadecimal") from error
    if len(packed) != EXPECTED_PAYLOAD_BYTES:
        raise BridgeCheckError("packed payload is not exactly 9216 bytes")
    _integer_list(value["input_fp32_bits"], 2, "input_fp32_bits")
    _integer_list(value["expected_output_fp32_bits"], 255, "expected_output_fp32_bits")
    return value


def _configure(library: Any) -> None:
    library.df_runtime_bridge_abi_version_v1.argtypes = []
    library.df_runtime_bridge_abi_version_v1.restype = ctypes.c_uint32
    library.df_runtime_create_neon_v1.argtypes = [
        _U8_POINTER,
        ctypes.c_size_t,
        _U8_POINTER,
        ctypes.c_size_t,
        _HANDLE_POINTER,
    ]
    library.df_runtime_create_neon_v1.restype = ctypes.c_int32
    library.df_runtime_last_error_v1.argtypes = [
        _CHAR_POINTER,
        ctypes.c_size_t,
        _SIZE_POINTER,
    ]
    library.df_runtime_last_error_v1.restype = ctypes.c_int32


def _last_error(library: Any) -> str:
    required = ctypes.c_size_t(0)
    status = int(library.df_runtime_last_error_v1(None, 0, ctypes.byref(required)))
    if (
        status not in (STATUS_OK, STATUS_TRUNCATED)
        or not 0 < required.value <= MAX_ERROR_BYTES
    ):
        return f"last-error query returned status={status}, bytes={required.value}"
    buffer = ctypes.create_string_buffer(required.value)
    status = int(
        library.df_runtime_last_error_v1(
            buffer,
            ctypes.c_size_t(required.value),
            ctypes.byref(required),
        )
    )
    if status not in (STATUS_OK, STATUS_TRUNCATED):
        return f"last-error copy returned status={status}"
    return bytes(buffer).split(b"\0", 1)[0].decode("ascii", errors="replace")


def _library_identity(path: Path) -> str:
    try:
        image = path.read_bytes()
    except OSError as error:
        raise BridgeCheckError(
            f"unable to hash release bridge {path}: {error}"
        ) from error
    return f"sha256:{hashlib.sha256(image).hexdigest()}"


def _check_darwin_eager(path: Path, envelope: dict[str, Any]) -> str:
    from decodeforge.torch_bridge import (
        NativeQ8Linear,
        RuntimeLibrary,
        close_binding,
        get_binding,
        load_binding,
    )

    try:
        torch = cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise BridgeCheckError(
            "the pytorch-cpu extra is required on Darwin arm64"
        ) from error

    library = RuntimeLibrary(path, _library_identity(path))
    manifest = cast(str, envelope["pack_manifest_json"]).encode("utf-8")
    packed = bytes.fromhex(cast(str, envelope["packed_weight_hex"]))
    binding_id = load_binding(library, manifest, packed)
    try:
        binding = get_binding(binding_id)
        if binding is None:
            raise BridgeCheckError("new binding is absent from the Python registry")
        descriptor = binding.descriptor
        if (
            descriptor.n,
            descriptor.k,
            descriptor.packed_weight_bytes,
        ) != (255, 2, EXPECTED_PAYLOAD_BYTES):
            raise BridgeCheckError("Python descriptor disagrees with the fixture shape")
        if descriptor.module_id != envelope["module_id"]:
            raise BridgeCheckError("Python descriptor module identity disagrees")
        if descriptor.packed_weight_id != envelope["packed_weight_id"]:
            raise BridgeCheckError("Python descriptor packed-weight identity disagrees")

        input_bits = _integer_list(envelope["input_fp32_bits"], 2, "input_fp32_bits")
        input_bytes = struct.pack("<2I", *input_bits)
        input_tensor = torch.frombuffer(
            bytearray(input_bytes), dtype=torch.float32
        ).reshape(1, 1, 2)
        original_input = input_tensor.view(torch.int32).clone()

        def unexpected_fallback(_input: Any) -> NoReturn:
            raise BridgeCheckError("eligible M=1 fixture unexpectedly used fallback")

        operator = NativeQ8Linear(binding_id, 255, 2, unexpected_fallback)
        output_tensor = operator(input_tensor)
        if (
            tuple(output_tensor.shape) != (1, 1, 255)
            or output_tensor.device.type != "cpu"
            or output_tensor.dtype is not torch.float32
            or not output_tensor.is_contiguous()
            or output_tensor.numel() != 255
        ):
            raise BridgeCheckError("eager operator returned invalid tensor metadata")
        if not torch.equal(input_tensor.view(torch.int32), original_input):
            raise BridgeCheckError("eager operator modified its borrowed input")
        actual = [
            int(word) & 0xFFFFFFFF
            for word in output_tensor.view(torch.int32).reshape(-1).tolist()
        ]
        expected = _integer_list(
            envelope["expected_output_fp32_bits"], 255, "expected_output_fp32_bits"
        )
        if actual != expected:
            first_difference = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(actual, expected, strict=True)
                    )
                    if left != right
                ),
                255,
            )
            raise BridgeCheckError(
                f"bitwise output mismatch at row {first_difference}: "
                f"actual=0x{actual[first_difference]:08x}, "
                f"expected=0x{expected[first_difference]:08x}"
            )
        counters = operator.counters
        if (
            counters.dispatch,
            counters.native_attempt,
            counters.native_success,
            counters.native_error,
            counters.fallback,
        ) != (1, 1, 1, 0, 0):
            raise BridgeCheckError(f"unexpected eager counters: {counters}")
        if operator.last_guard_reason is not None:
            raise BridgeCheckError("successful native call retained a guard reason")
    finally:
        close_binding(binding_id)
    if get_binding(binding_id) is not None:
        raise BridgeCheckError("closed binding remains in the Python registry")
    return "verified eager Torch bitwise M=1 check passed (Darwin arm64)"


def check_library(path: Path, envelope: dict[str, Any]) -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return _check_darwin_eager(path, envelope)
    if system != "Linux":
        raise BridgeCheckError(
            "bridge cdylib check supports Darwin arm64 or Linux, "
            f"got {system}:{machine}"
        )

    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise BridgeCheckError(
            f"unable to load release bridge {path}: {error}"
        ) from error
    _configure(library)
    abi_version = int(library.df_runtime_bridge_abi_version_v1())
    if abi_version != BRIDGE_ABI_VERSION:
        raise BridgeCheckError(f"bridge ABI version is {abi_version}")

    manifest = cast(str, envelope["pack_manifest_json"]).encode("utf-8")
    packed = bytes.fromhex(cast(str, envelope["packed_weight_hex"]))
    manifest_buffer = (ctypes.c_uint8 * len(manifest)).from_buffer_copy(manifest)
    packed_buffer = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
    handle = ctypes.c_uint64(0)
    status = int(
        library.df_runtime_create_neon_v1(
            manifest_buffer,
            ctypes.c_size_t(len(manifest)),
            packed_buffer,
            ctypes.c_size_t(len(packed)),
            ctypes.byref(handle),
        )
    )
    if status != STATUS_UNSUPPORTED_HOST or handle.value != 0:
        raise BridgeCheckError(
            f"Linux create expected UNSUPPORTED_HOST=10 and handle=0, "
            f"got status={status}, handle={handle.value}: {_last_error(library)}"
        )
    return "unsupported-host confirmed (Linux)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        envelope = _envelope_from_stdin()
        result = check_library(arguments.library, envelope)
    except (RuntimeError, OSError, ValueError, TypeError) as error:
        print(f"bridge-cdylib: error: {error}", file=sys.stderr)
        return 1
    print(f"bridge-cdylib: ok ({result})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

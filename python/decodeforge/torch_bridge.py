"""Eager-only PyTorch access to one verified DecodeForge Q8 linear.

The default decodeforge import remains framework-free.  This module loads
PyTorch lazily and keeps the foreign ABI behind RuntimeLibrary so the eager
operator only deals in tensor metadata and borrowed data pointers.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
import stat
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, TypeAlias

BRIDGE_ABI_VERSION: Final = 1
IDENTITY_CSTR_BYTES: Final = 72
MAX_MANIFEST_BYTES: Final = 16 * 1024
MAX_PACKED_WEIGHT_BYTES: Final = 128 * 1024 * 1024
MAX_VECTOR_ELEMENTS: Final = MAX_PACKED_WEIGHT_BYTES // 4
MAX_ERROR_BYTES: Final = 4096
MAX_DYLIB_BYTES: Final = 8 * 1024 * 1024
_POINTER_MAX: Final = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
OPERATOR_SCHEMA: Final = (
    "q8_linear_v1(Tensor x, int binding_id, int n, int k) -> Tensor"
)


class BridgeStatus(IntEnum):
    """Closed status vocabulary of the Rust bridge."""

    OK = 0
    TRUNCATED = 1
    NULL_ARGUMENT = 2
    ZERO_LENGTH = 3
    INVALID_HANDLE = 4
    INVALID_ARGUMENT = 5
    OVERLAP = 6
    LIMIT_VIOLATION = 7
    INVALID_MANIFEST = 8
    INVALID_PAYLOAD = 9
    UNSUPPORTED_HOST = 10
    BUILD_FAILED = 11
    LOAD_FAILED = 12
    EXECUTION_FAILED = 13
    NONFINITE_INPUT = 14
    NONFINITE_OUTPUT = 15
    PANIC = 16
    ALLOCATION_FAILED = 17
    INTERNAL = 18


class TorchBridgeError(RuntimeError):
    """A hard bridge or eager-operation failure."""

    def __init__(self, status: BridgeStatus | int, detail: str) -> None:
        self.status = status
        self.detail = detail
        status_name = (
            status.name.lower() if isinstance(status, BridgeStatus) else str(status)
        )
        super().__init__(f"decodeforge bridge {status_name}: {detail}")


class TorchUnavailableError(TorchBridgeError):
    """Raised when the optional PyTorch extra is not installed."""

    def __init__(self, detail: str = "install the pytorch-cpu extra") -> None:
        super().__init__(BridgeStatus.INTERNAL, detail)


@dataclass(frozen=True)
class RuntimeDescriptor:
    """The bounded descriptor returned by the Rust bridge."""

    n: int
    k: int
    packed_weight_bytes: int
    module_id: str
    packed_weight_id: str


class _CDescriptor(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("n", ctypes.c_uint32),
        ("k", ctypes.c_uint32),
        ("packed_weight_bytes", ctypes.c_uint64),
        ("module_id", ctypes.c_char * IDENTITY_CSTR_BYTES),
        ("packed_weight_id", ctypes.c_char * IDENTITY_CSTR_BYTES),
    ]


_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_F32_POINTER = ctypes.POINTER(ctypes.c_float)
_CHAR_POINTER = ctypes.POINTER(ctypes.c_char)
_SIZE_POINTER = ctypes.POINTER(ctypes.c_size_t)
_HANDLE_POINTER = ctypes.POINTER(ctypes.c_uint64)
_DESCRIPTOR_POINTER = ctypes.POINTER(_CDescriptor)


def _identity(value: str, field: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field} must be sha256:<64 lowercase hex digits>")
    return value


def _bounded_bytes(
    value: bytes | bytearray | memoryview, maximum: int, field: str
) -> bytes:
    if not isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(f"{field} must be bytes-like")
    source_length = value.nbytes if isinstance(value, memoryview) else len(value)
    if source_length == 0:
        raise ValueError(f"{field} must not be empty")
    if source_length > maximum:
        raise ValueError(f"{field} exceeds its {maximum}-byte bound")
    result = bytes(value)
    if len(result) > maximum:
        raise ValueError(f"{field} exceeds its {maximum}-byte bound")
    return result


def _require_pointer(address: int, length: int, field: str) -> None:
    if isinstance(address, bool) or not isinstance(address, int) or address <= 0:
        raise ValueError(f"{field} pointer must be a positive integer address")
    if address > _POINTER_MAX:
        raise ValueError(f"{field} pointer is outside the platform address range")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError(f"{field} length must be a positive integer")
    if length > MAX_VECTOR_ELEMENTS:
        raise ValueError(f"{field} length exceeds the bridge bound")


def _pointer_range(address: int, length: int, field: str) -> tuple[int, int]:
    _require_pointer(address, length, field)
    byte_length = length * ctypes.sizeof(ctypes.c_float)
    end = address + byte_length
    if end > _POINTER_MAX + 1:
        raise ValueError(f"{field} pointer range is outside the platform address range")
    return address, end


def _read_c_identity(raw: bytes, field: str) -> str:
    terminator = raw.find(b"\0")
    if terminator != IDENTITY_CSTR_BYTES - 1:
        raise TorchBridgeError(
            BridgeStatus.INTERNAL,
            f"{field} is not a fixed NUL-terminated identity",
        )
    try:
        value = raw[:terminator].decode("ascii")
    except UnicodeDecodeError as error:
        raise TorchBridgeError(
            BridgeStatus.INTERNAL, f"{field} is not ASCII"
        ) from error
    try:
        return _identity(value, field)
    except ValueError as error:
        raise TorchBridgeError(BridgeStatus.INTERNAL, str(error)) from error


def _descriptor_identity_bytes(value: _CDescriptor, field: str) -> bytes:
    offset = getattr(_CDescriptor, field).offset
    return ctypes.string_at(
        ctypes.addressof(value) + offset,
        IDENTITY_CSTR_BYTES,
    )


def _descriptor_from_c(value: _CDescriptor) -> RuntimeDescriptor:
    if value.abi_version != BRIDGE_ABI_VERSION:
        raise TorchBridgeError(
            BridgeStatus.INTERNAL,
            f"descriptor ABI version {value.abi_version} is unsupported",
        )
    if value.struct_size != ctypes.sizeof(_CDescriptor):
        raise TorchBridgeError(
            BridgeStatus.INTERNAL,
            f"descriptor size {value.struct_size} is unsupported",
        )
    if value.n <= 0 or value.k <= 0 or value.packed_weight_bytes <= 0:
        raise TorchBridgeError(BridgeStatus.INTERNAL, "descriptor shape is invalid")
    expected_payload_bytes = (int(value.n) + 3) // 4 * ((int(value.k) + 31) // 32) * 144
    if int(value.packed_weight_bytes) != expected_payload_bytes:
        raise TorchBridgeError(
            BridgeStatus.INTERNAL,
            "descriptor packed-weight length does not match its shape",
        )
    return RuntimeDescriptor(
        n=int(value.n),
        k=int(value.k),
        packed_weight_bytes=int(value.packed_weight_bytes),
        module_id=_read_c_identity(
            _descriptor_identity_bytes(value, "module_id"),
            "module_id",
        ),
        packed_weight_id=_read_c_identity(
            _descriptor_identity_bytes(value, "packed_weight_id"),
            "packed_weight_id",
        ),
    )


def _path_snapshot(path: Path) -> tuple[int, int, int, int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise TorchBridgeError(
            BridgeStatus.LOAD_FAILED, f"cannot stat bridge library {path}: {error}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TorchBridgeError(
            BridgeStatus.LOAD_FAILED,
            "bridge library must be a regular non-symlink file",
        )
    if metadata.st_size <= 0 or metadata.st_size > MAX_DYLIB_BYTES:
        raise TorchBridgeError(
            BridgeStatus.LIMIT_VIOLATION,
            f"bridge library size {metadata.st_size} is outside the supported bound",
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_verified_library(path: Path, expected_sha256: str) -> bytes:
    expected = _identity(expected_sha256, "required bridge SHA-256")
    before = _path_snapshot(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise TorchBridgeError(
            BridgeStatus.LOAD_FAILED, f"cannot read bridge library {path}: {error}"
        ) from error
    after = _path_snapshot(path)
    if before != after or len(data) != before[2]:
        raise TorchBridgeError(
            BridgeStatus.LOAD_FAILED,
            "bridge library changed while its content identity was read",
        )
    actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if actual != expected:
        raise TorchBridgeError(
            BridgeStatus.LOAD_FAILED,
            f"bridge SHA-256 {actual} does not equal required {expected}",
        )
    return data


class _PrivateLibrarySnapshot:
    """Owner-only load path containing exactly the hash-verified image."""

    def __init__(self, image: bytes, suffix: str) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="decodeforge-bridge-")
        directory = Path(self._directory.name)
        if stat.S_IMODE(directory.stat().st_mode) != 0o700:
            self.close()
            raise TorchBridgeError(
                BridgeStatus.LOAD_FAILED,
                "private bridge directory is not owner-only",
            )
        extension = suffix if suffix in {".dylib", ".so"} else ".bin"
        self.path = directory / f"libdecodeforge_bridge{extension}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(self.path, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                descriptor = -1
                written = output.write(image)
                if written != len(image):
                    raise OSError("short private bridge write")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(self.path, 0o400, follow_symlinks=False)
            metadata = self.path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o400
                or metadata.st_size != len(image)
            ):
                raise OSError("private bridge snapshot metadata is invalid")
        except OSError as error:
            if descriptor >= 0:
                os.close(descriptor)
            self.close()
            raise TorchBridgeError(
                BridgeStatus.LOAD_FAILED,
                f"cannot construct private bridge snapshot: {error}",
            ) from error

    def close(self) -> None:
        self._directory.cleanup()


class RuntimeLibrary:
    """ctypes adapter for one explicitly verified bridge dylib."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        required_sha256: str,
        *,
        cdll_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.path = Path(os.path.abspath(os.fspath(path)))
        self.required_sha256 = _identity(required_sha256, "required bridge SHA-256")
        self.image = _read_verified_library(self.path, self.required_sha256)
        snapshot = _PrivateLibrarySnapshot(self.image, self.path.suffix)
        loader = ctypes.CDLL if cdll_loader is None else cdll_loader
        try:
            self._cdll = loader(os.fspath(snapshot.path))
        except (OSError, TypeError, ValueError) as error:
            snapshot.close()
            raise TorchBridgeError(
                BridgeStatus.LOAD_FAILED,
                f"cannot load bridge library {self.path}: {error}",
            ) from error
        self._snapshot = snapshot
        try:
            self._configure()
            abi_version = int(self._abi_version())
            if abi_version != BRIDGE_ABI_VERSION:
                raise TorchBridgeError(
                    BridgeStatus.LOAD_FAILED,
                    f"bridge ABI version {abi_version} is unsupported",
                )
        except Exception:
            snapshot.close()
            raise

    def _configure(self) -> None:
        self._abi_version = self._symbol("df_runtime_bridge_abi_version_v1")
        self._abi_version.argtypes = []
        self._abi_version.restype = ctypes.c_uint32

        self._create = self._symbol("df_runtime_create_neon_v1")
        self._create.argtypes = [
            _U8_POINTER,
            ctypes.c_size_t,
            _U8_POINTER,
            ctypes.c_size_t,
            _HANDLE_POINTER,
        ]
        self._create.restype = ctypes.c_int32

        self._run = self._symbol("df_runtime_run_v1")
        self._run.argtypes = [
            ctypes.c_uint64,
            _F32_POINTER,
            ctypes.c_size_t,
            _F32_POINTER,
            ctypes.c_size_t,
        ]
        self._run.restype = ctypes.c_int32

        self._descriptor = self._symbol("df_runtime_get_descriptor_v1")
        self._descriptor.argtypes = [ctypes.c_uint64, _DESCRIPTOR_POINTER]
        self._descriptor.restype = ctypes.c_int32

        self._destroy = self._symbol("df_runtime_destroy_v1")
        self._destroy.argtypes = [ctypes.c_uint64]
        self._destroy.restype = ctypes.c_int32

        self._last_error = self._symbol("df_runtime_last_error_v1")
        self._last_error.argtypes = [_CHAR_POINTER, ctypes.c_size_t, _SIZE_POINTER]
        self._last_error.restype = ctypes.c_int32

    def _symbol(self, name: str) -> Any:
        try:
            return getattr(self._cdll, name)
        except AttributeError as error:
            raise TorchBridgeError(
                BridgeStatus.LOAD_FAILED, f"bridge library is missing {name}"
            ) from error

    def _last_error_text(self) -> str:
        required = ctypes.c_size_t(0)
        try:
            status = int(self._last_error(None, 0, ctypes.byref(required)))
            if status not in (BridgeStatus.OK, BridgeStatus.TRUNCATED):
                return f"last-error query failed with status {status}"
            if required.value == 0 or required.value > MAX_ERROR_BYTES:
                return "bridge returned an invalid last-error length"
            buffer = ctypes.create_string_buffer(required.value)
            status = int(
                self._last_error(buffer, required.value, ctypes.byref(required))
            )
            if status not in (BridgeStatus.OK, BridgeStatus.TRUNCATED):
                return f"last-error copy failed with status {status}"
            raw = bytes(buffer).split(b"\0", 1)[0]
            return raw.decode("utf-8", errors="replace")
        except (OSError, TypeError, ValueError) as error:
            return f"last-error query raised {type(error).__name__}: {error}"

    def _raise_status(self, raw_status: int) -> None:
        if raw_status == BridgeStatus.OK:
            return
        try:
            status: BridgeStatus | int = BridgeStatus(raw_status)
        except ValueError:
            status = raw_status
        raise TorchBridgeError(status, self._last_error_text())

    def create_binding(
        self,
        manifest_json: bytes | bytearray | memoryview,
        packed_weight: bytes | bytearray | memoryview,
    ) -> RuntimeBinding:
        manifest = _bounded_bytes(manifest_json, MAX_MANIFEST_BYTES, "manifest")
        payload = _bounded_bytes(
            packed_weight, MAX_PACKED_WEIGHT_BYTES, "packed payload"
        )
        manifest_buffer = (ctypes.c_uint8 * len(manifest)).from_buffer_copy(manifest)
        payload_buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        handle = ctypes.c_uint64(0)
        status = int(
            self._create(
                manifest_buffer,
                len(manifest),
                payload_buffer,
                len(payload),
                ctypes.byref(handle),
            )
        )
        try:
            self._raise_status(status)
        except Exception:
            if handle.value:
                with suppress(Exception):
                    self._destroy(handle.value)
            raise
        if handle.value == 0:
            raise TorchBridgeError(
                BridgeStatus.INTERNAL, "bridge returned a zero executable handle"
            )
        try:
            descriptor = self.get_descriptor(int(handle.value))
        except Exception:
            with suppress(Exception):
                self._destroy(handle.value)
            raise
        return RuntimeBinding(self, int(handle.value), descriptor)

    def get_descriptor(self, handle: int) -> RuntimeDescriptor:
        if isinstance(handle, bool) or not isinstance(handle, int) or handle <= 0:
            raise TorchBridgeError(BridgeStatus.INVALID_HANDLE, "handle is invalid")
        descriptor = _CDescriptor()
        status = int(
            self._descriptor(ctypes.c_uint64(handle), ctypes.byref(descriptor))
        )
        self._raise_status(status)
        return _descriptor_from_c(descriptor)

    def run(
        self,
        handle: int,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        if isinstance(handle, bool) or not isinstance(handle, int) or handle <= 0:
            raise TorchBridgeError(BridgeStatus.INVALID_HANDLE, "handle is invalid")
        input_range = _pointer_range(input_address, input_length, "input")
        output_range = _pointer_range(output_address, output_length, "output")
        if input_range[0] < output_range[1] and output_range[0] < input_range[1]:
            raise TorchBridgeError(
                BridgeStatus.OVERLAP, "input and output ranges overlap"
            )
        input_pointer = ctypes.cast(ctypes.c_void_p(input_address), _F32_POINTER)
        output_pointer = ctypes.cast(ctypes.c_void_p(output_address), _F32_POINTER)
        status = int(
            self._run(
                ctypes.c_uint64(handle),
                input_pointer,
                input_length,
                output_pointer,
                output_length,
            )
        )
        self._raise_status(status)

    def destroy(self, handle: int) -> None:
        if isinstance(handle, bool) or not isinstance(handle, int) or handle <= 0:
            raise TorchBridgeError(BridgeStatus.INVALID_HANDLE, "handle is invalid")
        self._raise_status(int(self._destroy(ctypes.c_uint64(handle))))


class BindingLike(Protocol):
    descriptor: RuntimeDescriptor

    @property
    def closed(self) -> bool: ...

    def run(
        self,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None: ...

    def close(self) -> None: ...


class RuntimeBinding:
    """A process-local owner of one bridge handle and its library."""

    def __init__(
        self,
        library: RuntimeLibrary,
        handle: int,
        descriptor: RuntimeDescriptor,
    ) -> None:
        self.library = library
        self.handle = handle
        self.descriptor = descriptor
        self._closed = False
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def run(
        self,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        with self._lock:
            if self._closed:
                raise TorchBridgeError(
                    BridgeStatus.INVALID_HANDLE, "runtime binding is closed"
                )
            self.library.run(
                self.handle,
                input_address,
                input_length,
                output_address,
                output_length,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.library.destroy(self.handle)
            finally:
                self._closed = True

    def __enter__(self) -> RuntimeBinding:
        if self.closed:
            raise TorchBridgeError(
                BridgeStatus.INVALID_HANDLE, "runtime binding is closed"
            )
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class BindingRegistry:
    """Synchronized process-local binding IDs; IDs are never reused."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._next_id = 1
        self._entries: dict[int, BindingLike] = {}

    def register(self, binding: BindingLike) -> int:
        if binding.closed:
            raise TorchBridgeError(
                BridgeStatus.INVALID_HANDLE, "cannot register a closed binding"
            )
        with self._lock:
            binding_id = self._next_id
            self._next_id += 1
            self._entries[binding_id] = binding
            return binding_id

    def get(self, binding_id: int) -> BindingLike | None:
        if isinstance(binding_id, bool) or not isinstance(binding_id, int):
            return None
        with self._lock:
            return self._entries.get(binding_id)

    def close(self, binding_id: int) -> None:
        if isinstance(binding_id, bool) or not isinstance(binding_id, int):
            raise TorchBridgeError(
                BridgeStatus.INVALID_HANDLE, "binding ID is not registered"
            )
        with self._lock:
            binding = self._entries.pop(binding_id, None)
        if binding is None:
            raise TorchBridgeError(
                BridgeStatus.INVALID_HANDLE, "binding ID is not registered"
            )
        binding.close()

    def clear(self) -> None:
        with self._lock:
            bindings = list(self._entries.values())
            self._entries.clear()
        for binding in bindings:
            binding.close()


_BINDINGS = BindingRegistry()


def register_binding(binding: BindingLike) -> int:
    """Retain a binding and return its nonzero process-local ID."""

    return _BINDINGS.register(binding)


def get_binding(binding_id: int) -> BindingLike | None:
    """Return a retained binding, or None for a forged/closed ID."""

    return _BINDINGS.get(binding_id)


def close_binding(binding_id: int) -> None:
    """Destroy and forget one retained binding."""

    _BINDINGS.close(binding_id)


def load_binding(
    library: RuntimeLibrary,
    manifest_json: bytes | bytearray | memoryview,
    packed_weight: bytes | bytearray | memoryview,
) -> int:
    """Create, describe, retain, and identify one runtime executable."""

    binding = library.create_binding(manifest_json, packed_weight)
    try:
        return register_binding(binding)
    except Exception:
        binding.close()
        raise


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise TorchUnavailableError() from error


def _tensor_guard_reason(
    x: Any,
    binding_id: int,
    n: int,
    k: int,
    *,
    torch_module: Any,
) -> str | None:
    binding = get_binding(binding_id)
    if binding is None or binding.closed:
        return "binding_unavailable"
    if binding.descriptor.n != n or binding.descriptor.k != k:
        return "binding_shape_mismatch"
    if getattr(x, "device", None) is None or x.device.type != "cpu":
        return "device"
    if getattr(x, "dtype", None) is not torch_module.float32:
        return "dtype"
    if getattr(x, "layout", None) is not torch_module.strided:
        return "layout"
    if bool(getattr(x, "requires_grad", False)):
        return "requires_grad"
    if bool(x.is_conj()):
        return "conjugate"
    if bool(x.is_neg()):
        return "negative"
    if not bool(x.is_contiguous()):
        return "non_contiguous"
    shape = tuple(int(dimension) for dimension in x.shape)
    if not shape or shape[-1] != k:
        return "shape"
    if any(dimension > 1 for dimension in shape[:-1]):
        return "m_gt_one"
    if any(dimension != 1 for dimension in shape[:-1]):
        return "shape"
    if int(x.numel()) != k:
        return "numel"
    return None


def _native_q8_linear(
    x: Any,
    binding_id: int,
    n: int,
    k: int,
    *,
    torch_module: Any | None = None,
) -> Any:
    """Run one native call; this function never invokes fallback."""

    torch = _torch_module() if torch_module is None else torch_module
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
        or isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
    ):
        raise TorchBridgeError(
            BridgeStatus.INVALID_ARGUMENT,
            "native q8_linear_v1 dimensions must be positive integers",
        )
    try:
        reason = _tensor_guard_reason(x, binding_id, n, k, torch_module=torch)
    except Exception as error:
        raise TorchBridgeError(
            BridgeStatus.INVALID_ARGUMENT,
            "native q8_linear_v1 guard inspection failed",
        ) from error
    if reason is not None:
        raise TorchBridgeError(
            BridgeStatus.INVALID_ARGUMENT,
            f"native q8_linear_v1 guard rejected input: {reason}",
        )
    binding = get_binding(binding_id)
    if binding is None:
        raise TorchBridgeError(
            BridgeStatus.INVALID_HANDLE, "binding disappeared before native call"
        )
    output_shape = (*map(int, x.shape[:-1]), n)
    output = torch.empty(output_shape, dtype=torch.float32, device="cpu")
    binding.run(
        int(x.data_ptr()),
        int(x.numel()),
        int(output.data_ptr()),
        int(output.numel()),
    )
    return output


_TORCH_LIBRARIES: tuple[Any, Any] | None = None
_TORCH_LIBRARY_LOCK = threading.RLock()


def register_torch_ops() -> None:
    """Register the eager-only DecodeForge operator exactly once."""

    global _TORCH_LIBRARIES
    torch = _torch_module()
    with _TORCH_LIBRARY_LOCK:
        if _TORCH_LIBRARIES is not None:
            return
        try:
            definition = torch.library.Library("decodeforge", "DEF")
        except RuntimeError:
            definition = torch.library.Library("decodeforge", "FRAGMENT")
        try:
            definition.define(OPERATOR_SCHEMA)
        except RuntimeError as error:
            if "already defined" not in str(error).lower():
                raise
        implementation = torch.library.Library("decodeforge", "IMPL", "CPU")
        try:
            implementation.impl("q8_linear_v1", _native_q8_linear)
        except RuntimeError as error:
            if "already" not in str(error).lower():
                raise
        _TORCH_LIBRARIES = (definition, implementation)


def _call_registered_native(x: Any, binding_id: int, n: int, k: int) -> Any:
    register_torch_ops()
    torch = _torch_module()
    return torch.ops.decodeforge.q8_linear_v1(x, binding_id, n, k)


@dataclass(frozen=True)
class DispatchCounters:
    """Immutable snapshot of completed eager dispatch accounting."""

    dispatch: int
    native_attempt: int
    native_success: int
    native_error: int
    fallback: int

    def validate(self) -> None:
        if self.dispatch != self.native_attempt + self.fallback:
            raise AssertionError("dispatch counter invariant failed")
        if self.native_attempt != self.native_success + self.native_error:
            raise AssertionError("native counter invariant failed")


FallbackCallable: TypeAlias = Callable[[Any], Any]
NativeCallable: TypeAlias = Callable[[Any, int, int, int], Any]


class NativeQ8Linear:
    """Low-level guarded callable with observable native/fallback counters.

    This is deliberately not an ``nn.Module`` or a binding owner. Model-level
    adapters must retain/close the binding and supply a fallback that implements
    the same logical Q8 weights identified by the binding descriptor.
    """

    def __init__(
        self,
        binding_id: int,
        n: int,
        k: int,
        fallback: FallbackCallable,
        *,
        native_operator: NativeCallable | None = None,
    ) -> None:
        if (
            isinstance(binding_id, bool)
            or not isinstance(binding_id, int)
            or binding_id <= 0
        ):
            raise ValueError("binding_id must be a positive integer")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not callable(fallback):
            raise TypeError("fallback must be callable")
        self.binding_id = binding_id
        self.n = n
        self.k = k
        self.fallback = fallback
        self._native_operator = (
            _call_registered_native if native_operator is None else native_operator
        )
        self._dispatch = 0
        self._native_attempt = 0
        self._native_success = 0
        self._native_error = 0
        self._fallback = 0
        self._last_guard_reason: str | None = None
        self._counter_lock = threading.Lock()

    @property
    def counters(self) -> DispatchCounters:
        with self._counter_lock:
            snapshot = DispatchCounters(
                dispatch=self._dispatch,
                native_attempt=self._native_attempt,
                native_success=self._native_success,
                native_error=self._native_error,
                fallback=self._fallback,
            )
        snapshot.validate()
        return snapshot

    @property
    def last_guard_reason(self) -> str | None:
        with self._counter_lock:
            return self._last_guard_reason

    def reset_counters(self) -> None:
        with self._counter_lock:
            self._dispatch = 0
            self._native_attempt = 0
            self._native_success = 0
            self._native_error = 0
            self._fallback = 0
            self._last_guard_reason = None

    def guard_reason(self, x: Any) -> str | None:
        torch = _torch_module()
        return _tensor_guard_reason(
            x, self.binding_id, self.n, self.k, torch_module=torch
        )

    def __call__(self, x: Any) -> Any:
        torch = _torch_module()
        try:
            reason = _tensor_guard_reason(
                x, self.binding_id, self.n, self.k, torch_module=torch
            )
        except Exception:
            reason = "guard_error"
        if reason is not None:
            with self._counter_lock:
                self._last_guard_reason = reason
            try:
                return self.fallback(x)
            finally:
                with self._counter_lock:
                    self._dispatch += 1
                    self._fallback += 1
        with self._counter_lock:
            self._last_guard_reason = None
        try:
            result = self._native_operator(x, self.binding_id, self.n, self.k)
        except Exception:
            with self._counter_lock:
                self._dispatch += 1
                self._native_attempt += 1
                self._native_error += 1
            raise
        with self._counter_lock:
            self._dispatch += 1
            self._native_attempt += 1
            self._native_success += 1
        return result


__all__ = [
    "BRIDGE_ABI_VERSION",
    "IDENTITY_CSTR_BYTES",
    "MAX_ERROR_BYTES",
    "MAX_MANIFEST_BYTES",
    "MAX_PACKED_WEIGHT_BYTES",
    "BindingRegistry",
    "BridgeStatus",
    "DispatchCounters",
    "NativeQ8Linear",
    "RuntimeBinding",
    "RuntimeDescriptor",
    "RuntimeLibrary",
    "TorchBridgeError",
    "TorchUnavailableError",
    "close_binding",
    "get_binding",
    "load_binding",
    "register_binding",
    "register_torch_ops",
]

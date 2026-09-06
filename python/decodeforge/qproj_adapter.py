"""Owning PyTorch adapter for one prepared DecodeForge query projection.

Importing :mod:`decodeforge` does not import this module or PyTorch.  Applications
that opt into model integration import this module explicitly after installing
the ``pytorch-cpu`` extra.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, TypeAlias, cast

import torch
import torch.nn.functional as functional
from torch import nn

from .torch_bridge import (
    NativeQ8Linear,
    RuntimeDescriptor,
    RuntimeLibrary,
    close_binding,
    get_binding,
    load_binding,
)

_IDENTITY_PREFIX: Final = "sha256:"
_MAX_LAYER_NAME_BYTES: Final = 256
SAME_Q8_REFERENCE_REASON: Final = "forced_same_q8_reference"

NativeCallable: TypeAlias = Callable[[Any, int, int, int], Any]


class QProjAdapterError(RuntimeError):
    """A prepared-asset, ownership, or lifecycle contract violation."""


class QProjExecutionMode(StrEnum):
    """Closed execution policies for one query-projection adapter."""

    HYBRID_NATIVE = "hybrid_native"
    SAME_Q8_REFERENCE = "same_q8_reference"


class QProjAdapterInitializationError(QProjAdapterError):
    """Own a binding whose cleanup failed during adapter construction."""

    def __init__(
        self,
        binding_id: int,
        initialization_error: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        super().__init__(
            "q-projection adapter initialization and binding cleanup both failed; "
            "call close() on this exception to retry cleanup"
        )
        self._binding_id = binding_id
        self._initialization_error = initialization_error
        self._cleanup_error = cleanup_error
        self._closed = False
        self._lock = threading.Lock()

    @property
    def binding_id(self) -> int:
        """Return the retained process-local binding ID."""

        return self._binding_id

    @property
    def initialization_error(self) -> BaseException:
        """Return the error that made adapter construction fail."""

        return self._initialization_error

    @property
    def cleanup_error(self) -> BaseException:
        """Return the first cleanup error retained for diagnosis."""

        return self._cleanup_error

    @property
    def closed(self) -> bool:
        """Return whether a cleanup retry successfully released the binding."""

        with self._lock:
            return self._closed

    def close(self) -> None:
        """Retry cleanup while retaining ownership after another failure."""

        with self._lock:
            if self._closed:
                return
            close_binding(self._binding_id)
            self._closed = True

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


@dataclass(frozen=True)
class QProjMetadata:
    """Immutable provenance exposed for result-bundle capture."""

    layer_name: str
    n: int
    k: int
    module_id: str
    packed_weight_id: str
    packed_weight_bytes: int
    fallback_weight_id: str


@dataclass(frozen=True)
class QProjCounters:
    """Immutable snapshot of completed adapter calls and current lifecycle."""

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

    def validate(self) -> None:
        """Raise when the snapshot violates its accounting invariants."""

        values = (
            self.forward,
            self.native_attempt,
            self.native_success,
            self.native_error,
            self.fallback_attempt,
            self.fallback_success,
            self.fallback_error,
            self.predispatch_error,
            self.rejected_closed,
            self.in_flight,
        )
        if any(value < 0 for value in values):
            raise AssertionError("q-projection counters must be nonnegative")
        if self.forward != (
            self.native_attempt + self.fallback_attempt + self.predispatch_error
        ):
            raise AssertionError("forward counter does not partition by dispatch path")
        if self.native_attempt != self.native_success + self.native_error:
            raise AssertionError("native counter invariant failed")
        if self.fallback_attempt != self.fallback_success + self.fallback_error:
            raise AssertionError("fallback counter invariant failed")


def _identity(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith(_IDENTITY_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field} must be sha256:<64 lowercase hex digits>")
    return value


def _layer_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("layer_name must be a string")
    encoded = value.encode("utf-8")
    if (
        not encoded
        or len(encoded) > _MAX_LAYER_NAME_BYTES
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ValueError("layer_name must be bounded printable text")
    return value


def _owned_fallback(weight: torch.Tensor) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("fallback_weight must be a torch.Tensor")
    if weight.device.type != "cpu":
        raise ValueError("fallback_weight must be on CPU")
    if weight.dtype is not torch.float32:
        raise ValueError("fallback_weight must have dtype torch.float32")
    if weight.layout is not torch.strided:
        raise ValueError("fallback_weight must use strided layout")
    if weight.ndim != 2 or any(int(dimension) <= 0 for dimension in weight.shape):
        raise ValueError("fallback_weight must have a nonempty [N,K] shape")
    if not weight.is_contiguous():
        raise ValueError("fallback_weight must be contiguous")
    if weight.requires_grad or weight.grad_fn is not None:
        raise ValueError("fallback_weight must be non-trainable")
    if weight.is_conj() or weight.is_neg():
        raise ValueError(
            "fallback_weight must not carry conjugate or negative view bits"
        )
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("fallback_weight must contain only finite values")
    return (
        weight.detach()
        .clone(memory_format=torch.contiguous_format)
        .requires_grad_(False)
    )


def fallback_weight_identity(weight: torch.Tensor) -> str:
    """Hash the canonical contiguous little-endian FP32 fallback bytes."""

    if sys.byteorder != "little":
        raise QProjAdapterError("fallback identities require a little-endian host")
    owned = _owned_fallback(weight)
    return _fallback_bytes_identity(owned)


def _fallback_bytes_identity(weight: torch.Tensor) -> str:
    array = weight.numpy()
    digest = hashlib.sha256(memoryview(cast(Any, array))).hexdigest()
    return f"{_IDENTITY_PREFIX}{digest}"


def _owned_fallback_with_identity(
    weight: torch.Tensor, required_identity: str
) -> tuple[torch.Tensor, str]:
    required = _identity(required_identity, "fallback_weight_id")
    if sys.byteorder != "little":
        raise QProjAdapterError("fallback identities require a little-endian host")
    owned = _owned_fallback(weight)
    actual = _fallback_bytes_identity(owned)
    if actual != required:
        raise QProjAdapterError(
            f"fallback weight identity {actual} does not equal declared {required}"
        )
    return owned, actual


class QProjAdapter(nn.Module):
    """Own one native binding and its exact same-Q8 FP32 fallback.

    This module is permanently in evaluation mode and rejects ``_apply``-based
    dtype/device transformations. The fallback buffer is intentionally
    nonpersistent: it is reconstructed from the identity-bound prepared asset
    rather than duplicated in a model ``state_dict``. Loading state into an
    installed adapter is rejected. ``close`` is idempotent, waits for every
    admitted forward call, and retains ownership when destruction fails.
    """

    same_q8_weight: torch.Tensor

    def __init__(
        self,
        *,
        layer_name: str,
        library: RuntimeLibrary,
        pack_manifest_json: bytes | bytearray | memoryview,
        packed_weight: bytes | bytearray | memoryview,
        fallback_weight: torch.Tensor,
        fallback_weight_id: str,
        fallback_parent_packed_weight_id: str,
        expected_module_id: str | None = None,
        native_operator: NativeCallable | None = None,
    ) -> None:
        super().__init__()
        validated_layer_name = _layer_name(layer_name)
        parent_id = _identity(
            fallback_parent_packed_weight_id,
            "fallback_parent_packed_weight_id",
        )
        required_module_id = (
            None
            if expected_module_id is None
            else _identity(expected_module_id, "expected_module_id")
        )
        owned_weight, actual_fallback_id = _owned_fallback_with_identity(
            fallback_weight, fallback_weight_id
        )

        self._lifecycle = threading.Condition(threading.RLock())
        self._dispatch_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._close_failed = False
        self._transitioning = False
        self._execution_mode = QProjExecutionMode.HYBRID_NATIVE
        self._forward_threads: set[int] = set()
        self._in_flight = 0
        self._forward = 0
        self._native_attempt = 0
        self._native_success = 0
        self._native_error = 0
        self._fallback_attempt = 0
        self._fallback_success = 0
        self._fallback_error = 0
        self._predispatch_error = 0
        self._rejected_closed = 0
        self._last_guard_reason: str | None = None
        self._binding_id: int | None = None

        binding_id = load_binding(library, pack_manifest_json, packed_weight)
        try:
            binding = get_binding(binding_id)
            if binding is None or binding.closed:
                raise QProjAdapterError("new runtime binding is unavailable")
            descriptor = binding.descriptor
            self._validate_descriptor(
                descriptor,
                owned_weight,
                parent_id,
                required_module_id,
            )
            self.register_buffer("same_q8_weight", owned_weight, persistent=False)
            self._fallback_object_id = id(self.same_q8_weight)
            self._fallback_data_pointer = int(self.same_q8_weight.data_ptr())
            self._fallback_version = int(self.same_q8_weight._version)
            self._metadata = QProjMetadata(
                layer_name=validated_layer_name,
                n=descriptor.n,
                k=descriptor.k,
                module_id=descriptor.module_id,
                packed_weight_id=descriptor.packed_weight_id,
                packed_weight_bytes=descriptor.packed_weight_bytes,
                fallback_weight_id=actual_fallback_id,
            )
            self._binding_id = binding_id
            self._callable = NativeQ8Linear(
                binding_id,
                descriptor.n,
                descriptor.k,
                self._fallback,
                native_operator=native_operator,
            )
            super().train(False)
        except BaseException as initialization_error:
            try:
                close_binding(binding_id)
            except BaseException as cleanup_error:
                self._binding_id = None
                self._closed = True
                raise QProjAdapterInitializationError(
                    binding_id,
                    initialization_error,
                    cleanup_error,
                ) from initialization_error
            self._binding_id = None
            self._closed = True
            raise

    @staticmethod
    def _validate_descriptor(
        descriptor: RuntimeDescriptor,
        fallback_weight: torch.Tensor,
        parent_id: str,
        required_module_id: str | None,
    ) -> None:
        shape = tuple(int(dimension) for dimension in fallback_weight.shape)
        if shape != (descriptor.n, descriptor.k):
            raise QProjAdapterError(
                f"fallback shape {shape} does not match binding "
                f"[{descriptor.n},{descriptor.k}]"
            )
        if descriptor.packed_weight_id != parent_id:
            raise QProjAdapterError(
                "fallback parent packed-weight identity does not match binding"
            )
        if (
            required_module_id is not None
            and descriptor.module_id != required_module_id
        ):
            raise QProjAdapterError(
                "binding module identity does not match expected asset"
            )

    @property
    def metadata(self) -> QProjMetadata:
        return self._metadata

    @property
    def closed(self) -> bool:
        with self._lifecycle:
            return self._closed

    @property
    def last_guard_reason(self) -> str | None:
        with self._counter_lock:
            return self._last_guard_reason

    @property
    def execution_mode(self) -> QProjExecutionMode:
        """Return the mode used by calls admitted after the latest transition."""

        with self._lifecycle:
            return self._execution_mode

    def set_execution_mode(self, mode: QProjExecutionMode) -> QProjExecutionMode:
        """Wait for admitted calls, change mode, and return the previous mode."""

        if not isinstance(mode, QProjExecutionMode):
            raise TypeError("mode must be a QProjExecutionMode")
        thread_id = threading.get_ident()
        with self._lifecycle:
            if thread_id in self._forward_threads:
                raise QProjAdapterError(
                    "cannot change execution mode from an admitted forward call"
                )
            while self._transitioning:
                self._lifecycle.wait()
            if self._closing or self._closed or self._close_failed:
                raise QProjAdapterError(
                    "cannot change execution mode while adapter is unavailable"
                )
            previous = self._execution_mode
            if mode is previous:
                return previous
            self._transitioning = True
            try:
                while self._in_flight:
                    self._lifecycle.wait()
                self._execution_mode = mode
            finally:
                self._transitioning = False
                self._lifecycle.notify_all()
            return previous

    @property
    def counters(self) -> QProjCounters:
        with self._counter_lock:
            forward = self._forward
            native_attempt = self._native_attempt
            native_success = self._native_success
            native_error = self._native_error
            fallback_attempt = self._fallback_attempt
            fallback_success = self._fallback_success
            fallback_error = self._fallback_error
            predispatch_error = self._predispatch_error
            rejected_closed = self._rejected_closed
        with self._lifecycle:
            in_flight = self._in_flight
            closed = self._closed
        snapshot = QProjCounters(
            forward=forward,
            native_attempt=native_attempt,
            native_success=native_success,
            native_error=native_error,
            fallback_attempt=fallback_attempt,
            fallback_success=fallback_success,
            fallback_error=fallback_error,
            predispatch_error=predispatch_error,
            rejected_closed=rejected_closed,
            in_flight=in_flight,
            closed=closed,
        )
        snapshot.validate()
        return snapshot

    def _validate_fallback_storage(self, weight: torch.Tensor | None = None) -> None:
        if weight is None:
            weight = self.same_q8_weight
        if (
            id(weight) != self._fallback_object_id
            or weight.device.type != "cpu"
            or weight.dtype is not torch.float32
            or weight.layout is not torch.strided
            or tuple(int(dimension) for dimension in weight.shape)
            != (self._metadata.n, self._metadata.k)
            or not weight.is_contiguous()
            or weight.requires_grad
            or weight.is_conj()
            or weight.is_neg()
            or int(weight.data_ptr()) != self._fallback_data_pointer
            or int(weight._version) != self._fallback_version
        ):
            raise QProjAdapterError("same-Q8 fallback buffer was replaced or mutated")

    def _fallback(self, x: Any) -> Any:
        weight = self.same_q8_weight
        self._validate_fallback_storage(weight)
        snapshot = weight.detach().clone(memory_format=torch.contiguous_format)
        actual_identity = _fallback_bytes_identity(snapshot)
        if actual_identity != self._metadata.fallback_weight_id:
            raise QProjAdapterError(
                "same-Q8 fallback buffer content no longer matches its identity"
            )
        return functional.linear(x, snapshot, bias=None)

    def _begin_forward(self) -> QProjExecutionMode:
        thread_id = threading.get_ident()
        with self._lifecycle:
            while self._transitioning:
                self._lifecycle.wait()
            if self._closing or self._closed or self._close_failed:
                with self._counter_lock:
                    self._rejected_closed += 1
                raise QProjAdapterError("q-projection adapter is closed")
            self._in_flight += 1
            self._forward_threads.add(thread_id)
            return self._execution_mode

    def _end_forward(self) -> None:
        thread_id = threading.get_ident()
        with self._lifecycle:
            self._forward_threads.discard(thread_id)
            self._in_flight -= 1
            if self._in_flight == 0:
                self._lifecycle.notify_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        execution_mode = self._begin_forward()
        try:
            with self._dispatch_lock:
                try:
                    self._validate_fallback_storage()
                except BaseException:
                    with self._counter_lock:
                        self._forward += 1
                        self._predispatch_error += 1
                    raise
                forced_reference = (
                    execution_mode is QProjExecutionMode.SAME_Q8_REFERENCE
                )
                try:
                    result = (
                        self._fallback(x) if forced_reference else self._callable(x)
                    )
                except BaseException:
                    reason = (
                        SAME_Q8_REFERENCE_REASON
                        if forced_reference
                        else self._callable.last_guard_reason
                    )
                    with self._counter_lock:
                        self._last_guard_reason = reason
                        self._forward += 1
                        if reason is None:
                            self._native_attempt += 1
                            self._native_error += 1
                        else:
                            self._fallback_attempt += 1
                            self._fallback_error += 1
                    raise
                if not isinstance(result, torch.Tensor):
                    reason = (
                        SAME_Q8_REFERENCE_REASON
                        if forced_reference
                        else self._callable.last_guard_reason
                    )
                    with self._counter_lock:
                        self._last_guard_reason = reason
                        self._forward += 1
                        if reason is None:
                            self._native_attempt += 1
                            self._native_error += 1
                        else:
                            self._fallback_attempt += 1
                            self._fallback_error += 1
                    raise QProjAdapterError(
                        "q-projection dispatch returned a non-tensor result"
                    )
                reason = (
                    SAME_Q8_REFERENCE_REASON
                    if forced_reference
                    else self._callable.last_guard_reason
                )
                with self._counter_lock:
                    self._last_guard_reason = reason
                    self._forward += 1
                    if reason is None:
                        self._native_attempt += 1
                        self._native_success += 1
                    else:
                        self._fallback_attempt += 1
                        self._fallback_success += 1
                return result
        finally:
            self._end_forward()

    def close(self) -> None:
        binding_id: int | None
        with self._lifecycle:
            while self._closing or self._transitioning:
                self._lifecycle.wait()
            if self._closed:
                return
            self._closing = True
            while self._in_flight:
                self._lifecycle.wait()
            binding_id = self._binding_id
        try:
            if binding_id is not None:
                close_binding(binding_id)
        except BaseException:
            with self._lifecycle:
                self._closing = False
                self._close_failed = True
                self._lifecycle.notify_all()
            raise
        with self._lifecycle:
            self._binding_id = None
            self._closed = True
            self._closing = False
            self._close_failed = False
            self._lifecycle.notify_all()

    def train(self, mode: bool = True) -> QProjAdapter:
        """Keep the native/fallback adapter permanently in evaluation mode."""

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if mode:
            raise QProjAdapterError("q-projection adapter is inference-only")
        return super().train(False)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> QProjAdapter:
        """Reject dtype/device/storage transformations of the owned fallback."""

        del fn, recurse
        raise QProjAdapterError(
            "q-projection adapter does not support dtype or device transformations"
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
        """Reject state loading; adapters must be rebuilt from prepared assets."""

        del (
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        raise QProjAdapterError(
            "q-projection adapter state must be rebuilt from prepared assets"
        )

    def __enter__(self) -> QProjAdapter:
        if self.closed:
            raise QProjAdapterError("q-projection adapter is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def extra_repr(self) -> str:
        metadata = self._metadata
        return (
            f"layer_name={metadata.layer_name!r}, n={metadata.n}, k={metadata.k}, "
            f"execution_mode={self.execution_mode.value!r}, closed={self.closed}"
        )


__all__ = [
    "SAME_Q8_REFERENCE_REASON",
    "QProjAdapter",
    "QProjAdapterError",
    "QProjAdapterInitializationError",
    "QProjCounters",
    "QProjExecutionMode",
    "QProjMetadata",
    "fallback_weight_identity",
]

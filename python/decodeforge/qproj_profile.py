"""Scoped diagnostic instrumentation of owning query-projection adapters.

Only selected adapter instances and their registry entries are changed. Callers
must exclusively own the model through setup, forwards, and restoration. The
outer NativeQ8Linear eligibility check remains unattributed. No production
adapter or bridge implementation is patched globally.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import torch
import torch.nn.functional as functional
from torch import nn

from . import torch_bridge as bridge
from .qproj_adapter import QProjAdapter, QProjAdapterError, _fallback_bytes_identity

QPROJ_PROFILE_BOUNDARIES: Final = (
    "adapter_storage_guard",
    "fallback",
    "fallback_clone",
    "fallback_hash",
    "fallback_linear",
    "guarded_native_operator",
    "guarded_binding_run",
)
SpanFactory: TypeAlias = Callable[[str, str], AbstractContextManager[Any]]
_MAX_MODULES: Final = 512
_MISSING: Final = object()
_ACTIVE: weakref.WeakSet[QProjAdapter] = weakref.WeakSet()
_ACTIVE_LOCK = threading.Lock()


class QProjProfileError(RuntimeError):
    """Internal profiling could not finish without ambiguity."""


class _Observer:
    """Defer observer failures until adapter dispatch accounting has completed."""

    def __init__(self, factory: SpanFactory) -> None:
        self.factory = factory
        self.thread = threading.get_ident()
        self.errors: list[BaseException] = []

    def _remember(self, error: BaseException) -> None:
        # Keep one error: a failing observer must not grow unbounded state.
        if not self.errors:
            self.errors.append(error)

    @contextmanager
    def span(self, boundary: str, path: str) -> Iterator[None]:
        if self.thread != threading.get_ident():
            self._remember(QProjProfileError("q-projection profile changed threads"))
        if self.errors:
            yield
            return
        try:
            manager = self.factory(boundary, path)
            manager.__enter__()
        except BaseException as error:
            self._remember(error)
            yield
            return
        try:
            yield
        except BaseException as error:
            try:
                # Observer suppression must never suppress a production error.
                manager.__exit__(type(error), error, error.__traceback__)
            except BaseException as observer_error:
                if observer_error is not error:
                    self._remember(observer_error)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except BaseException as error:
                self._remember(error)


class _ProfileBinding:
    """Borrow one registry entry; ownership stays with the original binding."""

    def __init__(
        self,
        original: bridge.BindingLike,
        registry: bridge.BindingRegistry,
        binding_id: int,
        observer: _Observer,
        path: str,
    ) -> None:
        self.original = original
        self.registry = registry
        self.binding_id = binding_id
        self.observer = observer
        self.path = path

    @property
    def descriptor(self) -> bridge.RuntimeDescriptor:
        return self.original.descriptor

    @property
    def closed(self) -> bool:
        return self.original.closed

    def run(
        self,
        input_address: int,
        input_length: int,
        output_address: int,
        output_length: int,
    ) -> None:
        with self.observer.span("guarded_binding_run", self.path):
            self.original.run(
                input_address, input_length, output_address, output_length
            )

    def close(self) -> None:
        if isinstance(self.original, bridge.RuntimeBinding):
            # Calling original.close() would reenter this registry proxy.
            self.original._close_from_registry(self.registry, self.binding_id)
        else:
            self.original.close()


@dataclass(frozen=True)
class _Target:
    path: str
    adapter: QProjAdapter
    binding_id: int
    binding: bridge.BindingLike


def _targets(model: nn.Module, paths: Sequence[str]) -> tuple[_Target, ...]:
    if len(paths) > _MAX_MODULES:
        raise QProjProfileError("too many q-projection profile module paths")
    result: list[_Target] = []
    adapters: set[int] = set()
    bindings: set[int] = set()
    for path in paths:
        if not isinstance(path, str) or not path or len(path) > 256:
            raise QProjProfileError("q-projection profile path is invalid")
        try:
            module = model.get_submodule(path)
        except (AttributeError, KeyError) as error:
            raise QProjProfileError(f"profile module is missing: {path}") from error
        adapter = (
            module
            if isinstance(module, QProjAdapter)
            else getattr(module, "adapter", None)
        )
        if not isinstance(adapter, QProjAdapter):
            continue
        if id(adapter) in adapters:
            raise QProjProfileError("one adapter cannot be profiled through two paths")
        with _ACTIVE_LOCK:
            if adapter in _ACTIVE:
                raise QProjProfileError(
                    "a q-projection internal profile is already active"
                )
        for fallback in (adapter._fallback, adapter._callable.fallback):
            if (
                getattr(fallback, "__self__", None) is not adapter
                or getattr(fallback, "__func__", None) is not QProjAdapter._fallback
            ):
                raise QProjProfileError("profile requires the production fallback body")
        counters = adapter.counters
        binding_id = adapter._binding_id
        if counters.closed or counters.in_flight or binding_id is None:
            raise QProjProfileError("profile requires idle, open owning adapters")
        binding = bridge.get_binding(binding_id)
        if binding is None or binding_id in bindings:
            raise QProjProfileError("profile requires distinct live adapter bindings")
        result.append(_Target(path, adapter, binding_id, binding))
        adapters.add(id(adapter))
        bindings.add(binding_id)
    if not result:
        raise QProjProfileError("profile requires at least one owning QProjAdapter")
    return tuple(result)


def _replace_attribute(
    owner: Any, name: str, value: Any, undo: list[Callable[[], None]]
) -> None:
    previous = owner.__dict__.get(name, _MISSING)

    def restore() -> None:
        if previous is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, previous)

    setattr(owner, name, value)
    undo.append(restore)


def _install_target(
    target: _Target,
    observer: _Observer,
    registry: bridge.BindingRegistry,
    undo: list[Callable[[], None]],
) -> None:
    adapter, path = target.adapter, target.path
    original_guard = adapter._validate_fallback_storage
    original_native = adapter._callable._native_operator

    def guard(weight: torch.Tensor | None = None) -> None:
        with observer.span("adapter_storage_guard", path):
            original_guard(weight)

    def fallback(x: Any) -> Any:
        # Keep these operations identical to QProjAdapter._fallback. Splitting
        # its short body here avoids adding instrumentation to production calls.
        with observer.span("fallback", path):
            weight = adapter.same_q8_weight
            adapter._validate_fallback_storage(weight)
            with observer.span("fallback_clone", path):
                snapshot = weight.detach().clone(memory_format=torch.contiguous_format)
            with observer.span("fallback_hash", path):
                actual_identity = _fallback_bytes_identity(snapshot)
                if actual_identity != adapter.metadata.fallback_weight_id:
                    raise QProjAdapterError(
                        "same-Q8 fallback buffer content no longer matches its identity"
                    )
            with observer.span("fallback_linear", path):
                return functional.linear(x, snapshot, bias=None)

    def native(x: Any, binding_id: int, n: int, k: int) -> Any:
        with observer.span("guarded_native_operator", path):
            return original_native(x, binding_id, n, k)

    proxy = _ProfileBinding(target.binding, registry, target.binding_id, observer, path)

    def restore_binding() -> None:
        with registry._lock:
            current = registry._entries.get(target.binding_id)
            if current is proxy:
                if target.binding.closed:
                    registry._entries.pop(target.binding_id)
                else:
                    registry._entries[target.binding_id] = target.binding
            elif current is not None or not target.binding.closed:
                raise QProjProfileError(
                    "profile binding entry changed before restoration"
                )

    with registry._lock:
        if registry._entries.get(target.binding_id) is not target.binding:
            raise QProjProfileError("profile binding changed before installation")
        registry._entries[target.binding_id] = proxy
        undo.append(restore_binding)
    _replace_attribute(adapter, "_validate_fallback_storage", guard, undo)
    _replace_attribute(adapter, "_fallback", fallback, undo)
    # NativeQ8Linear captured the bound fallback at adapter construction.
    _replace_attribute(adapter._callable, "fallback", fallback, undo)
    _replace_attribute(adapter._callable, "_native_operator", native, undo)


@contextmanager
def profile_qproj_internals(
    model: nn.Module, module_paths: Sequence[str], span_factory: SpanFactory
) -> Iterator[None]:
    """Temporarily instrument actual owning adapters under exclusive model use.

    Span callbacks must observe only. Their failures are deferred until this
    context exits so successful native/fallback calls keep successful counters.
    If production work fails, its exception remains primary. Closing an idle
    adapter within this context still releases its original binding normally.
    """

    targets = _targets(model, module_paths)
    observer = _Observer(span_factory)
    with _ACTIVE_LOCK:
        if any(target.adapter in _ACTIVE for target in targets):
            raise QProjProfileError("a q-projection internal profile is already active")
        for target in targets:
            _ACTIVE.add(target.adapter)
    undo: list[Callable[[], None]] = []
    failure: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        for target in targets:
            _install_target(target, observer, bridge._BINDINGS, undo)
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        for restore in reversed(undo):
            try:
                restore()
            except BaseException as error:
                cleanup_errors.append(error)
        with _ACTIVE_LOCK:
            for target in targets:
                _ACTIVE.discard(target.adapter)
        errors = [*observer.errors, *cleanup_errors]
        if errors:
            if failure is not None:
                for secondary_error in errors:
                    failure.add_note(
                        f"q-projection profile also failed: {secondary_error}"
                    )
            else:
                raise QProjProfileError(
                    "q-projection profiling or restoration failed"
                ) from errors[0]


__all__ = [
    "QPROJ_PROFILE_BOUNDARIES",
    "QProjProfileError",
    "SpanFactory",
    "profile_qproj_internals",
]

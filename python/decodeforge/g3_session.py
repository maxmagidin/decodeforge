"""Strict-offline runner for one frozen G3 TinyLlama generation session.

The base :mod:`decodeforge` package does not import this opt-in module.  A
successful call returns evidence for one session; any environment, identity,
correctness, coverage, timing, or reconciliation mismatch raises instead of
emitting an accepted-looking result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import pwd
import resource
import secrets
import shlex
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias, cast

import torch
from torch import nn
from torch.nn import functional as torch_functional

from .contracts import validate_data
from .g3_preparation import VerifiedPreparationReceipt, verify_preparation_receipt
from .qproj_adapter import QProjAdapter, QProjExecutionMode
from .qproj_model import (
    QProjAssetInventory,
    QProjModelCounters,
    VerifiedQProjAsset,
    install_tinyllama_qproj,
    tinyllama_qproj_paths,
)
from .torch_bridge import MAX_DYLIB_BYTES, RuntimeLibrary

SPEC_SHA256: Final = "3d4a5d8662cc8d8b41814d2a72d614349afc8e93c78d587489b28c82f290dd2f"
PROTOCOL_ID: Final = "g3-tinyllama-qproj-generation-v1"
TEXT_ROLE: Final = "demonstration_only_not_correctness_evidence"
_IDENTITY_PREFIX: Final = "sha256:"
_MAX_REPLAY_PATH_CHARS: Final = 1024
_MAX_REPLAY_COMMAND_CHARS: Final = 4096
_REPLAY_PATH_PUNCTUATION: Final = frozenset("/._+-:@")
_BUNDLE_INVENTORY: Final = (
    {"path": "README.md", "role": "human_summary"},
    {"path": "analysis.json", "role": "derived_analysis"},
    {"path": "asset-inventory.json", "role": "asset_provenance"},
    {"path": "correctness.json", "role": "correctness_evidence"},
    {"path": "coverage.json", "role": "dispatch_coverage"},
    {"path": "environment.txt", "role": "environment_capture"},
    {"path": "generated.txt", "role": "demonstration_text"},
    {"path": "manifest.json", "role": "closed_bundle_manifest"},
    {"path": "prompt.json", "role": "prompt_and_token_ids"},
    {"path": "timings.csv", "role": "raw_timing_samples"},
)

JsonValue: TypeAlias = Any
JsonObject: TypeAlias = dict[str, Any]


class G3SessionError(RuntimeError):
    """The frozen session could not produce acceptable evidence."""


@dataclass(frozen=True)
class SessionRequest:
    """Explicit local inputs for one fresh-process G3 session."""

    session_id: str
    session_index: int
    spec_path: Path
    model_directory: Path
    asset_directory: Path
    bridge_library: Path
    bridge_sha256: str
    preparation_receipt: Path
    session_output: Path
    process_start_ns: int


class InstallationLike(Protocol):
    @property
    def inventory(self) -> QProjAssetInventory: ...

    @property
    def counters(self) -> QProjModelCounters: ...

    @property
    def execution_mode(self) -> QProjExecutionMode: ...

    @property
    def closed(self) -> bool: ...

    def set_execution_mode(self, mode: QProjExecutionMode) -> QProjExecutionMode: ...

    def close(self) -> None: ...


class TokenizerLike(Protocol):
    eos_token_id: int | Sequence[int] | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: object) -> str: ...


ModelLoader: TypeAlias = Callable[[Path, Mapping[str, Any]], nn.Module]
TokenizerLoader: TypeAlias = Callable[[Path, Mapping[str, Any]], TokenizerLike]
RuntimeLoader: TypeAlias = Callable[[Path, str], RuntimeLibrary]
Installer: TypeAlias = Callable[
    [nn.Module, Path, RuntimeLibrary, float, float], "InstallResult"
]
InputVerifier: TypeAlias = Callable[
    [SessionRequest, Mapping[str, Any]], "VerifiedInputState"
]
PreparationLoader: TypeAlias = Callable[[Path], VerifiedPreparationReceipt]
PreparationCommandVerifier: TypeAlias = Callable[
    [SessionRequest, Sequence[Any], str], None
]


@dataclass(frozen=True)
class SessionDependencies:
    """Injectable process boundaries used by bounded deterministic tests."""

    verify_inputs: InputVerifier
    load_model: ModelLoader
    load_tokenizer: TokenizerLoader
    load_runtime: RuntimeLoader
    install: Installer
    configure_torch: Callable[[Mapping[str, Any]], None]
    load_preparation_receipt: PreparationLoader
    checkout_evidence: Callable[[Path], JsonObject]
    verify_preparation_command: PreparationCommandVerifier
    clock_ns: Callable[[], int] = time.perf_counter_ns
    peak_rss_bytes: Callable[[], int] | None = None


@dataclass
class VerifiedInputState:
    """Evidence plus a live anchored model-directory descriptor."""

    evidence: JsonObject
    model_directory_fd: int | None
    bridge_library: Path
    _bridge_descriptor: int | None = None
    _model_snapshot_path: Path | None = None
    _model_snapshot_files: tuple[str, ...] = ()

    def close(self) -> None:
        failures: list[BaseException] = []
        if self.model_directory_fd is not None:
            descriptor = self.model_directory_fd
            model_cleaned = False
            try:
                if self._model_snapshot_path is not None:
                    for name in self._model_snapshot_files:
                        with suppress(FileNotFoundError):
                            os.unlink(name, dir_fd=descriptor)
                    held = os.fstat(descriptor)
                    named = os.stat(self._model_snapshot_path, follow_symlinks=False)
                    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
                        raise G3SessionError(
                            "model snapshot pathname no longer names the held directory"
                        )
                    os.rmdir(self._model_snapshot_path)
                model_cleaned = True
            except BaseException as error:
                failures.append(error)
            if model_cleaned:
                try:
                    os.close(descriptor)
                except BaseException as error:
                    failures.append(error)
                else:
                    self.model_directory_fd = None
                    self._model_snapshot_path = None
        if self._bridge_descriptor is not None:
            try:
                os.close(self._bridge_descriptor)
            except BaseException as error:
                failures.append(error)
            self._bridge_descriptor = None
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("verified input cleanup failed", failures)


@dataclass(frozen=True)
class InstallResult:
    """Transactional owner plus all preinstallation direct checks."""

    installation: InstallationLike
    direct_checks: tuple[JsonObject, ...]


@dataclass
class _Generation:
    record: JsonObject
    logits: list[torch.Tensor]
    native_samples: list[tuple[int, torch.Tensor, torch.Tensor]]


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise G3SessionError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise G3SessionError(f"{field} must be an array")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise G3SessionError(f"{field} must be an integer")
    return cast(int, value)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise G3SessionError(f"{field} must be numeric")
    return float(value)


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise G3SessionError(f"{field} must be a string")
    return value


def _identity(value: str, field: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise G3SessionError(f"{field} must be 64 lowercase SHA-256 hex digits")
    return value


def _open_directory(path: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise G3SessionError("host lacks required no-follow directory primitives")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    parts = path.parts
    if path.is_absolute():
        descriptor = os.open(os.path.sep, flags)
        parts = parts[1:]
    else:
        descriptor = os.open(".", flags)
    try:
        for part in parts:
            if part in {"", ".", ".."}:
                raise G3SessionError("directory path contains an unsafe component")
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException as error:
        os.close(descriptor)
        if isinstance(error, G3SessionError):
            raise
        if isinstance(error, OSError):
            raise G3SessionError("unable to open directory without symlinks") from error
        raise


def publish_new_json(path: Path, value: object) -> None:
    """Durably publish JSON without following or replacing filesystem objects."""

    if path.name in {"", ".", ".."} or Path(path.name).name != path.name:
        raise ValueError("output must name one file in an existing directory")
    directory = _open_directory(path.parent)
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    linked = False
    temporary_exists = False
    linked_identity: tuple[int, int] | None = None
    try:
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory,
        )
        temporary_exists = True
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        linked = True
        linked_metadata = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        linked_identity = (linked_metadata.st_dev, linked_metadata.st_ino)
        os.fsync(directory)
        os.unlink(temporary_name, dir_fd=directory)
        temporary_exists = False
        os.fsync(directory)
    except BaseException as error:
        rollback_failures: list[BaseException] = [error]
        if linked:
            try:
                current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != linked_identity:
                    raise G3SessionError("session output changed during rollback")
                os.unlink(path.name, dir_fd=directory)
                os.fsync(directory)
            except BaseException as rollback_error:
                rollback_failures.append(rollback_error)
        if len(rollback_failures) > 1:
            raise BaseExceptionGroup(
                "JSON publication rollback failed", rollback_failures
            ) from error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory)
            os.fsync(directory)
        os.close(directory)


def require_external_session_output(output: Path, spec_path: Path) -> None:
    """Reject result publication inside the checkout whose clean state is evidence."""

    repository = spec_path.resolve(strict=True).parents[2]
    candidate = output.parent.resolve(strict=True) / output.name
    if candidate.is_relative_to(repository):
        raise G3SessionError("session output must be outside the measured checkout")


def _require_absolute_normalized_path(path: Path, label: str) -> None:
    normalized = Path(os.path.normpath(os.fspath(path)))
    if not path.is_absolute() or path != normalized or ".." in path.parts:
        raise G3SessionError(f"{label} must be an absolute normalized path")
    rendered = os.fspath(path)
    if len(rendered) > _MAX_REPLAY_PATH_CHARS or any(
        not (
            character.isascii()
            and (character.isalnum() or character in _REPLAY_PATH_PUNCTUATION)
        )
        for character in rendered
    ):
        raise G3SessionError(f"{label} must use the bounded replay-safe path grammar")


def _bridge_rebuild_command(
    library: Path,
    spec_path: Path,
) -> str:
    """Bind the verified dylib layout to its exact external Cargo target."""

    _require_absolute_normalized_path(library, "bridge library")
    if (
        library.name != "libdecodeforge_bridge.dylib"
        or library.parent.name != "release"
    ):
        raise G3SessionError(
            "bridge library must use the external Cargo target release layout"
        )
    target = library.parent.parent
    try:
        resolved_target = target.resolve(strict=True)
    except OSError as error:
        raise G3SessionError("bridge Cargo target does not exist") from error
    if resolved_target != target:
        raise G3SessionError("bridge Cargo target path is not canonical")
    release_descriptor = _open_directory(library.parent)
    os.close(release_descriptor)
    repository = spec_path.resolve(strict=True).parents[2]
    if resolved_target == repository or resolved_target.is_relative_to(repository):
        raise G3SessionError(
            "bridge Cargo target must be outside the measured checkout"
        )
    return shlex.join(
        [
            "env",
            f"CARGO_TARGET_DIR={resolved_target}",
            "make",
            "build-g3-bridge",
        ]
    )


def _session_rebuild_command(request: SessionRequest) -> str:
    """Render the complete shell-safe public Make invocation for this session."""

    for input_path, label in (
        (request.model_directory, "model directory"),
        (request.asset_directory, "asset directory"),
        (request.bridge_library, "bridge library"),
        (request.preparation_receipt, "preparation receipt"),
        (request.session_output, "session output"),
    ):
        _require_absolute_normalized_path(input_path, label)
    _identity(request.bridge_sha256, "bridge_sha256")
    arguments = [
        "make",
        "run-g3-demo",
        f"SESSION_ID={request.session_id}",
        f"SESSION_INDEX={request.session_index}",
        f"MODEL_DIR={request.model_directory}",
        f"ASSETS={request.asset_directory}",
        f"LIBRARY={request.bridge_library}",
        f"LIBRARY_SHA256={request.bridge_sha256}",
        f"PREPARATION_RECEIPT={request.preparation_receipt}",
        f"OUTPUT={request.session_output}",
    ]
    command = shlex.join(arguments)
    if len(command) > _MAX_REPLAY_COMMAND_CHARS:
        raise G3SessionError("session rebuild command exceeds its byte bound")
    return command


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_leaf(directory_fd: int, name: str, expected_bytes: int | None) -> int:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise G3SessionError("file path is not a safe basename")
    if not hasattr(os, "O_NOFOLLOW"):
        raise G3SessionError("host lacks required no-follow file primitives")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise G3SessionError(f"{name} must be a singly linked regular file")
        if expected_bytes is not None and metadata.st_size != expected_bytes:
            raise G3SessionError(
                f"{name} has {metadata.st_size} bytes, expected {expected_bytes}"
            )
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(error, G3SessionError):
            raise
        if isinstance(error, OSError):
            raise G3SessionError(f"unable to open {name}: {error}") from error
        raise
    return descriptor


def _read_descriptor(descriptor: int, label: str) -> bytes:
    before = os.fstat(descriptor)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise G3SessionError(f"{label} changed during read")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise G3SessionError(f"{label} was truncated during read")
    return data


def _read_stable(
    path: Path, expected_bytes: int | None = None, *, maximum: int | None = None
) -> bytes:
    directory = _open_directory(path.parent)
    try:
        descriptor = _open_leaf(directory, path.name, expected_bytes)
        try:
            if maximum is not None and os.fstat(descriptor).st_size > maximum:
                raise G3SessionError(f"{path} exceeds its byte bound")
            return _read_descriptor(descriptor, str(path))
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _hash_leaf(directory: int, name: str, expected_bytes: int) -> str:
    descriptor = _open_leaf(directory, name, expected_bytes)
    try:
        return _hash_descriptor(descriptor, name)
    finally:
        os.close(descriptor)


def _hash_descriptor(descriptor: int, label: str) -> str:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(after) or size != after.st_size:
        raise G3SessionError(f"{label} changed during hashing")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_preparation_command(
    request: SessionRequest,
    arguments: Sequence[Any],
    executable_identity: str,
) -> None:
    if len(arguments) != 5 or arguments[1] != "--source" or arguments[3] != "--output":
        raise G3SessionError("preparation command argv is not canonical")
    tool = Path(_string(arguments[0], "preparation executable"))
    source = Path(_string(arguments[2], "preparation source path"))
    output = Path(_string(arguments[4], "preparation output path"))
    if (
        not tool.is_absolute()
        or tool.name != "decodeforge-prepare-qproj"
        or source != request.model_directory.absolute() / "model.safetensors"
        or output != request.asset_directory.absolute()
    ):
        raise G3SessionError("preparation command inputs do not match this session")
    parent = _open_directory(tool.parent)
    try:
        descriptor = _open_leaf(parent, tool.name, None)
        try:
            size = os.fstat(descriptor).st_size
            if not 0 < size <= 128 * 1024 * 1024:
                raise G3SessionError("preparation executable exceeds its byte bound")
            actual = _IDENTITY_PREFIX + _hash_descriptor(descriptor, tool.name)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    if actual != executable_identity:
        raise G3SessionError("preparation executable identity mismatch")


def _load_spec(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_stable(path, maximum=1024 * 1024)
    digest = _sha256(raw)
    if digest != SPEC_SHA256:
        raise G3SessionError("experiment spec is not the exact frozen document")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G3SessionError("experiment spec is not valid JSON") from error
    spec = _object(parsed, "spec")
    diagnostics = validate_data(spec, "g3-experiment-spec", document_name=path.name)
    if diagnostics:
        raise G3SessionError(f"experiment spec schema failure: {diagnostics[0]}")
    if spec.get("protocol_id") != PROTOCOL_ID or spec.get("readiness") != "ready":
        raise G3SessionError("experiment spec is not ready for this protocol")
    return spec, digest


def _verify_file(directory_fd: int, record: Mapping[str, Any]) -> JsonObject:
    filename = _string(record.get("filename"), "artifact filename")
    if Path(filename).name != filename:
        raise G3SessionError("artifact filename is not a basename")
    expected_bytes = _integer(record.get("size_bytes"), f"{filename} size")
    expected_sha = _identity(
        _string(record.get("sha256"), f"{filename} SHA"), f"{filename} SHA"
    )
    actual_sha = _hash_leaf(directory_fd, filename, expected_bytes)
    if actual_sha != expected_sha:
        raise G3SessionError(f"{filename} identity mismatch")
    return {
        "role": _string(record.get("role"), f"{filename} role"),
        "filename": filename,
        "size_bytes": expected_bytes,
        "sha256": actual_sha,
    }


def _snapshot_verified_file(
    source_directory: int, destination_directory: int, record: Mapping[str, Any]
) -> JsonObject:
    filename = _string(record.get("filename"), "artifact filename")
    expected_bytes = _integer(record.get("size_bytes"), f"{filename} size")
    expected_sha = _identity(
        _string(record.get("sha256"), f"{filename} SHA"), f"{filename} SHA"
    )
    source = _open_leaf(source_directory, filename, expected_bytes)
    destination = -1
    try:
        before = os.fstat(source)
        destination = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=destination_directory,
        )
        digest = hashlib.sha256()
        copied = 0
        while chunk := os.read(source, 1024 * 1024):
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                if written <= 0:
                    raise G3SessionError(f"unable to snapshot {filename}")
                view = view[written:]
        os.fsync(destination)
        after = os.fstat(source)
        if (
            _stat_fingerprint(before) != _stat_fingerprint(after)
            or copied != expected_bytes
            or digest.hexdigest() != expected_sha
        ):
            raise G3SessionError(f"{filename} changed or mismatched during snapshot")
    finally:
        os.close(source)
        if destination >= 0:
            os.close(destination)
    return {
        "role": _string(record.get("role"), f"{filename} role"),
        "filename": filename,
        "size_bytes": expected_bytes,
        "sha256": expected_sha,
    }


def _require_directory_inventory(
    directory_fd: int, expected: set[str], label: str
) -> None:
    observed: set[str] = set()
    with os.scandir(directory_fd) as iterator:
        for entry in iterator:
            observed.add(entry.name)
            if len(observed) > len(expected):
                raise G3SessionError(f"{label} contains unpinned extra files")
    if observed != expected:
        raise G3SessionError(f"{label} contains missing or unpinned extra files")


def _command_output(command: Sequence[str], *, allow_empty: bool = False) -> str:
    if not command or not Path(command[0]).is_absolute():
        raise G3SessionError("provenance commands require absolute executables")
    clean_environment = {
        "HOME": pwd.getpwuid(os.getuid()).pw_dir,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    with tempfile.TemporaryFile() as output:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=output,
                stderr=output,
                env=clean_environment,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise G3SessionError(f"unable to inspect {' '.join(command)}") from error
        size = os.fstat(output.fileno()).st_size
        if size > 64 * 1024:
            raise G3SessionError("provenance command output exceeds its byte bound")
        output.seek(0)
        try:
            text = output.read().decode("utf-8")
        except UnicodeDecodeError as error:
            raise G3SessionError("provenance command output is not UTF-8") from error
    if not allow_empty and not text.strip():
        raise G3SessionError("provenance command returned no output")
    return text


def _command_line(command: Sequence[str]) -> str:
    return _command_output(command).splitlines()[0].strip()


def _checkout_evidence(spec_path: Path) -> JsonObject:
    repository = spec_path.resolve(strict=True).parents[2]
    revision = _command_line(
        ["/usr/bin/git", "-C", str(repository), "rev-parse", "HEAD"]
    )
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise G3SessionError("unable to resolve checkout revision")
    status = _command_output(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        allow_empty=True,
    )
    if status:
        raise G3SessionError("accepted evidence requires a clean checkout")
    index = _command_output(
        ["/usr/bin/git", "-C", str(repository), "ls-files", "-v"],
        allow_empty=True,
    )
    if any(
        line and (line[0].islower() or line[0] == "S") for line in index.splitlines()
    ):
        raise G3SessionError("checkout index contains hidden tracked-file flags")
    producer_names = (
        "decodeforge.g3_session",
        "decodeforge.g3_preparation",
        "decodeforge.qproj_adapter",
        "decodeforge.qproj_model",
        "decodeforge.torch_bridge",
    )
    for name in producer_names:
        module_path = getattr(sys.modules.get(name), "__file__", None)
        if module_path is None or not Path(module_path).resolve(
            strict=True
        ).is_relative_to(repository):
            raise G3SessionError(
                "session producer code is outside the measured checkout"
            )
    if Path(sys.argv[0]).name == "run_g3_session.py" and not Path(sys.argv[0]).resolve(
        strict=True
    ).is_relative_to(repository):
        raise G3SessionError("session CLI is outside the measured checkout")
    return {"revision": revision, "dirty": False}


def _sysctl(name: str) -> str:
    return _command_line(["/usr/sbin/sysctl", "-n", name])


def _normalized_architecture(machine: str) -> str:
    value = machine.lower()
    return "aarch64" if value == "arm64" else value


def _verify_environment(spec: Mapping[str, Any]) -> JsonObject:
    software = _object(spec.get("software"), "software")
    expected_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__.split("+", 1)[0],
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "safetensors": importlib.metadata.version("safetensors"),
    }
    for name, actual in expected_versions.items():
        expected = _string(_object(software.get(name), name).get("version"), name)
        if actual != expected:
            raise G3SessionError(
                f"{name} version mismatch: expected {expected}, got {actual}"
            )
    numpy_version = importlib.metadata.version("numpy")
    if numpy_version != "2.4.4":
        raise G3SessionError(
            f"numpy version mismatch: expected 2.4.4, got {numpy_version}"
        )
    expected_versions["numpy"] = numpy_version
    rustup = Path("/opt/homebrew/bin/rustup").resolve(strict=True)
    rust_line = _command_line([str(rustup), "run", "1.98.0", "rustc", "--version"])
    rust_version = rust_line.split()[1] if len(rust_line.split()) >= 2 else ""
    expected_rust = _string(
        _object(software.get("rust"), "rust").get("version"), "rust"
    )
    if rust_version != expected_rust:
        raise G3SessionError("Rust version mismatch")
    expected_versions["rust"] = rust_version
    clang_line = _command_line(["/usr/bin/clang", "--version"])
    expected_clang = _string(
        _object(software.get("clang"), "clang").get("version"), "clang"
    )
    if clang_line != expected_clang:
        raise G3SessionError("Clang version mismatch")
    expected_versions["clang"] = clang_line

    host = _object(spec.get("host"), "host")
    os_version = platform.mac_ver()[0]
    actual_host: dict[str, Any] = {
        "host_id": "apple-m4-primary",
        "os": "macos" if platform.system() == "Darwin" else platform.system().lower(),
        "os_version": os_version,
        "os_build": _command_line(["/usr/bin/sw_vers", "-buildVersion"]),
        "kernel_release": platform.release(),
        "arch": _normalized_architecture(platform.machine()),
        "cpu_model": _sysctl("machdep.cpu.brand_string"),
        "hardware_model": _sysctl("hw.model"),
        "physical_cores": int(_sysctl("hw.physicalcpu")),
        "logical_cores": int(_sysctl("hw.logicalcpu")),
        "features": ["neon"],
        "affinity_policy": "macOS default scheduler; no hard affinity requested",
    }
    for field, actual in actual_host.items():
        if host.get(field) != actual:
            raise G3SessionError(
                f"host {field} mismatch: expected {host.get(field)!r}, got {actual!r}"
            )
    return cast(JsonObject, {"software": expected_versions, "host": actual_host})


def _default_verify_inputs(
    request: SessionRequest, spec: Mapping[str, Any]
) -> VerifiedInputState:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    model = _object(spec.get("model"), "model")
    records = [
        _object(model.get("weights_file"), "weights_file"),
        *[
            _object(value, "configuration file")
            for value in _array(model.get("configuration_files"), "configuration_files")
        ],
        *[
            _object(value, "tokenizer file")
            for value in _array(model.get("tokenizer_files"), "tokenizer_files")
        ],
    ]
    expected_names = {
        _string(record.get("filename"), "artifact filename") for record in records
    }
    model_fd = -1
    snapshot_fd = -1
    bridge_fd = -1
    snapshot_path: Path | None = None
    snapshot_identity: tuple[int, int] | None = None
    try:
        bridge_rebuild_command = _bridge_rebuild_command(
            request.bridge_library, request.spec_path
        )
        model_fd = _open_directory(request.model_directory)
        temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
        snapshot_path = Path(
            tempfile.mkdtemp(prefix="decodeforge-g3-model-", dir=temporary_root)
        )
        snapshot_metadata = os.stat(snapshot_path, follow_symlinks=False)
        snapshot_identity = (snapshot_metadata.st_dev, snapshot_metadata.st_ino)
        snapshot_fd = _open_directory(snapshot_path)
        model_before = os.fstat(model_fd)
        _require_directory_inventory(model_fd, expected_names, "model directory")
        files = [
            _snapshot_verified_file(model_fd, snapshot_fd, record) for record in records
        ]
        os.fsync(snapshot_fd)
        _require_directory_inventory(model_fd, expected_names, "model directory")
        if _stat_fingerprint(model_before) != _stat_fingerprint(os.fstat(model_fd)):
            raise G3SessionError("model directory changed during verification")
        os.close(model_fd)
        model_fd = -1
        bridge_sha = _identity(request.bridge_sha256, "bridge_sha256")
        bridge_parent = _open_directory(request.bridge_library.parent)
        try:
            bridge_fd = _open_leaf(bridge_parent, request.bridge_library.name, None)
            bridge_metadata = os.fstat(bridge_fd)
            if not 0 < bridge_metadata.st_size <= MAX_DYLIB_BYTES:
                raise G3SessionError("runtime bridge library exceeds its byte bound")
            actual_bridge_sha = _hash_descriptor(bridge_fd, request.bridge_library.name)
        finally:
            os.close(bridge_parent)
        if actual_bridge_sha != bridge_sha:
            raise G3SessionError("runtime bridge library identity mismatch")
        environment = _verify_environment(spec)
        return VerifiedInputState(
            evidence={
                "offline": True,
                "model_files": files,
                "bridge_library": {
                    "path": request.bridge_library.name,
                    "size_bytes": bridge_metadata.st_size,
                    "sha256": bridge_sha,
                },
                "bridge_rebuild_command": bridge_rebuild_command,
                "environment": environment,
            },
            model_directory_fd=snapshot_fd,
            bridge_library=Path(f"/dev/fd/{bridge_fd}"),
            _bridge_descriptor=bridge_fd,
            _model_snapshot_path=snapshot_path,
            _model_snapshot_files=tuple(sorted(expected_names)),
        )
    except BaseException as error:
        failures: list[BaseException] = [error]
        if model_fd >= 0:
            try:
                os.close(model_fd)
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
        if snapshot_fd >= 0 and snapshot_path is not None:
            try:
                snapshot_files = tuple(entry.name for entry in os.scandir(snapshot_fd))
                VerifiedInputState(
                    {},
                    snapshot_fd,
                    Path("unused"),
                    _model_snapshot_path=snapshot_path,
                    _model_snapshot_files=snapshot_files,
                ).close()
                snapshot_fd = -1
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
        elif snapshot_path is not None and snapshot_identity is not None:
            try:
                named = os.stat(snapshot_path, follow_symlinks=False)
                if (named.st_dev, named.st_ino) != snapshot_identity:
                    raise G3SessionError("model snapshot pathname was replaced")
                os.rmdir(snapshot_path)
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
        if bridge_fd >= 0:
            try:
                os.close(bridge_fd)
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
        if len(failures) > 1:
            raise BaseExceptionGroup(
                "input verification and cleanup failed", failures
            ) from error
        raise


def _default_load_model(directory: Path, _spec: Mapping[str, Any]) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = cast(
        nn.Module,
        AutoModelForCausalLM.from_pretrained(
            directory,
            local_files_only=True,
            dtype=torch.float32,
            use_safetensors=True,
        ),
    )
    model.requires_grad_(False)
    model.eval()
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or (
            parameter.is_floating_point() and parameter.dtype is not torch.float32
        ):
            raise G3SessionError("loaded model is not entirely CPU FP32")
    return model


def _default_load_tokenizer(directory: Path, _spec: Mapping[str, Any]) -> TokenizerLike:
    from transformers import AutoTokenizer

    return cast(
        TokenizerLike,
        AutoTokenizer.from_pretrained(directory, local_files_only=True),
    )


def _default_load_runtime(path: Path, sha256: str) -> RuntimeLibrary:
    return RuntimeLibrary(path, f"{_IDENTITY_PREFIX}{sha256}")


def _default_install(
    model: nn.Module,
    assets: Path,
    runtime: RuntimeLibrary,
    direct_atol: float,
    direct_rtol: float,
) -> InstallResult:
    checks: list[JsonObject] = []
    value = torch.linspace(-1.0, 1.0, 2048, dtype=torch.float32).reshape(1, 1, 2048)

    def factory(asset: VerifiedQProjAsset) -> QProjAdapter:
        entry = asset.entry
        adapter = QProjAdapter(
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
        try:
            with torch.inference_mode():
                actual = adapter(value)
                reference = torch_functional.linear(value, adapter.same_q8_weight)
            maximum, allowed, excess, passed = _compare(
                actual, reference, direct_atol, direct_rtol
            )
            checks.append(
                {
                    "layer": entry.layer,
                    "layer_path": entry.layer_path,
                    "result": {
                        "max_abs": maximum,
                        "max_allowed": allowed,
                        "max_excess": excess,
                        "finite": bool(
                            torch.isfinite(actual).all().item()
                            and torch.isfinite(reference).all().item()
                        ),
                        "pass": passed,
                    },
                }
            )
            if not passed:
                raise G3SessionError("preinstallation q_proj probe exceeded tolerance")
            return adapter
        except BaseException as error:
            try:
                adapter.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "q_proj probe and adapter cleanup both failed",
                    [error, cleanup_error],
                ) from error
            raise

    installation = install_tinyllama_qproj(model, assets, factory)
    if len(checks) != 22:
        installation.close()
        raise G3SessionError("preinstallation probes did not cover all 22 layers")
    return InstallResult(installation, tuple(checks))


def default_dependencies() -> SessionDependencies:
    """Return real strict-offline process boundaries."""

    return SessionDependencies(
        verify_inputs=_default_verify_inputs,
        load_model=_default_load_model,
        load_tokenizer=_default_load_tokenizer,
        load_runtime=_default_load_runtime,
        install=_default_install,
        configure_torch=_configure_torch,
        load_preparation_receipt=verify_preparation_receipt,
        checkout_evidence=_checkout_evidence,
        verify_preparation_command=_verify_preparation_command,
        peak_rss_bytes=_peak_rss_bytes,
    )


def _configure_torch(generation: Mapping[str, Any]) -> None:
    torch.set_num_threads(_integer(generation.get("torch_num_threads"), "threads"))
    torch.set_num_interop_threads(
        _integer(generation.get("torch_num_interop_threads"), "interop threads")
    )
    torch.manual_seed(_integer(generation.get("seed"), "seed"))


def _peak_rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def _counter_snapshot(counters: QProjModelCounters) -> list[JsonObject]:
    result: list[JsonObject] = []
    for layer in counters.layers:
        values = asdict(layer)
        layer_number = values.pop("layer")
        layer_path = values.pop("layer_path")
        result.append(
            {"layer": layer_number, "layer_path": layer_path, "values": values}
        )
    return result


def _asset_record(inventory: QProjAssetInventory) -> JsonObject:
    entries: list[JsonObject] = []
    fallback_bytes = inventory.total_fallback_bytes // len(inventory.entries)
    for entry in inventory.entries:
        entries.append(
            {
                "layer": entry.layer,
                "directory": entry.directory,
                "layer_path": entry.layer_path,
                "tensor_name": entry.tensor_name,
                "manifest_identity": entry.manifest_identity,
                "tensor_identity": entry.tensor_identity,
                "logical_weight_identity": entry.logical_weight_identity,
                "packed_weight_identity": entry.packed_weight_identity,
                "packed_bytes": entry.packed_bytes,
                "module_identity": entry.module_identity,
                "fallback_identity": entry.fallback_weight_identity,
                "fallback_bytes": fallback_bytes,
            }
        )
    return {
        "format": "decodeforge_q_proj_inventory_v1",
        "source": asdict(inventory.source),
        "aggregate_identity": inventory.aggregate_identity,
        "layer_count": len(entries),
        "total_packed_bytes": inventory.total_packed_bytes,
        "total_fallback_bytes": inventory.total_fallback_bytes,
        "entries": entries,
    }


def _loaded_environment(model: nn.Module, tokenizer: TokenizerLike) -> JsonObject:
    values = (*model.parameters(), *model.buffers())
    devices = {value.device.type for value in values}
    dtypes = {
        str(value.dtype).removeprefix("torch.")
        for value in values
        if value.is_floating_point()
    }
    if devices != {"cpu"} or dtypes != {"float32"}:
        raise G3SessionError("loaded model identity is not uniformly CPU FP32")
    return {
        "model": {
            "class": type(model).__name__,
            "device": "cpu",
            "dtype": "float32",
            "eval_mode": not model.training,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "method": "AutoTokenizer.from_pretrained-local-files-only",
        },
    }


_COUNTER_FIELDS: Final = (
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


def _counter_delta(
    before: QProjModelCounters, after: QProjModelCounters
) -> list[JsonObject]:
    if len(before.layers) != 22 or len(after.layers) != 22:
        raise G3SessionError("counter snapshot does not contain 22 layers")
    result: list[JsonObject] = []
    for old, new in zip(before.layers, after.layers, strict=True):
        if old.layer != new.layer or old.layer_path != new.layer_path:
            raise G3SessionError("counter layer ordering changed")
        values: JsonObject = {}
        for field in _COUNTER_FIELDS:
            values[field] = int(getattr(new, field)) - int(getattr(old, field))
        values["closed_changed"] = old.closed != new.closed
        result.append(
            {"layer": old.layer, "layer_path": old.layer_path, "values": values}
        )
    return result


def _logits_identity(logits: torch.Tensor) -> str:
    value = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return _sha256(value.numpy().tobytes(order="C"))


def _model_output(value: Any) -> tuple[torch.Tensor, Any]:
    logits = getattr(value, "logits", None)
    past = getattr(value, "past_key_values", None)
    if not isinstance(logits, torch.Tensor) or past is None:
        raise G3SessionError("model output lacks logits or cached key/value state")
    if logits.device.type != "cpu" or logits.dtype is not torch.float32:
        raise G3SessionError("model logits must be CPU FP32")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] <= 0:
        raise G3SessionError("model logits have an unsupported shape")
    if not bool(torch.isfinite(logits).all().item()):
        raise G3SessionError("model produced nonfinite logits")
    return logits, past


def _eos_ids(tokenizer: TokenizerLike) -> set[int]:
    raw = tokenizer.eos_token_id
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    return {int(value) for value in raw}


def _generate(
    model: nn.Module,
    tokenizer: TokenizerLike,
    prompt_ids: list[int],
    generation_spec: Mapping[str, Any],
    clock_ns: Callable[[], int],
    native_samples: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> tuple[list[int], list[torch.Tensor], JsonObject]:
    maximum = _integer(generation_spec.get("max_new_tokens"), "max_new_tokens")
    minimum = _integer(generation_spec.get("min_new_tokens"), "min_new_tokens")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    retained_logits: list[torch.Tensor] = []
    cached_ns: list[JsonValue] = []
    eos = _eos_ids(tokenizer)

    total_start = clock_ns()
    prefill_start = total_start
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    logits, past = _model_output(output)
    prefill_end = clock_ns()
    step_logits = logits[0, -1].detach().clone()
    retained_logits.append(step_logits)
    token = int(torch.argmax(step_logits).item())
    generated.append(token)
    ttft_end = clock_ns()

    while len(generated) < maximum and not (
        len(generated) >= minimum and generated[-1] in eos
    ):
        decode_start = clock_ns()
        decode_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device="cpu")
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=torch.long)), dim=1
        )
        output = model(
            input_ids=decode_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits, past = _model_output(output)
        step_logits = logits[0, -1].detach().clone()
        retained_logits.append(step_logits)
        token = int(torch.argmax(step_logits).item())
        generated.append(token)
        decode_end = clock_ns()
        cached_ns.append(decode_end - decode_start)
    total_end = clock_ns()
    if len(generated) < minimum:
        raise G3SessionError("generation produced fewer than the frozen minimum tokens")
    if any(
        value <= 0
        for value in [
            prefill_end - prefill_start,
            ttft_end - total_start,
            total_end - total_start,
            *cast(list[int], cached_ns),
        ]
    ):
        raise G3SessionError("generation timing sample is missing or nonpositive")
    return (
        generated,
        retained_logits,
        {
            "prefill_ns": prefill_end - prefill_start,
            "time_to_first_token_ns": ttft_end - total_start,
            "cached_step_ns": cached_ns,
            "total_ns": total_end - total_start,
        },
    )


def _register_native_hooks(
    model: nn.Module,
    samples: list[tuple[int, torch.Tensor, torch.Tensor]],
    dispatch: list[JsonObject],
    execution_path: str,
    clock_ns: Callable[[], int],
) -> list[Any]:
    handles: list[Any] = []
    starts: dict[int, list[int]] = {layer: [] for layer in range(22)}
    steps: dict[int, int] = {layer: 0 for layer in range(22)}
    for layer, path in enumerate(tinyllama_qproj_paths()):
        module = model.get_submodule(path)

        def pre_hook(
            _module: nn.Module,
            _arguments: tuple[Any, ...],
            *,
            current_layer: int = layer,
        ) -> None:
            starts[current_layer].append(clock_ns())

        def hook(
            _module: nn.Module,
            arguments: tuple[Any, ...],
            output: Any,
            *,
            current_layer: int = layer,
        ) -> None:
            if not starts[current_layer]:
                raise G3SessionError("q_proj dispatch timing stack is inconsistent")
            elapsed = clock_ns() - starts[current_layer].pop()
            if elapsed <= 0:
                raise G3SessionError("q_proj dispatch timing is nonpositive")
            dispatch.append(
                {
                    "step_index": steps[current_layer],
                    "layer": current_layer,
                    "layer_path": tinyllama_qproj_paths()[current_layer],
                    "dispatch": (
                        "native"
                        if execution_path == "hybrid_native"
                        and len(arguments) == 1
                        and isinstance(arguments[0], torch.Tensor)
                        and arguments[0].ndim >= 2
                        and arguments[0].shape[-2] == 1
                        else "fallback"
                    ),
                    "dispatch_ns": elapsed,
                }
            )
            steps[current_layer] += 1
            if (
                len(arguments) == 1
                and isinstance(arguments[0], torch.Tensor)
                and isinstance(output, torch.Tensor)
                and arguments[0].ndim >= 2
                and arguments[0].shape[-2] == 1
            ):
                samples.append(
                    (
                        current_layer,
                        arguments[0].detach().clone(),
                        output.detach().clone(),
                    )
                )

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(hook))
    return handles


def _compare(
    actual: torch.Tensor, reference: torch.Tensor, atol: float, rtol: float
) -> tuple[float, float, float, bool]:
    if actual.shape != reference.shape:
        return float("inf"), float("nan"), float("inf"), False
    if not bool(torch.isfinite(actual).all().item()) or not bool(
        torch.isfinite(reference).all().item()
    ):
        return float("inf"), float("nan"), float("inf"), False
    differences = (actual - reference).abs()
    allowed = atol + rtol * reference.abs()
    excess = differences - allowed
    maximum_excess = float(excess.max().item())
    return (
        float(differences.max().item()),
        float(allowed.max().item()),
        maximum_excess,
        maximum_excess <= 0.0,
    )


def _validate_native_samples(
    model: nn.Module,
    installation: InstallationLike,
    samples: Sequence[tuple[int, torch.Tensor, torch.Tensor]],
    atol: float,
    rtol: float,
) -> list[JsonObject]:
    previous = installation.set_execution_mode(QProjExecutionMode.SAME_Q8_REFERENCE)
    checks: list[JsonObject] = []
    try:
        with torch.inference_mode():
            for index, (layer, value, actual) in enumerate(samples):
                reference = model.get_submodule(tinyllama_qproj_paths()[layer])(value)
                if not isinstance(reference, torch.Tensor):
                    raise G3SessionError("same-Q8 validation returned a non-tensor")
                maximum, allowed, excess, passed = _compare(
                    actual, reference, atol, rtol
                )
                checks.append(
                    {
                        "step_index": index // 22 + 1,
                        "layer": layer,
                        "layer_path": tinyllama_qproj_paths()[layer],
                        "result": {
                            "max_abs": maximum,
                            "max_allowed": allowed,
                            "max_excess": excess,
                            "finite": bool(
                                torch.isfinite(actual).all().item()
                                and torch.isfinite(reference).all().item()
                            ),
                            "pass": passed,
                        },
                    }
                )
                if not passed:
                    raise G3SessionError(
                        "hybrid native q_proj output exceeded tolerance"
                    )
    finally:
        installation.set_execution_mode(previous)
    return checks


def _validate_direct_checks(checks: Sequence[Mapping[str, Any]]) -> None:
    if len(checks) != 22:
        raise G3SessionError("preinstallation direct checks are incomplete")
    for layer, check in enumerate(checks):
        if (
            check.get("layer") != layer
            or check.get("layer_path") != tinyllama_qproj_paths()[layer]
        ):
            raise G3SessionError("preinstallation checks are not in exact layer order")
        result = _object(check.get("result"), "direct result")
        values = (
            _number(result.get("max_abs"), "direct max_abs"),
            _number(result.get("max_allowed"), "direct max_allowed"),
            _number(result.get("max_excess"), "direct max_excess"),
        )
        if (
            not all(torch.isfinite(torch.tensor(value)).item() for value in values)
            or result.get("finite") is not True
            or result.get("pass") is not True
            or values[2] > 0.0
        ):
            raise G3SessionError("preinstallation direct check failed")


def _validate_preinstall_baseline(counters: QProjModelCounters) -> None:
    if len(counters.layers) != 22:
        raise G3SessionError("preinstallation counter baseline is incomplete")
    for layer, value in enumerate(counters.layers):
        if (
            value.layer != layer
            or value.layer_path != tinyllama_qproj_paths()[layer]
            or value.forward != 1
            or value.native_attempt != 1
            or value.native_success != 1
            or any(
                (
                    value.native_error,
                    value.fallback_attempt,
                    value.fallback_success,
                    value.fallback_error,
                    value.predispatch_error,
                    value.rejected_closed,
                    value.in_flight,
                )
            )
            or value.closed
        ):
            raise G3SessionError("preinstallation counter baseline is invalid")


def _reconcile_run(
    path: str, token_count: int, deltas: Sequence[Mapping[str, JsonValue]]
) -> None:
    if len(deltas) != 22:
        raise G3SessionError("run does not contain 22 counter deltas")
    for layer, delta in enumerate(deltas):
        values = _object(delta.get("values"), "counter delta values")
        expected_native = token_count - 1 if path == "hybrid_native" else 0
        expected_fallback = 1 if path == "hybrid_native" else token_count
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
            "closed_changed": False,
        }
        for field, value in expected.items():
            observed = delta.get(field) if field == "layer" else values.get(field)
            if observed != value:
                raise G3SessionError(
                    f"counter reconciliation failed for layer {layer} field {field}"
                )


def _compare_pair(
    reference: _Generation,
    hybrid: _Generation,
    atol: float,
    rtol: float,
) -> None:
    reference_ids = cast(list[JsonValue], reference.record["output_ids"])
    hybrid_ids = cast(list[JsonValue], hybrid.record["output_ids"])
    if reference_ids != hybrid_ids:
        raise G3SessionError("hybrid and same-Q8 token IDs differ")
    if len(reference.logits) != len(hybrid.logits):
        raise G3SessionError("hybrid and same-Q8 generation lengths differ")
    hybrid_steps = cast(list[JsonValue], hybrid.record["steps"])
    for index, (actual, expected) in enumerate(
        zip(hybrid.logits, reference.logits, strict=True)
    ):
        maximum, allowed, excess, passed = _compare(actual, expected, atol, rtol)
        step = _object(hybrid_steps[index], "hybrid step")
        step["hybrid_comparison"] = {
            "max_abs": maximum,
            "max_allowed": allowed,
            "max_excess": excess,
            "pass": passed,
        }
        if not passed:
            raise G3SessionError(f"model logits exceeded tolerance at step {index}")


def _path_order(repetition: int) -> tuple[str, str]:
    if repetition % 2 == 0:
        return "same_q8_reference", "hybrid_native"
    return "hybrid_native", "same_q8_reference"


def _drift(runs: Sequence[JsonObject], lower: float, upper: float) -> JsonObject:
    paths: list[JsonObject] = []
    for path in ("same_q8_reference", "hybrid_native"):
        values = [
            _integer(_object(run["timing"], "timing")["total_ns"], "total")
            for run in runs
            if run["phase"] == "measured" and run["path"] == path
        ]
        if len(values) != 10:
            raise G3SessionError("drift calculation lacks ten measured samples")
        ratio = statistics.median(values[-3:]) / statistics.median(values[:3])
        if not lower <= ratio <= upper:
            raise G3SessionError(f"{path} timing drift ratio {ratio} is out of bounds")
        paths.append(
            {
                "path": path,
                "first_window_median_ns": statistics.median(values[:3]),
                "last_window_median_ns": statistics.median(values[-3:]),
                "ratio": ratio,
                "pass": True,
            }
        )
    return {
        "metric": (
            "ratio_of_last_window_median_to_first_window_median_"
            "total_generation_ns_per_path"
        ),
        "window_generations": 3,
        "ratio_lower": lower,
        "ratio_upper": upper,
        "paths": paths,
        "pass": True,
    }


def _load_verified_components(
    request: SessionRequest,
    spec: Mapping[str, Any],
    verified: VerifiedInputState,
    dependencies: SessionDependencies,
) -> tuple[nn.Module, TokenizerLike, RuntimeLibrary, int, int]:
    """Load only through retained verified descriptors.

    ``fchdir`` is process-global, so the real runner is intentionally a fresh,
    single-threaded process and no concurrent work is permitted during this
    bounded section. The retained directory descriptor continues to name the
    verified inode even if its original pathname is renamed.
    """

    original_directory = -1
    load_path = request.model_directory
    try:
        if verified.model_directory_fd is not None:
            if not hasattr(os, "fchdir") or not hasattr(os, "O_DIRECTORY"):
                raise G3SessionError("host lacks required anchored-load primitives")
            original_directory = os.open(
                ".", os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
            )
            os.fchdir(verified.model_directory_fd)
            load_path = Path(".")
        model_start = dependencies.clock_ns()
        model = dependencies.load_model(load_path, spec)
        model_end = dependencies.clock_ns()
        tokenizer_start = dependencies.clock_ns()
        tokenizer = dependencies.load_tokenizer(load_path, spec)
        tokenizer_end = dependencies.clock_ns()
    finally:
        if original_directory >= 0:
            try:
                os.fchdir(original_directory)
            finally:
                os.close(original_directory)
    runtime = dependencies.load_runtime(verified.bridge_library, request.bridge_sha256)
    return (
        model,
        tokenizer,
        runtime,
        model_end - model_start,
        tokenizer_end - tokenizer_start,
    )


def run_session(
    request: SessionRequest,
    *,
    dependencies: SessionDependencies | None = None,
) -> JsonObject:
    """Run one complete frozen session or raise without accepted evidence."""

    if (
        not request.session_id
        or len(request.session_id) > 64
        or not request.session_id[0].isalnum()
        or any(
            not (character.isascii() and (character.isalnum() or character in "._-"))
            for character in request.session_id
        )
    ):
        raise G3SessionError("session_id must be bounded portable text")
    if request.session_index not in range(3):
        raise G3SessionError("session_index must be in the frozen range 0..2")
    resolved_spec = request.spec_path.resolve(strict=True)
    repository = resolved_spec.parents[2]
    if resolved_spec != (repository / "benchmarks/g3/spec.json").resolve(strict=True):
        raise G3SessionError("session must use the canonical experiment spec path")
    for input_path, label in (
        (request.model_directory, "model directory"),
        (request.asset_directory, "asset directory"),
        (request.bridge_library, "bridge library"),
        (request.preparation_receipt, "preparation receipt"),
        (request.session_output, "session output"),
    ):
        _require_absolute_normalized_path(input_path, label)
        if input_path == repository or input_path.is_relative_to(repository):
            raise G3SessionError(f"{label} must be outside the measured checkout")
    run_rebuild_command = _session_rebuild_command(request)
    require_external_session_output(request.session_output, request.spec_path)
    spec, spec_sha = _load_spec(request.spec_path)
    deps = default_dependencies() if dependencies is None else dependencies
    checkout_preflight = deps.checkout_evidence(request.spec_path)
    verified = deps.verify_inputs(request, spec)
    try:
        preparation_receipt = deps.load_preparation_receipt(request.preparation_receipt)
        offline_preparation = preparation_receipt.session_projection()
        preparation_checkout = preparation_receipt.checkout_revision
        preparation_argv = preparation_receipt.command_argv
        preparation_executable_identity = preparation_receipt.tool_executable_identity
        deps.verify_preparation_command(
            request, preparation_argv, preparation_executable_identity
        )
    except BaseException as error:
        try:
            verified.close()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "session preflight and verified-input cleanup both failed",
                [error, cleanup_error],
            ) from error
        raise
    generation_spec = _object(spec.get("generation"), "generation")
    execution_spec = _object(spec.get("execution"), "execution")
    correctness = _object(spec.get("correctness"), "correctness")
    direct = _object(correctness.get("direct_operator"), "direct_operator")
    model_correctness = _object(correctness.get("model_logits"), "model_logits")
    direct_atol = _number(direct.get("absolute_tolerance"), "direct atol")
    direct_rtol = _number(direct.get("relative_tolerance"), "direct rtol")
    logits_atol = _number(model_correctness.get("absolute_tolerance"), "logits atol")
    logits_rtol = _number(model_correctness.get("relative_tolerance"), "logits rtol")

    deps.configure_torch(generation_spec)
    baseline_rss = deps.peak_rss_bytes() if deps.peak_rss_bytes is not None else 0

    try:
        model, tokenizer, runtime, model_load_ns, tokenizer_load_ns = (
            _load_verified_components(request, spec, verified, deps)
        )
    except BaseException as error:
        try:
            verified.close()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "component loading and verified-input cleanup both failed",
                [error, cleanup_error],
            ) from error
        raise
    else:
        verified.close()
    if model.training:
        raise G3SessionError("model loader did not return an eval model")
    loaded_environment = _loaded_environment(model, tokenizer)
    prompt = _object(spec.get("prompt"), "prompt")
    prompt_text = _string(prompt.get("text"), "prompt text")
    tokenization = _object(prompt.get("tokenization"), "tokenization")
    prompt_ids = [
        _integer(value, "prompt token")
        for value in _array(tokenization.get("input_ids"), "prompt IDs")
    ]
    if tokenizer.encode(prompt_text, add_special_tokens=True) != prompt_ids:
        raise G3SessionError("tokenizer did not reproduce the exact frozen prompt IDs")

    install_start = deps.clock_ns()
    install_result = deps.install(
        model,
        request.asset_directory,
        runtime,
        direct_atol,
        direct_rtol,
    )
    installation = install_result.installation
    try:
        assets_record = _asset_record(installation.inventory)
        if (
            offline_preparation.get("asset_inventory_identity")
            != assets_record["aggregate_identity"]
        ):
            raise G3SessionError("preparation receipt does not match installed assets")
        install_end = deps.clock_ns()
        if (
            installation.counters.live_adapters != 22
            or installation.counters.in_flight != 0
        ):
            raise G3SessionError(
                "transactional installation is not fully live and idle"
            )
    except BaseException as error:
        try:
            installation.close()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "session setup and installation cleanup both failed",
                [error, cleanup_error],
            ) from error
        raise

    runs: list[JsonObject] = []
    generations: list[_Generation] = []
    direct_checks = list(install_result.direct_checks)
    try:
        _validate_direct_checks(direct_checks)
        _validate_preinstall_baseline(installation.counters)
        warmups = _integer(
            execution_spec.get("warmup_complete_generations_per_path"), "warmups"
        )
        measured = _integer(
            execution_spec.get("measured_generations_per_path_per_session"),
            "measured repetitions",
        )
        if warmups != 2 or measured != 10:
            raise G3SessionError("runner supports only the frozen 2/10 session shape")

        global_order = 0
        for phase, count in (("warmup", warmups), ("measured", measured)):
            for repetition in range(count):
                pair: dict[str, _Generation] = {}
                for path in _path_order(repetition):
                    mode = (
                        QProjExecutionMode.SAME_Q8_REFERENCE
                        if path == "same_q8_reference"
                        else QProjExecutionMode.HYBRID_NATIVE
                    )
                    installation.set_execution_mode(mode)
                    before = installation.counters
                    if runs and _object(runs[-1]["counters"], "counters")[
                        "after"
                    ] != _counter_snapshot(before):
                        raise G3SessionError("run counter snapshots are not continuous")
                    samples: list[tuple[int, torch.Tensor, torch.Tensor]] = []
                    dispatch: list[JsonObject] = []
                    handles = _register_native_hooks(
                        model,
                        samples,
                        dispatch,
                        path,
                        deps.clock_ns,
                    )
                    try:
                        with torch.inference_mode():
                            output_ids, logits, timing = _generate(
                                model,
                                tokenizer,
                                prompt_ids,
                                generation_spec,
                                deps.clock_ns,
                                samples,
                            )
                    finally:
                        for handle in handles:
                            handle.remove()
                    after = installation.counters
                    delta = _counter_delta(before, after)
                    _reconcile_run(path, len(output_ids), delta)
                    if len(dispatch) != 22 * len(output_ids):
                        raise G3SessionError(
                            "q_proj dispatch timing count does not reconcile"
                        )
                    expected_samples = 22 * (len(output_ids) - 1)
                    if len(samples) != expected_samples:
                        raise G3SessionError(
                            "cached q_proj output capture count does not reconcile"
                        )
                    steps: list[JsonValue] = [
                        {
                            "step_index": index,
                            "token_id": output_ids[index],
                            "logit_count": int(value.numel()),
                            "finite": bool(torch.isfinite(value).all().item()),
                            "logits_sha256": _logits_identity(value),
                            "hybrid_comparison": None,
                        }
                        for index, value in enumerate(logits)
                    ]
                    record: JsonObject = {
                        "phase": phase,
                        "repetition": repetition,
                        "order_index": global_order,
                        "path": path,
                        "output_ids": output_ids,
                        "decoded_text": tokenizer.decode(
                            output_ids, skip_special_tokens=False
                        ),
                        "text_role": TEXT_ROLE,
                        "steps": steps,
                        "timing": {
                            **timing,
                            "q_projection_dispatches": dispatch,
                            "q_projection_ns": sum(
                                _integer(sample["dispatch_ns"], "dispatch_ns")
                                for sample in dispatch
                            ),
                            "native_work_unavailable_reason": (
                                "native boundary does not expose a separate "
                                "work-only timer"
                            ),
                            "native_work_ns": None,
                        },
                        "counters": {
                            "before": _counter_snapshot(before),
                            "after": _counter_snapshot(after),
                            "delta": delta,
                        },
                        "native_output_checks": [],
                    }
                    generation = _Generation(
                        record,
                        logits,
                        samples if path == "hybrid_native" else [],
                    )
                    pair[path] = generation
                    generations.append(generation)
                    runs.append(record)
                    global_order += 1
                _compare_pair(
                    pair["same_q8_reference"],
                    pair["hybrid_native"],
                    logits_atol,
                    logits_rtol,
                )

        validation_before = installation.counters
        for generation in generations:
            if generation.record["path"] != "hybrid_native":
                continue
            generation.record["native_output_checks"] = _validate_native_samples(
                model,
                installation,
                generation.native_samples,
                direct_atol,
                direct_rtol,
            )
        validation_after = installation.counters
        validation_delta = _counter_delta(validation_before, validation_after)
        expected_validation_per_layer = sum(
            len(generation.native_samples) // 22
            for generation in generations
            if generation.record["path"] == "hybrid_native"
        )
        for layer, validation_record in enumerate(validation_delta):
            expected = {
                "layer": layer,
                "forward": expected_validation_per_layer,
                "native_attempt": 0,
                "native_success": 0,
                "native_error": 0,
                "fallback_attempt": expected_validation_per_layer,
                "fallback_success": expected_validation_per_layer,
                "fallback_error": 0,
                "predispatch_error": 0,
                "rejected_closed": 0,
                "in_flight": 0,
                "closed_changed": False,
            }
            if any(
                (
                    validation_record.get(field)
                    if field == "layer"
                    else _object(
                        validation_record.get("values"), "validation values"
                    ).get(field)
                )
                != value
                for field, value in expected.items()
            ):
                raise G3SessionError(
                    f"post-run validation counters failed for layer {layer}"
                )
        validation_counters: JsonObject = {
            "before": _counter_snapshot(validation_before),
            "after": _counter_snapshot(validation_after),
            "delta": validation_delta,
        }
        rejection = _object(execution_spec.get("rejection_policy"), "rejection policy")
        drift = _drift(
            runs,
            _number(rejection.get("drift_ratio_lower"), "drift lower"),
            _number(rejection.get("drift_ratio_upper"), "drift upper"),
        )
    except BaseException as error:
        try:
            installation.close()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "session execution and installation cleanup both failed",
                [error, cleanup_error],
            ) from error
        raise
    else:
        installation.close()
    if not installation.closed or installation.counters.in_flight != 0:
        raise G3SessionError("adapter cleanup did not restore a closed idle model")

    peak_rss = deps.peak_rss_bytes() if deps.peak_rss_bytes is not None else 0
    native_output_check_count = sum(
        len(cast(list[Any], run["native_output_checks"])) for run in runs
    )
    logits_comparison_count = sum(
        len(cast(list[Any], run["steps"]))
        for run in runs
        if run["path"] == "hybrid_native"
    )
    verified_environment = _object(
        verified.evidence.get("environment", {}), "verified environment"
    )
    path_reconciliation: list[JsonObject] = []
    for path in ("same_q8_reference", "hybrid_native"):
        path_runs = [run for run in runs if run["path"] == path]
        cached = sum(len(cast(list[Any], run["output_ids"])) - 1 for run in path_runs)
        path_reconciliation.append(
            {
                "path": path,
                "warmup_runs": 2,
                "measured_runs": 10,
                "prefill_calls_per_layer": 12,
                "cached_decode_calls_per_layer": cached,
                "total_calls_per_layer": cached + 12,
                "pass": True,
            }
        )
    final_counters = installation.counters
    checkout_final = deps.checkout_evidence(request.spec_path)
    if checkout_preflight != checkout_final:
        raise G3SessionError("checkout changed during the generation session")
    if checkout_preflight.get("revision") != preparation_checkout:
        raise G3SessionError("preparation and session checkout revisions differ")
    result: JsonObject = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "format": "decodeforge_g3_generation_session_v1",
        "canonical_identity_algorithm": (
            "sha256_utf8_json_sort_keys_compact_unicode_no_nan_v1"
        ),
        "session_id": request.session_id,
        "session_index": request.session_index,
        "experiment_spec": {
            "path": "benchmarks/g3/spec.json",
            "sha256": spec_sha,
        },
        "bundle_inventory": list(_BUNDLE_INVENTORY),
        "state": {
            "status": "accepted",
            "stage": "completed",
            "accepted": True,
            "rejection_reasons": [],
        },
        "prompt": {
            "text": prompt_text,
            "encoding": "UTF-8",
            "tokenizer_method": "AutoTokenizer.from_pretrained-local-files-only",
            "add_special_tokens": True,
            "input_ids": prompt_ids,
            "attention_mask": [1] * len(prompt_ids),
        },
        "provenance": {
            "checkout": checkout_preflight,
            "model": assets_record["source"],
            "bridge_library": _object(
                verified.evidence.get("bridge_library"), "bridge library evidence"
            ),
            "asset_inventory_identity": assets_record["aggregate_identity"],
            "rebuild_commands": {
                "build_bridge": _string(
                    verified.evidence.get("bridge_rebuild_command"),
                    "bridge rebuild command",
                ),
                "prepare_assets": shlex.join(
                    _string(value, "preparation argument") for value in preparation_argv
                ),
                "run_session": run_rebuild_command,
            },
        },
        "environment": {
            **verified_environment,
            **loaded_environment,
            "runner_process_id": os.getpid(),
            "torch_num_threads": _integer(
                generation_spec.get("torch_num_threads"), "threads"
            ),
            "torch_num_interop_threads": _integer(
                generation_spec.get("torch_num_interop_threads"), "interop threads"
            ),
            "hf_hub_offline": True,
            "transformers_offline": True,
            "local_files_only": True,
        },
        "assets": assets_record,
        "offline_preparation": offline_preparation,
        "startup_timings": {
            "model_load_ns": model_load_ns,
            "tokenizer_load_ns": tokenizer_load_ns,
            "install_ns": install_end - install_start,
            "cold_startup_ns": install_end - request.process_start_ns,
            "baseline_rss_bytes": baseline_rss,
            "peak_rss_bytes": peak_rss,
        },
        "runs": runs,
        "correctness": {
            "direct_operator": {
                "absolute_tolerance": direct_atol,
                "relative_tolerance": direct_rtol,
                "preinstallation_layers": direct_checks,
                "native_output_check_count": native_output_check_count,
                "pass": True,
            },
            "model_logits": {
                "absolute_tolerance": logits_atol,
                "relative_tolerance": logits_rtol,
                "comparison_count": logits_comparison_count,
                "pass": True,
            },
            "tokens": {
                "policy": "exact_token_id_match",
                "comparison_count": 12,
                "pass": True,
            },
            "overall_pass": True,
        },
        "reconciliation": {
            "post_run_native_validation_counters": validation_counters,
            "installation": {
                "installed_modules": final_counters.installed_modules,
                "restored_modules": final_counters.restored_modules,
                "live_adapters": final_counters.live_adapters,
                "in_flight": final_counters.in_flight,
                "closed": installation.closed,
            },
            "paths": path_reconciliation,
            "pass": True,
        },
        "drift": drift,
    }
    diagnostics = validate_data(
        result, "g3-generation-session", document_name="session-result.json"
    )
    if diagnostics:
        raise G3SessionError(f"session evidence schema failure: {diagnostics[0]}")
    return result


__all__ = [
    "PROTOCOL_ID",
    "SPEC_SHA256",
    "G3SessionError",
    "SessionDependencies",
    "SessionRequest",
    "default_dependencies",
    "publish_new_json",
    "require_external_session_output",
    "run_session",
]

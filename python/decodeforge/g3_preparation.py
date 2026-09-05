"""Closed, separately timed G3 asset-preparation receipts.

The preparation receipt is deliberately separate from a generation session.
It records the monotonic interval around the already-atomic Rust preparation
command and can only be published after the resulting inventory is verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

JsonObject: TypeAlias = dict[str, Any]

RECEIPT_FORMAT: Final = "decodeforge_g3_offline_preparation_receipt_v1"
PROTOCOL_ID: Final = "g3-tinyllama-qproj-generation-v1"
CANONICAL_ASSET_INVENTORY_IDENTITY: Final = (
    "sha256:f659b26572357af84a5e5b66138331a2e35c319c5c9b8300cf81f7ea217ae0de"
)
_IDENTITY_DOMAIN: Final = b"DecodeForge/g3-offline-preparation-receipt/v1\0"
_IDENTITY_PREFIX: Final = "sha256:"
_MAX_RECEIPT_BYTES: Final = 1024 * 1024
_MAX_INVENTORY_BYTES: Final = 256 * 1024
_MAX_TOOL_BYTES: Final = 128 * 1024 * 1024
_GIT_TIMEOUT_SECONDS: Final = 10.0
_TOOL_TIMEOUT_SECONDS: Final = 3600.0
_DIAGNOSTIC_BYTES: Final = 16 * 1024
_TOTAL_PACKED_BYTES: Final = 103_809_024
_TOTAL_FALLBACK_BYTES: Final = 369_098_752
_SOURCE: Final[JsonObject] = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "filename": "model.safetensors",
    "bytes": 2_200_119_864,
    "identity": (
        "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
    ),
}
_OUTPUT: Final[JsonObject] = {
    "asset_inventory_identity": CANONICAL_ASSET_INVENTORY_IDENTITY,
    "layer_count": 22,
    "total_packed_bytes": _TOTAL_PACKED_BYTES,
    "total_fallback_bytes": _TOTAL_FALLBACK_BYTES,
}


class G3PreparationError(RuntimeError):
    """The timed preparation or its closed receipt is invalid."""


@dataclass(frozen=True)
class VerifiedPreparationReceipt:
    """Validated receipt data retained for checkout/tool/path binding."""

    checkout_revision: str
    command_argv: tuple[str, ...]
    tool_executable_identity: str
    receipt_identity: str
    elapsed_ns: int
    asset_inventory_identity: str

    def session_projection(self) -> JsonObject:
        """Return exactly the fields admitted by the G3 session schema."""

        return {
            "source": "separately_captured_prepare_command",
            "receipt_identity": self.receipt_identity,
            "elapsed_ns": self.elapsed_ns,
            "asset_inventory_identity": self.asset_inventory_identity,
        }


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_directory(
    name: str | os.PathLike[str], *, directory_fd: int | None = None
) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow == 0:
        raise G3PreparationError("secure receipt I/O requires O_NOFOLLOW")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | nofollow,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise G3PreparationError("unable to open directory without symlinks") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise G3PreparationError("receipt parent component is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_parent(path: Path) -> tuple[int, str]:
    if not path.name or path.name in {".", ".."}:
        raise G3PreparationError("receipt must name an explicit file")
    parent = path.parent
    parts = parent.parts
    descriptor = _open_directory("/" if parent.is_absolute() else ".")
    try:
        start = 1 if parent.is_absolute() else 0
        for component in parts[start:]:
            if component in {"", "."}:
                continue
            if component == "..":
                raise G3PreparationError("receipt path may not traverse a parent")
            child = _open_directory(component, directory_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, path.name
    except BaseException:
        os.close(descriptor)
        raise


def _read_leaf(
    path: Path, maximum: int, label: str, *, require_executable: bool = False
) -> bytes:
    parent_fd, name = _open_parent(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise G3PreparationError(
                f"unable to open {label} without symlinks"
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise G3PreparationError(f"{label} is not a regular file")
            if before.st_nlink != 1:
                raise G3PreparationError(f"{label} must have exactly one link")
            if require_executable and not (
                before.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            ):
                raise G3PreparationError(f"{label} is not executable")
            if before.st_size <= 0 or before.st_size > maximum:
                raise G3PreparationError(f"{label} byte extent is outside its bound")
            remaining = before.st_size
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise G3PreparationError(f"{label} changed during read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise G3PreparationError(f"{label} grew during read")
            after = os.fstat(descriptor)
            if _fingerprint(before) != _fingerprint(after):
                raise G3PreparationError(f"{label} changed during read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _json(raw: bytes, label: str) -> JsonObject:
    def closed_object(pairs: list[tuple[str, Any]]) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=closed_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise G3PreparationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise G3PreparationError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G3PreparationError("receipt is not canonical JSON data") from error


def receipt_identity(unsigned: Mapping[str, Any]) -> str:
    """Return the domain-separated identity of a receipt without its identity field."""

    return (
        _IDENTITY_PREFIX
        + hashlib.sha256(_IDENTITY_DOMAIN + _canonical_bytes(unsigned)).hexdigest()
    )


def _identity(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith(_IDENTITY_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise G3PreparationError(f"{label} is not a SHA-256 identity")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise G3PreparationError(f"{label} must be an integer")
    return cast(int, value)


def _object(value: Any, fields: set[str], label: str) -> JsonObject:
    if not isinstance(value, dict) or set(value) != fields:
        raise G3PreparationError(f"{label} is not a closed object")
    return cast(JsonObject, value)


def _validate_receipt(value: JsonObject) -> JsonObject:
    fields = {
        "schema_version",
        "format",
        "protocol_id",
        "checkout",
        "command",
        "source",
        "tool",
        "output",
        "timing",
        "receipt_identity",
    }
    if set(value) != fields:
        raise G3PreparationError("preparation receipt is not a closed document")
    claimed = _identity(value["receipt_identity"], "receipt identity")
    unsigned = {key: item for key, item in value.items() if key != "receipt_identity"}
    if claimed != receipt_identity(unsigned):
        raise G3PreparationError("preparation receipt identity mismatch")
    if (
        value["schema_version"] != 1
        or value["format"] != RECEIPT_FORMAT
        or value["protocol_id"] != PROTOCOL_ID
    ):
        raise G3PreparationError("preparation receipt protocol mismatch")

    checkout = _object(value["checkout"], {"revision", "dirty"}, "checkout")
    revision = checkout["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or checkout["dirty"] is not False
    ):
        raise G3PreparationError("preparation checkout is not exact and clean")

    command = _object(value["command"], {"argv"}, "command")
    argv = command["argv"]
    if (
        not isinstance(argv, list)
        or len(argv) != 5
        or not all(
            isinstance(argument, str)
            and 0 < len(argument) <= 4096
            and "\0" not in argument
            for argument in argv
        )
        or sum(len(cast(str, argument)) for argument in argv) > 16 * 1024
        or argv[1] != "--source"
        or argv[3] != "--output"
        or not all(os.path.isabs(cast(str, argv[index])) for index in (0, 2, 4))
        or any(
            os.path.normpath(cast(str, argv[index])) != argv[index]
            for index in (0, 2, 4)
        )
        or Path(cast(str, argv[0])).name != "decodeforge-prepare-qproj"
    ):
        raise G3PreparationError("preparation command is invalid")

    source = _object(
        value["source"],
        {"model_id", "revision", "filename", "bytes", "identity"},
        "source",
    )
    if source != _SOURCE:
        raise G3PreparationError("preparation source is not canonical")

    tool = _object(value["tool"], {"name", "version", "executable_identity"}, "tool")
    if tool["name"] != "decodeforge-prepare-qproj" or tool["version"] != "0.1.0":
        raise G3PreparationError("preparation tool is not canonical")
    _identity(tool["executable_identity"], "tool executable identity")

    output = _object(
        value["output"],
        {
            "asset_inventory_identity",
            "layer_count",
            "total_packed_bytes",
            "total_fallback_bytes",
        },
        "output",
    )
    if output != _OUTPUT:
        raise G3PreparationError("preparation output is not canonical")

    timing = _object(
        value["timing"], {"clock", "start_ns", "stop_ns", "elapsed_ns"}, "timing"
    )
    start = _integer(timing["start_ns"], "preparation start")
    stop = _integer(timing["stop_ns"], "preparation stop")
    elapsed = _integer(timing["elapsed_ns"], "preparation elapsed")
    if (
        timing["clock"] != "time.perf_counter_ns"
        or start < 0
        or stop < 0
        or elapsed <= 0
        or stop - start != elapsed
    ):
        raise G3PreparationError("preparation timing is invalid")
    return value


def verify_preparation_receipt(
    path: str | os.PathLike[str],
) -> VerifiedPreparationReceipt:
    """Verify a bounded receipt while retaining execution provenance."""

    value = load_preparation_receipt_document(path)
    timing = cast(JsonObject, value["timing"])
    output = cast(JsonObject, value["output"])
    checkout = cast(JsonObject, value["checkout"])
    command = cast(JsonObject, value["command"])
    tool = cast(JsonObject, value["tool"])
    argv = cast(list[str], command["argv"])
    tool_bytes = _read_leaf(
        Path(argv[0]), _MAX_TOOL_BYTES, "preparation tool", require_executable=True
    )
    observed_tool_identity = _IDENTITY_PREFIX + hashlib.sha256(tool_bytes).hexdigest()
    if observed_tool_identity != tool["executable_identity"]:
        raise G3PreparationError("preparation tool executable identity mismatch")
    return VerifiedPreparationReceipt(
        checkout_revision=cast(str, checkout["revision"]),
        command_argv=tuple(argv),
        tool_executable_identity=cast(str, tool["executable_identity"]),
        receipt_identity=cast(str, value["receipt_identity"]),
        elapsed_ns=cast(int, timing["elapsed_ns"]),
        asset_inventory_identity=cast(str, output["asset_inventory_identity"]),
    )


def load_preparation_receipt_document(
    path: str | os.PathLike[str],
) -> JsonObject:
    """Return an independent fully validated receipt document.

    This portable form validates the closed receipt and its domain-separated
    identity but deliberately does not require the recorded executable to
    remain present. Use :func:`verify_preparation_receipt` at capture/session
    time when the stronger live executable rehash is required.
    """

    value = _json(_read_leaf(Path(path), _MAX_RECEIPT_BYTES, "receipt"), "receipt")
    return validate_preparation_receipt_document(value)


def validate_preparation_receipt_document(value: Mapping[str, Any]) -> JsonObject:
    """Validate and detach one already parsed portable receipt document."""

    if not isinstance(value, dict):
        raise G3PreparationError("preparation receipt must be a JSON object")
    detached = deepcopy(dict(value))
    return _validate_receipt(detached)


def load_preparation_receipt(path: str | os.PathLike[str]) -> JsonObject:
    """Verify a bounded receipt and return its frozen session projection."""

    return verify_preparation_receipt(path).session_projection()


def _publish_new(path: Path, value: Mapping[str, Any]) -> os.stat_result:
    parent_fd, name = _open_parent(path)
    temporary = f".decodeforge-receipt-{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    linked = False
    temporary_exists = False
    published: os.stat_result | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_exists = True
        data = (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise G3PreparationError("receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.link(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary, dir_fd=parent_fd)
        temporary_exists = False
        published = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if published.st_nlink != 1 or _fingerprint(published) != _fingerprint(named):
            os.unlink(name, dir_fd=parent_fd)
            linked = False
            os.fsync(parent_fd)
            raise G3PreparationError("published receipt identity changed")
        os.fsync(parent_fd)
    except FileExistsError as error:
        raise G3PreparationError("receipt path already exists") from error
    except OSError as error:
        if linked:
            try:
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise G3PreparationError("unable to publish receipt atomically") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        os.close(parent_fd)
    if published is None:
        raise G3PreparationError("receipt publication did not complete")
    return published


def _remove_published(path: Path, expected: os.stat_result) -> None:
    parent_fd, name = _open_parent(path)
    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or _fingerprint(observed) != _fingerprint(expected)
        ):
            raise G3PreparationError("published receipt changed before rollback")
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as error:
        raise G3PreparationError("unable to roll back published receipt") from error
    finally:
        os.close(parent_fd)


def _require_new_receipt(path: Path) -> None:
    parent_fd, name = _open_parent(path)
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as error:
            raise G3PreparationError("unable to inspect receipt destination") from error
        raise G3PreparationError("receipt path already exists")
    finally:
        os.close(parent_fd)


def _index_is_unflagged(output: bytes) -> bool:
    """Reject index flags that can hide tracked-file changes from status."""

    if not output:
        return True
    if not output.endswith(b"\0"):
        raise G3PreparationError("checkout index metadata is malformed")
    permitted_tags = {
        b"H",
        b"h",
        b"S",
        b"s",
        b"M",
        b"m",
        b"R",
        b"r",
        b"C",
        b"c",
        b"K",
        b"k",
        b"?",
    }
    for record in output[:-1].split(b"\0"):
        if (
            len(record) < 3
            or record[1:2] != b" "
            or not record[2:]
            or record[:1] not in permitted_tags
        ):
            raise G3PreparationError("checkout index metadata is malformed")
        tag = record[:1]
        if tag.lower() == b"s" or tag.islower():
            return False
    return True


def _checkout_state(checkout: Path) -> tuple[str, bool]:
    def git_bytes(*arguments: str) -> bytes:
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(
                    ["git", "-C", os.fspath(checkout), *arguments],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
                    timeout=_GIT_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise G3PreparationError("checkout inspection timed out") from error
            except OSError as error:
                raise G3PreparationError(
                    "unable to launch checkout inspection"
                ) from error
            output.seek(0)
            stdout = output.read(1024 * 1024 + 1)
        if result.returncode != 0 or len(stdout) > 1024 * 1024:
            raise G3PreparationError("unable to bind the preparation checkout")
        return stdout

    def git_text(*arguments: str) -> str:
        try:
            return git_bytes(*arguments).decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise G3PreparationError("checkout metadata is not UTF-8") from error

    root = Path(git_text("rev-parse", "--show-toplevel"))
    try:
        if root.resolve(strict=True) != checkout.resolve(strict=True):
            raise G3PreparationError("checkout does not name its Git root")
    except OSError as error:
        raise G3PreparationError("checkout root is unavailable") from error
    revision = git_text("rev-parse", "HEAD")
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise G3PreparationError("checkout revision is not a full object ID")
    dirty = bool(
        git_bytes(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=no",
            "--ignore-submodules=none",
        )
    )
    if not _index_is_unflagged(git_bytes("ls-files", "-v", "-z", "--full-name")):
        raise G3PreparationError("checkout index contains hidden tracked-file flags")
    return revision, dirty


def _require_checkout_producer(checkout: Path) -> None:
    try:
        expected = (
            checkout.resolve(strict=True)
            / "python"
            / "decodeforge"
            / "g3_preparation.py"
        )
        actual = Path(__file__).resolve(strict=True)
    except OSError as error:
        raise G3PreparationError("preparation producer is unavailable") from error
    if actual != expected:
        raise G3PreparationError("preparation producer is not bound to the checkout")


def _run_tool(arguments: Sequence[str]) -> None:
    # A fixed allowlist prevents DYLD/LD/Python launch injection into the exact
    # snapshotted executable. stderr is retained only within a small bound.
    environment = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}
    with tempfile.TemporaryFile() as diagnostic:
        try:
            result = subprocess.run(
                arguments,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=diagnostic,
                env=environment,
                timeout=_TOOL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise G3PreparationError("q_proj preparation command timed out") from error
        except OSError as error:
            raise G3PreparationError(
                "unable to launch q_proj preparation command"
            ) from error
        diagnostic.seek(0)
        detail = diagnostic.read(_DIAGNOSTIC_BYTES + 1)
    if result.returncode != 0:
        suffix = ""
        if detail:
            bounded = (
                detail[:_DIAGNOSTIC_BYTES].decode("utf-8", errors="replace").strip()
            )
            marker = " [truncated]" if len(detail) > _DIAGNOSTIC_BYTES else ""
            suffix = f": {bounded}{marker}"
        message = "q_proj preparation or verification failed"
        raise G3PreparationError(f"{message} (exit {result.returncode}){suffix}")


def _inventory_output(output: Path) -> JsonObject:
    value = _json(
        _read_leaf(output / "inventory.json", _MAX_INVENTORY_BYTES, "asset inventory"),
        "asset inventory",
    )
    expected_fields = {
        "schema_version",
        "format",
        "source",
        "layer_count",
        "entries",
        "total_packed_bytes",
        "total_fallback_bytes",
        "aggregate_identity",
    }
    if set(value) != expected_fields:
        raise G3PreparationError("asset inventory is not a closed document")
    observed = {
        "asset_inventory_identity": value["aggregate_identity"],
        "layer_count": value["layer_count"],
        "total_packed_bytes": value["total_packed_bytes"],
        "total_fallback_bytes": value["total_fallback_bytes"],
    }
    if (
        value["schema_version"] != 1
        or value["format"] != "decodeforge_q_proj_inventory_v1"
        or value["source"] != _SOURCE
        or not isinstance(value["entries"], list)
        or len(value["entries"]) != 22
        or observed != _OUTPUT
    ):
        raise G3PreparationError("prepared asset inventory is not canonical")
    return observed


def capture_preparation_receipt(
    *,
    checkout: Path,
    source: Path,
    output: Path,
    receipt: Path,
    prepare_tool: Path,
    clock: Callable[[], int] = time.perf_counter_ns,
    run_tool: Callable[[Sequence[str]], None] = _run_tool,
    checkout_state: Callable[[Path], tuple[str, bool]] = _checkout_state,
) -> JsonObject:
    """Prepare, verify, and atomically publish one new closed receipt.

    The timed interval begins on the statement immediately before the prepare
    process call and stops on the statement immediately after it returns.  The
    Rust command returns only after its no-replace asset-directory rename and
    parent-directory sync have completed.
    """

    checkout = Path(checkout)
    source = Path(source)
    output = Path(output)
    receipt = Path(receipt)
    prepare_tool = Path(prepare_tool)
    if output.exists() or output.is_symlink():
        raise G3PreparationError("asset output must be a new path")
    output_absolute = Path(os.path.abspath(output))
    receipt_absolute = Path(os.path.abspath(receipt))
    try:
        checkout_absolute = checkout.resolve(strict=True)
    except OSError as error:
        raise G3PreparationError("preparation checkout is unavailable") from error
    try:
        receipt_absolute.relative_to(output_absolute)
    except ValueError:
        pass
    else:
        raise G3PreparationError("receipt must be outside the prepared asset directory")
    try:
        receipt_absolute.relative_to(checkout_absolute)
    except ValueError:
        pass
    else:
        raise G3PreparationError("receipt must be outside the preparation checkout")
    try:
        output_absolute.relative_to(checkout_absolute)
    except ValueError:
        pass
    else:
        raise G3PreparationError(
            "asset output must be outside the preparation checkout"
        )
    _require_new_receipt(receipt_absolute)

    _require_checkout_producer(checkout)
    revision, dirty = checkout_state(checkout)
    if dirty:
        raise G3PreparationError("preparation checkout must be clean")
    prepare_tool_absolute = Path(os.path.abspath(prepare_tool))
    tool_bytes = _read_leaf(
        prepare_tool_absolute,
        _MAX_TOOL_BYTES,
        "preparation tool",
        require_executable=True,
    )
    executable_identity = _IDENTITY_PREFIX + hashlib.sha256(tool_bytes).hexdigest()
    logical_argv = [
        os.fspath(prepare_tool_absolute),
        "--source",
        os.fspath(source_absolute := Path(os.path.abspath(source))),
        "--output",
        os.fspath(output_absolute),
    ]

    with tempfile.TemporaryDirectory(prefix="decodeforge-g3-tool-") as temporary:
        snapshot = Path(temporary) / "decodeforge-prepare-qproj"
        descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
        try:
            view = memoryview(tool_bytes)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise G3PreparationError("tool snapshot write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(snapshot, 0o700)

        prepare_arguments = [
            os.fspath(snapshot),
            "--source",
            os.fspath(source_absolute),
            "--output",
            os.fspath(output_absolute),
        ]
        start_ns = clock()
        run_tool(prepare_arguments)
        stop_ns = clock()
        elapsed_ns = stop_ns - start_ns
        if start_ns < 0 or stop_ns < 0 or elapsed_ns <= 0:
            raise G3PreparationError("preparation clock did not advance")

        run_tool([os.fspath(snapshot), "--verify", os.fspath(output_absolute)])

    observed_output = _inventory_output(output_absolute)
    final_revision, final_dirty = checkout_state(checkout)
    if final_revision != revision or final_dirty:
        raise G3PreparationError("checkout changed during preparation")
    unsigned: JsonObject = {
        "schema_version": 1,
        "format": RECEIPT_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "checkout": {"revision": revision, "dirty": False},
        "command": {"argv": logical_argv},
        "source": dict(_SOURCE),
        "tool": {
            "name": "decodeforge-prepare-qproj",
            "version": "0.1.0",
            "executable_identity": executable_identity,
        },
        "output": observed_output,
        "timing": {
            "clock": "time.perf_counter_ns",
            "start_ns": start_ns,
            "stop_ns": stop_ns,
            "elapsed_ns": elapsed_ns,
        },
    }
    value = {**unsigned, "receipt_identity": receipt_identity(unsigned)}
    _validate_receipt(value)
    published = _publish_new(receipt_absolute, value)
    try:
        published_revision, published_dirty = checkout_state(checkout)
        if published_revision != revision or published_dirty:
            raise G3PreparationError("checkout changed during receipt publication")
        return load_preparation_receipt(receipt_absolute)
    except BaseException as error:
        try:
            _remove_published(receipt_absolute, published)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "receipt verification and rollback both failed", [error, cleanup_error]
            ) from error
        raise

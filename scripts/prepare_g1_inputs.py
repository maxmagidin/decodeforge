#!/usr/bin/env python3
"""Extract and pin the one TinyLlama BF16 tensor used by G1.

This command intentionally has no model downloader.  The caller supplies a
local, already obtained ``model.safetensors`` file; the command first hashes
the complete file, then lazily reads only the pinned tensor through
``safetensors``/PyTorch, and finally atomically writes a small one-tensor file.
The output metadata contains source identities, never a local path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

MODEL_ID: Final = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION: Final = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
MODEL_FILENAME: Final = "model.safetensors"
MODEL_SIZE_BYTES: Final = 2_200_119_864
MODEL_SHA256: Final = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
TENSOR_NAME: Final = "model.layers.0.self_attn.q_proj.weight"
TENSOR_DTYPE: Final = "BF16"
TENSOR_SHAPE: Final = (2048, 2048)
TENSOR_BYTES: Final = TENSOR_SHAPE[0] * TENSOR_SHAPE[1] * 2
TENSOR_SHA256: Final = (
    "5abf98c51f903941a1592f3df83e2e56ca7149252f5d6665c7662927c83008ac"
)
METADATA_KEYS: Final = (
    "model",
    "revision",
    "model_filename",
    "model_size_bytes",
    "model_sha256",
    "tensor_name",
    "tensor_sha256",
)


class PrepareError(RuntimeError):
    """A source or output failed the closed G1 preparation contract."""


@dataclass(frozen=True, slots=True)
class PinnedSource:
    """Expected immutable identity of one source model file."""

    model: str = MODEL_ID
    revision: str = MODEL_REVISION
    model_filename: str = MODEL_FILENAME
    model_size_bytes: int = MODEL_SIZE_BYTES
    model_sha256: str = MODEL_SHA256
    tensor_name: str = TENSOR_NAME
    tensor_sha256: str = TENSOR_SHA256


PINNED_SOURCE: Final = PinnedSource()


def _open_safetensors(path: Path) -> Any:
    """Open a safetensors file through its untyped extension boundary."""

    return safe_open(path, framework="pt", device="cpu")  # type: ignore[no-untyped-call]


def _regular_file_signature(path: Path) -> tuple[int, int, int, int]:
    """Return a bounded identity for a non-symlink regular file."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise PrepareError(f"cannot stat source file: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise PrepareError("source file must not be a symbolic link")
    if not stat.S_ISREG(metadata.st_mode):
        raise PrepareError("source path must be a regular file")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _paths_alias(source: Path, output: Path) -> bool:
    """Return whether output would replace source through a path alias."""

    try:
        source_resolved = source.resolve(strict=True)
        output_resolved = output.resolve(strict=False)
    except OSError as error:
        raise PrepareError(f"cannot resolve source or output path: {error}") from error
    if source_resolved == output_resolved:
        return True
    try:
        return output.exists() and os.path.samefile(source, output)
    except OSError as error:
        raise PrepareError(
            f"cannot compare source and output paths: {error}"
        ) from error


def _sha256_file(path: Path) -> tuple[int, str, tuple[int, int, int, int]]:
    """Hash a complete source file and return its post-read signature."""

    before = _regular_file_signature(path)
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise PrepareError(f"cannot read source file: {error}") from error
    after = _regular_file_signature(path)
    if before != after or total != after[2]:
        raise PrepareError("source file changed while it was hashed")
    return total, digest.hexdigest(), after


def _raw_bf16_bytes(tensor: object) -> bytes:
    """Return the exact little-endian BF16 storage bytes of a CPU tensor."""

    # Importing torch here keeps the module import usable for CLI help and
    # makes the type/runtime dependency explicit at the point of extraction.
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise PrepareError("safetensors did not return a PyTorch tensor")
    if tensor.dtype is not torch.bfloat16:
        raise PrepareError(f"tensor dtype must be BF16, got {tensor.dtype}")
    if tuple(tensor.shape) != TENSOR_SHAPE:
        raise PrepareError(
            f"tensor shape must be {list(TENSOR_SHAPE)}, got {list(tensor.shape)}"
        )
    if tensor.device.type != "cpu":
        raise PrepareError("tensor extraction must remain on the CPU")
    try:
        words = tensor.detach().contiguous().view(torch.uint16).numpy()
        # The supported host is little-endian.  Make that requirement explicit
        # instead of allowing a native-endian view to silently change identity.
        if words.dtype.byteorder not in ("<", "=", "|"):
            words = words.astype("<u2", copy=False)
        raw = words.astype("<u2", copy=False).tobytes(order="C")
    except (RuntimeError, TypeError, ValueError) as error:
        raise PrepareError(f"could not read BF16 storage bytes: {error}") from error
    if len(raw) != TENSOR_BYTES:
        raise PrepareError(
            f"tensor has {len(raw)} storage bytes; expected {TENSOR_BYTES}"
        )
    return raw


def _source_metadata(source: PinnedSource) -> dict[str, str]:
    """Build the exact seven string metadata fields consumed by Rust."""

    return {
        "model": source.model,
        "revision": source.revision,
        "model_filename": source.model_filename,
        "model_size_bytes": str(source.model_size_bytes),
        "model_sha256": source.model_sha256,
        "tensor_name": source.tensor_name,
        "tensor_sha256": source.tensor_sha256,
    }


def _validate_source_identity(
    path: Path, source: PinnedSource
) -> tuple[tuple[int, int, int, int], bytes]:
    """Verify the complete model file and return the pinned tensor bytes."""

    size, digest, signature = _sha256_file(path)
    if size != source.model_size_bytes:
        raise PrepareError(
            f"model file size mismatch: expected {source.model_size_bytes}, got {size}"
        )
    if digest != source.model_sha256:
        raise PrepareError(
            f"model file SHA-256 mismatch: expected {source.model_sha256}, got {digest}"
        )

    try:
        with _open_safetensors(path) as tensors:
            tensor_names = tensors.keys()
            if source.tensor_name not in tensor_names:
                raise PrepareError(f"missing tensor {source.tensor_name!r}")
            tensor = tensors.get_tensor(source.tensor_name)
            raw = _raw_bf16_bytes(tensor)
    except PrepareError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, SafetensorError) as error:
        raise PrepareError(
            f"could not open pinned safetensors tensor: {error}"
        ) from error

    actual_tensor_hash = hashlib.sha256(raw).hexdigest()
    if actual_tensor_hash != source.tensor_sha256:
        raise PrepareError(
            "raw tensor SHA-256 mismatch: "
            f"expected {source.tensor_sha256}, got {actual_tensor_hash}"
        )
    if _regular_file_signature(path) != signature:
        raise PrepareError("source file changed while the tensor was extracted")
    return signature, raw


def _write_atomic(output: Path, raw_tensor: bytes, metadata: dict[str, str]) -> None:
    """Serialize one tensor into a same-directory temporary file and replace."""

    if output.name in ("", ".", ".."):
        raise PrepareError("output must name an explicit file")
    parent = output.parent if output.parent != Path("") else Path(".")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.tmp-", dir=parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            import torch

            storage = torch.frombuffer(bytearray(raw_tensor), dtype=torch.uint16).view(
                torch.bfloat16
            )
            storage = storage.reshape(TENSOR_SHAPE).contiguous()
            save_file({TENSOR_NAME: storage}, str(temporary), metadata=metadata)
            _canonicalize_safetensors_header(temporary)
            with temporary.open("rb") as written:
                os.fsync(written.fileno())
            os.replace(temporary, output)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
    except (OSError, RuntimeError, TypeError, ValueError, SafetensorError) as error:
        raise PrepareError(f"could not atomically write output: {error}") from error


def _canonicalize_safetensors_header(path: Path) -> None:
    """Sort a library-written header so equal prepared inputs are byte-stable."""

    try:
        with path.open("r+b") as artifact:
            length_bytes = artifact.read(8)
            if len(length_bytes) != 8:
                raise PrepareError("emitted safetensors header is truncated")
            header_length = int.from_bytes(length_bytes, "little")
            if header_length <= 0 or header_length > 1024 * 1024:
                raise PrepareError("emitted safetensors header length is invalid")
            encoded_header = artifact.read(header_length)
            if len(encoded_header) != header_length:
                raise PrepareError("emitted safetensors header is truncated")
            header = json.loads(encoded_header)
            canonical = json.dumps(
                header,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            if len(canonical) > header_length:
                raise PrepareError(
                    "canonical safetensors header exceeds its allocation"
                )
            artifact.seek(8)
            artifact.write(canonical)
            artifact.write(b" " * (header_length - len(canonical)))
            artifact.flush()
    except PrepareError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrepareError(
            f"could not canonicalize safetensors header: {error}"
        ) from error


def _verify_output(output: Path, source: PinnedSource) -> None:
    """Verify the emitted file is exactly one pinned BF16 tensor."""

    expected_metadata = _source_metadata(source)
    try:
        with _open_safetensors(output) as tensors:
            if list(tensors.keys()) != [TENSOR_NAME]:
                raise PrepareError("output must contain exactly the pinned tensor")
            if tensors.metadata() != expected_metadata:
                raise PrepareError("output metadata does not match the pinned fields")
            raw = _raw_bf16_bytes(tensors.get_tensor(TENSOR_NAME))
    except PrepareError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, SafetensorError) as error:
        raise PrepareError(f"could not verify emitted safetensors: {error}") from error
    if hashlib.sha256(raw).hexdigest() != source.tensor_sha256:
        raise PrepareError("emitted tensor identity does not match the pinned source")


def prepare_g1_input(
    model_path: Path,
    output_path: Path,
    *,
    source: PinnedSource = PINNED_SOURCE,
) -> None:
    """Prepare one immutable, portable G1 input artifact."""

    if _paths_alias(model_path, output_path):
        raise PrepareError("source and output paths must be different")
    _signature, raw_tensor = _validate_source_identity(model_path, source)
    # Recheck after the long full-model read so a pre-existing hard link or a
    # changed parent-directory symlink cannot turn the output into the source.
    if _paths_alias(model_path, output_path):
        raise PrepareError("source and output paths must be different")
    _write_atomic(output_path, raw_tensor, _source_metadata(source))
    _verify_output(output_path, source)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "verify a pinned local TinyLlama safetensors file and emit one "
            "BF16 q_proj tensor; this command never downloads"
        )
    )
    parser.add_argument(
        "--weights",
        "--model",
        dest="model_path",
        type=Path,
        required=True,
        help="existing full model.safetensors path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output one-tensor safetensors path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        prepare_g1_input(args.model_path, args.output)
    except PrepareError as error:
        print(f"prepare-g1-inputs: error: {error}", file=sys.stderr)
        return 2
    print("prepare-g1-inputs: verified pinned source and emitted one-tensor artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

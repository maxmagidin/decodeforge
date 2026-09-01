"""Focused tests for the pinned G1 safetensors preparation boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from decodeforge.contracts import ROOT
from safetensors.torch import save_file

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "decodeforge_test_prepare_g1_inputs", ROOT / "scripts" / "prepare_g1_inputs.py"
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
prepare: Any = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = prepare
_SCRIPT_SPEC.loader.exec_module(prepare)


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def _source_spec(model_path: Path, tensor: torch.Tensor) -> Any:
    model_bytes = model_path.read_bytes()
    return prepare.PinnedSource(
        model="test/TinyLlama",
        revision="0123456789abcdef0123456789abcdef01234567",
        model_filename="model.safetensors",
        model_size_bytes=len(model_bytes),
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        tensor_name=prepare.TENSOR_NAME,
        tensor_sha256=hashlib.sha256(_raw_bytes(tensor)).hexdigest(),
    )


@pytest.fixture
def source_artifact(tmp_path: Path) -> tuple[Path, Any, torch.Tensor]:
    tensor = torch.zeros(prepare.TENSOR_SHAPE, dtype=torch.bfloat16)
    model_path = tmp_path / "model.safetensors"
    save_file({prepare.TENSOR_NAME: tensor}, str(model_path))
    return model_path, _source_spec(model_path, tensor), tensor


def test_prepare_verifies_source_and_emits_exact_one_tensor(
    source_artifact: tuple[Path, Any, torch.Tensor], tmp_path: Path
) -> None:
    model_path, source, tensor = source_artifact
    output_path = tmp_path / "prepared.safetensors"

    prepare.prepare_g1_input(model_path, output_path, source=source)

    with prepare.safe_open(output_path, framework="pt", device="cpu") as output:
        assert list(output.keys()) == [prepare.TENSOR_NAME]
        assert output.metadata() == prepare._source_metadata(source)
        observed = output.get_tensor(prepare.TENSOR_NAME)
        assert observed.dtype is torch.bfloat16
        assert tuple(observed.shape) == prepare.TENSOR_SHAPE
        assert hashlib.sha256(_raw_bytes(observed)).hexdigest() == source.tensor_sha256
    assert output_path.stat().st_size < 16 * 1024 * 1024
    assert str(model_path).encode() not in output_path.read_bytes()
    assert not list(tmp_path.glob(".prepared.safetensors.tmp-*"))
    assert tensor.numel() == prepare.TENSOR_SHAPE[0] * prepare.TENSOR_SHAPE[1]


def test_prepare_rejects_a_changed_or_wrongly_hashed_complete_file(
    source_artifact: tuple[Path, Any, torch.Tensor], tmp_path: Path
) -> None:
    model_path, source, _tensor = source_artifact
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(prepare.PrepareError, match="size mismatch|SHA-256 mismatch"):
        prepare.prepare_g1_input(
            model_path, tmp_path / "prepared.safetensors", source=source
        )
    assert not (tmp_path / "prepared.safetensors").exists()


def test_prepare_rejects_missing_or_wrong_shape_tensor(tmp_path: Path) -> None:
    wrong = torch.zeros((2048, 2047), dtype=torch.bfloat16)
    missing_path = tmp_path / "missing.safetensors"
    save_file({"other": wrong}, str(missing_path))
    missing_source = _source_spec(missing_path, wrong)
    with pytest.raises(prepare.PrepareError, match="missing tensor"):
        prepare.prepare_g1_input(
            missing_path, tmp_path / "missing-output.safetensors", source=missing_source
        )

    wrong_path = tmp_path / "wrong.safetensors"
    save_file({prepare.TENSOR_NAME: wrong}, str(wrong_path))
    wrong_source = _source_spec(wrong_path, wrong)
    with pytest.raises(prepare.PrepareError, match="shape must be"):
        prepare.prepare_g1_input(
            wrong_path, tmp_path / "wrong-output.safetensors", source=wrong_source
        )


def test_prepare_replaces_output_atomically_and_rejects_symlink_source(
    source_artifact: tuple[Path, Any, torch.Tensor], tmp_path: Path
) -> None:
    model_path, source, _tensor = source_artifact
    output_path = tmp_path / "prepared.safetensors"
    output_path.write_bytes(b"old output")

    prepare.prepare_g1_input(model_path, output_path, source=source)
    first_bytes = output_path.read_bytes()
    prepare.prepare_g1_input(model_path, output_path, source=source)
    assert output_path.read_bytes() == first_bytes
    with prepare.safe_open(output_path, framework="pt", device="cpu") as output:
        assert list(output.keys()) == [prepare.TENSOR_NAME]
    header_length = int.from_bytes(first_bytes[:8], "little")
    header_bytes = first_bytes[8 : 8 + header_length]
    parsed_header = json.loads(header_bytes)
    canonical_header = json.dumps(
        parsed_header, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert header_bytes == canonical_header.ljust(header_length, b" ")
    assert not list(tmp_path.glob(".prepared.safetensors.tmp-*"))

    link = tmp_path / "model-link.safetensors"
    link.symlink_to(model_path)
    with pytest.raises(prepare.PrepareError, match="symbolic link"):
        prepare.prepare_g1_input(
            link, tmp_path / "link-output.safetensors", source=source
        )


def test_prepare_rejects_output_through_symlinked_parent(
    source_artifact: tuple[Path, Any, torch.Tensor], tmp_path: Path
) -> None:
    model_path, source, _tensor = source_artifact
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    real_source = real_parent / model_path.name
    model_path.replace(real_source)
    original = real_source.read_bytes()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(prepare.PrepareError, match="paths must be different"):
        prepare.prepare_g1_input(
            real_source, alias_parent / real_source.name, source=source
        )

    assert real_source.read_bytes() == original


def test_header_canonicalization_is_stable_across_fresh_processes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    child = """
import runpy
import sys
import torch
from pathlib import Path
from safetensors.torch import save_file

module = runpy.run_path(sys.argv[1])
output = Path(sys.argv[2])
save_file(
    {"tensor": torch.zeros((2, 2), dtype=torch.bfloat16)},
    str(output),
    metadata={"z": "last", "a": "first", "m": "middle"},
)
module["_canonicalize_safetensors_header"](output)
"""
    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                "-c",
                child,
                str(ROOT / "scripts/prepare_g1_inputs.py"),
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    assert first.read_bytes() == second.read_bytes()


def test_make_runs_g1_session_without_cargo_launch_environment(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "target dir"
    result = subprocess.run(
        [
            "make",
            "-n",
            "run-g1-session",
            f"CARGO_TARGET_DIR={target_dir}",
            "CASES=/tmp/cases.json",
            "OUTPUT=/tmp/session.json",
            "SESSION_ID=g1-test-session",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "cargo build --quiet --release --locked" in result.stdout
    assert "cargo run" not in result.stdout
    assert f'"{target_dir}/release/decodeforge-g1-bench" run-session' in result.stdout

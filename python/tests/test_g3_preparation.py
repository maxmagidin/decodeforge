"""Focused receipt, tamper, and capture-failure tests for G3 preparation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from decodeforge.g3_preparation import (
    CANONICAL_ASSET_INVENTORY_IDENTITY,
    G3PreparationError,
    capture_preparation_receipt,
    load_preparation_receipt,
    load_preparation_receipt_document,
    receipt_identity,
    verify_preparation_receipt,
)

JsonObject = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
REVISION = "1" * 40
SOURCE = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "filename": "model.safetensors",
    "bytes": 2_200_119_864,
    "identity": (
        "sha256:6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
    ),
}
OUTPUT = {
    "asset_inventory_identity": CANONICAL_ASSET_INVENTORY_IDENTITY,
    "layer_count": 22,
    "total_packed_bytes": 103_809_024,
    "total_fallback_bytes": 369_098_752,
}


def _tool(tmp_path: Path) -> Path:
    tool = tmp_path / "decodeforge-prepare-qproj"
    tool.write_bytes(b"test preparation executable\n")
    tool.chmod(0o700)
    return tool


def _unsigned(tool: Path, tmp_path: Path) -> JsonObject:
    return {
        "schema_version": 1,
        "format": "decodeforge_g3_offline_preparation_receipt_v1",
        "protocol_id": "g3-tinyllama-qproj-generation-v1",
        "checkout": {"revision": REVISION, "dirty": False},
        "command": {
            "argv": [
                str(tool.absolute()),
                "--source",
                str((tmp_path / "model.safetensors").absolute()),
                "--output",
                str((tmp_path / "assets").absolute()),
            ]
        },
        "source": dict(SOURCE),
        "tool": {
            "name": "decodeforge-prepare-qproj",
            "version": "0.1.0",
            "executable_identity": "sha256:"
            + hashlib.sha256(tool.read_bytes()).hexdigest(),
        },
        "output": dict(OUTPUT),
        "timing": {
            "clock": "time.perf_counter_ns",
            "start_ns": 100,
            "stop_ns": 250,
            "elapsed_ns": 150,
        },
    }


def _write_receipt(path: Path, unsigned: JsonObject) -> None:
    path.write_text(
        json.dumps({**unsigned, "receipt_identity": receipt_identity(unsigned)}),
        encoding="utf-8",
    )


def _inventory() -> JsonObject:
    return {
        "schema_version": 1,
        "format": "decodeforge_q_proj_inventory_v1",
        "source": dict(SOURCE),
        "layer_count": 22,
        "entries": [{"layer": layer} for layer in range(22)],
        "total_packed_bytes": OUTPUT["total_packed_bytes"],
        "total_fallback_bytes": OUTPUT["total_fallback_bytes"],
        "aggregate_identity": CANONICAL_ASSET_INVENTORY_IDENTITY,
    }


def test_receipt_parser_retains_provenance_and_projects_session(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"
    _write_receipt(receipt, _unsigned(tool, tmp_path))

    verified = verify_preparation_receipt(receipt)
    assert verified.checkout_revision == REVISION
    assert verified.command_argv[0] == str(tool.absolute())
    assert verified.tool_executable_identity.startswith("sha256:")
    assert load_preparation_receipt(receipt) == {
        "source": "separately_captured_prepare_command",
        "receipt_identity": verified.receipt_identity,
        "elapsed_ns": 150,
        "asset_inventory_identity": CANONICAL_ASSET_INVENTORY_IDENTITY,
    }


def test_portable_document_is_closed_identity_bound_and_deep_independent(
    tmp_path: Path,
) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"
    unsigned = _unsigned(tool, tmp_path)
    _write_receipt(receipt, unsigned)

    first = load_preparation_receipt_document(receipt)
    tool.unlink()
    second = load_preparation_receipt_document(receipt)
    first["source"]["model_id"] = "mutated/in-memory"
    assert second["source"] == SOURCE
    with pytest.raises(G3PreparationError, match="preparation tool"):
        verify_preparation_receipt(receipt)

    unsigned["source"]["unexpected"] = True
    _write_receipt(receipt, unsigned)
    with pytest.raises(G3PreparationError, match="source is not a closed object"):
        load_preparation_receipt_document(receipt)

    unsigned = _unsigned(_tool(tmp_path), tmp_path)
    _write_receipt(receipt, unsigned)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["timing"]["elapsed_ns"] = 149
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(G3PreparationError, match="identity mismatch"):
        load_preparation_receipt_document(receipt)


def test_receipt_rejects_identity_tamper_and_semantic_rehash(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"
    unsigned = _unsigned(tool, tmp_path)
    _write_receipt(receipt, unsigned)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["timing"]["elapsed_ns"] = 149
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(G3PreparationError, match="identity mismatch"):
        verify_preparation_receipt(receipt)

    unsigned["source"]["model_id"] = "other/model"
    _write_receipt(receipt, unsigned)
    with pytest.raises(G3PreparationError, match="source is not canonical"):
        verify_preparation_receipt(receipt)


def test_receipt_rejects_duplicate_keys_and_noncanonical_argv(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"
    unsigned = _unsigned(tool, tmp_path)
    _write_receipt(receipt, unsigned)
    raw = receipt.read_text(encoding="utf-8").replace(
        '"dirty": false', '"dirty": false, "dirty": false'
    )
    receipt.write_text(raw, encoding="utf-8")
    with pytest.raises(G3PreparationError, match="valid JSON"):
        verify_preparation_receipt(receipt)

    unsigned = _unsigned(tool, tmp_path)
    unsigned["command"]["argv"].append("--extra")
    _write_receipt(receipt, unsigned)
    with pytest.raises(G3PreparationError, match="command is invalid"):
        verify_preparation_receipt(receipt)


def test_receipt_rejects_replaced_tool_and_symlink_parent(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"
    _write_receipt(receipt, _unsigned(tool, tmp_path))
    tool.write_bytes(b"replacement")
    with pytest.raises(G3PreparationError, match="executable identity mismatch"):
        verify_preparation_receipt(receipt)

    extra_link = tmp_path / "tool-link"
    tool.unlink()
    tool.write_bytes(b"test preparation executable\n")
    extra_link.hardlink_to(tool)
    _write_receipt(receipt, _unsigned(tool, tmp_path))
    with pytest.raises(G3PreparationError, match="exactly one link"):
        verify_preparation_receipt(receipt)
    extra_link.unlink()

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(G3PreparationError, match="without symlinks"):
        verify_preparation_receipt(linked / "receipt.json")


def test_capture_times_only_prepare_then_verifies_and_publishes(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    output = tmp_path / "assets"
    receipt = tmp_path / "receipt.json"
    events: list[str] = []
    clock_values = iter((100, 250))

    def clock() -> int:
        events.append("clock")
        return next(clock_values)

    def run_tool(arguments: Sequence[str]) -> None:
        if "--source" in arguments:
            events.append("prepare")
            output.mkdir()
            (output / "inventory.json").write_text(
                json.dumps(_inventory()), encoding="utf-8"
            )
        else:
            events.append("verify")
            assert arguments[1:] == ["--verify", str(output.absolute())]

    calls = 0

    def checkout_state(_checkout: Path) -> tuple[str, bool]:
        nonlocal calls
        calls += 1
        return REVISION, False

    projection = capture_preparation_receipt(
        checkout=ROOT,
        source=tmp_path / "model.safetensors",
        output=output,
        receipt=receipt,
        prepare_tool=tool,
        clock=clock,
        run_tool=run_tool,
        checkout_state=checkout_state,
    )

    assert events == ["clock", "prepare", "clock", "verify"]
    assert calls == 3
    assert projection["elapsed_ns"] == 150
    verified = verify_preparation_receipt(receipt)
    assert verified.command_argv == (
        str(tool.absolute()),
        "--source",
        str((tmp_path / "model.safetensors").absolute()),
        "--output",
        str(output.absolute()),
    )


def test_capture_failure_or_checkout_change_never_publishes_receipt(
    tmp_path: Path,
) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"

    def fail(_arguments: Sequence[str]) -> None:
        raise G3PreparationError("synthetic failure")

    with pytest.raises(G3PreparationError, match="synthetic"):
        capture_preparation_receipt(
            checkout=ROOT,
            source=tmp_path / "model.safetensors",
            output=tmp_path / "failed-assets",
            receipt=receipt,
            prepare_tool=tool,
            clock=iter((1, 2)).__next__,
            run_tool=fail,
            checkout_state=lambda _checkout: (REVISION, False),
        )
    assert not receipt.exists()

    output = tmp_path / "assets"

    def prepare_then_verify(arguments: Sequence[str]) -> None:
        if "--source" in arguments:
            output.mkdir()
            (output / "inventory.json").write_text(
                json.dumps(_inventory()), encoding="utf-8"
            )

    states = iter(((REVISION, False), ("2" * 40, False)))
    with pytest.raises(G3PreparationError, match="checkout changed"):
        capture_preparation_receipt(
            checkout=ROOT,
            source=tmp_path / "model.safetensors",
            output=output,
            receipt=receipt,
            prepare_tool=tool,
            clock=iter((1, 2)).__next__,
            run_tool=prepare_then_verify,
            checkout_state=lambda _checkout: next(states),
        )
    assert not receipt.exists()


def test_capture_preflights_no_overwrite_and_external_receipt(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("retained", encoding="utf-8")
    ran = False

    def run_tool(_arguments: Sequence[str]) -> None:
        nonlocal ran
        ran = True

    with pytest.raises(G3PreparationError, match="already exists"):
        capture_preparation_receipt(
            checkout=ROOT,
            source=tmp_path / "model.safetensors",
            output=tmp_path / "assets",
            receipt=receipt,
            prepare_tool=tool,
            run_tool=run_tool,
        )
    assert receipt.read_text(encoding="utf-8") == "retained"
    assert not ran

    with pytest.raises(G3PreparationError, match="outside"):
        capture_preparation_receipt(
            checkout=tmp_path,
            source=tmp_path / "model.safetensors",
            output=tmp_path / "new-assets",
            receipt=tmp_path / "new-assets" / "receipt.json",
            prepare_tool=tool,
            run_tool=run_tool,
        )

    with pytest.raises(G3PreparationError, match="asset output must be outside"):
        capture_preparation_receipt(
            checkout=ROOT,
            source=tmp_path / "model.safetensors",
            output=ROOT / "untracked-g3-assets-test",
            receipt=tmp_path / "new-receipt.json",
            prepare_tool=tool,
            run_tool=run_tool,
        )
    with pytest.raises(G3PreparationError, match="receipt must be outside"):
        capture_preparation_receipt(
            checkout=ROOT,
            source=tmp_path / "model.safetensors",
            output=tmp_path / "other-assets",
            receipt=ROOT / "untracked-g3-receipt-test.json",
            prepare_tool=tool,
            run_tool=run_tool,
        )


def test_postpublication_checkout_change_rolls_back_receipt(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    output = tmp_path / "assets"
    receipt = tmp_path / "receipt.json"

    def prepare_then_verify(arguments: Sequence[str]) -> None:
        if "--source" in arguments:
            output.mkdir()
            (output / "inventory.json").write_text(
                json.dumps(_inventory()), encoding="utf-8"
            )

    states = iter(((REVISION, False), (REVISION, False), ("2" * 40, False)))
    with pytest.raises(G3PreparationError, match="receipt publication"):
        capture_preparation_receipt(
            checkout=ROOT,
            source=tmp_path / "model.safetensors",
            output=output,
            receipt=receipt,
            prepare_tool=tool,
            clock=iter((1, 2)).__next__,
            run_tool=prepare_then_verify,
            checkout_state=lambda _checkout: next(states),
        )
    assert not receipt.exists()

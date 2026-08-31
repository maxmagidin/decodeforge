"""Committed corpus and deterministic regeneration checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from decodeforge.contracts import ROOT, load_json, validate_data, validate_path

SCRIPT = ROOT / "scripts" / "generate_q8_fixtures.py"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "v1"
_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "decodeforge_fixture_generator", SCRIPT
)
assert _GENERATOR_SPEC is not None and _GENERATOR_SPEC.loader is not None
fixture_generator = importlib.util.module_from_spec(_GENERATOR_SPEC)
_GENERATOR_SPEC.loader.exec_module(fixture_generator)


def test_fixture_corpus_check_is_read_only_and_clean() -> None:
    before = {
        path: path.read_bytes() for path in FIXTURE_ROOT.rglob("*") if path.is_file()
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "16 deterministic fixtures" in result.stdout
    after = {
        path: path.read_bytes() for path in FIXTURE_ROOT.rglob("*") if path.is_file()
    }
    assert before == after


def test_every_committed_fixture_passes_semantic_schema() -> None:
    manifest = load_json(FIXTURE_ROOT / "manifest.json")
    records = manifest["artifacts"]
    assert [record["path"] for record in records] == sorted(
        record["path"] for record in records
    )
    assert manifest["format"] == "DFQ8_B32_V1"
    assert manifest["numeric_mode"] == "strict_f32_v1"
    for record in records:
        path = FIXTURE_ROOT / record["path"]
        assert validate_path(path, "quant-fixture") == []
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["blocks"] == (document["k"] + 31) // 32


def test_manifest_freezes_and_drives_the_counter_recipe() -> None:
    manifest = load_json(FIXTURE_ROOT / "manifest.json")
    recipe = manifest["corpus_recipe"]
    assert recipe == fixture_generator.EXPECTED_CORPUS_RECIPE
    counter = recipe["counter"]
    assert bytes.fromhex(counter["domain_hex"]) == (
        b"DecodeForge/DFQ8_B32_V1/corpus/v1\0"
    )
    assert counter["mapping"] == {
        "kind": "finite-binary32-exponent",
        "preserve_mask_hex": "807fffff",
        "forced_exponent": 124,
    }
    assert counter["streams"] == {
        "input": {"seed_hex": "696e707574", "word_count": 33},
        "source": {"seed_hex": "736f75726365", "word_count": 99},
    }

    documents = fixture_generator._generated_documents(recipe)
    records = manifest["artifacts"]
    assert [
        {
            "path": path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content in documents.items()
    ] == records


def test_manifest_recipe_mutation_and_unknown_fields_are_rejected() -> None:
    manifest = load_json(FIXTURE_ROOT / "manifest.json")

    mutated = deepcopy(manifest)
    mutated["corpus_recipe"]["counter"]["mapping"]["forced_exponent"] = 123
    diagnostics = validate_data(mutated, "fixture-manifest")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-007"]
    _assert_closed_diagnostics(diagnostics)

    unknown = deepcopy(manifest)
    unknown["corpus_recipe"]["counter"]["streams"]["source"]["salt"] = "no"
    diagnostics = validate_data(unknown, "fixture-manifest")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-005"]
    _assert_closed_diagnostics(diagnostics)


def test_manifest_recipe_integral_floats_and_booleans_are_rejected_before_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = load_json(FIXTURE_ROOT / "manifest.json")

    integral_float = deepcopy(manifest)
    integral_float["corpus_recipe"]["counter"]["counter_start"] = 0.0
    diagnostics = validate_data(integral_float, "fixture-manifest")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-007"]
    _assert_closed_diagnostics(diagnostics)

    boolean = deepcopy(manifest)
    boolean["corpus_recipe"]["counter"]["streams"]["source"]["word_count"] = False
    diagnostics = validate_data(boolean, "fixture-manifest")
    assert "DFE-SCHEMA-006" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(integral_float), encoding="utf-8")
    monkeypatch.setattr(fixture_generator, "FIXTURE_ROOT", tmp_path)
    monkeypatch.setattr(fixture_generator, "MANIFEST_PATH", manifest_path)

    def _must_not_generate(_: dict[str, Any]) -> dict[str, bytes]:
        raise AssertionError("an invalid recipe reached corpus generation")

    monkeypatch.setattr(fixture_generator, "_generated_documents", _must_not_generate)
    errors = fixture_generator._check()
    assert errors
    assert "DFE-SCHEMA-007" in errors[0]


def _mutable_fixture() -> dict[str, Any]:
    return deepcopy(load_json(FIXTURE_ROOT / "fixtures" / "ties-and-extrema.json"))


def _assert_closed_diagnostics(diagnostics: list[dict[str, Any]]) -> None:
    assert diagnostics
    for diagnostic in diagnostics:
        assert validate_data(diagnostic, "diagnostic") == []


def test_fixture_mutations_have_stable_diagnostics() -> None:
    wrong_scale = _mutable_fixture()
    scales = wrong_scale["expected_scale_bits"]
    assert isinstance(scales, list)
    scales[0] ^= 1
    diagnostics = validate_data(wrong_scale, "quant-fixture")
    assert "DFE-QUANT-011" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)

    wrong_padding = _mutable_fixture()
    q_bytes = wrong_padding["expected_q_bytes"]
    assert isinstance(q_bytes, list)
    q_bytes[-1] = 1
    diagnostics = validate_data(wrong_padding, "quant-fixture")
    assert "DFE-QUANT-007" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)

    wrong_output = _mutable_fixture()
    outputs = wrong_output["expected_output_fp32_bits"]
    assert isinstance(outputs, list)
    outputs[0] ^= 1
    diagnostics = validate_data(wrong_output, "quant-fixture")
    assert "DFE-QUANT-011" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)

    for field in ("logical_weight_identity", "fixture_identity"):
        wrong_identity = _mutable_fixture()
        wrong_identity[field] = "sha256:" + "0" * 64
        diagnostics = validate_data(wrong_identity, "quant-fixture")
        assert "DFE-QUANT-008" in {item["code"] for item in diagnostics}
        _assert_closed_diagnostics(diagnostics)


def test_fixture_required_fields_and_nonfinite_output_are_stable() -> None:
    for field in ("case_id", "logical_weight_identity", "fixture_identity"):
        missing = _mutable_fixture()
        del missing[field]
        diagnostics = validate_data(missing, "quant-fixture")
        assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-004"]
        _assert_closed_diagnostics(diagnostics)

    nonfinite = _mutable_fixture()
    outputs = nonfinite["expected_output_fp32_bits"]
    assert isinstance(outputs, list)
    outputs[0] = 0x7F800000
    diagnostics = validate_data(nonfinite, "quant-fixture")
    assert [item["code"] for item in diagnostics] == ["DFE-QUANT-010"]
    _assert_closed_diagnostics(diagnostics)


def test_fixture_manifest_paths_are_closed_and_canonical() -> None:
    manifest = load_json(FIXTURE_ROOT / "manifest.json")

    duplicate = deepcopy(manifest)
    artifacts = duplicate["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.append(deepcopy(artifacts[0]))
    diagnostics = validate_data(duplicate, "fixture-manifest")
    assert "DFE-QUANT-012" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)

    unsorted = deepcopy(manifest)
    artifacts = unsorted["artifacts"]
    assert isinstance(artifacts, list)
    artifacts.reverse()
    diagnostics = validate_data(unsorted, "fixture-manifest")
    assert "DFE-QUANT-012" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)

    unsafe = deepcopy(manifest)
    artifacts = unsafe["artifacts"]
    assert isinstance(artifacts, list)
    artifacts[0]["path"] = "manifest.json"
    diagnostics = validate_data(unsafe, "fixture-manifest")
    assert "DFE-QUANT-012" in {item["code"] for item in diagnostics}
    _assert_closed_diagnostics(diagnostics)


def test_fixture_write_refuses_target_ancestor_and_manifest_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_root = tmp_path / "v1"
    fixture_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (fixture_root / "fixtures").symlink_to(outside, target_is_directory=True)
    manifest_path = fixture_root / "manifest.json"
    monkeypatch.setattr(fixture_generator, "FIXTURE_ROOT", fixture_root)
    monkeypatch.setattr(fixture_generator, "MANIFEST_PATH", manifest_path)

    with pytest.raises(RuntimeError):
        fixture_generator._write(
            {
                "fixtures/safe.json": b"safe",
                "fixtures/redirected.json": b"redirected",
            },
            fixture_generator.EXPECTED_CORPUS_RECIPE,
        )
    assert not (fixture_root / "fixtures" / "safe.json").exists()
    assert not (outside / "safe.json").exists()
    assert not manifest_path.exists()

    direct_root = tmp_path / "direct-v1"
    direct_root_fixtures = direct_root / "fixtures"
    direct_root_fixtures.mkdir(parents=True)
    direct_outside = tmp_path / "direct-outside"
    direct_outside.mkdir()
    direct_target = direct_root_fixtures / "artifact.json"
    direct_target.symlink_to(direct_outside / "artifact.json")
    direct_manifest = direct_root / "manifest.json"
    monkeypatch.setattr(fixture_generator, "FIXTURE_ROOT", direct_root)
    monkeypatch.setattr(fixture_generator, "MANIFEST_PATH", direct_manifest)

    with pytest.raises(RuntimeError):
        fixture_generator._write(
            {"fixtures/artifact.json": b"artifact"},
            fixture_generator.EXPECTED_CORPUS_RECIPE,
        )
    assert not (direct_outside / "artifact.json").exists()
    assert not direct_manifest.exists()

    clean_root = tmp_path / "clean-v1"
    clean_root.mkdir()
    clean_outside = tmp_path / "clean-outside"
    clean_outside.mkdir()
    manifest_link = clean_root / "manifest.json"
    manifest_link.symlink_to(clean_outside / "manifest.json")
    monkeypatch.setattr(fixture_generator, "FIXTURE_ROOT", clean_root)
    monkeypatch.setattr(fixture_generator, "MANIFEST_PATH", manifest_link)

    with pytest.raises(RuntimeError):
        fixture_generator._write(
            {"fixtures/artifact.json": b"artifact"},
            fixture_generator.EXPECTED_CORPUS_RECIPE,
        )
    assert not (clean_root / "fixtures" / "artifact.json").exists()
    assert not (clean_outside / "manifest.json").exists()

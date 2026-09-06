from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from decodeforge import evaluation as ev
from torch import nn

_CLI_SPEC = importlib.util.spec_from_file_location(
    "decodeforge_test_run_evaluation",
    Path(__file__).parents[2] / "scripts" / "run_evaluation.py",
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(cli)


def _request(tmp_path: Path) -> ev.EvaluationRequest:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    model = tmp_path / "model"
    assets = tmp_path / "assets"
    model.mkdir()
    assets.mkdir()
    library = tmp_path / "bridge.dylib"
    library.write_bytes(b"bridge")
    return ev.EvaluationRequest(
        spec_path=spec,
        model_directory=model,
        asset_directory=assets,
        bridge_library=library,
        bridge_sha256="0" * 64,
    )


@pytest.mark.parametrize(
    "source",
    [
        {"git_revision": None, "git_dirty": False},
        {"git_revision": "0123456789abcdef", "git_dirty": True},
    ],
)
def test_invalid_or_dirty_source_is_rejected_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: dict[str, Any],
) -> None:
    monkeypatch.setattr(ev, "validate_spec", lambda _document: {"cases": []})
    monkeypatch.setattr(ev, "_source_identity", lambda _path: source)
    model_loader_called = False

    def model_loader(_path: Path) -> nn.Module:
        nonlocal model_loader_called
        model_loader_called = True
        raise AssertionError("source rejection must precede model loading")

    with pytest.raises(ev.EvaluationError):
        ev.run_evaluation(
            _request(tmp_path),
            model_verifier=lambda _path: {},
            model_loader=model_loader,
        )
    assert not model_loader_called


class _LogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(logits=self.logits)


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor(
            [[[0.0, float("nan")], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.float32,
        ),
        torch.zeros((1, 4), dtype=torch.float32),
    ],
)
def test_teacher_force_rejects_nonfinite_or_malformed_logits(
    logits: torch.Tensor,
) -> None:
    with pytest.raises(ev.EvaluationError):
        ev._teacher_force(_LogitModel(logits), [3, 4], [1, 0])


def test_compare_runs_shape_mismatch_is_failed_or_controlled_error() -> None:
    reference = ev.CachedGeneration(
        [3, 4, 1], [1], "max_new_tokens", False, [torch.zeros(3)], []
    )
    candidate = ev.CachedGeneration(
        [3, 4, 1], [1], "max_new_tokens", False, [torch.zeros(2)], []
    )
    try:
        result = ev.compare_runs(reference, candidate)
    except ev.EvaluationError:
        return
    assert result["passed"] is False


def test_hybrid_single_token_run_fails_cached_coverage_reconciliation() -> None:
    delta = []
    for layer in range(22):
        delta.append(
            {
                "layer": layer,
                "forward": 1,
                "native_attempt": 0,
                "native_success": 0,
                "native_error": 0,
                "fallback_attempt": 1,
                "fallback_success": 1,
                "fallback_error": 0,
                "predispatch_error": 0,
                "rejected_closed": 0,
                "in_flight": 0,
            }
        )
    with pytest.raises(ev.EvaluationError):
        ev._reconcile("hybrid_native", 1, delta)


@pytest.mark.parametrize("error_type", [ev.EvaluationError, RuntimeError])
def test_cli_fatal_error_writes_nonoverwriting_rejected_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    output = tmp_path / "rejected.json"
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda _request, **_kwargs: (_ for _ in ()).throw(
            error_type("synthetic fatal")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_evaluation.py",
            "--mode",
            "correctness",
            "--model-dir",
            str(tmp_path / "model"),
            "--assets",
            str(tmp_path / "assets"),
            "--library",
            str(tmp_path / "bridge.dylib"),
            "--library-sha256",
            "0" * 64,
            "--spec",
            str(tmp_path / "spec.json"),
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 2
    assert output.is_file()
    before = output.read_bytes()
    payload = json.loads(before)
    assert payload["accepted"] is False
    assert "synthetic fatal" in json.dumps(payload)

    assert cli.main() == 2
    assert output.read_bytes() == before

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_profile_sessions",
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_profile_sessions.py",
)
assert _SPEC is not None and _SPEC.loader is not None
cli: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x": 1, "x": 2}',
        b'{"nested": {"x": 1, "x": 2}}',
        b'{"x": NaN}',
        b'{"x": Infinity}',
        b'{"x": 1e999}',
        b"[]",
        b"null",
        b"{",
    ],
)
def test_reader_rejects_ambiguous_json(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "capture.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        cli.read_capture(path)


def test_reader_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "capture.json"
    raw = b'{ "session_index": 0 }\n'
    path.write_bytes(raw)
    document, digest = cli.read_capture(path)
    assert document == {"session_index": 0}
    assert digest == hashlib.sha256(raw).hexdigest()


def test_reader_bounds_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "capture.json"
    path.write_bytes(b'{"large": 1000}')
    monkeypatch.setattr(cli, "MAX_CAPTURE_BYTES", 5)
    with pytest.raises(ValueError, match="32 MiB"):
        cli.read_capture(path)


def test_reader_rejects_fifo_without_waiting(tmp_path: Path) -> None:
    path = tmp_path / "pipe"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        cli.read_capture(path)


def test_cli_writes_hashed_inputs_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / f"capture-{index}.json" for index in (2, 0, 1)]
    for index, path in zip((2, 0, 1), paths, strict=True):
        path.write_text(json.dumps({"session_index": index}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        cli, "analyze_profile_sessions", lambda _: {"performance_claim_allowed": False}
    )
    monkeypatch.setattr(
        "sys.argv", ["analyze", "--sessions", *map(str, paths), "--output", str(output)]
    )
    assert cli.main() == 0
    saved = output.read_bytes()
    report = json.loads(saved)
    assert report["performance_claim_allowed"] is False
    assert report["input_captures"] == [
        {
            "session_index": index,
            "sha256": hashlib.sha256(
                (tmp_path / f"capture-{index}.json").read_bytes()
            ).hexdigest(),
        }
        for index in range(3)
    ]
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert output.read_bytes() == saved


def test_cli_rejection_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture.json"
    path.write_text("{}")
    output = tmp_path / "report.json"

    def reject(_: Any) -> None:
        raise cli.ProfileAnalysisError("invalid capture")

    monkeypatch.setattr(cli, "analyze_profile_sessions", reject)
    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--sessions", *([str(path)] * 3), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not output.exists()

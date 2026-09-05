"""Regression tests for raw path handling in legacy Make recipes."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest
from decodeforge.contracts import ROOT


def _recorder(path: Path, name: str) -> Path:
    script = path / name
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "record = pathlib.Path(os.environ['MAKE_RECORDER'])\n"
        "keys = ('WEIGHTS', 'OUTPUT', 'CASES', 'PREPARED_WEIGHTS', 'CHECKOUT',\n"
        "        'BUNDLE', 'SESSION_1', 'SESSION_2', 'SESSION_3', 'OUTPUT_DIR',\n"
        "        'SESSION_ID', 'CARGO_TARGET_DIR')\n"
        "with record.open('a', encoding='utf-8') as stream:\n"
        "    json.dump({'argv': sys.argv[1:],\n"
        "              'env': {key: os.environ.get(key) for key in keys}},\n"
        "             stream, sort_keys=True)\n"
        "    stream.write('\\n')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _fake_uname(path: Path) -> Path:
    script = path / "uname"
    script.write_text(
        '#!/bin/sh\nif [ "$1" = "-s" ]; then echo Darwin; else echo arm64; fi\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _make(root: Path, fake_uname: Path, input_origin: str, *arguments: str) -> None:
    environment = os.environ.copy()
    environment["MAKE_RECORDER"] = str(root / "records.jsonl")
    environment["PATH"] = f"{fake_uname.parent}{os.pathsep}{environment['PATH']}"
    if input_origin == "environment":
        command_line = []
        for argument in arguments:
            if "=" not in argument:
                command_line.append(argument)
                continue
            key, value = argument.split("=", 1)
            if key in {"UV", "CARGO"}:
                command_line.append(argument)
            else:
                environment[key] = value
        arguments = tuple(command_line)
    subprocess.run(
        ["make", "-s", *arguments],
        cwd=ROOT,
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("input_origin", ["commandline", "environment"])
def test_legacy_make_recipes_preserve_raw_path_arguments(
    tmp_path: Path, input_origin: str
) -> None:
    uv = _recorder(tmp_path, "fake-uv")
    cargo = _recorder(tmp_path, "fake-cargo")
    fake_uname = _fake_uname(tmp_path)
    suffix = " spaces 'single' \"double\" `literal` $(shell printf harmless_token)"
    target = tmp_path / f"target{suffix}"
    executable = target / "release" / "decodeforge-g1-bench"
    executable.parent.mkdir(parents=True)
    executable = _recorder(executable.parent, "decodeforge-g1-bench")

    values = {
        "weights": str(tmp_path / f"weights{suffix}"),
        "prepared": str(tmp_path / f"prepared{suffix}"),
        "output": str(tmp_path / f"output{suffix}"),
        "cases": str(tmp_path / f"cases{suffix}"),
        "checkout": str(tmp_path / f"checkout{suffix}"),
        "bundle": str(tmp_path / f"bundle{suffix}"),
        "session_1": str(tmp_path / f"session-one{suffix}"),
        "session_2": str(tmp_path / f"session-two{suffix}"),
        "session_3": str(tmp_path / f"session-three{suffix}"),
        "output_dir": str(tmp_path / f"report-dir{suffix}"),
    }
    common = [f"UV={uv}", f"CARGO={cargo}", f"CARGO_TARGET_DIR={target}"]

    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "prepare-g1-input",
        *common,
        f"WEIGHTS={values['weights']}",
        f"OUTPUT={values['output']}",
    )
    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "prepare-g1-cases",
        *common,
        f"PREPARED_WEIGHTS={values['prepared']}",
        f"OUTPUT={values['output']}",
    )
    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "capture-g0-evidence",
        *common,
        f"OUTPUT={values['output']}",
        f"CHECKOUT={values['checkout']}",
    )
    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "verify-g0-repository",
        *common,
        f"BUNDLE={values['bundle']}",
        f"CHECKOUT={values['checkout']}",
    )
    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "verify-bundle",
        *common,
        f"BUNDLE={values['bundle']}",
    )
    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "analyze-g1",
        *common,
        f"SESSION_1={values['session_1']}",
        f"SESSION_2={values['session_2']}",
        f"SESSION_3={values['session_3']}",
        f"OUTPUT_DIR={values['output_dir']}",
    )
    _make(
        tmp_path,
        fake_uname,
        input_origin,
        "run-g1-session",
        *common,
        f"CASES={values['cases']}",
        f"OUTPUT={values['output']}",
        "SESSION_ID=session-id 'single' \"double\" `literal` "
        "$(shell printf harmless_token)",
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "records.jsonl").read_text().splitlines()
    ]
    assert any(
        record["argv"][-4:]
        == ["--weights", values["weights"], "--output", values["output"]]
        for record in records
    )
    assert any(
        record["argv"][-4:]
        == ["--weights", values["prepared"], "--output", values["output"]]
        for record in records
    )
    assert any(
        record["argv"][-4:]
        == ["--output", values["output"], "--checkout", values["checkout"]]
        for record in records
    )
    assert any(
        record["argv"][-4:]
        == ["--bundle", values["bundle"], "--checkout", values["checkout"]]
        for record in records
    )
    assert any(
        record["argv"][-2:] == ["--bundle", values["bundle"]] for record in records
    )
    assert any(
        record["argv"][-6:]
        == [
            "--sessions",
            values["session_1"],
            values["session_2"],
            values["session_3"],
            "--output-dir",
            values["output_dir"],
        ]
        for record in records
    )
    assert any(record["env"]["CHECKOUT"] == values["checkout"] for record in records)
    assert any(record["env"]["BUNDLE"] == values["bundle"] for record in records)
    assert any(record["env"]["SESSION_1"] == values["session_1"] for record in records)
    assert any(record["env"]["CARGO_TARGET_DIR"] == str(target) for record in records)

    executable_records = [
        record for record in records if record["argv"][:1] == ["run-session"]
    ]
    assert executable_records == [
        {
            "argv": [
                "run-session",
                "--cases",
                values["cases"],
                "--output",
                values["output"],
                "--session-id",
                "session-id 'single' \"double\" `literal` "
                "$(shell printf harmless_token)",
            ],
            "env": executable_records[0]["env"],
        }
    ]

"""Schema, diagnostic, and non-executing bundle validation tests."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import decodeforge.contracts as contracts
import pytest
from decodeforge.contracts import (
    ROOT,
    check_all,
    load_json,
    validate_data,
    validate_path,
    verify_bundle,
)

EXAMPLES = ROOT / "schemas" / "examples"
BUNDLES = ROOT / "tests" / "fixtures" / "bundles"
G3_SPEC = Path(__file__).resolve().parents[2] / "benchmarks" / "g3" / "spec.json"


def _deep_not_applicable() -> list[object]:
    values: list[object] = []
    for _ in range(2):
        value: object = "assembly"
        for _ in range(900):
            value = [value]
        values.append(value)
    return values


def test_catalog_and_directed_examples_are_consistent() -> None:
    assert check_all() == []


def test_canonical_g3_experiment_spec_is_accepted() -> None:
    assert validate_path(G3_SPEC, "g3-experiment-spec") == []


def test_schema_error_codes_are_stable() -> None:
    request = load_json(EXAMPLES / "compiler-request" / "valid-minimal.json")

    wrong_version = dict(request, schema_version=2)
    assert [
        item["code"] for item in validate_data(wrong_version, "compiler-request")
    ] == ["DFE-SCHEMA-002"]

    unknown_field = dict(request, weights_path="forbidden")
    assert [
        item["code"] for item in validate_data(unknown_field, "compiler-request")
    ] == ["DFE-SCHEMA-005"]

    wrong_type = dict(request, n="four")
    assert [item["code"] for item in validate_data(wrong_type, "compiler-request")] == [
        "DFE-SCHEMA-006"
    ]


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    diagnostics = validate_path(duplicate, "diagnostic")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-008"]


def test_nonfinite_numeric_token_is_rejected(tmp_path: Path) -> None:
    document = tmp_path / "overflow.json"
    document.write_text(
        '{"schema_version":1,"code":"DFE-BUNDLE-001",'
        '"severity":"error","component":"bundle",'
        '"summary":"bad","context":{"size":1e999}}',
        encoding="utf-8",
    )
    diagnostics = validate_path(document, "diagnostic")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-001"]


def test_nonfinite_numeric_token_in_permissive_context_is_rejected(
    tmp_path: Path,
) -> None:
    document = tmp_path / "context-overflow.json"
    document.write_text(
        '{"schema_version":1,"code":"DFE-BUNDLE-001",'
        '"severity":"error","component":"bundle",'
        '"summary":"bad","context":{"metadata":{"value":1e999}}}',
        encoding="utf-8",
    )
    diagnostics = validate_path(document, "diagnostic")
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-001"]


def test_empty_foundation_bundle_has_exact_missing_artifacts() -> None:
    diagnostics = verify_bundle(BUNDLES / "foundation-empty")
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-001"] * 3
    assert [item["context"]["artifact"] for item in diagnostics] == [
        "host.json",
        "report.md",
        "request.json",
    ]
    for item in diagnostics:
        assert validate_data(item, "diagnostic") == []


def test_minimal_foundation_bundle_is_accepted() -> None:
    assert verify_bundle(BUNDLES / "foundation-valid") == []


def test_deep_foundation_manifest_is_a_stable_parse_diagnostic(tmp_path: Path) -> None:
    bundle = tmp_path / "foundation-deep"
    manifest_path = bundle / "run-manifest.json"
    bundle.mkdir()
    payload = json.dumps(
        {"milestone": "foundation", "not_applicable": _deep_not_applicable()},
        separators=(",", ":"),
    )
    assert len(payload.encode("ascii")) == 3_667
    manifest_path.write_text(payload, encoding="ascii")

    diagnostics = verify_bundle(bundle)
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-001"]
    assert diagnostics[0]["context"] == {"path": ["run-manifest.json"]}


def test_registry_recursion_error_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _trusted_registry_failure() -> NoReturn:
        raise RecursionError("trusted schema registry")

    monkeypatch.setattr(contracts, "_schema_registry", _trusted_registry_failure)
    with pytest.raises(RecursionError, match="trusted schema registry"):
        verify_bundle(BUNDLES / "foundation-valid")


def test_bundle_hash_mutation_is_rejected(tmp_path: Path) -> None:
    mutated = tmp_path / "bundle"
    shutil.copytree(BUNDLES / "foundation-valid", mutated)
    report = mutated / "report.md"
    report.write_text(
        report.read_text(encoding="utf-8") + "mutated\n", encoding="utf-8"
    )
    diagnostics = verify_bundle(mutated)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-006"]
    assert diagnostics[0]["context"]["artifact"] == "report.md"


def test_bundle_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    mutated = tmp_path / "bundle"
    shutil.copytree(BUNDLES / "foundation-valid", mutated)
    report = mutated / "report.md"
    report.unlink()
    report.symlink_to("request.json")
    diagnostics = verify_bundle(mutated)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-005"]
    assert diagnostics[0]["context"]["artifact"] == "report.md"


def test_bundle_symlink_manifest_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run-manifest.json").symlink_to(
        BUNDLES / "foundation-valid" / "run-manifest.json"
    )
    diagnostics = verify_bundle(bundle)
    assert [item["code"] for item in diagnostics] == ["DFE-BUNDLE-005"]
    assert diagnostics[0]["context"]["artifact"] == "run-manifest.json"


def test_diagnostic_registry_codes_are_unique() -> None:
    registry = load_json(ROOT / "schemas" / "diagnostic-codes.json")
    raw_codes = registry["codes"]
    assert isinstance(raw_codes, list)
    codes = [entry["code"] for entry in raw_codes]
    assert len(codes) == len(set(codes))
    json.dumps(registry, allow_nan=False)


def _make_dry_run(target: str, assignments: list[str] | None = None) -> str:
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "--dry-run",
            *(assignments or []),
            target,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _capture_make_invocations(
    tmp_path: Path,
    target: str,
    assignments: list[str],
    *,
    cargo: bool = False,
    uv: bool = False,
    darwin_arm64: bool = False,
) -> list[dict[str, object]]:
    shim = tmp_path / "capture-tool.py"
    capture = tmp_path / "captured-invocations.jsonl"
    shim.write_text(
        """\
import json
import os
import sys

record = {
    "argv": sys.argv[1:],
    "cargo_target": os.environ.get("CARGO_TARGET_DIR"),
}
with open(os.environ["DECODEFORGE_TEST_CAPTURE"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\\n")
""",
        encoding="utf-8",
    )
    capture.unlink(missing_ok=True)
    tool = shlex.join([sys.executable, str(shim)])
    overrides = [
        *(["CARGO=" + tool] if cargo else []),
        *(["UV=" + tool] if uv else []),
    ]
    environment = dict(os.environ)
    if not any(value.startswith("CARGO_TARGET_DIR=") for value in assignments):
        environment.pop("CARGO_TARGET_DIR", None)
    if darwin_arm64:
        tools = tmp_path / "host-tools"
        tools.mkdir(exist_ok=True)
        uname = tools / "uname"
        uname.write_text(
            '#!/bin/sh\ncase "$1" in -s) echo Darwin ;; -m) echo arm64 ;; esac\n',
            encoding="utf-8",
        )
        uname.chmod(0o755)
        environment["PATH"] = str(tools) + os.pathsep + environment["PATH"]
    environment["DECODEFORGE_TEST_CAPTURE"] = str(capture)
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            target,
            *overrides,
            *assignments,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [
        json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()
    ]


def test_g3_make_recipes_never_render_raw_public_inputs(tmp_path: Path) -> None:
    sentinel = tmp_path / "dry-run-make-layer-executed"
    backtick = chr(96)
    hostile = (
        f"/opt/decodeforge/$(shell /usr/bin/touch {sentinel})"
        f"{backtick}/usr/bin/touch {sentinel}{backtick}$cash"
    )
    cases = (
        ("prepare-g3-assets", ("WEIGHTS", "OUTPUT"), ("WEIGHTS", "OUTPUT")),
        (
            "prepare-g3-assets-timed",
            ("CARGO_TARGET_DIR", "WEIGHTS", "OUTPUT", "RECEIPT"),
            ("CARGO_TARGET_DIR:-target", "WEIGHTS", "OUTPUT", "RECEIPT"),
        ),
        ("verify-g3-assets", ("ASSETS",), ("ASSETS",)),
        (
            "test-g3-adapter-real",
            ("CARGO_TARGET_DIR", "ASSETS", "SPEC"),
            ("CARGO_TARGET_DIR:-target", "ASSETS", "SPEC:-benchmarks/g3/spec.json"),
        ),
        ("build-g3-bridge", ("CARGO_TARGET_DIR",), ()),
        ("test-g3", (), ()),
        (
            "run-g3-session",
            (
                "SESSION_ID",
                "SESSION_INDEX",
                "MODEL_DIR",
                "ASSETS",
                "LIBRARY",
                "LIBRARY_SHA256",
                "PREPARATION_RECEIPT",
                "OUTPUT",
                "SPEC",
            ),
            (
                "SESSION_ID",
                "SESSION_INDEX",
                "MODEL_DIR",
                "ASSETS",
                "LIBRARY",
                "LIBRARY_SHA256",
                "PREPARATION_RECEIPT",
                "OUTPUT",
                "SPEC:-benchmarks/g3/spec.json",
            ),
        ),
        (
            "run-g3-demo",
            (
                "SESSION_ID",
                "SESSION_INDEX",
                "MODEL_DIR",
                "ASSETS",
                "LIBRARY",
                "LIBRARY_SHA256",
                "PREPARATION_RECEIPT",
                "OUTPUT",
                "SPEC",
            ),
            (
                "SESSION_ID",
                "SESSION_INDEX",
                "MODEL_DIR",
                "ASSETS",
                "LIBRARY",
                "LIBRARY_SHA256",
                "PREPARATION_RECEIPT",
                "OUTPUT",
                "SPEC:-benchmarks/g3/spec.json",
            ),
        ),
        (
            "analyze-g3",
            ("SESSION_1", "SESSION_2", "SESSION_3", "RECEIPT", "OUTPUT_DIR"),
            ("SESSION_1", "SESSION_2", "SESSION_3", "RECEIPT", "OUTPUT_DIR"),
        ),
        ("verify-g3-result", ("BUNDLE",), ("BUNDLE",)),
    )
    for target, variables, references in cases:
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "--dry-run",
                target,
                *(f"{variable}={hostile}" for variable in variables),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (target, result.stderr)
        rendered = result.stdout + result.stderr
        assert hostile not in rendered
        for reference in references:
            assert "$" + "{" + reference + "}" in rendered
        assert not sentinel.exists()


def test_g3_surfaces_transport_raw_public_inputs(
    tmp_path: Path,
) -> None:
    def assert_rust_preflight(invocations: list[dict[str, object]]) -> None:
        assert invocations[0]["argv"] == [
            "run",
            "--frozen",
            "python",
            "scripts/check_rust_toolchain.py",
            "--rust-version",
            "1.98.0",
        ]

    sentinel = tmp_path / "executed-make-layer"
    backtick = chr(96)
    cargo_target = "/opt/decodeforge/cargo$(error CARGO_AUDIT)"
    weights = f"/opt/decodeforge/model$(shell /usr/bin/touch {sentinel})$cash"
    output = '/opt/decodeforge/assets"quote'
    receipt = "/opt/decodeforge/receipt\nline\\tail.json"
    basic_preparation = _capture_make_invocations(
        tmp_path,
        "prepare-g3-assets",
        [f"WEIGHTS={weights}", f"OUTPUT={output}"],
        cargo=True,
        uv=True,
    )
    assert len(basic_preparation) == 2
    assert_rust_preflight(basic_preparation)
    basic_argv = basic_preparation[1]["argv"]
    assert isinstance(basic_argv, list)
    assert basic_argv[basic_argv.index("--source") + 1] == weights
    assert basic_argv[basic_argv.index("--output") + 1] == output

    preparation = _capture_make_invocations(
        tmp_path,
        "prepare-g3-assets-timed",
        [
            f"CARGO_TARGET_DIR={cargo_target}",
            f"WEIGHTS={weights}",
            f"OUTPUT={output}",
            f"RECEIPT={receipt}",
        ],
        cargo=True,
        uv=True,
    )
    # The timed preparation now runs the Rust dylib loader preflight before
    # building the helper and invoking the preparation wrapper.
    assert len(preparation) == 3
    assert_rust_preflight(preparation)
    assert preparation[0]["cargo_target"] == cargo_target
    preparation_argv = preparation[2]["argv"]
    assert isinstance(preparation_argv, list)
    assert preparation_argv[preparation_argv.index("--source") + 1] == weights
    assert preparation_argv[preparation_argv.index("--output") + 1] == output
    assert preparation_argv[preparation_argv.index("--receipt") + 1] == receipt
    assert preparation_argv[preparation_argv.index("--prepare-tool") + 1] == (
        cargo_target + "/release/decodeforge-prepare-qproj"
    )

    assets = (
        f"/opt/decodeforge/assets{backtick}/usr/bin/touch {sentinel}{backtick}$cash"
    )
    asset_verification = _capture_make_invocations(
        tmp_path,
        "verify-g3-assets",
        [f"ASSETS={assets}"],
        cargo=True,
    )
    assert len(asset_verification) == 1
    asset_argv = asset_verification[0]["argv"]
    assert isinstance(asset_argv, list)
    assert asset_argv[asset_argv.index("--verify") + 1] == assets

    default_build = _capture_make_invocations(
        tmp_path,
        "build-g3-bridge",
        [],
        cargo=True,
        uv=True,
    )
    assert len(default_build) == 2
    assert_rust_preflight(default_build)
    assert default_build[1]["cargo_target"] is None
    targeted_build = _capture_make_invocations(
        tmp_path,
        "build-g3-bridge",
        [f"CARGO_TARGET_DIR={cargo_target}"],
        cargo=True,
        uv=True,
    )
    assert len(targeted_build) == 2
    assert_rust_preflight(targeted_build)
    assert targeted_build[1]["cargo_target"] == cargo_target

    spec = "/opt/decodeforge/spec$(error SPEC_AUDIT)\nline.json"
    adapter = _capture_make_invocations(
        tmp_path,
        "test-g3-adapter-real",
        [
            f"CARGO_TARGET_DIR={cargo_target}",
            f"ASSETS={assets}",
            f"SPEC={spec}",
        ],
        cargo=True,
        uv=True,
        darwin_arm64=True,
    )
    # The adapter checkpoint preflights first, verifies assets, builds the
    # bridge, and finally invokes the Python checkpoint.
    assert len(adapter) == 4
    assert_rust_preflight(adapter)
    adapter_argv = adapter[3]["argv"]
    assert isinstance(adapter_argv, list)
    assert adapter_argv[adapter_argv.index("--library") + 1] == (
        cargo_target + "/release/libdecodeforge_bridge.dylib"
    )
    assert adapter_argv[adapter_argv.index("--assets") + 1] == assets
    assert adapter_argv[adapter_argv.index("--spec") + 1] == spec

    session_1 = (
        f"/opt/decodeforge/session{backtick}/usr/bin/touch {sentinel}{backtick}$cash"
    )
    output_directory = "/opt/decodeforge/report'quote"
    analysis = _capture_make_invocations(
        tmp_path,
        "analyze-g3",
        [
            f"SESSION_1={session_1}",
            "SESSION_2=/opt/decodeforge/session-2.json",
            "SESSION_3=/opt/decodeforge/session-3.json",
            f"RECEIPT={receipt}",
            f"OUTPUT_DIR={output_directory}",
        ],
        uv=True,
    )
    assert len(analysis) == 1
    analysis_argv = analysis[0]["argv"]
    assert isinstance(analysis_argv, list)
    sessions_index = analysis_argv.index("--sessions")
    assert analysis_argv[sessions_index + 1] == session_1
    assert analysis_argv[analysis_argv.index("--preparation-receipt") + 1] == receipt
    assert analysis_argv[analysis_argv.index("--output-dir") + 1] == output_directory

    bundle = "/opt/decodeforge/bundle$(error VERIFY_AUDIT)\\tail"
    verification = _capture_make_invocations(
        tmp_path,
        "verify-g3-result",
        [f"BUNDLE={bundle}"],
        uv=True,
    )
    assert len(verification) == 1
    verification_argv = verification[0]["argv"]
    assert isinstance(verification_argv, list)
    assert verification_argv[verification_argv.index("--bundle") + 1] == bundle
    assert not sentinel.exists()


def test_run_g3_demo_is_a_transparent_hardened_session_alias() -> None:
    assignments = [
        "SESSION_ID=make-surface-session",
        "SESSION_INDEX=2",
        "MODEL_DIR=/opt/decodeforge/model",
        "ASSETS=/opt/decodeforge/assets",
        "LIBRARY=/opt/decodeforge/cargo-target/release/libdecodeforge_bridge.dylib",
        f"LIBRARY_SHA256={'a' * 64}",
        "PREPARATION_RECEIPT=/opt/decodeforge/preparation-receipt.json",
        "OUTPUT=/opt/decodeforge/session-2.json",
    ]
    demo = _make_dry_run("run-g3-demo", assignments)
    session = _make_dry_run("run-g3-session", assignments)
    assert demo == session
    assert demo.count("scripts/run_g3_session.py") == 1
    assert "uv run --frozen --extra g3-generation" in demo
    for expected in (
        '--session-id "${SESSION_ID}"',
        '--session-index "${SESSION_INDEX}"',
        '--model-dir "${MODEL_DIR}"',
        '--assets "${ASSETS}"',
        '--library "${LIBRARY}"',
        '--library-sha256 "${LIBRARY_SHA256}"',
        '--preparation-receipt "${PREPARATION_RECEIPT}"',
        '--output "${OUTPUT}"',
        '--spec "${SPEC:-benchmarks/g3/spec.json}"',
    ):
        assert expected in demo
    for assignment in (assignments[0], *assignments[2:]):
        assert assignment.split("=", maxsplit=1)[1] not in demo


def test_run_g3_demo_transports_adversarial_paths_without_make_execution(
    tmp_path: Path,
) -> None:
    uv_shim = tmp_path / "uv-shim.py"
    uv_shim.write_text(
        """\
import os
import sys

expected = ["run", "--frozen", "--extra", "g3-generation", "python"]
if sys.argv[1:6] != expected:
    raise SystemExit(f"unexpected uv argv: {sys.argv[1:]!r}")
os.execv(sys.executable, [sys.executable, *sys.argv[6:]])
""",
        encoding="utf-8",
    )
    sentinel = tmp_path / "make-layer-executed"
    unsafe_models = (
        "/opt/decodeforge/model$cash",
        "/opt/decodeforge/model$(error AUDIT_SENTINEL)",
        f"/opt/decodeforge/model$(shell /usr/bin/touch {sentinel})",
        f"/opt/decodeforge/model`/usr/bin/touch {sentinel}`",
        '/opt/decodeforge/model"quote',
        "/opt/decodeforge/model\nnewline",
        "/opt/decodeforge/model\\escape",
    )
    stable_assignments = [
        "SESSION_ID=make-adversarial-session",
        "SESSION_INDEX=0",
        "ASSETS=/opt/decodeforge/assets",
        "LIBRARY=/opt/decodeforge/cargo-target/release/libdecodeforge_bridge.dylib",
        f"LIBRARY_SHA256={'a' * 64}",
        "PREPARATION_RECEIPT=/opt/decodeforge/preparation-receipt.json",
        "OUTPUT=/opt/decodeforge/session-0.json",
    ]
    for model in unsafe_models:
        result = subprocess.run(
            [
                "make",
                "--no-print-directory",
                "run-g3-demo",
                f"UV={sys.executable} {uv_shim}",
                f"MODEL_DIR={model}",
                *stable_assignments,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "bounded replay-safe path grammar" in result.stderr
        assert not sentinel.exists()


def test_test_g3_dry_run_has_the_closed_focused_gate() -> None:
    output = _make_dry_run("test-g3")
    assert (
        "cargo test --locked -p decodeforge-compiler --lib model_assets::tests::"
        in output
    )
    assert (
        "cargo test --locked -p decodeforge-compiler "
        "--bin decodeforge-prepare-qproj" in output
    )
    assert "python scripts/validate_schemas.py --all" in output
    expected_suites = [
        "python/tests/test_contracts.py",
        "python/tests/test_torch_bridge.py",
        "python/tests/test_qproj_adapter.py",
        "python/tests/test_qproj_model.py",
        "python/tests/test_g3_evidence.py",
        "python/tests/test_g3_preparation.py",
        "python/tests/test_g3_session.py",
        "python/tests/test_g3_session_cli.py",
        "python/tests/test_g3_results.py",
    ]
    positions = [output.index(suite) for suite in expected_suites]
    assert positions == sorted(positions)
    assert "python/tests/test_g3_*.py" not in output

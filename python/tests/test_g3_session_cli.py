from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_session_cli_reports_invalid_bridge_identity_without_traceback(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    external = tmp_path.resolve()
    output = external / "session.json"
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/run_g3_session.py"),
            "--session-id",
            "cli-invalid-hash",
            "--session-index",
            "0",
            "--model-dir",
            str(external / "model"),
            "--assets",
            str(external / "assets"),
            "--library",
            str(external / "target/release/libdecodeforge_bridge.dylib"),
            "--library-sha256",
            "sha256:invalid",
            "--preparation-receipt",
            str(external / "receipt.json"),
            "--output",
            str(output),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert (
        "g3-session: error: bridge_sha256 must be 64 lowercase SHA-256 hex digits"
        in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert "accepted" not in result.stdout
    assert not output.exists()

"""Keep the maintainer-selected license consistent across distributions."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_license_is_the_unmodified_apache_2_text() -> None:
    # Official https://www.apache.org/licenses/LICENSE-2.0.txt, including its
    # application appendix. Do not silently substitute a custom license.
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_python_distribution_declares_and_includes_license() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    sdist = metadata["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "/LICENSE" in sdist["include"]


def test_every_rust_member_inherits_the_workspace_license() -> None:
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]
    assert workspace["package"]["license"] == "Apache-2.0"
    for member in workspace["members"]:
        package = tomllib.loads((ROOT / member / "Cargo.toml").read_text())["package"]
        assert package["license"] == {"workspace": True}, member

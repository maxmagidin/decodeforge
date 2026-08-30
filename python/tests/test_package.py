"""Foundation import and version smoke tests."""

import decodeforge


def test_package_import_and_version() -> None:
    """The package imports without optional framework dependencies."""
    assert decodeforge.__version__ == "0.1.0"

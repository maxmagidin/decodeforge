"""Exponent overflow must fail at each evidence reader's JSON boundary."""

from pathlib import Path

import pytest
from decodeforge import contracts, g0_evidence, g1_evidence
from decodeforge import g3_preparation as preparation
from decodeforge import g3_results as results


@pytest.mark.parametrize("token", ["1e999", "-1e999"])
def test_evidence_readers_reject_overflow_before_semantic_validation(
    tmp_path: Path, token: str
) -> None:
    payload = ('{"metadata":{"nested":[' + token + "]}}").encode()
    paths = [tmp_path / f"session-{index}.json" for index in range(3)]
    for path in paths:
        path.write_bytes(payload)

    with pytest.raises(ValueError, match="non-finite"):
        contracts.load_json(paths[0])
    value, diagnostics = g0_evidence._decode_json_snapshot(payload, "manifest.json")
    assert value is None
    assert [item["code"] for item in diagnostics] == ["DFE-SCHEMA-001"]
    with pytest.raises(g1_evidence.G1AnalysisError, match="could not parse session"):
        g1_evidence.load_sessions(paths)
    with pytest.raises(preparation.G3PreparationError, match="not valid JSON"):
        preparation._json(payload, "receipt")
    with pytest.raises(results.G3ResultError, match="not strict UTF-8 JSON"):
        results._parse_json(payload, "analysis.json")


def test_evidence_readers_preserve_finite_numbers(tmp_path: Path) -> None:
    payload = b'{"metadata":{"nested":[1.25,1e308,-1e308]}}'
    expected = {"metadata": {"nested": [1.25, 1e308, -1e308]}}
    paths = [tmp_path / f"session-{index}.json" for index in range(3)]
    for path in paths:
        path.write_bytes(payload)

    assert contracts.load_json(paths[0]) == expected
    assert g0_evidence._decode_json_snapshot(payload, "manifest.json") == (expected, [])
    assert g1_evidence.load_sessions(paths) == [expected] * 3
    assert preparation._json(payload, "receipt") == expected
    assert results._parse_json(payload, "analysis.json") == expected

"""Strict JSON number hooks shared by evidence readers."""

from __future__ import annotations

import math


def reject_nonfinite_number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON value: {value}")
    return parsed

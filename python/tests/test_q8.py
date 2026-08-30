"""Directed tests for the strict-f32 DFQ8 reference."""

from __future__ import annotations

import struct
from collections.abc import Callable

import pytest
from decodeforge.contracts import validate_data
from decodeforge.q8 import (
    Q8Error,
    Q8Weights,
    _ratio_to_q,
    dequantize_f32_bits,
    f32_add_bits,
    f32_div_bits,
    f32_mul_bits,
    float_to_f32_bits,
    quantize_f32_bits,
)


def test_exact_subnormal_scale_boundaries() -> None:
    # 63 * 2**-149 / 127 is below half a subnormal; 64 * 2**-149 / 127
    # rounds to the minimum positive subnormal.
    assert f32_div_bits(0x0000003F, 0x42FE0000) == 0x00000000
    assert f32_div_bits(0x00000040, 0x42FE0000) == 0x00000001
    assert quantize_f32_bits(1, 1, [0x0000003F]).scale_bits == (0,)
    minimum = quantize_f32_bits(1, 1, [0x00000040])
    assert minimum.scale_bits == (1,)
    assert minimum.q_values[0] == 64


def test_q_layout_and_tail_padding() -> None:
    weights = quantize_f32_bits(2, 33, [0x3F800000] * 66)
    assert weights.blocks == 2
    assert weights.b == 2
    assert len(weights.q_bytes) == 2 * 2 * 32
    assert len(weights.scale_bits) == 4
    for row in range(weights.n):
        assert weights.q_at(row, 1, 0) == 127
        assert all(weights.q_at(row, 1, lane) == 0 for lane in range(1, 32))


def test_zero_and_signed_zero_are_canonicalized_for_quantization() -> None:
    positive = quantize_f32_bits(1, 2, [0x00000000, 0x80000000])
    negative = quantize_f32_bits(1, 2, [0x80000000, 0x00000000])
    assert positive == negative
    assert positive.scale_bits == (0,)
    assert positive.q_bytes == bytes(32)


def test_nonfinite_sources_have_stable_codes() -> None:
    with pytest.raises(Q8Error, match="DFE-QUANT-004") as source_error:
        quantize_f32_bits(1, 1, [0x7FC12345])
    assert source_error.value.diagnostic["code"] == "DFE-QUANT-004"


def test_division_is_not_a_binary64_double_round() -> None:
    # This midpoint is exact in the rational implementation.  The expected
    # result is the even lower significand.
    assert f32_div_bits(0x3F800001, 0x3F800000) == 0x3F800001


def test_quantization_accepts_positive_u32_k_before_length_check() -> None:
    with pytest.raises(Q8Error, match="DFE-QUANT-003") as error:
        quantize_f32_bits(1, 0xFFFFFFFF, ())
    assert error.value.context["expected"] == 0xFFFFFFFF


def test_f32_arithmetic_matches_packed_scalar_for_basic_values() -> None:
    def bits(value: float) -> int:
        return int(struct.unpack("<I", struct.pack("<f", value))[0])

    left, right = bits(1.25), bits(-0.5)
    assert f32_add_bits(left, right) == bits(0.75)
    assert f32_mul_bits(left, right) == bits(-0.625)
    assert len(dequantize_f32_bits(quantize_f32_bits(1, 1, [left]))) == 1


def test_ratio_to_q_clamps_after_ties_to_even() -> None:
    assert _ratio_to_q(float_to_f32_bits(127.5)) == 127
    assert _ratio_to_q(float_to_f32_bits(-127.5)) == -127
    assert _ratio_to_q(float_to_f32_bits(128.0)) == 127
    assert _ratio_to_q(float_to_f32_bits(-128.0)) == -127


def test_q8_weights_require_immutable_scale_tuple() -> None:
    with pytest.raises(Q8Error, match="DFE-QUANT-003"):
        Q8Weights(1, 1, 1, bytes(32), [0])  # type: ignore[arg-type]


def test_source_sequence_is_not_mutated() -> None:
    source = [0x3F800000, 0xBF000000]
    original = source.copy()
    quantize_f32_bits(1, 2, source)
    assert source == original


def test_one_hot_all_32_lanes_have_deterministic_layout() -> None:
    source: list[int] = []
    for lane in range(32):
        source.extend(0x3F800000 if index == lane else 0 for index in range(32))
    weights = quantize_f32_bits(32, 32, source)
    assert weights.blocks == 1
    assert weights.scale_bits == (0x3C010204,) * 32
    for row in range(32):
        assert weights.q_at(row, 0, row) == 127
        assert (
            sum(value != 0 for value in weights.q_values[row * 32 : (row + 1) * 32])
            == 1
        )


def test_representative_q8_diagnostics_match_closed_schema() -> None:
    operations: tuple[Callable[[], object], ...] = (
        lambda: quantize_f32_bits(0, 1, []),
        lambda: quantize_f32_bits(1, 1, [0x7F800000]),
        lambda: Q8Weights(1, 1, 1, bytes(32), [0]),  # type: ignore[arg-type]
        lambda: Q8Weights(1, 1, 1, bytes([0, 1] + [0] * 30), (0,)),
    )
    for operation in operations:
        with pytest.raises(Q8Error) as caught:
            operation()
        assert validate_data(caught.value.diagnostic, "diagnostic") == []

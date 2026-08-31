"""Source-versus-Q8 quantization quality evidence.

These tests characterize the error introduced by quantization and
dequantization of the source weights.  They do not exercise the generated
kernel comparator; generated-kernel correctness is a separate experiment.
"""

from __future__ import annotations

import random
from fractions import Fraction

from decodeforge.q8 import (
    Q8Weights,
    dequantize_f32_bits,
    float_to_f32_bits,
    quantize_f32_bits,
)

EXP_MASK = 0x7F800000
FRAC_MASK = 0x007FFFFF
SIGN_MASK = 0x80000000
MINSUB_BITS = 0x00000001


def _bits_to_fraction(bits: int) -> Fraction:
    """Decode a finite binary32 word exactly, independently of the reference."""

    exponent = (bits & EXP_MASK) >> 23
    fraction = bits & FRAC_MASK
    sign = -1 if bits & SIGN_MASK else 1
    if exponent == 0:
        return Fraction(sign * fraction, 1 << 149)
    significand = (1 << 23) | fraction
    power = exponent - 150
    if power >= 0:
        return Fraction(sign * significand * (1 << power))
    return Fraction(sign * significand, 1 << -power)


def _per_weight_bound(scale_bits: int) -> Fraction:
    """Conservative V1 error bound from the three rounded operations.

    With ``u = 2**-24``, the bound allows the integer RNE half-step, a
    conservative ``255*u`` allowance for division and dequantization-multiply
    rounding, and one half of a minimum-subnormal step for gradual underflow:

        scale * (0.5 + 255*u) + 2**-150

    All terms are Fractions so the assertion does not inherit binary64
    rounding from the test process.
    """

    scale = _bits_to_fraction(scale_bits)
    unit_roundoff = Fraction(1, 1 << 24)
    return scale * (Fraction(1, 2) + 255 * unit_roundoff) + Fraction(1, 1 << 150)


def _assert_interior_lane_bound(
    source_bits: tuple[int, ...], weights: Q8Weights
) -> int:
    decoded = dequantize_f32_bits(weights)
    checked = 0
    for index, (source_word, decoded_word) in enumerate(
        zip(source_bits, decoded, strict=True)
    ):
        row, logical_k = divmod(index, weights.k)
        block, lane = divmod(logical_k, 32)
        scale_bits = weights.scale_at(row, block)
        q = weights.q_at(row, block, lane)
        if scale_bits == 0:
            assert decoded_word == 0, f"zero-scale lane {index} did not decode to +0"
            continue
        # The serialized q value cannot tell whether an endpoint came from
        # the range clamp or was exactly representable.  Excluding q endpoints
        # leaves the unambiguously non-clamped lanes required by this bound.
        if abs(q) == 127:
            continue
        error = abs(_bits_to_fraction(source_word) - _bits_to_fraction(decoded_word))
        assert error <= _per_weight_bound(scale_bits), (
            f"lane {index}: q={q}, source={source_word:#010x}, "
            f"decoded={decoded_word:#010x}, error={error}"
        )
        checked += 1
    return checked


def test_directed_source_dequantization_respects_exact_v1_bound() -> None:
    source = tuple(
        float_to_f32_bits(value)
        for value in (
            1.3,
            -0.7,
            0.21,
            -0.11,
            3.75,
            -1.4,
            0.625,
            -0.03125,
            7.125,
            -2.25,
            0.875,
            -0.203125,
            0.015625,
            -4.5,
            1.0625,
            -0.8125,
            0.375,
            -0.0625,
            5.5,
            -3.125,
            0.09375,
            -0.875,
            2.75,
            -1.1875,
            0.5,
            -0.25,
            6.25,
            -2.875,
            0.15625,
            -0.046875,
            1.875,
            -0.5625,
        )
    )
    weights = quantize_f32_bits(1, 32, source)

    assert weights.scale_bits[0] != 0
    assert _assert_interior_lane_bound(source, weights) >= 20


def test_zero_blocks_padding_and_q8_storage_invariants_are_exact() -> None:
    source = tuple([0x80000000] * 33 + [0x3F800000] * 32 + [0x00000000])
    weights = quantize_f32_bits(2, 33, source)
    decoded = dequantize_f32_bits(weights)

    assert weights.scale_at(0, 0) == 0
    assert weights.scale_at(0, 1) == 0
    assert weights.scale_at(1, 1) == 0
    assert decoded[:33] == (0,) * 33
    assert decoded[33 + 32] == 0

    for row in range(weights.n):
        for block in range(weights.blocks):
            for lane in range(32):
                q = weights.q_at(row, block, lane)
                logical_k = block * 32 + lane
                assert -127 <= q <= 127
                if logical_k >= weights.k:
                    assert q == 0
    # The second row has a nonzero first block and an exactly zero tail block.
    assert weights.q_at(1, 0, 0) == 127
    assert all(weights.q_at(1, 1, lane) == 0 for lane in range(32))


def test_subnormal_clamp_is_reported_separately_from_half_step_bound() -> None:
    # This is the intentional clamp case: source magnitude is
    # 190 * 2**-149, scale rounds to minsub, and q=127.  Its error is
    # 63 * 2**-149, so the interior-lane bound is not applicable.
    source_word = 0x000000BE
    weights = quantize_f32_bits(1, 1, [source_word])
    decoded_word = dequantize_f32_bits(weights)[0]

    assert weights.scale_bits == (MINSUB_BITS,)
    assert weights.q_at(0, 0, 0) == 127
    assert decoded_word == 0x0000007F
    assert abs(_bits_to_fraction(source_word) - _bits_to_fraction(decoded_word)) == (
        Fraction(63, 1 << 149)
    )


def test_scale_underflow_to_zero_has_a_separate_exact_bound() -> None:
    # A nonzero block can still produce scale=+0 when amax/127 rounds below
    # half a minimum-subnormal step.  The largest such positive amax is
    # 63 * 2**-149; this is not an interior nonzero-scale lane and therefore
    # is intentionally outside _per_weight_bound.
    source_word = 0x0000003F
    weights = quantize_f32_bits(1, 1, [source_word])
    decoded_word = dequantize_f32_bits(weights)[0]

    assert weights.scale_bits == (0,)
    assert weights.q_at(0, 0, 0) == 0
    assert decoded_word == 0
    error = abs(_bits_to_fraction(source_word) - _bits_to_fraction(decoded_word))
    assert error == Fraction(63, 1 << 149)
    assert error <= Fraction(63, 1 << 149)


def test_seeded_moderate_random_rows_meet_bound_and_preserve_tails() -> None:
    rng = random.Random(0xDF08A11)
    n, k = 17, 37
    source = tuple(
        float_to_f32_bits(rng.randrange(-20000, 20001) / 4096.0) for _ in range(n * k)
    )
    weights = quantize_f32_bits(n, k, source)

    checked = _assert_interior_lane_bound(source, weights)
    assert checked >= 500
    for raw in weights.q_bytes:
        signed = raw if raw < 128 else raw - 256
        assert -127 <= signed <= 127
    for row in range(n):
        for lane in range(k, weights.blocks * 32):
            block, block_lane = divmod(lane, 32)
            assert weights.q_at(row, block, block_lane) == 0

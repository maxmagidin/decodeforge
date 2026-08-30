"""Independent exact IEEE-754 binary32 oracles for the Q8 arithmetic."""

from __future__ import annotations

import os
import random
from fractions import Fraction

from decodeforge.q8 import f32_add_bits, f32_div_bits, f32_mul_bits

SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
FRAC_MASK = 0x007FFFFF
ONE_BITS = 0x3F800000
MINNORMAL_BITS = 0x00800000
MINSUB_BITS = 0x00000001
MAXFINITE_BITS = 0x7F7FFFFF


def _round_positive(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and quotient & 1):
        return quotient + 1
    return quotient


def _floor_log2(numerator: int, denominator: int) -> int:
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    return exponent


def _bits_to_fraction(bits: int) -> Fraction:
    exponent = (bits & EXP_MASK) >> 23
    fraction = bits & FRAC_MASK
    assert exponent != 0xFF
    sign = -1 if bits & SIGN_MASK else 1
    if exponent == 0:
        return Fraction(sign * fraction, 1 << 149)
    significand = (1 << 23) | fraction
    power = exponent - 150
    if power >= 0:
        return Fraction(sign * significand * (1 << power))
    return Fraction(sign * significand, 1 << -power)


def _fraction_to_bits(value: Fraction, *, negative_zero: bool = False) -> int:
    if value == 0:
        return SIGN_MASK if negative_zero else 0
    sign_bit = SIGN_MASK if value < 0 else 0
    magnitude = abs(value)
    numerator = magnitude.numerator
    denominator = magnitude.denominator
    exponent = _floor_log2(numerator, denominator)
    if exponent < -126:
        subnormal = _round_positive(numerator << 149, denominator)
        if subnormal == 0:
            return sign_bit
        if subnormal >= 1 << 23:
            return sign_bit | MINNORMAL_BITS
        return sign_bit | subnormal

    significand = _round_positive(
        numerator << (exponent * -1 + 23) if exponent <= 23 else numerator,
        denominator if exponent <= 23 else denominator << (exponent - 23),
    )
    if significand >= 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        return sign_bit | EXP_MASK
    return sign_bit | ((exponent + 127) << 23) | (significand - (1 << 23))


def _oracle_add(left: int, right: int) -> int:
    left_zero = left & ~SIGN_MASK == 0
    right_zero = right & ~SIGN_MASK == 0
    if left_zero and right_zero:
        return SIGN_MASK if left & SIGN_MASK and right & SIGN_MASK else 0
    if left_zero:
        return right
    if right_zero:
        return left
    return _fraction_to_bits(_bits_to_fraction(left) + _bits_to_fraction(right))


def _oracle_mul(left: int, right: int) -> int:
    sign = bool((left ^ right) & SIGN_MASK)
    if left & ~SIGN_MASK == 0 or right & ~SIGN_MASK == 0:
        return SIGN_MASK if sign else 0
    return _fraction_to_bits(_bits_to_fraction(left) * _bits_to_fraction(right))


def _oracle_div(numerator: int, denominator: int) -> int:
    sign = bool((numerator ^ denominator) & SIGN_MASK)
    if numerator & ~SIGN_MASK == 0:
        return SIGN_MASK if sign else 0
    return _fraction_to_bits(
        _bits_to_fraction(numerator) / _bits_to_fraction(denominator)
    )


def test_directed_division_goldens_are_exact() -> None:
    cases = (
        (ONE_BITS, 0x40400000),  # 1 / 3
        (ONE_BITS, 0x41200000),  # 1 / 10
        (MINSUB_BITS, 0x40000000),  # minsub / 2 -> zero
        (0x00000003, 0x40000000),  # 3 minsub / 2 -> 2 minsubs
        (MINNORMAL_BITS, 0x40000000),  # minnormal / 2
        (0x00FFFFFF, 0x40000000),  # 0x00ffffff / 2 -> 0x00800000 carry
        (MAXFINITE_BITS, MINNORMAL_BITS),  # overflow
    )
    assert f32_div_bits(0x00FFFFFF, 0x40000000) == 0x00800000
    for numerator, denominator in cases:
        expected = _fraction_to_bits(
            _bits_to_fraction(numerator) / _bits_to_fraction(denominator)
        )
        assert f32_div_bits(numerator, denominator) == expected


def _random_class_word(rng: random.Random, index: int) -> int:
    sign = SIGN_MASK if rng.randrange(2) else 0
    category = index % 4
    if category == 0:
        return sign
    if category == 1:
        return sign | rng.randrange(1, 1 << 23)
    exponent = (1, 2, 126, 127, 254)[rng.randrange(5)]
    return sign | (exponent << 23) | rng.randrange(1 << 23)


def test_bounded_randomized_ieee_classes() -> None:
    raw_count = os.environ.get("DECODEFORGE_IEEE_DEEP_COUNT", "1024")
    count = max(0, int(raw_count))
    rng = random.Random(0xDF08)
    for index in range(count):
        left = _random_class_word(rng, index)
        right = _random_class_word(rng, index + 1)
        assert f32_add_bits(left, right) == _oracle_add(left, right)
        assert f32_mul_bits(left, right) == _oracle_mul(left, right)
        if right & ~SIGN_MASK:
            assert f32_div_bits(left, right) == _oracle_div(left, right)

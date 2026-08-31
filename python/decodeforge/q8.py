"""Bit-exact, dependency-free reference semantics for ``DFQ8_B32_V1``.

The module deliberately works with IEEE-754 binary32 bit words.  Keeping the
bits at the API boundary makes signed zeroes and NaN payloads observable and
prevents a host tensor library (or a platform rounding mode) from becoming
part of the format contract.

The reference is intentionally scalar and fairly conservative.  It is an
oracle for quantizers and generated kernels, not a high-throughput inference
implementation.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import TypeAlias

FORMAT = "DFQ8_B32_V1"
NUMERIC_MODE = "strict_f32_v1"
BLOCK_SIZE = 32
Q_MIN = -127
Q_MAX = 127
U32_MAX = 0xFFFFFFFF
# This is the largest K for which the predeclared comparator's t*u is < 1.
# Quantization itself accepts every positive-u32 K that can be represented by
# the derived storage checks below; this cap belongs only to comparator gamma.
MAX_COMPARATOR_K = 8_134_399
U = 2.0**-24

SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
FRAC_MASK = 0x007FFFFF
POSITIVE_MASK = 0x7FFFFFFF
F32_MAX_BITS = 0x7F7FFFFF
F32_ONE_BITS = 0x3F800000
F32_127_BITS = 0x42FE0000

Bits: TypeAlias = int
BitSequence: TypeAlias = Sequence[int]


class Q8Error(ValueError):
    """A stable semantic rejection with a machine-readable diagnostic."""

    def __init__(
        self,
        code: str,
        summary: str,
        context: dict[str, object] | None = None,
    ) -> None:
        self.code = code
        self.summary = summary
        self.context = dict(context or {})
        self.diagnostic = {
            "schema_version": 1,
            "code": code,
            "severity": "error",
            "component": "quant",
            "summary": summary,
            "context": self.context,
        }
        super().__init__(f"{code}: {summary}")


QuantizationError = Q8Error
ComparatorError = Q8Error
FixtureError = Q8Error


def _error(code: str, summary: str, **context: object) -> Q8Error:
    return Q8Error(code, summary, context)


def _validate_u32(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(
            "DFE-QUANT-001", "A dimension or bit word is not an integer.", field=name
        )
    if value < 0 or value > U32_MAX:
        raise _error(
            "DFE-QUANT-001",
            "A dimension or bit word is outside the unsigned binary32 range.",
            field=name,
            value=value,
        )
    return value


def _validate_bits(value: object, name: str = "bits") -> int:
    return _validate_u32(value, name)


def is_finite_f32_bits(bits: int) -> bool:
    """Return whether *bits* encodes a finite binary32 value."""

    bits = _validate_bits(bits)
    return (bits & EXP_MASK) != EXP_MASK


def f32_bits_to_float(bits: int) -> float:
    """Decode one validated binary32 bit word to a Python float."""

    bits = _validate_bits(bits)
    return float(struct.unpack("<f", struct.pack("<I", bits))[0])


def float_to_f32_bits(value: float) -> int:
    """Round a finite Python float to binary32 using IEEE round-to-nearest.

    This helper is for constructing test inputs.  Arithmetic in this module
    uses integer/rational operations instead; in particular division never
    passes through this conversion.
    """

    if not isinstance(value, float):
        raise TypeError("value must be a Python float")
    if not math.isfinite(value):
        raise _error(
            "DFE-QUANT-004", "A source value is not finite.", value=repr(value)
        )
    try:
        return int(struct.unpack("<I", struct.pack("<f", value))[0])
    except (OverflowError, struct.error) as exc:
        raise _error(
            "DFE-QUANT-004",
            "A source value is outside binary32 range.",
            value=repr(value),
        ) from exc


def _components(bits: int) -> tuple[int, int, int, bool]:
    """Return (sign, significand, power-of-two exponent, is_zero)."""

    bits = _validate_bits(bits)
    sign = -1 if bits & SIGN_MASK else 1
    exponent = (bits & EXP_MASK) >> 23
    fraction = bits & FRAC_MASK
    if exponent == 0:
        return sign, fraction, -149, fraction == 0
    if exponent == 0xFF:
        raise ValueError("non-finite binary32 value has no finite components")
    return sign, 0x800000 | fraction, exponent - 150, False


def _round_integer_ratio(numerator: int, denominator: int) -> int:
    """Round a non-negative rational to an integer, ties-to-even."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("invalid non-negative rational")
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder << 1
    if twice > denominator or (twice == denominator and quotient & 1):
        return quotient + 1
    return quotient


def _round_scaled(numerator: int, denominator: int, shift: int) -> int:
    """Return RNE(numerator / denominator * 2**shift)."""

    if shift >= 0:
        numerator <<= shift
    else:
        denominator <<= -shift
    return _round_integer_ratio(numerator, denominator)


def _floor_log2_ratio(numerator: int, denominator: int) -> int:
    """Return floor(log2(numerator / denominator)) for positive integers."""

    if numerator <= 0 or denominator <= 0:
        raise ValueError("log2 ratio requires positive integers")
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < (denominator << exponent):
            exponent -= 1
    elif (numerator << -exponent) < denominator:
        exponent -= 1
    return exponent


def _round_rational_f32(
    numerator: int,
    denominator: int = 1,
    binary_exponent: int = 0,
    sign: int = 1,
) -> int:
    """Round ``sign * numerator / denominator * 2**binary_exponent`` to f32.

    The implementation handles normal and gradual-underflow results directly
    as integer ratios.  It is therefore suitable for exact midpoint tests and
    does not depend on Python's binary64 division.
    """

    if numerator < 0 or denominator <= 0 or sign not in {-1, 1}:
        raise ValueError("invalid rational f32 arguments")
    sign_bit = SIGN_MASK if sign < 0 else 0
    if numerator == 0:
        return sign_bit

    exponent = _floor_log2_ratio(numerator, denominator) + binary_exponent
    if exponent < -126:
        # A subnormal unit is 2**-149.  Rounding can carry exactly to the
        # smallest normal, which is represented by exponent field one.
        subnormal = _round_scaled(numerator, denominator, binary_exponent + 149)
        if subnormal == 0:
            return sign_bit
        if subnormal >= 1 << 23:
            return sign_bit | (1 << 23)
        return sign_bit | subnormal

    # A normal has 24 significant bits, with the leading bit implicit.
    significand = _round_scaled(
        numerator,
        denominator,
        binary_exponent - exponent + 23,
    )
    if significand >= 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        return sign_bit | EXP_MASK
    if exponent < -126:  # defensive: the subnormal branch should handle this
        subnormal = _round_scaled(numerator, denominator, binary_exponent + 149)
        if subnormal == 0:
            return sign_bit
        if subnormal >= 1 << 23:
            return sign_bit | (1 << 23)
        return sign_bit | subnormal
    return sign_bit | ((exponent + 127) << 23) | (significand - (1 << 23))


def _finite_components(bits: int) -> tuple[int, int, int, bool]:
    if not is_finite_f32_bits(bits):
        raise ValueError("binary32 operand is not finite")
    return _components(bits)


def f32_mul_bits(left: int, right: int) -> int:
    """Multiply two finite binary32 values and round once to binary32."""

    sign_left, num_left, power_left, zero_left = _finite_components(left)
    sign_right, num_right, power_right, zero_right = _finite_components(right)
    sign = sign_left * sign_right
    if zero_left or zero_right:
        return SIGN_MASK if sign < 0 else 0
    return _round_rational_f32(
        num_left * num_right,
        binary_exponent=power_left + power_right,
        sign=sign,
    )


def f32_add_bits(left: int, right: int) -> int:
    """Add two finite binary32 values and round once to binary32."""

    sign_left, num_left, power_left, zero_left = _finite_components(left)
    sign_right, num_right, power_right, zero_right = _finite_components(right)
    if zero_left and zero_right:
        # IEEE round-to-nearest chooses +0 for opposite-signed zeroes and for
        # an exact cancellation.  Preserve -0 only when both operands are -0.
        if sign_left < 0 and sign_right < 0:
            return SIGN_MASK
        return 0
    if zero_left:
        return right
    if zero_right:
        return left

    common_power = min(power_left, power_right)
    signed_left = sign_left * (num_left << (power_left - common_power))
    signed_right = sign_right * (num_right << (power_right - common_power))
    total = signed_left + signed_right
    if total == 0:
        return 0
    return _round_rational_f32(
        abs(total),
        binary_exponent=common_power,
        sign=1 if total > 0 else -1,
    )


def f32_div_bits(numerator_bits: int, denominator_bits: int) -> int:
    """Divide finite binary32 values with exact rational RNE semantics."""

    sign_numerator, num, power_num, zero_num = _finite_components(numerator_bits)
    sign_denominator, den, power_denominator, zero_denominator = _finite_components(
        denominator_bits
    )
    if zero_denominator:
        raise ZeroDivisionError("binary32 division by zero")
    sign = sign_numerator * sign_denominator
    if zero_num:
        return SIGN_MASK if sign < 0 else 0
    return _round_rational_f32(
        num,
        den,
        power_num - power_denominator,
        sign,
    )


def f32_from_int(value: int) -> int:
    """Return the exactly rounded binary32 representation of an integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if value == 0:
        return 0
    return _round_rational_f32(abs(value), sign=1 if value > 0 else -1)

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

import hashlib
import math
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import FrozenInstanceError, dataclass
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


def _round_f32_to_int(bits: int) -> int:
    """Round one finite binary32 value to an integer, ties-to-even."""

    sign, numerator, power, is_zero = _finite_components(bits)
    if is_zero:
        return 0
    magnitude = _round_scaled(numerator, 1, power)
    return sign * magnitude


def _ratio_to_q(ratio_bits: int) -> int:
    """Round one binary32 ratio and clamp it to the signed DFQ8 range."""

    ratio = _validate_bits(ratio_bits, "ratio_bits")
    if not is_finite_f32_bits(ratio):
        # A finite source divided by a positive finite scale can overflow to
        # infinity.  It is still unambiguously beyond the representable q
        # range; NaN cannot arise from that operation and is rejected if an
        # alternate caller supplies it.
        if ratio & FRAC_MASK:
            raise _error(
                "DFE-QUANT-010",
                "A quantization ratio is not finite.",
                field="ratio_bits",
                bits=ratio,
            )
        return Q_MIN if ratio & SIGN_MASK else Q_MAX
    rounded = _round_f32_to_int(ratio)
    return max(Q_MIN, min(Q_MAX, rounded))


def _coerce_bits_sequence(values: Iterable[object], *, field: str) -> tuple[int, ...]:
    try:
        result = tuple(_validate_bits(value, field) for value in values)
    except TypeError as exc:
        raise _error(
            "DFE-QUANT-003", "A bit-word sequence is not iterable.", field=field
        ) from exc
    return result


def _shape(n: object, k: object) -> tuple[int, int, int]:
    n_value = _validate_u32(n, "n")
    k_value = _validate_u32(k, "k")
    if n_value == 0 or k_value == 0:
        raise _error(
            "DFE-QUANT-001", "N and K must both be positive.", n=n_value, k=k_value
        )
    blocks = (k_value + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Keep every derived byte count representable by Python's sequence APIs;
    # importantly this check runs before any result allocation.
    counts = (
        ("source", n_value, k_value),
        ("q", n_value, blocks * BLOCK_SIZE),
        ("scale", n_value, blocks),
    )
    for name, left, right in counts:
        if left > (2**63 - 1) // max(1, right):
            raise _error("DFE-QUANT-002", "Derived storage size overflows.", field=name)
    return n_value, k_value, blocks


def _check_length(values: Sequence[object], expected: int, field: str) -> None:
    if len(values) != expected:
        raise _error(
            "DFE-QUANT-003",
            "A bit-word sequence has the wrong length.",
            field=field,
            expected=expected,
            actual=len(values),
        )


def _check_finite(values: Sequence[int], field: str, code: str) -> None:
    for index, bits in enumerate(values):
        if not is_finite_f32_bits(bits):
            raise _error(
                code,
                "A binary32 value is not finite.",
                field=field,
                index=index,
                bits=bits,
            )


def _signed_q(raw: int) -> int:
    return raw if raw < 128 else raw - 256


def _raw_q(value: int) -> int:
    if value < Q_MIN or value > Q_MAX:
        raise ValueError("q value outside DFQ8 range")
    return value & 0xFF


@dataclass(frozen=True)
class Q8Weights:
    """Immutable logical shape plus raw Q8/scales storage.

    ``q_bytes`` is row-major ``[N][B][32]`` two's-complement storage.  The
    scale words are kept as integer binary32 bits in row-major ``[N][B]``
    order.  Both are intentionally exposed in raw form for hashing and parity
    checks; convenience properties expose signed q values and scale floats.
    """

    n: int
    k: int
    blocks: int
    q_bytes: bytes
    scale_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        n, k, blocks = _shape(self.n, self.k)
        if not isinstance(self.blocks, int) or isinstance(self.blocks, bool):
            raise _error(
                "DFE-QUANT-001",
                "The block count is not an integer.",
                field="blocks",
            )
        if self.blocks != blocks:
            raise _error(
                "DFE-QUANT-001",
                "The declared block count does not equal ceil(K/32).",
                expected=blocks,
                actual=self.blocks,
            )
        if not isinstance(self.q_bytes, bytes):
            raise _error(
                "DFE-QUANT-003", "q_bytes must be immutable bytes.", field="q_bytes"
            )
        if not isinstance(self.scale_bits, tuple):
            raise _error(
                "DFE-QUANT-003",
                "scale_bits must be an immutable tuple.",
                field="scale_bits",
            )
        expected_q = n * blocks * BLOCK_SIZE
        expected_scales = n * blocks
        if len(self.q_bytes) != expected_q or len(self.scale_bits) != expected_scales:
            raise _error(
                "DFE-QUANT-003",
                "Stored q or scale length does not match the shape.",
                expected_q=expected_q,
                actual_q=len(self.q_bytes),
                expected_scales=expected_scales,
                actual_scales=len(self.scale_bits),
            )
        for index, bits in enumerate(self.scale_bits):
            try:
                _validate_bits(bits, "scale_bits")
            except Q8Error as error:
                raise _error(
                    "DFE-QUANT-006",
                    "A scale is not a valid binary32 bit word.",
                    index=index,
                    value=bits,
                ) from error
            if not is_finite_f32_bits(bits) or bits & SIGN_MASK:
                raise _error(
                    "DFE-QUANT-006",
                    "A scale must be finite and non-negative.",
                    index=index,
                    bits=bits,
                )
        for row in range(n):
            for block in range(blocks):
                scale = self.scale_bits[row * blocks + block]
                start = (row * blocks + block) * BLOCK_SIZE
                end = start + BLOCK_SIZE
                for index in range(start, end):
                    q = _signed_q(self.q_bytes[index])
                    if q < Q_MIN or q > Q_MAX:
                        raise _error(
                            "DFE-QUANT-007",
                            "A q byte is outside the signed DFQ8 range.",
                            index=index,
                            value=q,
                        )
                    if (
                        block * BLOCK_SIZE + (index - start) >= k
                        and self.q_bytes[index] != 0
                    ):
                        raise _error(
                            "DFE-QUANT-007",
                            "A padded q lane is not zero.",
                            row=row,
                            block=block,
                            lane=index - start,
                        )
                    if scale == 0 and self.q_bytes[index] != 0:
                        raise _error(
                            "DFE-QUANT-007",
                            "A zero-scale block contains a nonzero q lane.",
                            row=row,
                            block=block,
                        )

    @property
    def b(self) -> int:
        """Read-only compatibility alias for the canonical ``blocks`` field."""

        return self.blocks

    @b.setter
    def b(self, value: int) -> None:
        del value
        raise FrozenInstanceError("cannot assign to field 'b'")

    @property
    def raw_q_bytes(self) -> bytes:
        return self.q_bytes

    @property
    def scales(self) -> tuple[int, ...]:
        return self.scale_bits

    @property
    def scale_bytes(self) -> bytes:
        return struct.pack(f"<{len(self.scale_bits)}I", *self.scale_bits)

    @property
    def q_values(self) -> tuple[int, ...]:
        return tuple(_signed_q(value) for value in self.q_bytes)

    def q_at(self, row: int, block: int, lane: int) -> int:
        if not (
            0 <= row < self.n and 0 <= block < self.blocks and 0 <= lane < BLOCK_SIZE
        ):
            raise IndexError("q index out of range")
        return _signed_q(self.q_bytes[(row * self.blocks + block) * BLOCK_SIZE + lane])

    def scale_at(self, row: int, block: int) -> int:
        if not (0 <= row < self.n and 0 <= block < self.blocks):
            raise IndexError("scale index out of range")
        return self.scale_bits[row * self.blocks + block]


def quantize_f32_bits(n: int, k: int, source_bits: Iterable[int]) -> Q8Weights:
    """Quantize row-major source binary32 words into immutable DFQ8 storage."""

    n_value, k_value, blocks = _shape(n, k)
    source = _coerce_bits_sequence(source_bits, field="source_fp32_bits")
    _check_length(source, n_value * k_value, "source_fp32_bits")
    _check_finite(source, "source_fp32_bits", "DFE-QUANT-004")

    q = bytearray(n_value * blocks * BLOCK_SIZE)
    scales: list[int] = []
    for row in range(n_value):
        source_row = source[row * k_value : (row + 1) * k_value]
        for block in range(blocks):
            first = block * BLOCK_SIZE
            last = min(first + BLOCK_SIZE, k_value)
            # Clearing the sign bit normalizes -0 while retaining all finite
            # positive magnitudes and their exact payload-free bit patterns.
            amax = max((bits & POSITIVE_MASK) for bits in source_row[first:last])
            if amax == 0:
                scale = 0
            else:
                scale = f32_div_bits(amax, F32_127_BITS)
                if not is_finite_f32_bits(scale) or scale & SIGN_MASK:
                    raise _error(
                        "DFE-QUANT-006",
                        "Quantization produced an invalid scale.",
                        row=row,
                        block=block,
                        bits=scale,
                    )
            scales.append(scale)
            out_start = (row * blocks + block) * BLOCK_SIZE
            if scale == 0:
                # bytearray was zero-initialized; keep the explicit branch as
                # an invariant marker for reviewers and alternate callers.
                continue
            for lane in range(last - first):
                source_value = source[row * k_value + first + lane]
                ratio = f32_div_bits(source_value, scale)
                q[out_start + lane] = _raw_q(_ratio_to_q(ratio))
            # Remaining lanes are already exactly zero padding.
    return Q8Weights(n_value, k_value, blocks, bytes(q), tuple(scales))


def dequantize_f32_bits(weights: Q8Weights) -> tuple[int, ...]:
    """Return logical row-major dequantized binary32 words (length ``N*K``)."""

    if not isinstance(weights, Q8Weights):
        raise TypeError("weights must be Q8Weights")
    result: list[int] = []
    for row in range(weights.n):
        for logical_k in range(weights.k):
            block, lane = divmod(logical_k, BLOCK_SIZE)
            q_bits = f32_from_int(weights.q_at(row, block, lane))
            result.append(f32_mul_bits(q_bits, weights.scale_at(row, block)))
    return tuple(result)


def dequantize_f32_rows_bits(weights: Q8Weights) -> tuple[tuple[int, ...], ...]:
    """Return dequantized words grouped by output row."""

    flat = dequantize_f32_bits(weights)
    return tuple(
        flat[row * weights.k : (row + 1) * weights.k] for row in range(weights.n)
    )


def canonical_linear_f32_bits(
    input_bits: Iterable[int],
    weights: Q8Weights,
) -> tuple[int, ...]:
    """Evaluate one ``[1,K]`` input in the strict canonical scalar order."""

    if not isinstance(weights, Q8Weights):
        raise TypeError("weights must be Q8Weights")
    input_words = _coerce_bits_sequence(input_bits, field="input_fp32_bits")
    _check_length(input_words, weights.k, "input_fp32_bits")
    _check_finite(input_words, "input_fp32_bits", "DFE-QUANT-005")

    outputs: list[int] = []
    for row in range(weights.n):
        output = 0  # canonical +0
        for block in range(weights.blocks):
            block_sum = 0  # canonical +0
            first = block * BLOCK_SIZE
            last = min(first + BLOCK_SIZE, weights.k)
            scale = weights.scale_at(row, block)
            for logical_k in range(first, last):
                q_bits = f32_from_int(weights.q_at(row, block, logical_k - first))
                product = f32_mul_bits(input_words[logical_k], q_bits)
                if not is_finite_f32_bits(product):
                    raise _error(
                        "DFE-QUANT-010",
                        "Canonical evaluation produced a non-finite product.",
                        row=row,
                        block=block,
                        lane=logical_k - first,
                    )
                block_sum = f32_add_bits(block_sum, product)
                if not is_finite_f32_bits(block_sum):
                    raise _error(
                        "DFE-QUANT-010",
                        "Canonical evaluation produced a non-finite block sum.",
                        row=row,
                        block=block,
                    )
            scaled = f32_mul_bits(block_sum, scale)
            if not is_finite_f32_bits(scaled):
                raise _error(
                    "DFE-QUANT-010",
                    "Canonical evaluation produced a non-finite scaled block.",
                    row=row,
                    block=block,
                )
            output = f32_add_bits(output, scaled)
            if not is_finite_f32_bits(output):
                raise _error(
                    "DFE-QUANT-010",
                    "Canonical evaluation produced a non-finite output.",
                    row=row,
                )
        outputs.append(output)
    return tuple(outputs)


def _u32_le(words: Iterable[int]) -> bytes:
    values = tuple(_validate_bits(word) for word in words)
    return struct.pack(f"<{len(values)}I", *values)


def logical_weight_identity(weights: Q8Weights) -> str:
    """Hash the canonical logical-weight preimage and return ``sha256:...``."""

    if not isinstance(weights, Q8Weights):
        raise TypeError("weights must be Q8Weights")
    preimage = (
        b"DecodeForge/DFQ8_B32_V1/logical-weight/v1\0"
        + struct.pack("<III", weights.n, weights.k, weights.blocks)
        + weights.q_bytes
        + weights.scale_bytes
    )
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def fixture_identity(
    n: int,
    k: int,
    source_bits: Iterable[int],
    weights: Q8Weights,
    input_bits: Iterable[int],
    output_bits: Iterable[int],
) -> str:
    """Hash a complete quantization/evaluation fixture preimage."""

    n_value, k_value, blocks = _shape(n, k)
    if weights.n != n_value or weights.k != k_value or weights.blocks != blocks:
        raise _error("DFE-QUANT-008", "Fixture dimensions disagree with weights.")
    source = _coerce_bits_sequence(source_bits, field="source_fp32_bits")
    inputs = _coerce_bits_sequence(input_bits, field="input_fp32_bits")
    outputs = _coerce_bits_sequence(output_bits, field="expected_output_fp32_bits")
    _check_length(source, n_value * k_value, "source_fp32_bits")
    _check_length(inputs, k_value, "input_fp32_bits")
    _check_length(outputs, n_value, "expected_output_fp32_bits")
    _check_finite(source, "source_fp32_bits", "DFE-QUANT-004")
    _check_finite(inputs, "input_fp32_bits", "DFE-QUANT-005")
    _check_finite(outputs, "expected_output_fp32_bits", "DFE-QUANT-010")
    preimage = (
        b"DecodeForge/DFQ8_B32_V1/strict_f32_v1/quant-fixture/v1\0"
        + struct.pack("<II", n_value, k_value)
        + _u32_le(source)
        + weights.scale_bytes
        + weights.q_bytes
        + _u32_le(inputs)
        + _u32_le(outputs)
    )
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def _ordered_ulp(bits: int) -> int:
    # Put both zero encodings at the same conceptual rank while retaining one
    # step from either zero to the adjacent signed min-subnormal.
    magnitude = bits & POSITIVE_MASK
    return 0x80000000 - magnitude if bits & SIGN_MASK else 0x80000000 + magnitude


def ulp_distance(left_bits: int, right_bits: int) -> int:
    """Return finite binary32 ULP distance, treating +/-0 as equal."""

    left = _validate_bits(left_bits, "left_bits")
    right = _validate_bits(right_bits, "right_bits")
    if not is_finite_f32_bits(left) or not is_finite_f32_bits(right):
        raise _error("DFE-QUANT-009", "ULP comparison requires finite values.")
    if left & POSITIVE_MASK == 0 and right & POSITIVE_MASK == 0:
        return 0
    return abs(_ordered_ulp(left) - _ordered_ulp(right))


@dataclass(frozen=True, slots=True)
class OutputComparison:
    """One output's strict-f32 comparison metrics and pass/fail decision."""

    index: int
    actual_bits: int
    reference_bits: int
    absolute_error: float
    relative_error: float
    ulp: int
    l1_magnitude: float
    tolerance: float
    factor: float
    passed: bool

    @property
    def ulp_distance(self) -> int:
        return self.ulp


def _comparison_bits(values: Iterable[object], field: str) -> tuple[int, ...]:
    result: list[int] = []
    try:
        iterator: Iterator[object] = iter(values)
    except TypeError as exc:
        raise _error(
            "DFE-QUANT-009", "Comparator values are not iterable.", field=field
        ) from exc
    for index, value in enumerate(iterator):
        if isinstance(value, bool):
            raise _error(
                "DFE-QUANT-009",
                "Comparator value is not finite.",
                field=field,
                index=index,
            )
        if isinstance(value, int):
            bits = _validate_bits(value, field)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise _error(
                    "DFE-QUANT-009",
                    "Comparator value is not finite.",
                    field=field,
                    index=index,
                )
            try:
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
            except (OverflowError, struct.error) as exc:
                raise _error(
                    "DFE-QUANT-009",
                    "Comparator value is outside binary32 range.",
                    field=field,
                    index=index,
                ) from exc
        else:
            raise _error(
                "DFE-QUANT-009",
                "Comparator value is not a bit word or float.",
                field=field,
                index=index,
            )
        if not is_finite_f32_bits(bits):
            raise _error(
                "DFE-QUANT-009",
                "Comparator value is not finite.",
                field=field,
                index=index,
                bits=bits,
            )
        result.append(bits)
    return tuple(result)


def compare_strict_f32_v1(
    actual: Iterable[int | float],
    input_bits: Iterable[int],
    weights: Q8Weights,
    *,
    generated_scalar: bool = False,
) -> tuple[OutputComparison, ...]:
    """Compare candidate outputs against internally computed canonical output.

    Values may be finite Python floats or raw f32 bit words.  The canonical
    reference is always computed from ``input_bits`` and ``weights`` in the
    strict scalar order.  ``A`` is summed in ascending row-major order using
    binary64 Python floats.  Generated scalar candidates use the fixed
    factor-one and ULP-2 policy; all other candidates use fixed factor four.
    """

    if not isinstance(weights, Q8Weights):
        raise TypeError("weights must be Q8Weights")
    if weights.k > MAX_COMPARATOR_K:
        raise _error(
            "DFE-QUANT-009",
            "Comparator K exceeds the strict-f32 reduction-bound limit.",
            k=weights.k,
            maximum_k=MAX_COMPARATOR_K,
        )
    actual_bits = _comparison_bits(actual, "actual")
    if len(actual_bits) != weights.n:
        raise _error(
            "DFE-QUANT-003",
            "Comparator output length does not match N.",
            expected=weights.n,
            actual=len(actual_bits),
        )
    inputs = _coerce_bits_sequence(input_bits, field="input_fp32_bits")
    _check_length(inputs, weights.k, "input_fp32_bits")
    _check_finite(inputs, "input_fp32_bits", "DFE-QUANT-009")
    reference_bits = canonical_linear_f32_bits(inputs, weights)
    factor_value = 1.0 if generated_scalar else 4.0

    t = 2 * weights.k + 2 * weights.blocks + 16
    denominator = 1.0 - t * U
    if denominator <= 0:
        raise _error(
            "DFE-QUANT-009", "Comparator reduction bound is outside its domain.", t=t
        )
    gamma = (t * U) / denominator
    comparisons: list[OutputComparison] = []
    for row, (actual_word, reference_word) in enumerate(
        zip(actual_bits, reference_bits, strict=False)
    ):
        actual_value = f32_bits_to_float(actual_word)
        reference_value = f32_bits_to_float(reference_word)
        magnitude = 0.0
        # Canonical ascending order is explicit here; no sum/reduction helper
        # is allowed to choose an alternate order.
        for logical_k, input_word in enumerate(inputs):
            block, lane = divmod(logical_k, BLOCK_SIZE)
            q_value = weights.q_at(row, block, lane)
            scale_value = f32_bits_to_float(weights.scale_at(row, block))
            term = abs(f32_bits_to_float(input_word) * float(q_value) * scale_value)
            magnitude += term
        absolute = abs(actual_value - reference_value)
        if reference_word & POSITIVE_MASK == 0:
            relative = 0.0 if actual_word & POSITIVE_MASK == 0 else math.inf
        else:
            relative = absolute / abs(reference_value)
        tolerance = max(1e-7, factor_value * gamma * magnitude)
        distance = ulp_distance(actual_word, reference_word)
        passed = absolute <= tolerance
        if generated_scalar:
            passed = passed and distance <= 2
        comparisons.append(
            OutputComparison(
                index=row,
                actual_bits=actual_word,
                reference_bits=reference_word,
                absolute_error=absolute,
                relative_error=relative,
                ulp=distance,
                l1_magnitude=magnitude,
                tolerance=tolerance,
                factor=factor_value,
                passed=passed,
            )
        )
    return tuple(comparisons)


__all__ = [
    "BLOCK_SIZE",
    "FORMAT",
    "MAX_COMPARATOR_K",
    "NUMERIC_MODE",
    "OutputComparison",
    "Q8Error",
    "Q8Weights",
    "canonical_linear_f32_bits",
    "compare_strict_f32_v1",
    "dequantize_f32_bits",
    "dequantize_f32_rows_bits",
    "f32_add_bits",
    "f32_bits_to_float",
    "f32_div_bits",
    "f32_from_int",
    "f32_mul_bits",
    "fixture_identity",
    "float_to_f32_bits",
    "is_finite_f32_bits",
    "logical_weight_identity",
    "quantize_f32_bits",
    "ulp_distance",
]

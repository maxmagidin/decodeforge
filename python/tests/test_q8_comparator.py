"""Fault-injection evidence for the strict-f32 generated-kernel comparator."""

from __future__ import annotations

import random

from decodeforge.q8 import (
    Q8Weights,
    compare_strict_f32_v1,
    f32_bits_to_float,
    float_to_f32_bits,
)


def _hand_built_weights() -> Q8Weights:
    """Build one two-block object without going through the quantizer."""

    q_values = [(-1 if lane % 2 else 1) * (lane % 9 + 1) for lane in range(32)]
    q_values.extend([-23])  # logical tail lane K-32
    q_values.extend([0] * 31)  # physical padding
    return Q8Weights(
        n=1,
        k=33,
        blocks=2,
        q_bytes=bytes(value & 0xFF for value in q_values),
        scale_bits=(0x3DCCCCCD, 0x3E4CCCCD),  # approximately 0.1 and 0.2
    )


def _inputs() -> tuple[int, ...]:
    return tuple(
        float_to_f32_bits((-1.0 if lane % 3 == 0 else 1.0) * ((lane % 11 + 1) / 17.0))
        for lane in range(33)
    )


def _independent_evaluator(
    weights: Q8Weights,
    input_bits: tuple[int, ...],
    *,
    use_block_scales: bool = True,
    include_tail: bool = True,
    sign_extend_q: bool = True,
    reorder_first_block: bool = False,
) -> tuple[int, ...]:
    """Evaluate candidates with Python floats, independently of the oracle.

    Each switch intentionally models one common generated-kernel mistake.
    The comparator itself remains the sole source of the canonical reference.
    """

    outputs: list[int] = []
    for row in range(weights.n):
        total = 0.0
        for block in range(weights.blocks):
            if not include_tail and block == weights.blocks - 1:
                continue
            block_total = 0.0
            first = block * 32
            last = min(first + 32, weights.k)
            for logical_k in range(first, last):
                lane = logical_k - first
                address_lane = lane
                if reorder_first_block and block == 0:
                    address_lane = (lane + 1) % 32
                raw_q = weights.q_bytes[
                    (row * weights.blocks + block) * 32 + address_lane
                ]
                q = raw_q
                if sign_extend_q and raw_q >= 128:
                    q -= 256
                block_total += f32_bits_to_float(input_bits[logical_k]) * float(q)
            scale = (
                f32_bits_to_float(weights.scale_at(row, block))
                if use_block_scales
                else 1.0
            )
            total += block_total * scale
        outputs.append(float_to_f32_bits(float(total)))
    return tuple(outputs)


def _assert_rejected(
    candidate: tuple[int, ...], input_bits: tuple[int, ...], weights: Q8Weights
) -> None:
    comparison = compare_strict_f32_v1(candidate, input_bits, weights)
    assert len(comparison) == 1
    assert comparison[0].factor == 4.0
    assert not comparison[0].passed


def test_comparator_rejects_hand_built_q8_faults() -> None:
    weights = _hand_built_weights()
    input_bits = _inputs()
    correct = _independent_evaluator(weights, input_bits)
    correct_comparison = compare_strict_f32_v1(correct, input_bits, weights)
    assert correct_comparison[0].passed

    _assert_rejected(
        _independent_evaluator(weights, input_bits, use_block_scales=False),
        input_bits,
        weights,
    )
    _assert_rejected(
        _independent_evaluator(weights, input_bits, include_tail=False),
        input_bits,
        weights,
    )
    _assert_rejected(
        _independent_evaluator(weights, input_bits, sign_extend_q=False),
        input_bits,
        weights,
    )
    _assert_rejected(
        _independent_evaluator(weights, input_bits, reorder_first_block=True),
        input_bits,
        weights,
    )


def test_comparator_rejects_deterministic_material_corruption_loop() -> None:
    weights = _hand_built_weights()
    input_bits = _inputs()
    correct = _independent_evaluator(weights, input_bits)
    assert compare_strict_f32_v1(correct, input_bits, weights)[0].passed

    rng = random.Random(0xDF08C0DE)
    base = f32_bits_to_float(correct[0])
    for _ in range(32):
        delta = (0.25 + rng.random() * 0.75) * (-1.0 if rng.randrange(2) else 1.0)
        corrupted = (float_to_f32_bits(base + delta),)
        _assert_rejected(corrupted, input_bits, weights)

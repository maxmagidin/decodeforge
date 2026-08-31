# ADR 0002: Fix strict output lanes and OI4 packing for the first G1 slice

- **Status:** Accepted for G1 contract work
- **Date:** 2026-08-31
- **Decision owner:** Repository owner

## Context

The G0 oracle is intentionally target-independent, while a first compiler slice
needs one reproducible physical layout that both scalar and future NEON kernels
can consume. Reduction-vector layouts make the strict-f32 recurrence ambiguous:
they invite horizontal reductions, reassociation, and multiple `K` partial sums.

## Decision

The first G1 Region/Loop contract is `M=1`, `strict_f32_v1`, and output-vector
oriented. A panel has `T=4` output lanes and `B=ceil(K/32)` records. The fixed
layout name is `DFQ8_B32_OI4_V1`; each `(panel, block)` record is exactly 144
bytes: four little-endian FP32 scale words followed by 32 groups of four q
bytes. Payload bytes begin at offset zero and contain no header. Missing output
lanes and physical `K` padding are zero-filled, while logical evaluation never
visits `K` padding. The payload requirement is 16-byte alignment and is
represented by the pack specification/manifest.

Loop IR fixes one `K` partial accumulator, ascending blocks and logical lanes,
and separate RN32 multiply/add operations. Scalar uses one lane; NEON uses four
output lanes. `N` tails use scalar cleanup. Both variants must consume identical
pack bytes.

## Consequences

- A deterministic pack identity can be checked independently of the logical
  G0 weight identity.
- Future scalar and NEON code generation share offsets and cannot silently
  evaluate padded lanes or change reduction order.
- Other vector axes, unrolling, horizontal reductions, FMA, C emission, and
  target-specific scheduling remain later work and are not implied by this
  contract.

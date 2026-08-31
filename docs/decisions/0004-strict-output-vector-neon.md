# ADR 0004: Freeze the strict output-vector NEON lowering

- **Status:** Accepted for the G1 ARM64 implementation
- **Date:** 2026-08-31
- **Decision owner:** Repository owner

## Context

The verified Q8 region, Loop IR, and `DFQ8_B32_OI4_V1` pack already define a
four-output physical panel. A first SIMD kernel needs to make its mapping
explicit so that source, assembly, and correctness evidence describe the same
algorithm. The strict-f32 contract also rules out a reduction across lanes,
fused multiply-add, reassociation, and evaluation of physically padded K
lanes.

## Decision

The first NEON backend emits one fixed C11/ARM64 schedule:

1. vectorize four independent output rows across the OI4 panel (`N` axis);
2. traverse blocks in ascending order and lanes within each block in ascending
   order, with one independent scalar-equivalent recurrence in each vector
   lane;
3. load four signed Q8 values from the current lane, widen through
   `int8 -> int16 -> int32`, convert to four FP32 values, and broadcast the
   scalar activation to all four lanes;
4. use explicit `vmulq_f32` followed by `vaddq_f32` for each product and
   accumulation, then multiply the four block sums by the four stored scales;
5. evaluate only the logical K lanes (`lane_count = min(32, K-block_start)`);
6. use the scalar ABI-compatible cleanup loop for the final one to three N
   rows when `N` is not a multiple of four.

The generated source contains compile-time guards for AArch64, NEON, Clang,
and little-endian targets. It does not select a schedule at runtime, inspect
host state, or embed weights. Its module identity uses a distinct
`DecodeForge/generated-module/neon-c/v1` domain and includes the canonical
NEON Loop IR schedule, numeric mode, ABI version, and source format. Thus a
NEON artifact cannot be confused with a scalar artifact, while changing only
weight values does not change the code identity.

The generated public boundary is exactly the scalar G1 ABI:
`df_abi_version`, `df_artifact_id`, and `df_run_v1`. The guard order, floating
point environment handling, status values, and private-output responsibility
remain those frozen by [ADR 0003](0003-scalar-generated-abi.md). This ADR
freezes source lowering only; native compilation, dynamic loading, and runtime
execution are separate follow-on work.

## Evidence required before promotion

- deterministic source and identity tests for repeated emission, shape/tail
  cases, domain separation, and absence of host/weight data;
- strict C syntax validation on an ARM64 Apple Clang toolchain;
- assembly audit showing widening/conversion, explicit multiply/add, no FMA,
  no horizontal reduction, logical-K bounds, and the scalar N-tail;
- bit-exact or contract-equivalent comparison with the scalar oracle across the
  complete fixture corpus before any performance result is reported.

## Consequences

- The first SIMD result is small enough to review: four output rows, one K
  recurrence, one pack layout, and one target.
- It intentionally leaves K-unrolling, multiple accumulators, prefetching,
  alternate packing, and runtime schedule selection for G2.
- A vector speedup is an empirical question. If the fixed schedule loses to
  generated scalar, the report must preserve that negative result rather than
  silently changing semantics or claiming a win.

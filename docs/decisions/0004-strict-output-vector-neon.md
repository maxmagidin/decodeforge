# ADR 0004: Freeze the strict output-vector NEON lowering

- **Status:** Accepted and implemented for the fixed G1 ARM64 correctness checkpoint
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
3. load one packed 32-bit Q word, express signed `int8 -> int16 -> int32`
   widening in generated C intrinsics, convert to four FP32 values, and
   broadcast the scalar activation; final instruction selection remains
   Clang's responsibility;
4. use explicit `vmulq_f32` followed by `vaddq_f32` for each product and
   accumulation, then load the four scale words as one raw 16-byte vector,
   bit-reinterpret them as FP32, and scale the four block sums;
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

Direct Q-word materialization makes the selected Apple toolchain produce a
register-connected `sshll.8h -> sshll.4s -> scvtf.4s` sequence. Loading the
scale bits directly removes the temporary stack array and its stack-canary
failure path without disabling stack protection. The audit checks the selected
instructions and requires the literal signed
`sshll.8h -> sshll.4s -> scvtf.4s` chain; an unsigned or differently collapsed
sign-extension sequence is outside this fixed checkpoint.

This is deliberately a structural machine-code audit rather than a second
general AArch64 verifier. Packed-address provenance comes from the exact
verified source-to-private-snapshot build path. The audit independently checks
control-flow dominance and uninterrupted SIMD value flow from the raw vector
load to the scale multiply.

The generated public boundary is exactly the scalar G1 ABI:
`df_abi_version`, `df_artifact_id`, and `df_run_v1`. The guard order, floating
point environment handling, status values, and private-output responsibility
remain those frozen by [ADR 0003](0003-scalar-generated-abi.md). This ADR
originally froze source lowering. Native compilation, shape-aware disassembly
audit, checked loading, and exact runtime execution are now implemented for
this fixed schedule; schedule selection and performance evidence remain
follow-on work.

## Implemented correctness evidence

- deterministic source, identity, shape/tail, and domain-separation tests;
- strict Apple-arm64 C compilation and retained Mach-O/disassembly audit;
- register-connected signed widening, vector conversion, separate
  multiply/add, activation broadcast, store, logical-K loop, and required
  scalar N-tail checks;
- bit-exact native scalar/NEON execution across all 16 fixtures, plus explicit
  `N=4` and `N=5` vector/tail cases;
- no performance claim until the prepared-call benchmark gate produces
  accepted evidence.

## Consequences

- The first SIMD result is small enough to review: four output rows, one K
  recurrence, one pack layout, and one target.
- It intentionally leaves K-unrolling, multiple accumulators, prefetching,
  alternate packing, and runtime schedule selection for G2.
- A vector speedup is an empirical question. If the fixed schedule loses to
  generated scalar, the report must preserve that negative result rather than
  silently changing semantics or claiming a win.

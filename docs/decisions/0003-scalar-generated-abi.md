# ADR 0003: Freeze the scalar generated-module ABI

- **Status:** Accepted for the G1 scalar contract
- **Date:** 2026-08-31
- **Decision owner:** Repository owner

## Context

G1 has a verified Region/Loop contract and deterministic
`DFQ8_B32_OI4_V1` packing, but generated code needs a small, stable boundary
before emitter and runtime work begins. The boundary must make failed calls
safe and diagnosable without adding a runtime API or making a performance
claim.

## Decision

The generated module includes
[`include/decodeforge/abi_v1.h`](../../include/decodeforge/abi_v1.h). It keeps
the existing 48-byte `df_call_v1` layout and exactly these public functions:

```c
uint32_t df_abi_version(void);
const char *df_artifact_id(void);
int32_t df_run_v1(
    const df_call_v1 *call,
    const float *x,
    const uint8_t *packed_weight,
    float *y);
```

`df_artifact_id` returns a C string occupying exactly 72 bytes including its
terminating NUL (`DF_ARTIFACT_ID_CSTR_BYTES_V1`). It identifies the generated
code contract (operator, schedule, numeric mode, ABI, and source format), not a
particular weight payload. The safe runtime separately binds the exact packed
weight identity before crossing the raw ABI. The status values returned by
`df_run_v1` are frozen as follows:

| Value | Macro | Meaning |
|---:|---|---|
| 0 | `DF_STATUS_OK_V1` | call completed |
| 1 | `DF_STATUS_NULL_ARGUMENT_V1` | a required pointer is null |
| 2 | `DF_STATUS_ABI_VERSION_V1` | `abi_version` is not the generated ABI version |
| 3 | `DF_STATUS_STRUCT_SIZE_V1` | `struct_size` is not the 48-byte descriptor size |
| 4 | `DF_STATUS_FLAGS_V1` | flags are not zero |
| 5 | `DF_STATUS_RESERVED_V1` | reserved bits/fields are not zero |
| 6 | `DF_STATUS_SHAPE_V1` | the call shape is not the compiled shape |
| 7 | `DF_STATUS_STRIDE_V1` | a stride is not the required stride |
| 8 | `DF_STATUS_PACKED_WEIGHT_BYTES_V1` | packed byte count is not expected |
| 9 | `DF_STATUS_PACKED_WEIGHT_ALIGNMENT_V1` | packed data is not 16-byte aligned |
| 10 | `DF_STATUS_FP_ENVIRONMENT_V1` | required FP environment cannot be established, verified, or preserved |
| 11 | `DF_STATUS_NONFINITE_INPUT_V1` | a logical input element is not finite |
| 12 | `DF_STATUS_NONFINITE_RESULT_V1` | the computed logical result is not finite |

The first scalar specialization supports only `M=1`, with `flags=0` and
`reserved0=0`. Strides are measured in `float` elements, not bytes, and must
be exactly `x_stride=K` and `y_stride=N` for the compiled shape. Let
`P=ceil(N/4)` and `B=ceil(K/32)`; the headerless OI4 payload must contain
exactly `P*B*144` bytes. `packed_weight` must point to storage aligned to 16
bytes. The ABI does not validate the extent of any buffer, pointer aliasing,
or that packed bytes identify the artifact. A safe Rust caller must establish
those properties before calling the generated module.

Each implementation evaluates guards in this deterministic order:

1. null `call`;
2. ABI version;
3. descriptor size;
4. flags;
5. reserved field;
6. compiled shape;
7. strides;
8. packed byte count;
9. data pointers;
10. packed-weight alignment;
11. successful floating-point-environment hold, round-to-nearest-even, and
    gradual-subnormal input/output canaries;
12. finiteness of the logical `x` elements; and
13. the scalar helper.

Every failure through the nonfinite-input check leaves `y` untouched. Once the
helper starts, it may write earlier logical outputs before detecting a
nonfinite result. In that case it returns `DF_STATUS_NONFINITE_RESULT_V1`; the
safe runtime discards its private output allocation and exposes no partial
result. More generally, that runtime discards the whole private output on any
nonzero status. A direct raw-ABI caller must treat `y` as unspecified after any
post-helper failure.

Before observing or performing floating-point work, the generated entrypoint
uses `feholdexcept` to save the caller environment, clear exception flags, and
install non-stop exception handling. Every path after a successful hold passes
through one `fesetenv` cleanup, including unsupported rounding, failed
subnormal canaries, nonfinite input, and nonfinite output. Thus the call
preserves the caller's rounding mode, exception flags, and trap configuration;
a hold or restore failure reports `DF_STATUS_FP_ENVIRONMENT_V1`.

The gradual-underflow canaries compare the object-representation bits of their
computed subnormal results. A floating-point comparison is insufficient: when
flush-to-zero or flush-inputs-to-zero is active, ARM can treat both a flushed
result and the expected subnormal comparison operand as zero. Integer bit
checks make either mode an observable `DF_STATUS_FP_ENVIRONMENT_V1` failure.

This ABI freezes correctness and failure semantics only. It is not evidence of
native code generation, throughput, latency, or any other performance result.

## Consequences

- Scalar generated modules and their future runtime bridge share one checked,
  versioned declaration and one status vocabulary.
- Header smoke tests compile the contract as C11 and C++17 and assert every
  frozen value, declaration, field offset, and size without invoking generated
  code.
- Emitter, runtime, toolchain, aliasing/extent validation, and weight-identity
  policy remain separate implementation work.

#ifndef DECODEFORGE_ABI_V1_H
#define DECODEFORGE_ABI_V1_H

#include <stddef.h>
#include <stdint.h>

#define DF_GENERATED_ABI_VERSION_V1 UINT32_C(1)

/*
 * Frozen df_run_v1 results.  These values are part of the generated-module
 * ABI; keep them representable as int32_t, the return type of df_run_v1.
 */
#define DF_STATUS_OK_V1 INT32_C(0)
#define DF_STATUS_NULL_ARGUMENT_V1 INT32_C(1)
#define DF_STATUS_ABI_VERSION_V1 INT32_C(2)
#define DF_STATUS_STRUCT_SIZE_V1 INT32_C(3)
#define DF_STATUS_FLAGS_V1 INT32_C(4)
#define DF_STATUS_RESERVED_V1 INT32_C(5)
#define DF_STATUS_SHAPE_V1 INT32_C(6)
#define DF_STATUS_STRIDE_V1 INT32_C(7)
#define DF_STATUS_PACKED_WEIGHT_BYTES_V1 INT32_C(8)
#define DF_STATUS_PACKED_WEIGHT_ALIGNMENT_V1 INT32_C(9)
#define DF_STATUS_FP_ENVIRONMENT_V1 INT32_C(10)
#define DF_STATUS_NONFINITE_INPUT_V1 INT32_C(11)
#define DF_STATUS_NONFINITE_RESULT_V1 INT32_C(12)

/* The artifact ID C string length includes its terminating NUL byte. */
#define DF_ARTIFACT_ID_CSTR_BYTES_V1 UINT32_C(72)

/*
 * The fixed generated-module call ABI.  Keep this definition in one header:
 * generated C, the future runtime bridge, and layout smoke tests all consume
 * it directly.  Pointers are intentionally passed separately from this
 * fixed-size descriptor so its layout is stable on supported 64-bit hosts.
 */
typedef struct df_call_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t flags;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t x_stride;
    uint32_t y_stride;
    uint32_t reserved0;
    uint64_t packed_weight_bytes;
} df_call_v1;

/*
 * Scalar G1 contract (the full guard and failure semantics are in ADR 0003):
 * strides count float elements; flags and reserved0 are zero; m is one;
 * x_stride is k and y_stride is n.  For P=ceil(n/4), B=ceil(k/32), the
 * headerless DFQ8_B32_OI4_V1 payload is P*B*144 bytes and its pointer is
 * required to be 16-byte aligned.  The implementation checks guards in a
 * fixed order and leaves y untouched through NONFINITE_INPUT; a caller or
 * runtime must discard the whole output if any later check fails.
 */

#if defined(__cplusplus)
static_assert(sizeof(df_call_v1) == 48, "df_call_v1 must be 48 bytes");
static_assert(offsetof(df_call_v1, abi_version) == 0, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, struct_size) == 4, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, flags) == 8, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, m) == 16, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, n) == 20, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, k) == 24, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, x_stride) == 28, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, y_stride) == 32, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, reserved0) == 36, "df_call_v1 ABI layout");
static_assert(offsetof(df_call_v1, packed_weight_bytes) == 40, "df_call_v1 ABI layout");
#else
_Static_assert(sizeof(df_call_v1) == 48, "df_call_v1 must be 48 bytes");
_Static_assert(offsetof(df_call_v1, abi_version) == 0, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, struct_size) == 4, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, flags) == 8, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, m) == 16, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, n) == 20, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, k) == 24, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, x_stride) == 28, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, y_stride) == 32, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, reserved0) == 36, "df_call_v1 ABI layout");
_Static_assert(offsetof(df_call_v1, packed_weight_bytes) == 40, "df_call_v1 ABI layout");
#endif

#if defined(__cplusplus)
extern "C" {
#endif

uint32_t df_abi_version(void);
const char *df_artifact_id(void);
int32_t df_run_v1(
    const df_call_v1 *call,
    const float *x,
    const uint8_t *packed_weight,
    float *y);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* DECODEFORGE_ABI_V1_H */

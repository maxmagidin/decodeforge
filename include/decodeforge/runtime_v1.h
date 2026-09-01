#ifndef DECODEFORGE_RUNTIME_V1_H
#define DECODEFORGE_RUNTIME_V1_H

#include <stddef.h>
#include <stdint.h>

/*
 * This bridge ABI is deliberately separate from the generated-module ABI in
 * abi_v1.h; include that header separately when using df_run_v1. A bridge
 * handle owns one verified generated NEON module and its exact aligned packed
 * payload. Handles are process-local and must never be guessed, persisted, or
 * reused by a foreign caller.
 */
#define DF_RUNTIME_ABI_VERSION_V1 UINT32_C(1)
#define DF_RUNTIME_IDENTITY_CSTR_BYTES_V1 UINT32_C(72)
#define DF_RUNTIME_MAX_MANIFEST_BYTES_V1 (UINT64_C(16) * UINT64_C(1024))
/* One manifest/payload pair may contain at most 128 MiB of packed bytes. */
#define DF_RUNTIME_MAX_PACKED_WEIGHT_BYTES_V1 \
    (UINT64_C(128) * UINT64_C(1024) * UINT64_C(1024))
/* All live handles together may retain at most 2 GiB of packed bytes. */
#define DF_RUNTIME_MAX_AGGREGATE_PACKED_WEIGHT_BYTES_V1 \
    (UINT64_C(2) * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024))
#define DF_RUNTIME_MAX_LIVE_ENTRIES_V1 UINT32_C(256)
#define DF_RUNTIME_MAX_ERROR_BYTES_V1 UINT32_C(4096)

typedef uint64_t df_runtime_handle_v1;

/* Closed bridge status values.  These are not generated df_run_v1 statuses. */
#define DF_RUNTIME_STATUS_OK_V1 INT32_C(0)
#define DF_RUNTIME_STATUS_TRUNCATED_V1 INT32_C(1)
#define DF_RUNTIME_STATUS_NULL_ARGUMENT_V1 INT32_C(2)
#define DF_RUNTIME_STATUS_ZERO_LENGTH_V1 INT32_C(3)
#define DF_RUNTIME_STATUS_INVALID_HANDLE_V1 INT32_C(4)
#define DF_RUNTIME_STATUS_INVALID_ARGUMENT_V1 INT32_C(5)
#define DF_RUNTIME_STATUS_OVERLAP_V1 INT32_C(6)
#define DF_RUNTIME_STATUS_LIMIT_VIOLATION_V1 INT32_C(7)
#define DF_RUNTIME_STATUS_INVALID_MANIFEST_V1 INT32_C(8)
#define DF_RUNTIME_STATUS_INVALID_PAYLOAD_V1 INT32_C(9)
#define DF_RUNTIME_STATUS_UNSUPPORTED_HOST_V1 INT32_C(10)
#define DF_RUNTIME_STATUS_BUILD_FAILED_V1 INT32_C(11)
#define DF_RUNTIME_STATUS_LOAD_FAILED_V1 INT32_C(12)
#define DF_RUNTIME_STATUS_EXECUTION_FAILED_V1 INT32_C(13)
#define DF_RUNTIME_STATUS_NONFINITE_INPUT_V1 INT32_C(14)
#define DF_RUNTIME_STATUS_NONFINITE_OUTPUT_V1 INT32_C(15)
#define DF_RUNTIME_STATUS_PANIC_V1 INT32_C(16)
#define DF_RUNTIME_STATUS_ALLOCATION_FAILED_V1 INT32_C(17)
#define DF_RUNTIME_STATUS_INTERNAL_V1 INT32_C(18)

/*
 * Fixed-size query result.  module_id and packed_weight_id are NUL-terminated
 * sha256:<64 lowercase hex> identities.  The bridge writes an all-zero
 * descriptor on a failed query whenever the caller supplied a valid pointer.
 * Descriptor fields are usable only when the return value is OK.
 */
typedef struct df_runtime_descriptor_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t n;
    uint32_t k;
    uint64_t packed_weight_bytes;
    char module_id[DF_RUNTIME_IDENTITY_CSTR_BYTES_V1];
    char packed_weight_id[DF_RUNTIME_IDENTITY_CSTR_BYTES_V1];
} df_runtime_descriptor_v1;

#if defined(__cplusplus)
static_assert(
    sizeof(df_runtime_descriptor_v1) == 168,
    "df_runtime_descriptor_v1 must be 168 bytes");
static_assert(
    offsetof(df_runtime_descriptor_v1, abi_version) == 0,
    "df_runtime_descriptor_v1 ABI layout");
static_assert(
    offsetof(df_runtime_descriptor_v1, struct_size) == 4,
    "df_runtime_descriptor_v1 ABI layout");
static_assert(
    offsetof(df_runtime_descriptor_v1, n) == 8,
    "df_runtime_descriptor_v1 ABI layout");
static_assert(
    offsetof(df_runtime_descriptor_v1, k) == 12,
    "df_runtime_descriptor_v1 ABI layout");
static_assert(
    offsetof(df_runtime_descriptor_v1, packed_weight_bytes) == 16,
    "df_runtime_descriptor_v1 ABI layout");
static_assert(
    offsetof(df_runtime_descriptor_v1, module_id) == 24,
    "df_runtime_descriptor_v1 ABI layout");
static_assert(
    offsetof(df_runtime_descriptor_v1, packed_weight_id) == 96,
    "df_runtime_descriptor_v1 ABI layout");
#else
_Static_assert(
    sizeof(df_runtime_descriptor_v1) == 168,
    "df_runtime_descriptor_v1 must be 168 bytes");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, abi_version) == 0,
    "df_runtime_descriptor_v1 ABI layout");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, struct_size) == 4,
    "df_runtime_descriptor_v1 ABI layout");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, n) == 8,
    "df_runtime_descriptor_v1 ABI layout");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, k) == 12,
    "df_runtime_descriptor_v1 ABI layout");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, packed_weight_bytes) == 16,
    "df_runtime_descriptor_v1 ABI layout");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, module_id) == 24,
    "df_runtime_descriptor_v1 ABI layout");
_Static_assert(
    offsetof(df_runtime_descriptor_v1, packed_weight_id) == 96,
    "df_runtime_descriptor_v1 ABI layout");
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* Return the bridge ABI version, without creating a handle. */
uint32_t df_runtime_bridge_abi_version_v1(void);

/*
 * Parse canonical PackManifestV1 JSON, copy and verify the exact OI4 payload,
 * build/audit/load the fixed NEON module, and return one nonzero handle.
 * Caller preconditions: each input pointer is readable for its exact byte
 * count, out_handle is writable for one handle, and all ranges are disjoint.
 * Inputs and output are borrowed only for the duration of this call. At most
 * one create/build is admitted at a time; failed creates do not consume quota.
 */
int32_t df_runtime_create_neon_v1(
    const uint8_t *pack_manifest_json,
    size_t pack_manifest_json_bytes,
    const uint8_t *packed_weight,
    size_t packed_weight_bytes,
    df_runtime_handle_v1 *out_handle);

/*
 * Execute one exact M=1 call. Caller preconditions: input/output are aligned,
 * readable/writable for their exact element counts, and disjoint. Invalid
 * handles, lengths, and other structural failures leave output untouched;
 * output is usable only when the return value is OK. Concurrent runs are
 * supported.
 */
int32_t df_runtime_run_v1(
    df_runtime_handle_v1 handle,
    const float *input,
    size_t input_len,
    float *output,
    size_t output_len);

/*
 * Query the fixed shape and content identities owned by a handle. The output
 * pointer must be aligned and writable for one descriptor. A valid pointer is
 * zeroed on every failure, including INVALID_HANDLE; fields are usable only
 * on OK.
 */
int32_t df_runtime_get_descriptor_v1(
    df_runtime_handle_v1 handle,
    df_runtime_descriptor_v1 *out_descriptor);

/* Destroy one handle.  A second destroy returns INVALID_HANDLE. */
int32_t df_runtime_destroy_v1(df_runtime_handle_v1 handle);

/*
 * Copy the calling thread's last bridge error as bounded printable ASCII,
 * NUL-terminated string.
 * required_bytes includes the terminating NUL.  A NULL buffer with zero
 * capacity is a length query; a short buffer returns TRUNCATED and is always
 * NUL-terminated when its capacity is nonzero. Caller preconditions:
 * required_bytes is writable for one size_t; a nonzero buffer is writable for
 * its exact capacity and disjoint from required_bytes.
 */
int32_t df_runtime_last_error_v1(
    char *buffer,
    size_t buffer_bytes,
    size_t *required_bytes);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* DECODEFORGE_RUNTIME_V1_H */

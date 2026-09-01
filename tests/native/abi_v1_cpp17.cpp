#include <cstddef>
#include <type_traits>

#include "decodeforge/abi_v1.h"
#include "decodeforge/runtime_v1.h"

static_assert(DF_GENERATED_ABI_VERSION_V1 == 1, "generated ABI version drift");
static_assert(DF_RUNTIME_ABI_VERSION_V1 == 1, "runtime ABI version drift");
static_assert(DF_RUNTIME_STATUS_OK_V1 == 0, "bridge status drift");
static_assert(DF_RUNTIME_STATUS_PANIC_V1 == 16, "bridge status drift");
static_assert(
    DF_RUNTIME_MAX_PACKED_WEIGHT_BYTES_V1 ==
        UINT64_C(128) * UINT64_C(1024) * UINT64_C(1024),
    "bridge per-pack byte bound drift");
static_assert(
    DF_RUNTIME_MAX_AGGREGATE_PACKED_WEIGHT_BYTES_V1 ==
        UINT64_C(2) * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024),
    "bridge aggregate byte bound drift");
static_assert(
    DF_RUNTIME_IDENTITY_CSTR_BYTES_V1 == 72,
    "bridge identity size drift");
static_assert(
    sizeof(df_runtime_descriptor_v1) == 168,
    "bridge descriptor size drift");
static_assert(DF_STATUS_OK_V1 == 0, "status drift");
static_assert(DF_STATUS_NULL_ARGUMENT_V1 == 1, "status drift");
static_assert(DF_STATUS_ABI_VERSION_V1 == 2, "status drift");
static_assert(DF_STATUS_STRUCT_SIZE_V1 == 3, "status drift");
static_assert(DF_STATUS_FLAGS_V1 == 4, "status drift");
static_assert(DF_STATUS_RESERVED_V1 == 5, "status drift");
static_assert(DF_STATUS_SHAPE_V1 == 6, "status drift");
static_assert(DF_STATUS_STRIDE_V1 == 7, "status drift");
static_assert(DF_STATUS_PACKED_WEIGHT_BYTES_V1 == 8, "status drift");
static_assert(DF_STATUS_PACKED_WEIGHT_ALIGNMENT_V1 == 9, "status drift");
static_assert(DF_STATUS_FP_ENVIRONMENT_V1 == 10, "status drift");
static_assert(DF_STATUS_NONFINITE_INPUT_V1 == 11, "status drift");
static_assert(DF_STATUS_NONFINITE_RESULT_V1 == 12, "status drift");
static_assert(DF_ARTIFACT_ID_CSTR_BYTES_V1 == 72, "artifact ID length drift");

static_assert(sizeof(df_call_v1) == 48, "df_call_v1 size drift");
static_assert(offsetof(df_call_v1, abi_version) == 0, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, struct_size) == 4, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, flags) == 8, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, m) == 16, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, n) == 20, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, k) == 24, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, x_stride) == 28, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, y_stride) == 32, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, reserved0) == 36, "df_call_v1 offset drift");
static_assert(offsetof(df_call_v1, packed_weight_bytes) == 40, "df_call_v1 offset drift");

using df_abi_version_signature_v1 = uint32_t (*)(void);
using df_artifact_id_signature_v1 = const char *(*)(void);
using df_run_signature_v1 = int32_t (*)(
    const df_call_v1 *, const float *, const uint8_t *, float *);
using df_bridge_version_signature_v1 = uint32_t (*)();
using df_bridge_create_signature_v1 = int32_t (*)(
    const uint8_t *, size_t, const uint8_t *, size_t, df_runtime_handle_v1 *);
using df_bridge_run_signature_v1 = int32_t (*)(
    df_runtime_handle_v1, const float *, size_t, float *, size_t);
using df_bridge_descriptor_signature_v1 = int32_t (*)(
    df_runtime_handle_v1, df_runtime_descriptor_v1 *);
using df_bridge_destroy_signature_v1 = int32_t (*)(df_runtime_handle_v1);
using df_bridge_error_signature_v1 = int32_t (*)(char *, size_t, size_t *);
static_assert(
    std::is_same<decltype(&df_abi_version), df_abi_version_signature_v1>::value,
    "df_abi_version declaration drift");
static_assert(
    std::is_same<decltype(&df_artifact_id), df_artifact_id_signature_v1>::value,
    "df_artifact_id declaration drift");
static_assert(
    std::is_same<decltype(&df_run_v1), df_run_signature_v1>::value,
    "df_run_v1 declaration drift");
static_assert(
    std::is_same<
        decltype(&df_runtime_bridge_abi_version_v1),
        df_bridge_version_signature_v1>::value,
    "bridge version declaration drift");
static_assert(
    std::is_same<
        decltype(&df_runtime_create_neon_v1),
        df_bridge_create_signature_v1>::value,
    "bridge create declaration drift");
static_assert(
    std::is_same<decltype(&df_runtime_run_v1), df_bridge_run_signature_v1>::value,
    "bridge run declaration drift");
static_assert(
    std::is_same<
        decltype(&df_runtime_get_descriptor_v1),
        df_bridge_descriptor_signature_v1>::value,
    "bridge descriptor declaration drift");
static_assert(
    std::is_same<
        decltype(&df_runtime_destroy_v1),
        df_bridge_destroy_signature_v1>::value,
    "bridge destroy declaration drift");
static_assert(
    std::is_same<
        decltype(&df_runtime_last_error_v1),
        df_bridge_error_signature_v1>::value,
    "bridge error declaration drift");

int main() {
    df_call_v1 call{};
    auto version_fn = &df_abi_version;
    auto artifact_fn = &df_artifact_id;
    auto run_fn = &df_run_v1;
    auto bridge_version_fn = &df_runtime_bridge_abi_version_v1;
    auto bridge_create_fn = &df_runtime_create_neon_v1;
    auto bridge_run_fn = &df_runtime_run_v1;
    auto bridge_descriptor_fn = &df_runtime_get_descriptor_v1;
    auto bridge_destroy_fn = &df_runtime_destroy_v1;
    auto bridge_error_fn = &df_runtime_last_error_v1;

    (void)call;
    (void)version_fn;
    (void)artifact_fn;
    (void)run_fn;
    (void)bridge_version_fn;
    (void)bridge_create_fn;
    (void)bridge_run_fn;
    (void)bridge_descriptor_fn;
    (void)bridge_destroy_fn;
    (void)bridge_error_fn;
    return 0;
}

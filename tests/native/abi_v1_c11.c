#include <stddef.h>

#include "decodeforge/abi_v1.h"
#include "decodeforge/runtime_v1.h"

_Static_assert(DF_GENERATED_ABI_VERSION_V1 == 1, "generated ABI version drift");
_Static_assert(DF_RUNTIME_ABI_VERSION_V1 == 1, "runtime ABI version drift");
_Static_assert(DF_STATUS_OK_V1 == 0, "status drift");
_Static_assert(DF_STATUS_NULL_ARGUMENT_V1 == 1, "status drift");
_Static_assert(DF_STATUS_ABI_VERSION_V1 == 2, "status drift");
_Static_assert(DF_STATUS_STRUCT_SIZE_V1 == 3, "status drift");
_Static_assert(DF_STATUS_FLAGS_V1 == 4, "status drift");
_Static_assert(DF_STATUS_RESERVED_V1 == 5, "status drift");
_Static_assert(DF_STATUS_SHAPE_V1 == 6, "status drift");
_Static_assert(DF_STATUS_STRIDE_V1 == 7, "status drift");
_Static_assert(DF_STATUS_PACKED_WEIGHT_BYTES_V1 == 8, "status drift");
_Static_assert(DF_STATUS_PACKED_WEIGHT_ALIGNMENT_V1 == 9, "status drift");
_Static_assert(DF_STATUS_FP_ENVIRONMENT_V1 == 10, "status drift");
_Static_assert(DF_STATUS_NONFINITE_INPUT_V1 == 11, "status drift");
_Static_assert(DF_STATUS_NONFINITE_RESULT_V1 == 12, "status drift");
_Static_assert(DF_ARTIFACT_ID_CSTR_BYTES_V1 == 72, "artifact ID length drift");

_Static_assert(sizeof(df_call_v1) == 48, "df_call_v1 size drift");
_Static_assert(offsetof(df_call_v1, abi_version) == 0, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, struct_size) == 4, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, flags) == 8, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, m) == 16, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, n) == 20, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, k) == 24, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, x_stride) == 28, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, y_stride) == 32, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, reserved0) == 36, "df_call_v1 offset drift");
_Static_assert(offsetof(df_call_v1, packed_weight_bytes) == 40, "df_call_v1 offset drift");

typedef uint32_t (*df_abi_version_signature_v1)(void);
typedef const char *(*df_artifact_id_signature_v1)(void);
typedef int32_t (*df_run_signature_v1)(
    const df_call_v1 *, const float *, const uint8_t *, float *);
_Static_assert(
    _Generic(&df_abi_version, df_abi_version_signature_v1: 1, default: 0),
    "df_abi_version declaration drift");
_Static_assert(
    _Generic(&df_artifact_id, df_artifact_id_signature_v1: 1, default: 0),
    "df_artifact_id declaration drift");
_Static_assert(
    _Generic(&df_run_v1, df_run_signature_v1: 1, default: 0),
    "df_run_v1 declaration drift");

int main(void) {
    df_call_v1 call = {0};
    df_runtime_v1 *runtime = NULL;
    df_abi_version_signature_v1 version_fn = &df_abi_version;
    df_artifact_id_signature_v1 artifact_fn = &df_artifact_id;
    df_run_signature_v1 run_fn = &df_run_v1;

    (void)call;
    (void)runtime;
    (void)version_fn;
    (void)artifact_fn;
    (void)run_fn;
    return 0;
}

#include <stddef.h>

#include "decodeforge/abi_v1.h"
#include "decodeforge/runtime_v1.h"

int main(void) {
    df_call_v1 call = {0};
    df_runtime_v1 *runtime = NULL;
    uint32_t (*version_fn)(void) = &df_abi_version;
    const char *(*artifact_fn)(void) = &df_artifact_id;
    int32_t (*run_fn)(const df_call_v1 *, const float *, const uint8_t *, float *) = &df_run_v1;

    (void)call;
    (void)runtime;
    (void)version_fn;
    (void)artifact_fn;
    (void)run_fn;
    return DF_GENERATED_ABI_VERSION_V1 == DF_RUNTIME_ABI_VERSION_V1 ? 0 : 1;
}

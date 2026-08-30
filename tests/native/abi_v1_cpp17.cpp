#include <cstddef>

#include "decodeforge/abi_v1.h"
#include "decodeforge/runtime_v1.h"

int main() {
    df_call_v1 call{};
    df_runtime_v1 *runtime = nullptr;
    auto version_fn = &df_abi_version;
    auto artifact_fn = &df_artifact_id;
    auto run_fn = &df_run_v1;

    (void)call;
    (void)runtime;
    (void)version_fn;
    (void)artifact_fn;
    (void)run_fn;
    return DF_GENERATED_ABI_VERSION_V1 == DF_RUNTIME_ABI_VERSION_V1 ? 0 : 1;
}

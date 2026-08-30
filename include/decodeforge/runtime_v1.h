#ifndef DECODEFORGE_RUNTIME_V1_H
#define DECODEFORGE_RUNTIME_V1_H

#include <stdint.h>

/*
 * The foundation reserves the runtime ABI version and opaque handle name only.
 * Runtime operations are intentionally not declared until their semantics are
 * implemented and tested.
 */
#define DF_RUNTIME_ABI_VERSION_V1 UINT32_C(1)

typedef struct df_runtime_v1 df_runtime_v1;

#endif /* DECODEFORGE_RUNTIME_V1_H */

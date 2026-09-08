# DecodeForge Rust workspace

The Rust side owns exact Q8 semantics, compilation, native artifact validation,
runtime safety, and the C boundary used by Python/PyTorch.

```text
decodeforge-core
      │ exact Q8 semantics and identities
      ▼
decodeforge-compiler ───► decodeforge-runtime
      │ generated artifacts       │ validated ownership/execution
      └──────────────┬────────────┘
                     ▼
             decodeforge-bridge
                     │ versioned C ABI
                     ▼
              Python / PyTorch
```

| Crate | Responsibility |
| --- | --- |
| [`decodeforge-core`](decodeforge-core) | `DFQ8_B32_V1` quantization, canonical evaluation, fixture parsing, and identities |
| [`decodeforge-compiler`](decodeforge-compiler) | Region/Loop IR, OI4 packing, scalar/NEON generation, toolchain control, Mach-O audit, and model-asset preparation |
| [`decodeforge-runtime`](decodeforge-runtime) | Generated-module ABI validation, loading, scalar/native execution, and ownership |
| [`decodeforge-bridge`](decodeforge-bridge) | Hardened process-local C ABI with opaque handles, limits, lifecycle linearization, and diagnostics |
| [`decodeforge-cli`](decodeforge-cli) | Version reporting and Q8 fixture verification |

The compiler currently generates strict scalar and Apple ARM64 NEON code for
the completed path. An AVX2 target is deferred work, not an implemented backend.

Run Rust checks through the repository commands so toolchain, native, Python,
schema, and retained-evidence expectations stay aligned:

```sh
make setup
make check
```

See the [technical design](../docs/DESIGN.md) and
[Q8 contract](../docs/Q8_FORMAT_V1.md) before changing public formats, ABI
versions, numerical behavior, or result identities.

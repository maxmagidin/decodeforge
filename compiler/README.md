# DecodeForge Rust workspace

The Rust workspace turns a typed Q8 projection into generated native code,
checks the result before loading it, and exposes a small C boundary to
Python/PyTorch.

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
| [`decodeforge-core`](decodeforge-core) | Defines Q8 quantization, reference evaluation, fixtures, and content identities |
| [`decodeforge-compiler`](decodeforge-compiler) | Lowers typed IR, packs OI4 weights, generates scalar/NEON C, and validates the compiled artifact |
| [`decodeforge-runtime`](decodeforge-runtime) | Loads validated modules and manages scalar/native execution and ownership |
| [`decodeforge-bridge`](decodeforge-bridge) | Exposes a versioned, process-local C ABI with opaque handles, bounds, lifecycle rules, and diagnostics |
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

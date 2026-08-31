# DecodeForge V1 schemas

This directory contains seven semantic JSON Schema Draft 2020-12 contracts:

- `compiler-request.schema.json`;
- `quant-fixture.schema.json`;
- `fixture-manifest.schema.json`;
- `schedule.schema.json`;
- `diagnostic.schema.json`;
- `host-manifest.schema.json`;
- `run-manifest.schema.json`.

`common.schema.json` contains shared definitions and
`diagnostic-codes.json` is the append-only stable-code registry. Every `$ref`
is resolved from the checked-in URN catalog; validation never retrieves a
schema from the network. Parsers reject duplicate JSON keys before schema
validation.

`schema_version` is the integer major version and is exactly `1`. Semantic
objects reject unknown properties. Quant fixtures and their manifests are
closed vocabularies with no free-form metadata; their fixed format and numeric
mode are `DFQ8_B32_V1` and `strict_f32_v1`.

The schedule schema currently describes the fixed G1 Loop IR contract: explicit
scalar or NEON target variants with exact lane/target pairings, a fixed `N` tile
of four, one ascending logical `K` accumulator, separate rounded multiply/add
steps, output-interleaved OI4 packing with 16-byte alignment, logical-only `K`
padding, and scalar cleanup for an `N` tail. Generated kernels and performance
evidence are not part of this contract yet. Because JSON Schema cannot express
the product of panel and block counts, the offline semantic validator also
rejects any combined shape whose `P*B*144` payload size exceeds the frozen
64-bit representation. That validator also requires integer-valued Loop IR
fields to use JSON integer tokens, matching Rust's canonical serde boundary.

The `portable` scalar target is the architecture-neutral baseline for the
project's supported 64-bit little-endian hosts; it is not a 32-bit target.

Run the complete offline contract check with:

```sh
uv run --frozen python scripts/validate_schemas.py --all
```

Validate a foundation fixture bundle without executing any artifact with:

```sh
make verify-bundle BUNDLE=tests/fixtures/bundles/foundation-valid
```

The `foundation-empty` fixture is intentionally invalid and must emit three
ordered `DFE-BUNDLE-001` diagnostics. Later milestones extend bundle
requirements. The Q8 fixture schema records a physical 32-lane q block even
for a `K` tail; the Python semantic validator derives all array lengths from
`N`, `K`, and `blocks`. The exact identities and corpus manifest are defined
in `docs/Q8_FORMAT_V1.md`.

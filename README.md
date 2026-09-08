# DecodeForge

[![CI](https://github.com/maxmagidin/decodeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/maxmagidin/decodeforge/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024-orange.svg)](Cargo.toml)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.14-blue.svg)](pyproject.toml)

**A Rust compiler that turns quantized LLM linear operations into audited
ARM64 NEON kernels and executes them inside PyTorch.**

DecodeForge follows one performance-critical operation from numerical
semantics to generated machine code: a fixed-weight, single-token query
projection used during LLM decode. It lowers typed Q8 IR, packs weights into an
output-interleaved layout, emits deterministic scalar or NEON C, audits the
compiled artifact, and calls it through a guarded native bridge.

The completed Apple M4 path runs generated code across **all 22 TinyLlama query
projections during cached single-token decode**. The repository includes the
source, disassembly audits, raw samples, correctness results, model integration
evidence, and offline verification needed to check that claim.

> **Headline result:** generated NEON was approximately **3.96× faster** than
> generated scalar across three independent Apple M4 sessions at the same-Q8
> prepared-call boundary. This is a kernel result—not a 3.96× whole-model or
> stock-PyTorch speedup.

[Read the plain-language primer](docs/PRIMER.md) ·
[Inspect the results](results/README.md) ·
[Understand the design](docs/DESIGN.md)

## Results at a glance

| Completed proof | Result | Evidence |
| --- | ---: | --- |
| Generated NEON vs generated scalar | **3.956×–3.962×** across 3 independent sessions | [G1 kernel result](results/g1/apple-m4-primary/README.md) |
| Native vs same-Q8 model behavior | **30/30 prompts**, **1,070 decode steps**, exact token agreement | [Apple M4 evaluation](results/evaluation/apple-m4-v1/README.md) |
| Maximum native/reference logit difference | **0.0000171661** | [Correctness capture](results/evaluation/apple-m4-v1/correctness-v1.json) |
| Native execution coverage | **22/22** TinyLlama query projections during cached decode | [G3 evidence](results/g3/apple-m4-primary/README.md) |
| Q8 vs original FP32 sensitivity | **99.7099%** next-token argmax agreement on 1,034 fixed tokens | [Evaluation summary](results/evaluation/apple-m4-v1/summary.json) |
| Reproducible model benchmark | **81 measured generations + 27 warmups** across 3 fresh processes | [Evaluation protocol](docs/EVALUATION_V1.md) |

The broader model benchmark found that the hybrid native path was substantially
faster than the guarded same-Q8 reference, but **did not consistently beat
original FP32 PyTorch**. That negative boundary is retained because it separates
the compiler's verified kernel result from the performance of the surrounding
model and guard path.

![DecodeForge Apple M4 results overview: three generated-kernel confidence intervals, all-22 projection coverage, exact-token correctness, and model throughput ranges.](docs/assets/decodeforge-results-overview.svg)

## What DecodeForge builds

```text
fixed TinyLlama q_proj weights + static [N,K] + Apple M4 target
                              │
                              ▼
                    exact DFQ8_B32_V1 semantics
                              │
                              ▼
                     typed Region IR + Loop IR
                              │
                              ▼
                   output-interleaved OI4 packing
                              │
                              ▼
              deterministic scalar / ARM64 NEON C
                              │
                              ▼
               Clang build + Mach-O/disassembly audit
                              │
                              ▼
              versioned runtime ABI + guarded ownership
                              │
                              ▼
                   eager PyTorch q_proj adapters
                       ┌──────┴────────┐
                       ▼               ▼
             same-Q8 prefill     native M=1 decode
                       └──────┬────────┘
                              ▼
               correctness + timing + lifecycle evidence
```

PyTorch and Transformers still own model loading, tokenization, attention, KV
state, sampling, and every unsupported operation. DecodeForge replaces only the
eligible query-projection work it can guard and verify.

## Why this is a compiler

DecodeForge does more than wrap a handwritten kernel:

- **Typed IR:** Region and Loop IR keep operator semantics separate from the
  execution schedule, vector width, reduction order, tails, and packed offsets.
- **Deterministic lowering:** the same versioned request and weights produce
  byte-stable source and identity-bound artifacts.
- **Generated code:** the compiler emits strict scalar and ARM64 NEON C rather
  than calling an existing matrix-multiplication primitive.
- **Verified packing:** OI4 packing aligns memory with output-lane
  vectorization while preserving exact logical weight identities.
- **Native artifact audit:** compiled modules are checked for architecture,
  symbols, relocations, helpers, stack behavior, and expected instruction forms
  before loading.
- **Guarded execution:** versioned C ABIs validate shapes, buffer extents,
  features, artifact identities, ownership, and lifecycle state.
- **Evidence-first measurement:** correctness gates run before timing; raw
  observations, rejected states, specifications, and analyzers remain checked
  in.

Follow the implementation from [IR](compiler/decodeforge-compiler/src/ir.rs)
to [packing](compiler/decodeforge-compiler/src/pack.rs),
[NEON generation](compiler/decodeforge-compiler/src/codegen/neon_c.rs),
[artifact auditing](compiler/decodeforge-compiler/src/native/audit.rs), the
[native bridge](compiler/decodeforge-bridge/src/lib.rs), and the
[PyTorch model adapter](python/decodeforge/qproj_model.py).

## Measured Apple M4 behavior

### Generated kernel

G1 compares generated scalar and generated NEON code for the same
`M=1, N=2048, K=2048` Q8 projection through the same allocation-free
prepared-call boundary. Each of three processes ran 40 balanced scalar/NEON
pairs after warmup and calibration. A deterministic 10,000-resample paired BCa
bootstrap produced one 95% interval per session; every lower bound exceeded the
predeclared 1.0 gate.

| Session | Paired speedup | 95% paired-BCa interval |
| --- | ---: | ---: |
| 1 | 3.95671× | [3.95103, 3.96705] |
| 2 | 3.96176× | [3.95085, 3.96960] |
| 3 | 3.95648× | [3.95351, 3.95997] |

The timed boundary includes output-sentinel fill, the generated-module ABI
call, status decoding, and a finite-output scan. It excludes packing,
compilation, dynamic loading, and allocation.

### Integrated model

Decode throughput is reported in tokens/second. Each range spans the three
per-process medians, not a confidence interval or selected best runs.

![Apple M4 decode throughput: native execution is faster than the guarded same-Q8 reference but does not consistently beat FP32 PyTorch.](docs/assets/apple-m4-decode-throughput.svg)

| Case / output cap | Original FP32 PyTorch | Hybrid native | Guarded same-Q8 reference |
| --- | ---: | ---: | ---: |
| Short / 16 | 11.61–15.47 | 11.29–14.74 | 3.84–4.70 |
| Medium / 32 | 14.78–14.86 | 14.17–14.29 | 4.27–4.53 |
| Long / 64 | 12.23–13.74 | 10.85–13.23 | 4.37–4.47 |

Production guards remained enabled. Correctness-comparison hooks were excluded
from performance timing, but the guarded reference path still includes its own
weight clone/hash work, and outer finite-logit checks remain. Only query
projections use native decode; the rest of the model stays in PyTorch.

For the full numerical, lifecycle, setup, memory, and limitations record, read
the [Apple M4 evaluation](results/evaluation/apple-m4-v1/README.md).

## Reproduce the checked-in evidence

The fastest review path does not download a model or rerun an experiment. It
recomputes the retained analyses and verifies their identities:

```sh
make verify-g1-result verify-evaluation-result
```

Run the complete portable development suite with:

```sh
make setup
make check
```

`make check` covers Rust and Python formatting, linting, typing, unit and native
tests, Q8 fixture parity, actual bridge-library execution where supported,
schema validation, documentation, and retained-evidence verification. Shared
CI intentionally has no performance threshold.

A new model execution requires the pinned TinyLlama/tokenizer files, prepared
Q8 assets, and a verified native library; model weights are not committed.
Follow the [presentation guide](docs/PRESENTATION_DEMO.md) for an interactive
run or the [evaluation protocol](docs/EVALUATION_V1.md) for a new measurement.

## Repository guide

| Path | Responsibility |
| --- | --- |
| [`compiler/decodeforge-core`](compiler/decodeforge-core) | Exact Q8 semantics, identities, and fixture contracts |
| [`compiler/decodeforge-compiler`](compiler/decodeforge-compiler) | Typed IR, lowering, OI4 packing, source generation, toolchain and artifact audits |
| [`compiler/decodeforge-runtime`](compiler/decodeforge-runtime) | Generated-module validation, ownership, loading, and execution |
| [`compiler/decodeforge-bridge`](compiler/decodeforge-bridge) | Hardened process-local C ABI over verified modules and packs |
| [`python/decodeforge`](python/decodeforge) | Evidence tooling, eager PyTorch binding, query-projection adapters, evaluation |
| [`benchmarks`](benchmarks) | Frozen experiment specifications and capture inputs |
| [`results`](results) | Checked-in raw observations, summaries, generated source, and audits |
| [`schemas`](schemas) | Closed JSON contracts and stable diagnostics |
| [`docs`](docs) | Design, methods, decisions, project narrative, and reproduction guides |

Additional navigation is available in the
[compiler](compiler/README.md), [Python](python/README.md),
[benchmark](benchmarks/README.md), [results](results/README.md), and
[documentation](docs/README.md) indexes.

## Scope and limitations

The completed G0–G3 path is intentionally narrow:

- Apple M4 / ARM64 NEON;
- one thread;
- frozen, bias-free `DFQ8_B32_V1` weights;
- static `M=1, N=2048, K=2048` native query projections;
- all 22 TinyLlama `q_proj` modules during cached decode;
- eager PyTorch integration rather than a general FX or `torch.compile`
  backend.

The remaining 133 TinyLlama linear modules, native prefill, schedule search,
x86-64 AVX2, fusion, multicore execution, and general graph compilation are
not implemented claims. The current evidence comes from one physical M4 and a
synthetic 30-prompt corpus; it establishes implementation agreement and a
repeatable kernel result, not general model quality or cross-host performance.

These boundaries are deliberate. They make it possible to attribute each
result to a concrete layer of the system rather than hiding several unfinished
projects behind one benchmark number.

## Documentation

- [Reader's primer](docs/PRIMER.md) — the project, results, and methodology in
  plain language.
- [Documentation map](docs/README.md) — every design, protocol, decision, and
  review document by purpose.
- [Technical design](docs/DESIGN.md) — IR, packing, code generation, runtime,
  and model boundaries.
- [Q8 format](docs/Q8_FORMAT_V1.md) — normative numerical and serialization
  contract.
- [Benchmark methodology](docs/BENCHMARKS.md) — timing, statistical, and
  evidence policy.
- [Evaluation protocol](docs/EVALUATION_V1.md) — frozen broader model study.

## Status and next work

The original G0–G3 path is complete:

- **G0:** independent Python/Rust Q8 semantics and a provenance-bound corpus;
- **G1:** typed IR, OI4 packing, deterministic scalar/NEON generation, audited
  native artifacts, and the retained M4 kernel result;
- **G2:** a hardened native ABI and guarded eager PyTorch operator;
- **G3:** all-22 TinyLlama query-projection execution with retained model
  correctness, dispatch, lifecycle, and timing evidence.

The next optimization should be selected from profiling rather than feature
count: reduce guard/prefill overhead, broaden native linear coverage, add
schedule selection, reproduce on another physical Mac, or implement the
deferred AVX2 target.

DecodeForge's original code is licensed under [Apache 2.0](LICENSE).
Third-party dependencies and model artifacts retain their own licenses.

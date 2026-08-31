# DecodeForge

**A shape-specializing compiler for quantized LLM linear layers on commodity CPUs.**

DecodeForge's required path compiles the dominant operation in autoregressive
LLM decode—large matrix-vector products—into a guarded ARM64 NEON kernel on an
Apple M4. An explicit model pass quantizes frozen weights; a `torch.compile`
backend then recognizes supported Llama-family linear subgraphs, packs
constants, searches a bounded schedule space, builds a guarded native module,
and returns a callable that PyTorch can use in place of the captured graph.
An x86-64 AVX2 backend remains a deferred, optional portability extension after
the local Mac compiler path works end to end.

The project asks one question:

> Can one small schedule compiler produce a credible weight-only Q8 decode
> kernel on an Apple M4, and explain which tiling, packing, and fusion decisions
> work on that required local path?

The Ryzen/AVX2 design is retained as a possible later portability extension;
it is not a G0–G3 acceptance dependency. This owner-approved scope change is
recorded in [ADR 0001](docs/decisions/0001-mac-first-required-path.md).

This is deliberately not an inference server, work-stealing runtime, KV-cache
manager, general tensor framework, or GPU compiler. PyTorch/Transformers owns
model loading, tokenization, attention, KV state, and generation. DecodeForge
owns the compiler path for a narrow set of hot CPU operators.

## Why this scope

The original all-in-one engine concept packages several independent systems
questions. That makes a result hard to attribute and leaves too many components
half-finished. DecodeForge has one measurable contribution:

```text
FX graph + example shapes + constant weights + CPU target
                         |
                         v
                supported-region matcher
                         |
                         v
              typed contraction/fusion IR
                         |
       quantize + pack + enumerate legal schedules
                         |
            calibrate / benchmark / select schedule
                         |
                         v
                generated scalar / NEON native module
                (AVX2 deferred)
                         |
                         v
              guarded torch.compile callable
```

The required development host is:

- Apple M4 MacBook Air: ARM64 NEON, 10 physical cores.

The compiler/runtime contract targets 64-bit little-endian hosts. In schedule
records, `portable` means the scalar baseline across those supported hosts; it
does not claim 32-bit or big-endian portability.

The deferred portability host is:

- Ryzen 5 3600: x86-64 AVX2 hardware, 6 cores / 12 threads (the strict-f32
  compiler contract still requires separate multiply and add).
- Radeon RX 5700 XT: intentionally out of scope.

The initial weight-only format keeps activations in FP32, so its vector kernels
use widening/conversion plus FP32 arithmetic. Integer dot-product instructions
are not an MVP claim; using them would require a separately specified activation
quantization path.

## Initial compiler surface

Supported graph regions, in promotion order:

1. `linear(x, W)` with constant FP32/BF16 weights compiled to weight-only Q8;
2. `rms_norm(x) -> linear(...)`, avoiding a materialized normalized vector;
3. paired gate/up linears followed by `silu(gate) * up`;
4. small batches (`M = 1, 2, 4, 8`) after the decode `M = 1` path is stable.

Only item 1 belongs to the required vertical slice. Fusion, small batches, a
predictive cost model, and the HTML viewer remain locked until the scalar and
NEON kernels are correct and measured on real projection shapes on the M4.
AVX2 work is deferred until a later evidence-selected extension.

The first reference model is
[`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0),
an Apache-2.0 Llama-compatible 1.1B model with realistic decode projection
shapes. The compiler is shape-driven rather than hard-coded to that model, but
only those shapes are required for the first complete result.

## What makes it a compiler

- A typed IR represents contraction, quantization, scale, epilogue, layout, and
  reduction semantics independently of the M4 target; future targets reuse it.
- A legality layer rejects schedules that violate vector width, alignment,
  reduction, tail, or numeric-contract constraints.
- A schedule pass chooses loop order, row/output blocking, `K` unroll, vector
  width, prefetch distance, parallel split, and packed-weight layout.
- Code generation emits inspectable C/intrinsics plus a stable C ABI, then uses
  the host toolchain to build a loadable native module.
- Guards bind a kernel to the shapes, strides, dtype, alignment, quantization
  format, and CPU features for which it was compiled.
- The compiler cache is content-addressed by graph, weight, schedule, target,
  numeric mode, and compiler version.
- Every optimization is benchmarked against the same quantized scalar semantics,
  not only against a different precision or framework.

## Evidence, not architecture alone

The primary project artifact is a reproducible compiler run, not this design.
For each winning schedule, DecodeForge retains:

- canonical Region IR and Loop IR;
- generated C/intrinsics and the exact compiler invocation;
- disassembly with the hot loop identified, vector instructions checked, and
  stack spills or scalarized paths called out;
- the logical and physical weight-layout manifests;
- correctness results against the Q8 oracle;
- raw timing samples, host state, and available hardware-counter measurements;
- the selected and rejected schedules with reasons;
- compile time, tuning time, cache-hit latency, code size, and break-even calls.

Every performance claim must be reconstructible from a checked-in result bundle.
If a counter is unavailable on a host, the manifest records that fact instead of
substituting an estimate.

| Skill signal | Required proof |
|---|---|
| compiler construction | typed/verified IR, legal schedule enumeration, deterministic lowering |
| SIMD and machine code | retained M4 NEON source, disassembly audit, scalarization/spill checks; AVX2 only if selected as G4 extension |
| memory-system reasoning | packed-layout accounting, bandwidth calibration, cache counters when available |
| ABI and FFI safety | versioned C ABI, pointer/shape guards, negative tests, corrupt-artifact recovery |
| performance engineering | raw randomized samples, uncertainty, same-semantics baselines, break-even analysis |
| target judgment | one measured schedule tradeoff explained on M4; a cross-target comparison is optional G4 evidence |

## Delivery gates

| Gate | Required result | Scope unlocked |
|---|---|---|
| G0: semantics | `DFQ8_B32_V1` Python and Rust scalar semantics, fixtures, and schema agree, followed by a checked-in provenance/evidence bundle | none until the G0 evidence bundle is complete |
| G1: M4 vertical slice | A TinyLlama `M=1` projection lowers to generated scalar and ARM64 NEON on the M4, with source, disassembly, correctness, and timings | bounded Mac schedule evidence |
| G2: Mac schedule evidence | Bounded schedule selection on the M4 is correctness-gated, reproducible, and measured | guarded Mac PyTorch integration |
| G3: Mac framework proof | A guarded `torch.compile` region executes end to end on the M4, falls back safely, and reports coverage | evidence-selected extension |
| G4: evidence-selected extension | One measured extension—AVX2 portability, fusion, small batch, or multicore—wins or yields an honest negative result under the same Q8 contract | — |

Failure at a gate causes investigation or a scope cut; it does not unlock more
surface area. The dashboard is presentation polish and comes after G3.

## Repository plan

```text
compiler/                 Rust workspace
  decodeforge-core/       G0 Q8 semantic oracle, identities, fixture gates
  decodeforge-compiler/   G1 IR, OI4 packing, scalar codegen, and Apple artifact audit
  decodeforge-runtime/    module cache, guards, loading, ABI
python/decodeforge/       torch.compile backend, FX partitioning, test harness
native/torch_bridge/      thin ATen/PyTorch pointer/shape bridge
benchmarks/               kernel, layer, and TinyLlama suites
results/                  reproducible run manifests, raw data, assembly, reports
dashboard/                optional post-G3 compiler report viewer
docs/                     design, implementation gates, benchmark methodology
```

The AVX2 code-generation entry is retained for the deferred G4 portability
extension; it is not part of the required Mac-first G0–G3 path.

## Credible success

The project is résumé-ready when it can demonstrate all of the following:

- scalar and NEON kernels agree with a dequantize-then-matmul oracle within a
  documented numeric tolerance on the M4; AVX2 is required only if selected as
  the G4 portability extension;
- compiled regions preserve the PyTorch callable contract and reject guard
  violations safely;
- schedule selection is reproducible and beats the untuned generated schedule
  on at least one real TinyLlama projection shape on the M4;
- compiler time, tuning time, cache behavior, packed-weight size, throughput,
  latency, and quantization error are reported together;
- a TinyLlama decoder block or generation path uses compiled regions end to end;
- any optional G4 cross-target result is explained—for example, a tile that
  helps the M4 but hurts Zen 2;
- at least one optimization is supported by annotated disassembly and available
  CPU counters, connecting the source-level schedule to observed machine behavior;
- comparisons to PyTorch Inductor, llama.cpp, or vendor libraries are labeled as
  contextual rather than falsely identical when formats/semantics differ.

No target speedup is assumed in advance.

Hand-written assembly, a custom thread runtime, and operating-system internals
are not required claims. The low-level contribution is CPU-kernel generation,
data layout, native ABI integration, and evidence-based microarchitectural
analysis.

## Documents

- [Design and technical specification](docs/DESIGN.md)
- [Benchmark and experimental methodology](docs/BENCHMARKS.md)
- [Implementation plan and decision gates](docs/IMPLEMENTATION_PLAN.md)
- [G0 evidence contract](docs/G0_EVIDENCE_V1.md)
- [ADR 0001: Mac-first required path](docs/decisions/0001-mac-first-required-path.md)

## Status

G0 is complete. The independent Python and Rust scalar oracles, closed fixture
schemas, and 16-case corpus pass byte-for-byte parity gates. The checked-in
[Apple M4 correctness bundle](results/g0/apple-m4-primary/sha256-311053f53efd9c28ab3e4338ca83e78e53acf8c969d9f8a76c6e56f7c2d79d86/report.md)
binds those checks to source revision `cc838b0`, the exact toolchain and host
profile, and hashed artifacts; CI verifies both its portable contents and Git
provenance. It deliberately makes no native-kernel or performance claim. The
first G1 slice now implements verified Region/Loop lowering, one shared OI4
pack, the frozen generated-module ABI, deterministic strict scalar C, and an
Apple-arm64 builder that retains and structurally audits the exact private
Mach-O artifact and scalar helper disassembly. A checked runtime now consumes
that owner, makes an independently verified read-only load copy, binds the
frozen ABI, and executes all 16 fixtures bit-exactly through a safe Rust API.
NEON code generation, benchmark evidence, and performance claims do not yet
exist. See [the normative Q8 contract](docs/Q8_FORMAT_V1.md).

Rust independently generates the expected fixture documents in memory, and
`q8 verify` only reads and verifies an existing fixture tree. Run the read-only
Rust gate with `make rust-fixture-check`, or directly:

```sh
PATH="$(dirname "$(rustup which --toolchain 1.98.0 cargo)"):$PATH" cargo run --offline --locked -p decodeforge -- q8 verify
```

The Python generator is the sole explicit fixture writer:

```sh
uv run --frozen python scripts/generate_q8_fixtures.py --write
```

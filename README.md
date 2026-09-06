# DecodeForge

**A shape-specializing compiler for quantized LLM linear layers on commodity CPUs.**

New here? Start with the [plain-language primer](docs/PRIMER.md), jump to the
[benchmark results](#measured-results-apple-m4), or read the
[contributor setup guide](CONTRIBUTING.md).

DecodeForge's original code is licensed under [Apache 2.0](LICENSE).
Third-party dependencies, model artifacts, and separately attributed material
retain their own licenses.

DecodeForge compiles the dominant operation in autoregressive LLM decode—large
matrix-vector products with frozen weights—into guarded ARM64 NEON kernels on
an Apple M4. Its completed compiler path lowers a typed Q8 linear operation,
packs weights into an output-interleaved layout, emits scalar and NEON C, asks
Clang/LLVM to build the machine code, audits the artifact, and executes it
through a versioned native ABI. The completed G2 boundary exposes that artifact
as a guarded eager PyTorch operator, and the G3 implementation installs it for
all 22 TinyLlama query projections during cached single-token decode. Three
accepted model sessions and their independently verified result bundle complete
the frozen G3 technical gate. A separate chat-formatted presentation demo now
produces useful text with identical reference/native tokens; it does not replace
that accepted evidence.

PyTorch and Transformers still own model loading, tokenization, attention, KV
state, sampling, and unsupported operations. Prompt prefill uses a reference
path reconstructed from the same Q8 weights; only eligible `M=1` decode calls
enter generated native code. General `torch.compile`/FX integration, schedule
search, all-model linear coverage, fusion, multicore execution, and x86-64 AVX2
are evidence-selected extensions rather than prerequisites for a usable demo.

The project asks one question:

> Can one small compiler own a real hot path in CPU text generation—Q8 lowering,
> packing, ARM64 NEON code generation, and native execution across all 22 query
> projections—while producing evidence strong enough to defend every claim?

The Mac-first target choice is recorded in
[ADR 0001](docs/decisions/0001-mac-first-required-path.md). The shorter path
from the completed compiler to an eager query-projection generation demo is
recorded in
[ADR 0005](docs/decisions/0005-prioritize-eager-q-projection-demo.md).

This is deliberately not an inference server, work-stealing runtime, KV-cache
manager, general tensor framework, or GPU compiler. PyTorch/Transformers owns
model loading, tokenization, attention, KV state, and generation. DecodeForge
owns the compiler path for a narrow set of hot CPU operators.

## Measured results: Apple M4

TinyLlama already runs locally with PyTorch. DecodeForge demonstrates a custom
compiler taking over **22 query projections during cached single-token decode**,
not a new ability to run the model locally or a general-purpose inference engine.

| Evaluation | Observed result | What it establishes |
| --- | --- | --- |
| Generated NEON vs generated scalar | **~3.96x speedup** in each of 3 independent sessions | Same-Q8 single-projection prepared-call performance, **not whole-model speedup** |
| Native vs same-Q8 reference correctness | **30/30 prompts; 1,070 generated steps; exact token agreement** | Correct native dispatch across all 22 layers and clean restoration; maximum absolute logit difference `0.0000171661` |
| Q8 vs original FP32 sensitivity | **30/30 greedy sequences matched; 99.7099% next-token argmax agreement** on 1,034 fixed reference tokens | Small observed quantization effect on this synthetic corpus, not a general quality guarantee |
| Model performance protocol | **81 measured generations + 27 warmups across 3 fresh processes** | Separate performance measurements without correctness-comparison hooks |
| Setup and memory | **16.30–16.63 s** setup components; **4.62–4.80 GiB** peak process RSS | Reused pinned artifacts/caches; RSS covers all paths, not per-path memory savings |

Decode throughput below is in **tokens/second**. Each range spans the three
per-process medians, not a confidence interval or selected best runs. The cases
use short/medium/long prompts and output caps of 16/32/64 tokens respectively.

![Apple M4 decode throughput: native execution is faster than the guarded same-Q8 reference but does not consistently beat FP32 PyTorch. Exact ranges follow in the table.](docs/assets/apple-m4-decode-throughput.svg)

| Case / output cap | Original FP32 PyTorch | Hybrid native | Guarded same-Q8 reference |
| --- | --- | --- | --- |
| Short / 16 | 11.61–15.47 | 11.29–14.74 | 3.84–4.70 |
| Medium / 32 | 14.78–14.86 | 14.17–14.29 | 4.27–4.53 |
| Long / 64 | 12.23–13.74 | 10.85–13.23 | 4.37–4.47 |

**The native path beats the guarded same-Q8 reference, but does not consistently
beat original FP32 PyTorch.** Production guards remain enabled, including the
reference fallback's weight cloning/hashing; finite-logit checks remain in decode
timing. These are guarded model-boundary measurements, not isolated kernel
timings. Only query projections use native decode; the rest remains in PyTorch.
Also, 29/30 correctness generations hit their token cap, and a second physical
Mac remains untested: neither broad instruction-following quality nor cross-host
performance is established.

See the [G1 kernel measurements](results/g1/apple-m4-primary/README.md) and
[full model evaluation](results/evaluation/apple-m4-v1/README.md) for raw evidence,
confidence intervals where applicable, prefill/total-generation timings, and
limitations. Recompute and verify the retained analyses without running the model:

```sh
make verify-g1-result verify-evaluation-result
```

## Why this scope

The original all-in-one engine concept packages several independent systems
questions. That makes a result hard to attribute and leaves too many components
half-finished. DecodeForge has one measurable contribution:

```text
frozen q_proj weight + static [N,K] + CPU target
                         |
                         v
                  typed Q8Linear IR
                         |
                         v
               quantize + OI4 weight pack
                         |
                         v
             generated scalar / NEON source
                         |
                   Clang + artifact audit
                         |
                         v
              versioned native runtime bridge
                         |
                         v
          guarded eager PyTorch operator (G2)
                         |
                         v
  22 q_proj adapters: same-Q8 prefill / native decode (G3)
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

The required vertical slice is deliberately narrower than a general graph
compiler: frozen, bias-free weight-only Q8 query projections with
`[N,K]=[2048,2048]`. TinyLlama contains 22 of them. Prompt prefill has `M>1`
and therefore uses the identity-bound same-Q8 reference path; cached decode has
`M=1` and is eligible for generated NEON execution.

The remaining 133 TinyLlama linear modules, additional shapes, `M>1` native
kernels, RMSNorm or SwiGLU fusion, general FX matching, predictive scheduling,
and AVX2 stay outside G0–G3. This yields repeated real-model coverage without
pretending that one fixed kernel is already a whole-model inference engine.

The first reference model is
[`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0),
an Apache-2.0 Llama-compatible 1.1B model with realistic decode projection
shapes. The compiler is shape-driven rather than hard-coded to that model, but
only those shapes are required for the first complete result.

## What makes it a compiler

- A typed IR represents contraction, quantization, scale, epilogue, layout, and
  reduction semantics independently of the M4 target; future targets reuse it.
- Region and Loop IR make the fixed G1 schedule, reduction order, vector width,
  tail behavior, and packed addressing explicit and verifiable.
- The existing lowering is structured so a later legality layer and bounded
  schedule selector can vary those decisions without changing Q8 semantics.
- Code generation emits inspectable C/intrinsics plus a stable C ABI, then uses
  the host toolchain to build a loadable native module.
- Guards bind a kernel to the shapes, strides, dtype, alignment, quantization
  format, and CPU features for which it was compiled.
- Content identities bind the logical weight, physical pack, module, shape,
  numeric mode, and ABI at the native boundary.
- Every optimization is benchmarked against the same quantized scalar semantics,
  not only against a different precision or framework.

## Evidence, not architecture alone

The primary project artifact is a reproducible compiler run, not this design.
For each published generated kernel, DecodeForge retains:

- canonical Region IR and Loop IR;
- generated C/intrinsics and the exact compiler invocation;
- disassembly with the hot loop identified, vector instructions checked, and
  stack spills or scalarized paths called out;
- the logical and physical weight-layout manifests;
- correctness results against the Q8 oracle;
- raw timing samples, host state, and available hardware-counter measurements;
- the fixed schedule, plus selected and rejected candidates once schedule search
  is implemented;
- compile/pack time, code size, and—where applicable—tuning time, cache-hit
  latency, and break-even calls.

Every performance claim must be reconstructible from a checked-in result bundle.
If a counter is unavailable on a host, the manifest records that fact instead of
substituting an estimate.

| Skill signal | Required proof |
|---|---|
| compiler construction | typed/verified IR, explicit schedule representation, deterministic lowering and code generation |
| SIMD and machine code | retained M4 NEON source, disassembly audit, scalarization/spill checks; AVX2 only if selected as G4 extension |
| memory-system reasoning | packed-layout accounting, bandwidth calibration, cache counters when available |
| ABI and FFI safety | versioned C ABI, pointer/shape guards, negative tests, corrupt-artifact recovery |
| performance engineering | raw randomized samples, uncertainty, same-semantics baselines, break-even analysis |
| target judgment | one measured schedule tradeoff explained on M4; a cross-target comparison is optional G4 evidence |

## Delivery gates

| Gate | Required result | Scope unlocked |
|---|---|---|
| G0: semantics — complete | `DFQ8_B32_V1` Python and Rust scalar semantics, fixtures, schema, and checked-in provenance bundle agree | generated code |
| G1: M4 compiler/kernel — complete | A real TinyLlama `M=1` query projection lowers to generated scalar and ARM64 NEON with retained source, disassembly, correctness, and timings | framework boundary |
| G2: native eager PyTorch boundary — complete | The hardened versioned C ABI and guarded eager `q8_linear_v1` operator execute the real release library with observable native, fallback, error, and lifecycle paths | model adapter |
| G3: 22-projection generation proof — complete under the frozen protocol | Three accepted fresh-process sessions and the verified ten-file bundle establish all-22 native coverage, numerical/token agreement, and clean restoration; text quality is a separate presentation limitation | evidence-selected extension |
| G4: evidence-selected extension | One measured next step—schedule selection, broader linear coverage, FX/`torch.compile`, fusion, AVX2, or multicore—wins or yields an honest negative result | — |

Failure at a gate causes investigation or a scope cut; it does not unlock more
surface area. In particular, the G1 kernel result cannot be relabeled as a
model speedup, and a text demo without same-Q8 comparison and coverage counters
does not complete G3.

## Repository plan

```text
compiler/                 Rust workspace
  decodeforge-core/       G0 Q8 semantic oracle, identities, fixture gates
  decodeforge-compiler/   G1 IR, OI4 packing, scalar codegen, and Apple artifact audit
  decodeforge-runtime/    generated-module ownership, validation, loading, ABI
  decodeforge-bridge/     hardened process-local C ABI for native modules and packs
python/decodeforge/       eager PyTorch binding, q_proj adapter, test harness
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
- the eager operator passes real tensor storage to the release bridge, preserves
  the PyTorch callable contract, and rejects guard violations safely;
- all 22 TinyLlama query projections use packs prepared by the compiler path;
  prefill uses the same Q8 identity and cached `M=1` decode enters native code;
- compiler/pack time, library and model load time, packed-weight size, bridge
  dispatch, prefill, per-token decode, total latency, and quantization quality
  are reported separately;
- a pinned prompt generates text with native coverage proven by counters and
  agrees with an all-same-Q8-fallback run under a predeclared policy;
- any optional G4 cross-target result is explained—for example, a tile that
  helps the M4 but hurts Zen 2;
- the completed G1 optimization is supported by retained source, annotated
  disassembly, and raw paired measurements;
- comparisons to PyTorch Inductor, llama.cpp, or vendor libraries are labeled as
  contextual rather than falsely identical when formats/semantics differ.

No target speedup is assumed in advance.

Hand-written assembly, a custom thread runtime, and operating-system internals
are not required claims. The low-level contribution is CPU-kernel generation,
data layout, native ABI integration, and evidence-based microarchitectural
analysis.

## Documents

- [Plain-language project primer](docs/PRIMER.md)
- [Design and technical specification](docs/DESIGN.md)
- [Benchmark and experimental methodology](docs/BENCHMARKS.md)
- [Implementation plan and decision gates](docs/IMPLEMENTATION_PLAN.md)
- [Delivery progress and remaining issues](docs/PROGRESS_2026_09_05.md)
- [G0 evidence contract](docs/G0_EVIDENCE_V1.md)
- [ADR 0001: Mac-first required path](docs/decisions/0001-mac-first-required-path.md)
- [ADR 0004: Strict output-vector NEON lowering](docs/decisions/0004-strict-output-vector-neon.md)
- [ADR 0005: Eager Q-projection generation demo](docs/decisions/0005-prioritize-eager-q-projection-demo.md)

## Status

G0 is complete. The independent Python and Rust scalar oracles, closed fixture
schemas, and 16-case corpus pass byte-for-byte parity gates. The checked-in
[Apple M4 correctness bundle](results/g0/apple-m4-primary/sha256-311053f53efd9c28ab3e4338ca83e78e53acf8c969d9f8a76c6e56f7c2d79d86/report.md)
binds those checks to source revision `cc838b0`, the exact toolchain and host
profile, and hashed artifacts; CI verifies both its portable contents and Git
provenance. G1 is also complete. Its compiler path implements verified
Region/Loop lowering, one shared
OI4 pack, the frozen generated-module ABI, deterministic strict scalar and
output-vector NEON C, and separately identified Apple-arm64 dylibs whose
Mach-O structure and hidden helpers are audited before loading. The NEON
source expresses signed `int8 -> int16 -> int32` widening; the retained machine
code contains the corresponding `sshll.8h -> sshll.4s -> scvtf.4s` path,
lane-form activation multiply, separate adds, a raw vector scale load, and
guarded vector/tail stores. Direct Q-word and 16-byte scale loads removed
unnecessary byte reconstruction and a temporary stack-array/canary path
without disabling stack protection. A backend-neutral checked runtime executes
scalar and NEON modules through the same ABI: all 16 frozen fixtures are
bit-exact, and dedicated `N=4` and `N=5` cases prove vector-only and
vector-plus-tail execution. This is correctness and machine-code evidence
for the generated path. A backend-neutral prepared-call API
now validates buffer extents once, reuses caller-owned output storage, and
includes deterministic output scrubbing and validation around every native
invocation. See [the normative Q8 contract](docs/Q8_FORMAT_V1.md).

The checked-in
[Apple M4 G1 result](results/g1/apple-m4-primary/README.md) prepares one
byte-stable, provenance-pinned
TinyLlama tensor; independently reconstructs the canonical Q8 pack and oracle;
and runs generated scalar and NEON artifacts through the same allocation-free
prepared-call boundary. Each raw session retains the Region/Loop IR, pack
manifest, generated source, disassembly audit, exact toolchain, host and Git
state, correctness gates, calibration, and all 80 balanced observations. The
portable analyzer accepts exactly three clean-checkout processes, rejects
thermal drift above the declared 10% policy, and computes deterministic paired
BCa intervals. The three session speedups are `3.95671x`, `3.96176x`, and
`3.95648x`; their respective 95% paired-BCa intervals are
`[3.95103, 3.96705]`, `[3.95085, 3.96960]`, and `[3.95351, 3.95997]`.
All three lower bounds exceed `1.0`, so the predeclared G1 speedup gate passes.
This is a generated scalar-versus-NEON kernel result at the complete prepared
call boundary, not an end-to-end model speedup.

G2 is also complete. `decodeforge-bridge` exports the versioned six-function C
ABI in
[`include/decodeforge/runtime_v1.h`](include/decodeforge/runtime_v1.h). It owns
verified generated modules and exact aligned OI4 payloads behind opaque,
process-local handles; enforces per-pack, aggregate-byte, and live-handle
limits; linearizes run/destroy; contains panics; and exposes bounded
thread-local diagnostics. `make test-bridge-cdylib` builds the actual release
library and verifies the frozen `N=255,K=2` fixture through that C boundary
(bit-exact with real Torch buffers on Apple ARM64, explicit unsupported-host
behavior on Linux). The guarded eager PyTorch operator adds verified private
library snapshots, exact tensor guards, observable fallback/error counters, and
tested lifecycle ownership around that release boundary.

G3 is complete under its frozen execution/correctness protocol. The checked-in
[ten-file M4 result](results/g3/apple-m4-primary/README.md) retains three
accepted fresh-process sessions from clean revision `ad15f5d`, with all 22
query projections exercising native cached decode, exact token agreement,
numerical checks and clean restoration. `make check` now independently
regenerates and verifies this bundle as well as the G1 result.

The pooled median total-generation times are 1.732 seconds for the guarded
same-Q8 reference and 0.751 seconds for hybrid native execution. The reference
includes per-call fallback-weight cloning and hashing; both paths include hook
instrumentation. These are not isolated kernel timings or a stock-PyTorch
comparison. The frozen prompt produced control-token text rather than a useful
sentence. A separate [chat-template presentation demo](docs/PRESENTATION_DEMO.md)
now produces a meaningful answer with identical tokens, all-22 native coverage,
and clean restoration; it gives two sentences rather than the requested one.
See the [original result interpretation](docs/G3_RESULT_2026_09_05.md).

A separate [30-prompt M4 evaluation](results/evaluation/apple-m4-v1/README.md)
passed native/reference correctness and preserved all 30 greedy FP32 sequences
on its synthetic corpus. Three fresh performance processes show an advantage
over the guarded same-Q8 fallback, but not a consistent advantage over original
FP32. Most generations reached their token cap, and a second physical Mac is
still untested. Run `make verify-evaluation-result` to check the retained
observations and summary; this verification is included in `make check`.

Reproduce the checked-in analysis with `make verify-g1-result`.

Run the closed G1 path with explicit artifacts and session IDs:

```sh
make prepare-g1-input WEIGHTS=/path/to/model.safetensors OUTPUT=/tmp/g1-weight.safetensors
make prepare-g1-cases PREPARED_WEIGHTS=/tmp/g1-weight.safetensors OUTPUT=/tmp/g1-cases
make run-g1-session CASES=/tmp/g1-cases/manifest.json OUTPUT=/tmp/session-01.json SESSION_ID=session-01
make run-g1-session CASES=/tmp/g1-cases/manifest.json OUTPUT=/tmp/session-02.json SESSION_ID=session-02
make run-g1-session CASES=/tmp/g1-cases/manifest.json OUTPUT=/tmp/session-03.json SESSION_ID=session-03
make analyze-g1 SESSION_1=/tmp/session-01.json SESSION_2=/tmp/session-02.json SESSION_3=/tmp/session-03.json OUTPUT_DIR=/tmp/g1-report
```

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

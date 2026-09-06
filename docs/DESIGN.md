# DecodeForge design and technical specification

**Status:** G0 and G1 are complete with checked-in evidence. The fixed G1
vertical slice includes verified lowering, shared OI4
packing, deterministic scalar and strict output-vector NEON source, audited
Apple-arm64 scalar/NEON dylibs, and checked loading through the frozen ABI.
Both backends execute all 16 frozen fixtures bit-exactly; dedicated `N=4` and
`N=5` cases verify vector-only and vector-plus-tail machine code. A prepared
safe-call API validates buffer extents once and reuses caller-owned output while
preserving complete output scrubbing on every failure. A closed real-shape G1
harness prepares the pinned TinyLlama tensor, reconstructs and binds the
Q8/oracle assets, records 40 balanced prepared-call pairs per process, and
retains IR, source, disassembly, pack, host, toolchain, and checkout evidence.
Its analyzer requires three clean independent processes plus drift and paired
BCa gates. The checked-in sessions measured `3.95671x`, `3.96176x`, and
`3.95648x` paired speedups; all three 95% paired-BCa lower bounds exceed
`3.95x`, passing the predeclared G1 gate. This is a generated-kernel result at
the complete prepared-call boundary, not an end-to-end model speedup. The
normative contract is [Q8_FORMAT_V1](Q8_FORMAT_V1.md). G2 is complete:
`decodeforge-bridge` provides a bounded, panic-contained, versioned C ABI over
verified generated modules and exact OI4 packs, and the guarded eager PyTorch
operator exercises it through the actual release library with closed guard,
fallback, error, and lifecycle tests. G3.0–G3.3 are code-complete, including
deterministic 22-layer asset preparation, the identity-bound owning adapter,
and transactional all-layer installation. The hardened generation runner and
ten-file analyzer/verifier now retain three accepted G3.4 sessions and verified
G3.5 evidence. G3 is complete under its frozen protocol. Its generated text is
control-token output; the separate [chat-template demo](PRESENTATION_DEMO.md)
now produces useful text without replacing that evidence.
Generation timings compare guarded/instrumented same-Q8 and hybrid-native
paths, not stock PyTorch or isolated kernel work; see the
[result interpretation](G3_RESULT_2026_09_05.md).

**Primary contribution:** A shape-specializing schedule compiler for frozen,
weight-only Q8 LLM linear regions, with the required vertical slice on an Apple
M4 using ARM64 NEON. x86-64 AVX2 is a deferred optional portability extension,
as recorded in [ADR 0001](decisions/0001-mac-first-required-path.md).

The compiler and generated-runtime contract is limited to 64-bit little-endian
hosts. The scalar target label `portable` means portable within that supported
host class, not across 32-bit or big-endian systems.

## 1. Goals

1. Accept a small, normalized graph region containing frozen Llama-family linear
   operations and supported epilogues.
2. Represent contraction, Q8 dequantization, reductions, layouts, fusion, and
   numeric behavior in a target-independent typed IR.
3. Emit target-specific scalar/NEON code for the required M4 path and preserve a
   verifier-visible fixed schedule. Keep bounded schedule selection and an
   x86-64 AVX2 lowering as deferred extension points.
4. Integrate generated modules through a hardened C ABI and a guarded eager
   PyTorch operator before attempting a general graph backend.
5. Demonstrate all 22 TinyLlama 1.1B query projections in prompt-to-text
   generation on the Apple M4: identity-bound same-Q8 fallback for prefill and
   native generated code for cached `M=1` decode.
6. Separate the fixed kernel, packing, bridge, query-projection coverage, and
   whole-model measurements rather than attributing attention/KV/sampling work
   to the compiler.
7. Produce an inspectable compiler report: IR, fixed schedule, generated source,
   packed layout, guards, code size, build time, correctness, and raw runtime
   metrics. If G4 adds tuning, include every rejected and selected candidate.
8. Tie at least one source-level optimization in the required ARM64 NEON path to
   generated assembly and available hardware counters, so low-level claims are
   independently inspectable rather than inferred from latency alone.
   Apply the same evidence standard to an optional x86 extension if selected.

## 2. Non-goals

- a full LLM inference engine, tokenizer, sampler, HTTP server, or KV manager;
- training, backward graphs, mutable weights, distributed execution, or GPUs;
- arbitrary PyTorch graphs or every ATen operator;
- a general-purpose replacement for TorchInductor, TVM, MLIR, or llama.cpp;
- every quantization format; Q4, activation quantization, and mixed precision are
  separate future work;
- integer dot-product code paths while activations remain FP32;
- custom thread scheduling or work stealing;
- an x86-64 AVX2 backend in the required first result; it is a deferred optional
  portability extension after the Mac path works end to end;
- promising to beat vendor libraries or llama.cpp before measurement.

### 2.1 Promotion discipline

Implementation advances through evidence gates rather than component count:

| Gate | Exit evidence |
|---|---|
| G0: semantics — complete | `DFQ8_B32_V1` Python and Rust scalar semantics, fixtures, schema, and checked-in provenance bundle agree |
| G1: M4 compiler/kernel — complete | One real TinyLlama q-projection flows through verified IR to generated scalar and ARM64 NEON; the bundle contains source, assembly, correctness, timings, and host metadata |
| G2: native eager PyTorch boundary — complete | The hardened C ABI and guarded eager `q8_linear_v1` operator execute the release artifact with tested native, fallback, error, and lifecycle paths |
| G3: 22-projection generation proof — complete under the frozen protocol | Three accepted fresh-process sessions and the verified ten-file bundle prove all-22 native coverage, numerical/token agreement, and clean restoration; useful text is a separate presentation task |
| G4: evidence-selected extension | One measured extension—schedule selection, broader linear coverage, FX/`torch.compile`, fusion, AVX2, native small batch, or multicore—wins or yields an honest negative result |

Work that belongs to a later gate is kept out of the critical path. In
particular, schedule search, all-155-linear coverage, general FX integration,
the visual dashboard, predictive cost-model claims, both fusions, multicore
tuning, native `M > 1`, and x86-64 work cannot delay G0–G3. See
[ADR 0005](decisions/0005-prioritize-eager-q-projection-demo.md).

## 3. Compiler/runtime boundary

PyTorch/Transformers owns everything except the explicitly replaced query
projections:

```text
tokenizer + TinyLlama + generation loop
  | attention, KV cache, sampling, every non-q_proj operator
  |
  +--> ordinary PyTorch execution -----------------------------+
  |                                                            |
  +--> 22 owning q_proj adapters                               |
          |                                                     |
          +--> prompt prefill (M>1): same-Q8 fallback ----------+
          |                                                     |
          +--> cached decode (M=1): guarded eager operator      |
                                      |                         |
                                      v                         |
                            versioned runtime C ABI              |
                                      |                         |
                                      v                         |
                         audited generated NEON module ----------+
```

This boundary prevents the compiler project from silently becoming an inference
engine. Text generation is a consumer of the compiler. Tokenization, attention,
KV state, sampling, and the remaining 133 TinyLlama linear modules are reused
from standard frameworks.

## 4. Frontend contract

The required G2 frontend is one eager-only logical operator:

```text
decodeforge::q8_linear_v1(Tensor x, int binding_id, int n, int k) -> Tensor
```

`binding_id` resolves to an owned process-local native binding whose descriptor
fixes `N`, `K`, module identity, packed-weight identity, and packed byte count.
The operator allocates a contiguous CPU FP32 output, lends the input and output
tensor storage directly to the C ABI, and returns the output only after the
native call succeeds. It never passes model weights through the dispatcher.

The Python binding:

1. lazily imports Torch so the base package stays framework-independent;
2. verifies and privately snapshots the exact bridge library bytes before
   loading them;
3. binds the frozen C signatures and owns handles through synchronized objects;
4. registers bindings in a synchronized process-local registry;
5. validates device, dtype, layout, contiguity, inference state, static shape,
   and singleton leading dimensions before native entry;
6. records completed native/fallback/error calls and in-flight work;
7. allows an explicit fallback only on a guard miss—never after a native attempt
   has begun.

General Dynamo/FX capture, fake/meta behavior, graph partitioning, compilation
from symbolic graphs, and a `torch.compile` backend are possible G4 work. They
are not required to prove that the existing compiler artifact runs inside a
real model.

### 4.1 Quantization and packing are explicit

Framework integration must not quietly reinterpret the original FP32 model.
The G3 preparation step explicitly reads each frozen source `q_proj` weight,
quantizes it under `DFQ8_B32_V1`, and routes it through the canonical Rust OI4
packer. Each layer receives a manifest/payload pair bound to the model revision,
source tensor key and hash, logical shape, Q8 identity, packed identity, and
module identity.

The canonical preparation path also dequantizes those exact Q8 values into a
frozen FP32 fallback tensor and records its hash plus parent pack identity.
The adapter uses ordinary `torch.nn.functional.linear` over that buffer for
`M>1`; Python does not implement a second quantizer or OI4 unpacker. The source
FP32 `nn.Linear` may be retained for quality context or restoration, but it is
not a legal fallback for the hybrid Q8 experiment. Generated/native and
reference paths may differ only by their declared FP32 reduction
implementation, not by quantized weight values.

### 4.2 Required model shape and dispatch

G3 replaces exactly:

```text
model.layers.{0..21}.self_attn.q_proj
```

Every module is bias-free with `[N,K]=[2048,2048]`. The owning adapter preserves
the normal linear callable shape:

- prompt prefill such as `[1,S,2048]`, `S>1`, runs through same-Q8 fallback;
- cached decode `[1,1,2048]` runs through the native eager operator;
- all static `N`, `K`, dtype, device, contiguity, inference, and view-bit guards
  must pass before native entry.

The 22 replacements are validated before mutation and installed
transactionally. Partial installation rolls back. The final result must prove
prefill fallback and cached native coverage independently for every adapter.

Native `M>1`, other projection shapes, all-155-linear replacement, RMSNorm
fusion, gate/up fusion, and small batches are promotion-gated G4 extensions.

### 4.3 Guard misses and errors

These conditions are ordinary native guard misses when an owning adapter has an
explicit same-Q8 fallback:

- `M` is not exactly one;
- tensor is not CPU FP32, strided, contiguous, finite, and inference-only;
- rank/leading dimensions or `K` do not match the binding;
- conjugate or negative view bits are set.

A wrong binding identity, descriptor mismatch, invalid/closed handle, failed
module build/load, native nonfinite result, or bridge failure is not a fallback
condition. It raises a structured error. This distinction prevents a broken
native attempt from disappearing behind successful framework execution.

## 5. Reference model and shapes

The integration target is
[`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0),
an Apache-2.0 Llama-compatible model.

Published configuration relevant to linear kernels:

| Property | Value |
|---|---:|
| Layers | 22 |
| Hidden width | 2,048 |
| Intermediate width | 5,632 |
| Query heads | 32 |
| KV heads | 4 |
| Vocabulary | 32,000 |

TinyLlama linear inventory and promotion status:

| Region | Logical matrix shape `[N, K]` | Count | G0–G3 status |
|---|---:|---:|---|
| Q projection | `[2048, 2048]` | 22 | required G3 native-decode path |
| attention output | `[2048, 2048]` | 22 | deferred despite shared shape |
| K projection | `[256, 2048]` | 22 | deferred |
| V projection | `[256, 2048]` | 22 | deferred |
| gate projection | `[5632, 2048]` | 22 | deferred |
| up projection | `[5632, 2048]` | 22 | deferred |
| down projection | `[2048, 5632]` | 22 | deferred |
| language-model head | `[32000, 2048]` | 1 | deferred |

This is 155 bias-free linear modules in total. G3 deliberately targets only the
22 query projections: they repeat one real compiler shape across every decoder
layer while keeping memory use and correctness attribution bounded.

The exact checkpoint revision and hashes are pinned by the benchmark manifest,
not by a mutable `main` reference.

## 6. Quantization semantics

The normative readable contract, including exact raw-bit identities and the
fixture/check command, is [Q8_FORMAT_V1](Q8_FORMAT_V1.md).

Initial logical format: `DFQ8_B32_V1`.

For every consecutive block of 32 weights:

```text
amax  = max(abs(w[0:32]))
scale = 0                         if amax == 0
        amax / 127                otherwise
q[i]  = 0                         if scale == 0
        clamp(round(w[i]/scale), -127, 127) otherwise
```

Reference dot product:

```text
y[n] = Σ_blocks scale[n, b] × Σ_i x[b×32+i] × float(q[n,b,i])
```

Contract:

- weights are signed int8;
- scales are FP32 initially;
- activations and accumulation are FP32;
- block length is exactly 32; a padded tail is zero-filled and guarded;
- rounding is specified; Python, Rust scalar, generated scalar, and generated
  NEON results have bit-exact parity across the 16 frozen fixtures on the M4;
- NaN/Inf source weights are rejected by default;
- source-vs-Q8 quality compares source weights with `dequantize_f32_bits`,
  separately from generated-kernel comparator correctness.

Both the Python and Rust strict-f32 helpers use integer/rational rounding for
binary32 add, multiply, divide, and integer conversion. Native floating-point
arithmetic is not part of the semantic oracle; it cannot change fixture bits.

The complete bit-level contract is frozen in
[`docs/Q8_FORMAT_V1.md`](Q8_FORMAT_V1.md), including raw-word finite checks,
strict operation order, gradual underflow, identities, and the generated
fixture corpus. The q layout is physically `[N][B][32]`; only logical lanes
participate in `amax` and evaluation, and all tail lanes serialize as zero.

Logical storage is approximately 36 bytes per 32 weights (32 int8 + one FP32
scale) before target packing, versus 128 bytes in FP32.

Reduced-size scales, activation quantization, per-channel formats, and Q4 are not
part of the first result.

## 7. Intermediate representation

The compiler uses two small IR levels rather than a general tensor framework.

### 7.1 Region IR

Region IR preserves operator semantics and fusion opportunities:

```rust
struct Region {
    args: Vec<Value>,
    ops: Vec<RegionOp>,
    results: Vec<ValueId>,
    constraints: ShapeConstraints,
}

enum RegionOp {
    RmsNorm(RmsNormOp),
    Q8Linear(Q8LinearOp),
    Silu(SiluOp),
    Mul(BinaryOp),
}
```

Tensor types contain rank, static/symbolic dimensions, dtype, strides, alignment,
and alias class. Q8 weight values include quantization format and constant ID.

### 7.2 Loop IR

Loop IR makes schedule and target decisions explicit:

```rust
struct LoopKernel {
    loops: Vec<Loop>,
    loads: Vec<Load>,
    accumulators: Vec<Accumulator>,
    epilogue: Vec<ScalarExpr>,
    stores: Vec<Store>,
    parallel: ParallelPlan,
}
```

It represents:

- `M`, output-channel, quant-block, and within-block loops;
- loop order, split factors, unroll, vector lanes, and tails;
- Q8/scales/activation loads and packed address formulas;
- FP32 accumulators and reduction order;
- optional input RMS scale and SwiGLU epilogue;
- output-channel parallel partition and grain;
- prefetch annotations.

No arbitrary control flow is needed in the MVP.

### 7.3 Text and verification

Both IR levels have deterministic text forms used in snapshots and compiler
reports. The verifier checks:

- SSA/use-def and operator type/shape rules;
- `K` compatibility between activation and weight;
- Q8 block/tail legality;
- packed address bounds and alignment;
- every output element is written once;
- reduction covers every logical `K` exactly once;
- SIMD target features satisfy the selected instructions;
- fusion preserves uses and numeric contract;
- parallel workers write disjoint output ranges;
- guards cover all assumptions embedded in generated code.

## 8. Deferred G4 canonicalization and fusion

The following graph work is retained as a design extension, not implemented or
required by the eager G2/G3 path.

Canonicalization normalizes equivalent FX patterns, removes redundant views,
folds constant shapes/scales, and makes frozen parameters explicit.

### 8.1 RMSNorm + linear

Ordinary execution writes a normalized vector, then reads it for the linear.
The fused region:

1. reduces `x²` once per row;
2. computes `r = rsqrt(mean(x²) + eps)`;
3. uses `x[k] × gamma[k] × r` directly in the Q8 dot product;
4. does not allocate/materialize the normalized vector.

The compiler reports bytes and dispatches removed. Fusion is rejected when the
normalized value has another use not included in the region.

### 8.2 Paired gate/up + SwiGLU

Gate and up projections share the same input vector. A fused schedule may load
an activation block once, update accumulators for both weight panels, and apply
`silu(gate) × up` before storing the intermediate output.

The fused schedule still reads two weight matrices. Its benefit is not assumed:
larger accumulator pressure may cause spills, and separate projections may offer
more thread-level parallelism. Fused and unfused forms are tuning candidates.

The down projection remains a separate region in the MVP.

### 8.3 Numeric legality

No fusion uses `-ffast-math` by default. Reduction reassociation, reciprocal
approximations, scale precision changes, or approximate SiLU each require an
explicit numeric-mode flag and independent accuracy results.

## 9. Deferred G4 schedule space

G1 records one explicit verified schedule; G2/G3 reuse it. The following
bounded search space describes a future evidence-selected tuner and is not a
prerequisite for framework or model integration.

```rust
struct Schedule {
    m_tile: u16,
    n_tile: u16,
    k_blocks_unrolled: u8,
    vector_lanes: u8,
    loop_order: LoopOrder,
    pack: PackSpec,
    prefetch_distance: u16,
    parallel_grain_n: u32,
    fuse_kind: FuseKind,
    reduction: ReductionPlan,
    tail: TailStrategy,
}
```

### 9.1 Candidate dimensions

- output-channel tile `Ntile`;
- number of 32-weight blocks unrolled in `K`;
- activation broadcast/reload strategy;
- scale load grouping;
- scale placement and reduction structure within the numeric contract;
- accumulator count;
- panel-major packed layout;
- prefetch distance or no prefetch;
- output-channel grain passed to PyTorch's CPU parallel runtime;
- fused vs unfused supported epilogue;
- scalar cleanup vs padded/guarded tail.

`M=1` eliminates many GEMM schedules and keeps the initial search tractable.

### 9.2 Legality pruning

Before generating code, reject schedules that:

- require unsupported target features;
- exceed an architecture-specific accumulator/register budget estimate;
- misalign vector loads without an allowed unaligned path;
- produce out-of-range accesses at shape tails;
- violate quant-block boundaries;
- create overlapping parallel output ranges;
- exceed configured code-size or candidate-count budgets.

This distinction—legal schedule generation before empirical selection—is a core
compiler feature.

### 9.3 Heuristic ranking

The initial cost estimate uses:

- bytes read per output and expected weight-cache reuse;
- vector operations and conversions per quant block;
- accumulator/register estimate;
- loop/branch overhead;
- packed padding and artifact size;
- measured call/parallel-launch overhead;
- target calibration for memory bandwidth and a few primitive kernels.

It ranks candidates; it does not claim to predict exact latency.

## 10. Weight packing

Logical Q8 values are target-independent. The packer creates immutable physical
panels selected by a schedule.

The first fixed target panel is `DFQ8_B32_OI4_V1`: `T=4`, `B=ceil(K/32)`,
`P=ceil(N/4)`, and a headerless payload of exactly `P*B*144` bytes. Record
`(p,b)` starts at `(p*B+b)*144`; bytes `0..15` are four exact little-endian
scale words and bytes `16..143` are q bytes at `16+4*l+j` for logical lane `l`
and output lane `j`. Missing output lanes and physical K padding are zero, but
padding is never evaluated. The payload requirement is 16-byte alignment and
is carried by the pack specification/manifest, not by a hidden header.

Example panel concept:

```text
[Ntile output channels]
  [Kblock 0: scales for Ntile][q bytes arranged for vector loads]
  [Kblock 1: scales for Ntile][q bytes arranged for vector loads]
  ...
```

Scalar and NEON consume this same exact byte order; the payload never changes
with the target backend. A separate `PackManifestV1` carries format/schema,
logical shape and identity, packed identity, payload byte count, and required
alignment. Target schedule metadata belongs to the schedule/artifact manifest,
not to the headerless payload. Artifact parsing validates the manifest and
payload together, including their identities and byte count. Before kernel
entry, the runtime copies or maps those bytes into storage that actually meets
the manifest's alignment requirement and checks the resulting pointer.

Packing is compile time. It never occurs in the hot inference path. The compiler
reports padding and scale overhead, and the cache deduplicates identical packs.

## 11. Code generation

The Rust compiler emits inspectable C with architecture intrinsics and a stable
C ABI. The host Clang toolchain performs register allocation, instruction
selection, object generation, and linking.

### 11.1 Backends

1. **Scalar:** portable semantics oracle; vectorization explicitly disabled for
   compiler validation where necessary.
2. **ARM64 NEON (required):** uses AArch64 NEON widening/conversion and FP32
   arithmetic primitives supported by the guarded M4 target.
3. **x86-64 AVX2 (deferred):** may use AVX2 widening/conversion and FP32 vector
   arithmetic if selected as the G4 portability extension, but strict-f32
   lowering still requires separate multiply and add; no AVX-512/VNNI
   assumption on Zen 2.

The first kernels dequantize int8 weights into FP32 vectors and accumulate with
FP32 activations. This contract cannot directly use integer dot-product
instructions because one operand remains FP32. Dot-product, VNNI, or similar
claims require a future activation-quantized numeric mode with its own accuracy
and baseline results; they are not variants of `DFQ8_B32_V1`.

#### Reference SIMD dataflow

The initial vector mapping is deliberately simple and inspectable. For each
output channel and 32-weight block:

```text
for output lane n:
    out = +0
    for block b in ascending order:
        block_sum = +0
        for logical lane l in ascending order:
            product   = RN32(x[b*32+l] * float32(q[n,b,l]))
            block_sum = RN32(block_sum + product)
        scaled = RN32(block_sum * scale[n,b])
        out    = RN32(out + scaled)
```

The generated C expresses signed widening with `vmovl_s8` and `vmovl_s16`,
followed by four-lane conversion. Directly materializing the packed Q word lets
Apple Clang select the corresponding `sshll.8h -> sshll.4s -> scvtf.4s`
sequence rather than a longer byte-reconstruction/sign-extension path. The
four little-endian scale words are loaded as one raw `uint8x16_t` and
bit-reinterpreted as `float32x4_t`; this preserves their bits without a
temporary stack array or stack-canary failure path.

The fixed G1 Loop IR retains one `K` partial accumulator and the same logical
block/lane recurrence. It does not use FMA, horizontal reduction,
reassociation, or a second `K` accumulator. Four output lanes map to one panel
record; a 128-bit store completes a full tile and scalar cleanup handles an
`N` tail. The assembly audit confirms that contract in the selected machine
instructions rather than assuming it from the intrinsic names.

The deferred AVX2 design is a target extension only. If selected for G4 it must
implement the same separate RN32 multiply/add recurrence; a fused multiply-add
is not a strict-f32 schedule and is not implied by this G1 contract.

The packer stores `K` tails in full 32-lane records with zero q bytes, but the
logical evaluator and every generated kernel must never evaluate padded lanes.
Logical `K`, padded `K`, and byte bounds remain in the manifest and guards.
`N` tails use scalar cleanup and never permit a store beyond the logical output
tensor.

### 11.2 Generated function ABI

The canonical call descriptor and function declarations live in
[`include/decodeforge/abi_v1.h`](../include/decodeforge/abi_v1.h); generated
code and ABI checks must include that header rather than copying a pseudocode
struct into this document.

The first scalar generated-module contract, including its frozen status values,
72-byte artifact-ID C string, exact `M=1` shape/stride rules, OI4 byte count,
16-byte packed-data alignment, deterministic guard order, and whole-output
failure behavior, is specified in [ADR 0003](decisions/0003-scalar-generated-abi.md).
Generated inner loops can rely on the assumptions proven by those guards; the
ABI does not prove buffer extents, aliasing, or packed-weight identity.

### 11.3 Compilation

The build command is explicit and captured in the manifest. Default mode uses
optimization without fast-math. Target modes:

- portable scalar;
- explicit ARM64 feature set;
- explicit x86-64 AVX2 (deferred optional G4 extension; strict-f32 kernels must
  still use separate multiply and add);
- optional host-native artifact, labeled non-portable.

macOS emits a `.dylib`; Linux emits a `.so`. Generated source is retained in a
debug artifact and hash-addressed in normal caches.

The implemented Apple native checkpoint has backend-specific fixed Clang/link
policies for scalar and NEON modules. Both invoke Apple Clang through
`/usr/bin/xcrun` with a closed argument and environment policy, a pinned
SDK/developer directory, strict floating-point flags, disabled
auto-vectorization, explicit exports, and fatal linker warnings. Accepted NEON
work must originate from explicit generated intrinsics and survive the
shape-aware machine-code audit. The bounded runner contains the complete tool
process group and drains output without detached threads.

Compiler output is opened once with no-follow/nonblocking semantics and must be
one bounded, owner-controlled regular file. Those bytes are copied into a
second owner-only directory, then parsed as one little-endian ARM64-all
`MH_DYLIB` with the fixed ID, macOS 15 deployment target, one dyld-required
UUID, only the direct `libSystem` dependency, exactly three ordinary text
exports, the expected local text helper, and no initializer/interposition or
rpath surface. `llvm-objdump` audits that retained copy; the held descriptor,
path identity, metadata, and bytes are revalidated after disassembly. The
artifact owner exposes immutable bytes and hashes, not a public mutable path.

The backend-neutral checked runtime consumes either unforgeable compiler owner
and revalidates the region, schedule, pack, shape, and module identity before
entering its documented unsafe boundary. It copies the exact audited image
into another owner-only directory, closes the writable descriptor, retains a
read-only descriptor, and loads through `/dev/fd` so pathname replacement
cannot redirect the image. Because dyld applies launch-time search overrides
even to paths containing a slash, the runtime rejects every visible `DYLD_*`
variable before loading. It uses eager, local, first-image symbol lookup and
rechecks pathname identity, mode `0400`, and every byte after `dlopen`. Its ABI
version, fixed-length module ID, input extent, pack extent/alignment, status,
and complete finite output are checked. A failed call exposes no partial
output. Tests execute both backends across all 16 frozen fixtures bit-exactly.
Dedicated `N=4` and `N=5` builds prove vector-only and vector-plus-tail paths,
while mismatch, ownership, floating-point-environment, and interleaved-module
tests exercise the guarded loading boundary.

The safe compiler entrypoint assumes an ordinary safe process: no hostile
same-UID actor changes the retained inode in place, and no C or unsafe code
erases a launch-time `DYLD_*` variable after dyld has cached it. Those stronger
process-compromise cases require code-signature or mapped-image attestation and
are outside the G1 threat model. A launch-time subprocess regression and a
pathname-substitution regression enforce the supported boundary.

### 11.4 Why C/intrinsics first

This project is about LLM schedule, packing, and retargetable code generation.
Using Clang for final machine instruction selection provides native code without
first taking on MLIR/LLVM build integration. An MLIR backend is a legitimate
later comparison only after the core compiler works; it is not required to make
the compiler real.

### 11.5 Machine-code audit

Clang remains responsible for instruction selection and register allocation,
but DecodeForge verifies the result rather than assuming the intrinsics produced
the intended loop. Each published generated kernel's evidence bundle includes an
`objdump` or `llvm-objdump` listing and a short audit of:

- the hot-loop boundaries and vector width;
- widening/conversion, separate multiply and add, load, and prefetch
  instructions actually emitted; a horizontal reduction or fused multiply-add
  is a contract failure for this strict-f32 slice;
- loop branches, tail handling, and unexpected scalarization;
- stack-frame size and accumulator spills;
- alignment assumptions visible in the generated loads;
- code size and compiler version/flags.

For the fixed G1 NEON checkpoint, the audit is shape-aware: `N=4` requires a
complete vector path with no scalar floating-point recurrence, while `N=5`
also requires a scalar tail and store. It checks register-connected signed
widening, conversion, lane-form activation multiply, separate scale/accumulator
arithmetic, and a 128-bit output store. It rejects fused, dot-product, matrix,
or horizontal arithmetic; calls; indirect or out-of-range branches; malformed
returns; and unexpected scalarization. These checks establish code-shape
correctness, not performance.

The audit does not attempt to reconstruct full general-purpose-register pointer
provenance from arbitrary AArch64. The exact verified generated source, closed
compiler invocation, and retained byte-identical snapshot establish the packed
address boundary. Within that boundary, the audit requires the vector load to
dominate its scale multiply and rejects any intervening use or redefinition of
the loaded SIMD register, including secondary destinations of paired loads.

If schedule search is selected for G4, at least one rejected or losing M4
schedule is audited far enough to connect a concrete machine-code difference—
such as a spill, extra shuffle, or larger tail—to its measured result. If AVX2
is selected for G4, the same audit applies to that extension. Hand-written
assembly is not required; understanding the emitted assembly is.

## 12. Deferred G4 parallel execution

DecodeForge does not implement a scheduler. G1–G3 use the existing
single-threaded generated-call contract. If multicore output-channel execution
is selected for G4, an integration layer reuses PyTorch's existing intra-op CPU
runtime to partition disjoint ranges; generated kernels never start threads.

Tests run:

- one thread, which isolates code generation and packing;
- physical-core sweeps;
- M4 worker-count sweeps because performance and efficiency cores differ.

Ryzen SMT and other second-host measurements are deferred to G4 if AVX2
portability is selected. Nested parallelism is disabled.

## 13. Native bridge and callable

The merged Rust `decodeforge-bridge` cdylib exports the six functions frozen in
[`include/decodeforge/runtime_v1.h`](../include/decodeforge/runtime_v1.h): ABI
version, create NEON handle, run, query descriptor, destroy, and thread-local
last error. This bridge ABI is versioned independently from the generated-module
ABI.

Create parses a bounded canonical `PackManifestV1`, copies and verifies the
exact OI4 payload, builds/audits/loads the fixed NEON module, and returns one
unforgeable process-local handle. The registry limits one pack to 128 MiB, all
live packs to 2 GiB, and live entries to 256. One build is admitted at a time;
concurrent runs are supported; destroy linearizes against in-flight work. Every
export contains Rust panics at the FFI boundary, uses a closed status set, and
records bounded printable thread-local diagnostics. A failed descriptor query
zeros its output.

The external bridge test builds the actual release cdylib/so. On Apple ARM64 it
passes real CPU float32 Torch buffers for the frozen `N=255,K=2` fixture and
requires all 255 result words to match. Other supported CI hosts verify the
explicit unsupported-host status rather than pretending to execute NEON.

The G2 Python layer is deliberately thin:

- it requires a caller-supplied SHA-256 library identity, rejects symlinks and
  non-regular/oversized inputs, and loads a private immutable snapshot of the
  verified bytes;
- `ctypes` signatures exactly mirror the frozen header;
- owned binding objects synchronize run/close and expose immutable descriptors
  and completed-call counters;
- a `torch.library` eager operator allocates the output and passes input/output
  `data_ptr()` values directly to the bridge;
- Python owns dispatch guards and same-Q8 fallback policy, not code-generation
  optimization.

G3 adds an owning `nn.Module` adapter over that low-level callable. A C++ ATen
extension, fake/meta implementation, and compiled-graph frontend are not needed
for G2/G3 and remain possible later work.

## 14. Guards, cache, and failure behavior

### 14.1 Binding identity

```text
Region/Loop IR and generated-module identity
+ logical constant-weight hash
+ logical Q8 format/version
+ exact static dimensions and strides
+ target triple and CPU features
+ fixed schedule and pack identity/version
+ numeric mode
+ compiler Git/version
+ generated-module ABI and native bridge ABI
```

The bridge descriptor exposes `N`, `K`, module ID, packed-weight ID, and byte
extent. The Python binding and G3 adapter must cross-check those values against
their asset inventory and same-Q8 fallback before model mutation.

### 14.2 Guard miss

For the G2 low-level callable, policy is explicit: invoke a caller-supplied
same-Q8 fallback or raise. For the G3 adapter, prompt `M>1` and other ordinary
eligibility misses use its identity-bound same-Q8 fallback. Native entry never
occurs outside the binding's assumptions, and a failure after native entry is
always raised rather than hidden by fallback.

### 14.3 Cache states

G2 uses a bounded process-local handle registry and does not require a persistent
compiler cache. G3 prepares immutable manifest/payload assets atomically and
rebuilds native handles during controlled setup. If persistent caching becomes a
G4 priority, writes are temporary-file-plus-rename, locked per content key, and
checksummed; corrupt or ABI-incompatible entries are rejected and rebuilt.

## 15. Deferred G4 autotuner

The fixed verified G1 schedule is sufficient for G2/G3. This section specifies
a future tuner if G3 evidence makes schedule selection the highest-value next
extension.

The tuner is offline or first-use opt-in; production callable execution never
launches surprise tuning.

Pipeline:

```text
enumerate -> legality prune -> heuristic rank -> generate top K
          -> correctness smoke test -> warm -> randomized measurements
          -> robust aggregate -> select -> validate -> cache
```

### 15.1 Measurement

- use fixed input buffers and touch outputs to prevent elimination;
- warm code and packed weights separately;
- randomize candidate order across rounds;
- use enough inner iterations to exceed timer noise;
- report median, dispersion, and tail, not only best sample;
- reject outlier-corrupted or thermally drifting sessions rather than cherry-pick;
- validate output before a candidate can win;
- include tuning/compile time and break-even call count.

Objective starts with single-call median latency at the target shape. Secondary
objectives are p95/p99, code size, and pack size. Multi-objective choices remain
visible; one metric never silently hides another.

### 15.2 Overfitting controls

- benchmark real TinyLlama shapes and a held-out synthetic shape suite on the
  M4;
- repeat on a second host only if an AVX2 portability extension is selected for
  G4;
- keep a simple heuristic schedule as a non-tuned baseline;
- do not use test-run samples to claim an independent predictive cost model;
- record every evaluated candidate, including losing schedules.

### 15.3 Hardware evidence and bandwidth model

Wall-clock latency is the primary portable metric. When the host exposes them,
the run also records cycles, instructions, branches/branch misses, cache-load
events, and other stable counters relevant to the tested CPU. Linux runs use
`perf stat` with the exact event list captured; macOS runs record the selected
Instruments or `xctrace` template and available counters. Missing or multiplexed
counters are labeled and never silently converted into exact values.

Each host is calibrated with sustained memory-bandwidth and call-overhead tests.
For `M=1`, the report estimates bytes read per output, operations per byte, and
the fraction of calibrated bandwidth achieved. The estimate is explicitly a
model; it is compared with counter evidence and timings before describing a
kernel as bandwidth-bound, compute-bound, or latency-bound.

The benchmark manifest records, where observable:

- exact CPU model, OS, compiler, target features, and thread-affinity policy;
- worker count, warmup, iteration count, candidate randomization seed, and input
  seed;
- power mode and evidence of frequency or thermal drift;
- raw samples and counter availability, not only aggregates.

## 16. Correctness and numeric validation

### 16.1 Oracles

1. Python quantizer/dequantize + PyTorch FP32 matmul;
2. Rust scalar Q8 implementation;
3. generated scalar C;
4. generated NEON candidate through the prepared-call and release bridge;
5. eager PyTorch operator through the actual release library;
6. G3 hybrid native query projections vs all-same-Q8-fallback generation;
7. fused vs materialized Q8 graph only if fusion is selected for G4.

Each level is compared before end-to-end integration.

### 16.2 Test cases

- all-zero, constant, alternating-sign, and random weights;
- zero and extreme but finite activations;
- `K` exactly/above/below block boundaries;
- `N` exactly/above/below vector/tile boundaries;
- non-multiple tails and padded lanes;
- the required `[2048,2048]` TinyLlama query-projection shape and 22 distinct
  layer weight identities;
- random small shapes suitable for exhaustive scalar checking;
- alignment and deliberately unaligned rejected/fallback paths;
- G2 library hash/snapshot, eager guard, counter, error, and lifecycle cases;
- G3 exact-module inventory, transactional replacement, prefill fallback,
  cached native coverage, and repeated setup/teardown;
- fused RMSNorm and SwiGLU cases only when those G4 features are selected.

### 16.3 Metrics

- maximum absolute and relative error;
- mean squared error and cosine similarity;
- top-k/logit agreement at model integration;
- greedy-token agreement over fixed prompts;
- perplexity delta on a pinned small evaluation slice, if full Q8 replacement is
  used for claims.

Schedule variants using the same Q8 semantics should differ only within the
declared FP32 reduction tolerance. Quantization error is reported separately.

## 17. Evidence bundle and compiler report

Every published claim has a self-contained result directory with a
machine-readable manifest and generated Markdown report. G1 retains canonical
Region/Loop IR, the fixed schedule, pack metadata, complete generated source,
disassembly and audit, build command, guards/ABI/features, correctness, raw
samples, analysis, and host/tool/source provenance.

G3 adds:

1. pinned model/tokenizer revision and hashes;
2. fixed prompt bytes, tokenized IDs, decode settings, and output IDs/text;
3. ordered 22-entry q-projection asset inventory with source/Q8/pack/module
   identities and extents;
4. eager operator and adapter guard/lifecycle configuration;
5. per-adapter native/fallback/error/in-flight coverage;
6. direct operator and model-level correctness deltas against all-same-Q8
   fallback;
7. raw preparation, startup, prefill, per-token decode, total, and memory
   measurements;
8. a verifier-generated summary that makes no model-speed claim unless its own
   measurements support one.

The optional visualizer is a compiler artifact viewer, not a live inference
dashboard. It must render the same checked-in manifest and must not become the
only way to inspect a result.

## 18. Repository components

| Component | Responsibility |
|---|---|
| `decodeforge-core` | G0 DFQ8 semantics, reference quantizer/evaluator, identities, fixture gates |
| `decodeforge-compiler` | G1 verification, lowering, OI4 packing, scalar/NEON source generation, Apple artifact construction, and shape-aware disassembly audit |
| `decodeforge-runtime` | generated-module ownership, validation, guarded dynamic loading, and prepared calls |
| `decodeforge-bridge` | versioned C ABI, bounded handle/pack ownership, lifecycle synchronization, panic/status boundary |
| Python package | lazy bridge loading, eager operator, binding registry/counters, same-Q8 model adapter |
| benchmarks | correctness, microkernels, projection/layer/model integration |
| results | manifests, raw samples, generated source, assembly, and reports |
| dashboard | optional post-G3 compiler-report rendering |

The target-independent runtime crate cannot depend on the compiler pipeline.
The bridge is an explicit orchestration boundary that can ask the compiler to
build the fixed artifact during handle creation. Persistent caching and schedule
enumeration are not prerequisites for G2/G3.

## 19. Safety and security

- generated/native modules are local compiler outputs, not accepted over the
  network in the MVP;
- constant sizes/offsets use checked arithmetic and checksums;
- compiler invokes the toolchain without shell interpolation;
- asset names derive from bounded layer indices/content identities, not raw
  model/user-provided paths;
- the Python wrapper hashes a bounded non-symlink regular bridge library and
  loads an owner-only private snapshot of the verified bytes;
- native handles are CSPRNG-derived, process-local, quota-bounded, and never
  persisted or accepted from model data;
- C ABI boundaries validate all pointer-related assumptions before entry;
- unsafe Rust is isolated to dynamic loading/FFI, with ownership documented;
- native code never writes outside disjoint guarded output/scratch ranges;
- large source weights and packed payloads are not embedded in reports; the
  pinned public demonstration prompt and generated text may be retained.

## 20. Main risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Scope expands to full model compiler | project never finishes | G3 replaces only the 22 same-shaped query projections |
| `torch.compile`/FX integration dominates | visible model proof is delayed | use the guarded eager operator first; defer graph capture to G4 |
| Q8 format makes comparison unfair | speedup is precision change | compare schedules against same Q8 scalar semantics; quality separately |
| G1 `~3.96x` is mistaken for model speedup | misleading résumé claim | name the prepared-call scalar boundary beside every number; measure G3 separately |
| Same-Q8 fallback accidentally uses FP32 source weight | correctness and timing attribution fail | bind fallback and native descriptor to one checked pack identity |
| Partial 22-layer replacement | untracked mixed semantics | validate all assets/modules first, install transactionally, and roll back on failure |
| Native path silently falls back | false coverage/performance claim | per-adapter completion/error counters and hard failure after native entry |
| NEON dequant kernel is not competitive end to end | weak speed headline | publish compiler, machine-code, ABI, and coverage evidence; report neutral/negative model result honestly |
| Future tuner overfits one CPU/shape | weak generality | held-out shapes and a fixed heuristic baseline when schedule search is selected for G4 |
| M4 thermal drift | misleading winner | randomized rounds and thermal/run-order reporting |
| Generated code relies on host-native flags | artifact crashes elsewhere | exact feature guards and portable fallback |
| Documentation outpaces implementation | impressive plan but weak résumé evidence | promotion gates and checked-in result bundles; no “built” claim before proof |
| Intrinsics compile into scalar or spill-heavy code | low-level claim is superficial | retain the existing shape-aware disassembly audit for every published kernel |

## 21. Settled baseline decisions

- project name: DecodeForge;
- compiler focus: frozen Q8 query projections for cached decode in G0–G3;
- explicit quantization and canonical Rust OI4 packing before model integration;
- TinyLlama 1.1B supplies one required real shape repeated across 22 layers;
- Rust compiler/bridge, generated C/intrinsics, host Clang, and a thin lazy
  Python eager binding;
- scalar → ARM64 NEON order for the required path; x86 AVX2 is deferred to G4;
- one thread before multi-core scaling;
- Transformers/PyTorch owns tokenizer, attention, KV cache, sampling, and
  unsupported model operations;
- same-Q8 fallback for prompt prefill and native execution only for guarded
  cached `M=1` calls;
- no KV paging, HTTP server, work stealing, GPU backend, Q4, or generic MLIR
  frontend in the first project;
- generated-source backend before any MLIR experiment;
- no integer dot-product claim for the FP32-activation `DFQ8_B32_V1` path;
- checked-in source, disassembly, raw measurements, and manifests are required
  evidence, not optional polish;
- schedule selection, general FX/`torch.compile`, all-155-linear coverage,
  dashboard, both fusions, multicore tuning, native small-batch support, and
  AVX2 are locked behind G3 and an evidence-selected G4 extension;
- performance numbers appear only as checked-in completed evidence with their
  exact boundaries; no unmeasured end-to-end target is promised.

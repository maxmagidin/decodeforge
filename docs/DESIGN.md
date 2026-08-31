# DecodeForge design and technical specification

**Status:** G0 is complete with checked-in provenance evidence. The fixed G1
native-correctness checkpoint now includes verified lowering, shared OI4
packing, deterministic scalar and strict output-vector NEON source, audited
Apple-arm64 scalar/NEON dylibs, and checked loading through the frozen ABI.
Both backends execute all 16 frozen fixtures bit-exactly; dedicated `N=4` and
`N=5` cases verify vector-only and vector-plus-tail machine code. No timing,
speedup, real-shape benchmark, or completed-G1 claim exists yet. The normative
contract is [Q8_FORMAT_V1](Q8_FORMAT_V1.md).

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
3. Enumerate only legal schedules, emit target-specific scalar/NEON code for the
   required M4 path, benchmark a bounded candidate set, and cache the result.
   Keep an x86-64 AVX2 lowering as a deferred extension point.
4. Integrate compiled regions through a registered `torch.compile` backend and a
   thin PyTorch CPU bridge.
5. Demonstrate real projection shapes from TinyLlama 1.1B on the Apple M4;
   evaluate another CPU only if it is selected as a later extension.
6. Attribute performance to schedule, packing, and fusion choices rather than to
   a simultaneous server/scheduler/KV change.
7. Produce an inspectable compiler report: normalized graph, IR, rejected and
   selected schedules, generated source, packed layout, guards, code size, build
   time, tuning samples, and runtime metrics.
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
| G0: semantics | `DFQ8_B32_V1` Python and Rust scalar semantics, fixtures, and schema agree, followed by a checked-in provenance/evidence bundle |
| G1: M4 vertical slice | One real TinyLlama shape flows through minimal IR to generated scalar and ARM64 NEON code on the M4; the bundle contains source, assembly, correctness, timings, and host metadata |
| G2: Mac schedule evidence | Bounded schedule selection on the M4 is correctness-gated, reproducible, and measured |
| G3: Mac PyTorch slice | A guarded `torch.compile` region runs end to end on the M4, demonstrates a guard miss, and reports supported-region coverage |
| G4: evidence-selected extension | One measured extension—AVX2 portability, fusion, small batch, or multicore—wins or yields an honest negative result under unchanged semantics |

Work that belongs to a later gate is kept out of the critical path. In
particular, the visual dashboard, predictive cost-model claims, both fusions,
multi-core tuning, `M > 1`, and x86-64 work cannot delay G0–G3.

## 3. Compiler/runtime boundary

PyTorch owns everything outside a supported region:

```text
Transformers model
  | tokenization, attention, KV cache, sampling, unsupported operators
  v
TorchDynamo / FX graph
  |
  +--> unsupported partitions ------------------------+
  |                                                   |
  +--> DecodeForge-supported frozen linear regions    |
          |                                            |
          v                                            |
       native compiled module                          |
          |                                            |
          +---------------- output tensors ------------+
```

This boundary prevents the compiler project from silently becoming an inference
engine. The final end-to-end demo can generate text, but text generation is a
consumer of the compiler, not code the project reimplements.

## 4. Frontend contract

PyTorch's custom backend contract is:

```python
backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable
```

DecodeForge registers the backend name `decodeforge`. The backend:

1. verifies inference mode and CPU tensors;
2. normalizes supported FX/ATen forms;
3. propagates tensor metadata from example/fake inputs;
4. identifies maximal supported regions;
5. lowers each region into DecodeForge IR;
6. compiles or retrieves a guarded native module;
7. returns a callable that executes supported regions through the native bridge
   and unsupported regions through the preserved FX graph.

### 4.1 Quantization is explicit

A compiler backend should not quietly change FP32 model semantics. Weight-only
Q8 is therefore an explicit model transformation:

```python
model = decodeforge.quantize(model, format="DFQ8_B32_V1")
compiled = torch.compile(model, backend="decodeforge", fullgraph=False)
```

`decodeforge.quantize` replaces eligible frozen `nn.Linear` modules with a
logical `decodeforge.q8_linear` operator and stores Q8 constants. A scalar
dequantize-and-dot implementation defines that operator's reference semantics.
The compiler optimizes the already-quantized graph; it is expected to agree with
the Q8 reference within the declared floating-point reduction tolerance.

A tiny FP32 bridge fixture may be used to validate pointer, shape, and ownership
behavior, but a complete FP32 compiler path is not a prerequisite. Q8 compiler
correctness is established against the explicit Q8 reference contract.

### 4.2 Supported region shapes

MVP:

```text
df.q8_linear(x, Wq, scales)
```

Then:

```text
rms_norm(x, gamma, eps) -> df.q8_linear(...)

gate = df.q8_linear(x, W_gate)
up   = df.q8_linear(x, W_up)
out  = silu(gate) * up
```

Required decode shape is `M=1`. Later small-batch values are `M ∈ {2,4,8}`.
`K` and `N` are static per compiled weight. Dynamic unsupported dimensions cause
a guard miss and recompile or fallback according to configuration.

The first complete result supports only `df.q8_linear`. RMSNorm fusion, paired
gate/up fusion, and `M > 1` are promotion-gated extensions, not parallel MVP
workstreams.

### 4.3 Rejected cases

- weights require gradients or can mutate;
- non-CPU tensors;
- unsupported dtype, stride, alias, or rank;
- output used by an in-place op the partitioner cannot prove safe;
- symbolic dimensions without a legal guarded fallback;
- non-contiguous activations before a copy cost is explicitly modeled;
- numeric modes not requested by the user.

The compiler emits a rejection reason for every candidate region.

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

Required projection families:

| Region | Logical matrix shape `[N, K]` | Repetitions/layer |
|---|---:|---:|
| Q projection | `[2048, 2048]` | 1 |
| K projection | `[256, 2048]` | 1 |
| V projection | `[256, 2048]` | 1 |
| attention output | `[2048, 2048]` | 1 |
| gate projection | `[5632, 2048]` | 1 |
| up projection | `[5632, 2048]` | 1 |
| down projection | `[2048, 5632]` | 1 |
| language-model head | `[32000, 2048]` | 1/model |

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

## 8. Canonicalization and fusion

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

## 9. Schedule space

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
the intended loop. Each selected schedule's evidence bundle includes an
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

At least one rejected or losing M4 schedule is audited far enough to connect a
concrete machine-code difference—such as a spill, extra shuffle, or larger tail—
to its measured result. If AVX2 is selected for G4, the same audit applies to
that extension. Hand-written assembly is not required; understanding the
emitted assembly is.

## 12. Parallel execution

DecodeForge does not implement a scheduler. The PyTorch C++ bridge uses the
existing intra-op CPU runtime to partition disjoint output-channel ranges. The
compiled schedule supplies the grain size and range kernel.

Tests run:

- one thread, which isolates code generation and packing;
- physical-core sweeps;
- M4 worker-count sweeps because performance and efficiency cores differ.

Ryzen SMT and other second-host measurements are deferred to G4 if AVX2
portability is selected. Nested parallelism is disabled. Generated kernels
never start threads.

## 13. Native bridge and callable

A thin C++ extension:

- validates CPU device, dtype, rank, shape, stride, and alignment;
- obtains `at::Tensor` data pointers;
- allocates outputs with PyTorch;
- calls the selected native module directly or through `at::parallel_for`;
- releases the Python GIL where the PyTorch extension API permits;
- converts nonzero kernel status into a structured exception;
- owns no compiler optimization policy.

A fake/meta implementation supplies shapes for Dynamo/export tracing. The
bridge ABI is versioned independently from compiler artifacts.

## 14. Guards, cache, and failure behavior

### 14.1 Guard key

```text
Region IR fingerprint
+ constant-weight hash
+ logical Q8 format/version
+ exact static dimensions and strides
+ target triple and CPU features
+ schedule and pack version
+ numeric mode
+ compiler Git/version
+ native bridge ABI
```

### 14.2 Guard miss

Policy is configurable and visible:

- compile a new supported specialization;
- execute the scalar Q8 reference;
- fall back to preserved FX/PyTorch partition;
- fail in strict benchmark mode.

Never call a kernel on a shape or CPU outside its assumptions.

### 14.3 Cache states

Cache writes are atomic (temporary file then rename), locked per key, and
checksummed. Corrupt or ABI-incompatible entries are ignored and rebuilt. A
cache entry contains IR, schedule, source, native module, packed constants,
manifest, and optional tuning evidence.

## 15. Autotuner

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
4. generated NEON candidate (and AVX2 only if selected for G4);
5. fused vs materialized Q8 graph.

Each level is compared before end-to-end integration.

### 16.2 Test cases

- all-zero, constant, alternating-sign, and random weights;
- zero and extreme but finite activations;
- `K` exactly/above/below block boundaries;
- `N` exactly/above/below vector/tile boundaries;
- non-multiple tails and padded lanes;
- every required TinyLlama projection shape;
- random small shapes suitable for exhaustive scalar checking;
- alignment and deliberately unaligned rejected/fallback paths;
- fused RMSNorm and SwiGLU positive/negative pattern cases.

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

Every benchmarked compilation emits a self-contained result directory with a
machine-readable manifest. A Markdown report is required; a standalone HTML
viewer is optional after G3. The bundle contains:

1. captured FX region and support/rejection decisions;
2. canonical Region IR and lowered Loop IR;
3. legal schedule count and pruning reasons;
4. top candidates with parameters and cost estimates;
5. target packed layout diagram and size overhead;
6. complete generated source, disassembly, machine-code audit, and build command;
7. guards, cache key, ABI, and required CPU features;
8. raw samples, available counters, tuning distributions, and selected schedule;
9. scalar/vector/fused correctness deltas;
10. kernel and layer benchmark comparison.

The optional visualizer is a compiler artifact viewer, not a live inference
dashboard. It must render the same checked-in manifest and must not become the
only way to inspect a result.

## 18. Repository components

| Component | Responsibility |
|---|---|
| `decodeforge-core` | G0 DFQ8 semantics, reference quantizer/evaluator, identities, fixture gates |
| `decodeforge-compiler` | G1 verification, lowering, OI4 packing, scalar/NEON source generation, Apple artifact construction, and shape-aware disassembly audit |
| `decodeforge-runtime` | cache, artifact validation, guards, dynamic loading |
| Python package | model transformation, backend registration, FX normalization/partition |
| native bridge | ATen tensors, output allocation, PyTorch CPU parallel runtime, ABI |
| benchmarks | correctness, microkernels, projection/layer/model integration |
| results | manifests, raw samples, generated source, assembly, and reports |
| dashboard | optional post-G3 compiler-report rendering |

The runtime crate cannot depend on the compiler pipeline. A cached artifact is
usable without schedule enumeration or code generation.

## 19. Safety and security

- generated/native modules are local compiler outputs, not accepted over the
  network in the MVP;
- constant sizes/offsets use checked arithmetic and checksums;
- compiler invokes the toolchain without shell interpolation;
- cache filenames derive from hashes, not graph/user-provided names;
- dynamic libraries are loaded only from the configured cache root;
- C ABI boundaries validate all pointer-related assumptions before entry;
- unsafe Rust is isolated to dynamic loading/FFI, with ownership documented;
- native code never writes outside disjoint guarded output/scratch ranges;
- source model prompts/data are not embedded in compiler reports.

## 20. Main risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Scope expands to full model compiler | project never finishes | support only frozen Q8 linear regions and two fusions |
| `torch.compile` integration dominates | little codegen progress | standalone IR/compiler harness precedes frontend |
| Q8 format makes comparison unfair | speedup is precision change | compare schedules against same Q8 scalar semantics; quality separately |
| AVX2/NEON dequant kernel is not competitive | weak headline | result can focus on compiler/tuning insights; use contextual ceilings honestly |
| Tuner overfits one CPU/shape | no portability | two architectures, held-out shapes, heuristic baseline |
| M4 thermal drift | misleading winner | randomized rounds and thermal/run-order reporting |
| Fused gate/up spills registers | regression | fused and unfused are candidates, not doctrine |
| PyTorch fallback hides unsupported work | inflated end-to-end claim | report compiled-region coverage and kernel-only/layer/full metrics separately |
| Generated code relies on host-native flags | artifact crashes elsewhere | exact feature guards and portable fallback |
| Documentation outpaces implementation | impressive plan but weak résumé evidence | promotion gates and checked-in result bundles; no “built” claim before proof |
| Intrinsics compile into scalar or spill-heavy code | low-level claim is superficial | mandatory disassembly audit for selected schedules and representative losers |

## 21. Settled baseline decisions

- project name: DecodeForge;
- compiler focus: frozen Q8 linear regions for decode;
- explicit quantized model transformation before backend compilation;
- TinyLlama 1.1B supplies required real shapes;
- Rust compiler, generated C/intrinsics, host Clang, thin C++ ATen bridge;
- scalar → ARM64 NEON order for the required path; x86 AVX2 is deferred to G4;
- one thread before multi-core scaling;
- PyTorch CPU runtime supplies parallel ranges; no custom scheduler;
- no KV paging, HTTP server, work stealing, GPU backend, Q4, or generic MLIR
  frontend in the first project;
- generated-source backend before any MLIR experiment;
- no integer dot-product claim for the FP32-activation `DFQ8_B32_V1` path;
- checked-in source, disassembly, raw measurements, and manifests are required
  evidence, not optional polish;
- dashboard, both fusions, multi-core tuning, small-batch support, and AVX2 are
  locked behind the Mac scalar/NEON path and an evidence-selected G4 extension;
- no performance number written into the design.

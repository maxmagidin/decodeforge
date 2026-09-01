# DecodeForge benchmark and experimental methodology

**Rule:** a benchmark may support a claim only when it compares the intended
variable, preserves the declared numeric contract, and produces a reproducible
result bundle.

**Current status:** G0 and G1 are complete with checked-in evidence. The fixed
G1 scalar/NEON path passes all 16 frozen fixtures bit-exactly on the M4.
Dedicated `N=4` and `N=5` tests establish vector-only and
vector-plus-tail execution, and retained disassembly verifies signed widening,
conversion, lane-form activation multiply, separate scale/accumulator
arithmetic, raw vector scale loading, and guarded stores. On the real
TinyLlama `M=1, N=K=2048` projection, the three checked-in sessions measured
paired scalar/NEON speedups of `3.95671x`, `3.96176x`, and `3.95648x`; all 95%
paired-BCa lower bounds exceed `3.95x`. The declared speedup gate passes. This
claim covers the complete prepared-call kernel boundary only. Ryzen/AVX2
remains deferred to optional G4 work.

## Claim classes

DecodeForge keeps four kinds of claims separate:

1. **Compiler correctness:** generated code implements the `DFQ8_B32_V1` contract.
2. **Schedule improvement:** one legal schedule outperforms another under the
   same format, inputs, host, and callable boundary.
3. **Framework integration:** a supported PyTorch region executes through the
   guarded artifact without hidden fallback.
4. **Contextual performance:** DecodeForge is compared with PyTorch, Inductor,
   llama.cpp, or another library even when precision, packing, or boundaries
   differ.

Only class 2 isolates the compiler's optimization contribution. Class 4 is
useful context but cannot be presented as an apples-to-apples speedup unless the
format, inputs, thread count, operator boundary, and output semantics match.

## Required baselines

| Baseline | Purpose | Claim permitted |
|---|---|---|
| Python dequantize plus PyTorch FP32 matmul | numeric oracle | quantization/correctness only |
| Rust scalar Q8 | cross-language semantic oracle | correctness only |
| generated scalar C with vectorization disabled | codegen and ABI baseline | vector-vs-scalar speedup |
| designated untuned vector schedule | schedule baseline | tuner improvement |
| materialized Q8 graph | fusion baseline | bytes/dispatches removed and fusion effect |
| PyTorch/Inductor at its native dtype | ecosystem context | contextual latency/throughput only |
| llama.cpp or vendor kernel with different format | practical ceiling/context | labeled non-equivalent comparison only |

Compiler flags for the generated scalar baseline must prevent accidental
auto-vectorization; the build log and disassembly verify this.

## Shape suites

### Required real shapes on the M4

All use `M=1` first:

- `[N,K] = [2048,2048]`;
- `[256,2048]`;
- `[5632,2048]`;
- `[2048,5632]`;
- `[32000,2048]` after the smaller projections are stable.

The Ryzen 5 3600 shape suite is optional G4 evidence if AVX2 portability is
selected; it is not a G0–G3 requirement.

### Held-out shapes

A small synthetic suite varies `N` around tile boundaries and `K` around the
32-weight block boundary. Held-out shapes are fixed before tuning results are
examined. They test whether legality and heuristic ranking generalize; they are
not additional search data.

### Correctness corpus

- zero, constant, alternating-sign, and seeded random weights;
- zero, small, large, and mixed-sign finite activations;
- `K` below, equal to, and above block/tile boundaries;
- `N` below, equal to, and above vector/tile boundaries;
- aligned and intentionally unsupported alignment/stride cases;
- invalid feature, shape, artifact version, checksum, and guard cases.

## Numeric policy

Quantization error and schedule error are different experiments:

- Source-vs-Q8 quality compares source FP32/BF16 weight words with
  `dequantize_f32_bits` output. For nonzero-scale, non-clamped lanes, the
  deterministic V1 evidence uses the conservative per-weight bound
  `scale * (0.5 + 255*u) + 2^-150`, where `u = 2^-24`, with exact rational
  arithmetic in the tests. Zero blocks, padding, and the intentional
  subnormal-clamp case are reported separately.
- Generated scalar and NEON schedules are compared with the internally computed
  canonical Q8 output to test compiler correctness. A future AVX2 candidate is
  compared the same way only if selected as G4.
- Fusion is compared with the materialized Q8 graph.

The source-vs-Q8 report is a quantization-quality result, not the generated-
kernel comparator and not evidence that a native kernel has passed.

Reports include maximum absolute and relative error, mean squared error, and
cosine similarity. Model-level reports add top-k/logit agreement, fixed-prompt
greedy-token agreement, and a pinned perplexity slice when claims cover the
whole quantized path. The tolerance and its rationale are versioned before
candidate timing.

## Timing protocol

The generated-kernel timing boundary uses a prepared safe call with borrowed
input and caller-owned output storage. Each measured invocation includes the
complete `0x7fc0_0000` quiet-NaN sentinel fill, the native `df_run_v1` ABI
call, status decoding, and the complete finite-output scan. Compilation,
dynamic loading, weight packing, and input/output allocation happen before
measurement. Reports must name this checked boundary; they must not relabel it
as raw kernel-only latency.

Each run:

1. pins the source, model revision, inputs, schedule candidates, compiler, flags,
   CPU features, numeric mode, and random seeds in the manifest;
2. allocates inputs and outputs before timed regions;
3. touches outputs so work cannot be eliminated;
4. warms module loading, code, and packed weights separately;
5. calibrates inner repetitions so a sample exceeds timer granularity;
6. randomizes candidate order across rounds;
7. validates a candidate before allowing it into performance selection;
8. records raw samples and rejects an entire compromised session by a declared
   policy rather than deleting inconvenient points;
9. repeats important results in at least three independent sessions;
10. reports median, dispersion, p95 when supported by sample count, and the
    effect size versus baseline.

The first real-shape G1 protocol fixes 40 paired rounds per fresh process with
exactly 20 scalar-first and 20 NEON-first orders, retains all 80 integer timing
observations, and repeats in exactly three processes. Each backend warms for at
least 16 calls and 500 ms; calibration doubles repetitions until a batch reaches
25 ms. A session is rejected when the geometric center of paired backend
latencies changes by more than 10% between the first and final ten pairs. The
reported per-session effect is the exponentiated median log paired scalar/NEON
latency ratio with a deterministic 10,000-resample 95% paired BCa interval. A
real-shape speedup claim is permitted only when every session's lower bound is
greater than one. The 120-pair aggregate is a pooled descriptive point estimate
only (it has no confidence interval and never overrides that gate). Every
session also reports batch-normalized backend latency in `ns/invocation`
(`elapsed_ns / repetitions`) using the arithmetic median, median absolute
deviation (`median(abs(x - median(x)))`), and nearest-rank p95
(`sorted[ceil(0.95*n)-1]`, one-indexed).

The hashed timing specification names the complete prepared-call boundary and
rejects undefined or degenerate BCa intervals. For the real case it also pins
the generated scalar/NEON source and retained disassembly identities; the
analyzer recomputes module identities from the retained Region/Loop IR and
requires runner-native Apple clang, SDK, and `llvm-objdump` provenance.

M4 single-thread results come first. Physical-core sweeps and Apple
performance/efficiency-core behavior are separate experiments with explicit
worker and affinity policies. Ryzen SMT and other second-host measurements are
deferred to G4 if AVX2 portability is selected. Nested parallelism is disabled.

## Hardware and machine-code evidence

The runner records the exact event names and tool invocation. Candidate counters
include cycles, instructions, branches, branch misses, cache references/misses,
and platform-specific memory events when reliable. Linux uses `perf stat`;
macOS uses an explicitly named Instruments or `xctrace` configuration. Reports
mark unavailable, multiplexed, permission-denied, and model-derived values.

Every selected kernel retains disassembly. The audit identifies:

- vector width and instruction families in the hot loop;
- int8 widening/conversion and FP32 accumulation sequence;
- load pattern and scale handling;
- unroll structure, tail branches, and horizontal reduction;
- stack frame and spills;
- unexpected scalarization or extra shuffles.

Reports distinguish generated intrinsics from Clang-selected instructions. In
the fixed checkpoint, direct packed-Q materialization produces the audited
`sshll.8h -> sshll.4s -> scvtf.4s` sequence, while four scale words become one
raw 16-byte load and bit reinterpretation. Reports describe those selected
instructions rather than claiming that source intrinsics executed directly.

At least one winning-versus-losing comparison on the M4 connects a schedule
choice to both emitted code and a measurement. If AVX2 is selected for G4, the
same evidence standard applies there. Latency alone is not used to invent a
microarchitectural explanation.

## Bandwidth and overhead calibration

Each host records sustained single-thread and selected multi-thread memory-copy
or read bandwidth, empty bridge-call overhead, dynamic-module call overhead, and
parallel-launch overhead. For each `M=1` kernel, the report estimates:

- logical and physically packed bytes read;
- scale and padding overhead;
- operations per byte;
- outputs and effective weight bytes per second;
- achieved fraction of calibrated bandwidth;
- compiler/tuning cost and cache-hit latency;
- break-even calls against the chosen baseline.

The model is labeled as an estimate and reconciled with observed counters before
using “bandwidth-bound,” “compute-bound,” or “launch-bound.”

## Schedule-selection acceptance

A selected schedule must:

- pass correctness before timing eligibility;
- have all assumptions represented by guards;
- fit configured code-size, pack-size, register-budget, and candidate-count
  limits;
- be no slower than the untuned vector baseline within the documented noise
  policy on reported required shapes;
- show a supported improvement on at least one required M4 shape for the
  project to claim successful empirical selection;
- if an AVX2 extension is selected for G4, report its result separately rather
  than treating it as a prerequisite for the Mac claim.

If no candidate wins, the result is reported as a negative result and the tuner
does not claim an optimization. The compiler may still be correct.

## End-to-end reporting

Framework reports show a latency decomposition:

```text
total region/block time
  = graph/bridge dispatch
  + guard and cache lookup
  + compiled kernel time
  + unsupported/fallback work
  + output handling
```

They also report supported-region coverage and distinguish cold compile, cold
cache, warm cache, and steady-state execution. Text-generation numbers are not
attributed entirely to DecodeForge when attention, KV handling, sampling, or
fallback operators dominate.

## Result bundle layout

```text
results/<run-id>/
  manifest.json
  report.md
  correctness.json
  samples.csv
  counters.json
  region.ir
  loop.ir
  schedule.json
  pack-manifest.json
  generated.c
  generated.s
  build-command.txt
  stdout.txt
```

Large model weights and generated native binaries are not required in version
control. The manifest pins their hashes and records the command that rebuilds
them. A summary table is generated from raw files rather than hand-transcribed.

## Minimum publishable result

The first credible public report contains:

- one required shape on M4 ARM64 NEON;
- scalar and untuned-vector baselines under identical Q8 semantics;
- a selected schedule and all losing candidates;
- correctness distributions and raw timing samples;
- generated source and audited assembly;
- available counters plus a bandwidth/overhead model;
- compiler time, tuning time, cache-hit time, code size, and pack size;
- an honest M4 explanation, including a negative result if that is what the
  evidence shows; any cross-target explanation is optional G4 evidence.

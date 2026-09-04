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
remains deferred to optional G4 work. The hardened versioned runtime C ABI is
the first completed G2 piece and is exercised through the actual release
dynamic library. The eager PyTorch operator and G3 22-query-projection
generation experiment remain unmerged; G3 has not started on `main`. This
document defines their claim boundaries before results exist.

## Claim classes

DecodeForge keeps five kinds of claims separate:

1. **Compiler correctness:** generated code implements the `DFQ8_B32_V1` contract.
2. **Generated-kernel improvement:** the fixed NEON kernel outperforms generated
   scalar under the same format, inputs, host, and prepared-call boundary.
3. **Framework integration:** an eligible eager PyTorch call executes through
   the guarded release artifact without hidden fallback.
4. **Model-path coverage:** all 22 TinyLlama query projections use same-Q8
   fallback for prefill and native code for eligible cached `M=1` decode.
5. **Contextual performance:** DecodeForge is compared with PyTorch, Inductor,
   llama.cpp, or another library even when precision, packing, or boundaries
   differ.

Class 2 isolates the completed compiler's current optimization contribution.
Class 4 proves deployment coverage but does not itself prove a speedup. Class 5
is useful context but cannot be presented as an apples-to-apples speedup unless
the format, inputs, thread count, operator boundary, and output semantics match.

## Required baselines

| Baseline | Purpose | Claim permitted |
|---|---|---|
| Python dequantize plus PyTorch FP32 matmul | numeric oracle | quantization/correctness only |
| Rust scalar Q8 | cross-language semantic oracle | correctness only |
| generated scalar C with vectorization disabled | codegen and ABI baseline | vector-vs-scalar speedup |
| same-Q8 eager fallback using the identical packed-weight identity | G2/G3 semantic and framework baseline | native operator and model-path correctness; performance when boundaries match |
| all-22 same-Q8-fallback TinyLlama run | G3 model baseline | hybrid native-vs-fallback query-projection effect |
| designated untuned vector schedule (future G4) | schedule baseline | tuner improvement |
| materialized Q8 graph | fusion baseline | bytes/dispatches removed and fusion effect |
| original FP32/BF16 PyTorch model or Inductor | quality/ecosystem context | contextual latency/throughput and quality only |
| llama.cpp or vendor kernel with different format | practical ceiling/context | labeled non-equivalent comparison only |

Compiler flags for the generated scalar baseline must prevent accidental
auto-vectorization; the build log and disassembly verify this.

## Shape suites

### Required real shape on the M4

G1–G3 require `[N,K]=[2048,2048]`, the shape of every TinyLlama query
projection. G1 measures one prepared `M=1` kernel. G3 prepares 22 distinct
weight identities at that shape, uses `M>1` same-Q8 fallback for prompt prefill,
and requires native coverage for each projection during cached `M=1` decode.

The remaining model shapes—`[256,2048]`, `[5632,2048]`, `[2048,5632]`, and
`[32000,2048]`—are possible G4 broader-coverage work. Ryzen 5 3600 measurements
are optional G4 evidence if AVX2 portability is selected; neither is a G0–G3
requirement.

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

G3 adds model-integration cases for exactly 22 ordered layer paths, swapped or
missing asset identities, partial-install rollback, prefill-versus-decode
dispatch, forced all-fallback execution, repeated teardown, and call-counter
reconciliation.

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
- The G2 eager operator is compared with the canonical output on a frozen real
  release-library fixture. Guard-miss fallback must be built from the same Q8
  weight identity as the binding; source FP32 weights are not interchangeable.
- G3 compares the hybrid native path with an all-same-Q8-fallback run from the
  identical prompt and prepared packs. Operator deltas, downstream logit deltas,
  and greedy token IDs are reported separately under thresholds frozen before
  the final run.
- Fusion is compared with the materialized Q8 graph.

The source-vs-Q8 report is a quantization-quality result, not the generated-
kernel comparator and not evidence that a native kernel has passed.

Reports include maximum absolute and relative error, mean squared error, and
cosine similarity. Model-level reports add top-k/logit agreement and
fixed-prompt greedy-token agreement. A pinned perplexity slice is needed only
if a future claim covers a broad quantized model path. The tolerance and its
rationale are versioned before candidate timing.

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

### G2 eager-operator protocol

The eager operator has three separately reported boundaries:

1. direct bridge run after handle creation;
2. `torch.library` dispatch plus bridge run with preallocated input and newly
   allocated output;
3. same-Q8 fallback dispatch for an intentional guard miss.

Correctness and guard/lifecycle acceptance come before latency. The real
release-library test records the library hash, bridge ABI, module/pack
identities, shape, tensor dtype/stride/device, direct data pointers, completion
counters, and output bits. Library verification/snapshotting, module build,
pack copying, and handle creation are cold setup and are not included in a
steady-state operator number. Reports may show them separately.

### G3 prompt-to-text protocol

Freeze the model revision, tokenizer assets, prompt bytes and token IDs,
greedy-decoding settings, `use_cache=True`, new-token count, seeds, Torch thread
count, warmup, repetitions, and software/host versions before the final run.
The prompt must tokenize to more than one token and generation must request at
least two new tokens: the first token is selected from prefill logits, so a
subsequent token is needed to exercise cached `M=1` decode.
Compare two executions using the exact same 22 Q8 pack identities:

- **all fallback:** every q-projection uses the same-Q8 reference;
- **hybrid native:** prompt prefill uses same-Q8 fallback and every eligible
  cached `M=1` q-projection uses its native binding.

At minimum, retain per-step latency and aggregate counters per adapter. The
coverage verifier requires all 22 adapters to observe prefill fallback and
cached native calls, zero errors, zero eligible silent fallbacks, and zero
in-flight calls after completion. It reconciles totals with observed forward
calls instead of assuming a generation-library call count.

Report these intervals separately:

- safetensors validation, Q8 quantization, and OI4 packing;
- dynamic-library verification/snapshot, module build/load, and handle creation;
- model/tokenizer load and adapter installation;
- first prompt prefill;
- time to first generated token;
- every cached decode step and tokens per second;
- total generation and peak resident memory.

Cold setup must not be hidden inside a decode number or silently excluded from
an end-to-end number. The first public result can be a correctness/coverage
proof even if hybrid generation is not faster. Any speedup claim requires raw
paired model runs, a predeclared session-quality policy, uncertainty, and a
boundary that names all fallback work.

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

A future schedule-selection G4 claim must connect at least one
winning-versus-losing M4 choice to both emitted code and a measurement. If AVX2
is selected for G4, the same evidence standard applies there. G0–G3 require the
fixed G1 source/disassembly/result evidence but do not require a tuner. Latency
alone is never used to invent a microarchitectural explanation.

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

## Deferred G4 schedule-selection acceptance

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
q_proj adapter call
  = eager dispatch + guard
  + (native bridge + generated kernel | same-Q8 fallback)
  + output handling

total prompt-to-text time
  = model/tokenizer setup (cold report only)
  + prompt prefill
  + cached decode steps
  + attention, KV handling, remaining model operators, and sampling
```

They report per-adapter native/fallback/error coverage and distinguish offline
packing, cold module construction, warmed execution, and steady-state decode.
Text-generation numbers are not attributed entirely to DecodeForge when
attention, KV handling, sampling, or fallback operators dominate.

## Result bundle layout

Kernel bundles retain the existing manifest, report, correctness data, raw
samples, Region/Loop IR, fixed schedule, pack manifest, generated source,
assembly, build command, and captured tool output. The G3 model bundle adds the
pinned prompt/token IDs, ordered 22-pack inventory, per-adapter coverage,
model-level correctness, per-step timing samples, decoded text, and environment
record. Shared compiler artifacts may be referenced by content hash rather than
duplicated.

Large model weights and generated native binaries are not required in version
control. The manifest pins their hashes and records the command that rebuilds
them. A summary table is generated from raw files rather than hand-transcribed.

## Minimum publishable result

The first credible public report combines the completed G1 evidence with G2/G3
model-use evidence:

- the real `[2048,2048]` M4 ARM64 NEON kernel and same-semantics generated scalar
  baseline, with G1 raw timings and paired uncertainty;
- canonical IR, OI4 layout, generated source, audited assembly, build provenance,
  and correctness evidence;
- the real release bridge exercised through the guarded eager PyTorch operator;
- exactly 22 identity-bound query-projection assets and adapters;
- a pinned prompt whose prefill fallback and cached native coverage reconcile
  through counters, compared with an all-same-Q8-fallback run;
- separated preparation, cold startup, prefill, per-token decode, total time,
  pack size, and peak-memory measurements;
- an honest result even when the first hybrid model run is neutral or slower.

Schedule candidates, tuning time, cache-hit analysis, broader operators,
cross-target evidence, and external performance comparisons become required
only when the corresponding G4 claim is made.

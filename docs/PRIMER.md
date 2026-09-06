# DecodeForge: a reader's primer

Read this as a guided tour: [the system](#how-the-pieces-fit-together),
[the findings](#what-did-the-benchmarks-show),
[the experimental method](#experimental-method-step-by-step),
[the testing strategy](#testing-the-system-and-the-measurement-code), and
[reproduction](#what-can-a-visitor-run).

## The project in one paragraph

DecodeForge is a small compiler that generates specialized CPU code for part of
a language model. It takes fixed, quantized weight matrices, checks their
mathematical and memory-layout contracts, generates scalar or ARM64 NEON code,
and uses Clang/LLVM to build executable machine code. A native bridge connects
that code to PyTorch. The completed demonstration runs all **22 query
projections in TinyLlama-1.1B during cached single-token decoding** on an Apple
M4. PyTorch and Transformers still run the rest of the model. This is a compiler
and integration project, not a new model or a complete inference engine.

## Why build it if TinyLlama already runs locally?

TinyLlama already runs on a Mac using PyTorch. The engineering question is
whether a small, independently built compiler can take responsibility for a
real computation inside that model, execute specialized native code correctly,
and demonstrate exactly where it helps.

A query projection transforms a token's internal representation into the
"query" used by attention. During single-token decoding, this projection is a
matrix-vector product: multiply a fixed weight matrix by the current token's
activation vector. Fixed dimensions and weights make specialization possible.
DecodeForge targets bias-free query projections with a `[2048, 2048]` weight
matrix. Other shapes and the remaining 133 linear modules are outside the
completed scope.

## How the pieces fit together

```text
Fixed model weights + shape + CPU target
                  |
          Typed Q8 linear representation
                  |
          Quantize and pack weights
                  |
       Lower to explicit regions and loops
                  |
       Generate scalar / ARM64 NEON C
                  |
       Clang/LLVM builds machine code
                  |
       Audit artifact; load native bridge
                  |
       Guarded eager PyTorch adapters
                  |
    +-------------+---------------------+
    |                                   |
Prompt prefill                 Cached single-token decode
Same-Q8 reference              Native query projections
    |                                   |
    +--------- Rest of model: PyTorch --+
```

The compiler makes operations, reduction order, vector width, and packed
addressing explicit in intermediate representations. It does not merely call
an existing matrix-multiplication library. Generated source, machine-code
audits, artifact identities, and timing samples are retained for inspection.

The bridge checks shapes, data types, buffers, and artifact identities before
native execution. Installation is transactional: an unsuccessful installation
must not leave a partially modified model. Closing the adapters restores the
original PyTorch modules and releases native resources.

Follow the implementation in order: [typed IR](../compiler/decodeforge-compiler/src/ir.rs)
→ [weight packing](../compiler/decodeforge-compiler/src/pack.rs)
→ [lowering](../compiler/decodeforge-compiler/src/lower.rs)
→ [NEON code generation](../compiler/decodeforge-compiler/src/codegen/neon_c.rs)
→ [artifact audit](../compiler/decodeforge-compiler/src/native/audit.rs)
→ [native bridge](../compiler/decodeforge-bridge/src/lib.rs)
→ [PyTorch model adapters](../python/decodeforge/qproj_model.py).

## Four terms worth knowing

| Term | Meaning here |
| --- | --- |
| Q8 / weight-only quantization | Store weights as signed 8-bit values with scales. Activations and arithmetic remain FP32; this is not an integer dot-product kernel. |
| NEON / SIMD | ARM vector instructions that operate on several values at once, rather than one scalar value at a time. |
| Prefill vs decode | Prefill processes the input prompt. Cached decode processes one new token at a time while reusing the model's attention cache. |
| Same-Q8 reference | A comparison path reconstructed from the identical quantized weights. It isolates compiler differences from changes caused by quantization. |

## What did the benchmarks show?

There are two different performance results, and they must not be combined.

**The kernel result:** generated NEON code was approximately **3.96x faster
than generated scalar code** across three independent Apple M4 sessions. Both
used the same Q8 projection and complete prepared-call boundary, including
output checks. Packing, compilation, loading, and allocation were outside the
timed boundary. This is not a 3.96x speedup over PyTorch or over the whole model.
The [G1 report](../results/g1/apple-m4-primary/README.md) retains the exact
speedups, confidence intervals, and raw observations.

**The model result:** a separate evaluation ran **81 measured generations and
27 warmups across three fresh processes**, comparing original FP32 PyTorch,
the guarded same-Q8 reference, and hybrid native execution. Native decode
delivered roughly **11–15 tokens/s**, versus **4–5 tokens/s** for the guarded
Q8 reference. Original FP32 was around **12–15 tokens/s**: native execution
did **not consistently beat FP32**. The
[README chart and exact table](../README.md#measured-results-apple-m4) show all
three cases and their run-to-run ranges.

The model benchmark excludes correctness-comparison hooks, but retains
production guards, reference fallback weight cloning/hashing, and outer
finite-logit checks in decode timing. Those costs matter. Most of the model
also remains in PyTorch, so a faster query-projection kernel does not translate
directly into the same whole-model speedup. No comparison with llama.cpp, MLX,
or a GPU inference engine was established.

## How do we know it works?

- **30/30 fixed prompts and 1,070 generated steps:** native and same-Q8
  reference token sequences agreed exactly. Maximum absolute logit difference
  was about **0.00001717**, within the declared tolerance.
- **Actual native execution:** counters reconciled all 22 query projections;
  matching outputs alone would not prove the native path ran.
- **Clean restoration:** all 22 original modules were restored, with no live
  adapters or in-flight calls remaining.
- **Quantization sensitivity:** all 30 greedy sequences also matched FP32 on
  this synthetic corpus. A separate fixed-reference probe showed **99.7099%**
  next-token argmax agreement over 1,034 tokens. This is not a general quality
  guarantee or proof that quantization improves the model.

These are correctness and integration findings, not a broad capability score.
**29/30 generations reached their token cap** rather than stopping naturally.
A separate sentence-formatted presentation demo is not a benchmark or proof
of improved instruction following. Clean-checkout evaluation succeeded on the
same M4; another physical Mac remains untested.

## Experimental method, step by step

The method separates three questions: **did compilation preserve the
computation, what changed because of quantization, and how fast is the
resulting implementation?** Each comparison has its own baseline and rules.

### 1. Fix the experiment before inspecting outputs

The broader evaluation uses a committed specification: 30 unique, original
synthetic prompts, ten in each input-length class, with output caps of 16,
32, and 64 occurring ten times each. Three performance cases are named in
advance, not chosen because they produced favorable timings. These are
sensitivity probes, not a representative sample of real user tasks or an
external benchmark dataset.

The runner requires the supplied specification to byte-match the committed
copy. It pins the model artifacts, uses a clean source revision, and records
model/tokenizer, Q8 pack, and native-library identities. CPU FP32 execution,
one Torch thread, one inter-op thread, seed 0, the local chat template, and
greedy decoding keep these choices fixed. A seed alone would not establish
reproducibility; the artifact and execution settings matter too.

Read: [fixed cases and settings](../benchmarks/evaluation-v1/spec.json),
[input protocol](EVALUATION_V1.md#frozen-inputs-and-decoding), and
[specification tests](../python/tests/test_evaluation_spec.py).

### 2. Choose controls that isolate different causes

| Comparison | What stays fixed | What changes | Question answered |
| --- | --- | --- | --- |
| Generated scalar vs generated NEON | Q8 projection, inputs, prepared-call boundary | Generated execution schedule | Does vectorization improve this kernel boundary? |
| Same-Q8 reference vs hybrid native | Quantized weight identities, prompts, greedy settings | Query-projection execution path | Does native execution preserve model behavior? |
| Original FP32 vs same-Q8 reference | Prompts or fixed teacher-forced tokens | Query-projection weight representation and reference path | How sensitive is this probe to quantization? |
| Original FP32 vs hybrid native timing | Named prompt/cap and measurement boundaries | Combined quantization, adapters, and generated execution | How does the integrated implementation compare in practice? |

The final row is a practical comparison, not a controlled estimate of the
compiler's isolated contribution. Likewise, the guarded Q8 fallback includes
production work that original FP32 does not perform. Keeping both baselines
visible prevents attributing all differences to machine-code quality.

Read: [numerical comparison policy](BENCHMARKS.md#numeric-policy),
[reference/native paths](../python/decodeforge/qproj_adapter.py), and
[model observations](../results/evaluation/apple-m4-v1/README.md#practical-performance).

### 3. Require numerical agreement and observable execution

During correctness capture, compare model logits at every shared input
prefix. Every compared value must be finite and satisfy the predeclared rule:

```text
abs(native - reference) <= 0.001 + 0.001 * abs(reference)
```

That is an elementwise absolute-plus-relative tolerance, not an average error
that could hide one bad output. Exact generated token-ID equality is a
separate requirement. If tokens diverge, the runner does not compare unrelated
continuations and call them equivalent. Matching rendered strings is not
enough, and matching tokens cannot excuse a failed logit check.

Per-layer counters independently establish execution: all 22 layers must
show the expected prefill/reference and cached/native calls without errors.
Installation and cleanup checks verify module ownership and restoration.
An early EOS that prevents any cached decode is retained as a coverage
failure, not suppressed to force a pass.

Read: [correctness and lifecycle rules](EVALUATION_V1.md#correctness-and-lifecycle),
[generation/comparison code](../python/decodeforge/evaluation.py),
[cached-loop tests](../python/tests/test_evaluation.py), and
[installation/rollback tests](../python/tests/test_qproj_model.py).

### 4. Measure quantization sensitivity on identical target sequences

Free-running generations can diverge and then receive different future
inputs. The separate teacher-forced probe avoids that confound: both FP32
and same-Q8 receive the same fixed token sequence, including the same previous
target tokens at each prediction position. No extra EOS is appended. The
project-authored reference text is a fixed stimulus, not a ground-truth answer.

For each target token, negative log-likelihood (NLL) is
`-ln(probability assigned to that target token)`. Lower NLL means more
probability assigned to these particular fixed targets, not necessarily
better answers. The analyzer sums per-token NLL and divides by the total
number of target tokens; it does not give short and long prompts equal weight
by averaging their averages. Argmax agreement separately asks whether the
two paths prefer the same next token.

Across 1,034 target tokens, mean NLL was **3.4827915980 FP32** and
**3.4825643588 Q8**, a difference of **−0.0002272392 nats/token**. This tiny
descriptive difference is not evidence of a general quality improvement.
Teacher forcing uses a multi-token forward and therefore tests the same-Q8
fallback, not native cached decode; step 3 tests the native path separately.

Read: [fixed-sequence protocol](EVALUATION_V1.md#original-fp32-context-and-fixed-sequence-probe),
[metric implementation](../python/decodeforge/evaluation_metrics.py), and
[known-answer metric tests](../python/tests/test_evaluation_metrics.py).

### 5. Benchmark the kernel with paired trials and a declared decision rule

G1 uses **40 paired rounds per process**, balanced between 20 scalar-first
and 20 NEON-first orders. Each backend warms for at least 16 calls and 500 ms;
calibration increases repetitions until a batch reaches at least 25 ms.
There are three fresh processes, yielding 120 pairs and 240 raw observations.
The measured boundary includes the native call, output sentinel fill, status
decoding, and finite-output scan; it excludes packing, compilation, loading,
and allocation.

The speedup estimator is the exponentiated median of paired log latency
ratios. A deterministic 10,000-resample paired BCa (bias-corrected and
accelerated bootstrap) produces a 95% interval for each session. The declared
claim rule requires **all three lower confidence bounds to exceed 1.0**.
All three passed. The pooled result is descriptive; it does not replace this
per-session rule or turn repeated measurements into independent machines.

A session is rejected if the geometric center of paired backend latencies
drifts by more than 10% between the first and last ten pairs. That detects
timing drift; it is not a direct measurement or elimination of thermal effects.
The protocol rejects a compromised session rather than deleting individual
inconvenient samples.

Read: [frozen G1 specification](../benchmarks/g1/spec.json),
[timing method](BENCHMARKS.md#timing-protocol),
[analyzer](../scripts/analyze_g1_benchmark.py), and
[retained report and intervals](../results/g1/apple-m4-primary/report.md).

### 6. Measure model performance separately from correctness capture

The broader model evaluation deliberately uses a smaller descriptive design:
**3 processes × 3 fixed cases × 3 paths × 3 measured repetitions = 81
generations**, plus one warmup per process/case/path, or 27 warmups.
Correctness-comparison hooks and teacher-forced scoring are outside performance
timers. Production guards, finite-output checks, and the fallback's own
clone/hash work remain included and disclosed.

Raw nanosecond samples support per-process median/min/max summaries of prefill,
cached decode, and total generation. Setup components and process-lifetime
peak RSS are reported separately. The README chart shows the minimum and
maximum of the **three process medians**, not a confidence interval. The 81
generations are not 81 independent process samples.

FP32 runs first, before adapter installation; the two Q8 paths reverse order
in the middle process. This is only partial order balancing. No concurrent
builds or other evaluation processes are permitted during capture, but order,
cache, scheduling, and thermal effects are not eliminated. The slow native
short-prompt observation in process 2 is retained. No model-level confidence
interval or statistically significant FP32 speedup is claimed, and the G1
bootstrap/drift protocol must not be implied to apply to this separate study.

Read: [timing inclusions and exclusions](EVALUATION_V1.md#practical-performance-protocol),
[raw process 0](../results/evaluation/apple-m4-v1/performance-0.json),
[process 1](../results/evaluation/apple-m4-v1/performance-1.json),
[process 2](../results/evaluation/apple-m4-v1/performance-2.json), and
[recomputed summary](../results/evaluation/apple-m4-v1/summary.json).

### 7. Preserve evidence and make rejection behavior testable

The broader runner refuses a dirty or unidentified producer, altered
specification, invalid model outputs, and existing output paths. Acceptance
requires consistent identities, exact native/reference tokens, numerical
checks, all-layer counters, clean teardown, and complete timing samples.
Rejected attempts and their reasons remain separate from accepted summaries;
a failure must not become a success-looking JSON report.

The analyzer checks the retained records and recomputes token-weighted metrics
and timing summaries. This gives a reader an inexpensive audit path. It is
not independent model re-execution: full-vocabulary logits were transient,
so the summary verifier cannot reconstruct them from saved error metrics.
Clean source and hashes make the evidence traceable, not externally certified
or immune to every measurement error.

Read: [acceptance policy](EVALUATION_V1.md#result-acceptance-and-retention),
[raw correctness capture](../results/evaluation/apple-m4-v1/correctness-v1.json),
[summary analyzer](../scripts/analyze_evaluation.py),
[runner rejection tests](../python/tests/test_evaluation_rejections.py), and
[tampered-evidence tests](../python/tests/test_evaluation_analysis.py).

## Testing the system and the measurement code

The tests form layers of evidence. Small fixtures check precise failure modes;
compiled-library integration checks cross the real native boundary; model
captures test the combined system. None substitutes for the others.

| Layer | Strategy and examples | Inspect the tests |
| --- | --- | --- |
| Q8 semantics | Python/Rust fixture parity; rounding, zero blocks, boundaries, and numerical comparison behavior | [Q8 tests](../python/tests/test_q8.py), [IEEE cases](../python/tests/test_q8_ieee.py), [fixture parity](../python/tests/test_fixtures.py) |
| Compiler and native artifacts | Verify lowering/packing contracts and generated artifacts, including vector/tail behavior | [Compiler tests and implementation](../compiler/decodeforge-compiler/src/lib.rs), [asset validation tests](../compiler/decodeforge-compiler/src/model_assets/tests.rs) |
| Actual FFI integration | Execute the release shared library using real tensor buffers; check ABI behavior, not only mocks | [Release-library check](../scripts/check_bridge_cdylib.py), [PyTorch bridge tests](../python/tests/test_torch_bridge.py) |
| Model lifecycle | Detect missing/swapped assets, partial-install failures, incorrect dispatch, and repeated teardown | [Adapter tests](../python/tests/test_qproj_adapter.py), [all-layer tests](../python/tests/test_qproj_model.py) |
| Measurement mathematics | Known NLL for uniform logits; target-vs-argmax distinction; exact tolerance boundary; NaN/Inf and shape rejection | [Metric tests](../python/tests/test_evaluation_metrics.py) |
| Experiment control | Reject altered specs; verify cache/EOS behavior; ensure one timing sample per cached forward | [Specification tests](../python/tests/test_evaluation_spec.py), [runner tests](../python/tests/test_evaluation.py) |
| Evidence integrity | Mutate tokens, digests, counters, session count, and source cleanliness; require rejection | [Analyzer tests](../python/tests/test_evaluation_analysis.py), [CLI failure tests](../python/tests/test_evaluation_rejections.py) |

Linux x86-64 and macOS ARM64 CI exercise their supported checks, including
offline operation. Linux CI is not evidence of an implemented AVX2 backend,
and macOS CI is not a second-host TinyLlama evaluation. See the
[workflow](../.github/workflows/ci.yml) and [aggregate commands](../Makefile)
for what actually runs. Unit tests use small fixtures and test doubles where
appropriate; only the retained model captures support the model-level claims.

## Limits and threats to validity

- **External validity:** one physical M4, one model, 22 query projections,
  and 30 synthetic prompts do not establish results on other machines,
  models, workloads, or all-model quantization.
- **Measurement attribution:** guards and fallback overhead remain. FP32 runs
  first, caches are reused, and only three process clusters are observed.
  The model timings are descriptive, not a causal estimate of compiler-only
  acceleration or a cold-storage startup benchmark.
- **Quality measurement:** exact tokens and small logit differences establish
  implementation agreement on the probe, not helpfulness. Most generations
  hit their cap, and fixed reference texts are not task-answer labels.
- **Reproducibility:** a clean same-host checkout and retained analyses were
  verified. New model execution on another physical host is still needed for
  a cross-host claim.

These limits define the next experiments: reproduce on another compatible
Mac, expand externally meaningful task-quality coverage, and profile the
guarded boundary/prefill costs before proposing a new optimization. They are
future work, not results silently included in the completed scope.

## What can a visitor run?

Start with [CONTRIBUTING.md](../CONTRIBUTING.md) for the pinned tools and setup.
Then choose the level of verification you need:

```sh
# Recompute retained benchmark analyses; no model execution or download.
make verify-g1-result verify-evaluation-result

# Run code checks, tests, fixtures, and retained evidence verification.
make check
```

The analysis commands verify the saved observations; they do not independently
rerun the model. For a new local generation, follow the artifact prerequisites
and command in the [presentation guide](PRESENTATION_DEMO.md), with the optional
[explicit sentence-stopping mode](PRESENTATION_POLISH.md). For new model
measurements, use the separate [evaluation protocol](EVALUATION_V1.md).
These require the pinned local model files, prepared Q8 assets, and verified
native library; the model weights are not committed to this repository.

## Where to go next

| Interest | Start here |
| --- | --- |
| Exact model results, setup, memory, and limitations | [Apple M4 evaluation](../results/evaluation/apple-m4-v1/README.md) |
| Compiler architecture and design choices | [Design](DESIGN.md) |
| Quantization mathematics and numerical contract | [Q8 format](Q8_FORMAT_V1.md) |
| Benchmark methodology | [Benchmarks](BENCHMARKS.md) |
| Compiler implementation | [Rust compiler crate](../compiler/decodeforge-compiler/src/lib.rs) |
| PyTorch model integration | [Query-projection model adapters](../python/decodeforge/qproj_model.py) |

The original G0–G3 compiler/demo scope is complete. Broader layer coverage,
less boundary overhead, native prefill, additional CPU targets, and multicore
execution are possible follow-ups, not demonstrated results. Original project
code is under [Apache 2.0](../LICENSE); model and dependency terms remain separate.

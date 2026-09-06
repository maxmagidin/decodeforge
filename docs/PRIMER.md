# DecodeForge: a reader's primer

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

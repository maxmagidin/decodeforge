# DecodeForge implementation plan

**Purpose:** deliver the smallest sequence of independently demonstrable compiler
results. A milestone is complete only when its evidence is checked in; creating
the planned directory or API is not progress by itself.

## Operating rules

1. Keep one executable vertical slice at all times.
2. Optimize only after the current path has a correctness oracle and benchmark.
3. Preserve losing schedules and negative results.
4. Do not begin two architecture backends simultaneously.
5. Do not describe a component as built until its acceptance command succeeds
   from a clean checkout.
6. Prefer a smaller compiler with inspectable output over additional supported
   operators.

## Critical path

```text
Q8 semantics
    -> scalar kernel and corpus
    -> minimal Region/Loop IR
    -> generated scalar kernel
    -> NEON vertical slice
    -> AVX2 vertical slice
    -> bounded schedule selection
    -> guarded torch.compile integration
    -> optional fusion or small-batch extension
```

The report viewer, predictive cost-model experiments, multi-core tuning, and
additional graph patterns are off the critical path.

## G0 — Freeze semantics and evidence schema

### Build

- Specify `DFQ8_B32`, including rounding, zero blocks, padding, NaN/Inf policy,
  and the exact accumulation expression.
- Implement Python and Rust scalar quantize/dequantize-and-dot references.
- Define a versioned run manifest before producing benchmark numbers.
- Create deterministic fixtures for zero, sign-alternating, extreme finite,
  block-boundary, tail, and random cases.

### Exit evidence

- Python and Rust references agree on serialized fixtures.
- Re-running the fixture generator with the recorded seed reproduces its hashes.
- The manifest records inputs, source revision, toolchain, CPU, target features,
  numeric mode, and output hashes.

### Stop condition

Do not create SIMD code until rounding and tail behavior are identical across
the references.

## G1 — Complete one NEON vertical slice

Use the `M=1`, `[N,K]=[2048,2048]` TinyLlama projection first, then add the
remaining required projection families. The first slice contains only the IR
needed by Q8 linear.

### Build

- Lower a `Q8Linear` Region IR op into a minimal Loop IR.
- Emit scalar C through the same lowering and code-generation interface intended
  for vector schedules.
- Pack one immutable target panel format with a versioned manifest.
- Emit NEON widening/conversion and FP32 accumulation intrinsics.
- Compile and load the artifact through the stable C ABI.
- Add schedule knobs only after a fixed vector kernel is correct.

### Exit evidence

- Generated scalar and NEON output pass the complete correctness corpus.
- One TinyLlama projection bundle includes Region IR, Loop IR, pack metadata,
  source, build command, disassembly, raw samples, and a Markdown report.
- The disassembly audit identifies the hot loop, vector instructions, tail path,
  stack frame, and any spills.
- The vector path is faster than generated scalar for at least one required real
  shape under the same Q8 semantics. Failure triggers analysis, not a new feature.

## G2 — Prove the abstraction on AVX2

The x86 backend must consume the same semantic IR and differ only at explicit
schedule, packing, target-lowering, guard, and target-artifact boundaries. The
logical kernel call contract remains versioned and stable.

### Build

- Add AVX2/FMA widening/conversion and FP32 accumulation lowering.
- Add x86 feature detection and negative feature-guard tests.
- Re-run the same fixture and real-shape suites on the Ryzen 5 3600.
- Compare at least two legal schedule or packing choices on both CPUs.

### Exit evidence

- AVX2 passes the same scalar oracle and tail corpus.
- A wrong-feature or wrong-shape artifact is rejected before kernel entry.
- The report contains one cross-target case study where a schedule parameter has
  a measurably different effect and connects that effect to layout, assembly,
  counters, or a clearly labeled model.
- Both selected vector paths beat generated scalar on at least one real shape.

## G2.5 — Add bounded schedule selection

Only after fixed NEON and AVX2 kernels work:

- expose `N` tile, `K`-block unroll, accumulator count, panel layout, prefetch,
  and tail strategy selectively;
- reject illegal schedules before compilation;
- benchmark a capped, recorded candidate set;
- require correctness before a candidate can win;
- retain every candidate, including failures and losers.

The first useful tuner may be exhaustive over a deliberately tiny space. A
predictive cost model is not required for the résumé-ready result.

### Exit evidence

- Selection is deterministic given a result bundle and policy.
- The selected schedule is no slower than the designated untuned vector baseline
  within the documented noise policy on reported required shapes.
- At least one shape on each architecture shows a statistically supported win
  over that untuned baseline.
- Tuning time and break-even call count are reported.

## G3 — Integrate through PyTorch

### Build

- Register the explicit logical `q8_linear` operator and fake/meta behavior.
- Recognize only frozen, CPU, supported-layout regions.
- Compile or retrieve a guarded artifact and call it through the ATen bridge.
- Preserve the unsupported FX graph and make fallback policy observable.

### Exit evidence

- A TinyLlama projection and then one decoder-block path execute through
  `torch.compile`.
- Tests demonstrate the successful path, shape/stride/feature guard misses,
  fallback, cache miss, cache hit, corrupt-cache rebuild, and numeric failure.
- The report separates kernel time, bridge/dispatch time, compiled-region
  coverage, block time, and contextual end-to-end time.

## G4 — Choose one extension from evidence

Choose exactly one first:

- RMSNorm plus Q8 linear fusion;
- paired gate/up plus SwiGLU fusion;
- `M in {2, 4, 8}`;
- multi-core output-channel scheduling.

Selection is based on the largest measured bottleneck after G3, not on which
feature sounds most impressive. The extension must compare against a
materialized or untuned path with identical semantics and report regressions as
well as wins.

## Explicit cut order

If completion risk rises, remove work in this order:

1. dashboard;
2. predictive cost model;
3. second fusion;
4. first fusion;
5. `M > 1`;
6. multi-core tuning;
7. broad FX pattern support.

Do not cut scalar oracles, both ISA backends, generated-source retention,
assembly inspection, raw measurements, guards, or one end-to-end PyTorch path;
those are the core evidence.

## Résumé promotion checklist

Résumé bullets may use these verbs only after the matching evidence exists:

| Verb or claim | Required evidence |
|---|---|
| designed | reviewed specification or checked-in IR/ABI decision |
| implemented | tests execute the component from a clean checkout |
| generated NEON/AVX2 | retained source and disassembly confirm vector instructions |
| optimized | controlled before/after result under the same numeric contract |
| autotuned | recorded candidate set, correctness gating, selection policy, and reproducible winner |
| integrated with PyTorch | guarded compiled region executes and fallback is tested |
| achieved `X%` improvement | raw samples, baseline definition, uncertainty, host manifest, and reproduction command |

Until G2 and G3 are complete, the honest project description is “designed a
cross-target Q8 kernel compiler and implemented its current completed gates,”
with those gates named explicitly.

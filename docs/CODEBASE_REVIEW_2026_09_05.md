# DecodeForge codebase review — 2026-09-05

DecodeForge has a working Mac-first kernel compiler and eager PyTorch bridge.
The highest-value next milestone is the formal G3 model-generation evidence,
followed by an optimization chosen from that evidence. More compiler surface
is not currently the main missing deliverable.

## Which checkout this review covers

The reviewed base is `integration/g2-g3` at `cf5f970`, which was clean at the
start of this review. It is 38 commits ahead of `origin/main` (`b978cd7`). The
ordinary local `main` is still at `009a7ad`, 94 commits behind `origin/main`,
with a substantial uncommitted scaffold. These are materially different
implementations. The initial scaffold-only assessment does not describe the
current integration branch.

Three `gpt-5.6-luna` executors reviewed the native compiler/runtime, Python
bridge/model integration, and evidence/schema tooling. The parent reviewed
the architecture, numeric reference, command surface, retained results, and
integration of changes. Preliminary edits to the old scaffold were undone;
its pre-existing changes were preserved. Review changes are local and have
not been committed or pushed.

## Maturity and evidence

| Area | Current state | Evidence and limit |
| --- | --- | --- |
| G0 semantics | Implemented | Independent Python/Rust binary32 arithmetic, Q8 quantization, identities, and a 16-case frozen corpus. A checked-in provenance bundle exists. |
| G1 compiler/kernel | Implemented and measured | Verified Region/Loop IR, OI4 packing, scalar/NEON C, Clang builds, Mach-O/disassembly checks, and native execution. Three retained sessions report approximately 3.96x NEON over generated scalar. |
| G2 eager integration | Implemented | Six-function native C ABI, owned handles, tensor guards, explicit fallback, counters, and a test through the actual release dylib and Torch storage. |
| G3 model integration | Implemented; formal evidence pending | Preparation, owning adapters for all 22 query projections, transactional installation, session runner, and bundle verifier exist. No accepted three-session generation bundle is checked in. |
| G4 expansion | Deferred | Schedule search, general graph compilation, broader linear coverage, AVX2, fusion, and multicore remain extensions. |

The G1 speedups are `3.95671x`, `3.96176x`, and `3.95648x`, with respective
95% paired-BCa intervals `[3.95103, 3.96705]`, `[3.95085, 3.96960]`, and
`[3.95351, 3.95997]`. They compare the complete prepared-call boundary,
including output scrubbing and validation, for one `[2048,2048]` projection.
They exclude packing, compilation, loading, and allocation. They establish
neither whole-model acceleration nor a win over optimized PyTorch kernels.
See [the retained G1 evidence](../results/g1/apple-m4-primary/README.md).

## Strengths

- **Numerical correctness is unusually explicit.** Independent integer/rational
  binary32 oracles avoid accidental host rounding and contraction. Fixtures
  exercise tails, signed zero, subnormals, ties, and finite extremes; strict
  generated kernels are checked against their exact expected output words.
- **The compiler has inspectable intermediate steps.** Typed IR, explicit
  packing, deterministic source, and retained disassembly make it possible to
  connect a semantic operator to emitted SIMD instructions.
- **Native ownership is carefully bounded.** The bridge checks sizes,
  alignment, overlap, identities, and resource limits; handles and private
  library copies have explicit lifetimes. Tests cover failure scrubbing,
  concurrent run/close behavior, and panic containment.
- **The integration has observable behavior.** Native success, fallback,
  errors, and in-flight operations are accounted separately. Model adapter
  installation and cleanup restore original modules transactionally.
- **Evidence can be recomputed.** Raw G1 samples and deterministic analysis are
  retained. G3 reconstructs session evidence and verifies the closed result
  inventory instead of trusting summary numbers.
- **The scope is defensible.** The current accepted direction is Mac-first,
  one fixed shape, and all 22 query projections. Framework and inference
  responsibilities remain with PyTorch/Transformers.

## Improvements made in this review

| Defect | Change | Regression boundary |
| --- | --- | --- |
| Older G0/G1 Make recipes interpreted public path arguments as shell text. Quotes and command syntax could alter an argument before the underlying tool received it. | Extend the existing G3 raw-variable/environment approach to the older commands, including external Cargo target paths. | Execute Make with recorder tools and confirm exact argument delivery for unusual filenames. |
| JSON exponent overflow such as `1e999` became Python infinity despite the documented finite-JSON contract. Rejecting the literal `Infinity` token alone was insufficient. | Reject float overflow at evidence parser boundaries through a shared finite-number helper. | Exercise positive/negative overflow and metadata through parser/validator entry points. |
| Model preflight did not check that original query-projection weights were finite. | Reject non-finite source weights before adapter factories or native handles are created. | A NaN source weight must fail before any factory call. |

These changes preserve the numeric mode, ABI, schema versions, and measured
results. No native arithmetic or benchmark acceptance thresholds were changed.

## Weaknesses and remaining risks

1. **G3 delivery is unfinished.** Passing adapter and session tests is not a
   substitute for three accepted fresh-process model runs and their verified
   ten-file bundle. The integration branch also contains 38 commits not yet
   on main; existing hosted CI results do not certify this exact revision.
2. **The reference timing includes substantial integrity work.**
   `QProjAdapter._fallback()` clones and hashes the full FP32 fallback weight
   before `functional.linear` on every call. That is a 16 MiB clone per
   `[2048,2048]` layer, or 352 MiB of cloned weight content across 22 layers
   per model pass, plus hashing reads. Both paths incur this during prefill;
   reference-only decode also incurs it at every token, while native decode
   avoids it. A future G3 delta therefore compares these entire guarded
   implementations, not just matrix-vector compute. Preserve the integrity
   checks and make this cost explicit before interpreting a speedup.
3. **The measured path is instrumented.** G3 hooks clone projection inputs
   and outputs on both execution paths. Dispatch timing includes guards,
   Python overhead, and fallback integrity work; `native_work_ns` is not
   measured. Native-output oracle comparisons happen after measured runs,
   which is good, but the headline must still name the timed boundary.
4. **Coverage and generality remain narrow.** Only query projections use
   native decode; the other 133 TinyLlama linear modules and non-linear
   operations remain outside the compiler. There is one fixed OI4 schedule,
   no measured schedule selection, and no AVX2 or general `torch.compile`
   path. These are limits of the current claim, not broken requirements.
5. **Cold-start and memory costs need model evidence.** Native handle creation
   builds a generated module per binding. Adapters retain dense FP32 fallback
   weights and original modules for restoration. Q8 pack size alone cannot
   establish whole-model memory savings or startup efficiency.
6. **Maintenance weight is growing.** Several evidence, session, and native
   audit modules exceed 1,000 lines. Shared parsing policy helps; future
   extraction should follow demonstrated repeated behavior and preserve the
   existing regression corpus.
7. **Repository hygiene obscures the current project.** A stale dirty local
   main, an ahead integration worktree, and an open eager-bridge PR make a
   quick checkout easy to misread. Consolidate the reviewed integration path
   through the normal commit/PR workflow. The repository also still lacks a
   maintainer-selected license.

## Validation

The final aggregate passed after all executor patches settled:

| Command/check | Result |
| --- | --- |
| `CARGO_NET_OFFLINE=true UV_OFFLINE=true make format` | Pass |
| `CARGO_NET_OFFLINE=true UV_OFFLINE=true make check` | Pass; 382 Python tests, Rust debug/release tests, real Torch/dylib execution, fixture parity, and G1 report regeneration |
| Rust workspace suites within `make check` | 149 tests per profile, including native compiler/runtime/bridge execution |
| `actionlint .github/workflows/ci.yml` | Pass |
| `git diff --check` | Pass |
| G0 repository verification using a clean base snapshot | Pass |

The final Python suite completed in 217.99 seconds. The full local aggregate
log is retained in `.lavish/review-check.log`. No fresh performance measurement
or full-model generation session was taken during this review.

The checked-in G0 bundle's repository provenance was accepted against a clean
detached snapshot of reviewed base `cf5f970`. Its checker correctly rejects
the working checkout after edits because that checkout is dirty. No historical
evidence was rewritten to conceal this distinction.

The local Rust toolchain emitted `rust-objcopy` warnings about a missing
`libLLVM.dylib` while stripping release debug information. Builds and native
execution continued. This host-toolchain warning is separate from generated
kernel correctness and should be resolved before a new certified capture.

Changed implementation and regression paths:

- `Makefile`, `python/tests/test_make_inputs.py`, and the G1 launcher assertion
  in `python/tests/test_prepare_g1_inputs.py`;
- `python/decodeforge/_json.py`, `contracts.py`, `g0_evidence.py`,
  `g1_evidence.py`, `g3_preparation.py`, and `g3_results.py`, with regressions
  in `python/tests/test_contracts.py` and `test_json_boundaries.py`;
- `python/decodeforge/qproj_model.py` and `python/tests/test_qproj_model.py`.

## Direction

Integrate and validate the reviewed changes, then capture the existing G3
protocol from a clean revision with explicit model assets and library identity.
Publish its coverage, correctness, startup, prefill, decode, and integrity-work
boundaries. Only then choose a G4 extension using the measured bottleneck.
The most useful immediate deliverable is a defensible end-to-end demonstration.

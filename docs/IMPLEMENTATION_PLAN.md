# DecodeForge implementation plan

**Purpose:** deliver the shortest sequence of independently demonstrable
compiler results. A gate is complete only when its acceptance command succeeds
from a clean checkout and its evidence is checked in. This plan implements the
delivery decision in
[ADR 0005](decisions/0005-prioritize-eager-q-projection-demo.md).

## Operating rules

1. Keep one executable vertical slice at all times.
2. Optimize only after the current boundary has a correctness oracle and a
   benchmark.
3. Keep quantization error, generated-kernel correctness, and framework
   performance as separate claims.
4. Preserve raw samples, negative results, source, disassembly, and provenance;
   do not promote a result from a hand-written summary alone.
5. Do not describe a component as built until its clean-checkout acceptance
   command succeeds.
6. Prefer standard tooling for model loading, tensor dispatch, machine-code
   generation, serialization, hashing, and statistics. DecodeForge's original
   work is Q8 lowering, OI4 packing, scalar/NEON code generation, the guarded
   native boundary, and its use in a real model.
7. A fallback must preserve the same logical Q8 weight identity. Falling back to
   the original FP32 `nn.Linear` changes semantics and is not an acceptable G3
   comparator.
8. No kernel-only result may be described as an end-to-end speedup.

## Critical path

```text
G0 Q8 semantics [complete]
    -> G1 typed lowering + OI4 pack + scalar/NEON codegen [complete]
    -> G2 hardened C ABI + guarded eager PyTorch op [complete]
    -> G3 frozen experiment + 22 packs + owning adapters [code-complete]
         -> replace all 22 q_proj modules transactionally [code-complete]
         -> capture three pinned prompt-to-text sessions [complete]
         -> verify and check in correctness/coverage/timing bundle [complete]
    -> G4 choose one extension from measured evidence
```

The Apple M4/ARM64 NEON path is the required résumé path. Schedule search,
general FX/`torch.compile` integration, all 155 TinyLlama linears, fusion,
native `M>1`, multicore execution, a dashboard, and x86-64 AVX2 are not G0–G3
dependencies.

## Status ledger

| Gate | State | Evidence boundary |
|---|---|---|
| G0: semantics | complete | independent Python/Rust semantics, closed fixtures, and Apple M4 provenance bundle |
| G1: compiler/kernel | complete | real `[2048,2048]` q-projection, generated scalar/NEON, audited dylibs, bit-exact corpus, three paired sessions |
| G2: framework boundary | complete | hardened C ABI, guarded eager operator, lifecycle hardening, and real release-dylib checkpoint pass |
| G3: model proof | complete | three accepted clean-source processes, all-22 native decode coverage, exact token agreement and verified ten-file bundle; generated control-token text limits the presentation demo |
| G4: extension | deferred | selected only from G3 bottleneck evidence |

The three G1 Apple M4 sessions measured `3.95671x`, `3.96176x`, and
`3.95648x` NEON-over-scalar speedups. Every paired-BCa 95% lower bound exceeds
`3.95x`. The timed unit is one complete allocation-free prepared call, including
sentinel fill, native ABI invocation, status handling, and finite-output scan.
It is not a model or raw-inner-loop number.

## G0 — Freeze semantics and evidence schema (complete)

Delivered:

- normative `DFQ8_B32_V1` rounding, zero-block, padding, NaN/Inf, and strict
  accumulation behavior;
- independent Python and Rust quantize/dequantize-and-dot references;
- deterministic boundary and random fixtures with byte-for-byte parity;
- a closed evidence schema and checked-in Apple M4 correctness bundle.

The normative contract remains
[`docs/Q8_FORMAT_V1.md`](Q8_FORMAT_V1.md). Fixture verification is read-only
unless the explicit Python writer is invoked with `--write`.

Acceptance commands:

```sh
make fixture-check
make rust-fixture-check
make verify-g0-result
```

## G1 — Complete one compiler/kernel vertical slice (complete)

Delivered for TinyLlama `M=1`, `[N,K]=[2048,2048]`:

- verified `Q8Linear` Region IR to Loop IR lowering;
- canonical `DFQ8_B32_OI4_V1` packing shared by scalar and NEON;
- deterministic strict scalar and output-vector NEON C generation;
- closed Apple Clang build policy and shape-aware Mach-O/disassembly audit;
- checked loading and prepared-call execution through the generated-module ABI;
- bit-exact execution across all 16 frozen fixtures and dedicated `N=4`/`N=5`
  vector/tail cases;
- a provenance-pinned three-session result bundle with all 80 observations per
  session and deterministic paired BCa analysis.

Acceptance commands:

```sh
make test-native
make verify-g1-result
```

## G2 — Native eager PyTorch boundary

G2 is intentionally a binding milestone, not a graph-compiler milestone.

### G2.1 Hardened runtime C ABI (complete)

The merged `decodeforge-bridge` crate and
[`include/decodeforge/runtime_v1.h`](../include/decodeforge/runtime_v1.h):

- expose exactly six versioned C functions;
- parse the canonical pack manifest, copy and verify the exact OI4 payload,
  build/audit/load one NEON module, and return an opaque process-local handle;
- bind the handle to static `N`, `K`, module identity, packed-weight identity,
  and byte extent;
- limit one pack to 128 MiB, all live packs to 2 GiB, and live entries to 256;
- serialize create/build, support concurrent runs, and linearize run/destroy;
- contain Rust panics and provide bounded printable-ASCII thread-local errors;
- zero failed descriptors and map concrete build/load/execution failures to a
  closed status set.

The external acceptance path builds the real release dynamic library. On Apple
ARM64 it executes the frozen `N=255,K=2` fixture with real Torch storage and
checks all 255 output words bit-for-bit; Linux proves the explicit
unsupported-host status path.

```sh
make test-bridge-cdylib
```

### G2.2 Guarded eager operator (complete)

Build a lazy Python integration with these boundaries:

1. Load only a bounded regular, non-symlink library whose caller-supplied
   `sha256:<64 lowercase hex>` identity matches. Load an owner-only immutable
   private snapshot of the verified bytes so path replacement cannot redirect
   `dlopen` after hashing.
2. Bind exact `ctypes` signatures for the frozen C ABI and translate every
   nonzero status into a structured Python exception containing the bridge's
   bounded diagnostic.
3. Wrap each native handle in an owned binding with synchronized run/close,
   idempotent cleanup, immutable descriptor data, and completed-call counters
   for native success, fallback success, native error, fallback error, and
   in-flight work.
4. Register the eager-only logical operator
   `decodeforge::q8_linear_v1(Tensor x, int binding_id, int n, int k) -> Tensor`.
   Pass input and output `data_ptr()` storage directly to the C ABI; do not copy
   the input or reconstruct the pack in Python.
5. Admit only CPU, FP32, strided, contiguous, finite, inference-only tensors
   without conjugate/negative view bits. Static `n` and `k` must match the
   binding, and every leading dimension must be one so the call is exactly
   `M=1`.
6. A guard miss may invoke an explicitly supplied same-Q8 fallback. Once a
   native call begins, any native failure is an error and must never silently
   re-run through fallback.
7. Importing base `decodeforge` must not import or require Torch. Framework
   loading remains lazy and uses the pinned CPU Torch extra.

Required tests:

- fake-ABI status, descriptor, guard, counter, and lifetime tests;
- concurrent run/close and in-flight success/error snapshots;
- replacement-after-hash regression for the private library snapshot;
- eager schema and shape tests with real Torch for `[1,1,2048]` native and
  `[1,23,2048]` fallback paths;
- direct-pointer assertions for the native input/output buffers;
- actual release-library integration against a frozen fixture;
- Linux unsupported-host behavior without requiring the Torch extra.

Focused acceptance commands (names become stable when the wrapper lands):

```sh
uv run --frozen --extra pytorch-cpu python -m pytest -q python/tests/test_torch_bridge.py
make test-bridge-cdylib
make check
```

### G2 exit evidence

- The release dynamic library is exercised through the Python wrapper and eager
  operator, not only through a mock.
- Valid `M=1` execution is bit-exact on the frozen integration fixture.
- Every guard miss, native error, fallback outcome, and lifecycle transition is
  observable and tested.
- The package remains importable without Torch installed.

## G3 — All 22 query projections in prompt-to-text generation

G3 is the first recruiter-visible product proof. It must generate text, but the
acceptance evidence is the controlled compiler boundary underneath the text.

### G3.0 Freeze the experiment before implementation (code-complete)

Record in a machine-readable spec:

- model ID `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, exact repository revision, and
  hashes of the model, tokenizer, and configuration files;
- Python, Torch, Transformers, tokenizers, safetensors, Rust, Clang, OS, and CPU
  versions;
- one fixed prompt, tokenized input IDs, greedy decoding, `use_cache=True`,
  maximum new-token count, seed, and Torch thread count; the prompt must contain
  more than one token and the run must request at least two new tokens so both
  prefill and a subsequent cached-decode call are unavoidable;
- direct operator tolerance, model-logit tolerance, token-agreement policy,
  warmup, repetitions, and session rejection policy;
- the exact distinction between offline preparation, cold startup, prompt
  prefill, cached decode, and total generation.

Do not choose tolerances or timing exclusions after inspecting the final result.

### G3.1 Prepare canonical assets for 22 `q_proj` weights (code-complete)

Implement one bounded preparation command that:

1. reads a pinned safetensors file without following symlinks or accepting a
   mutable model reference;
2. requires exactly the tensor keys
   `model.layers.{0..21}.self_attn.q_proj.weight`, each finite, bias-free, and
   exactly FP32/BF16 `[2048,2048]` after a documented conversion;
3. routes quantization and OI4 packing through the canonical Rust implementation
   rather than duplicating the physical pack algorithm in Python;
4. derives a non-trainable FP32 fallback tensor by dequantizing those exact Q8
   values in the canonical preparation path; Python must not reconstruct a
   second quantizer or OI4 unpacker;
5. emits one manifest/payload/fallback-tensor asset set per layer atomically,
   with the source tensor key and hash, model revision, logical shape, Q8 format,
   pack format, module identity, packed-weight identity, fallback tensor hash
   and parent pack identity, byte counts, and tool/source versions;
6. emits a top-level inventory whose ordered 22 entries and aggregate identity
   are deterministic.

For this shape, `B=64`, `P=512`, and each OI4 payload must be exactly
`512 * 64 * 144 = 4,718,592` bytes (4.5 MiB). All 22 payloads total
`103,809,024` bytes (99 MiB), below both the per-handle and aggregate bridge
limits. The FP32 fallback tensors are 16 MiB each and 352 MiB total; they do
not count against bridge packed-byte quotas but do count toward the G3 memory
report. Tests recompute these values instead of trusting the manifest.

Checkpoint tests:

- a tiny synthetic safetensors file proves exact extraction, ordering, and
  deterministic reruns;
- missing, duplicate, wrong-shape, nonfinite, unsupported-dtype, symlink,
  oversized, and changed-during-read inputs fail closed;
- Rust independently verifies every emitted pack identity and byte extent;
- preparing the real checkpoint yields exactly 22 unique layer entries and no
  untracked model weights are copied into the repository.

### G3.2 Prove one owning model adapter (code-complete)

Create an `nn.Module` adapter only after the low-level eager binding is stable.
The adapter must:

- own one binding and close it exactly once;
- register the prepared dequantized-Q8 FP32 `[2048,2048]` fallback as a frozen
  buffer, verify its hash/parent pack identity, and use ordinary
  `torch.nn.functional.linear` for `M>1`; never use the source FP32 weight;
- preserve the original projection's input/output shape and bias-free callable
  contract;
- send only guarded cached `M=1` CPU FP32 contiguous inference inputs to native
  code and send prompt `M>1` calls to the same-Q8 fallback;
- hard-error on a native attempt that fails, while making ordinary guard misses
  and fallback completions visible;
- expose immutable layer name, shape, module ID, pack ID, and counters for the
  result bundle.

The one-layer checkpoint runs deterministic `M=1`, `M>1`, wrong dtype, wrong
shape, noncontiguous, gradient-enabled, close, and injected-error cases. It
compares native output with the canonical same-Q8 reference under the frozen
operator policy before any model module is replaced.

### G3.3 Replace all 22 modules transactionally (code-complete)

The model integration must discover and validate all target modules before
mutating the model. It then installs one adapter at each exact layer path and
provides a cleanup operation that restores or closes every owned resource.

Acceptance invariants:

- exactly 22 expected paths are present, bias-free, `[2048,2048]`, frozen, and
  bound to the matching ordered asset entry;
- zero or 22 adapters are installed—partial replacement rolls back;
- every descriptor's pack identity matches its same-Q8 fallback identity;
- `state_dict`, evaluation mode, device restrictions, and cleanup behavior are
  explicit and tested;
- aggregate payload accounting remains below the bridge limit before any handle
  is created;
- repeated setup/teardown leaves no live bindings and no in-flight calls.

### G3.4 Run the pinned generation checkpoint (complete)

Run two paths from the same tokenized prompt and same prepared Q8 assets:

1. **same-Q8 reference:** force every adapter through fallback;
2. **hybrid native:** use same-Q8 fallback for prefill and native execution for
   every eligible cached `M=1` query projection.

The result is accepted only when:

- both paths complete with finite outputs and decode the recorded token IDs;
- direct operator errors stay within the frozen policy and the model-level
  logits/token IDs satisfy the predeclared agreement policy;
- all 22 adapters record prefill fallback coverage;
- all 22 adapters record native cached-decode coverage, with no eligible call
  silently falling back and zero native/fallback errors;
- call totals reconcile with the observed model invocations and end with zero
  in-flight work;
- the generated text is shown as demonstration output, not used as correctness
  proof by itself.

Measure preparation, module build/load, model load, first prompt prefill,
time-to-first-token, each cached decode step, q-projection dispatch/native work
where instrumentable, total generation, and peak resident memory separately.
Report cold and warmed paths; do not include offline packing in steady-state
decode latency. The all-same-Q8 path is the semantic/performance baseline. The
original FP32 model may be shown only as labeled quality/ecosystem context.

### G3.5 Check in a closed result bundle (complete)

The bundle contains:

```text
results/g3/apple-m4-primary/
  README.md
  manifest.json
  prompt.json
  asset-inventory.json
  correctness.json
  coverage.json
  timings.csv
  analysis.json
  generated.txt
  environment.txt
```

Generated native source and disassembly may be referenced by hash when they are
identical to retained compiler artifacts. Large model weights, OI4 payloads, and
dynamic libraries stay out of Git; the bundle records hashes and exact rebuild
commands. A verifier must reject modified, missing, extra, nonfinite, symlinked,
or schema-invalid evidence and recompute all summary values from raw data.

Run the pre-evidence command surface from the frozen clean checkout root. Keep
every model, tool, asset, library, receipt, session, and bundle path below one
stable external root whose parent components are real directories rather than
symlinks; source, tool, receipt, and library inputs must be single-link regular
files. Every `run-g3-demo`/session replay path is at most 1024 ASCII
characters and uses only letters, digits, `/._+-:@`; whitespace and shell/Make
metacharacters are rejected before model loading. The preparation outputs and
every session output must be new.

```sh
G3_WORK=/opt/homebrew/var/decodeforge-g3-evidence
mkdir -p "$G3_WORK"
chmod 700 "$G3_WORK"

CARGO_TARGET_DIR="$G3_WORK/cargo-target" make prepare-g3-assets-timed \
  WEIGHTS="$G3_WORK/model/model.safetensors" \
  OUTPUT="$G3_WORK/assets" \
  RECEIPT="$G3_WORK/preparation-receipt.json"
CARGO_TARGET_DIR="$G3_WORK/cargo-target" make build-g3-bridge
CARGO_TARGET_DIR="$G3_WORK/cargo-target" make test-g3

G3_LIBRARY="$G3_WORK/cargo-target/release/libdecodeforge_bridge.dylib"
G3_LIBRARY_SHA256="$(shasum -a 256 "$G3_LIBRARY" | awk '{print $1}')"

make run-g3-demo \
  SESSION_ID=apple-m4-g3-0 SESSION_INDEX=0 \
  MODEL_DIR="$G3_WORK/model" ASSETS="$G3_WORK/assets" \
  LIBRARY="$G3_LIBRARY" LIBRARY_SHA256="$G3_LIBRARY_SHA256" \
  PREPARATION_RECEIPT="$G3_WORK/preparation-receipt.json" \
  OUTPUT="$G3_WORK/session-0.json"
make run-g3-demo \
  SESSION_ID=apple-m4-g3-1 SESSION_INDEX=1 \
  MODEL_DIR="$G3_WORK/model" ASSETS="$G3_WORK/assets" \
  LIBRARY="$G3_LIBRARY" LIBRARY_SHA256="$G3_LIBRARY_SHA256" \
  PREPARATION_RECEIPT="$G3_WORK/preparation-receipt.json" \
  OUTPUT="$G3_WORK/session-1.json"
make run-g3-demo \
  SESSION_ID=apple-m4-g3-2 SESSION_INDEX=2 \
  MODEL_DIR="$G3_WORK/model" ASSETS="$G3_WORK/assets" \
  LIBRARY="$G3_LIBRARY" LIBRARY_SHA256="$G3_LIBRARY_SHA256" \
  PREPARATION_RECEIPT="$G3_WORK/preparation-receipt.json" \
  OUTPUT="$G3_WORK/session-2.json"

make analyze-g3 \
  SESSION_1="$G3_WORK/session-0.json" \
  SESSION_2="$G3_WORK/session-1.json" \
  SESSION_3="$G3_WORK/session-2.json" \
  RECEIPT="$G3_WORK/preparation-receipt.json" \
  OUTPUT_DIR="$G3_WORK/result"
make verify-g3-result BUNDLE="$G3_WORK/result"
```

`run-g3-demo` is a transparent alias of the hardened `run-g3-session` target;
it does not derive the session identity, index, library hash, receipt, or any
path. Each call starts one fresh runner process. If a session rejects, stop and
audit it rather than silently retrying the same formal capture set. The runner
authenticates the exact bridge artifact and retains a shell-safe rebuild command
whose external `CARGO_TARGET_DIR` is derived from the required
`<target>/release/libdecodeforge_bridge.dylib` layout. The checkout revision
supplies the rebuild command's working tree; no machine-local checkout path is
serialized. The library SHA-256 authenticates the exact bytes used by the run;
the rebuild command is provenance, not a promise that unrelated Cargo-home or
tool-installation paths produce a byte-identical dynamic library. Keep this
exact external target path fixed through the bridge build and all three session
runs.

The accepted ten-file bundle is checked in at `results/g3/apple-m4-primary`.
Default `make check` now verifies that canonical `G3_RESULT` in addition to G1.
The [result interpretation](G3_RESULT_2026_09_05.md) records exact measurement
boundaries and the control-token output limitation. Future captures must still
pass their own frozen acceptance checks; command completion alone is not proof.

## G4 — Choose one extension from G3 evidence

Choose exactly one first:

- bounded correctness-gated schedule selection for the existing q-projection;
- additional TinyLlama linear families, up to all 155 linears;
- general FX/`torch.compile` capture and fake/meta support;
- RMSNorm+linear or paired gate/up+SwiGLU fusion;
- native `M in {2,4,8}` prefill kernels;
- multicore output-channel scheduling;
- x86-64 AVX2 portability.

Selection follows the largest measured G3 bottleneck or the clearest missing
compiler proof. It must compare identical Q8 semantics and may honestly conclude
that a candidate does not win. No G4 feature is required for the first strong
résumé result.

## Explicit cut order

If G3 completion risk rises, remove work in this order:

1. dashboard or HTML report;
2. external library comparisons;
3. multi-prompt quality suite;
4. cold-start optimization;
5. per-layer fine-grained timers beyond coverage counters.

Do not cut the canonical 22-pack preparation, one-layer adapter checkpoint,
same-Q8 fallback identity, all-22 transactional replacement, cached `M=1`
native coverage, pinned generation comparison, lifecycle checks, or result
verification. Those are the visible project proof.

## Résumé promotion checklist

| Verb or claim | Required evidence |
|---|---|
| designed | reviewed specification or checked-in IR/ABI decision |
| implemented a Q8 compiler | deterministic typed lowering, canonical pack, retained generated source, and clean-checkout tests |
| generated ARM64 NEON | retained source plus audited machine code containing the required vector path |
| achieved `~3.96x` | exact G1 prepared-call scalar baseline, raw samples, intervals, host manifest, and boundary qualification |
| integrated with PyTorch | the real release library executes through the guarded eager operator and fallback/error paths are tested |
| accelerated TinyLlama query projections | all 22 adapters record native cached-decode coverage against the same-Q8 baseline |
| improved end-to-end generation by `X%` | G3 raw model timings and uncertainty support that exact claim; G1 cannot substitute |
| autotuned | recorded candidate set, correctness gate, selection policy, reproducible winner, and break-even analysis from a future G4 bundle |

The evidence supports: “Built and measured a Mac-first Q8 linear compiler with
generated scalar/ARM64 NEON kernels and a hardened eager PyTorch boundary;
verified native execution across all 22 TinyLlama query projections in three
independent generation sessions.” A polished text demo and broader performance
claims require separate evidence; the accepted output contains control tokens.

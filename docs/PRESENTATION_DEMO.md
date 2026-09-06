# Separate TinyLlama presentation experiment

This experiment addresses the user-facing text limitation of the accepted G3
capture. It is not a new G3 session, a replacement result, or a benchmark.
The frozen `benchmarks/g3/spec.json` and `results/g3/apple-m4-primary` remain
unchanged.

## Question and pre-run configuration

Does the pinned chat model produce useful text when its own local chat template
is used, with an explicit assistant-generation prefix and normal EOS stopping?

- Use the same pinned local TinyLlama model/tokenizer, canonical Q8 assets and
  verified native bridge as the accepted capture. No network loading or remote
  model code is allowed.
- Keep the user message `Write one short sentence about a compiler.`
- Apply the tokenizer's chat template to that user message with
  `add_generation_prompt=True`. Do not manually append a second BOS token.
- Use CPU FP32 eager evaluation, deterministic greedy selection, the existing
  all-22 projection adapters, and cached single-token decode.
- Generate at most 64 new tokens; stop at the first EOS. Do not carry over G3's
  two-token minimum or suppress EOS to manufacture a sentence.
- Run once through the same-Q8 reference and once through hybrid native
  execution. Keep both raw token IDs and raw/clean decoded text.

## Checks and interpretation

Require exact token-ID agreement across the two paths, prefill fallback and
native cached-decode coverage across all 22 layers, and complete restoration
with no live adapters or in-flight calls after cleanup. Failures must not emit
a success-looking report. A token sequence that ends immediately without any
native decode is not evidence of native execution.

Inspect the actual text separately: numerical agreement does not imply a
meaningful answer. Report whether EOS was reached or the token limit truncated
the answer. Preserve failed or unhelpful observations when describing the
experiment; do not silently replace the prompt after seeing an output.

This two-path demonstration does not have G3's repeated sessions, drift checks,
or full correctness instrumentation. It cannot support a performance claim or
replace the accepted numerical evidence. Any further prompt or sampling change
must be recorded as a distinct experiment.

## Command

After preparing the existing G3 assets and release bridge, run the standalone
entry point from the active checkout. Replace the example absolute paths and
bridge digest with the verified local artifacts; the digest is 64 lowercase
hexadecimal characters without a `sha256:` prefix.

```sh
UV_OFFLINE=true uv run --frozen --extra g3-generation \
  python scripts/run_presentation_demo.py \
  --model-dir /absolute/path/to/model \
  --assets /absolute/path/to/assets \
  --library /absolute/path/to/libdecodeforge_bridge.dylib \
  --library-sha256 BRIDGE_SHA256 \
  --prompt 'Write one short sentence about a compiler.' \
  --max-new-tokens 64 \
  --output /absolute/path/to/new-presentation-result.json
```

This command does not replace `make run-g3-session` or `make run-g3-demo` (the
latter is an alias for the frozen session runner). Its output is deliberately
marked `benchmark: false` and `g3_evidence: false`.

## Observed result — 2026-09-05 (local date)

The original question was run from clean source revision
`662d137ebb7e834ef15d7de486185dcf880a8dd8` on the Apple M4. The
[retained presentation JSON](../results/presentation/apple-m4-chat-template-v1.json)
records the rendered input, exact tokens, file identities, per-layer counters
and cleanup state. Both paths produced this same text:

> A compiler is a software tool that translates high-level programming languages
> into machine code. It is responsible for converting complex syntax into a
> series of instructions that can be executed by a computer's processor.

- Both paths generated 42 tokens, including EOS; neither hit the 64-token cap.
- Each of the 22 reference layers made 42 same-Q8 fallback calls.
- Each of the 22 hybrid layers made one prefill fallback and 41 native cached
  decode calls, with no dispatch errors.
- Exact token IDs agreed; all original modules were restored, with zero
  installed/live adapters and in-flight calls after cleanup.

The template change resolves the control-token-only presentation failure for
this prompt. It does not establish general instruction-following quality:
the model returned two sentences despite the request for one. No timing or
speedup is claimed by this presentation experiment, and no new full G3-style
logit or native-output comparison was performed.

The accepted G3 result remains byte-for-byte unchanged. This presentation used
the previously verified assets and bridge from `capture-ad15f5d`, with bridge
SHA-256 `ea9132762af1847599256a984021bd1a99dff139f55e01867bf38250702d9e94`.
The standalone loader checks the pinned files before and after loading; it is
not the descriptor-snapshot/provenance protocol of the frozen G3 runner.

An earlier presentation implementation attempt at `e482d0a` was rejected before
generation because its copied special-token-map digest had a typo. No report
was emitted. Revision `662d137` corrected the digest and added a regression
that compares every presentation model pin with the frozen G3 model records.
The prompt and generation settings were unchanged between those attempts.

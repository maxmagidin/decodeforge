# DecodeForge broader evaluation V1

`benchmarks/evaluation-v1/spec.json` is the predeclared contract for the
broader TinyLlama query-projection evaluation. It is deliberately separate
from the frozen G3 experiment and does not change or replace
`benchmarks/g3/spec.json` or `results/g3/apple-m4-primary`.

The evaluation answers four narrower questions:

1. Does the all-22 adapter installation, dispatch split, and teardown remain
   correct over a small but varied chat corpus?
2. Do same-Q8 reference and hybrid-native execution produce the same logits
   within the inherited G3 tolerance and the same greedy token IDs?
3. How does Q8 behave in the context of the original FP32 model on fixed
   teacher-forced sequences?
4. What are the guarded, end-to-end prefill/decode costs in three fresh
   processes on the available host?

It is not a general instruction-following or human-preference benchmark. The
30 prompts are original, project-authored synthetic sensitivity probes. Their
`reference_text` values are fixed sequences for a reproducible numerical
probe, not answer keys and not a claim that the model must produce those
words.

## Frozen inputs and decoding

The spec pins the same local TinyLlama revision as G3 and refers to G3 for the
model file identities. Runs are offline and CPU FP32, with evaluation and
inference mode enabled, one Torch thread and one inter-op thread, and seed
0. Every case is one user message passed through the tokenizer's native
chat template with `add_generation_prompt=True`; the rendered input IDs and a
hash of the rendered template are retained in the result.

There are 30 distinct prompts, ten in each short, medium, and long input
category. Output caps (16, 32, and 64) occur exactly ten times each. Inputs
must tokenize to at most 512 tokens;
the fixed reference sequence must remain at most 640 tokens after adding the
chat-template input. The performance subset is explicitly named by
the top-level `performance_case_ids` rather than selected after observing
results:

| Case | Input class | Cap |
| --- | --- | --- |
| `short-01-16` | short | 16 |
| `medium-02-32` | medium | 32 |
| `long-03-64` | long | 64 |

Generation is deterministic greedy decoding (`do_sample=false`, one beam,
`use_cache=true`). It stops only at the tokenizer's natural EOS or the case's
`max_new_tokens`; `min_new_tokens=0` and `stop_after_sentence=false`. No
sentence-boundary stopping, post-hoc truncation, prompt replacement, or
sampling is permitted. Cases used for dispatch coverage must generate at least
two tokens so that prefill and cached single-token decode are both observed;
an early natural EOS is retained as a failed/inconclusive coverage case, not
silently suppressed.

## Correctness and lifecycle

Each path uses the same ordered 22 Q-projection assets and the same Q8 pack
identities:

- `same_q8_reference`: same-Q8 fallback for every q-projection forward;
- `hybrid_native`: same-Q8 fallback for prompt prefill, then native dispatch
  for every eligible cached `M=1` decode.

For each case, retain the input IDs, full output IDs, generated IDs, stop
reason, per-step finite logit comparison metrics, and per-layer counters. Full
vocabulary logits are transient and are not retained. The comparison is made
at every common prefix step. The inherited numeric condition is

```text
abs(native - reference) <= 0.001 + 0.001 * abs(reference)
```

with finite values required. Exact full-sequence token-ID equality is a
separate mandatory check; matching decoded strings alone is insufficient.
The same-Q8/native comparison requires identical prefixes at each step before
comparing logits, so a token divergence cannot be hidden by comparing
unrelated continuations.

The lifecycle verifier checks, before generation, 22 open adapters with
`installed_modules=22`, `restored_modules=0`, `live_adapters=22`, and no
in-flight calls. Every layer must have the canonical ordered path
`model.layers.{0..21}.self_attn.q_proj`. For each path it checks all-22
coverage, fallback/native dispatch counts, zero native/fallback/pre-dispatch
errors, zero rejected-closed calls, and zero in-flight calls after each case.
After `close`, it requires `installed_modules=0`, `restored_modules=22`,
`live_adapters=0`, no in-flight calls, and identity restoration of every
original module. A second close must be harmless and is checked explicitly.

The verifier does not infer coverage from a generation-library call count. It
reconciles observed per-adapter counters with the actual number of prefill and
cached decode forwards. This preserves the G3 lifecycle lesson that an
installation-time count and a post-cleanup live count are different facts.

## Original FP32 context and fixed-sequence probe

The untouched CPU FP32 model (evidence path key `fp32`) is run with the same chat template and greedy
settings as the two adapter paths. Its generated IDs, token count, natural
stop reason, and decoded text are retained as contextual baseline output. No
claim of answer quality is made, and the report must not synthesize a quality
score from these outputs.

The numerical sensitivity probe uses the fixed `reference_text` in
each case. Construct one target sequence from the chat-template input IDs,
the tokenizer's IDs for that reference text; do not append EOS. Feed that
identical sequence with teacher forcing through original FP32 and same-Q8
reference. (A teacher-forced sequence has M>1 and therefore exercises fallback
rather than the native cached-decode path.) For every
target position, retain:

- the target token ID;
- each path's finite per-token negative log-likelihood;
- each path's argmax token ID; and
- argmax agreement with original FP32.

Report per-token values and their means, plus the raw agreement fractions.
These are descriptive NLL and argmax-agreement measurements on a fixed
sequence, not an invented quality grade, pass percentage, perplexity claim,
or broad language-model evaluation. There is no arbitrary NLL threshold that
can turn a sensitivity probe into a quality verdict. Any numerical failure
(non-finite logits/NLL or a path execution error) remains a correctness
failure; ordinary FP32-vs-Q8 differences are reported without relabeling them
as product quality.

## Practical performance protocol

Performance is deliberately modest: three fresh processes, each running one
warmup and three measured generations per path for the three predeclared
performance cases. Original FP32, same-Q8 reference, and hybrid-native paths
(`fp32`, `same_q8_reference`, and `hybrid_native` in evidence) are paired within
each process. Startup, model/tokenizer load, runtime verification and
all-22 installation are reported separately from warmed generation. Setup is
the sum of model-preparation and runtime/installation components, excluding
intervening FP32 evaluation runs. RSS is the operating-system process-lifetime
peak sampled after cleanup, including interpreter/import memory.

Retain raw `perf_counter_ns` samples and report per-process median/min/max for:

- startup component timing from the CLI timestamp, after interpreter/standard
  library startup but before importing the evaluation module; direct API calls
  instead begin timing at model preparation;
- prompt prefill (one fixed input forward without past key values);
- each cached `M=1` decode step and aggregate decode throughput; and
- total generation from prefill start through final token selection.

Text decoding, JSON serialization, setup, teardown, and result comparison are
outside the timed regions. The prefill interval ends at model return before
the outer finite-logit validation; TTFT and total-generation intervals include
that validation. A warmed generation number must not hide cold startup or
asset installation.

Production guards remain enabled. Their overhead is included in the timed
production dispatch and reported as included rather than isolated. The report names their contribution,
including identity-bound asset checks, shape/dtype/device/stride checks,
native eligibility checks, status decoding, and finite-output validation. No
guard-bypass comparison or mutation of production guards is part of this
protocol. Hooks, activation capture, output
comparisons, weight cloning/hashing for evidence, and teacher-forced scoring
are not inside any performance timer. The production result therefore
measures the guarded implementation boundary without turning instrumentation
into a claimed kernel speedup.

The reference fallback's production clone/hash work, when performed by the
adapter, remains in the timed path and is disclosed; it is not removed to
improve the number. Output comparisons and teacher-forced scoring happen only
outside timing regions.

No second physical Mac is available. The reproducibility claim is consequently
limited to three fresh processes from the same-host clean checkout. The result
must record the clean git revision, model/tokenizer/Q8/bridge identities,
software and host details, thread settings, relevant environment variables,
rendered-input hashes, token IDs, raw timing samples, and exact command line.
No cross-host conclusion or confidence interval may be inferred from this
evaluation. A paired original-FP32 comparison is allowed only for the named
measured boundaries and must not be generalized into an unqualified
stock-PyTorch speedup claim.

## Result acceptance and retention

Emit a success result only if artifact identity, tokenization, finite outputs,
same-prefix logit checks, exact same-Q8/native IDs, all-22 dispatch/lifecycle
checks, and required timing samples pass. Keep rejected attempts and their
reason fields; do not silently retry or replace a case. A successful report
must retain raw observations sufficient to recompute the summaries. This
evaluation's result is additional evidence and must never be copied into the
frozen G3 result bundle.

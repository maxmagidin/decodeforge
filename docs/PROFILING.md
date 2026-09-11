# Decode profiling

DecodeForge's kernel benchmark answers a narrow question: how quickly does one
generated Q8 projection run? The diagnostic profiler answers the next one:
where does time go during complete cached generation?

This distinction matters. Instrumentation adds observer overhead, and a fast
kernel can still be hidden by tensor preparation, adapter checks, fallback
work, the rest of the transformer, output validation, or token selection. A
profile is therefore a tool for choosing the next experiment—not proof of a
speedup.

## What the profiler records

[`decode_profile.py`](../python/decodeforge/decode_profile.py) wraps one greedy,
CPU, batch-one generation with explicitly nested spans:

- complete generation;
- prompt prefill and each cached-decode step;
- input preparation, model forward, output validation, token selection, and
  bookkeeping; and
- selected module forwards inside the model.

`tinyllama_component_paths()` selects 157 nonoverlapping components: token
embedding; both layer norms, query/key/value/output projections, and the whole
MLP in each of 22 transformer layers; final norm; and language-model head.
Whole MLPs are selected instead of their nested projections so direct children
do not overlap.

Each event includes inclusive and exclusive nanoseconds. Exclusive time is the
event duration minus its direct child spans. In particular, exclusive
`model_forward` time is an **unattributed remainder** that includes unselected
model work and profiler overhead; it is not a pure estimate of "everything
else."

For installed query-projection adapters, dispatch is derived from the adapter's
before/after counter delta. It is not guessed from tensor shape. Events can be
labelled `native`, `fallback`, an error path, or `ambiguous` when the observed
counters do not describe exactly one forward.

Detailed query-projection profiling adds nested diagnostic spans without
changing the normal adapter or bridge implementations. The spans cover the
adapter's fallback-storage checks, the existing fallback clone, identity hash,
and linear operation, the guarded native operator, and the runtime binding
call. The guarded native operator includes PyTorch dispatch, its internal input
guard, output allocation, and binding execution. The binding span includes
locking, pointer checks, the C ABI call, and status handling. Neither boundary
is kernel-only timing. The outer native eligibility check remains in the
query-projection event's exclusive remainder.

## Programmatic use

```python
from decodeforge.decode_profile import (
    profile_cached_generation,
    tinyllama_component_paths,
)

profile = profile_cached_generation(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=16,
    module_paths=tinyllama_component_paths(),
    qproj_details=True,
)
document = profile.to_wire()
```

The returned `CachedGeneration` keeps the generated token IDs beside the trace,
making behavioral parity easy to check against an uninstrumented control. The
preliminary diagnostic document identifies itself as `diagnostic_profile` and
sets `performance_claim_allowed` to `false`; it is not yet a retained evidence
contract or independently verifiable schema.

## Fresh-process session capture

[`run_profile_capture.py`](../scripts/run_profile_capture.py) turns the API into
one reproducible local session. Each invocation verifies the pinned checkpoint,
loads and installs all 22 query-projection adapters, and then runs both same-Q8
reference and hybrid-native modes. Each mode gets one warmup, one matched
unprofiled control, and one profiled generation. Odd session indexes reverse
both path and observer order to make order effects visible.

Run it three times from three new interpreter processes:

```sh
uv run --frozen --extra g3-generation python scripts/run_profile_capture.py \
  --model-dir /absolute/path/to/model \
  --assets /absolute/path/to/assets \
  --library /absolute/path/to/libdecodeforge_bridge.dylib \
  --library-sha256 <64-lowercase-hex-digits> \
  --prompt "Write one short sentence about a compiler." \
  --max-new-tokens 16 --session-index 0 \
  --output /absolute/path/to/profile-0.json
```

Repeat with session indexes `1` and `2` and distinct output paths. A supplied
index is only a label; separate CLI launches provide process isolation. Output
publication refuses existing files, symlinks, and symlinked parent directories.

The control executes the same explicit cached-generation workload without
hooks or timer reads. The recorded outer ratio is an observer-cost indicator,
not a corrected timing: hook installation, collection, and cleanup are included
in the profiled side, and no overhead is subtracted. Every run retains its own
adapter counter delta, generated IDs, stop reason, and elapsed time. The runner
rejects missing cached-native coverage, path disagreement, incomplete teardown,
or a profile whose q-projection dispatch events disagree with the counters.

## Safety and limitations

Profiling is opt-in: no hooks or timer reads are added to existing G1, G3, or
evaluation-v1 measurement paths. Hook installation is transactional, cleanup
runs after failures, shared module aliases are rejected, and overlapping
profiles of one model are not allowed. One trace must remain on one Python
thread so nested event order stays unambiguous.

Current support is intentionally bounded to synchronous CPU inference, one
sequence, greedy cached generation, and at most 64 new tokens. Timings include
Python hooks and clock reads and do not subtract their cost. A capture also
limits selected modules, event count, nesting depth, and module-path length so
unexpected model behavior cannot grow a trace without bound. The output is not
a retained benchmark artifact and must not be used for a resume or performance
claim.

## Comparing three sessions

[`analyze_profile_sessions.py`](../scripts/analyze_profile_sessions.py) reads the
three captures and recomputes their cost attribution:

```sh
make analyze-profile \
  SESSION_1=/absolute/path/to/profile-0.json \
  SESSION_2=/absolute/path/to/profile-1.json \
  SESSION_3=/absolute/path/to/profile-2.json \
  OUTPUT=/absolute/path/to/new-analysis.json
```

The captures must come from the same clean source revision and have matching
source hashes, environment, checkpoint, quantized assets, bridge, and workload.
The analyzer requires indexes `0`, `1`, and `2`, with distinct recorded process
IDs. Those IDs are a consistency check; the JSON cannot prove process isolation
or authenticate the producer. Launch the capture command separately each time.

The analyzer revalidates the complete nested traces, token agreement, adapter
counter continuity, execution order, and final restoration. It recomputes each
observer ratio from the control and profile durations. Missing fields,
inconsistent claims, duplicate JSON keys, and oversized input files are rejected.
An existing output is never replaced. The report includes SHA-256 hashes of the
exact input files, so keep all three raw captures alongside it.

### Reading the result

For a readable Markdown report, repeat the same command with
`REPORT_FORMAT=markdown` and a new `OUTPUT=/absolute/path/to/analysis.md`.
The direct CLI equivalent is `--output-format markdown`. Both formats validate
the raw captures before writing; Markdown is a presentation of the same analysis,
not a separate evidence format.

The Markdown report explains each mode's largest cost and ranking stability,
then shows a compact cost table. Times are the median across sessions of each
session's average milliseconds per cached step. The three share columns retain
each session's own fraction of cached-decode time. This displayed median order
is descriptive and does not replace the per-session stability checks. Prompt
text, command lines, process IDs, and developer paths are omitted from the
readable report; source and input-file hashes identify its inputs.

For each session and execution mode, the report separates prompt prefill from
cached decode. Each cost bucket adds **exclusive** event durations: time spent
in a child is not counted again in its parent. These buckets sum to the measured
phase duration. The report retains time outside the steps separately as the
generation remainder.

Query-projection internals have their own buckets. Other module events are
grouped by component family, such as MLP, key projection, or layer norm. Adapter,
model-forward, and step remainders remain visible; they include uninstrumented
work and observer overhead. The native binding bucket still includes locking,
pointer checks, and status handling, so it is not kernel-only time.

`stability` compares cached-decode rankings separately for the same-Q8 reference
and hybrid-native paths. It reports agreement across the **entire ordering** and
agreement on the **top group** separately. Equal totals form an explicit tie
group. An alphabetical tie-break never turns a tie into a winner, and averaging
sessions never conceals a changed order. Exact ordering agreement is descriptive,
not a statistical confidence test; close totals can change order with noise.

`observer` contains outer control/profile call times and their ratio, including
profile hook setup and restoration outside the generation-root span.
The control has no per-step timers, so this ratio cannot correct individual
cached-decode costs. No observer overhead is subtracted, and the report always
sets `performance_claim_allowed` to `false`.

## Completing P0

The capture and analysis tools are available. Completing the empirical milestone
still requires three comparable real captures with stable cost attribution. An
unstable ordering is a useful result, but does not satisfy that criterion. Keep
it visible and investigate workload duration and measurement noise before
selecting an optimization.

Once the largest stable, actionable cost is established, fix the optimization's
acceptance rule and baseline before measuring it. The outer native eligibility
check remains in the adapter remainder; profiling does not isolate every guard.

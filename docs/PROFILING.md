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
)
document = profile.to_wire()
```

The returned `CachedGeneration` keeps the generated token IDs beside the trace,
making behavioral parity easy to check against an uninstrumented control. The
preliminary diagnostic document identifies itself as `diagnostic_profile` and
sets `performance_claim_allowed` to `false`; it is not yet a retained evidence
contract or independently verifiable schema.

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

## Completing P0

The instrumentation is the foundation, not the conclusion. Completing P0 also
requires separate spans for adapter guards, fallback weight cloning and
hashing, and guarded native bridge execution. Only after those sub-boundaries
exist should the capture run in fresh processes and include:

1. an uninstrumented control to verify tokens and estimate observer impact;
2. three profiled runs for same-Q8 reference and hybrid-native execution;
3. raw traces plus environment, model, asset, and bridge identities; and
4. a cost ranking that remains stable across the three runs.

Only then should the largest stable, actionable cost determine the first
optimization. The acceptance rule and baseline must be fixed before that
optimization is measured.

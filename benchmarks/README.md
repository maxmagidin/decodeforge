# DecodeForge benchmark protocols

Each benchmark answers one bounded question. Its inputs, timed boundary,
acceptance rule, and saved evidence are fixed before the result is interpreted.

| Protocol | Question | Current result |
| --- | --- | --- |
| [`g1/spec.json`](g1/spec.json) | Does generated NEON improve one same-Q8 projection over generated scalar? | Approximately 3.96× across three accepted M4 sessions |
| [`g3/spec.json`](g3/spec.json) | Do all 22 query projections execute natively during cached decode and restore cleanly? | Three accepted sessions with exact tokens and clean restoration |
| [`evaluation-v1/spec.json`](evaluation-v1/spec.json) | How do native, same-Q8, and original FP32 paths compare across correctness, sensitivity, and practical timing? | 30/30 correctness cases; native does not consistently beat FP32 |

The [benchmark methodology](../docs/BENCHMARKS.md) defines shared timing,
statistical, baseline, and retention rules. The
[evaluation protocol](../docs/EVALUATION_V1.md) defines the broader model study.

Read the human-facing [G1 kernel result](../results/g1/apple-m4-primary/README.md),
[G3 integration result](../results/g3/apple-m4-primary/README.md), or
[broader evaluation](../results/evaluation/apple-m4-v1/README.md). Recompute
their checked-in analyses without running or downloading the model:

```sh
make verify-g1-result verify-g3-result verify-evaluation-result
make verify-results-visual
```

Regenerate the checked-in SVG overview directly from those retained summaries:

```sh
make render-results-visual
```

Captured performance is host- and boundary-specific. CI runs correctness and
evidence verification but intentionally enforces no performance threshold.

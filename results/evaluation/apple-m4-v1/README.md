# Apple M4 broader evaluation V1

All four captures passed from clean source
`535540ee2e76d5965639af786d4401623865bc0c`: one 30-prompt correctness/quality
process and three sequential performance processes. The
[predeclared protocol](../../../docs/EVALUATION_V1.md) and original synthetic
prompts were frozen before any model outputs were collected. This is separate
from accepted G3; neither its specification nor its evidence was changed.

## Correctness and quantization sensitivity

- **30/30 cases passed**, covering 1,070 generated tokens with exact
  same-Q8/native agreement. Maximum absolute model-logit difference was
  `0.0000171661376953125`, within the declared elementwise tolerance.
- Across the 1,040 cached generation steps, counters recorded 22,880 native
  calls: one query projection in each of TinyLlama's 22 layers.
  Cleanup restored the exact original modules: 0 installed, 22 restored,
  0 live adapters, 0 in-flight calls.
- All **30 greedy sequences also matched original FP32** on this corpus.
- On 1,034 fixed reference tokens, token-weighted mean NLL was
  **3.4827915980 FP32** versus **3.4825643588 same-Q8**, a difference of
  **-0.0002272392 nats/token**. Argmax agreement was **99.7099%** (1,031/1,034).
  The tiny negative difference is not evidence of generally improved quality.
- Rendered chat input lengths were 25–30 tokens (short), 33–40 (medium), and
  158–202 (long), with output caps balanced across 16/32/64 tokens.
- **29/30 generations reached their cap; only one stopped at EOS.** No sentence
  stopping was enabled. These results establish implementation agreement, not
  reliable instruction following or naturally completed answers.

## Practical performance

Each process used one warmup and three measured generations per path for three
fixed cases: **81 measured generations and 27 warmups** across all processes.
All 81 measured token sequences also matched the corresponding correctness
capture. Below are ranges of the **three per-process medians**, not confidence
intervals or selected best runs.

| Case / output cap | FP32 decode tokens/s | Hybrid-native decode tokens/s | Guarded same-Q8 reference tokens/s |
| --- | --- | --- | --- |
| Short / 16 | 11.61–15.47 | 11.29–14.74 | 3.84–4.70 |
| Medium / 32 | 14.78–14.86 | 14.17–14.29 | 4.27–4.53 |
| Long / 64 | 12.23–13.74 | 10.85–13.23 | 4.37–4.47 |

| Case / output cap | FP32 total generation seconds | Hybrid-native total generation seconds | Guarded same-Q8 reference seconds |
| --- | --- | --- | --- |
| Short / 16 | 1.102–1.423 | 1.301–2.542 | 3.468–4.193 |
| Medium / 32 | 2.264–2.278 | 2.502–2.516 | 7.171–7.580 |
| Long / 64 | 5.032–5.589 | 5.336–6.479 | 14.672–15.155 |

The native path is faster than the guarded same-Q8 reference here, but **does
not establish a consistent advantage over original FP32**. The medium case
is slower than FP32 in every process; short/long results show variability.
The slow short/native observation in process 2 is retained, not excluded or
silently rerun. FP32 ran first in each process, followed by the Q8 paths with
their order reversed in process 1; this does not eliminate thermal/order effects.

Fresh-process setup components totaled **16.30–16.63 seconds**, using existing
pinned artifacts and reusable OS/file caches. This is not cold-storage or
offline asset-preparation timing. Peak process RSS was **4.62–4.80 GiB**;
it includes all paths and their setup, not a separate memory measurement for
each implementation. The retained summary also contains prefill medians and
all setup components for each process.

No evidence hooks, activation cloning for comparison, or teacher-forced scoring
ran inside the performance region. Production guards remained enabled, including
the reference fallback's clone/hash work. Outer finite-logit checks are included
in TTFT, cached decode and total-generation timing but not the prefill-forward
interval. These are guarded model-boundary measurements, not isolated kernels.

## Reproducibility and verification

The capture used a separate clean checkout with an isolated uv environment on
the same Apple M4, reusing Rust/uv caches and pinned external model/Q8/bridge
artifacts. The exact capture head passed full offline `make check` with
**497 Python tests**, Rust debug/release and native suites, fixture checks,
and G1/G3 verification. Strict Rust preflight and G0 provenance also passed.
The earlier harness head `d180594` passed the aggregate with 496 Python tests.
Their full logs remain under `/opt/homebrew/var/decodeforge-evaluation-v1`.

Hosted macOS also exercised an actual missing-link repair and then passed
normal and offline preflight in
[run 34003095842](https://github.com/maxmagidin/decodeforge/actions/runs/34003095842).
That is toolchain evidence, **not a second-Mac model evaluation**. A second
physical host remains untested; there is no cross-host performance claim.

The five JSON files here are byte-identical copies of the external captures
and recomputed summary. To verify their summary, identities, dispatch counters,
sample counts, and token agreement without downloading or running a model:

```sh
make verify-evaluation-result
```

This verification is also part of `make check`. It does not independently
re-execute the model or recover full-vocabulary logits from retained error
metrics. Raw observations and failures must remain visible when interpreting
the results; the summary is not a general quality grade or a release approval.

The next performance work should profile and reduce the guarded boundary and
prefill overhead, preserving numerical/lifecycle checks. Broader capability
claims need a larger task-quality evaluation; reproduction on another physical
Mac is still required for a cross-host claim.

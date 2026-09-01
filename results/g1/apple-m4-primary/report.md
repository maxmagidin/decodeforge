# DecodeForge G1 paired benchmark

Protocol: `g1-prepared-call-paired-v1`

| Case | Session speedups (95% BCa CI) | Pooled descriptive speedup | Claim |
| --- | --- | --- | --- |
| `tinyllama-q-proj-2048x2048` | 3.95671 [3.95103, 3.96705]<br>3.96176 [3.95085, 3.9696]<br>3.95648 [3.95351, 3.95997] | 3.95828 (point only) | allowed |

The aggregate is pooled-descriptive (not used for the claim).
BCa intervals use exactly 10,000 deterministic paired resamples; each pair remains intact.
Latency summaries normalize every batch as elapsed_ns / repetitions and report median, median absolute deviation, and nearest-rank p95 in ns/invocation.

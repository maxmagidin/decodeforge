# DecodeForge G3 TinyLlama generation evidence

Protocol: `g3-tinyllama-qproj-generation-v1`

This closed bundle contains three independently captured, schema-valid
accepted sessions. Generated text is demonstration output only; direct
operator, model-logit, token, and dispatch evidence determine correctness.

Accepted sessions: 3

| Path | Measured samples | Minimum ns | Median fraction ns | Maximum ns |
| --- | ---: | ---: | ---: | ---: |
| `same_q8_reference` | 30 | 1657430500 | 3464222208/2 | 1900694916 |
| `hybrid_native` | 30 | 714516417 | 1502683166/2 | 919110208 |

## Timing boundary and interpretation

The measured generation paths use guarded, instrumented `q_proj` adapters and are not isolated kernel timings. Hook instrumentation clones projection inputs and outputs. The same-Q8 fallback also clones and hashes its FP32 fallback weight on every call; those integrity costs are included in the reported generation timings.

`native_work_ns` is unavailable because the bridge exposes no kernel-only timer. The recorded dispatch and generation timings therefore include the guarded boundary and instrumentation. Offline preparation and cold-start components are recorded separately and are excluded from the warmed samples in the table above.

Coverage is limited to the 22 TinyLlama `q_proj` adapters and their cached single-token decode calls. This bundle does not establish a blanket whole-model speedup or a comparison with stock PyTorch.

`analysis.json` retains the canonical session objects needed to
reconstruct and independently regenerate every bundle member.

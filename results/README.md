# DecodeForge results

This directory contains the evidence behind DecodeForge's public claims. Start
with the human-readable README in each result family; machine-readable files
retain raw observations, identities, generated source, audits, and summaries.

![DecodeForge Apple M4 results overview](../docs/assets/decodeforge-results-overview.svg)

| Result | What it establishes | What it does not establish |
| --- | --- | --- |
| [G0 Apple M4](g0/apple-m4-primary/sha256-311053f53efd9c28ab3e4338ca83e78e53acf8c969d9f8a76c6e56f7c2d79d86/report.md) | Python/Rust Q8 semantic and fixture agreement bound to source and host provenance | Generated-code or performance behavior |
| [G1 Apple M4](g1/apple-m4-primary/README.md) | Approximately 3.96× generated NEON/scalar speedup for one same-Q8 projection boundary | Whole-model or stock-PyTorch speedup |
| [G3 Apple M4](g3/apple-m4-primary/README.md) | All-22 query-projection native coverage, exact same-Q8 tokens, lifecycle reconciliation | Broad model quality or all-linear coverage |
| [Broader Apple M4 evaluation](evaluation/apple-m4-v1/README.md) | Native/reference correctness, Q8/FP32 sensitivity, and separate model timing | Consistent FP32 advantage or cross-host generalization |
| [Presentation runs](presentation) | Human-readable output under explicitly varied formatting choices | Replacement for the frozen G3 or broader evaluation |

Verify retained analyses and identities without rerunning TinyLlama:

```sh
make verify-g0-result verify-g1-result verify-g3-result verify-evaluation-result
make verify-results-visual
```

The summaries are reproducible views over retained evidence, not substitutes
for it. New benchmark claims must preserve raw samples and rejected outcomes,
name the exact comparison boundary, and follow the relevant frozen protocol.

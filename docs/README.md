# DecodeForge documentation

Use this page to choose the shortest document that answers your question.

## Start here

| Goal | Document |
| --- | --- |
| Understand the project without compiler background | [Reader's primer](PRIMER.md) |
| Review the strongest results and limitations | [Apple M4 evaluation](../results/evaluation/apple-m4-v1/README.md) |
| Follow the implementation architecture | [Technical design](DESIGN.md) |
| Reproduce or contribute | [Contributing guide](../CONTRIBUTING.md) |

## Normative contracts

| Document | Authority |
| --- | --- |
| [Q8 format V1](Q8_FORMAT_V1.md) | Numerical behavior, serialization, identities, and comparison rules |
| [Technical design](DESIGN.md) | IR, packing, code generation, native module, runtime, and PyTorch boundaries |
| [Benchmark methodology](BENCHMARKS.md) | Timing boundaries, statistics, baselines, retention, and claim policy |
| [G0 evidence V1](G0_EVIDENCE_V1.md) | Provenance and correctness-bundle contract |
| [Evaluation V1](EVALUATION_V1.md) | Frozen broader model experiment and acceptance rules |

## Delivery and review

| Document | Purpose |
| --- | --- |
| [Implementation plan](IMPLEMENTATION_PLAN.md) | Gate definitions, command ownership, and scoped extensions |
| [Progress snapshot](PROGRESS_2026_09_05.md) | Completed G0–G3 state and remaining work at the recorded date |
| [Review stack](REVIEW_STACK.md) | Historical integration and validation sequence |
| [Codebase review](CODEBASE_REVIEW_2026_09_05.md) | Findings from the final implementation review |
| [G3 capture audit](G3_CAPTURE_AUDIT_2026_09_05.md) | Model-evidence capture and lifecycle review |
| [G3 result interpretation](G3_RESULT_2026_09_05.md) | Boundaries on the accepted generation result |

## Demonstration guides

- [Presentation demo](PRESENTATION_DEMO.md) explains the interactive
  chat-formatted run.
- [Presentation polish](PRESENTATION_POLISH.md) records sentence-formatting
  experiments without changing the frozen correctness result.

## Architecture decisions

- [ADR 0001](decisions/0001-mac-first-required-path.md): make Apple M4 the
  required path and defer AVX2.
- [ADR 0002](decisions/0002-strict-output-lane-oi4-pack.md): use strict
  output-lane vectorization and OI4 packing.
- [ADR 0003](decisions/0003-scalar-generated-abi.md): freeze the generated
  scalar ABI.
- [ADR 0004](decisions/0004-strict-output-vector-neon.md): freeze strict
  output-vector NEON lowering.
- [ADR 0005](decisions/0005-prioritize-eager-q-projection-demo.md): prioritize
  the eager all-query-projection model proof.

Historical documents are retained because the repository treats decisions and
evidence as versioned artifacts. The current public overview lives in the
[root README](../README.md).

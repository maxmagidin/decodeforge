# ADR 0001: Make the Apple M4 path the required vertical slice

- **Status:** Accepted
- **Date:** 2026-08-30
- **Decision owner:** Repository owner

## Context

The earlier plan made both ARM64 NEON and x86-64 AVX2 mandatory before the
compiler could reach framework integration. That made a second computer part
of the critical path and risked turning a buildable compiler project into a
cross-host research program. The owner explicitly chose to complete and
measure the project on the local Apple M4 first.

## Decision

G0 still requires independent Python and Rust implementations of the portable
`DFQ8_B32_V1` semantics. After G0, the required résumé path is:

1. generated scalar plus ARM64 NEON on the Apple M4;
2. bounded, correctness-gated schedule selection on that Mac;
3. guarded `torch.compile` integration on that Mac.

An x86-64 AVX2 backend remains designed as a retargeting extension, but it is
not a G0–G3 acceptance dependency. It may be selected at G4 after the local
compiler path works end to end. No AVX2 or cross-host claim is permitted without
its own correctness, assembly, and measurement evidence.

## Consequences

- The complete required project can be built and demonstrated on one owned
  machine.
- Target-independent IR, format, manifests, and ABI boundaries must remain
  retargetable; Mac-first does not permit hard-coding the compiler to one model
  or silently weakening target guards.
- Public status and résumé language name only completed gates. AVX2 remains a
  future option, not implied evidence.
- This decision supersedes the earlier sequencing requirement that AVX2 finish
  before bounded schedule selection and PyTorch integration.

## Revisit conditions

Revisit only after G3 evidence exists, or if an accessible second host makes
AVX2 the highest-value measured extension.

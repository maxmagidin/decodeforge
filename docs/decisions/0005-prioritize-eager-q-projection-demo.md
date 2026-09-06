# ADR 0005: Prioritize the eager Q-projection generation demo

- **Status:** Accepted
- **Date:** 2026-08-31
- **Decision owner:** Repository owner

## Context

G0 froze and independently verified the Q8 semantics. G1 then proved that one
real TinyLlama `[N,K]=[2048,2048]` query projection can flow through verified
IR, OI4 packing, scalar and ARM64 NEON code generation, audited machine code,
and the complete prepared-call boundary. The three checked-in Apple M4 sessions
measured `3.95671x`, `3.96176x`, and `3.95648x` NEON-over-scalar speedups. Those
numbers are kernel-boundary results, not model or text-generation speedups.

The original plan placed a schedule tuner and general `torch.compile`/FX
integration between that evidence and a visible model demonstration. Both are
valuable extensions, but neither is needed to prove that the compiler can own a
real hot path during generation. Keeping them on the critical path would delay
the smallest recruiter-visible result and expand the implementation surface
before the existing compiler is exercised across a model.

The versioned Rust C ABI in `decodeforge-bridge` is now the first completed G2
piece. It owns audited generated modules and exact packed weights behind opaque
handles, enforces bounded resources and lifecycle rules, contains panics, and
is tested through the actual release dynamic library. A guarded eager PyTorch
operator is the narrowest next consumer of that boundary.

## Decision

The required Mac-first path is now:

1. **G2 — native eager PyTorch boundary.** Keep the merged hardened runtime C
   ABI and add a low-level eager-only `decodeforge::q8_linear_v1` operator. The
   operator accepts only guarded CPU, FP32, contiguous, inference-only `M=1`
   inputs with static `N` and `K`, passes tensor storage directly to the bridge,
   and makes native, fallback, and error outcomes observable. General
   `torch.compile`, Dynamo/FX partitioning, fake/meta support, and persistent
   compiler caching are not G2 requirements.
2. **G3 — all-query-projection prompt-to-text proof.** Prepare one canonical Q8
   pack for each of TinyLlama's 22 bias-free `q_proj` weights and install an
   owning `nn.Module` adapter at every query-projection site. Prompt prefill
   (`M>1`) uses a reference implementation reconstructed from the *same* Q8
   weights and identity. Cached single-token decode (`M=1`) uses the native
   operator. A pinned greedy-generation run must prove coverage with counters,
   compare native execution with an all-same-Q8-fallback run, and report asset
   preparation, model load, prefill, per-token decode, dispatch, and total time
   separately.
3. **G4 — evidence-selected expansion.** Choose the next bottleneck only after
   G3 measurements. Candidates include bounded schedule selection, broader
   projection coverage, general FX/`torch.compile` integration, fusion, AVX2,
   and multicore execution.

Transformers/PyTorch continues to own tokenizer behavior, attention, KV-cache
management, sampling, and every unsupported operator. DecodeForge owns the
following original contribution:

```text
Q8Linear semantics
    -> compiler lowering
    -> OI4 weight packing
    -> scalar/NEON code generation
    -> guarded native artifact
    -> eager PyTorch execution inside 22 real model projections
```

The natural FP32 `nn.Linear` is a contextual quality baseline, not the G3
fallback comparator. Falling back from Q8 decode to FP32 prefill would silently
change the operator's semantics and make attribution impossible. No end-to-end
speedup may be inferred from the G1 `~3.96x` result; model-level performance is
claimed only from the G3 protocol and its own measurements.

## Consequences

- The next public artifact generates real text and visibly exercises compiled
  machine code without turning DecodeForge into a full inference engine.
- The required path stays achievable on the owner's Mac and reuses PyTorch,
  Transformers, Clang/LLVM, safetensors, and standard statistical tooling.
- The fixed G1 schedule remains legitimate compiler output. Empirical schedule
  search becomes an optional optimization rather than a prerequisite for model
  integration.
- G3 requires an identity-bound same-Q8 fallback, deterministic asset
  preparation, explicit resource cleanup, coverage counters, and a reproducible
  result bundle. A demo that merely emits text without those checks does not
  complete the gate.
- Supporting all 155 TinyLlama linear modules is explicitly unnecessary for the
  first end-to-end proof. The 22 same-shaped query projections provide repeated
  real-model coverage while keeping one compiled shape and one native contract.

## Supersession

This ADR supersedes the sequencing portions of
[ADR 0001](0001-mac-first-required-path.md) that require bounded schedule
selection before framework integration and specifically require
`torch.compile` for G3. It also supersedes ADR 0004's assignment of runtime
schedule selection to G2. ADR 0001's Apple-M4-first target decision and
optional AVX2 status remain in force. The technical decisions in ADRs 0002–0004
and the frozen Q8, packing, generated ABI, and NEON contracts are unchanged.

## Revisit conditions

Revisit after the checked-in G3 prompt-to-text bundle identifies the dominant
remaining cost, or if the eager boundary cannot preserve the same-Q8 fallback
and lifecycle contract without a materially different integration design.

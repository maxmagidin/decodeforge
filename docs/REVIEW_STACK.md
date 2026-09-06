# Review stack and validation

The implementation is split at dependency boundaries, preserving original
commits and experimental outcomes. These are stacked PRs, not seven independent
changes that can be merged in arbitrary order. Existing
[#32](https://github.com/maxmagidin/decodeforge/pull/32) is the prerequisite.

## Merge order and independently checked heads

Every row passed the full local `make check`, including the Rust and Python
checks present at that revision. Test counts grow as later work adds coverage.

| Order | PR / scope | Source head | Python tests |
| --- | --- | --- | --- |
| 1 | [#34: adapters, canonical assets, checkpoint](https://github.com/maxmagidin/decodeforge/pull/34) | `9805877` | 209 |
| 2 | [#35: transactional all-layer installation](https://github.com/maxmagidin/decodeforge/pull/35) | `60ec795` | 238 |
| 3 | [#36: offline G3 session and publication pipeline](https://github.com/maxmagidin/decodeforge/pull/36) | `cf5f970` | 374 |
| 4 | [#33: accepted model evidence and chat demo](https://github.com/maxmagidin/decodeforge/pull/33) | `084a491` | 422 |
| 5 | [#38: Apache 2.0 licensing](https://github.com/maxmagidin/decodeforge/pull/38) | `becc1e6` | 425 |
| 6 | [#39: explicit sentence formatting and retained experiments](https://github.com/maxmagidin/decodeforge/pull/39) | `660d1ab` | 429 |
| 7 | [#40: guarded macOS Rust loader repair](https://github.com/maxmagidin/decodeforge/pull/40) | `1a662b9` | 443 |

The 422-test aggregate predates its documentation/result-retention commit;
the presentation model run used clean source `662d137`. The other listed
split heads were checked from isolated or clean worktrees. #40 also carries
documentation-only handoff updates after the tested source head.

At the first handoff, #33, #34 and #35 had green hosted checks and were ready;
#36 and #38–#40 remained draft while hosted jobs ran. The actual hosted macOS
repair/preflight step passed for `1a662b9`; the full workflow had not yet
completed. This snapshot does not replace the live checks on each PR.

During the subsequent evaluation pass, all hosted checks on the open #32–#40
stack completed successfully. Hosted logs confirmed an actual missing-link
repair followed by successful normal and offline preflight. The separate
`evaluation/v1` branch contains the checked evaluation harness at `535540e`;
`evaluation/v1-results` retains its observations and adds default summary
verification. Integrate these follow-ups only after #40. See the
[evaluation result](../results/evaluation/apple-m4-v1/README.md) for scope and
limitations; no main merge or #33 merge has been performed.

## Why one proposed boundary was consolidated

The first proposed #36 head, `c24b1fa`, failed the isolated workspace-portability
lint because its documentation retained machine-specific paths. Its immediate
publication follow-up contained the necessary corrections, so #36 was advanced
normally to `cf5f970` and passed all 374 Python tests plus the full aggregate.
GitHub automatically marked provisional
[#37](https://github.com/maxmagidin/decodeforge/pull/37) integrated when its
intermediate base reached its exact head. That did not merge into main or #33.

No history was rewritten, and no author or test narrative was fabricated.
Preserve commit ancestry during integration: retained evidence refers to exact
producer revisions. Review dependency-first; after each merge, verify the next
PR's base, diff and checks before continuing. No main merge or #33 merge was
performed during this pass.

## Evidence and remaining limits

- Frozen G3 specification, runner and accepted bundle are unchanged.
- The successful one-sentence run is presentation evidence, not a new benchmark.
  The failed stricter-prompt run is retained alongside it.
- The local toolchain repair check was a healthy no-op. Hosted preflight success
  and fixture mutation tests are reported separately.
- Broader instruction following, additional model layers, AVX2, multicore and
  uninstrumented whole-model performance are not established by this work.
- The original dirty checkout is untouched. Active work is in the
  `decodeforge-improvements` worktree on `review/08-macos-readiness`.

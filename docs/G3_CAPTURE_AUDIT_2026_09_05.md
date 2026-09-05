# G3 capture audit — 2026-09-05

## Rejected capture at 47b444e

The first session for source revision `47b444e` passed preparation and the real
adapter preflight, loaded the pinned TinyLlama model, and reached final evidence
validation. It was rejected at:

```text
reconciliation.installation.installed_modules
DFE-SCHEMA-007 / const
```

The external capture directory is `capture-47b444e` beneath the configured G3
evidence root. Its preparation receipt, assets, bridge and `session-0.log` are
retained. No accepted session JSON or result bundle was published. Indices 1
and 2 were not run. No performance result is claimed from this attempt.

## Cause and correction

`QProjModelInstallation.counters.installed_modules` is a live count: it is 22
while adapters are installed and zero after successful restoration. The G3
evidence field records how many modules were installed during the session, but
the runner populated it from the live count after cleanup. Its test double
incorrectly returned 22 even after closing, masking the mismatch.

The runner now retains the validated installation-time counter snapshot for
that evidence field and independently requires zero installed modules after
cleanup. Restored count, live adapters and in-flight work still come from the
final snapshot. The test double now follows the real lifecycle, and negative
tests reject inconsistent installation and cleanup counts.

The frozen schema, experiment, numerical tolerances, timing windows, generated
kernel and runtime ABI are unchanged. The corrected source needs a new clean
revision, new preparation receipt and an entirely new three-session capture
set; no session from the rejected implementation is reused or silently retried.

## Validation before recapture

- The corrected test double first reproduced the exact final-schema rejection.
- The focused session/model suites then passed all 65 tests, including negative
  checks for inconsistent initial and post-cleanup counts.
- `CARGO_NET_OFFLINE=true UV_OFFLINE=true make check` passed after the fix,
  including 405 Python tests, Rust debug/release suites, real Torch/dylib
  execution, fixture parity and G1 report regeneration.
- `git diff --check` passed. The local aggregate log is
  `.lavish/g3-lifecycle-check.log`.

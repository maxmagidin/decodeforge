# G3 frozen generation experiment

[`spec.json`](spec.json) is the closed version-1 contract for the G3 TinyLlama
generation comparison. Validate it offline with the repository-wide schema
check before implementing or running the checkpoint.

The canonical spec is ready: all model, configuration, and tokenizer file
identities were independently verified against the immutable model revision.
The fixed prompt IDs were produced with the exact locked Transformers and
tokenizers versions under strict offline, local-files-only loading. Execution
must reject any artifact, dependency, host, or tokenization mismatch.

Capture the required offline-preparation timing separately from generation:

```console
G3_WORK="/opt/homebrew/var/decodeforge-g3-evidence"
mkdir -p "$G3_WORK"
CARGO_TARGET_DIR="$G3_WORK/cargo-target" make prepare-g3-assets-timed \
  WEIGHTS="$G3_WORK/model/model.safetensors" \
  OUTPUT="$G3_WORK/assets" \
  RECEIPT="$G3_WORK/preparation-receipt.json"
```

Both the asset and receipt outputs must be new and outside the clean source
checkout; the receipt must also be outside the asset directory. The clock
surrounds only the descriptor-stable copy of `decodeforge-prepare-qproj
--source ... --output ...`; that command returns after its atomic no-replace
directory publication and parent sync. The wrapper then verifies the complete
prepared inventory and publishes the identity-bound receipt atomically without
replacement. Retain the exact preparation executable named by the receipt:
receipt verification rehashes it and generation additionally binds its
checkout, source, and asset paths.

For a bundle intended to be checked in, place the preparation tool, source
model, and outputs below the stable, real (not symlinked)
`/opt/homebrew/var/decodeforge-g3-evidence` root shown above. Absolute command
arguments are authenticated receipt evidence and are retained exactly; the
analyzer never redacts or rewrites them. User-home and platform-private
temporary paths would therefore fail the repository portability check.

`session-template.json` is the closed runner/result envelope. It is
intentionally marked `not_run` and contains no measured values or acceptance
claim. A runner may populate its accepted or rejected evidence branch only
under the schema's ordered-run, identity, counter, correctness, timing,
lifecycle, and drift invariants.

Completed session JSON files and the separately captured preparation receipt
are transient analyzer inputs, not members of the published ten-file result
bundle. The analyzer live-rehashes the receipt's preparation tool and must
losslessly project every session field into that closed inventory, including
ordered per-dispatch timing samples and counter snapshots. It must record each
input's canonical semantic
identity in `manifest.json`, computed as SHA-256 over UTF-8 JSON with sorted
keys, compact separators, and non-finite values forbidden. The ten retained
members must permit exact reconstruction and canonical reserialization of each
session so the verifier can reproduce that identity. An original presentation
byte hash is provenance-only unless its exact byte preimage is separately
retained.

The downstream analyzer preserves the complete session-to-member mapping by
retaining the three canonical session objects and the complete, identity-bound
portable preparation receipt in `analysis.json`, and emitting
deterministic role-specific projections in the other members. The verifier
reconstructs those sessions, revalidates the generation-session schema and its
semantic rules, and regenerates all ten members byte for byte. This recomputes
asset identities, correctness, coverage, timings, drift, acceptance, session
identities, member hashes, and the bundle identity without trusting retained
summaries.

Create and verify a bundle without overwriting an existing output:

```sh
make analyze-g3 \
  SESSION_1="$G3_WORK/session-0.json" \
  SESSION_2="$G3_WORK/session-1.json" \
  SESSION_3="$G3_WORK/session-2.json" \
  RECEIPT="$G3_WORK/preparation-receipt.json" \
  OUTPUT_DIR="$G3_WORK/result"
make verify-g3-result BUNDLE="$G3_WORK/result"
```

Both commands accept only the closed G3 contract. They do not create or imply a
checked-in measured result.

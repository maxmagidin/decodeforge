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
make prepare-g3-assets-timed \
  WEIGHTS=/absolute/path/to/model.safetensors \
  OUTPUT=/private/tmp/decodeforge-g3-assets \
  RECEIPT=/private/tmp/decodeforge-g3-prepare-receipt.json
```

Both output paths must be new, and the receipt must be outside both the asset
directory and the clean source checkout. The clock surrounds only the
descriptor-stable copy of `decodeforge-prepare-qproj --source ... --output
...`; that command returns after its atomic no-replace directory publication
and parent sync. The wrapper then verifies the complete prepared inventory and
publishes the identity-bound receipt atomically without replacement. Retain
the exact preparation executable named by the receipt: receipt verification
rehashes it and generation additionally binds its checkout, source, and asset
paths.

`session-template.json` is the closed runner/result envelope. It is
intentionally marked `not_run` and contains no measured values or acceptance
claim. A runner may populate its accepted or rejected evidence branch only
under the schema's ordered-run, identity, counter, correctness, timing,
lifecycle, and drift invariants.

Completed session JSON files are transient analyzer inputs, not members of the
published ten-file result bundle. The analyzer must losslessly project every
session field into that closed inventory, including ordered per-dispatch timing
samples and counter snapshots. It must record each input's canonical semantic
identity in `manifest.json`, computed as SHA-256 over UTF-8 JSON with sorted
keys, compact separators, and non-finite values forbidden. The ten retained
members must permit exact reconstruction and canonical reserialization of each
session so the verifier can reproduce that identity. An original presentation
byte hash is provenance-only unless its exact byte preimage is separately
retained.

The downstream analyzer must preserve the complete session-to-member mapping;
the downstream verifier must reconstruct sessions, revalidate this schema and
its semantic rules, and independently recompute asset identities, correctness,
coverage, timings, drift, rejection, and acceptance. Neither executable is
implemented by this contract slice.

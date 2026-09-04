# G3 frozen generation experiment

[`spec.json`](spec.json) is the closed version-1 contract for the G3 TinyLlama
generation comparison. Validate it offline with the repository-wide schema
check before implementing or running the checkpoint.

The canonical spec is ready: all model, configuration, and tokenizer file
identities were independently verified against the immutable model revision.
The fixed prompt IDs were produced with the exact locked Transformers and
tokenizers versions under strict offline, local-files-only loading. Execution
must reject any artifact, dependency, host, or tokenization mismatch.

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

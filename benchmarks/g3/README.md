# G3 frozen generation experiment

[`spec.json`](spec.json) is the closed version-1 contract for the G3 TinyLlama
generation comparison. Validate it offline with the repository-wide schema
check before implementing or running the checkpoint.

The canonical spec is ready: all model, configuration, and tokenizer file
identities were independently verified against the immutable model revision.
The fixed prompt IDs were produced with the exact locked Transformers and
tokenizers versions under strict offline, local-files-only loading. Execution
must reject any artifact, dependency, host, or tokenization mismatch.

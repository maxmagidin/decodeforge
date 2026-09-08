# Security policy

DecodeForge generates, audits, loads, and executes native code. Treat model
artifacts, compiled modules, result bundles, and compiler inputs as untrusted
until their version, size, architecture, and content identities have been
validated.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/maxmagidin/decodeforge/security/advisories/new).
If that channel is unavailable, contact the maintainer through the public
contact information on the [GitHub profile](https://github.com/maxmagidin).

Include the affected revision, platform and architecture, a minimal sanitized
reproduction, expected impact, and whether the issue crosses a native ABI or
artifact-identity boundary. Do not attach model weights, secrets, private data,
hostnames, usernames, serial numbers, or absolute developer paths.

## Supported versions

Security fixes target the current `main` branch. DecodeForge is a research and
portfolio project rather than a production inference service; no long-term
support window is promised for older revisions or evidence bundles.

## Security boundaries

- Content hashes and manifests provide identity and integrity checks; they are
  not a sandbox for malicious native code.
- Generated modules must still pass architecture, ABI, symbol, relocation,
  helper, size, and feature checks before loading.
- Runtime guards validate shapes, buffer extents, alignment, identities, limits,
  and lifecycle state before execution.
- Model checkpoints and tokenizer assets are external dependencies and retain
  their own provenance, trust, and license requirements.
- Benchmark evidence may prove what ran under a recorded protocol; it does not
  certify the repository for adversarial production workloads.

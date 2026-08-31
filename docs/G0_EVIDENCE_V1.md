# G0 evidence contract V1

G0 remains **open**. This document specifies evidence machinery and test
fixtures only; DecodeForge has not checked in a real G0 results bundle or made
a native-kernel, performance, or completion claim.

## Correctness-bundle identity

A G0 run uses `milestone: "g0"` and `bundle_class: "correctness"`. Its
`bundle_id` is:

```text
sha256(ASCII("DecodeForge/run-bundle/v1\\0") || canonical-run-manifest-without-bundle_id)
```

The canonical run manifest is compact ASCII JSON with lexicographically sorted
object keys, no whitespace, and integers as its only numeric values. Its stored
root bytes have no terminal newline. The exact `created_utc` field is part of
that preimage. Therefore two separately captured runs can have different IDs;
reproducibility is established by their declared fixture and artifact bytes,
not by pretending that independently captured runs have one timestamp-free
identity.

`reproduction` is data only. It has cwd `.`, policy
`g0-correctness-v1`, the full source revision, a fixed three-key offline
environment, and an ordered `commands` array. Each command declares its ID,
argv array, and required zero exit code. A verifier must never execute it.

## Closed G0 inventory

The root `run-manifest.json` is not self-listed. A correctness bundle declares
exactly these sorted path/role pairs:

- `fixture-manifest.json` / `fixture-manifest`
- `host.json` / `host-manifest`
- `report.md` / `report`

No extra path, directory, symlink, or special file belongs in the bundle. The
JSON artifacts use the same ASCII/integer-only canonical representation used
for the ID. `host.json` and the copied fixture manifest each carry exactly one
terminal newline; the fixture spelling is therefore canonical JSON plus its
existing contract-required newline.

## Portable snapshot verification

The portable verifier opens the bundle root once as a directory descriptor,
rejecting a symlink at that final path component while allowing ordinary
symlinked ancestors. It opens each of the four flat names relative to that
descriptor without following the final component, accepts only bounded regular
files (64 KiB for the root and copied fixture, 32 KiB for the host, and 256
KiB for the report), and compares descriptor metadata before and after each
read. After the hashes and JSON checks, it re-stats each flat name through that
same descriptor and requires the original descriptor identity. It also lists
the closed inventory through the descriptor before and after the artifact
snapshots, and once more as the last verification operation. This prevents a
later pathname swap from redirecting an in-progress G0 verification. A missing,
special, malformed, or oversized root manifest is a bounded verification
failure, never a reason to invoke an unbounded reader.

The inventory scan retains at most the four permitted names and stops on a
fifth directory entry, reporting a stable entry-cap failure rather than
allocating an attacker-controlled list of names.

The copied fixture manifest and host manifest are parsed from those same
snapshots, canonicalized, and checked against their pinned single-file schemas.
Their relationship to each other and to the run manifest is a later portable
cross-file verification layer.

The distributable Python wheel force-includes the complete schema corpus,
examples, and diagnostic registry under `decodeforge/_schemas`; installed
verification reads that packaged copy. Source and editable checkouts fall back
to the checked-in `schemas/` tree so repository tools retain their
repository-relative diagnostics.

## Two verification boundaries

The portable verifier only reads bounded file snapshots, parses them, and
checks hashes and cross-file contracts. It deliberately has no Git dependency.
The separate repository gate is allowed to use checked-in source material and
Git history: it checks the copied fixture-manifest bytes and that the clean full
revision has the required ancestry. This separation makes a bundle inspectable
from a source archive while keeping provenance claims auditable in a checkout.

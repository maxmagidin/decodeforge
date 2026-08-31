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

For G0, that record is exactly:

```text
environment = {
  "CARGO_NET_OFFLINE": "true",
  "DECODEFORGE_SOURCE_REVISION": R,
  "UV_OFFLINE": "true"
}
commands = [
  {id: "schema-contracts-v1", argv: ["uv", "run", "--frozen", "python", "scripts/validate_schemas.py", "--all"], expected_exit_code: 0},
  {id: "q8-python-fixtures-v1", argv: ["uv", "run", "--frozen", "python", "scripts/generate_q8_fixtures.py", "--check"], expected_exit_code: 0},
  {id: "q8-rust-fixtures-v1", argv: ["make", "rust-fixture-check"], expected_exit_code: 0}
]
```

Here `R` is a lowercase, full 40-hex commit ID. `project.revision`,
`reproduction.source_revision`,
`reproduction.environment.DECODEFORGE_SOURCE_REVISION`, and
`host.source.revision` are all exactly `R`; both `project.dirty` and
`host.source.dirty` are `false`.

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
Their relationship to each other and to the run manifest is a portable
cross-file verification layer; it consumes the retained snapshots and never
reopens an artifact pathname.

The distributable Python wheel force-includes the complete schema corpus,
examples, and diagnostic registry under `decodeforge/_schemas`; installed
verification reads that packaged copy. Source and editable checkouts fall back
to the checked-in `schemas/` tree so repository tools retain their
repository-relative diagnostics.

## Closed G0 Apple-M4 profile

G0 is one declared Apple-M4 correctness profile, not a generic future-host
verifier. Its host record has `role: "mac-primary"`,
`architecture: "aarch64"`, `os.name: "Darwin"`, and the non-identifying
`host_id: "apple-m4-primary"`; its target is exactly
`aarch64-apple-darwin`. CPU identity and topology are exactly `Apple M4`, 10
physical cores, and 10 logical cores. Host CPU features are sorted and unique.
Target features are also sorted and unique, include `neon`, and are a subset of
the host feature list. A host may report additional detected features; equality
is not required.

This closed profile permits only the fields produced by its capture machinery.
The run root contains exactly `schema_version`, `milestone`, `bundle_class`,
`bundle_id`, `created_utc`, `project`, `target`, `reproduction`, `artifacts`,
`checks`, and `not_applicable`; it does not admit generic root metadata,
selection, operator, or redaction fields. `project` contains exactly revision,
cleanliness, format, numeric mode, compiler version, and the generated/runtime
ABI fields. `target` contains exactly triple and features, with no metadata.

The host root contains exactly schema version, host ID, role, architecture,
CPU, OS, toolchains, and source; `execution` and arbitrary host metadata are
not admitted. CPU fields are exactly model, physical cores, logical cores, and
features. OS fields are exactly name, version, and kernel. Source fields are
exactly revision and dirty. Toolchain keys are exactly the four keys below.
These JSON shape restrictions prevent schema-generic metadata, environment,
path, or identifying fields from becoming G0 provenance. They do not interpret
the free-form `report.md`; the capture template owns that privacy boundary.

The copied fixture manifest's `format` and `numeric_mode` equal the run
manifest's `project` fields. `project.compiler_version` is exactly `0.1.0` as
historical G0 profile data, not a comparison against the ambient verifier. The
host toolchain map is exactly:

```text
clang  = Apple clang version 17.0.0 (clang-1700.0.13.5)
python = 3.12.14
rust   = 1.98.0
uv     = 0.12.5
```

The full Apple Clang first-line identity is retained deliberately; unlike the
lockfile-pinned Python, Rust, and uv versions, its build suffix is useful G0
provenance. This closed map is not a constraint on a future milestone's host.

The G0 check map is exactly `schema: pass`, `correctness: pass`,
`assembly: not-applicable`, and `certified_performance: not-applicable`.
`not_applicable` is exactly the ordered list
`["assembly", "certified_performance"]`. The reproduction record above is
also exact. These comparisons do not grant the verifier authority to run any
recorded command.

## Two verification boundaries

The portable verifier only reads bounded file snapshots, parses them, and
checks hashes and cross-file contracts. It deliberately has no Git dependency.
The separate repository gate has an explicit interface:

```sh
uv run --frozen python scripts/verify_g0_repository.py \
  --bundle PATH --checkout PATH
```

It first performs the pure portable verification and then, only for a verified
G0 bundle, checks Git provenance against the supplied checkout. It accepts a
clean checked-in bundle such as `results/g0/...` as well as a bundle outside
the checkout; no working-directory or source-tree auto-discovery occurs. The
later capture workflow stages a not-yet-committed bundle outside the checkout
before atomically installing it, but that capture safety rule is not an
inside-bundle prohibition for this verifier.

The repository gate requires the baseline, `R`, and `HEAD` to be commit
objects and enforces the fixed baseline
`d2fe5c77a97f6dd55a48ef1bc58d51cc872dc69c <= R <= HEAD` in Git ancestry
order. A detached, clean `HEAD` is valid. It rejects staged, unstaged, and
untracked changes, while ignored-only files are allowed. It compares the
already verified in-memory bytes of `fixture-manifest.json` against the exact
blob `R:tests/fixtures/v1/manifest.json`; it does not reread the bundle after
the portable snapshot pass.

Before provenance queries, the gate resolves the supplied path once and
requires it to be the exact non-bare worktree root reported by Git; a nested
path that discovers a parent repository, a redirected local `core.worktree`,
and a bare repository are rejected. It records that worktree's Git directory,
then supplies explicit `--git-dir` and `--work-tree` values on every later
query. This prevents current-directory discovery or local worktree redirection
from changing which checkout is attested.

Every Git query uses fixed argv with no shell, no stdin, no network operation,
and a minimal controlled environment. Replacement objects, lazy promisor
fetches, system/global configuration, prompts, pagers, optional locks,
untracked caches, and local `core.fsmonitor` helpers are disabled. Status also
forces ordinary file-mode, symlink, stat, ctime, and ignore-stat behavior; a
bounded `ls-files -v -z` pass rejects assume-unchanged and skip-worktree index
entries that could hide an edit. Each query has a five-second deadline and a
64 KiB stdout cap, runs in its own process group, and has that group terminated
and reaped on timeout, output-limit, or runner failure. Diagnostics omit raw
stderr and filesystem paths; any stderr byte from a fixed Git query is an
operationally unavailable, fail-closed result. The final repository
observations are clean status, unflagged index state, then equality of
`HEAD^{commit}` with its captured value, so a late branch movement is rejected.
This separation makes a bundle inspectable from a source archive while keeping
provenance claims auditable in an explicit checkout.

# Apple M4 G1 result

This directory is the checked-in result for the fixed DecodeForge G1 vertical
slice. It measures the generated scalar and ARM64 NEON implementations of the
same weight-only Q8 `M=1, N=2048, K=2048` TinyLlama projection through the
complete prepared-call boundary.

On the pinned Apple M4, the three independent sessions measured paired
speedups of `3.95671x`, `3.96176x`, and `3.95648x`. Their 95% paired-BCa
confidence intervals were `[3.95103, 3.96705]`, `[3.95085, 3.96960]`, and
`[3.95351, 3.95997]`. Every lower bound exceeds `1.0`, so the predeclared G1
speedup gate passes.

The measured boundary includes output sentinel fill, the generated-module ABI
call, status decoding, and the complete finite-output scan. It excludes weight
packing, compilation, dynamic loading, and allocation. This is a generated
kernel result, not an end-to-end text-generation speedup.

## Contents

- `session-01.json` through `session-03.json` retain all 240 raw observations,
  generated source, disassembly audit, build flags, artifact hashes, host and
  checkout metadata, and pre/post timing correctness checks.
- `report.json` is the machine-readable deterministic analysis.
- `report.md` is the compact human-readable table.

The sessions were captured from clean revision
`046a73d6c1577808769d50cac7df6897a2c98ad7`. Reproduce and byte-compare the
report from the checked-in sessions with:

```sh
make verify-g1-result
```

# DFQ8_B32_V1 strict-f32 reference contract

This document is the normative readable contract for the independent Python
and Rust references and their cross-language fixtures. The format is frozen as
`DFQ8_B32_V1` and the executable numeric mode is `strict_f32_v1`.

## Shape and storage

An operator has `N > 0` output rows and `K > 0` logical input columns. The
number of blocks is `B = ceil(K / 32)`. Both references accept positive u32
`N` and `K` values subject to derived-storage checks. The fixed forward-error
comparator is defined only through `MAX_COMPARATOR_K = 8,134,399`, where its
declared reduction bound remains in domain. Storage is row-major:

```text
q       [N][B][32]  raw signed-int8 bytes (two's complement)
scales  [N][B]      little-endian IEEE-754 binary32 words
```

Every lane beyond logical `K` in a final block is exactly zero. `-128` is not
valid. A scale is finite, non-negative, and canonical `+0` or a positive
binary32 value.

## Quantization

Inputs are raw binary32 bit words. A NaN or infinity is rejected, while source
`-0` is normalized to `+0` for quantization math. For each logical row and
block, `amax` is the maximum absolute value of logical lanes only:

```text
scale = +0                         if amax == +0
        RN32(amax / 127.0f)        otherwise
q     = 0                          if scale == +0
        clamp(RNE(RN32(w / scale)), -127, 127) otherwise
```

`RN32` means one correctly rounded binary32 operation (round-to-nearest,
ties-to-even) with gradual underflow and no flush-to-zero. Both the Python and
Rust references implement these operations with integer/rational helpers, so
their results do not depend on an accidental binary64 double round, host FPU
mode, or contraction. Integer rounding and clamping happen in the shown order.
The source bit words remain inspectable in fixtures and are never mutated.

## Source-vs-Q8 quality evidence

Quantization quality is measured separately from generated-kernel correctness.
The source FP32 words are compared with `dequantize_f32_bits` output, not with
the generated-kernel comparator. For every nonzero-scale lane that did not hit
the `[-127, 127]` clamp, the deterministic tests use exact `Fraction`
arithmetic to check the conservative V1 per-weight bound

```text
scale * (0.5 + 255*u) + 2**-150,  where u = 2**-24
```

This combines the integer-RNE half-step with allowances for binary32 division
and dequantization-multiply rounding, plus half a minimum-subnormal step. Zero
blocks and physical padding are checked for exact zero and q-range invariants.
When a nonzero source block's scale itself underflows to `+0`, every q and
dequantized value is zero and the separate per-weight error bound is
`63 * 2**-149`; the committed `source=0x0000003f` case reaches that boundary.
The intentional subnormal-clamp case is reported independently: source
magnitude `190 * 2**-149` rounds its scale to the minimum subnormal, stores
`q=127`, and has absolute error `63 * 2**-149`; the interior-lane bound does
not apply to that clamped lane.

## Canonical scalar evaluation

For one finite input row `x[K]`, evaluation is strictly ordered. Start `out`
and each `block_sum` at `+0` binary32. Visit blocks ascending, then logical
lanes ascending; never visit padding:

```text
product    = RN32(x[k] * float32(q[row, block, lane]))
block_sum  = RN32(block_sum + product)
scaled     = RN32(block_sum * scale[row, block])
out        = RN32(out + scaled)
```

No FMA, BLAS, NumPy, vector reduction, reassociation, or padded-lane read is
part of this oracle. Any non-finite source, input, intermediate, or output is a
stable `DFE-QUANT` rejection and is never serialized as a valid fixture.

## Identities and JSON

Fixture JSON has required `case_id` and `blocks` fields (there is no `b`
wire field), as well as both identities below. Its `error_bound` is exactly
`{"policy":"strict_f32_v1","comparator":"dfq8_forward_v1"}`. Quant fixture
and fixture-manifest objects reject free-form metadata. The logical-weight
identity is SHA-256 over this exact byte preimage:

```text
ASCII("DecodeForge/DFQ8_B32_V1/logical-weight/v1\\0")
|| LE32(N) || LE32(K) || LE32(B)
|| raw_q_bytes || scale_u32_le
```

The fixture identity is SHA-256 over:

```text
ASCII("DecodeForge/DFQ8_B32_V1/strict_f32_v1/quant-fixture/v1\\0")
|| LE32(N) || LE32(K) || source_u32_le || scale_u32_le
|| raw_q_bytes || input_u32_le || output_u32_le
```

Both are written as `sha256:<lowercase-hex>`. Fixture JSON is encoded as UTF-8
using sorted keys, compact separators, `ensure_ascii=true`, `allow_nan=false`,
and exactly one terminal newline. The manifest root pins `format` to
`DFQ8_B32_V1` and `numeric_mode` to `strict_f32_v1`; it requires at least one
artifact and each artifact record contains only a safe relative `path`, `bytes`,
and SHA-256 value. Hashed artifacts contain no timestamps or free-form
metadata. The committed corpus is listed by the closed, sorted
`tests/fixtures/v1/manifest.json`. Run
`uv run --frozen python scripts/generate_q8_fixtures.py --check` to verify it
without writing. Rust independently generates the expected documents in memory,
and `q8 verify` only reads and verifies the supplied fixture tree. Reproduce
the read-only Rust gate with:

```sh
PATH="$(dirname "$(rustup which --toolchain 1.98.0 cargo)"):$PATH" cargo run --offline --locked -p decodeforge -- q8 verify
```

The Python generator is the sole explicit fixture writer:

```sh
uv run --frozen python scripts/generate_q8_fixtures.py --write
```

## Comparator

The strict comparator accepts finite candidate outputs and computes its
canonical reference internally from the input bits and weights. It sums
`A = sum(abs(float64(x) * float64(q) * float64(scale)))` in ascending logical
order, with `u = 2^-24`, `t = 2*K + 2*B + 16`, and
`gamma = t*u/(1-t*u)`. The default pass bound per output is:

```text
abs(candidate - canonical) <= max(1e-7, 4 * gamma * A)
```

`+0` and `-0` compare equal with ULP distance zero. The negative minimum
subnormal is one ULP from either zero and two ULPs from the positive minimum
subnormal. Relative error for a zero reference is `0` when the actual value is
also zero and `+infinity` otherwise. Ordinary candidates use fixed factor four,
while generated-scalar validation uses factor one and additionally requires ULP
distance at most two.

## Stable diagnostics

The Python API raises a structured diagnostic-bearing `Q8Error` for invalid
dimensions (`DFE-QUANT-001`), size overflow (`002`), length mismatch (`003`),
non-finite source/input (`004`/`005`), invalid scale (`006`), invalid q or
padding (`007`), identity mismatch (`008`), comparator domain (`009`),
non-finite evaluation (`010`), fixture mismatch (`011`), and unsafe fixture
artifacts (`012`). The same codes are registered in the V1 diagnostic catalog.

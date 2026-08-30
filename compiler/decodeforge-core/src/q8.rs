//! Independent semantics for `DFQ8_B32_V1`.
//!
//! The public API intentionally deals in binary32 bit words.  This keeps
//! signed zeroes and NaN payloads observable and makes the Rust oracle
//! independent of a tensor library or a host language's floating point
//! conversions.  Storage and evaluation follow the normative contract in
//! `docs/Q8_FORMAT_V1.md`.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

/// The frozen Q8 storage format.
pub const FORMAT: &str = "DFQ8_B32_V1";
/// The frozen scalar arithmetic mode.
pub const NUMERIC_MODE: &str = "strict_f32_v1";
/// Number of lanes in one physical Q8 block.
pub const BLOCK_SIZE: usize = 32;
/// Smallest representable signed Q8 value (negative 128 is forbidden).
pub const Q_MIN: i16 = -127;
/// Largest representable signed Q8 value.
pub const Q_MAX: i16 = 127;
/// Largest K for which the comparator's declared reduction bound is defined.
pub const MAX_COMPARATOR_K: u32 = 8_134_399;

const SIGN_MASK: u32 = 0x8000_0000;
const EXP_MASK: u32 = 0x7f80_0000;
const FRAC_MASK: u32 = 0x007f_ffff;
const POSITIVE_MASK: u32 = 0x7fff_ffff;
const F32_127_BITS: u32 = 0x42fe_0000;
const I64_MAX_U64: u64 = i64::MAX as u64;

pub mod fixture;

/// A stable semantic rejection with a machine-readable diagnostic code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8Error {
    /// Stable catalog code (for example `DFE-QUANT-003`).
    pub code: &'static str,
    /// Frozen human-readable summary.
    pub summary: &'static str,
    /// Small, deterministic context map useful to callers and diagnostics.
    pub context: BTreeMap<String, Value>,
}

impl Q8Error {
    fn new(code: &'static str, summary: &'static str) -> Self {
        Self {
            code,
            summary,
            context: BTreeMap::new(),
        }
    }

    fn with(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        let value = serde_json::to_value(value).expect("diagnostic context values serialize");
        self.context.insert(key.into(), value);
        self
    }

    /// Render the closed diagnostic object as canonical JSON.
    pub fn diagnostic_json(&self) -> String {
        #[derive(Serialize)]
        struct Diagnostic<'a> {
            schema_version: u32,
            code: &'a str,
            severity: &'static str,
            component: &'static str,
            summary: &'a str,
            context: &'a BTreeMap<String, Value>,
        }
        serde_json::to_string(&Diagnostic {
            schema_version: 1,
            code: self.code,
            severity: "error",
            component: component_for_code(self.code),
            summary: self.summary,
            context: &self.context,
        })
        .expect("diagnostic values serialize")
    }
}

impl fmt::Display for Q8Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.summary)
    }
}

impl std::error::Error for Q8Error {}

fn quant_error(code: &'static str, summary: &'static str) -> Q8Error {
    Q8Error::new(code, summary)
}

fn component_for_code(code: &str) -> &'static str {
    if code.starts_with("DFE-SCHEMA-") {
        "schema"
    } else {
        "quant"
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    blocks: u32,
    source_len: usize,
    q_len: usize,
    scale_len: usize,
}

fn checked_len(name: &'static str, left: u64, right: u64) -> Result<usize, Q8Error> {
    let value = left.checked_mul(right).ok_or_else(|| {
        quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", name)
    })?;
    // The Python reference deliberately performs this check before result
    // allocation, even on 64-bit hosts where usize can represent a little
    // more than the accepted sequence size.
    if value > I64_MAX_U64 || value > usize::MAX as u64 {
        return Err(
            quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", name),
        );
    }
    Ok(value as usize)
}

fn shape(n: u32, k: u32) -> Result<Shape, Q8Error> {
    if n == 0 || k == 0 {
        return Err(
            quant_error("DFE-QUANT-001", "N and K must both be positive.")
                .with("n", n)
                .with("k", k),
        );
    }
    let blocks_u64 = (k as u64).div_ceil(BLOCK_SIZE as u64);
    let blocks = u32::try_from(blocks_u64).map_err(|_| {
        quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", "blocks")
    })?;
    let source_len = checked_len("source", n as u64, k as u64)?;
    let q_lanes = checked_len("q", blocks as u64, BLOCK_SIZE as u64)?;
    let q_len = checked_len("q", n as u64, q_lanes as u64)?;
    let scale_len = checked_len("scale", n as u64, blocks as u64)?;
    Ok(Shape {
        blocks,
        source_len,
        q_len,
        scale_len,
    })
}

fn try_zeroed_u8(length: usize, field: &'static str) -> Result<Vec<u8>, Q8Error> {
    let mut result = Vec::new();
    result.try_reserve_exact(length).map_err(|_| {
        quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", field)
    })?;
    result.resize(length, 0);
    Ok(result)
}

fn try_zeroed_u32(length: usize, field: &'static str) -> Result<Vec<u32>, Q8Error> {
    let mut result = Vec::new();
    result.try_reserve_exact(length).map_err(|_| {
        quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", field)
    })?;
    result.resize(length, 0);
    Ok(result)
}

/// Whether a bit word encodes a finite binary32 number.
pub const fn is_finite_f32_bits(bits: u32) -> bool {
    (bits & EXP_MASK) != EXP_MASK
}

/// Decode a validated binary32 word to the host's f32 value.
pub fn f32_bits_to_float(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn round_integer_ratio(numerator: &BigUint, denominator: &BigUint) -> BigUint {
    debug_assert!(!denominator.is_zero());
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let twice = &remainder << 1usize;
    if twice > *denominator || (twice == *denominator && quotient.bit(0)) {
        quotient + BigUint::one()
    } else {
        quotient
    }
}

fn round_scaled(numerator: &BigUint, denominator: &BigUint, shift: i32) -> BigUint {
    let mut numerator = numerator.clone();
    let mut denominator = denominator.clone();
    if shift >= 0 {
        numerator <<= usize::try_from(shift).expect("non-negative f32 shift fits usize");
    } else {
        denominator <<= usize::try_from(-shift).expect("negative f32 shift fits usize");
    }
    round_integer_ratio(&numerator, &denominator)
}

fn floor_log2_ratio(numerator: &BigUint, denominator: &BigUint) -> i32 {
    debug_assert!(!numerator.is_zero() && !denominator.is_zero());
    let mut exponent = i32::try_from(numerator.bits()).expect("binary32 ratio fits i32")
        - i32::try_from(denominator.bits()).expect("binary32 ratio fits i32");
    if exponent >= 0 {
        let shifted = denominator << usize::try_from(exponent).expect("f32 exponent fits usize");
        if numerator < &shifted {
            exponent -= 1;
        }
    } else {
        let shifted = numerator << usize::try_from(-exponent).expect("f32 exponent fits usize");
        if shifted < *denominator {
            exponent -= 1;
        }
    }
    exponent
}

/// Round a signed rational binary value to one IEEE-754 binary32 word.
///
/// All intermediate arithmetic here is integer arithmetic.  This mirrors the
/// Python reference's rational implementation and therefore does not depend
/// on the host FPU, floating-point environment, contraction, or extended
/// precision.  The normal and gradual-underflow paths include the carry into
/// the next exponent and overflow to infinity.
fn round_rational_f32(
    numerator: BigUint,
    denominator: BigUint,
    binary_exponent: i32,
    sign: i8,
) -> u32 {
    debug_assert!(sign == -1 || sign == 1);
    debug_assert!(!denominator.is_zero());
    let sign_bit = if sign < 0 { SIGN_MASK } else { 0 };
    if numerator.is_zero() {
        return sign_bit;
    }

    let mut exponent = floor_log2_ratio(&numerator, &denominator) + binary_exponent;
    if exponent < -126 {
        let subnormal = round_scaled(&numerator, &denominator, binary_exponent + 149);
        if subnormal.is_zero() {
            return sign_bit;
        }
        let normal_threshold = BigUint::one() << 23usize;
        if subnormal >= normal_threshold {
            return sign_bit | (1 << 23);
        }
        return sign_bit | subnormal.to_u32().expect("binary32 subnormal fits u32");
    }

    let mut significand = round_scaled(&numerator, &denominator, binary_exponent - exponent + 23);
    let significand_threshold = BigUint::one() << 24usize;
    if significand >= significand_threshold {
        significand >>= 1usize;
        exponent += 1;
    }
    if exponent > 127 {
        return sign_bit | EXP_MASK;
    }
    if exponent < -126 {
        let subnormal = round_scaled(&numerator, &denominator, binary_exponent + 149);
        if subnormal.is_zero() {
            return sign_bit;
        }
        let normal_threshold = BigUint::one() << 23usize;
        if subnormal >= normal_threshold {
            return sign_bit | (1 << 23);
        }
        return sign_bit | subnormal.to_u32().expect("binary32 subnormal fits u32");
    }
    let hidden_bit = BigUint::one() << 23usize;
    let fraction = (significand - hidden_bit)
        .to_u32()
        .expect("binary32 significand fits u32");
    sign_bit | (((exponent + 127) as u32) << 23) | fraction
}

fn components(bits: u32) -> (i8, BigUint, i32, bool) {
    let sign = if bits & SIGN_MASK == 0 { 1 } else { -1 };
    let exponent_field = (bits & EXP_MASK) >> 23;
    let fraction = bits & FRAC_MASK;
    if exponent_field == 0 {
        return (sign, BigUint::from(fraction), -149, fraction == 0);
    }
    debug_assert_ne!(exponent_field, 0xff);
    (
        sign,
        BigUint::from((1u32 << 23) | fraction),
        exponent_field as i32 - 150,
        false,
    )
}

fn finite_components(bits: u32) -> (i8, BigUint, i32, bool) {
    debug_assert!(is_finite_f32_bits(bits));
    components(bits)
}

/// Multiply two finite binary32 values, rounding once to binary32.
///
/// The operation is implemented with integer/rational arithmetic, including
/// signed zero and gradual underflow, so no native floating-point operation
/// can influence oracle bits.
pub fn f32_mul_bits(left: u32, right: u32) -> Result<u32, Q8Error> {
    if !is_finite_f32_bits(left) || !is_finite_f32_bits(right) {
        return Err(quant_error(
            "DFE-QUANT-010",
            "A binary32 arithmetic operand is not finite.",
        ));
    }
    let (sign_left, numerator_left, power_left, zero_left) = finite_components(left);
    let (sign_right, numerator_right, power_right, zero_right) = finite_components(right);
    let sign = sign_left * sign_right;
    if zero_left || zero_right {
        return Ok(if sign < 0 { SIGN_MASK } else { 0 });
    }
    Ok(round_rational_f32(
        numerator_left * numerator_right,
        BigUint::one(),
        power_left + power_right,
        sign,
    ))
}

/// Add two finite binary32 values, rounding once to binary32.
pub fn f32_add_bits(left: u32, right: u32) -> Result<u32, Q8Error> {
    if !is_finite_f32_bits(left) || !is_finite_f32_bits(right) {
        return Err(quant_error(
            "DFE-QUANT-010",
            "A binary32 arithmetic operand is not finite.",
        ));
    }
    let (sign_left, numerator_left, power_left, zero_left) = finite_components(left);
    let (sign_right, numerator_right, power_right, zero_right) = finite_components(right);
    if zero_left && zero_right {
        return Ok(if sign_left < 0 && sign_right < 0 {
            SIGN_MASK
        } else {
            0
        });
    }
    if zero_left {
        return Ok(right);
    }
    if zero_right {
        return Ok(left);
    }

    let common_power = power_left.min(power_right);
    let mut signed_left = BigInt::from_biguint(
        Sign::Plus,
        numerator_left
            << usize::try_from(power_left - common_power).expect("f32 exponent fits usize"),
    );
    let mut signed_right = BigInt::from_biguint(
        Sign::Plus,
        numerator_right
            << usize::try_from(power_right - common_power).expect("f32 exponent fits usize"),
    );
    if sign_left < 0 {
        signed_left = -signed_left;
    }
    if sign_right < 0 {
        signed_right = -signed_right;
    }
    let total = signed_left + signed_right;
    if total.is_zero() {
        return Ok(0);
    }
    Ok(round_rational_f32(
        total.magnitude().clone(),
        BigUint::one(),
        common_power,
        if total.sign() == Sign::Minus { -1 } else { 1 },
    ))
}

/// Divide two finite binary32 values, rounding once to binary32.
pub fn f32_div_bits(numerator: u32, denominator: u32) -> Result<u32, Q8Error> {
    if !is_finite_f32_bits(numerator) || !is_finite_f32_bits(denominator) {
        return Err(quant_error(
            "DFE-QUANT-006",
            "A scale or division operand is not finite.",
        ));
    }
    if denominator & POSITIVE_MASK == 0 {
        return Err(quant_error(
            "DFE-QUANT-006",
            "A scale or division operand is zero.",
        ));
    }
    let (sign_numerator, numerator, power_numerator, zero_numerator) = finite_components(numerator);
    let (sign_denominator, denominator, power_denominator, zero_denominator) =
        finite_components(denominator);
    let sign = sign_numerator * sign_denominator;
    if zero_numerator {
        return Ok(if sign < 0 { SIGN_MASK } else { 0 });
    }
    debug_assert!(!zero_denominator);
    Ok(round_rational_f32(
        numerator,
        denominator,
        power_numerator - power_denominator,
        sign,
    ))
}

/// Convert a small integer exactly to binary32.
pub fn f32_from_int(value: i32) -> u32 {
    if value == 0 {
        return 0;
    }
    round_rational_f32(
        BigUint::from(value.unsigned_abs()),
        BigUint::one(),
        0,
        if value < 0 { -1 } else { 1 },
    )
}

/// Convert a positive integer ratio exactly to binary32 for corpus constants.
pub(crate) fn f32_from_ratio(numerator: u32, denominator: u32, negative: bool) -> u32 {
    assert!(denominator != 0);
    round_rational_f32(
        BigUint::from(numerator),
        BigUint::from(denominator),
        0,
        if negative { -1 } else { 1 },
    )
}

fn round_f32_to_i32(bits: u32) -> i32 {
    let (sign, numerator, power, is_zero) = finite_components(bits);
    if is_zero {
        return 0;
    }
    let magnitude = round_scaled(&numerator, &BigUint::one(), power);
    match magnitude.to_i32() {
        Some(value) => i32::from(sign) * value,
        None if sign < 0 => i32::MIN,
        None => i32::MAX,
    }
}

fn ratio_to_q(ratio_bits: u32) -> Result<i16, Q8Error> {
    if !is_finite_f32_bits(ratio_bits) {
        if ratio_bits & FRAC_MASK != 0 {
            return Err(
                quant_error("DFE-QUANT-010", "A quantization ratio is not finite.")
                    .with("bits", ratio_bits),
            );
        }
        return Ok(if ratio_bits & SIGN_MASK == 0 {
            Q_MAX
        } else {
            Q_MIN
        });
    }
    let rounded = round_f32_to_i32(ratio_bits);
    Ok(rounded.clamp(Q_MIN as i32, Q_MAX as i32) as i16)
}

/// Immutable logical shape and physical Q8/scales storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8Weights {
    /// Number of output rows.
    n: u32,
    /// Number of logical input columns.
    k: u32,
    /// Number of physical 32-lane blocks.
    blocks: u32,
    /// Row-major `[N][B][32]` two's-complement bytes.
    q_bytes: Vec<u8>,
    /// Row-major `[N][B]` binary32 scale words.
    scale_bits: Vec<u32>,
}

impl Q8Weights {
    /// Construct and validate immutable-in-practice Q8 storage.
    pub fn try_new(
        n: u32,
        k: u32,
        blocks: u32,
        q_bytes: Vec<u8>,
        scale_bits: Vec<u32>,
    ) -> Result<Self, Q8Error> {
        let shape = shape(n, k)?;
        if blocks != shape.blocks {
            return Err(quant_error(
                "DFE-QUANT-001",
                "The declared block count does not equal ceil(K/32).",
            )
            .with("expected", shape.blocks)
            .with("actual", blocks));
        }
        if q_bytes.len() != shape.q_len || scale_bits.len() != shape.scale_len {
            return Err(quant_error(
                "DFE-QUANT-003",
                "Stored q or scale length does not match the shape.",
            )
            .with("expected_q", shape.q_len)
            .with("actual_q", q_bytes.len())
            .with("expected_scales", shape.scale_len)
            .with("actual_scales", scale_bits.len()));
        }
        for (index, bits) in scale_bits.iter().copied().enumerate() {
            if !is_finite_f32_bits(bits) || bits & SIGN_MASK != 0 {
                return Err(quant_error(
                    "DFE-QUANT-006",
                    "A scale must be finite and non-negative.",
                )
                .with("index", index)
                .with("bits", bits));
            }
        }
        for row in 0..n as usize {
            for block in 0..shape.blocks as usize {
                let scale = scale_bits[row * shape.blocks as usize + block];
                let start = (row * shape.blocks as usize + block) * BLOCK_SIZE;
                for lane in 0..BLOCK_SIZE {
                    let raw = q_bytes[start + lane];
                    let q = raw as i16 - if raw >= 128 { 256 } else { 0 };
                    if !(Q_MIN..=Q_MAX).contains(&q) {
                        return Err(quant_error(
                            "DFE-QUANT-007",
                            "A q byte is outside the signed DFQ8 range.",
                        )
                        .with("index", start + lane)
                        .with("value", q));
                    }
                    if block * BLOCK_SIZE + lane >= k as usize && raw != 0 {
                        return Err(quant_error("DFE-QUANT-007", "A padded q lane is not zero.")
                            .with("row", row)
                            .with("block", block)
                            .with("lane", lane));
                    }
                    if scale == 0 && raw != 0 {
                        return Err(quant_error(
                            "DFE-QUANT-007",
                            "A zero-scale block contains a nonzero q lane.",
                        )
                        .with("row", row)
                        .with("block", block));
                    }
                }
            }
        }
        Ok(Self {
            n,
            k,
            blocks,
            q_bytes,
            scale_bits,
        })
    }

    /// Compatibility alias for the physical block count.
    pub const fn b(&self) -> u32 {
        self.blocks
    }

    /// Number of output rows.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Number of logical input columns.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// Number of physical 32-lane blocks.
    pub const fn blocks(&self) -> u32 {
        self.blocks
    }

    /// Raw physical q bytes in row-major order.
    pub fn q_bytes(&self) -> &[u8] {
        &self.q_bytes
    }

    /// Scale words in row-major order.
    pub fn scale_bits(&self) -> &[u32] {
        &self.scale_bits
    }

    /// Signed q value at one physical lane, or `None` for an invalid index.
    pub fn q_at(&self, row: u32, block: u32, lane: usize) -> Option<i8> {
        if row >= self.n || block >= self.blocks || lane >= BLOCK_SIZE {
            return None;
        }
        let raw = self.q_bytes
            [(row as usize * self.blocks as usize + block as usize) * BLOCK_SIZE + lane];
        Some((raw as i16 - if raw >= 128 { 256 } else { 0 }) as i8)
    }

    /// Scale word at one block, or `None` for an invalid index.
    pub fn scale_at(&self, row: u32, block: u32) -> Option<u32> {
        if row >= self.n || block >= self.blocks {
            return None;
        }
        Some(self.scale_bits[row as usize * self.blocks as usize + block as usize])
    }

    /// Signed q bytes in row-major physical order.
    pub fn q_values(&self) -> Vec<i16> {
        self.q_bytes
            .iter()
            .map(|raw| (*raw as i16 - if *raw >= 128 { 256 } else { 0 }) as i8)
            .map(i16::from)
            .collect()
    }

    /// Alias exposing raw q storage without granting mutable access.
    pub fn raw_q_bytes(&self) -> &[u8] {
        &self.q_bytes
    }

    /// Alias exposing scale storage without granting mutable access.
    pub fn scales(&self) -> &[u32] {
        &self.scale_bits
    }

    /// Scale words serialized little-endian.
    pub fn scale_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.scale_bits.len() * 4);
        for bits in &self.scale_bits {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    }
}

/// Quantize row-major source binary32 words into DFQ8 storage.
pub fn quantize_f32_bits(n: u32, k: u32, source_bits: &[u32]) -> Result<Q8Weights, Q8Error> {
    let shape = shape(n, k)?;
    if source_bits.len() != shape.source_len {
        return Err(
            quant_error("DFE-QUANT-003", "A bit-word sequence has the wrong length.")
                .with("field", "source_fp32_bits")
                .with("expected", shape.source_len)
                .with("actual", source_bits.len()),
        );
    }
    for (index, bits) in source_bits.iter().copied().enumerate() {
        if !is_finite_f32_bits(bits) {
            return Err(
                quant_error("DFE-QUANT-004", "A binary32 value is not finite.")
                    .with("field", "source_fp32_bits")
                    .with("index", index)
                    .with("bits", bits),
            );
        }
    }
    let mut q = try_zeroed_u8(shape.q_len, "q")?;
    let mut scales = try_zeroed_u32(shape.scale_len, "scale")?;
    for row in 0..n as usize {
        let source_row = &source_bits[row * k as usize..(row + 1) * k as usize];
        for block in 0..shape.blocks as usize {
            let first = block * BLOCK_SIZE;
            let last = (first + BLOCK_SIZE).min(k as usize);
            let mut amax = 0u32;
            for bits in &source_row[first..last] {
                amax = amax.max(*bits & POSITIVE_MASK);
            }
            let scale = if amax == 0 {
                0
            } else {
                let scale = f32_div_bits(amax, F32_127_BITS)?;
                if !is_finite_f32_bits(scale) || scale & SIGN_MASK != 0 {
                    return Err(quant_error(
                        "DFE-QUANT-006",
                        "Quantization produced an invalid scale.",
                    )
                    .with("row", row)
                    .with("block", block)
                    .with("bits", scale));
                }
                scale
            };
            scales[row * shape.blocks as usize + block] = scale;
            if scale == 0 {
                continue;
            }
            let q_start = (row * shape.blocks as usize + block) * BLOCK_SIZE;
            for lane in 0..last - first {
                let ratio = f32_div_bits(source_row[first + lane], scale)?;
                q[q_start + lane] = ratio_to_q(ratio)? as i8 as u8;
            }
        }
    }
    Q8Weights::try_new(n, k, shape.blocks, q, scales)
}

/// Dequantize logical row-major weights into binary32 words.
pub fn dequantize_f32_bits(weights: &Q8Weights) -> Result<Vec<u32>, Q8Error> {
    let length = (weights.n as usize)
        .checked_mul(weights.k as usize)
        .ok_or_else(|| quant_error("DFE-QUANT-002", "Derived storage size overflows."))?;
    let mut result = Vec::new();
    result.try_reserve_exact(length).map_err(|_| {
        quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", "dequantized")
    })?;
    for row in 0..weights.n {
        for logical_k in 0..weights.k {
            let (block, lane) = (
                logical_k / BLOCK_SIZE as u32,
                (logical_k % BLOCK_SIZE as u32) as usize,
            );
            let q = weights.q_at(row, block, lane).ok_or_else(|| {
                quant_error("DFE-QUANT-007", "A q index is outside the stored layout.")
            })?;
            let product = f32_mul_bits(
                f32_from_int(i32::from(q)),
                weights.scale_at(row, block).unwrap(),
            )?;
            result.push(product);
        }
    }
    Ok(result)
}

/// Evaluate one `[1,K]` input in strict ascending block/lane order.
pub fn canonical_linear_f32_bits(
    input_bits: &[u32],
    weights: &Q8Weights,
) -> Result<Vec<u32>, Q8Error> {
    if input_bits.len() != weights.k as usize {
        return Err(
            quant_error("DFE-QUANT-003", "A bit-word sequence has the wrong length.")
                .with("field", "input_fp32_bits")
                .with("expected", weights.k)
                .with("actual", input_bits.len()),
        );
    }
    for (index, bits) in input_bits.iter().copied().enumerate() {
        if !is_finite_f32_bits(bits) {
            return Err(
                quant_error("DFE-QUANT-005", "A binary32 value is not finite.")
                    .with("field", "input_fp32_bits")
                    .with("index", index)
                    .with("bits", bits),
            );
        }
    }
    let mut outputs = Vec::new();
    outputs.try_reserve_exact(weights.n as usize).map_err(|_| {
        quant_error("DFE-QUANT-002", "Derived storage size overflows.").with("field", "output")
    })?;
    for row in 0..weights.n {
        let mut output = 0u32;
        for block in 0..weights.blocks {
            let mut block_sum = 0u32;
            let first = block as usize * BLOCK_SIZE;
            let last = (first + BLOCK_SIZE).min(weights.k as usize);
            let scale = weights.scale_at(row, block).unwrap();
            for (logical_k, input_word) in input_bits.iter().enumerate().take(last).skip(first) {
                let lane = logical_k - first;
                let q_bits = f32_from_int(i32::from(weights.q_at(row, block, lane).unwrap()));
                let product = f32_mul_bits(*input_word, q_bits).map_err(|_| {
                    quant_error(
                        "DFE-QUANT-010",
                        "Canonical evaluation produced a non-finite product.",
                    )
                    .with("row", row)
                    .with("block", block)
                    .with("lane", lane)
                })?;
                if !is_finite_f32_bits(product) {
                    return Err(quant_error(
                        "DFE-QUANT-010",
                        "Canonical evaluation produced a non-finite product.",
                    )
                    .with("row", row)
                    .with("block", block)
                    .with("lane", lane));
                }
                block_sum = f32_add_bits(block_sum, product).map_err(|_| {
                    quant_error(
                        "DFE-QUANT-010",
                        "Canonical evaluation produced a non-finite block sum.",
                    )
                    .with("row", row)
                    .with("block", block)
                })?;
                if !is_finite_f32_bits(block_sum) {
                    return Err(quant_error(
                        "DFE-QUANT-010",
                        "Canonical evaluation produced a non-finite block sum.",
                    )
                    .with("row", row)
                    .with("block", block));
                }
            }
            let scaled = f32_mul_bits(block_sum, scale).map_err(|_| {
                quant_error(
                    "DFE-QUANT-010",
                    "Canonical evaluation produced a non-finite scaled block.",
                )
                .with("row", row)
                .with("block", block)
            })?;
            if !is_finite_f32_bits(scaled) {
                return Err(quant_error(
                    "DFE-QUANT-010",
                    "Canonical evaluation produced a non-finite scaled block.",
                )
                .with("row", row)
                .with("block", block));
            }
            output = f32_add_bits(output, scaled).map_err(|_| {
                quant_error(
                    "DFE-QUANT-010",
                    "Canonical evaluation produced a non-finite output.",
                )
                .with("row", row)
            })?;
            if !is_finite_f32_bits(output) {
                return Err(quant_error(
                    "DFE-QUANT-010",
                    "Canonical evaluation produced a non-finite output.",
                )
                .with("row", row));
            }
        }
        outputs.push(output);
    }
    Ok(outputs)
}

fn update_u32_le(hasher: &mut Sha256, words: &[u32]) {
    for word in words {
        hasher.update(word.to_le_bytes());
    }
}

/// SHA-256 over the canonical logical-weight preimage.
pub fn logical_weight_identity(weights: &Q8Weights) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"DecodeForge/DFQ8_B32_V1/logical-weight/v1\0");
    hasher.update(weights.n.to_le_bytes());
    hasher.update(weights.k.to_le_bytes());
    hasher.update(weights.blocks.to_le_bytes());
    hasher.update(&weights.q_bytes);
    update_u32_le(&mut hasher, &weights.scale_bits);
    format!("sha256:{}", hex_lower(&hasher.finalize()))
}

/// SHA-256 over a complete quantization/evaluation fixture preimage.
pub fn fixture_identity(
    n: u32,
    k: u32,
    source_bits: &[u32],
    weights: &Q8Weights,
    input_bits: &[u32],
    output_bits: &[u32],
) -> Result<String, Q8Error> {
    let expected = shape(n, k)?;
    if weights.n != n || weights.k != k || weights.blocks != expected.blocks {
        return Err(quant_error(
            "DFE-QUANT-008",
            "Fixture dimensions disagree with weights.",
        ));
    }
    if source_bits.len() != expected.source_len {
        return Err(
            quant_error("DFE-QUANT-003", "A bit-word sequence has the wrong length.")
                .with("field", "source_fp32_bits")
                .with("expected", expected.source_len)
                .with("actual", source_bits.len()),
        );
    }
    if input_bits.len() != k as usize {
        return Err(
            quant_error("DFE-QUANT-003", "A bit-word sequence has the wrong length.")
                .with("field", "input_fp32_bits")
                .with("expected", k)
                .with("actual", input_bits.len()),
        );
    }
    if output_bits.len() != n as usize {
        return Err(
            quant_error("DFE-QUANT-003", "A bit-word sequence has the wrong length.")
                .with("field", "expected_output_fp32_bits")
                .with("expected", n)
                .with("actual", output_bits.len()),
        );
    }
    for (field, values, code) in [
        ("source_fp32_bits", source_bits, "DFE-QUANT-004"),
        ("input_fp32_bits", input_bits, "DFE-QUANT-005"),
        ("expected_output_fp32_bits", output_bits, "DFE-QUANT-010"),
    ] {
        for (index, bits) in values.iter().copied().enumerate() {
            if !is_finite_f32_bits(bits) {
                return Err(quant_error(code, "A binary32 value is not finite.")
                    .with("field", field)
                    .with("index", index)
                    .with("bits", bits));
            }
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(b"DecodeForge/DFQ8_B32_V1/strict_f32_v1/quant-fixture/v1\0");
    hasher.update(n.to_le_bytes());
    hasher.update(k.to_le_bytes());
    update_u32_le(&mut hasher, source_bits);
    update_u32_le(&mut hasher, &weights.scale_bits);
    hasher.update(&weights.q_bytes);
    update_u32_le(&mut hasher, input_bits);
    update_u32_le(&mut hasher, output_bits);
    Ok(format!("sha256:{}", hex_lower(&hasher.finalize())))
}

fn ordered_ulp(bits: u32) -> i64 {
    let magnitude = (bits & POSITIVE_MASK) as i64;
    if bits & SIGN_MASK != 0 {
        0x8000_0000i64 - magnitude
    } else {
        0x8000_0000i64 + magnitude
    }
}

/// Return finite binary32 ULP distance, treating either zero encoding equally.
pub fn ulp_distance(left: u32, right: u32) -> Result<u64, Q8Error> {
    if !is_finite_f32_bits(left) || !is_finite_f32_bits(right) {
        return Err(quant_error(
            "DFE-QUANT-009",
            "ULP comparison requires finite values.",
        ));
    }
    if left & POSITIVE_MASK == 0 && right & POSITIVE_MASK == 0 {
        return Ok(0);
    }
    Ok(ordered_ulp(left).abs_diff(ordered_ulp(right)))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

#[cfg(test)]
fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_subnormal_boundaries_match_contract() {
        assert_eq!(f32_div_bits(0x0000_003f, F32_127_BITS).unwrap(), 0);
        assert_eq!(f32_div_bits(0x0000_0040, F32_127_BITS).unwrap(), 1);
        assert_eq!(f32_mul_bits(0x3f80_0000, 0xbf00_0000).unwrap(), 0xbf00_0000);
        assert_eq!(f32_add_bits(0x3f80_0000, 0x3400_0000).unwrap(), 0x3f80_0001);
    }

    #[test]
    fn arithmetic_matches_python_golden_vectors() {
        let additions = [
            (0x0000_0000, 0x8000_0000, 0x0000_0000),
            (0x8000_0000, 0x8000_0000, 0x8000_0000),
            (0x3f80_0000, 0xbf80_0000, 0x0000_0000),
            (0x007f_ffff, 0x0000_0001, 0x0080_0000),
            (0x0080_0000, 0x8080_0000, 0x0000_0000),
            (0x3f80_0000, 0x3400_0000, 0x3f80_0001),
            (0x7f7f_ffff, 0x7f7f_ffff, 0x7f80_0000),
            (0xff7f_ffff, 0xff7f_ffff, 0xff80_0000),
        ];
        for (left, right, expected) in additions {
            assert_eq!(f32_add_bits(left, right).unwrap(), expected);
        }

        let multiplications = [
            (0x8000_0000, 0x3f80_0000, 0x8000_0000),
            (0x0000_0001, 0x3f80_0000, 0x0000_0001),
            (0x0000_0001, 0x3f00_0000, 0x0000_0000),
            (0x3fc0_0000, 0x4020_0000, 0x4070_0000),
            (0x7f7f_ffff, 0x7f7f_ffff, 0x7f80_0000),
            (0xbf80_0000, 0x4040_0000, 0xc040_0000),
        ];
        for (left, right, expected) in multiplications {
            assert_eq!(f32_mul_bits(left, right).unwrap(), expected);
        }

        let divisions = [
            (0x8000_0000, 0x4040_0000, 0x8000_0000),
            (0x0000_0001, 0x3f00_0000, 0x0000_0002),
            (0x0080_0000, 0x4000_0000, 0x0040_0000),
            (0x3f80_0000, 0x4040_0000, 0x3eaa_aaab),
            (0x7f7f_ffff, 0x0080_0000, 0x7f80_0000),
            (0xbf80_0000, 0x3f00_0000, 0xc000_0000),
        ];
        for (numerator, denominator, expected) in divisions {
            assert_eq!(f32_div_bits(numerator, denominator).unwrap(), expected);
        }
    }

    #[test]
    fn q_ratio_rounding_handles_half_and_above_half() {
        // 0.5 ties to the even integer zero, while the next representable
        // value and 0.75 round away from zero.  Negative ties mirror the same
        // magnitude rule.
        assert_eq!(ratio_to_q(0x3f00_0000).unwrap(), 0);
        assert_eq!(ratio_to_q(0x3f00_0001).unwrap(), 1);
        assert_eq!(ratio_to_q(0x3f40_0000).unwrap(), 1);
        assert_eq!(ratio_to_q(0xbf00_0000).unwrap(), 0);
        assert_eq!(ratio_to_q(0xbf00_0001).unwrap(), -1);
        assert_eq!(ratio_to_q(0x3fc0_0000).unwrap(), 2);
        assert_eq!(ratio_to_q(0xbfc0_0000).unwrap(), -2);
        assert_eq!(ratio_to_q(0x3fc0_0000).unwrap(), 2); // 1.5 ties to even 2
        assert_eq!(ratio_to_q(0x4020_0000).unwrap(), 2); // 2.5 ties to even 2
    }

    fn next_down(bits: u32) -> u32 {
        if bits & SIGN_MASK != 0 {
            bits + 1
        } else {
            bits - 1
        }
    }

    fn next_up(bits: u32) -> u32 {
        if bits & SIGN_MASK != 0 {
            bits - 1
        } else {
            bits + 1
        }
    }

    #[test]
    fn every_interior_half_integer_and_adjacent_f32_values_round_correctly() {
        // There is one midpoint between each neighboring integer in the
        // interior of [-127, 127].  Test the exact tie and both neighboring
        // binary32 values against the Python oracle's RNE result.
        for lower in -127i32..=126 {
            let midpoint = if lower < 0 {
                f32_from_ratio((-2 * lower - 1) as u32, 2, true)
            } else {
                f32_from_ratio((2 * lower + 1) as u32, 2, false)
            };
            let expected_tie = if lower % 2 == 0 { lower } else { lower + 1 };
            assert_eq!(ratio_to_q(next_down(midpoint)).unwrap(), lower as i16);
            assert_eq!(ratio_to_q(midpoint).unwrap(), expected_tie as i16);
            assert_eq!(ratio_to_q(next_up(midpoint)).unwrap(), (lower + 1) as i16);
        }
    }

    #[test]
    fn q_layout_and_rounding_are_deterministic() {
        let weights = quantize_f32_bits(1, 33, &[0x3f80_0000; 33]).unwrap();
        assert_eq!(weights.blocks, 2);
        assert_eq!(weights.q_at(0, 1, 0), Some(127));
        assert!(weights.q_bytes[33..64].iter().all(|value| *value == 0));
        let ties = quantize_f32_bits(
            1,
            8,
            &[
                0x42fe_0000,
                0xc2fe_0000,
                0x4020_0000,
                0x4060_0000,
                0xc020_0000,
                0xc060_0000,
                0,
                0x8000_0000,
            ],
        )
        .unwrap();
        assert_eq!(ties.q_values()[2..6], [2, 4, -2, -4]);
    }

    #[test]
    fn sha256_identity_has_known_empty_digest() {
        let digest = sha256_digest(b"");
        assert_eq!(
            hex_lower(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn strict_accumulation_is_not_a_double_reduction() {
        let weights = Q8Weights::try_new(
            1,
            3,
            1,
            vec![1, 1, 1].into_iter().chain([0; 29]).collect(),
            vec![0x3f80_0000],
        )
        .unwrap();
        let result =
            canonical_linear_f32_bits(&[0x3f80_0000, 0x3400_0000, 0x3380_0000], &weights).unwrap();
        assert_eq!(result, [0x3f80_0002]);
    }

    #[test]
    fn contraction_sensitive_vector_uses_separate_product_then_add() {
        // Fusing x0 * -127 + x1 * -1 into one operation yields 0xc2ffed91;
        // the contract rounds each product before the ascending sum and
        // therefore requires 0xc2ffed90.
        let weights = Q8Weights::try_new(
            1,
            2,
            1,
            vec![129, 255].into_iter().chain([0; 30]).collect(),
            vec![0x3f80_0000],
        )
        .unwrap();
        let result = canonical_linear_f32_bits(&[0x3f7f_ed5d, 0x3f80_0392], &weights).unwrap();
        assert_eq!(result, [0xc2ff_ed90]);
    }
}

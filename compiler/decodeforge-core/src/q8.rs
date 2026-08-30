//! Exact binary32 primitives for `DFQ8_B32_V1`.
//!
//! The public API intentionally deals in binary32 bit words.  This keeps
//! signed zeroes and NaN payloads observable and makes the Rust oracle
//! independent of a tensor library or a host language's floating point
//! conversions.  This layer supplies host-independent integer/rational
//! arithmetic; Q8 storage and evaluation are introduced separately.

use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};
use serde::Serialize;
use serde_json::Value;
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

const SIGN_MASK: u32 = 0x8000_0000;
const EXP_MASK: u32 = 0x7f80_0000;
const FRAC_MASK: u32 = 0x007f_ffff;
const POSITIVE_MASK: u32 = 0x7fff_ffff;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_subnormal_boundaries_match_contract() {
        assert_eq!(f32_div_bits(0x0000_003f, 0x42fe_0000).unwrap(), 0);
        assert_eq!(f32_div_bits(0x0000_0040, 0x42fe_0000).unwrap(), 1);
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
}

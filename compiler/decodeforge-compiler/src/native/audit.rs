//! Structural audit of the generated scalar helper's AArch64 disassembly.
//!
//! The audit is deliberately narrow: it examines only the hidden helper which
//! the scalar emitter creates, and rejects instructions that would violate the
//! scalar, non-contracted recurrence. It is a guardrail, not a disassembler
//! or a general Mach-O validation framework.

use crate::{Result, invalid};

/// Findings from the actual hidden-helper disassembly audit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarDylibAuditReport {
    helper_symbol: String,
    scalar_scvtf_count: usize,
    scalar_fmul_count: usize,
    scalar_fadd_count: usize,
    return_count: usize,
    conditional_branch_count: usize,
    comparison_count: usize,
    logical_lane_loop_observed: bool,
}

impl ScalarDylibAuditReport {
    /// The C helper symbol whose Mach-O spelling was audited.
    pub fn helper_symbol(&self) -> &str {
        &self.helper_symbol
    }

    /// Number of signed-integer-to-scalar-binary32 conversions observed.
    pub const fn scalar_scvtf_count(&self) -> usize {
        self.scalar_scvtf_count
    }

    /// Number of scalar binary32 multiply instructions observed.
    pub const fn scalar_fmul_count(&self) -> usize {
        self.scalar_fmul_count
    }

    /// Number of scalar binary32 add instructions observed.
    pub const fn scalar_fadd_count(&self) -> usize {
        self.scalar_fadd_count
    }

    /// Number of ordinary return instructions observed.
    pub const fn return_count(&self) -> usize {
        self.return_count
    }

    /// Number of conditional branches observed in the helper.
    pub const fn conditional_branch_count(&self) -> usize {
        self.conditional_branch_count
    }

    /// Number of compare/decrement instructions observed in the helper.
    pub const fn comparison_count(&self) -> usize {
        self.comparison_count
    }

    /// Whether the emitted helper still contains structural loop evidence.
    ///
    /// The generated source has one logical `lane < lane_count` loop. The
    /// compiler may choose different labels and counters, so the bounded audit
    /// checks the stable machine-level evidence for it: a comparison/decrement
    /// and a conditional branch inside the named helper.
    pub const fn logical_lane_loop_observed(&self) -> bool {
        self.logical_lane_loop_observed
    }
}

/// Audit the disassembly returned by the fixed `llvm-objdump` invocation.
///
/// This is crate-visible rather than public so production callers cannot
/// provide an arbitrary symbol. The native builder gets the symbol directly
/// from a verified [`crate::ScalarCModule`]. The builder relaxes loop evidence
/// only for the unforgeable `K == 1` module case, where Clang legally folds the
/// one-iteration source loop into straight-line code.
pub(crate) fn audit_scalar_helper_disassembly(
    hidden_symbol: &str,
    disassembly: &str,
    require_logical_lane_loop: bool,
) -> Result<ScalarDylibAuditReport> {
    let mach_o_symbol = format!("_{hidden_symbol}");
    let mut in_helper = false;
    let mut found_helper = false;
    let mut scalar_scvtf_count = 0_usize;
    let mut scalar_fmul_count = 0_usize;
    let mut scalar_fadd_count = 0_usize;
    let mut return_count = 0_usize;
    let mut conditional_branch_count = 0_usize;
    let mut comparison_count = 0_usize;
    let mut first_instruction_address = None;
    let mut last_instruction_address = None;
    let mut instruction_addresses = Vec::new();
    let mut final_instruction_is_return = false;
    let mut scalar_scvtf_addresses = Vec::new();
    let mut scalar_fmul_addresses = Vec::new();
    let mut scalar_fadd_addresses = Vec::new();
    let mut direct_branches = Vec::new();

    for raw_line in disassembly.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if !in_helper {
            if is_target_label(line, &mach_o_symbol) {
                in_helper = true;
                found_helper = true;
            }
            continue;
        }

        if is_other_global_label(line, &mach_o_symbol) {
            break;
        }

        let Some((address, instruction)) = parse_instruction(line) else {
            if has_hex_address_prefix(line) {
                return Err(invalid(
                    "DFE-NATIVE-006",
                    "scalar helper disassembly contains an unparseable instruction record.",
                ));
            }
            continue;
        };
        if last_instruction_address.is_some_and(|previous| address <= previous) {
            return Err(invalid(
                "DFE-NATIVE-006",
                "scalar helper instruction addresses are not strictly increasing.",
            ));
        }
        first_instruction_address.get_or_insert(address);
        last_instruction_address = Some(address);
        instruction_addresses.push(address);
        final_instruction_is_return = instruction.base_mnemonic == "ret";

        if is_forbidden_mnemonic(instruction.base_mnemonic) {
            return Err(invalid(
                "DFE-NATIVE-006",
                format!(
                    "scalar helper disassembly contains forbidden instruction {}.",
                    instruction.base_mnemonic
                ),
            ));
        }
        if instruction.base_mnemonic.starts_with("bl")
            || instruction.base_mnemonic.starts_with("br")
            || matches!(instruction.base_mnemonic, "retaa" | "retab")
        {
            return Err(invalid(
                "DFE-NATIVE-006",
                "scalar helper disassembly contains a call or indirect branch.",
            ));
        }
        if has_vector_arrangement(instruction.operands) || has_vector_register(instruction.operands)
        {
            return Err(invalid(
                "DFE-NATIVE-006",
                "scalar helper disassembly contains a vector register or arrangement.",
            ));
        }

        if instruction.base_mnemonic == "scvtf" && first_operand_is_scalar_s(instruction.operands) {
            scalar_scvtf_count += 1;
            scalar_scvtf_addresses.push(address);
        }
        if instruction.base_mnemonic == "fmul" && first_operand_is_scalar_s(instruction.operands) {
            scalar_fmul_count += 1;
            scalar_fmul_addresses.push(address);
        }
        if instruction.base_mnemonic == "fadd" && first_operand_is_scalar_s(instruction.operands) {
            scalar_fadd_count += 1;
            scalar_fadd_addresses.push(address);
        }
        if is_conditional_branch(instruction.mnemonic) {
            conditional_branch_count += 1;
        }
        if matches!(instruction.base_mnemonic, "cmp" | "cmn" | "subs") {
            comparison_count += 1;
        }
        if instruction.base_mnemonic == "ret" {
            return_count += 1;
        }
        if instruction.base_mnemonic == "b" || is_conditional_branch(instruction.mnemonic) {
            direct_branches.push((
                address,
                parse_direct_branch_target(instruction.operands).ok_or_else(|| {
                    invalid(
                        "DFE-NATIVE-006",
                        "scalar helper has a direct branch with an unparseable target.",
                    )
                })?,
                is_conditional_branch(instruction.mnemonic),
            ));
        }
    }

    if !found_helper {
        return Err(invalid(
            "DFE-NATIVE-006",
            "llvm-objdump output does not contain the generated hidden helper.",
        ));
    }
    if scalar_scvtf_count == 0 {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper disassembly has no scalar scvtf instruction.",
        ));
    }
    if scalar_fmul_count == 0 {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper disassembly has no scalar fmul instruction.",
        ));
    }
    if scalar_fadd_count == 0 {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper disassembly has no scalar fadd instruction.",
        ));
    }
    if return_count != 1 {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper disassembly must contain exactly one ordinary return.",
        ));
    }
    if !final_instruction_is_return {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper ordinary return is not its final instruction.",
        ));
    }
    let (Some(first_address), Some(last_address)) =
        (first_instruction_address, last_instruction_address)
    else {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper disassembly contains no instruction records.",
        ));
    };
    if direct_branches.iter().any(|(_, target, _)| {
        *target < first_address || *target > last_address || !instruction_addresses.contains(target)
    }) {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper contains a direct branch outside its instruction range.",
        ));
    }
    let logical_lane_loop_observed = comparison_count > 0
        && direct_branches.iter().any(|(branch, target, conditional)| {
            *conditional
                && target < branch
                && scalar_scvtf_addresses
                    .iter()
                    .any(|address| target <= address && address < branch)
                && scalar_fmul_addresses
                    .iter()
                    .any(|address| target <= address && address < branch)
                && scalar_fadd_addresses
                    .iter()
                    .any(|address| target <= address && address < branch)
        });
    if require_logical_lane_loop && !logical_lane_loop_observed {
        return Err(invalid(
            "DFE-NATIVE-006",
            "scalar helper disassembly lacks logical lane-loop branch evidence.",
        ));
    }

    Ok(ScalarDylibAuditReport {
        helper_symbol: hidden_symbol.to_owned(),
        scalar_scvtf_count,
        scalar_fmul_count,
        scalar_fadd_count,
        return_count,
        conditional_branch_count,
        comparison_count,
        logical_lane_loop_observed,
    })
}

struct Instruction<'a> {
    mnemonic: &'a str,
    base_mnemonic: &'a str,
    operands: &'a str,
}

fn parse_instruction(line: &str) -> Option<(u64, Instruction<'_>)> {
    let (address, text) = line.split_once(':')?;
    let address = u64::from_str_radix(address.trim(), 16).ok()?;
    let text = text.trim();
    let mnemonic = text.split_whitespace().next()?;
    let bytes = mnemonic.as_bytes();
    if bytes.is_empty()
        || !bytes[0].is_ascii_alphabetic()
        || bytes
            .iter()
            .any(|byte| !byte.is_ascii_alphanumeric() && *byte != b'.')
        || (bytes.len() == 2 && bytes.iter().all(u8::is_ascii_hexdigit))
    {
        return None;
    }
    let mnemonic_start = text.find(mnemonic)?;
    let operands = text[mnemonic_start + mnemonic.len()..].trim();
    let base_mnemonic = mnemonic.split('.').next().unwrap_or(mnemonic);
    Some((
        address,
        Instruction {
            mnemonic,
            base_mnemonic,
            operands,
        },
    ))
}

fn has_hex_address_prefix(line: &str) -> bool {
    line.split_once(':').is_some_and(|(address, _)| {
        let address = address.trim();
        !address.is_empty() && address.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn parse_direct_branch_target(operands: &str) -> Option<u64> {
    let token = operands
        .split(|character: char| character == ',' || character.is_whitespace())
        .rfind(|token| !token.is_empty())?;
    let token = token.strip_prefix("0x").unwrap_or(token);
    u64::from_str_radix(token, 16).ok()
}

fn is_target_label(line: &str, expected: &str) -> bool {
    line == format!("{expected}:")
        || line.ends_with(&format!("<{expected}>:"))
        || line.ends_with(&format!(" {expected}:"))
}

fn is_other_global_label(line: &str, expected: &str) -> bool {
    if is_target_label(line, expected) {
        return false;
    }
    let Some(label) = line.strip_suffix(':') else {
        return false;
    };
    let label = label.trim();
    label.starts_with('_') && !label.contains(char::is_whitespace)
}

fn is_forbidden_mnemonic(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "fmadd"
            | "fmsub"
            | "fnmadd"
            | "fnmsub"
            | "fmla"
            | "fmls"
            | "fmad"
            | "fmsb"
            | "fmmla"
            | "bfmmla"
            | "faddv"
            | "faddp"
            | "addv"
            | "addp"
            | "saddlv"
            | "uaddlv"
            | "sadalp"
            | "uadalp"
    )
}

fn first_operand_is_scalar_s(operands: &str) -> bool {
    let first = operands.trim_start().trim_start_matches(['[', '{']);
    let bytes = first.as_bytes();
    bytes.len() >= 2
        && bytes[0] == b's'
        && bytes[1].is_ascii_digit()
        && (bytes.len() == 2 || !bytes[2].is_ascii_alphanumeric())
}

fn is_conditional_branch(mnemonic: &str) -> bool {
    mnemonic.starts_with("b.") || matches!(mnemonic, "cbz" | "cbnz" | "tbz" | "tbnz")
}

fn has_vector_arrangement(operands: &str) -> bool {
    let bytes = operands.as_bytes();
    let mut index = 0_usize;
    while index < bytes.len() {
        if bytes[index] != b'.' {
            index += 1;
            continue;
        }
        let digit_start = index + 1;
        let mut digit_end = digit_start;
        while digit_end < bytes.len() && bytes[digit_end].is_ascii_digit() {
            digit_end += 1;
        }
        if digit_end > digit_start
            && digit_end < bytes.len()
            && matches!(bytes[digit_end], b's' | b'd' | b'h' | b'b')
        {
            return true;
        }
        index += 1;
    }
    false
}

fn has_vector_register(operands: &str) -> bool {
    let bytes = operands.as_bytes();
    for index in 0..bytes.len() {
        if !matches!(bytes[index], b'q' | b'v')
            || index + 1 >= bytes.len()
            || !bytes[index + 1].is_ascii_digit()
        {
            continue;
        }
        let left_boundary = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
        if left_boundary {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELPER: &str =
        "df_kernel_scalar_v1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn scalar_snippet(extra: &str) -> String {
        let extra = if extra.is_empty() {
            String::new()
        } else {
            format!("0000000100000020:\t{extra}\n")
        };
        format!(
            "\
(__TEXT,__text) section\n\
_{HELPER}:\n\
0000000100000000:\tscvtf s2, w8\n\
0000000100000004:\tfmul s0, s0, s2\n\
0000000100000008:\tfadd s0, s0, s1\n\
000000010000000c:\tmadd w8, w8, w9, wzr\n\
0000000100000010:\tmovi d0, #0000000000000000\n\
0000000100000014:\tsubs w8, w8, #1\n\
0000000100000018:\tb.ne 0x100000000\n\
000000010000001c:\tret\n\
{extra}"
        )
    }

    #[test]
    fn accepts_scalar_aarch64_without_false_positives() {
        let (_, conversion) =
            parse_instruction("0000000100000000:\tscvtf s2, w8").expect("instruction parses");
        assert_eq!(conversion.base_mnemonic, "scvtf");
        assert!(first_operand_is_scalar_s(conversion.operands));
        let report = audit_scalar_helper_disassembly(HELPER, &scalar_snippet(""), true).unwrap();
        assert_eq!(report.scalar_scvtf_count(), 1);
        assert_eq!(report.scalar_fmul_count(), 1);
        assert_eq!(report.scalar_fadd_count(), 1);
        assert_eq!(report.return_count(), 1);
        assert!(report.logical_lane_loop_observed());
    }

    #[test]
    fn accepts_a_verified_singleton_reduction_without_a_machine_loop() {
        let straight_line = scalar_snippet("")
            .replace("0000000100000014:\tsubs w8, w8, #1\n", "")
            .replace("0000000100000018:\tb.ne 0x100000000\n", "");
        let report = audit_scalar_helper_disassembly(HELPER, &straight_line, false).unwrap();
        assert!(!report.logical_lane_loop_observed());
        assert_eq!(report.scalar_scvtf_count(), 1);
        assert_eq!(report.scalar_fmul_count(), 1);
        assert_eq!(report.scalar_fadd_count(), 1);
    }

    #[test]
    fn rejects_every_forbidden_instruction_class() {
        for forbidden in [
            "fmadd s0, s1, s2, s3",
            "fmsub s0, s1, s2, s3",
            "fnmadd s0, s1, s2, s3",
            "fnmsub s0, s1, s2, s3",
            "fmla v0.4s, v1.4s, v2.4s",
            "fmls v0.4s, v1.4s, v2.4s",
            "fmad s0, s1, s2, s3",
            "fmsb s0, s1, s2, s3",
            "fmmla v0.4s, v1.16b, v2.16b",
            "bfmmla v0.4s, v1.8h, v2.8h",
            "faddv s0, v1.4s",
            "faddp s0, v1.2s",
            "addv b0, v1.16b",
            "addp v0.4s, v1.4s, v2.4s",
            "saddlv s0, v1.8h",
            "uaddlv d0, v1.4s",
            "sadalp v0.4s, v1.8h",
            "uadalp v0.4s, v1.8h",
            "bl _another_function",
            "blr x8",
            "blraa x8, x9",
            "br x8",
            "fadd s0, s1, q2",
            "fadd v0.4s, v1.4s, v2.4s",
        ] {
            let error = audit_scalar_helper_disassembly(HELPER, &scalar_snippet(forbidden), true)
                .expect_err(forbidden);
            assert_eq!(error.code(), "DFE-NATIVE-006", "{forbidden}");
        }
    }

    #[test]
    fn rejects_raw_opcode_records_and_out_of_range_branches() {
        let raw_record = scalar_snippet("c0 03 5f d6 ret");
        let raw_error = audit_scalar_helper_disassembly(HELPER, &raw_record, true)
            .expect_err("raw opcode bytes must not be mistaken for a mnemonic");
        assert_eq!(raw_error.code(), "DFE-NATIVE-006");

        let branch_error =
            audit_scalar_helper_disassembly(HELPER, &scalar_snippet("b 0x0000000200000000"), true)
                .expect_err("a branch outside the helper must be rejected");
        assert_eq!(branch_error.code(), "DFE-NATIVE-006");

        let middle_of_instruction =
            scalar_snippet("").replace("b.ne 0x100000000", "b.ne 0x100000002");
        let boundary_error = audit_scalar_helper_disassembly(HELPER, &middle_of_instruction, true)
            .expect_err("a branch into the middle of an instruction must be rejected");
        assert_eq!(boundary_error.code(), "DFE-NATIVE-006");

        let after_return = scalar_snippet("nop");
        let return_error = audit_scalar_helper_disassembly(HELPER, &after_return, true)
            .expect_err("the only return must be the final helper instruction");
        assert_eq!(return_error.code(), "DFE-NATIVE-006");
    }
}

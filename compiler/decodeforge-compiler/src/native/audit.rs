//! Structural audit of the generated scalar helper's AArch64 disassembly.
//!
//! The audit is deliberately narrow: it examines only the hidden helper which
//! the scalar emitter creates, and rejects instructions that would violate the
//! scalar, non-contracted recurrence. It is a guardrail, not a disassembler
//! or a general Mach-O validation framework. The NEON checks prove code shape,
//! control flow, and SIMD value flow; packed-address provenance comes from the
//! exact verified source-to-snapshot builder rather than a second GPR verifier.

use crate::{Result, invalid};
use std::collections::{HashMap, HashSet, VecDeque};

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

/// Findings from the actual hidden-helper disassembly audit for one strict
/// output-vector NEON module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeonDylibAuditReport {
    helper_symbol: String,
    vector_path_observed: bool,
    scalar_tail_observed: bool,
    signed_widen_8_to_16_count: usize,
    signed_widen_16_to_32_count: usize,
    signed_q8_to_i32_count: usize,
    vector_scvtf_count: usize,
    vector_fmul_count: usize,
    vector_fadd_count: usize,
    vector_broadcast_count: usize,
    vector_store_count: usize,
    return_count: usize,
    conditional_branch_count: usize,
    logical_vector_lane_loop_observed: bool,
}

impl NeonDylibAuditReport {
    /// The C helper symbol whose Mach-O spelling was audited.
    pub fn helper_symbol(&self) -> &str {
        &self.helper_symbol
    }

    /// Whether the helper contains a complete four-output vector recurrence.
    pub const fn vector_path_observed(&self) -> bool {
        self.vector_path_observed
    }

    /// Whether scalar multiply/add evidence remains for a partial output tile.
    pub const fn scalar_tail_observed(&self) -> bool {
        self.scalar_tail_observed
    }

    /// Number of signed 8-bit to 16-bit vector widen instructions observed.
    pub const fn signed_widen_8_to_16_count(&self) -> usize {
        self.signed_widen_8_to_16_count
    }

    /// Number of signed 16-bit to 32-bit vector widen instructions observed.
    pub const fn signed_widen_16_to_32_count(&self) -> usize {
        self.signed_widen_16_to_32_count
    }

    /// Total number of proven signed q8-to-i32 vector extension chains.
    pub const fn signed_q8_to_i32_count(&self) -> usize {
        self.signed_q8_to_i32_count
    }

    /// Number of four-lane signed-integer-to-binary32 conversions observed.
    pub const fn vector_scvtf_count(&self) -> usize {
        self.vector_scvtf_count
    }

    /// Number of four-lane binary32 multiplies observed.
    pub const fn vector_fmul_count(&self) -> usize {
        self.vector_fmul_count
    }

    /// Number of four-lane binary32 adds observed.
    pub const fn vector_fadd_count(&self) -> usize {
        self.vector_fadd_count
    }

    /// Number of four-lane activation broadcasts observed.
    pub const fn vector_broadcast_count(&self) -> usize {
        self.vector_broadcast_count
    }

    /// Number of 128-bit vector stores observed.
    pub const fn vector_store_count(&self) -> usize {
        self.vector_store_count
    }

    /// Number of ordinary return instructions observed.
    pub const fn return_count(&self) -> usize {
        self.return_count
    }

    /// Number of conditional branches observed in the helper.
    pub const fn conditional_branch_count(&self) -> usize {
        self.conditional_branch_count
    }

    /// Whether one backwards conditional branch encloses the ordered explicit
    /// widen, convert, broadcast, multiply, and add recurrence.
    pub const fn logical_vector_lane_loop_observed(&self) -> bool {
        self.logical_vector_lane_loop_observed
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

/// Audit the strict output-vector helper selected by a verified NEON module.
///
/// The shape is part of the generated module identity. It determines whether
/// a full four-output vector path can exist, whether a scalar output tail must
/// exist, and whether Clang may legally fold the one-iteration lane loop.
pub(crate) fn audit_neon_helper_disassembly(
    hidden_symbol: &str,
    disassembly: &str,
    n: u32,
    k: u32,
) -> Result<NeonDylibAuditReport> {
    let mach_o_symbol = format!("_{hidden_symbol}");
    let mut in_helper = false;
    let mut found_helper = false;
    let mut first_instruction_address = None;
    let mut last_instruction_address = None;
    let mut instruction_addresses = Vec::new();
    let mut final_instruction_is_return = false;
    let mut return_count = 0_usize;
    let mut conditional_branch_count = 0_usize;
    let mut direct_branches = Vec::new();
    let mut control_flow = Vec::new();
    let mut widen_8_to_16_addresses = Vec::new();
    let mut widen_16_to_32_addresses = Vec::new();
    let mut literal_widen_8_to_16 = Vec::new();
    let mut literal_widen_16_to_32 = Vec::new();
    let mut vector_scvtf_addresses = Vec::new();
    let mut vector_fmul_addresses = Vec::new();
    let mut vector_fadd_addresses = Vec::new();
    let mut vector_scvtf_instructions = Vec::new();
    let mut vector_fmul_instructions = Vec::new();
    let mut vector_fadd_instructions = Vec::new();
    let mut vector_load_instructions = Vec::new();
    let mut vector_store_instructions = Vec::new();
    let mut activation_load_instructions = Vec::new();
    let mut broadcast_instructions = Vec::new();
    let mut vector_broadcast_addresses = Vec::new();
    let mut vector_store_addresses = Vec::new();
    let mut scalar_scvtf_count = 0_usize;
    let mut scalar_fmul_count = 0_usize;
    let mut scalar_fadd_count = 0_usize;
    let mut scalar_fadd_addresses = Vec::new();
    let mut scalar_store_addresses = Vec::new();
    let mut scalar_fadd_instructions = Vec::new();
    let mut scalar_store_instructions = Vec::new();
    let mut simd_register_mentions = Vec::new();

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
                return Err(neon_audit_error(
                    "NEON helper disassembly contains an unparseable instruction record.",
                ));
            }
            continue;
        };
        if last_instruction_address.is_some_and(|previous| address != previous + 4) {
            return Err(neon_audit_error(
                "NEON helper instruction addresses are not one complete fixed-width sequence.",
            ));
        }
        first_instruction_address.get_or_insert(address);
        last_instruction_address = Some(address);
        instruction_addresses.push(address);
        final_instruction_is_return = instruction.base_mnemonic == "ret";

        if is_forbidden_mnemonic(instruction.base_mnemonic) {
            return Err(neon_audit_error(format!(
                "NEON helper disassembly contains forbidden instruction {}.",
                instruction.base_mnemonic
            )));
        }
        if instruction.base_mnemonic.starts_with("bl")
            || instruction.base_mnemonic.starts_with("br")
            || matches!(instruction.base_mnemonic, "retaa" | "retab")
        {
            return Err(neon_audit_error(
                "NEON helper disassembly contains a call or indirect branch.",
            ));
        }
        for register in simd_registers(instruction.operands) {
            simd_register_mentions.push(VectorRegisterInstruction { address, register });
        }

        if is_signed_widen(
            instruction.mnemonic,
            instruction.base_mnemonic,
            instruction.operands,
            ".8h",
            ".8b",
        ) {
            widen_8_to_16_addresses.push(address);
            if let Some(registers) = unary_vector_registers(address, instruction.operands) {
                literal_widen_8_to_16.push(registers);
            }
        }
        if is_signed_widen(
            instruction.mnemonic,
            instruction.base_mnemonic,
            instruction.operands,
            ".4s",
            ".4h",
        ) {
            widen_16_to_32_addresses.push(address);
            if let Some(registers) = unary_vector_registers(address, instruction.operands) {
                literal_widen_16_to_32.push(registers);
            }
        }
        if instruction.base_mnemonic == "scvtf"
            && instruction_has_arrangement(instruction.mnemonic, instruction.operands, ".4s")
        {
            vector_scvtf_addresses.push(address);
            if let Some(registers) = unary_vector_registers(address, instruction.operands) {
                vector_scvtf_instructions.push(registers);
            }
        }
        if instruction.base_mnemonic == "fmul"
            && instruction_has_arrangement(instruction.mnemonic, instruction.operands, ".4s")
        {
            vector_fmul_addresses.push(address);
            if let Some(registers) = ternary_vector_registers(address, instruction.operands) {
                vector_fmul_instructions.push(registers);
            }
        }
        if instruction.base_mnemonic == "fadd"
            && instruction_has_arrangement(instruction.mnemonic, instruction.operands, ".4s")
        {
            vector_fadd_addresses.push(address);
            if let Some(registers) = ternary_vector_registers(address, instruction.operands) {
                vector_fadd_instructions.push(registers);
            }
        }
        if is_four_lane_broadcast(
            instruction.mnemonic,
            instruction.base_mnemonic,
            instruction.operands,
        ) {
            vector_broadcast_addresses.push(address);
        }
        if matches!(instruction.base_mnemonic, "ld1r" | "dup")
            && instruction_has_arrangement(instruction.mnemonic, instruction.operands, ".4s")
            && !uses_stack_pointer_address(instruction.operands)
            && let Some(destination) = first_simd_register(instruction.operands)
        {
            broadcast_instructions.push(VectorRegisterInstruction {
                address,
                register: destination,
            });
        }
        if matches!(instruction.base_mnemonic, "ldr" | "ldur")
            && first_operand_is_scalar_s(instruction.operands)
            && !uses_stack_pointer_address(instruction.operands)
            && let Some(destination) = first_simd_register(instruction.operands)
        {
            activation_load_instructions.push(VectorRegisterInstruction {
                address,
                register: destination,
            });
        }
        if matches!(instruction.base_mnemonic, "ldr" | "ldur")
            && first_operand_is_q_register(instruction.operands)
            && !uses_stack_pointer_address(instruction.operands)
            && let Some(destination) = first_simd_register(instruction.operands)
        {
            vector_load_instructions.push(VectorRegisterInstruction {
                address,
                register: destination,
            });
        }
        if is_128_bit_vector_store(instruction.base_mnemonic, instruction.operands) {
            vector_store_addresses.push(address);
            if let Some(source) = first_simd_register(instruction.operands) {
                vector_store_instructions.push(VectorRegisterInstruction {
                    address,
                    register: source,
                });
            }
        }
        if instruction.base_mnemonic == "scvtf" && first_operand_is_scalar_s(instruction.operands) {
            scalar_scvtf_count += 1;
        }
        if instruction.base_mnemonic == "fmul" && first_operand_is_scalar_s(instruction.operands) {
            scalar_fmul_count += 1;
        }
        if instruction.base_mnemonic == "fadd" && first_operand_is_scalar_s(instruction.operands) {
            scalar_fadd_count += 1;
            scalar_fadd_addresses.push(address);
            if let Some(registers) = ternary_simd_registers(address, instruction.operands) {
                scalar_fadd_instructions.push(registers);
            }
        }
        if matches!(instruction.base_mnemonic, "str" | "stur")
            && first_operand_is_scalar_s(instruction.operands)
            && !uses_stack_pointer_address(instruction.operands)
        {
            scalar_store_addresses.push(address);
            if let Some(source) = first_simd_register(instruction.operands) {
                scalar_store_instructions.push(VectorRegisterInstruction {
                    address,
                    register: source,
                });
            }
        }
        if is_conditional_branch(instruction.mnemonic) {
            conditional_branch_count += 1;
        }
        if instruction.base_mnemonic == "ret" {
            return_count += 1;
        }
        let conditional_branch = is_conditional_branch(instruction.mnemonic);
        let branch_target = if instruction.base_mnemonic == "b" || conditional_branch {
            Some(
                parse_direct_branch_target(instruction.operands).ok_or_else(|| {
                    neon_audit_error("NEON helper has a direct branch with an unparseable target.")
                })?,
            )
        } else {
            None
        };
        if let Some(target) = branch_target {
            direct_branches.push((address, target, conditional_branch));
        }
        control_flow.push((
            address,
            if instruction.base_mnemonic == "ret" {
                ControlFlow::Return
            } else if conditional_branch {
                ControlFlow::Conditional(branch_target.expect("conditional target parsed"))
            } else if instruction.base_mnemonic == "b" {
                ControlFlow::Jump(branch_target.expect("jump target parsed"))
            } else {
                ControlFlow::Fallthrough
            },
        ));
    }

    if !found_helper {
        return Err(neon_audit_error(
            "llvm-objdump output does not contain the generated hidden NEON helper.",
        ));
    }
    if return_count != 1 {
        return Err(neon_audit_error(
            "NEON helper disassembly must contain exactly one ordinary return.",
        ));
    }
    if !final_instruction_is_return {
        return Err(neon_audit_error(
            "NEON helper ordinary return is not its final instruction.",
        ));
    }
    let (Some(first_address), Some(last_address)) =
        (first_instruction_address, last_instruction_address)
    else {
        return Err(neon_audit_error(
            "NEON helper disassembly contains no instruction records.",
        ));
    };
    if direct_branches.iter().any(|(_, target, _)| {
        *target < first_address || *target > last_address || !instruction_addresses.contains(target)
    }) {
        return Err(neon_audit_error(
            "NEON helper contains a direct branch outside its instruction range.",
        ));
    }
    if !all_instructions_reachable(&control_flow) {
        return Err(neon_audit_error(
            "NEON helper contains unreachable instruction records.",
        ));
    }

    let literal_signed_extensions =
        chained_instruction_addresses(&literal_widen_8_to_16, &literal_widen_16_to_32);
    let mut signed_q8_to_i32_instructions = literal_signed_extensions;
    signed_q8_to_i32_instructions.sort_by_key(|instruction| instruction.address);
    signed_q8_to_i32_instructions.dedup_by(|left, right| {
        left.address == right.address && left.destination == right.destination
    });

    let vector_recurrences = vector_recurrences(
        &signed_q8_to_i32_instructions,
        &vector_scvtf_instructions,
        &vector_fmul_instructions,
        &vector_fadd_instructions,
        &activation_load_instructions,
        &broadcast_instructions,
    );
    let vector_schedules = vector_schedules(
        &vector_recurrences,
        &vector_load_instructions,
        &vector_fmul_instructions,
        &vector_fadd_instructions,
        &vector_store_instructions,
        &simd_register_mentions,
        &control_flow,
    );
    let vector_path_observed = !vector_schedules.is_empty();
    let scalar_tail_observed = scalar_scvtf_count > 0
        && scalar_fmul_count > 0
        && scalar_fadd_count > 0
        && scalar_store_addresses
            .iter()
            .any(|store| scalar_fadd_addresses.iter().any(|add| add < store))
        && scalar_store_instructions.iter().any(|store| {
            scalar_fadd_instructions
                .iter()
                .any(|add| add.address < store.address && add.destination == store.register)
        });
    let logical_vector_lane_loop_observed = vector_schedules.iter().any(|schedule| {
        direct_branches.iter().any(|(branch, target, conditional)| {
            *conditional
                && target < branch
                && target <= &schedule.recurrence.extension_address
                && schedule.recurrence.activation_address >= *target
                && schedule.recurrence.lane_add_address < *branch
                && *branch < schedule.scale_multiply_address
        })
    });

    if n >= 4 && !vector_path_observed {
        return Err(neon_audit_error(
            "NEON helper lacks the ordered four-output vector recurrence or output store.",
        ));
    }
    let any_vector_evidence = !widen_8_to_16_addresses.is_empty()
        || !widen_16_to_32_addresses.is_empty()
        || !vector_scvtf_addresses.is_empty()
        || !vector_fmul_addresses.is_empty()
        || !vector_fadd_addresses.is_empty()
        || !vector_broadcast_addresses.is_empty()
        || !vector_load_instructions.is_empty()
        || !vector_store_addresses.is_empty();
    if n < 4 && any_vector_evidence {
        return Err(neon_audit_error(
            "NEON helper retains vector work for a scalar-only shape.",
        ));
    }
    if n.is_multiple_of(4)
        && (scalar_scvtf_count > 0 || scalar_fmul_count > 0 || scalar_fadd_count > 0)
    {
        return Err(neon_audit_error(
            "NEON helper retains scalar floating-point work without an output tail.",
        ));
    }
    if !n.is_multiple_of(4) && !scalar_tail_observed {
        return Err(neon_audit_error(
            "NEON helper lacks the required scalar output-tail recurrence.",
        ));
    }
    if n >= 4 && k > 1 && !logical_vector_lane_loop_observed {
        return Err(neon_audit_error(
            "NEON helper lacks backwards lane-loop evidence around its vector recurrence.",
        ));
    }

    Ok(NeonDylibAuditReport {
        helper_symbol: hidden_symbol.to_owned(),
        vector_path_observed,
        scalar_tail_observed,
        signed_widen_8_to_16_count: widen_8_to_16_addresses.len(),
        signed_widen_16_to_32_count: widen_16_to_32_addresses.len(),
        signed_q8_to_i32_count: signed_q8_to_i32_instructions.len(),
        vector_scvtf_count: vector_scvtf_addresses.len(),
        vector_fmul_count: vector_fmul_addresses.len(),
        vector_fadd_count: vector_fadd_addresses.len(),
        vector_broadcast_count: vector_broadcast_addresses.len(),
        vector_store_count: vector_store_addresses.len(),
        return_count,
        conditional_branch_count,
        logical_vector_lane_loop_observed,
    })
}

fn neon_audit_error(summary: impl Into<String>) -> crate::CompilerError {
    invalid("DFE-NATIVE-006", summary)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ControlFlow {
    Fallthrough,
    Conditional(u64),
    Jump(u64),
    Return,
}

fn all_instructions_reachable(instructions: &[(u64, ControlFlow)]) -> bool {
    let Some(&(entry, _)) = instructions.first() else {
        return false;
    };
    let indices = instructions
        .iter()
        .enumerate()
        .map(|(index, (address, _))| (*address, index))
        .collect::<HashMap<_, _>>();
    let mut reachable = HashSet::new();
    let mut pending = VecDeque::from([entry]);
    while let Some(address) = pending.pop_front() {
        if !reachable.insert(address) {
            continue;
        }
        let Some(&index) = indices.get(&address) else {
            return false;
        };
        let next = instructions.get(index + 1).map(|(next, _)| *next);
        match instructions[index].1 {
            ControlFlow::Fallthrough => pending.extend(next),
            ControlFlow::Conditional(target) => {
                pending.push_back(target);
                pending.extend(next);
            }
            ControlFlow::Jump(target) => pending.push_back(target),
            ControlFlow::Return => {}
        }
    }
    reachable.len() == instructions.len()
}

fn instruction_dominates(
    instructions: &[(u64, ControlFlow)],
    candidate: u64,
    use_address: u64,
) -> bool {
    let Some(&(entry, _)) = instructions.first() else {
        return false;
    };
    if candidate == use_address || candidate == entry {
        return true;
    }
    if !instructions
        .iter()
        .any(|(address, _)| *address == candidate)
        || !instructions
            .iter()
            .any(|(address, _)| *address == use_address)
    {
        return false;
    }

    let indices = instructions
        .iter()
        .enumerate()
        .map(|(index, (address, _))| (*address, index))
        .collect::<HashMap<_, _>>();
    let mut reachable_without_candidate = HashSet::new();
    let mut pending = VecDeque::from([entry]);
    while let Some(address) = pending.pop_front() {
        if address == candidate || !reachable_without_candidate.insert(address) {
            continue;
        }
        if address == use_address {
            return false;
        }
        let Some(&index) = indices.get(&address) else {
            return false;
        };
        let next = instructions.get(index + 1).map(|(next, _)| *next);
        match instructions[index].1 {
            ControlFlow::Fallthrough => pending.extend(next),
            ControlFlow::Conditional(target) => {
                pending.push_back(target);
                pending.extend(next);
            }
            ControlFlow::Jump(target) => pending.push_back(target),
            ControlFlow::Return => {}
        }
    }
    true
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VectorRegisterInstruction {
    address: u64,
    register: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VectorRecurrenceEvidence {
    extension_address: u64,
    activation_address: u64,
    lane_add_address: u64,
    lane_add_destination: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VectorScheduleEvidence {
    recurrence: VectorRecurrenceEvidence,
    scale_multiply_address: u64,
}

fn vector_recurrences(
    signed_extensions: &[UnaryVectorInstruction],
    conversions: &[UnaryVectorInstruction],
    multiplies: &[TernaryVectorInstruction],
    adds: &[TernaryVectorInstruction],
    activation_loads: &[VectorRegisterInstruction],
    broadcasts: &[VectorRegisterInstruction],
) -> Vec<VectorRecurrenceEvidence> {
    let mut evidence = Vec::new();
    for extension in signed_extensions {
        for conversion in conversions {
            if extension.address >= conversion.address || extension.destination != conversion.source
            {
                continue;
            }
            for multiply in multiplies {
                if conversion.address >= multiply.address {
                    continue;
                }
                let activation = if conversion.destination == multiply.first_source {
                    activation_evidence(
                        &multiply.second_source,
                        multiply.second_source_is_lane_zero,
                        multiply.address,
                        activation_loads,
                        broadcasts,
                    )
                } else if conversion.destination == multiply.second_source {
                    activation_evidence(
                        &multiply.first_source,
                        multiply.first_source_is_lane_zero,
                        multiply.address,
                        activation_loads,
                        broadcasts,
                    )
                } else {
                    None
                };
                let Some(activation_address) = activation else {
                    continue;
                };
                for add in adds {
                    if multiply.address < add.address
                        && (multiply.destination == add.first_source
                            || multiply.destination == add.second_source)
                    {
                        evidence.push(VectorRecurrenceEvidence {
                            extension_address: extension.address,
                            activation_address,
                            lane_add_address: add.address,
                            lane_add_destination: add.destination.clone(),
                        });
                    }
                }
            }
        }
    }
    evidence
}

fn activation_evidence(
    activation_register: &str,
    vector_by_element: bool,
    multiply_address: u64,
    activation_loads: &[VectorRegisterInstruction],
    broadcasts: &[VectorRegisterInstruction],
) -> Option<u64> {
    let candidates = if vector_by_element {
        activation_loads
    } else {
        broadcasts
    };
    candidates
        .iter()
        .filter(|instruction| {
            instruction.address < multiply_address && instruction.register == activation_register
        })
        .map(|instruction| instruction.address)
        .max()
}

fn vector_schedules(
    recurrences: &[VectorRecurrenceEvidence],
    scale_loads: &[VectorRegisterInstruction],
    multiplies: &[TernaryVectorInstruction],
    adds: &[TernaryVectorInstruction],
    stores: &[VectorRegisterInstruction],
    simd_register_mentions: &[VectorRegisterInstruction],
    control_flow: &[(u64, ControlFlow)],
) -> Vec<VectorScheduleEvidence> {
    let mut evidence = Vec::new();
    for recurrence in recurrences {
        for scale_load in scale_loads {
            if scale_loads.iter().any(|earlier| {
                earlier.register == scale_load.register && earlier.address < scale_load.address
            }) {
                continue;
            }
            for multiply in multiplies {
                if multiply.address <= scale_load.address
                    || !instruction_dominates(control_flow, scale_load.address, multiply.address)
                    || register_is_mentioned_between(
                        &scale_load.register,
                        scale_load.address,
                        multiply.address,
                        simd_register_mentions,
                    )
                {
                    continue;
                }
                let consumes_block_sum_and_scale = (multiply.first_source
                    == recurrence.lane_add_destination
                    && multiply.second_source == scale_load.register)
                    || (multiply.second_source == recurrence.lane_add_destination
                        && multiply.first_source == scale_load.register);
                if !consumes_block_sum_and_scale {
                    continue;
                }
                for add in adds {
                    if add.address <= multiply.address
                        || (multiply.destination != add.first_source
                            && multiply.destination != add.second_source)
                    {
                        continue;
                    }
                    if stores.iter().any(|store| {
                        add.address < store.address && store.register == add.destination
                    }) {
                        evidence.push(VectorScheduleEvidence {
                            recurrence: recurrence.clone(),
                            scale_multiply_address: multiply.address,
                        });
                    }
                }
            }
        }
    }
    evidence
}

fn register_is_mentioned_between(
    register: &str,
    definition_address: u64,
    use_address: u64,
    mentions: &[VectorRegisterInstruction],
) -> bool {
    mentions.iter().any(|mention| {
        mention.register == register
            && definition_address < mention.address
            && mention.address < use_address
    })
}

pub(super) struct Instruction<'a> {
    pub(super) mnemonic: &'a str,
    pub(super) base_mnemonic: &'a str,
    pub(super) operands: &'a str,
}

pub(super) fn parse_instruction(line: &str) -> Option<(u64, Instruction<'_>)> {
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

pub(super) fn has_hex_address_prefix(line: &str) -> bool {
    line.split_once(':').is_some_and(|(address, _)| {
        let address = address.trim();
        !address.is_empty() && address.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

pub(super) fn parse_direct_branch_target(operands: &str) -> Option<u64> {
    let token = operands
        .split(|character: char| character == ',' || character.is_whitespace())
        .rfind(|token| !token.is_empty())?;
    let token = token.strip_prefix("0x").unwrap_or(token);
    u64::from_str_radix(token, 16).ok()
}

pub(super) fn is_target_label(line: &str, expected: &str) -> bool {
    line == format!("{expected}:")
        || line.ends_with(&format!("<{expected}>:"))
        || line.ends_with(&format!(" {expected}:"))
}

pub(super) fn is_other_global_label(line: &str, expected: &str) -> bool {
    if is_target_label(line, expected) {
        return false;
    }
    let Some(label) = line.strip_suffix(':') else {
        return false;
    };
    let label = label.trim();
    label.starts_with('_') && !label.contains(char::is_whitespace)
}

pub(super) fn is_forbidden_mnemonic(mnemonic: &str) -> bool {
    if [
        "fmadd", "fmsub", "fnmadd", "fnmsub", "fmla", "fmls", "fmad", "fmsb", "bfmla", "fcmla",
    ]
    .iter()
    .any(|prefix| mnemonic.starts_with(prefix))
        || ["dot", "mmla", "mopa", "mops"]
            .iter()
            .any(|suffix| mnemonic.ends_with(suffix))
    {
        return true;
    }
    matches!(
        mnemonic,
        "faddv"
            | "faddp"
            | "addv"
            | "addp"
            | "saddlv"
            | "saddlp"
            | "uaddlv"
            | "uaddlp"
            | "sadalp"
            | "uadalp"
            | "fadda"
    )
}

pub(super) fn first_operand_is_scalar_s(operands: &str) -> bool {
    let first = operands.trim_start().trim_start_matches(['[', '{']);
    let bytes = first.as_bytes();
    bytes.len() >= 2
        && bytes[0] == b's'
        && bytes[1].is_ascii_digit()
        && (bytes.len() == 2 || !bytes[2].is_ascii_alphanumeric())
}

pub(super) fn is_conditional_branch(mnemonic: &str) -> bool {
    mnemonic.starts_with("b.") || matches!(mnemonic, "cbz" | "cbnz" | "tbz" | "tbnz")
}

fn operands(operands: &str) -> impl Iterator<Item = &str> {
    operands.split(',').map(str::trim)
}

fn operand_has_arrangement(operand: &str, arrangement: &str) -> bool {
    operand
        .trim_matches(|character: char| {
            character.is_whitespace()
                || matches!(character, '{' | '}' | '[' | ']' | '!' | '(' | ')')
        })
        .split_whitespace()
        .next()
        .is_some_and(|register| register.ends_with(arrangement))
}

fn first_operand_has_arrangement(operands_text: &str, arrangement: &str) -> bool {
    operands(operands_text)
        .next()
        .is_some_and(|operand| operand_has_arrangement(operand, arrangement))
}

fn first_two_operands_have_arrangement(
    operands_text: &str,
    first_arrangement: &str,
    second_arrangement: &str,
) -> bool {
    let mut parsed = operands(operands_text);
    parsed
        .next()
        .is_some_and(|operand| operand_has_arrangement(operand, first_arrangement))
        && parsed
            .next()
            .is_some_and(|operand| operand_has_arrangement(operand, second_arrangement))
}

fn instruction_has_arrangement(mnemonic: &str, operands_text: &str, arrangement: &str) -> bool {
    mnemonic.ends_with(arrangement)
        || operands(operands_text).any(|operand| operand_has_arrangement(operand, arrangement))
}

fn is_signed_widen(
    full_mnemonic: &str,
    base_mnemonic: &str,
    operands_text: &str,
    destination_arrangement: &str,
    source_arrangement: &str,
) -> bool {
    if !matches!(base_mnemonic, "sshll" | "sxtl") {
        return false;
    }
    let explicit_arrangements = first_two_operands_have_arrangement(
        operands_text,
        destination_arrangement,
        source_arrangement,
    );
    let llvm_macho_arrangement =
        full_mnemonic.ends_with(destination_arrangement) && !has_vector_arrangement(operands_text);
    if !explicit_arrangements && !llvm_macho_arrangement {
        return false;
    }
    if base_mnemonic == "sxtl" {
        return operands(operands_text).count() == 2;
    }
    operands(operands_text)
        .nth(2)
        .is_some_and(|shift| matches!(shift, "#0" | "#0x0"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UnaryVectorInstruction {
    address: u64,
    destination: String,
    source: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TernaryVectorInstruction {
    address: u64,
    destination: String,
    first_source: String,
    second_source: String,
    first_source_is_lane_zero: bool,
    second_source_is_lane_zero: bool,
}

fn vector_register_name(operand: &str) -> Option<String> {
    let token = operand
        .trim()
        .trim_start_matches(['{', '['])
        .trim_start()
        .split(['.', '[', ']', ' ', '}'])
        .next()?;
    let bytes = token.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'v' || !bytes[1..].iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(token.to_owned())
}

fn simd_register_name(operand: &str) -> Option<String> {
    let token = operand
        .trim()
        .trim_start_matches(['{', '['])
        .trim_start()
        .split(['.', '[', ']', ' ', '}'])
        .next()?;
    let bytes = token.as_bytes();
    if bytes.len() < 2
        || !matches!(bytes[0], b'v' | b'q' | b'd' | b's' | b'h' | b'b')
        || !bytes[1..].iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    Some(format!("v{}", &token[1..]))
}

fn first_simd_register(operands_text: &str) -> Option<String> {
    simd_register_name(operands(operands_text).next()?)
}

fn simd_registers(operands_text: &str) -> impl Iterator<Item = String> + '_ {
    operands(operands_text).filter_map(simd_register_name)
}

fn unary_vector_registers(address: u64, operands_text: &str) -> Option<UnaryVectorInstruction> {
    let mut parsed = operands(operands_text);
    Some(UnaryVectorInstruction {
        address,
        destination: vector_register_name(parsed.next()?)?,
        source: vector_register_name(parsed.next()?)?,
    })
}

fn ternary_vector_registers(address: u64, operands_text: &str) -> Option<TernaryVectorInstruction> {
    let mut parsed = operands(operands_text);
    let destination = parsed.next()?;
    let first_source = parsed.next()?;
    let second_source = parsed.next()?;
    Some(TernaryVectorInstruction {
        address,
        destination: vector_register_name(destination)?,
        first_source: vector_register_name(first_source)?,
        second_source: vector_register_name(second_source)?,
        first_source_is_lane_zero: operand_is_lane_zero(first_source),
        second_source_is_lane_zero: operand_is_lane_zero(second_source),
    })
}

fn ternary_simd_registers(address: u64, operands_text: &str) -> Option<TernaryVectorInstruction> {
    let mut parsed = operands(operands_text);
    let destination = parsed.next()?;
    let first_source = parsed.next()?;
    let second_source = parsed.next()?;
    Some(TernaryVectorInstruction {
        address,
        destination: simd_register_name(destination)?,
        first_source: simd_register_name(first_source)?,
        second_source: simd_register_name(second_source)?,
        first_source_is_lane_zero: operand_is_lane_zero(first_source),
        second_source_is_lane_zero: operand_is_lane_zero(second_source),
    })
}

fn operand_is_lane_zero(operand: &str) -> bool {
    operand.trim().ends_with("[0]")
}

fn chained_instruction_addresses(
    first: &[UnaryVectorInstruction],
    second: &[UnaryVectorInstruction],
) -> Vec<UnaryVectorInstruction> {
    second
        .iter()
        .filter(|later| {
            first.iter().any(|earlier| {
                earlier.address < later.address && earlier.destination == later.source
            })
        })
        .cloned()
        .collect()
}

fn has_vector_element_operand(operands_text: &str) -> bool {
    operands(operands_text).any(|operand| {
        let operand = operand.trim();
        operand.starts_with('v')
            && operand.contains('[')
            && operand.ends_with(']')
            && vector_register_name(operand).is_some()
    })
}

fn is_four_lane_broadcast(full_mnemonic: &str, base_mnemonic: &str, operands_text: &str) -> bool {
    match base_mnemonic {
        "ld1r" => instruction_has_arrangement(full_mnemonic, operands_text, ".4s"),
        "dup" => instruction_has_arrangement(full_mnemonic, operands_text, ".4s"),
        "fmul" => {
            instruction_has_arrangement(full_mnemonic, operands_text, ".4s")
                && has_vector_element_operand(operands_text)
        }
        _ => false,
    }
}

fn first_operand_is_q_register(operands_text: &str) -> bool {
    let Some(first) = operands(operands_text).next() else {
        return false;
    };
    let first = first.trim_start_matches(['{', '[']).trim_start();
    let bytes = first.as_bytes();
    bytes.len() >= 2
        && bytes[0] == b'q'
        && bytes[1].is_ascii_digit()
        && (bytes.len() == 2 || !bytes[2].is_ascii_alphanumeric())
}

fn is_128_bit_vector_store(mnemonic: &str, operands_text: &str) -> bool {
    if uses_stack_pointer_address(operands_text) {
        return false;
    }
    match mnemonic {
        "str" | "stur" => first_operand_is_q_register(operands_text),
        "st1" => first_operand_has_arrangement(operands_text, ".4s"),
        _ => false,
    }
}

fn uses_stack_pointer_address(operands_text: &str) -> bool {
    operands_text
        .split_once('[')
        .is_some_and(|(_, address)| address.trim_start().starts_with("sp"))
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
    const NEON_HELPER: &str =
        "df_kernel_neon_v1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

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

    fn neon_vector_snippet(with_tail: bool) -> String {
        let tail = if with_tail {
            "0000000100000030:\tscvtf s2, w8\n\
0000000100000034:\tfmul s2, s2, s3\n\
0000000100000038:\tfadd s1, s1, s2\n\
000000010000003c:\tstr s1, [x2, x7, lsl #2]\n\
0000000100000040:\tret\n"
        } else {
            "0000000100000030:\tret\n"
        };
        format!(
            "\
(__TEXT,__text) section\n\
_{NEON_HELPER}:\n\
0000000100000000:\tsshll.8h v2, v2, #0x0\n\
0000000100000004:\tsshll.4s v2, v2, #0x0\n\
0000000100000008:\tscvtf.4s v2, v2\n\
000000010000000c:\tldr s3, [x0, x12, lsl #2]\n\
0000000100000010:\tfmul.4s v2, v2, v3[0]\n\
0000000100000014:\tfadd.4s v1, v1, v2\n\
0000000100000018:\tsubs w8, w8, #1\n\
000000010000001c:\tb.ne 0x100000000\n\
0000000100000020:\tldr q4, [x1]\n\
0000000100000024:\tfmul.4s v1, v1, v4\n\
0000000100000028:\tfadd.4s v0, v0, v1\n\
000000010000002c:\tstr q0, [x2]\n\
{tail}"
        )
    }

    fn scalar_only_neon_snippet() -> String {
        format!(
            "\
(__TEXT,__text) section\n\
_{NEON_HELPER}:\n\
0000000100000000:\tscvtf s2, w8\n\
0000000100000004:\tfmul s2, s2, s3\n\
0000000100000008:\tfadd s1, s1, s2\n\
000000010000000c:\tsubs w8, w8, #1\n\
0000000100000010:\tb.ne 0x100000000\n\
0000000100000014:\tstr s1, [x2]\n\
0000000100000018:\tret\n"
        )
    }

    fn singleton_neon_snippet(scale_register_clobbered: bool) -> String {
        let between_load_and_use = if scale_register_clobbered {
            "movi.2d v2, #0000000000000000"
        } else {
            "nop"
        };
        format!(
            "\
(__TEXT,__text) section\n\
_{NEON_HELPER}:\n\
0000000100000000:\tldr w8, [x1, #0x10]\n\
0000000100000004:\tfmov d0, x8\n\
0000000100000008:\tsshll.8h v0, v0, #0x0\n\
000000010000000c:\tsshll.4s v0, v0, #0x0\n\
0000000100000010:\tldr s1, [x0]\n\
0000000100000014:\tldr q2, [x1]\n\
0000000100000018:\t{between_load_and_use}\n\
000000010000001c:\tscvtf.4s v0, v0\n\
0000000100000020:\tfmul.4s v0, v0, v1[0]\n\
0000000100000024:\tmovi.2d v1, #0000000000000000\n\
0000000100000028:\tfadd.4s v0, v1, v0\n\
000000010000002c:\tfmul.4s v0, v0, v2\n\
0000000100000030:\tfadd.4s v0, v1, v0\n\
0000000100000034:\tstr q0, [x2]\n\
0000000100000038:\tret\n"
        )
    }

    fn hoisted_loop_neon_snippet(bypass_scale_load: bool, intervening: &str) -> String {
        let entry = if bypass_scale_load {
            "cbz w7, 0x100000010"
        } else {
            "nop"
        };
        format!(
            "\
(__TEXT,__text) section\n\
_{NEON_HELPER}:\n\
0000000100000000:\t{entry}\n\
0000000100000004:\tldr q0, [x1], #0x10\n\
0000000100000008:\t{intervening}\n\
000000010000000c:\tmovi.2d v1, #0000000000000000\n\
0000000100000010:\tldr w9, [x1, x8, lsl #2]\n\
0000000100000014:\tfmov d2, x9\n\
0000000100000018:\tsshll.8h v2, v2, #0x0\n\
000000010000001c:\tsshll.4s v2, v2, #0x0\n\
0000000100000020:\tscvtf.4s v2, v2\n\
0000000100000024:\tldr s3, [x0, x8, lsl #2]\n\
0000000100000028:\tfmul.4s v2, v2, v3[0]\n\
000000010000002c:\tfadd.4s v1, v1, v2\n\
0000000100000030:\tadd x8, x8, #0x1\n\
0000000100000034:\tcmp x8, #0x2\n\
0000000100000038:\tb.lo 0x100000010\n\
000000010000003c:\tfmul.4s v0, v1, v0\n\
0000000100000040:\tmovi.2d v1, #0000000000000000\n\
0000000100000044:\tfadd.4s v0, v1, v0\n\
0000000100000048:\tstr q0, [x2]\n\
000000010000004c:\tret\n"
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
            "saddlp.4s v0, v1",
            "uaddlp.4s v0, v1",
            "fadda s0, p0, s1, z0.s",
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

    #[test]
    fn accepts_literal_signed_neon_recurrence_and_broadcast_forms() {
        let literal =
            audit_neon_helper_disassembly(NEON_HELPER, &neon_vector_snippet(false), 4, 33).unwrap();
        assert!(literal.vector_path_observed());
        assert!(!literal.scalar_tail_observed());
        assert_eq!(literal.signed_widen_8_to_16_count(), 1);
        assert_eq!(literal.signed_widen_16_to_32_count(), 1);
        assert_eq!(literal.signed_q8_to_i32_count(), 1);
        assert_eq!(literal.vector_broadcast_count(), 1);
        assert_eq!(literal.vector_store_count(), 1);
        assert!(literal.logical_vector_lane_loop_observed());

        let ld1r = neon_vector_snippet(false)
            .replace("ldr s3, [x0, x12, lsl #2]", "ld1r.4s { v3 }, [x0]")
            .replace("fmul.4s v2, v2, v3[0]", "fmul.4s v2, v2, v3");
        audit_neon_helper_disassembly(NEON_HELPER, &ld1r, 4, 33).unwrap();

        let mut dup = neon_vector_snippet(false);
        for (from, to) in [
            ("0000000100000030:", "0000000100000034:"),
            ("000000010000002c:", "0000000100000030:"),
            ("0000000100000028:", "000000010000002c:"),
            ("0000000100000024:", "0000000100000028:"),
            ("0000000100000020:", "0000000100000024:"),
            ("000000010000001c:", "0000000100000020:"),
            ("0000000100000018:", "000000010000001c:"),
            ("0000000100000014:", "0000000100000018:"),
            ("0000000100000010:", "0000000100000014:"),
        ] {
            dup = dup.replace(from, to);
        }
        dup = dup
            .replace(
                "000000010000000c:\tldr s3, [x0, x12, lsl #2]",
                "000000010000000c:\tldr s3, [x0, x12, lsl #2]\n0000000100000010:\tdup.4s v3, v3[0]",
            )
            .replace("fmul.4s v2, v2, v3[0]", "fmul.4s v2, v2, v3");
        audit_neon_helper_disassembly(NEON_HELPER, &dup, 4, 33).unwrap();
    }

    #[test]
    fn enforces_vector_tile_tail_and_singleton_shape_contracts() {
        let tail =
            audit_neon_helper_disassembly(NEON_HELPER, &neon_vector_snippet(true), 5, 33).unwrap();
        assert!(tail.vector_path_observed());
        assert!(tail.scalar_tail_observed());

        let scalar_only =
            audit_neon_helper_disassembly(NEON_HELPER, &scalar_only_neon_snippet(), 3, 33).unwrap();
        assert!(!scalar_only.vector_path_observed());
        assert!(scalar_only.scalar_tail_observed());

        let straight_line = neon_vector_snippet(false)
            .replace("subs w8, w8, #1", "nop")
            .replace("b.ne 0x100000000", "nop");
        let singleton = audit_neon_helper_disassembly(NEON_HELPER, &straight_line, 4, 1).unwrap();
        assert!(!singleton.logical_vector_lane_loop_observed());

        let hoisted_scale = singleton_neon_snippet(false);
        let hoisted = audit_neon_helper_disassembly(NEON_HELPER, &hoisted_scale, 4, 1).unwrap();
        assert!(hoisted.vector_path_observed());
        assert!(!hoisted.logical_vector_lane_loop_observed());
        assert!(
            audit_neon_helper_disassembly(NEON_HELPER, &singleton_neon_snippet(true), 4, 1)
                .is_err()
        );

        let hoisted_loop = hoisted_loop_neon_snippet(false, "nop");
        let loop_report = audit_neon_helper_disassembly(NEON_HELPER, &hoisted_loop, 4, 2).unwrap();
        assert!(loop_report.logical_vector_lane_loop_observed());
        for intervening in [
            "movi.2d v0, #0000000000000000",
            "ldr q0, [x2]",
            "ldp q3, q0, [x0]",
        ] {
            assert!(
                audit_neon_helper_disassembly(
                    NEON_HELPER,
                    &hoisted_loop_neon_snippet(false, intervening),
                    4,
                    2,
                )
                .is_err(),
                "intervening scale-register use must be rejected: {intervening}"
            );
        }
        assert!(
            audit_neon_helper_disassembly(
                NEON_HELPER,
                &hoisted_loop_neon_snippet(true, "nop"),
                4,
                2,
            )
            .is_err(),
            "a scale load that does not dominate its multiply must be rejected"
        );

        assert!(audit_neon_helper_disassembly(NEON_HELPER, &straight_line, 4, 2).is_err());
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &hoisted_scale, 4, 2).is_err());
        assert!(
            audit_neon_helper_disassembly(NEON_HELPER, &neon_vector_snippet(false), 5, 33).is_err()
        );
        assert!(
            audit_neon_helper_disassembly(NEON_HELPER, &neon_vector_snippet(true), 4, 33).is_err()
        );
        assert!(
            audit_neon_helper_disassembly(NEON_HELPER, &neon_vector_snippet(true), 3, 33).is_err()
        );
    }

    #[test]
    fn rejects_neon_recurrence_and_tail_mutations() {
        let vector = neon_vector_snippet(false);
        for (from, to) in [
            ("sshll.8h v2, v2, #0x0", "mov v2.8b, v2.8b"),
            ("sshll.4s v2, v2, #0x0", "mov v2.4s, v2.4s"),
            ("scvtf.4s v2, v2", "mov v2.4s, v2.4s"),
            ("fmul.4s v2, v2, v3[0]", "fmul.4s v2, v2, v3"),
            ("fadd.4s v1, v1, v2", "mov v1.4s, v2.4s"),
            ("str q0, [x2]", "str x0, [x2]"),
            ("ldr q4, [x1]", "ldr q4, [sp]"),
            ("fmul.4s v1, v1, v4", "mov.4s v1, v1"),
            ("fadd.4s v0, v0, v1", "mov.4s v0, v1"),
            ("str q0, [x2]", "str q0, [sp]"),
            ("ldr s3, [x0, x12, lsl #2]", "ldr s3, [sp]"),
        ] {
            let mutated = vector.replace(from, to);
            assert!(
                audit_neon_helper_disassembly(NEON_HELPER, &mutated, 4, 33).is_err(),
                "mutation {from} -> {to} must fail"
            );
        }

        let tail_without_store = neon_vector_snippet(true).replace(
            "000000010000003c:\tstr s1, [x2, x7, lsl #2]\n",
            "000000010000003c:\tstr w1, [x2, x7, lsl #2]\n",
        );
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &tail_without_store, 5, 33).is_err());
        let tail_stack_spill =
            neon_vector_snippet(true).replace("str s1, [x2, x7, lsl #2]", "str s1, [sp]");
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &tail_stack_spill, 5, 33).is_err());

        let bad_dataflow = neon_vector_snippet(false)
            .replace("sshll.8h v2, v2", "sshll.8h v4, v2")
            .replace("sshll.4s v2, v2", "sshll.4s v2, v3");
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &bad_dataflow, 4, 33).is_err());

        let outer_loop_only = neon_vector_snippet(false)
            .replace("subs w8, w8, #1", "nop")
            .replace("b.ne 0x100000000", "nop")
            .replace(
                "000000010000002c:\tstr q0, [x2]",
                "000000010000002c:\tb.ne 0x100000000",
            )
            .replace(
                "0000000100000030:\tret",
                "0000000100000030:\tstr q0, [x2]\n0000000100000034:\tret",
            );
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &outer_loop_only, 4, 33).is_err());
    }

    #[test]
    fn rejects_neon_forbidden_control_flow_and_arithmetic() {
        let vector = neon_vector_snippet(false);
        for forbidden in [
            "fmla.4s v1, v2, v3",
            "fmadd s0, s1, s2, s3",
            "sdot.4s v0, v1, v2",
            "smmla.4s v0, v1, v2",
            "fmlal.4s v0, v1, v2",
            "fmlsl2.4s v0, v1, v2",
            "bfmla.4s v0, v1, v2",
            "bfdot.4s v0, v1, v2",
            "fcmla.4s v0, v1, v2, #0",
            "smopa.4s za0, p0, p1, z0, z1",
            "faddv s0, v1.4s",
            "bl _outside",
            "blr x8",
            "br x8",
        ] {
            let mutated = vector.replace(
                "0000000100000018:\tsubs w8, w8, #1",
                &format!("0000000100000018:\t{forbidden}"),
            );
            assert!(
                audit_neon_helper_disassembly(NEON_HELPER, &mutated, 4, 33).is_err(),
                "{forbidden} must fail"
            );
        }

        let outside = vector.replace("b.ne 0x100000000", "b.ne 0x200000000");
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &outside, 4, 33).is_err());
        let middle = vector.replace("b.ne 0x100000000", "b.ne 0x100000002");
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &middle, 4, 33).is_err());
        let multiple_returns =
            vector.replace("000000010000002c:\tstr q0, [x2]", "000000010000002c:\tret");
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &multiple_returns, 4, 33).is_err());
        let after_return = format!("{}0000000100000034:\tnop\n", vector);
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &after_return, 4, 33).is_err());

        let unreachable = vector
            .replace(
                "0000000100000000:\tsshll.8h v2, v2, #0x0",
                "0000000100000000:\tb 0x100000020\n0000000100000002:\tsshll.8h v2, v2, #0x0",
            )
            .replace("0000000100000004:", "0000000100000006:")
            .replace("0000000100000008:", "000000010000000a:")
            .replace("000000010000000c:", "000000010000000e:")
            .replace("0000000100000010:", "0000000100000012:")
            .replace("0000000100000014:", "0000000100000016:")
            .replace("0000000100000018:", "000000010000001a:")
            .replace("000000010000001c:", "000000010000001e:");
        assert!(audit_neon_helper_disassembly(NEON_HELPER, &unreachable, 4, 33).is_err());
    }
}

//! Region-to-Loop lowering for the fixed G1 Q8 linear contract.

use crate::Result;
use crate::ir::{KernelVariant, LoopKernelV1, Q8LinearRegion};

/// Lower one verified Q8 linear region into a fixed scalar or NEON Loop IR.
pub fn lower_q8_linear(region: &Q8LinearRegion, variant: KernelVariant) -> Result<LoopKernelV1> {
    LoopKernelV1::new(region, variant)
}

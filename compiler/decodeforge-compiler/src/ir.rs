//! Verified target-independent Region and fixed Loop IR for the G1 slice.

use crate::pack::PackSpecV1;
use crate::{
    BLOCK_SIZE, OUTPUT_TILE, RECORD_BYTES, Result, SCHEMA_VERSION, checked_mul, checked_usize,
    invalid,
};
use decodeforge_core::q8::{self, Q8Weights};
use serde::{Deserialize, Deserializer, Serialize};

/// The only region operation admitted by this G1 slice.
pub const OPERATOR_Q8_LINEAR: &str = "q8_linear";
/// The fixed G1 numeric contract.
pub const NUMERIC_MODE_STRICT_F32_V1: &str = "strict_f32_v1";

/// Static `M=1`, `[N,K]` shape for a Q8 linear region.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct Q8LinearShape {
    m: u32,
    n: u32,
    k: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShapeWire {
    m: u32,
    n: u32,
    k: u32,
}

impl<'de> Deserialize<'de> for Q8LinearShape {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ShapeWire::deserialize(deserializer)?;
        Self::from_mnk(wire.m, wire.n, wire.k).map_err(serde::de::Error::custom)
    }
}

impl Q8LinearShape {
    /// Construct a checked `M=1` shape.
    pub fn new(n: u32, k: u32) -> Result<Self> {
        Self::from_mnk(1, n, k)
    }

    /// Construct from host-sized dimensions, rejecting values outside the ABI
    /// `u32` domain before any derived arithmetic.
    pub fn from_usize(n: usize, k: usize) -> Result<Self> {
        let n = u32::try_from(n).map_err(|_| {
            invalid(
                "DFE-COMP-001",
                "N is not representable by the ABI u32 field.",
            )
        })?;
        let k = u32::try_from(k).map_err(|_| {
            invalid(
                "DFE-COMP-001",
                "K is not representable by the ABI u32 field.",
            )
        })?;
        Self::new(n, k)
    }

    /// Construct from an explicit matrix shape, accepting only `M=1`.
    pub fn from_mnk(m: u32, n: u32, k: u32) -> Result<Self> {
        if m != 1 {
            return Err(invalid(
                "DFE-COMP-001",
                format!("Q8LinearShape requires M=1, got M={m}."),
            ));
        }
        if n == 0 || k == 0 {
            return Err(invalid(
                "DFE-COMP-001",
                format!("N and K must be positive, got N={n}, K={k}."),
            ));
        }

        let blocks = (k as u64).div_ceil(u64::from(BLOCK_SIZE));
        let panels = (n as u64).div_ceil(u64::from(OUTPUT_TILE));
        let padded_n = checked_mul(panels, u64::from(OUTPUT_TILE), "padded N")?;
        let padded_k = checked_mul(blocks, u64::from(BLOCK_SIZE), "padded K")?;
        if padded_n > u64::from(u32::MAX) || padded_k > u64::from(u32::MAX) {
            return Err(invalid(
                "DFE-COMP-002",
                "padded dimensions are not representable by the ABI u32 fields.",
            ));
        }

        // Check every derived count used by the region, packer, and future ABI
        // before retaining the shape.  No allocation occurs here.
        let _ = checked_usize(
            checked_mul(n as u64, k as u64, "logical elements")?,
            "logical elements",
        )?;
        let _ = checked_usize(
            checked_mul(n as u64, blocks, "logical q storage")?,
            "logical q storage",
        )?;
        let _ = checked_usize(checked_mul(panels, blocks, "record count")?, "record count")?;
        let payload = checked_mul(
            checked_mul(panels, blocks, "payload records")?,
            u64::from(RECORD_BYTES),
            "payload",
        )?;
        let _ = checked_usize(payload, "payload")?;

        Ok(Self { m, n, k })
    }

    /// Matrix batch dimension, always `1`.
    pub const fn m(&self) -> u32 {
        self.m
    }

    /// Logical output-channel count.
    pub const fn n(&self) -> u32 {
        self.n
    }

    /// Logical reduction/input-channel count.
    pub const fn k(&self) -> u32 {
        self.k
    }

    /// Number of physical 32-lane K blocks.
    pub const fn blocks(&self) -> u32 {
        // `self.k > 0` is guaranteed by the constructor, so this cannot
        // overflow and remains a pure accessor.
        (self.k.saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE
    }

    /// Number of four-output physical panels.
    pub const fn panels(&self) -> u32 {
        (self.n.saturating_add(OUTPUT_TILE - 1)) / OUTPUT_TILE
    }

    /// Padded physical output count.
    pub const fn padded_n(&self) -> u32 {
        self.panels().saturating_mul(OUTPUT_TILE)
    }

    /// Padded physical K count.  Padding is storage-only; it is not logical.
    pub const fn padded_k(&self) -> u32 {
        self.blocks().saturating_mul(BLOCK_SIZE)
    }

    /// Number of bytes in the exact headerless OI4 payload.
    pub fn payload_bytes(&self) -> Result<usize> {
        let records = checked_mul(
            self.panels() as u64,
            self.blocks() as u64,
            "payload records",
        )?;
        let payload = checked_mul(records, u64::from(RECORD_BYTES), "payload")?;
        checked_usize(payload, "payload")
    }

    /// Byte offset of one `(panel, block)` record, or `None` if out of range.
    pub fn record_offset(&self, panel: u32, block: u32) -> Option<usize> {
        if panel >= self.panels() || block >= self.blocks() {
            return None;
        }
        let index = (panel as u64)
            .checked_mul(self.blocks() as u64)?
            .checked_add(block as u64)?;
        let offset = index.checked_mul(u64::from(RECORD_BYTES))?;
        usize::try_from(offset).ok()
    }

    /// Checked record offset with a diagnostic instead of `None`.
    pub fn checked_record_offset(&self, panel: u32, block: u32) -> Result<usize> {
        self.record_offset(panel, block).ok_or_else(|| {
            invalid(
                "DFE-COMP-003",
                format!("record index ({panel}, {block}) is outside the shape."),
            )
        })
    }

    /// Whether the logical output count needs scalar N-tail cleanup.
    pub const fn has_n_tail(&self) -> bool {
        !self.n.is_multiple_of(OUTPUT_TILE)
    }

    /// Whether the logical K count has physical zero padding.
    pub const fn has_k_padding(&self) -> bool {
        !self.k.is_multiple_of(BLOCK_SIZE)
    }

    /// Verify all shape invariants again at an API boundary.
    pub fn verify(&self) -> Result<()> {
        let expected = Self::from_mnk(self.m, self.n, self.k)?;
        if *self == expected {
            Ok(())
        } else {
            Err(invalid("DFE-COMP-006", "Q8LinearShape invariant mismatch."))
        }
    }
}

impl TryFrom<(u32, u32)> for Q8LinearShape {
    type Error = crate::CompilerError;

    fn try_from((n, k): (u32, u32)) -> Result<Self> {
        Self::new(n, k)
    }
}

impl TryFrom<(usize, usize)> for Q8LinearShape {
    type Error = crate::CompilerError;

    fn try_from((n, k): (usize, usize)) -> Result<Self> {
        Self::from_usize(n, k)
    }
}

/// A target-independent Q8 linear region bound to a logical weight identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Q8LinearRegion {
    schema_version: u32,
    operator: String,
    shape: Q8LinearShape,
    logical_weight_identity: String,
    numeric_mode: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegionWire {
    schema_version: u32,
    operator: String,
    shape: Q8LinearShape,
    logical_weight_identity: String,
    numeric_mode: String,
}

impl<'de> Deserialize<'de> for Q8LinearRegion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegionWire::deserialize(deserializer)?;
        Self::try_new_with_schema(
            wire.schema_version,
            &wire.operator,
            wire.shape,
            &wire.logical_weight_identity,
            &wire.numeric_mode,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl Q8LinearRegion {
    /// Construct a region from a checked shape and G0 logical identity.
    pub fn new(shape: Q8LinearShape, logical_weight_identity: impl Into<String>) -> Result<Self> {
        let identity = logical_weight_identity.into();
        Self::try_new_with_schema(
            SCHEMA_VERSION,
            OPERATOR_Q8_LINEAR,
            shape,
            &identity,
            NUMERIC_MODE_STRICT_F32_V1,
        )
    }

    /// Construct a region directly from immutable G0 Q8 weights.
    pub fn from_weights(weights: &Q8Weights) -> Result<Self> {
        let shape = Q8LinearShape::new(weights.n(), weights.k())?;
        Self::new(shape, q8::logical_weight_identity(weights))
    }

    fn try_new_with_schema(
        schema_version: u32,
        operator: &str,
        shape: Q8LinearShape,
        logical_weight_identity: &str,
        numeric_mode: &str,
    ) -> Result<Self> {
        shape.verify()?;
        if schema_version != SCHEMA_VERSION {
            return Err(invalid(
                "DFE-COMP-001",
                format!("unsupported region schema version {schema_version}."),
            ));
        }
        if operator != OPERATOR_Q8_LINEAR {
            return Err(invalid(
                "DFE-COMP-001",
                format!("unsupported region operator {operator:?}."),
            ));
        }
        if numeric_mode != NUMERIC_MODE_STRICT_F32_V1 {
            return Err(invalid(
                "DFE-COMP-001",
                format!("unsupported numeric mode {numeric_mode:?}."),
            ));
        }
        if !crate::is_sha256_identity(logical_weight_identity) {
            return Err(invalid(
                "DFE-COMP-004",
                "logical weight identity must be sha256:<64 lowercase hex digits>.",
            ));
        }
        Ok(Self {
            schema_version,
            operator: operator.to_owned(),
            shape,
            logical_weight_identity: logical_weight_identity.to_owned(),
            numeric_mode: numeric_mode.to_owned(),
        })
    }

    /// Region schema major version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Region operator name.
    pub fn operator(&self) -> &str {
        &self.operator
    }

    /// Region shape.
    pub const fn shape(&self) -> Q8LinearShape {
        self.shape
    }

    /// G0 logical-weight identity bound to this region.
    pub fn logical_weight_identity(&self) -> &str {
        &self.logical_weight_identity
    }

    /// Fixed arithmetic mode.
    pub fn numeric_mode(&self) -> &str {
        &self.numeric_mode
    }

    /// Recheck region invariants.
    pub fn verify(&self) -> Result<()> {
        Self::try_new_with_schema(
            self.schema_version,
            &self.operator,
            self.shape,
            &self.logical_weight_identity,
            &self.numeric_mode,
        )
        .map(|_| ())
    }

    /// Deterministic compact JSON representation.
    pub fn canonical_json(&self) -> Result<String> {
        self.verify()?;
        serde_json::to_string(self)
            .map_err(|error| invalid("DFE-COMP-006", format!("region JSON failed: {error}")))
    }
}

/// Fixed target variant represented by this slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelVariant {
    /// One scalar output lane.
    Scalar,
    /// Four output lanes (the future ARM64 NEON mapping).
    Neon,
}

impl KernelVariant {
    /// Number of output lanes represented by this variant.
    pub const fn vector_lanes(self) -> u8 {
        match self {
            Self::Scalar => 1,
            Self::Neon => 4,
        }
    }

    /// Stable lower-case variant name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
        }
    }
}

/// Fixed vector axis: output channels, never reduction lanes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum VectorAxis {
    /// Output-channel vectorization.
    #[serde(rename = "output")]
    Output,
}

impl VectorAxis {
    pub const fn as_str(self) -> &'static str {
        "output"
    }
}

/// Fixed strict reduction traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReductionOrder {
    /// Blocks ascending, then logical lanes ascending.
    #[serde(rename = "ascending-block-lane")]
    AscendingBlockLane,
}

impl ReductionOrder {
    pub const fn as_str(self) -> &'static str {
        "ascending-block-lane"
    }
}

/// Fixed non-contracting arithmetic operation order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArithmeticContract {
    /// RN32 product followed by RN32 add; no FMA or reassociation.
    #[serde(rename = "separate-mul-add")]
    SeparateMulAdd,
}

impl ArithmeticContract {
    pub const fn as_str(self) -> &'static str {
        "separate-mul-add"
    }
}

/// Physical K padding policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum KPadding {
    /// Zero bytes may be stored, but padded lanes are never evaluated.
    #[serde(rename = "logical-only")]
    LogicalOnly,
}

impl KPadding {
    pub const fn as_str(self) -> &'static str {
        "logical-only"
    }
}

/// N tail policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NTail {
    /// Process 1–3 remaining output lanes with scalar cleanup.
    #[serde(rename = "scalar-cleanup")]
    ScalarCleanup,
}

impl NTail {
    pub const fn as_str(self) -> &'static str {
        "scalar-cleanup"
    }
}

/// Fixed, verified Loop IR kernel for one Q8 linear region.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LoopKernelV1 {
    schema_version: u32,
    shape: Q8LinearShape,
    variant: KernelVariant,
    vector_axis: VectorAxis,
    vector_lanes: u8,
    n_tile: u32,
    k_block: u32,
    k_unroll: u8,
    accumulators: u8,
    reduction_order: ReductionOrder,
    arithmetic: ArithmeticContract,
    k_padding: KPadding,
    pack: PackSpecV1,
    n_tail: NTail,
    numeric_mode: String,
    logical_weight_identity: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoopKernelWire {
    schema_version: u32,
    shape: Q8LinearShape,
    variant: KernelVariant,
    vector_axis: VectorAxis,
    vector_lanes: u8,
    n_tile: u32,
    k_block: u32,
    k_unroll: u8,
    accumulators: u8,
    reduction_order: ReductionOrder,
    arithmetic: ArithmeticContract,
    k_padding: KPadding,
    pack: PackSpecV1,
    n_tail: NTail,
    numeric_mode: String,
    logical_weight_identity: String,
}

impl<'de> Deserialize<'de> for LoopKernelV1 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LoopKernelWire::deserialize(deserializer)?;
        let kernel = Self {
            schema_version: wire.schema_version,
            shape: wire.shape,
            variant: wire.variant,
            vector_axis: wire.vector_axis,
            vector_lanes: wire.vector_lanes,
            n_tile: wire.n_tile,
            k_block: wire.k_block,
            k_unroll: wire.k_unroll,
            accumulators: wire.accumulators,
            reduction_order: wire.reduction_order,
            arithmetic: wire.arithmetic,
            k_padding: wire.k_padding,
            pack: wire.pack,
            n_tail: wire.n_tail,
            numeric_mode: wire.numeric_mode,
            logical_weight_identity: wire.logical_weight_identity,
        };
        kernel.verify().map_err(serde::de::Error::custom)?;
        Ok(kernel)
    }
}

impl LoopKernelV1 {
    /// Lower one verified region into the fixed Loop IR variant.
    pub fn new(region: &Q8LinearRegion, variant: KernelVariant) -> Result<Self> {
        region.verify()?;
        let shape = region.shape();
        let kernel = Self {
            schema_version: SCHEMA_VERSION,
            shape,
            variant,
            vector_axis: VectorAxis::Output,
            vector_lanes: variant.vector_lanes(),
            n_tile: OUTPUT_TILE,
            k_block: BLOCK_SIZE,
            k_unroll: 1,
            accumulators: 1,
            reduction_order: ReductionOrder::AscendingBlockLane,
            arithmetic: ArithmeticContract::SeparateMulAdd,
            k_padding: KPadding::LogicalOnly,
            pack: PackSpecV1::new(),
            n_tail: NTail::ScalarCleanup,
            numeric_mode: NUMERIC_MODE_STRICT_F32_V1.to_owned(),
            logical_weight_identity: region.logical_weight_identity().to_owned(),
        };
        kernel.verify()?;
        Ok(kernel)
    }

    /// Recheck all fixed Loop IR invariants.
    pub fn verify(&self) -> Result<()> {
        self.shape.verify()?;
        if self.schema_version != SCHEMA_VERSION
            || self.vector_axis != VectorAxis::Output
            || self.vector_lanes != self.variant.vector_lanes()
            || self.n_tile != OUTPUT_TILE
            || self.k_block != BLOCK_SIZE
            || self.k_unroll != 1
            || self.accumulators != 1
            || self.reduction_order != ReductionOrder::AscendingBlockLane
            || self.arithmetic != ArithmeticContract::SeparateMulAdd
            || self.k_padding != KPadding::LogicalOnly
            || self.pack != PackSpecV1::new()
            || self.n_tail != NTail::ScalarCleanup
            || self.numeric_mode != NUMERIC_MODE_STRICT_F32_V1
            || !crate::is_sha256_identity(&self.logical_weight_identity)
        {
            return Err(invalid(
                "DFE-COMP-006",
                "LoopKernelV1 violates a fixed G1 invariant.",
            ));
        }
        Ok(())
    }

    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub const fn shape(&self) -> Q8LinearShape {
        self.shape
    }

    pub const fn variant(&self) -> KernelVariant {
        self.variant
    }

    pub const fn vector_axis(&self) -> VectorAxis {
        self.vector_axis
    }

    pub const fn vector_lanes(&self) -> u8 {
        self.vector_lanes
    }

    pub const fn n_tile(&self) -> u32 {
        self.n_tile
    }

    pub const fn k_block(&self) -> u32 {
        self.k_block
    }

    pub const fn k_unroll(&self) -> u8 {
        self.k_unroll
    }

    /// Number of K partial accumulators, fixed to one.
    pub const fn accumulators(&self) -> u8 {
        self.accumulators
    }

    pub const fn reduction_order(&self) -> ReductionOrder {
        self.reduction_order
    }

    pub const fn arithmetic(&self) -> ArithmeticContract {
        self.arithmetic
    }

    /// Whether this fixed kernel uses fused multiply-add (always false).
    pub const fn uses_fma(&self) -> bool {
        false
    }

    /// Whether this fixed kernel performs a horizontal reduction (always
    /// false; output lanes are independent accumulators).
    pub const fn uses_horizontal_reduction(&self) -> bool {
        false
    }

    /// The fixed arithmetic contract never fuses a product with an add.
    pub const fn separate_mul_add(&self) -> bool {
        true
    }

    pub const fn k_padding(&self) -> KPadding {
        self.k_padding
    }

    pub const fn n_tail(&self) -> NTail {
        self.n_tail
    }

    pub const fn pack(&self) -> &PackSpecV1 {
        &self.pack
    }

    pub fn numeric_mode(&self) -> &str {
        &self.numeric_mode
    }

    pub fn logical_weight_identity(&self) -> &str {
        &self.logical_weight_identity
    }

    /// Deterministic compact JSON representation of the Loop IR.
    pub fn canonical_json(&self) -> Result<String> {
        self.verify()?;
        serde_json::to_string(self)
            .map_err(|error| invalid("DFE-COMP-006", format!("loop JSON failed: {error}")))
    }

    /// Deterministic schedule-schema-shaped JSON without the bound weight ID.
    pub fn schedule_json(&self) -> Result<String> {
        self.verify()?;
        #[derive(Serialize)]
        struct Schedule<'a> {
            schema_version: u32,
            shape: Q8LinearShape,
            variant: KernelVariant,
            vector_axis: VectorAxis,
            vector_lanes: u8,
            n_tile: u32,
            k_block: u32,
            k_unroll: u8,
            accumulators: u8,
            reduction_order: ReductionOrder,
            arithmetic: ArithmeticContract,
            k_padding: KPadding,
            pack: SchedulePack,
            n_tail: NTail,
            target: TargetDescriptor,
            numeric_mode: &'a str,
        }
        #[derive(Serialize)]
        struct SchedulePack {
            layout: &'static str,
            alignment: u32,
        }
        #[derive(Serialize)]
        struct TargetDescriptor {
            triple: &'static str,
            features: Vec<&'static str>,
        }
        let target = match self.variant {
            KernelVariant::Scalar => TargetDescriptor {
                triple: "portable",
                features: Vec::new(),
            },
            KernelVariant::Neon => TargetDescriptor {
                triple: "aarch64-apple-darwin",
                features: vec!["neon"],
            },
        };
        serde_json::to_string(&Schedule {
            schema_version: self.schema_version,
            shape: self.shape,
            variant: self.variant,
            vector_axis: self.vector_axis,
            vector_lanes: self.vector_lanes,
            n_tile: self.n_tile,
            k_block: self.k_block,
            k_unroll: self.k_unroll,
            accumulators: self.accumulators,
            reduction_order: self.reduction_order,
            arithmetic: self.arithmetic,
            k_padding: self.k_padding,
            pack: SchedulePack {
                layout: "output-interleaved",
                alignment: 16,
            },
            n_tail: self.n_tail,
            target,
            numeric_mode: &self.numeric_mode,
        })
        .map_err(|error| invalid("DFE-COMP-006", format!("schedule JSON failed: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO_ID: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

    #[test]
    fn shape_and_loop_invariants_cover_g1_contract() {
        let shape = Q8LinearShape::new(5, 33).unwrap();
        assert_eq!(shape.m(), 1);
        assert_eq!(shape.blocks(), 2);
        assert_eq!(shape.panels(), 2);
        assert_eq!(shape.padded_n(), 8);
        assert_eq!(shape.padded_k(), 64);
        assert!(shape.has_n_tail());
        assert!(shape.has_k_padding());

        let region = Q8LinearRegion::new(shape, ZERO_ID).unwrap();
        let scalar = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        let neon = LoopKernelV1::new(&region, KernelVariant::Neon).unwrap();
        for kernel in [&scalar, &neon] {
            kernel.verify().unwrap();
            assert_eq!(kernel.vector_axis(), VectorAxis::Output);
            assert_eq!(kernel.n_tile(), 4);
            assert_eq!(kernel.k_block(), 32);
            assert_eq!(kernel.k_unroll(), 1);
            assert_eq!(kernel.accumulators(), 1);
            assert_eq!(kernel.reduction_order(), ReductionOrder::AscendingBlockLane);
            assert_eq!(kernel.arithmetic(), ArithmeticContract::SeparateMulAdd);
            assert_eq!(kernel.k_padding(), KPadding::LogicalOnly);
            assert_eq!(kernel.n_tail(), NTail::ScalarCleanup);
            assert!(!kernel.uses_fma());
            assert!(!kernel.uses_horizontal_reduction());
            assert_eq!(
                kernel.canonical_json().unwrap(),
                kernel.canonical_json().unwrap()
            );
            assert!(kernel.schedule_json().unwrap().contains("separate-mul-add"));
        }
        assert_eq!(scalar.vector_lanes(), 1);
        assert_eq!(neon.vector_lanes(), 4);
    }

    #[test]
    fn invalid_shapes_and_serialized_invariants_are_rejected() {
        assert!(Q8LinearShape::new(0, 1).is_err());
        assert!(Q8LinearShape::new(1, 0).is_err());
        assert!(Q8LinearShape::from_mnk(2, 1, 1).is_err());
        assert!(Q8LinearShape::new(u32::MAX, u32::MAX).is_err());
        assert!(Q8LinearShape::from_usize(usize::MAX, 1).is_err());

        let region = Q8LinearRegion::new(Q8LinearShape::new(1, 1).unwrap(), ZERO_ID).unwrap();
        let kernel = LoopKernelV1::new(&region, KernelVariant::Scalar).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&kernel.canonical_json().unwrap()).unwrap();
        value["k_unroll"] = serde_json::Value::from(2);
        assert!(serde_json::from_value::<LoopKernelV1>(value).is_err());

        let mut value: serde_json::Value =
            serde_json::from_str(&kernel.canonical_json().unwrap()).unwrap();
        value["n_tile"] = serde_json::Value::from(8);
        assert!(serde_json::from_value::<LoopKernelV1>(value).is_err());
    }
}

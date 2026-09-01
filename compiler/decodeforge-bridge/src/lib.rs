#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

//! A deliberately small C ABI for one verified DecodeForge NEON executable.
//!
//! The bridge owns all generated code and packed weights behind a process-local
//! handle. Foreign callers only lend manifest, payload, and invocation
//! buffers for the duration of an individual call.

use decodeforge_compiler::{
    CompilerError, GeneratedRuntimeError, GeneratedStatusV1, KernelVariant, PackManifestV1,
    PackedWeightsV1, Q8LinearRegion, build_apple_neon_dylib, emit_neon_c, load_apple_neon_v1,
    lower_q8_linear,
};
use serde_json::from_slice;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::{align_of, offset_of, size_of};
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

const BRIDGE_ABI_VERSION: u32 = 1;
const IDENTITY_CSTR_BYTES: usize = 72;
const MAX_MANIFEST_BYTES: usize = 16 * 1024;
const MAX_PACKED_WEIGHT_BYTES: usize = 128 * 1024 * 1024;
const MAX_AGGREGATE_PACKED_WEIGHT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_VECTOR_BYTES: usize = MAX_PACKED_WEIGHT_BYTES;
const MAX_ERROR_BYTES: usize = 4096;
const QUIET_NAN_BITS: u32 = 0x7fc0_0000;
const MAX_LIVE_ENTRIES: usize = 256;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BridgeStatus {
    Ok = 0,
    Truncated = 1,
    NullArgument = 2,
    ZeroLength = 3,
    InvalidHandle = 4,
    InvalidArgument = 5,
    Overlap = 6,
    LimitViolation = 7,
    InvalidManifest = 8,
    InvalidPayload = 9,
    UnsupportedHost = 10,
    BuildFailed = 11,
    LoadFailed = 12,
    ExecutionFailed = 13,
    NonfiniteInput = 14,
    NonfiniteOutput = 15,
    Panic = 16,
    AllocationFailed = 17,
    Internal = 18,
}

impl BridgeStatus {
    const fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Truncated => "truncated",
            Self::NullArgument => "null_argument",
            Self::ZeroLength => "zero_length",
            Self::InvalidHandle => "invalid_handle",
            Self::InvalidArgument => "invalid_argument",
            Self::Overlap => "overlap",
            Self::LimitViolation => "limit_violation",
            Self::InvalidManifest => "invalid_manifest",
            Self::InvalidPayload => "invalid_payload",
            Self::UnsupportedHost => "unsupported_host",
            Self::BuildFailed => "build_failed",
            Self::LoadFailed => "load_failed",
            Self::ExecutionFailed => "execution_failed",
            Self::NonfiniteInput => "nonfinite_input",
            Self::NonfiniteOutput => "nonfinite_output",
            Self::Panic => "panic",
            Self::AllocationFailed => "allocation_failed",
            Self::Internal => "internal",
        }
    }

    const fn as_i32(self) -> i32 {
        self as i32
    }
}

#[derive(Debug)]
struct BridgeError {
    status: BridgeStatus,
    message: String,
}

impl BridgeError {
    fn new(status: BridgeStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy)]
struct LastError {
    bytes: [u8; MAX_ERROR_BYTES],
    length: usize,
}

impl LastError {
    const fn new() -> Self {
        Self {
            bytes: [0; MAX_ERROR_BYTES],
            length: 0,
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<LastError> = const { RefCell::new(LastError::new()) };
}

fn set_last_error(status: BridgeStatus, message: &str) {
    // Error reporting is itself inside the C boundary.  In particular, a
    // poisoned/borrowed TLS cell or an allocation panic must never escape.
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = LAST_ERROR.try_with(|cell| {
            let Ok(mut error) = cell.try_borrow_mut() else {
                return;
            };
            let mut length = 0;
            for byte in status.label().bytes().chain(b": ".iter().copied()) {
                if length + 1 >= MAX_ERROR_BYTES {
                    break;
                }
                error.bytes[length] = byte;
                length += 1;
            }
            // Keep diagnostics printable ASCII and NUL-free.  Iterating chars
            // avoids splitting UTF-8; non-ASCII and control characters are
            // intentionally replaced rather than copied from untrusted text.
            for character in message.chars() {
                if length + 1 >= MAX_ERROR_BYTES {
                    break;
                }
                let byte = character as u32;
                error.bytes[length] = if (0x20..=0x7e).contains(&byte) {
                    byte as u8
                } else {
                    b'?'
                };
                length += 1;
            }
            error.bytes[length] = 0;
            error.length = length;
        });
    }));
}

fn clear_last_error() {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = LAST_ERROR.try_with(|cell| {
            if let Ok(mut error) = cell.try_borrow_mut() {
                error.bytes[0] = 0;
                error.length = 0;
            }
        });
    }));
}

fn last_error_snapshot() -> ([u8; MAX_ERROR_BYTES], usize) {
    LAST_ERROR
        .try_with(|cell| {
            cell.try_borrow()
                .map(|error| (error.bytes, error.length))
                .ok()
        })
        .ok()
        .flatten()
        .unwrap_or(([0; MAX_ERROR_BYTES], 0))
}

fn report(error: BridgeError) -> BridgeStatus {
    set_last_error(error.status, &error.message);
    error.status
}

fn ffi_boundary(function: impl FnOnce() -> Result<BridgeStatus, BridgeError>) -> i32 {
    // Keep operation, status mapping/reporting, and TLS recording in one
    // containment envelope.  This is intentionally defensive around OOM,
    // poisoned TLS, and any future status conversion code.
    let result = catch_unwind(AssertUnwindSafe(function));
    match catch_unwind(AssertUnwindSafe(|| match result {
        Ok(Ok(status)) => status.as_i32(),
        Ok(Err(error)) => report(error).as_i32(),
        Err(_) => {
            let status = BridgeStatus::Panic;
            set_last_error(status, "panic crossed the bridge boundary");
            status.as_i32()
        }
    })) {
        Ok(status) => status,
        Err(_) => {
            set_last_error(
                BridgeStatus::Panic,
                "panic while recording bridge diagnostic",
            );
            BridgeStatus::Panic.as_i32()
        }
    }
}

struct Admission {
    live_entries: AtomicUsize,
    packed_bytes: AtomicUsize,
}

impl Admission {
    fn reserve(self: &Arc<Self>, packed_bytes: usize) -> Result<AdmissionLease, BridgeError> {
        if packed_bytes > MAX_PACKED_WEIGHT_BYTES {
            return Err(BridgeError::new(
                BridgeStatus::LimitViolation,
                "one packed weight exceeds the per-pack byte limit",
            ));
        }
        let entries = self.live_entries.load(Ordering::Relaxed);
        let bytes = self.packed_bytes.load(Ordering::Relaxed);
        if entries >= MAX_LIVE_ENTRIES
            || bytes
                .checked_add(packed_bytes)
                .is_none_or(|total| total > MAX_AGGREGATE_PACKED_WEIGHT_BYTES)
        {
            return Err(BridgeError::new(
                BridgeStatus::LimitViolation,
                "bridge live-entry or aggregate packed-byte limit exceeded",
            ));
        }
        self.live_entries.fetch_add(1, Ordering::Relaxed);
        self.packed_bytes.fetch_add(packed_bytes, Ordering::Relaxed);
        Ok(AdmissionLease {
            admission: Arc::clone(self),
            packed_bytes,
        })
    }

    fn release(&self, packed_bytes: usize) {
        self.live_entries.fetch_sub(1, Ordering::Relaxed);
        self.packed_bytes.fetch_sub(packed_bytes, Ordering::Relaxed);
    }
}

/// One admission's accounting ownership, released exactly once at last drop.
///
/// An executable entry stores this lease rather than a bare accounting
/// pointer.  A run/query `Arc<Entry>` clone therefore keeps both the live
/// entry and its packed bytes accounted after registry removal.
struct AdmissionLease {
    admission: Arc<Admission>,
    packed_bytes: usize,
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        self.admission.release(self.packed_bytes);
    }
}

struct Entry {
    executable: decodeforge_runtime::GeneratedExecutableV1,
    _admission: AdmissionLease,
}

struct Registry {
    entries: HashMap<u64, Arc<Entry>>,
    admission: Arc<Admission>,
    building: AtomicBool,
}

impl Registry {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            admission: Arc::new(Admission {
                live_entries: AtomicUsize::new(0),
                packed_bytes: AtomicUsize::new(0),
            }),
            building: AtomicBool::new(false),
        }
    }

    fn random_handle(&self) -> Result<u64, BridgeError> {
        let mut bytes = [0_u8; size_of::<u64>()];
        for _ in 0..128 {
            getrandom::fill(&mut bytes).map_err(|_| {
                BridgeError::new(BridgeStatus::Internal, "secure handle generation failed")
            })?;
            let handle = u64::from_ne_bytes(bytes);
            if handle != 0 && !self.entries.contains_key(&handle) {
                return Ok(handle);
            }
        }
        Err(BridgeError::new(
            BridgeStatus::Internal,
            "unable to generate a collision-free handle",
        ))
    }
}

struct CreateReservation {
    admission: Option<AdmissionLease>,
    registry: &'static Mutex<Registry>,
    active: bool,
}

impl CreateReservation {
    fn transfer_accounting(&mut self) -> AdmissionLease {
        self.admission
            .take()
            .expect("create reservation accounting was already transferred")
    }

    fn commit(mut self) {
        self.active = false;
        if let Ok(guard) = self.registry.lock() {
            guard.building.store(false, Ordering::Release);
        }
    }
}

impl Drop for CreateReservation {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        if let Ok(guard) = self.registry.lock() {
            guard.building.store(false, Ordering::Release);
        }
    }
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry::new()))
}

fn lock_registry() -> Result<MutexGuard<'static, Registry>, BridgeError> {
    registry().lock().map_err(|_| {
        BridgeError::new(
            BridgeStatus::Internal,
            "bridge executable registry lock is poisoned",
        )
    })
}

fn reserve_create(packed_bytes: usize) -> Result<CreateReservation, BridgeError> {
    let guard = lock_registry()?;
    if guard.building.swap(true, Ordering::AcqRel) {
        return Err(BridgeError::new(
            BridgeStatus::LimitViolation,
            "only one bridge create/build may run at a time",
        ));
    }
    let admission = match guard.admission.reserve(packed_bytes) {
        Ok(admission) => admission,
        Err(error) => {
            guard.building.store(false, Ordering::Release);
            return Err(error);
        }
    };
    drop(guard);
    Ok(CreateReservation {
        admission: Some(admission),
        registry: registry(),
        active: true,
    })
}

fn pointer_range<T>(
    pointer: *const T,
    count: usize,
    maximum_bytes: usize,
    field: &'static str,
) -> Result<Range<usize>, BridgeError> {
    if pointer.is_null() {
        return Err(BridgeError::new(
            BridgeStatus::NullArgument,
            format!("{field} pointer is null"),
        ));
    }
    let byte_count = count.checked_mul(size_of::<T>()).ok_or_else(|| {
        BridgeError::new(
            BridgeStatus::LimitViolation,
            format!("{field} byte length overflows"),
        )
    })?;
    if byte_count == 0 {
        return Err(BridgeError::new(
            BridgeStatus::ZeroLength,
            format!("{field} length is zero"),
        ));
    }
    if byte_count > maximum_bytes {
        return Err(BridgeError::new(
            BridgeStatus::LimitViolation,
            format!("{field} exceeds the bridge input bound"),
        ));
    }
    let address = pointer as usize;
    if !address.is_multiple_of(align_of::<T>()) {
        return Err(BridgeError::new(
            BridgeStatus::InvalidArgument,
            format!("{field} pointer is misaligned"),
        ));
    }
    let end = address.checked_add(byte_count).ok_or_else(|| {
        BridgeError::new(
            BridgeStatus::InvalidArgument,
            format!("{field} address range overflows"),
        )
    })?;
    Ok(address..end)
}

fn byte_range(
    pointer: *const u8,
    length: usize,
    maximum_bytes: usize,
    field: &'static str,
) -> Result<Range<usize>, BridgeError> {
    pointer_range(pointer, length, maximum_bytes, field)
}

fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn copy_bytes(pointer: *const u8, length: usize) -> Result<Vec<u8>, BridgeError> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| {
        BridgeError::new(
            BridgeStatus::AllocationFailed,
            "unable to copy packed-weight payload",
        )
    })?;
    // SAFETY: The caller validated pointer non-null, alignment, and the exact
    // bounded byte range before invoking this helper.
    unsafe {
        bytes.set_len(length);
        ptr::copy_nonoverlapping(pointer, bytes.as_mut_ptr(), length);
    }
    Ok(bytes)
}

fn parse_manifest(pointer: *const u8, length: usize) -> Result<PackManifestV1, BridgeError> {
    // SAFETY: The caller validated the bounded non-null input range.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    let manifest = from_slice::<PackManifestV1>(bytes).map_err(|error| {
        BridgeError::new(
            BridgeStatus::InvalidManifest,
            format!("manifest is not valid PackManifestV1 JSON: {error}"),
        )
    })?;
    let canonical = manifest.canonical_json().map_err(|error| {
        BridgeError::new(
            BridgeStatus::InvalidManifest,
            format!("manifest verification failed: {error}"),
        )
    })?;
    if canonical.as_bytes() != bytes {
        return Err(BridgeError::new(
            BridgeStatus::InvalidManifest,
            "manifest JSON is not the exact compact canonical encoding",
        ));
    }
    Ok(manifest)
}

fn compiler_status(error: &CompilerError, fallback: BridgeStatus) -> BridgeStatus {
    match error.code() {
        "DFE-NATIVE-001" => BridgeStatus::UnsupportedHost,
        "DFE-COMP-002" => BridgeStatus::AllocationFailed,
        "DFE-NATIVE-008" => BridgeStatus::AllocationFailed,
        code if code.starts_with("DFE-NATIVE-007") => BridgeStatus::LoadFailed,
        code if code.starts_with("DFE-NATIVE-") => fallback,
        _ => fallback,
    }
}

fn compiler_error(error: CompilerError, fallback: BridgeStatus) -> BridgeError {
    let status = compiler_status(&error, fallback);
    BridgeError::new(status, error.to_string())
}

fn create_neon_impl(
    manifest_pointer: *const u8,
    manifest_bytes: usize,
    payload_pointer: *const u8,
    payload_bytes: usize,
    out_handle: *mut u64,
) -> Result<BridgeStatus, BridgeError> {
    let manifest_range = byte_range(
        manifest_pointer,
        manifest_bytes,
        MAX_MANIFEST_BYTES,
        "manifest",
    )?;
    let payload_range = byte_range(
        payload_pointer,
        payload_bytes,
        MAX_PACKED_WEIGHT_BYTES,
        "packed payload",
    )?;
    let handle_range = pointer_range(out_handle.cast_const(), 1, size_of::<u64>(), "handle")?;
    if ranges_overlap(&manifest_range, &payload_range)
        || ranges_overlap(&manifest_range, &handle_range)
        || ranges_overlap(&payload_range, &handle_range)
    {
        return Err(BridgeError::new(
            BridgeStatus::Overlap,
            "manifest, payload, and output handle ranges overlap",
        ));
    }
    // SAFETY: The output pointer has a checked aligned range and does not
    // overlap either borrowed input.
    unsafe {
        *out_handle = 0;
    }

    let manifest = parse_manifest(manifest_pointer, manifest_bytes)?;
    let expected_payload = manifest
        .shape()
        .payload_bytes()
        .map_err(|error| compiler_error(error, BridgeStatus::InvalidManifest))?;
    if expected_payload > MAX_PACKED_WEIGHT_BYTES {
        return Err(BridgeError::new(
            BridgeStatus::LimitViolation,
            "manifest requires a payload larger than the bridge bound",
        ));
    }
    if payload_bytes != expected_payload {
        return Err(BridgeError::new(
            BridgeStatus::InvalidPayload,
            format!(
                "payload length {payload_bytes} does not equal manifest length {expected_payload}"
            ),
        ));
    }
    // Admission is taken before copying caller-owned bytes or doing any
    // compiler/build work.  The reservation is released automatically by
    // `CreateReservation` on every failed path.
    let mut reservation = reserve_create(payload_bytes)?;
    let payload = copy_bytes(payload_pointer, payload_bytes)?;
    let packed = PackedWeightsV1::from_artifact_parts(manifest, payload)
        .map_err(|error| compiler_error(error, BridgeStatus::InvalidPayload))?;
    let region = Q8LinearRegion::new(packed.shape(), packed.logical_weight_identity().to_owned())
        .map_err(|error| compiler_error(error, BridgeStatus::InvalidManifest))?;
    let kernel = lower_q8_linear(&region, KernelVariant::Neon)
        .map_err(|error| compiler_error(error, BridgeStatus::InvalidManifest))?;
    let module = emit_neon_c(&region, &kernel, &packed)
        .map_err(|error| compiler_error(error, BridgeStatus::BuildFailed))?;
    let artifact = build_apple_neon_dylib(&module)
        .map_err(|error| compiler_error(error, BridgeStatus::BuildFailed))?;
    let executable = load_apple_neon_v1(artifact, &region, &kernel, &packed)
        .map_err(|error| compiler_error(error, BridgeStatus::LoadFailed))?;

    // Construct the Arc before reacquiring the registry lock.  If this can
    // fail, its Drop releases the accounting; ownership is then transferred
    // from the reservation exactly once.
    let entry = Arc::new(Entry {
        executable,
        _admission: reservation.transfer_accounting(),
    });
    let mut guard = lock_registry()?;
    guard.entries.try_reserve(1).map_err(|_| {
        BridgeError::new(
            BridgeStatus::AllocationFailed,
            "unable to reserve bridge handle storage",
        )
    })?;
    let handle = guard.random_handle()?;
    guard.entries.insert(handle, entry);
    // The Arc now owns the reservation's accounting; destruction decrements
    // aggregate bytes only when the last concurrent run/query handle drops.
    drop(guard);
    reservation.commit();
    // SAFETY: The output range remains valid for the duration of this call by
    // the foreign ABI contract, and no input can alias it.
    unsafe {
        *out_handle = handle;
    }
    clear_last_error();
    Ok(BridgeStatus::Ok)
}

fn scrub_output(output: &mut [f32]) {
    output.fill(f32::from_bits(QUIET_NAN_BITS));
}

fn map_runtime_error(error: GeneratedRuntimeError) -> BridgeError {
    // The runtime intentionally groups allocation and extent errors under one
    // diagnostic code, so bridge status must be selected from the concrete
    // variant rather than from `error.code()`.
    let status = match &error {
        GeneratedRuntimeError::KernelStatus(GeneratedStatusV1::NonFiniteInput) => {
            BridgeStatus::NonfiniteInput
        }
        GeneratedRuntimeError::KernelStatus(GeneratedStatusV1::NonFiniteResult)
        | GeneratedRuntimeError::InvalidSuccessOutput { .. } => BridgeStatus::NonfiniteOutput,
        GeneratedRuntimeError::InputLength { .. } | GeneratedRuntimeError::OutputLength { .. } => {
            BridgeStatus::InvalidArgument
        }
        GeneratedRuntimeError::AllocationFailed { .. } => BridgeStatus::AllocationFailed,
        _ => BridgeStatus::ExecutionFailed,
    };
    BridgeError::new(status, error.to_string())
}

fn run_impl(
    handle: u64,
    input_pointer: *const f32,
    input_length: usize,
    output_pointer: *mut f32,
    output_length: usize,
) -> Result<BridgeStatus, BridgeError> {
    if handle == 0 {
        return Err(BridgeError::new(
            BridgeStatus::InvalidHandle,
            "handle zero is invalid",
        ));
    }
    // Linearization: cloning the Arc while holding the registry lock makes a
    // run observe either the entry before destroy or INVALID_HANDLE after it.
    // Invocation itself is deliberately outside the lock, so destroy may
    // remove the map entry while an already-admitted run safely finishes.
    let entry = {
        let guard = lock_registry()?;
        Arc::clone(guard.entries.get(&handle).ok_or_else(|| {
            BridgeError::new(
                BridgeStatus::InvalidHandle,
                "handle is not present in the bridge registry",
            )
        })?)
    };
    let expected_input = usize::try_from(entry.executable.k())
        .map_err(|_| BridgeError::new(BridgeStatus::Internal, "compiled K is not representable"))?;
    let expected_output = usize::try_from(entry.executable.n())
        .map_err(|_| BridgeError::new(BridgeStatus::Internal, "compiled N is not representable"))?;

    let input_range = pointer_range(input_pointer, input_length, MAX_VECTOR_BYTES, "input")?;
    let output_range = pointer_range(
        output_pointer.cast_const(),
        output_length,
        MAX_VECTOR_BYTES,
        "output",
    )?;
    if ranges_overlap(&input_range, &output_range) {
        return Err(BridgeError::new(
            BridgeStatus::Overlap,
            "input and output ranges overlap",
        ));
    }
    if input_length != expected_input || output_length != expected_output {
        return Err(BridgeError::new(
            BridgeStatus::InvalidArgument,
            format!(
                "call lengths are input={input_length}, output={output_length}; expected K={expected_input}, N={expected_output}"
            ),
        ));
    }
    // SAFETY: Both ranges have checked non-null aligned pointers and bounded
    // lengths, and the ranges are disjoint.  Extents were validated above;
    // malformed calls leave output untouched.
    let output = unsafe { std::slice::from_raw_parts_mut(output_pointer, output_length) };
    // SAFETY: The input range is checked and disjoint from output.
    let input = unsafe { std::slice::from_raw_parts(input_pointer, input_length) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut prepared = entry
            .executable
            .prepare_call(input, output)
            .map_err(map_runtime_error)?;
        prepared.invoke().map_err(map_runtime_error)?;
        Ok::<(), BridgeError>(())
    }));
    match result {
        Ok(Ok(())) => {
            clear_last_error();
            Ok(BridgeStatus::Ok)
        }
        Ok(Err(error)) => Err(error),
        Err(_) => {
            scrub_output(output);
            Err(BridgeError::new(
                BridgeStatus::Panic,
                "panic was contained while invoking generated code",
            ))
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeDescriptorV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub n: u32,
    pub k: u32,
    pub packed_weight_bytes: u64,
    pub module_id: [u8; IDENTITY_CSTR_BYTES],
    pub packed_weight_id: [u8; IDENTITY_CSTR_BYTES],
}

impl Default for RuntimeDescriptorV1 {
    fn default() -> Self {
        Self {
            abi_version: 0,
            struct_size: 0,
            n: 0,
            k: 0,
            packed_weight_bytes: 0,
            module_id: [0; IDENTITY_CSTR_BYTES],
            packed_weight_id: [0; IDENTITY_CSTR_BYTES],
        }
    }
}

const _: () = assert!(size_of::<RuntimeDescriptorV1>() == 168);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, abi_version) == 0);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, struct_size) == 4);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, n) == 8);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, k) == 12);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, packed_weight_bytes) == 16);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, module_id) == 24);
const _: () = assert!(offset_of!(RuntimeDescriptorV1, packed_weight_id) == 96);

fn copy_identity(
    destination: &mut [u8; IDENTITY_CSTR_BYTES],
    identity: &str,
) -> Result<(), BridgeError> {
    if identity.len() != IDENTITY_CSTR_BYTES - 1
        || !identity.starts_with("sha256:")
        || !identity.as_bytes()[7..]
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(BridgeError::new(
            BridgeStatus::Internal,
            "executable identity is not a canonical SHA-256 string",
        ));
    }
    destination[..IDENTITY_CSTR_BYTES - 1].copy_from_slice(identity.as_bytes());
    destination[IDENTITY_CSTR_BYTES - 1] = 0;
    Ok(())
}

fn descriptor_impl(
    handle: u64,
    output_pointer: *mut RuntimeDescriptorV1,
) -> Result<BridgeStatus, BridgeError> {
    if handle == 0 {
        // Preserve INVALID_HANDLE even when the caller supplied no output;
        // when a valid output pointer is supplied, honor the zero-on-failure
        // contract without dereferencing an invalid foreign pointer.
        if pointer_range(
            output_pointer.cast_const(),
            1,
            size_of::<RuntimeDescriptorV1>(),
            "descriptor",
        )
        .is_ok()
        {
            // SAFETY: The range check above proves this fixed descriptor write
            // is aligned, non-null, and within the caller's writable object.
            unsafe { *output_pointer = RuntimeDescriptorV1::default() };
        }
        return Err(BridgeError::new(
            BridgeStatus::InvalidHandle,
            "handle zero is invalid",
        ));
    }
    let _output_range = pointer_range(
        output_pointer.cast_const(),
        1,
        size_of::<RuntimeDescriptorV1>(),
        "descriptor",
    )?;
    // A valid descriptor pointer is cleared for every failed query, including
    // INVALID_HANDLE.  Build the replacement locally so callers never see a
    // partially written descriptor.
    // SAFETY: The fixed output range was validated above.
    unsafe { *output_pointer = RuntimeDescriptorV1::default() };
    let entry = {
        let guard = lock_registry()?;
        Arc::clone(guard.entries.get(&handle).ok_or_else(|| {
            BridgeError::new(
                BridgeStatus::InvalidHandle,
                "handle is not present in the bridge registry",
            )
        })?)
    };
    let mut descriptor = RuntimeDescriptorV1 {
        abi_version: BRIDGE_ABI_VERSION,
        struct_size: size_of::<RuntimeDescriptorV1>() as u32,
        n: entry.executable.n(),
        k: entry.executable.k(),
        packed_weight_bytes: entry
            .executable
            .n()
            .div_ceil(4)
            .checked_mul(entry.executable.k().div_ceil(32))
            .and_then(|records| records.checked_mul(144))
            .ok_or_else(|| {
                BridgeError::new(
                    BridgeStatus::Internal,
                    "compiled packed-weight size overflows",
                )
            })? as u64,
        module_id: [0; IDENTITY_CSTR_BYTES],
        packed_weight_id: [0; IDENTITY_CSTR_BYTES],
    };
    copy_identity(&mut descriptor.module_id, entry.executable.module_id())?;
    copy_identity(
        &mut descriptor.packed_weight_id,
        entry.executable.packed_identity(),
    )?;
    // SAFETY: The fixed output range was validated above and remains borrowed
    // exclusively for this call.
    unsafe { *output_pointer = descriptor };
    clear_last_error();
    Ok(BridgeStatus::Ok)
}

fn destroy_impl(handle: u64) -> Result<BridgeStatus, BridgeError> {
    if handle == 0 {
        return Err(BridgeError::new(
            BridgeStatus::InvalidHandle,
            "handle zero is invalid",
        ));
    }
    // Linearization: removal under the lock makes future lookups fail.  The
    // Arc is dropped after unlocking, so an already-cloned run/query owner can
    // finish while destruction releases its accounting safely.
    let removed = {
        let mut guard = lock_registry()?;
        guard.entries.remove(&handle)
    };
    if removed.is_none() {
        return Err(BridgeError::new(
            BridgeStatus::InvalidHandle,
            "handle is not present in the bridge registry",
        ));
    }
    drop(removed);
    clear_last_error();
    Ok(BridgeStatus::Ok)
}

fn last_error_impl(
    buffer_pointer: *mut u8,
    buffer_length: usize,
    required_pointer: *mut usize,
) -> Result<BridgeStatus, BridgeError> {
    let required_range = pointer_range(
        required_pointer.cast_const(),
        1,
        size_of::<usize>(),
        "required-bytes",
    )?;
    let buffer_range = if buffer_pointer.is_null() && buffer_length == 0 {
        None
    } else {
        Some(byte_range(
            buffer_pointer.cast_const(),
            buffer_length,
            MAX_ERROR_BYTES,
            "error buffer",
        )?)
    };
    if let Some(buffer_range) = &buffer_range
        && ranges_overlap(buffer_range, &required_range)
    {
        return Err(BridgeError::new(
            BridgeStatus::Overlap,
            "error buffer overlaps required-bytes output",
        ));
    }
    let (bytes, length) = last_error_snapshot();
    let required = length + 1;
    // SAFETY: required_pointer was validated and does not overlap the optional
    // error buffer.
    unsafe {
        *required_pointer = required;
    }
    if buffer_pointer.is_null() && buffer_length == 0 {
        return Ok(BridgeStatus::Ok);
    }
    if buffer_length == 0 {
        return Err(BridgeError::new(
            BridgeStatus::ZeroLength,
            "non-null error buffer has zero capacity",
        ));
    }
    // SAFETY: buffer_pointer was validated and does not overlap the required
    // output; the copy is bounded by the validated capacity.
    let destination = unsafe { std::slice::from_raw_parts_mut(buffer_pointer, buffer_length) };
    let copied = length.min(buffer_length - 1);
    destination[..copied].copy_from_slice(&bytes[..copied]);
    destination[copied] = 0;
    if copied != length {
        Ok(BridgeStatus::Truncated)
    } else {
        Ok(BridgeStatus::Ok)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn df_runtime_bridge_abi_version_v1() -> u32 {
    match catch_unwind(|| BRIDGE_ABI_VERSION) {
        Ok(version) => version,
        Err(_) => BRIDGE_ABI_VERSION,
    }
}

#[unsafe(no_mangle)]
/// Create a verified generated NEON executable and return its process-local handle.
///
/// # Safety
/// `manifest_pointer` and `payload_pointer` must point to readable buffers of
/// exactly the supplied lengths for the duration of this call. `out_handle`
/// must be non-null, writable for one `u64`, and aligned; none of the three
/// ranges may overlap.
pub unsafe extern "C" fn df_runtime_create_neon_v1(
    manifest_pointer: *const u8,
    manifest_bytes: usize,
    payload_pointer: *const u8,
    payload_bytes: usize,
    out_handle: *mut u64,
) -> i32 {
    ffi_boundary(|| {
        create_neon_impl(
            manifest_pointer,
            manifest_bytes,
            payload_pointer,
            payload_bytes,
            out_handle,
        )
    })
}

#[unsafe(no_mangle)]
/// Run one exact generated call for a live handle.
///
/// # Safety
/// `input_pointer` must point to `input_length` readable, aligned `f32`s and
/// `output_pointer` to `output_length` writable, aligned `f32`s for the
/// duration of this call. The ranges must not overlap. A handle is process
/// local and must have been returned by create and not destroyed; concurrent
/// run/query calls are permitted, as is destroy (which linearizes separately).
pub unsafe extern "C" fn df_runtime_run_v1(
    handle: u64,
    input_pointer: *const f32,
    input_length: usize,
    output_pointer: *mut f32,
    output_length: usize,
) -> i32 {
    ffi_boundary(|| {
        run_impl(
            handle,
            input_pointer,
            input_length,
            output_pointer,
            output_length,
        )
    })
}

#[unsafe(no_mangle)]
/// Query shape and identities for a live handle.
///
/// # Safety
/// `output_pointer` must be non-null, aligned, and writable for exactly one
/// `RuntimeDescriptorV1`. On any failure after pointer validation it is set to
/// an all-zero descriptor; its contents are usable only when the return value
/// is `DF_RUNTIME_STATUS_OK_V1`.
pub unsafe extern "C" fn df_runtime_get_descriptor_v1(
    handle: u64,
    output_pointer: *mut RuntimeDescriptorV1,
) -> i32 {
    ffi_boundary(|| descriptor_impl(handle, output_pointer))
}

#[unsafe(no_mangle)]
/// Destroy one process-local executable handle.
///
/// # Safety
/// `handle` must be a value returned by create that the caller coordinates so
/// no new operation begins after destruction. Existing operations that have
/// already cloned the handle remain valid until they return.
pub unsafe extern "C" fn df_runtime_destroy_v1(handle: u64) -> i32 {
    ffi_boundary(|| destroy_impl(handle))
}

#[unsafe(no_mangle)]
/// Read the calling thread's bounded diagnostic string.
///
/// # Safety
/// `required_pointer` must be non-null, aligned, and writable for one
/// `size_t`. If `buffer_length` is nonzero, `buffer_pointer` must be non-null,
/// aligned for bytes, writable for that many bytes, and disjoint from
/// `required_pointer`; a null buffer is valid only with zero length.
pub unsafe extern "C" fn df_runtime_last_error_v1(
    buffer_pointer: *mut std::ffi::c_char,
    buffer_length: usize,
    required_pointer: *mut usize,
) -> i32 {
    ffi_boundary(|| last_error_impl(buffer_pointer.cast::<u8>(), buffer_length, required_pointer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use decodeforge_compiler::PackedWeightsV1;
    use decodeforge_core::q8::Q8Weights;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    use std::process::Command;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    const CLEAN_DYLD_CHILD: &str = "DECODEFORGE_BRIDGE_CLEAN_DYLD_CHILD";

    // These wrappers keep every unsafe FFI call in one reviewed test helper.
    // The fixture-owned slices and stack outputs satisfy each export's
    // pointer preconditions; invalid-pointer cases intentionally pass null.
    fn df_runtime_create_neon_v1(
        manifest_pointer: *const u8,
        manifest_bytes: usize,
        payload_pointer: *const u8,
        payload_bytes: usize,
        out_handle: *mut u64,
    ) -> i32 {
        // SAFETY: Callers in this module use fixture-owned buffers or the
        // explicit null/overlap cases being tested.
        unsafe {
            super::df_runtime_create_neon_v1(
                manifest_pointer,
                manifest_bytes,
                payload_pointer,
                payload_bytes,
                out_handle,
            )
        }
    }

    fn df_runtime_run_v1(
        handle: u64,
        input_pointer: *const f32,
        input_length: usize,
        output_pointer: *mut f32,
        output_length: usize,
    ) -> i32 {
        // SAFETY: Tests pass aligned fixture/stack buffers, disjoint ranges,
        // or deliberate null/overlap inputs to exercise validation.
        unsafe {
            super::df_runtime_run_v1(
                handle,
                input_pointer,
                input_length,
                output_pointer,
                output_length,
            )
        }
    }

    fn df_runtime_get_descriptor_v1(handle: u64, output_pointer: *mut RuntimeDescriptorV1) -> i32 {
        // SAFETY: The descriptor test output is a valid aligned stack object;
        // null is passed only for invalid-handle short-circuit coverage.
        unsafe { super::df_runtime_get_descriptor_v1(handle, output_pointer) }
    }

    fn df_runtime_destroy_v1(handle: u64) -> i32 {
        // SAFETY: Handles are plain process-local values and no pointer is
        // involved; tests coordinate their fixture ownership.
        unsafe { super::df_runtime_destroy_v1(handle) }
    }

    fn df_runtime_last_error_v1(
        buffer_pointer: *mut std::ffi::c_char,
        buffer_length: usize,
        required_pointer: *mut usize,
    ) -> i32 {
        // SAFETY: Error buffers and required outputs are fixture-owned and
        // disjoint, with null used only for the documented length query.
        unsafe { super::df_runtime_last_error_v1(buffer_pointer, buffer_length, required_pointer) }
    }

    fn fixture() -> (String, Vec<u8>) {
        let weights = Q8Weights::try_new(4, 32, 1, vec![1_u8; 4 * 32], vec![1.0_f32.to_bits(); 4])
            .expect("frozen bridge fixture is valid");
        let packed = PackedWeightsV1::pack(&weights).expect("fixture packs");
        (
            packed
                .canonical_manifest_json()
                .expect("fixture manifest serializes"),
            packed.bytes().to_vec(),
        )
    }

    fn status(value: BridgeStatus) -> i32 {
        value.as_i32()
    }

    fn admission() -> Arc<Admission> {
        Arc::new(Admission {
            live_entries: AtomicUsize::new(0),
            packed_bytes: AtomicUsize::new(0),
        })
    }

    fn admission_counts(admission: &Admission) -> (usize, usize) {
        (
            admission.live_entries.load(Ordering::Relaxed),
            admission.packed_bytes.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn admission_enforces_per_pack_limit() {
        let admission = admission();
        let lease = admission
            .reserve(MAX_PACKED_WEIGHT_BYTES)
            .expect("one maximum-size pack is admitted");
        assert_eq!(admission_counts(&admission), (1, MAX_PACKED_WEIGHT_BYTES));
        drop(lease);
        assert_eq!(admission_counts(&admission), (0, 0));
        let error = admission
            .reserve(MAX_PACKED_WEIGHT_BYTES + 1)
            .err()
            .expect("a pack above the per-pack limit must be rejected");
        assert_eq!(error.status, BridgeStatus::LimitViolation);
        assert_eq!(admission_counts(&admission), (0, 0));
    }

    #[test]
    fn admission_enforces_aggregate_byte_limit() {
        let admission = admission();
        let mut leases = Vec::new();
        for _ in 0..(MAX_AGGREGATE_PACKED_WEIGHT_BYTES / MAX_PACKED_WEIGHT_BYTES) {
            leases.push(
                admission
                    .reserve(MAX_PACKED_WEIGHT_BYTES)
                    .expect("aggregate limit admits exactly sixteen packs"),
            );
        }
        assert_eq!(
            admission_counts(&admission),
            (16, MAX_AGGREGATE_PACKED_WEIGHT_BYTES)
        );
        let error = admission
            .reserve(1)
            .err()
            .expect("one byte above the aggregate limit must be rejected");
        assert_eq!(error.status, BridgeStatus::LimitViolation);
        drop(leases);
        assert_eq!(admission_counts(&admission), (0, 0));
    }

    #[test]
    fn admission_enforces_live_entry_limit() {
        let admission = admission();
        let mut leases = Vec::new();
        for _ in 0..MAX_LIVE_ENTRIES {
            leases.push(admission.reserve(1).expect("live-entry limit admits 256"));
        }
        assert_eq!(admission_counts(&admission), (MAX_LIVE_ENTRIES, 256));
        let error = admission
            .reserve(1)
            .err()
            .expect("the 257th live entry must be rejected");
        assert_eq!(error.status, BridgeStatus::LimitViolation);
        drop(leases);
        assert_eq!(admission_counts(&admission), (0, 0));
    }

    #[test]
    fn failed_reservation_and_lease_drop_release_no_accounting() {
        let admission = admission();
        assert!(admission.reserve(MAX_PACKED_WEIGHT_BYTES + 1).is_err());
        assert_eq!(admission_counts(&admission), (0, 0));
        let lease = admission.reserve(123).expect("small pack is admitted");
        assert_eq!(admission_counts(&admission), (1, 123));
        drop(lease);
        assert_eq!(admission_counts(&admission), (0, 0));
    }

    #[test]
    fn admission_lease_survives_registry_style_arc_removal() {
        let admission = admission();
        let lease = admission.reserve(123).expect("pack is admitted");
        let registry_entry = Arc::new(lease);
        let in_flight_entry = Arc::clone(&registry_entry);
        drop(registry_entry);
        assert_eq!(admission_counts(&admission), (1, 123));
        drop(in_flight_entry);
        assert_eq!(admission_counts(&admission), (0, 0));
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn rerun_without_dyld_overrides(test_filter: &str) -> bool {
        if std::env::var_os(CLEAN_DYLD_CHILD).is_some() {
            return false;
        }
        let keys = std::env::vars_os()
            .map(|(key, _)| key)
            .filter(|key| key.to_string_lossy().starts_with("DYLD_"))
            .collect::<Vec<_>>();
        if keys.is_empty() {
            return false;
        }
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .arg(test_filter)
            .arg("--nocapture")
            .env(CLEAN_DYLD_CHILD, "1");
        for key in keys {
            command.env_remove(key);
        }
        assert!(command.status().expect("clean child status").success());
        true
    }

    fn read_last_error() -> String {
        let mut required = 0_usize;
        assert_eq!(
            df_runtime_last_error_v1(ptr::null_mut(), 0, &mut required),
            status(BridgeStatus::Ok)
        );
        assert!(required > 0);
        let mut bytes = vec![0_u8; required];
        assert_eq!(
            df_runtime_last_error_v1(bytes.as_mut_ptr().cast(), bytes.len(), &mut required,),
            status(BridgeStatus::Ok)
        );
        assert_eq!(bytes.last(), Some(&0));
        String::from_utf8(bytes[..bytes.len() - 1].to_vec()).expect("error is UTF-8")
    }

    #[test]
    fn bridge_status_and_descriptor_layout_are_frozen() {
        assert_eq!(df_runtime_bridge_abi_version_v1(), BRIDGE_ABI_VERSION);
        assert_eq!(size_of::<RuntimeDescriptorV1>(), 168);
        assert_eq!(offset_of!(RuntimeDescriptorV1, module_id), 24);
        assert_eq!(offset_of!(RuntimeDescriptorV1, packed_weight_id), 96);
        assert_eq!(status(BridgeStatus::Ok), 0);
        assert_eq!(status(BridgeStatus::Panic), 16);
    }

    #[test]
    fn runtime_extent_and_allocation_variants_map_to_distinct_bridge_statuses() {
        for (error, expected) in [
            (
                GeneratedRuntimeError::InputLength {
                    expected: 8,
                    actual: 7,
                },
                BridgeStatus::InvalidArgument,
            ),
            (
                GeneratedRuntimeError::OutputLength {
                    expected: 4,
                    actual: 3,
                },
                BridgeStatus::InvalidArgument,
            ),
            (
                GeneratedRuntimeError::AllocationFailed { object: "output" },
                BridgeStatus::AllocationFailed,
            ),
        ] {
            assert_eq!(map_runtime_error(error).status, expected);
        }
    }

    #[test]
    fn handles_are_nonzero_and_collision_checked() {
        let registry = Registry::new();
        let first = registry.random_handle().expect("CSPRNG handle");
        assert_ne!(first, 0);
        let second = registry.random_handle().expect("CSPRNG handle");
        assert_ne!(second, 0);
        assert_ne!(first, second);
    }

    #[test]
    fn invalid_handles_never_observe_foreign_pointers() {
        assert_eq!(
            df_runtime_destroy_v1(0),
            status(BridgeStatus::InvalidHandle)
        );
        assert_eq!(
            df_runtime_destroy_v1(0xfeed_beef),
            status(BridgeStatus::InvalidHandle)
        );
        assert_eq!(
            df_runtime_run_v1(0, ptr::null(), 0, ptr::null_mut(), 0),
            status(BridgeStatus::InvalidHandle)
        );
        assert_eq!(
            df_runtime_get_descriptor_v1(0, ptr::null_mut()),
            status(BridgeStatus::InvalidHandle)
        );
        assert!(read_last_error().contains("invalid_handle"));
    }

    #[test]
    fn invalid_descriptor_clears_a_valid_output_pointer() {
        let mut descriptor = RuntimeDescriptorV1 {
            abi_version: 99,
            struct_size: 99,
            n: 99,
            k: 99,
            packed_weight_bytes: 99,
            module_id: [0xff; IDENTITY_CSTR_BYTES],
            packed_weight_id: [0xff; IDENTITY_CSTR_BYTES],
        };
        assert_eq!(
            df_runtime_get_descriptor_v1(0xfeed_beef, &mut descriptor),
            status(BridgeStatus::InvalidHandle)
        );
        assert_eq!(descriptor, RuntimeDescriptorV1::default());
    }

    #[test]
    fn create_rejects_null_zero_noncanonical_and_overlapping_inputs() {
        let (manifest, payload) = fixture();
        let mut handle = 99_u64;
        assert_eq!(
            df_runtime_create_neon_v1(
                ptr::null(),
                manifest.len(),
                payload.as_ptr(),
                payload.len(),
                &mut handle,
            ),
            status(BridgeStatus::NullArgument)
        );
        assert_eq!(handle, 99);
        assert_eq!(
            df_runtime_create_neon_v1(
                manifest.as_ptr(),
                0,
                payload.as_ptr(),
                payload.len(),
                &mut handle,
            ),
            status(BridgeStatus::ZeroLength)
        );
        let pretty = serde_json::to_vec_pretty(
            &serde_json::from_str::<serde_json::Value>(&manifest).expect("fixture JSON"),
        )
        .expect("pretty JSON");
        assert_eq!(
            df_runtime_create_neon_v1(
                pretty.as_ptr(),
                pretty.len(),
                payload.as_ptr(),
                payload.len(),
                &mut handle,
            ),
            status(BridgeStatus::InvalidManifest)
        );
        assert_eq!(handle, 0);
        assert_eq!(
            df_runtime_create_neon_v1(
                manifest.as_ptr(),
                manifest.len(),
                manifest.as_ptr(),
                manifest.len(),
                &mut handle,
            ),
            status(BridgeStatus::Overlap)
        );
        assert_eq!(handle, 0);
        assert_eq!(
            df_runtime_create_neon_v1(
                manifest.as_ptr(),
                manifest.len(),
                payload.as_ptr(),
                payload.len(),
                ptr::null_mut(),
            ),
            status(BridgeStatus::NullArgument)
        );
    }

    #[test]
    fn create_rejects_payload_length_and_identity_mutations() {
        let (manifest, mut payload) = fixture();
        let mut handle = 0_u64;
        assert_eq!(
            df_runtime_create_neon_v1(
                manifest.as_ptr(),
                manifest.len(),
                payload.as_ptr(),
                payload.len() - 1,
                &mut handle,
            ),
            status(BridgeStatus::InvalidPayload)
        );
        assert_eq!(handle, 0);
        payload[0] ^= 1;
        assert_eq!(
            df_runtime_create_neon_v1(
                manifest.as_ptr(),
                manifest.len(),
                payload.as_ptr(),
                payload.len(),
                &mut handle,
            ),
            status(BridgeStatus::InvalidPayload)
        );
        assert_eq!(handle, 0);
        let mut invalid_manifest = manifest.into_bytes();
        let identity = invalid_manifest
            .windows(b"sha256:".len())
            .position(|window| window == b"sha256:")
            .expect("manifest contains an identity");
        invalid_manifest[identity + 7] = b'Z';
        let (_, valid_payload) = fixture();
        assert_eq!(
            df_runtime_create_neon_v1(
                invalid_manifest.as_ptr(),
                invalid_manifest.len(),
                valid_payload.as_ptr(),
                valid_payload.len(),
                &mut handle,
            ),
            status(BridgeStatus::InvalidManifest)
        );
    }

    #[test]
    fn error_copy_is_nul_terminated_and_reports_truncation() {
        let mut required = 0_usize;
        assert_eq!(
            df_runtime_destroy_v1(1),
            status(BridgeStatus::InvalidHandle)
        );
        assert_eq!(
            df_runtime_last_error_v1(ptr::null_mut(), 0, &mut required),
            status(BridgeStatus::Ok)
        );
        assert!(required > 4);
        let mut short = [0xff_u8; 4];
        assert_eq!(
            df_runtime_last_error_v1(short.as_mut_ptr().cast(), short.len(), &mut required),
            status(BridgeStatus::Truncated)
        );
        assert_eq!(short[3], 0);
        assert!(!short[..3].contains(&0));
    }

    #[test]
    fn panic_is_contained_at_the_internal_boundary() {
        let result = ffi_boundary(|| -> Result<BridgeStatus, BridgeError> {
            panic!("test panic");
        });
        assert_eq!(result, status(BridgeStatus::Panic));
        assert!(read_last_error().contains("panic"));
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn valid_fixture_reports_unsupported_host_without_a_handle() {
        let (manifest, payload) = fixture();
        let mut handle = 0_u64;
        assert_eq!(
            df_runtime_create_neon_v1(
                manifest.as_ptr(),
                manifest.len(),
                payload.as_ptr(),
                payload.len(),
                &mut handle,
            ),
            status(BridgeStatus::UnsupportedHost)
        );
        assert_eq!(handle, 0);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn macos_arm64_fixture_runs_with_descriptor_and_scrubbing_policy() {
        if rerun_without_dyld_overrides(
            "macos_arm64_fixture_runs_with_descriptor_and_scrubbing_policy",
        ) {
            return;
        }
        let (manifest, payload) = fixture();
        let mut handle = 0_u64;
        let create_status = df_runtime_create_neon_v1(
            manifest.as_ptr(),
            manifest.len(),
            payload.as_ptr(),
            payload.len(),
            &mut handle,
        );
        assert_eq!(
            create_status,
            status(BridgeStatus::Ok),
            "create failed: {}",
            read_last_error()
        );
        assert_ne!(handle, 0);

        let mut descriptor = RuntimeDescriptorV1 {
            abi_version: 0,
            struct_size: 0,
            n: 0,
            k: 0,
            packed_weight_bytes: 0,
            module_id: [0xff; IDENTITY_CSTR_BYTES],
            packed_weight_id: [0xff; IDENTITY_CSTR_BYTES],
        };
        assert_eq!(
            df_runtime_get_descriptor_v1(handle, &mut descriptor),
            status(BridgeStatus::Ok)
        );
        assert_eq!(descriptor.abi_version, BRIDGE_ABI_VERSION);
        assert_eq!(
            descriptor.struct_size as usize,
            size_of::<RuntimeDescriptorV1>()
        );
        assert_eq!((descriptor.n, descriptor.k), (4, 32));
        assert_eq!(descriptor.packed_weight_bytes, 144);
        assert_eq!(descriptor.module_id[71], 0);
        assert_eq!(descriptor.packed_weight_id[71], 0);

        let input = [1.0_f32; 32];
        let mut output = [0.0_f32; 4];
        assert_eq!(
            df_runtime_run_v1(
                handle,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len()
            ),
            status(BridgeStatus::Ok)
        );
        assert_eq!(output, [32.0; 4]);

        let nonfinite_input = [f32::NAN; 32];
        output.fill(0.0);
        assert_eq!(
            df_runtime_run_v1(
                handle,
                nonfinite_input.as_ptr(),
                nonfinite_input.len(),
                output.as_mut_ptr(),
                output.len(),
            ),
            status(BridgeStatus::NonfiniteInput)
        );
        assert!(output.iter().all(|value| value.to_bits() == QUIET_NAN_BITS));

        let overflowing_input = [f32::MAX; 32];
        output.fill(0.0);
        assert_eq!(
            df_runtime_run_v1(
                handle,
                overflowing_input.as_ptr(),
                overflowing_input.len(),
                output.as_mut_ptr(),
                output.len(),
            ),
            status(BridgeStatus::NonfiniteOutput)
        );
        assert!(output.iter().all(|value| value.to_bits() == QUIET_NAN_BITS));

        assert_eq!(
            df_runtime_run_v1(
                handle,
                input.as_ptr(),
                input.len() - 1,
                output.as_mut_ptr(),
                output.len(),
            ),
            status(BridgeStatus::InvalidArgument)
        );
        assert_eq!(
            df_runtime_run_v1(
                handle,
                input.as_ptr(),
                input.len(),
                input.as_ptr().cast_mut(),
                output.len(),
            ),
            status(BridgeStatus::Overlap)
        );
        assert_eq!(df_runtime_destroy_v1(handle), status(BridgeStatus::Ok));
        assert_eq!(
            df_runtime_destroy_v1(handle),
            status(BridgeStatus::InvalidHandle)
        );
    }
}

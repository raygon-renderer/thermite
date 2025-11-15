use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU8, Ordering};

use super::InstructionSet;

const UNINITIALIZED: u8 = 0;
const INITIALIZING: u8 = 1;
const INITIALIZED: u8 = 2;

pub struct DetectInstructionSet {
    state: AtomicU8,
    isa: UnsafeCell<Option<InstructionSet>>,
}

// SAFETY: We ensure proper synchronization using atomic operations
unsafe impl Sync for DetectInstructionSet {}

impl DetectInstructionSet {
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(UNINITIALIZED),
            isa: UnsafeCell::new(None),
        }
    }

    #[inline(never)] #[rustfmt::skip]
    fn initialize(&self) -> InstructionSet {
        let previous_state = self.state.compare_exchange(UNINITIALIZED, INITIALIZING, Ordering::AcqRel, Ordering::Acquire);

        match previous_state.unwrap_or_else(|s| s) {
            INITIALIZING => while self.state.load(Ordering::Acquire) != INITIALIZED {
                core::hint::spin_loop(); // Wait for initialization to complete
            }
            UNINITIALIZED => {
                // SAFETY: We are the only thread initializing at this point
                unsafe { *self.isa.get() = Some(Self::detect_internal()) };
                self.state.store(INITIALIZED, Ordering::Release);
            }
            _ => {} // Already initialized
        }

        // SAFETY: At this point, the state is guaranteed to be INITIALIZED
        unsafe { (*self.isa.get()).unwrap_unchecked() }
    }

    #[inline(always)]
    pub fn get_or_init(&self) -> InstructionSet {
        if self.state.load(Ordering::Acquire) == INITIALIZED {
            // SAFETY: The state is INITIALIZED, so isa is guaranteed to be Some
            return unsafe { (*self.isa.get()).unwrap_unchecked() };
        }

        self.initialize()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect_internal() -> InstructionSet {
        let mut best = InstructionSet::Scalar;

        if core_detect::is_x86_feature_detected!("avx512f") {
            best = InstructionSet::X86V4; // TODO: Check if more AVX512 features are needed
        } else if core_detect::is_x86_feature_detected!("avx2") && core_detect::is_x86_feature_detected!("fma") {
            best = InstructionSet::X86V3;
        } else if core_detect::is_x86_feature_detected!("sse4.2") && core_detect::is_x86_feature_detected!("popcnt") {
            best = InstructionSet::X86V2;
        } else if core_detect::is_x86_feature_detected!("sse2") {
            best = InstructionSet::X86V1;
        }

        best
    }

    #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
    fn detect_internal() -> InstructionSet {
        InstructionSet::NEON // Assume Neon is always available on ARM with the feature enabled
    }

    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    fn detect_internal() -> InstructionSet {
        InstructionSet::WASM32 // Assume Wasm SIMD is always available on wasm32 with the feature enabled
    }

    #[cfg(not(any(
        all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")),
        all(feature = "wasm32", target_arch = "wasm32"),
        any(target_arch = "x86", target_arch = "x86_64")
    )))]
    fn detect_internal() -> InstructionSet {
        InstructionSet::Scalar // Fallback for unsupported architectures
    }
}

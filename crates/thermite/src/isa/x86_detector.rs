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
        // Hand-rolled `cpuid` (see `crate::cpu::x86`) rather than a detection
        // crate: the same module already has to speak `cpuid` for cache and
        // topology queries, so this costs nothing and drops a dependency.
        // Crucially, its AVX-class flags already fold in the `XCR0` check --
        // the OS must save YMM/ZMM state, not just the CPU implement it.
        let features = crate::cpu::x86::features();

        let mut best = InstructionSet::Scalar;

        // Which hardware earns the X86V4 rung depends on what the build did
        // with it. With an `avx512-tier*` feature, the v4 backend's
        // `#[target_feature]` trampolines assume the compiled tier's whole
        // set, so hardware below `COMPILED_TIER` must not select it (it would
        // execute encodings the CPU lacks) and falls through to the v3 arm.
        // Without a tier feature there is no v4 backend. The dispatch table
        // maps `X86V4` onto x86-v3, so reporting it on bare AVX512F is both
        // safe and informative (`InstructionSet` consumers still see the
        // wider-register, masked-ops rung).
        #[cfg(feature = "avx512-tier1")]
        let v4 = features.avx512_tier() >= Some(crate::backend::x86_v4::COMPILED_TIER);
        #[cfg(not(feature = "avx512-tier1"))]
        let v4 = features.avx512f;

        if v4 {
            best = InstructionSet::X86V4;
        } else if features.avx2 && features.fma && features.popcnt {
            // POPCNT predates AVX2 by five years (Nehalem, 2008) and is present on
            // every AVX2 CPU; checking it here lets dispatched code assume it, the
            // same way the V2 level already does.
            best = InstructionSet::X86V3;
        } else if features.sse42 && features.popcnt {
            best = InstructionSet::X86V2;
        } else if features.sse2 {
            best = InstructionSet::X86V1;
        }

        best
    }

    // No other arch variants: this module is only compiled on x86/x86_64 (see
    // `isa/mod.rs`). NEON/wasm/spirv have constant `InstructionSet::get()`
    // implementations there and never need a cached detector.
}


/// Enum of supported instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SimdInstructionSet {
    Scalar,

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE42,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX2,

    #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
    NEON,

    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    WASM32,
}

impl SimdInstructionSet {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "static_init"))]
    #[inline]
    pub fn runtime_detect() -> SimdInstructionSet {
        #[static_init::dynamic(0)]
        static SIS: SimdInstructionSet = SimdInstructionSet::runtime_detect_x86_internal();

        unsafe { *SIS }
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "static_init")))]
    pub fn runtime_detect() -> SimdInstructionSet {
        unsafe {
            static mut CACHED: Option<SimdInstructionSet> = None;

            match CACHED {
                Some(value) => value,
                None => {
                    // Allow this to race, they all converge to the same result
                    let isa = Self::runtime_detect_x86_internal();
                    CACHED = Some(isa);
                    isa
                }
            }
        }
    }

    #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
    const fn runtime_detect() -> SimdInstructionSet {
        SimdInstructionSet::NEON
    }

    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    const fn runtime_detect() -> SimdInstructionSet {
        SimdInstructionSet::WASM32
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn runtime_detect_x86_internal() -> SimdInstructionSet {
        if core_detect::is_x86_feature_detected!("fma") {
            // TODO: AVX512
            if core_detect::is_x86_feature_detected!("avx2") {
                return SimdInstructionSet::AVX2;
            }
        }

        if core_detect::is_x86_feature_detected!("avx") {
            SimdInstructionSet::AVX
        } else if core_detect::is_x86_feature_detected!("sse4.2") {
            SimdInstructionSet::SSE42
        } else if core_detect::is_x86_feature_detected!("sse2") {
            SimdInstructionSet::SSE2
        } else {
            SimdInstructionSet::Scalar
        }
    }

    /// True fused multiply-add instructions are only used on AVX2 and above, so this checks for that ergonomically.
    pub const fn has_true_fma(self) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self as u8 >= SimdInstructionSet::AVX2 as u8 {
            return true;
        }

        false
    }

    /// On older platforms, fused multiply-add instructions can be emulated (expensively),
    /// but only if the `"emulate_fma"` Cargo feature is enabled.
    pub const fn has_emulated_fma(self) -> bool {
        !self.has_true_fma() && cfg!(feature = "emulate_fma")
    }

    /// The number of general-purpose registers that can be expected to be allocated to algorithms
    pub const fn num_registers(self) -> usize {
        #[allow(unreachable_patterns)]
        match self {
            // #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SimdInstructionSet::AVX512 => 32,

            //
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdInstructionSet::Scalar => 8,

            // x86 has at least 16 registers for xmms, ymms, zmms
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            _ => 16,

            // 32x64-bit or 32x128-bit registers
            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            SimdInstructionSet::NEON => 32,

            _ => 1,
        }
    }
}

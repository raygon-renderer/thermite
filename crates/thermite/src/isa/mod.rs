/// Enum of supported instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum InstructionSet {
    /// Scalar (no SIMD)
    Scalar,

    /// x86/x86_64 SIMD instruction set level 1 (SSE2)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V1,

    /// x86/x86_64 SIMD instruction set level 2 (SSE4.1)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V2,

    /// x86/x86_64 SIMD instruction set level 3 (AVX2 + FMA)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V3,

    /// x86/x86_64 SIMD instruction set level 4 (AVX-512F)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V4,

    /// ARM Neon SIMD instruction set
    #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
    NEON,

    /// WebAssembly SIMD instruction set
    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    WASM32,
}

// WASM32 may not have atomics to support the detector
#[cfg(not(all(feature = "wasm32", target_arch = "wasm32")))]
mod detector;

impl InstructionSet {
    #[cfg(not(all(feature = "wasm32", target_arch = "wasm32")))]
    pub fn get() -> InstructionSet {
        static DETECTOR: detector::DetectInstructionSet = detector::DetectInstructionSet::new();

        DETECTOR.get_or_init()
    }

    #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
    pub fn get() -> InstructionSet {
        InstructionSet::WASM32
    }

    #[inline(always)]
    pub const fn num_registers(&self) -> usize {
        match self {
            InstructionSet::Scalar => 1, // Scalar has 1 "register"

            // x86-v1 has 8 XMM registers
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V1 => 8,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V2 | InstructionSet::X86V3 => 16,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V4 => 32,

            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            InstructionSet::NEON => 32,

            #[cfg(all(feature = "wasm32", target_arch = "wasm32"))]
            InstructionSet::WASM32 => 16,
        }
    }

    #[inline(always)]
    pub const fn has_fma(&self) -> bool {
        match self {
            InstructionSet::Scalar => false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V3 | InstructionSet::X86V4 => true,

            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            InstructionSet::NEON => true,

            _ => false,
        }
    }
}

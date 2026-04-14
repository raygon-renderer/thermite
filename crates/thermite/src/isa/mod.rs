#![allow(unexpected_cfgs)]

//! Instruction Set Architecture detection and utilities

/// Enum of supported instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum InstructionSet {
    /// Scalar (no SIMD)
    Scalar,

    /// Standard library SIMD types (e.g. std::simd::Simd) when available.
    #[cfg(feature = "std_simd")]
    StdSimd,

    /// Unknown ISA, usually the result of register emulation,
    /// such as with Glam vectors as registers.
    Unknown,

    /// x86/x86_64 SIMD instruction set level 1 (SSE2)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V1,

    /// x86/x86_64 SIMD instruction set level 2 (SSE4.2 + POPCNT)
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

    /// WebAssembly SIMD instruction set (32-bit)
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    WASM32,

    /// WebAssembly SIMD instruction set (64-bit)
    #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
    WASM64,

    /// SPIR-V (Vulkan/OpenCL compute shader)
    #[cfg(all(feature = "spirv", target_arch = "spirv"))]
    SPIRV,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_detector;

impl InstructionSet {
    /// Detect the current instruction set at runtime. This result is cached for future calls.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get() -> InstructionSet {
        static DETECTOR: x86_detector::DetectInstructionSet = x86_detector::DetectInstructionSet::new();

        DETECTOR.get_or_init()
    }

    /// Detect the current instruction set at runtime. This result is cached for future calls.
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    pub fn get() -> InstructionSet {
        InstructionSet::WASM32
    }

    /// Detect the current instruction set at runtime. This result is cached for future calls.
    #[cfg(all(feature = "spirv", target_arch = "spirv"))]
    pub fn get() -> InstructionSet {
        InstructionSet::SPIRV
    }

    /// Returns an estimate of the number of SIMD registers available
    /// for the given instruction set.
    #[inline(always)]
    pub const fn num_registers(&self) -> usize {
        match self {
            // x86-v1 has 8 XMM registers
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V1 => 8,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V2 | InstructionSet::X86V3 => 16,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V4 => 32,

            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            InstructionSet::NEON => 32,

            #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
            InstructionSet::WASM32 => 16, // TODO: Verify

            #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
            InstructionSet::WASM64 => 16, // TODO: Verify

            // SPIR-V/SIMT: each invocation is scalar; report 1
            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            InstructionSet::SPIRV => 1,

            _ => 1,
        }
    }

    /// Returns whether the given instruction set supports Fused Multiply-Add (FMA) operations.
    #[inline(always)]
    pub const fn has_fma(&self) -> bool {
        match self {
            InstructionSet::Scalar => false,

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V3 | InstructionSet::X86V4 => true,

            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            InstructionSet::NEON => true,

            // SPIR-V supports FMA via OpFma
            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            InstructionSet::SPIRV => true,

            _ => false,
        }
    }

    /// Returns whether the given instruction set is a SIMD instruction set.
    #[inline(always)]
    pub const fn is_simd(&self) -> bool {
        match self {
            InstructionSet::Scalar | InstructionSet::Unknown => false,

            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            InstructionSet::SPIRV => false, // SPIR-V is SIMT, not SIMD

            _ => true,
        }
    }

    #[inline(always)]
    pub const fn min(a: InstructionSet, b: InstructionSet) -> InstructionSet {
        if (a as u8) < (b as u8) { a } else { b }
    }

    #[inline(always)]
    pub const fn max(a: InstructionSet, b: InstructionSet) -> InstructionSet {
        if (a as u8) > (b as u8) { a } else { b }
    }

    #[inline(always)]
    pub const fn assert_eq(a: InstructionSet, b: InstructionSet) -> InstructionSet {
        assert!((a as u8) == (b as u8), "InstructionSet equality assertion failed");

        a
    }

    #[inline(always)]
    pub const fn unaligned_is_cheap(self) -> bool {
        match self {
            // Scalar loads are always cheap
            InstructionSet::Scalar => true,

            // only x86 v3+ has efficient unaligned loads/stores usually
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V1 | InstructionSet::X86V2 => false,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V3 | InstructionSet::X86V4 => true,

            // Neon generally has efficient unaligned loads/stores
            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            InstructionSet::NEON => true,

            // WASM is uncertain, so assume not cheap
            #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
            InstructionSet::WASM32 => false,
            #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
            InstructionSet::WASM64 => false,

            // SPIR-V has no concept of alignment penalties
            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            InstructionSet::SPIRV => true,

            // unknown ISA
            _ => false,
        }
    }

    pub const fn unroll_factor(self) -> usize {
        match self {
            InstructionSet::Scalar => 4,

            // only x86 v3+ has efficient unaligned loads/stores usually
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V1 | InstructionSet::X86V2 | InstructionSet::X86V3 => 4,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V4 => 8, // twice as many registers

            // Neon generally has efficient unaligned loads/stores
            #[cfg(all(feature = "neon", any(target_arch = "arm", target_arch = "aarch64")))]
            InstructionSet::NEON => 4,

            // WASM is uncertain, so assume not cheap
            #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
            InstructionSet::WASM32 => 2,
            #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
            InstructionSet::WASM64 => 2,

            // SPIR-V/GPU: each invocation is scalar; extra unrolling increases register
            // pressure and hurts occupancy, so keep it minimal
            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            InstructionSet::SPIRV => 1,

            // unknown ISA
            _ => 1,
        }
    }

    pub const fn has_masked_operations(self) -> bool {
        match self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V4 => true, // AVX-512 has masked ops

            _ => false,
        }
    }

    pub const fn has_instruction_level_parallelism(self) -> bool {
        match self {
            #[cfg(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "arm",
                target_arch = "aarch64"
            ))]
            _ => true,

            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            InstructionSet::SPIRV => false, // SPIR-V is SIMT, not SIMD

            _ => false,
        }
    }
}

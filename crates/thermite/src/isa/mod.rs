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
    #[cfg(target_arch = "aarch64")]
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
    ///
    /// NEON (AdvSIMD) is a mandatory part of AArch64, so no actual runtime
    /// detection is needed.
    #[cfg(target_arch = "aarch64")]
    pub fn get() -> InstructionSet {
        InstructionSet::NEON
    }

    /// Detect the current instruction set at runtime. This result is cached for future calls.
    ///
    /// SIMD128 is decided by the engine before the module runs, so there is
    /// nothing to detect.
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    pub fn get() -> InstructionSet {
        InstructionSet::WASM32
    }

    /// Detect the current instruction set at runtime. This result is cached for future calls.
    #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
    pub fn get() -> InstructionSet {
        InstructionSet::WASM64
    }

    /// Detect the current instruction set at runtime. This result is cached for future calls.
    #[cfg(all(feature = "spirv", target_arch = "spirv"))]
    pub fn get() -> InstructionSet {
        InstructionSet::SPIRV
    }

    /// Detect the current instruction set at runtime. This result is cached for future calls.
    ///
    /// Fallback for targets with no SIMD backend compiled in -- an unlisted
    /// architecture, or wasm without its opt-in feature. Without this the method
    /// would simply not exist on those targets, so anything calling it
    /// (including `dispatch_dyn!`) failed to compile rather than falling back to
    /// scalar.
    #[cfg(not(any(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_arch = "aarch64",
        all(feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")),
        all(feature = "spirv", target_arch = "spirv"),
    )))]
    pub fn get() -> InstructionSet {
        InstructionSet::Scalar
    }

    /// Order two sets by capability, returning the weaker.
    ///
    /// Ordering is the enum's declaration order, which ascends by capability
    /// *within* an architecture (`Scalar < X86V1 < .. < X86V4`). Across
    /// architectures it is meaningless -- but two architectures' variants never
    /// coexist, since each is `cfg`-gated to its own target.
    #[inline(always)]
    pub const fn min(a: InstructionSet, b: InstructionSet) -> InstructionSet {
        if (a as u8) < (b as u8) { a } else { b }
    }

    /// Order two sets by capability, returning the stronger. See [`min`](Self::min).
    #[inline(always)]
    pub const fn max(a: InstructionSet, b: InstructionSet) -> InstructionSet {
        if (a as u8) > (b as u8) { a } else { b }
    }

    /// Assert two sets are the same, returning it. Compares discriminants
    /// because `PartialEq` is not callable in a `const fn`.
    #[inline(always)]
    pub const fn assert_eq(a: InstructionSet, b: InstructionSet) -> InstructionSet {
        assert!((a as u8) == (b as u8), "InstructionSet equality assertion failed");

        a
    }

    /// Whether the *target* executes independent instructions in parallel, so
    /// that breaking a dependency chain into several accumulators pays off.
    ///
    /// A property of the hardware, not of the instruction set: a superscalar CPU
    /// reorders scalar code just as happily as SIMD code, so this does not vary
    /// by variant. SIMT targets (SPIR-V) hide latency with occupancy instead and
    /// gain nothing from extra accumulators.
    #[inline(always)]
    pub const fn has_instruction_level_parallelism(self) -> bool {
        cfg!(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "arm",
            target_arch = "aarch64"
        ))
    }
}

/// Per-ISA properties, one row per variant.
///
/// Written as a table because the alternative -- a separate `match` per property
/// -- repeated the same three `#[cfg]` predicates on every arm, roughly
/// `variants x properties` times, and scattered one ISA's characteristics across
/// the whole file. Here each variant carries its `cfg` once and all of its
/// properties are visible together.
///
/// The generated matches are **exhaustive**: adding a variant without a row is a
/// compile error rather than silently inheriting a `_ => ..` default.
macro_rules! isa_properties {
    ($(
        $(#[cfg $cfg:tt])?
        $variant:ident {
            registers: $registers:expr,
            fma: $fma:expr,
            simd: $simd:expr,
            unaligned_cheap: $unaligned:expr,
            unroll: $unroll:expr,
            masked: $masked:expr,
        }
    )*) => {
        impl InstructionSet {
            /// Estimate of how many SIMD registers the ISA exposes. Used by
            /// inlining/unrolling heuristics; see also [`NativeIsa::Registers`],
            /// the type-level equivalent.
            ///
            /// [`NativeIsa::Registers`]: crate::simd::NativeIsa::Registers
            #[inline(always)]
            pub const fn num_registers(self) -> usize {
                match self { $( $(#[cfg $cfg])? Self::$variant => $registers, )* }
            }

            /// Whether the ISA has a true fused multiply-add (one rounding).
            #[inline(always)]
            pub const fn has_fma(self) -> bool {
                match self { $( $(#[cfg $cfg])? Self::$variant => $fma, )* }
            }

            /// Whether the ISA is actually SIMD. False for `Scalar`, `Unknown`,
            /// and SPIR-V (which is SIMT: one lane per invocation).
            #[inline(always)]
            pub const fn is_simd(self) -> bool {
                match self { $( $(#[cfg $cfg])? Self::$variant => $simd, )* }
            }

            /// Whether unaligned loads/stores cost about the same as aligned
            /// ones, so an unaligned iterator need not be avoided.
            #[inline(always)]
            pub const fn unaligned_is_cheap(self) -> bool {
                match self { $( $(#[cfg $cfg])? Self::$variant => $unaligned, )* }
            }

            /// Suggested unroll factor for bulk loops, scaled to the register
            /// file: more registers allow more accumulators in flight.
            #[inline(always)]
            pub const fn unroll_factor(self) -> usize {
                match self { $( $(#[cfg $cfg])? Self::$variant => $unroll, )* }
            }

            /// Whether the ISA has first-class masked operations (AVX-512
            /// opmask registers), letting the `_c`/`_m`/`_z` variants lower to a
            /// single instruction instead of a blend.
            #[inline(always)]
            pub const fn has_masked_operations(self) -> bool {
                match self { $( $(#[cfg $cfg])? Self::$variant => $masked, )* }
            }
        }
    };
}

isa_properties! {
    Scalar {
        registers: 1, fma: false, simd: false, unaligned_cheap: true, unroll: 4, masked: false,
    }

    #[cfg(feature = "std_simd")]
    StdSimd {
        registers: 1, fma: false, simd: true, unaligned_cheap: false, unroll: 1, masked: false,
    }

    Unknown {
        registers: 1, fma: false, simd: false, unaligned_cheap: false, unroll: 1, masked: false,
    }

    // 8 XMM registers on legacy SSE, 16 from SSE4.2/AVX2, 32 with AVX-512.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V1 {
        registers: 8, fma: false, simd: true, unaligned_cheap: false, unroll: 4, masked: false,
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V2 {
        registers: 16, fma: false, simd: true, unaligned_cheap: false, unroll: 4, masked: false,
    }

    // Unaligned access stops being penalised around AVX2.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V3 {
        registers: 16, fma: true, simd: true, unaligned_cheap: true, unroll: 4, masked: false,
    }

    // Twice the registers, so twice the unroll; and the only ISA here with real
    // masked operations.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86V4 {
        registers: 32, fma: true, simd: true, unaligned_cheap: true, unroll: 8, masked: true,
    }

    #[cfg(target_arch = "aarch64")]
    NEON {
        registers: 32, fma: true, simd: true, unaligned_cheap: true, unroll: 4, masked: false,
    }

    // TODO: verify the wasm register count and unaligned cost; the engine's JIT
    // decides both, so these are conservative guesses.
    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    WASM32 {
        registers: 16, fma: false, simd: true, unaligned_cheap: false, unroll: 2, masked: false,
    }

    #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
    WASM64 {
        registers: 16, fma: false, simd: true, unaligned_cheap: false, unroll: 2, masked: false,
    }

    // SIMT: one lane per invocation, FMA via `OpFma`, no alignment penalty, and
    // extra unrolling only raises register pressure and hurts occupancy.
    #[cfg(all(feature = "spirv", target_arch = "spirv"))]
    SPIRV {
        registers: 1, fma: true, simd: false, unaligned_cheap: true, unroll: 1, masked: false,
    }
}

#[cfg(test)]
mod tests {
    use super::InstructionSet;

    /// Every variant compiled on this target.
    fn all() -> &'static [InstructionSet] {
        &[
            InstructionSet::Scalar,
            InstructionSet::Unknown,
            #[cfg(feature = "std_simd")]
            InstructionSet::StdSimd,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V2,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V3,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            InstructionSet::X86V4,
            #[cfg(target_arch = "aarch64")]
            InstructionSet::NEON,
            #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
            InstructionSet::WASM32,
            #[cfg(all(feature = "wasm", target_arch = "wasm64"))]
            InstructionSet::WASM64,
        ]
    }

    /// Properties must be sane for every variant, whatever the target. These are
    /// consumed by codegen heuristics, so a zero would be actively harmful.
    #[test]
    fn properties_are_sane() {
        for &isa in all() {
            assert!(isa.num_registers() >= 1, "{isa:?}: zero registers");
            assert!(isa.unroll_factor() >= 1, "{isa:?}: zero unroll factor");
            // Masked operations are a SIMD feature; nothing scalar can have them.
            assert!(!isa.has_masked_operations() || isa.is_simd(), "{isa:?}: masked but not SIMD");
        }

        assert!(!InstructionSet::Scalar.is_simd());
        assert!(!InstructionSet::Unknown.is_simd());
        assert!(!InstructionSet::Scalar.has_fma());
    }

    /// Pins the x86 rows of the table. These feed real codegen decisions
    /// (`unroll_factor` in the transform loops, `has_fma` in the math kernels),
    /// so changing one should be deliberate.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_rows() {
        use InstructionSet::{X86V1, X86V2, X86V3, X86V4};

        assert_eq!((X86V1.num_registers(), X86V2.num_registers()), (8, 16));
        assert_eq!((X86V3.num_registers(), X86V4.num_registers()), (16, 32));

        // FMA arrives with AVX2 (v3).
        assert!(!X86V1.has_fma() && !X86V2.has_fma() && X86V3.has_fma() && X86V4.has_fma());

        // Unaligned access stops being penalised at v3.
        assert!(!X86V1.unaligned_is_cheap() && !X86V2.unaligned_is_cheap());
        assert!(X86V3.unaligned_is_cheap() && X86V4.unaligned_is_cheap());

        // Only AVX-512 has real masked operations.
        assert!(!X86V1.has_masked_operations() && !X86V2.has_masked_operations());
        assert!(!X86V3.has_masked_operations() && X86V4.has_masked_operations());

        // Twice the registers, twice the accumulators.
        assert_eq!(X86V3.unroll_factor(), 4);
        assert_eq!(X86V4.unroll_factor(), 8);

        // Declaration order ascends by capability.
        assert!(InstructionSet::Scalar < X86V1 && X86V1 < X86V2 && X86V2 < X86V3 && X86V3 < X86V4);
        assert_eq!(InstructionSet::min(X86V2, X86V4), X86V2);
        assert_eq!(InstructionSet::max(X86V2, X86V4), X86V4);
        assert_eq!(InstructionSet::assert_eq(X86V3, X86V3), X86V3);
    }

    #[test]
    #[should_panic(expected = "InstructionSet equality assertion failed")]
    fn assert_eq_rejects_mismatch() {
        InstructionSet::assert_eq(InstructionSet::Scalar, InstructionSet::Unknown);
    }

    /// ILP is a property of the host, not of the ISA variant -- previously this
    /// was a `match` whose first arm was a `_` wildcard, making every later arm
    /// dead code.
    #[test]
    fn ilp_does_not_vary_by_variant() {
        let expected = InstructionSet::Scalar.has_instruction_level_parallelism();
        for &isa in all() {
            assert_eq!(isa.has_instruction_level_parallelism(), expected, "{isa:?}");
        }
        assert_eq!(
            expected,
            cfg!(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "arm",
                target_arch = "aarch64"
            ))
        );
    }

    /// `get()` must exist and return something this build can actually run.
    #[test]
    fn get_is_available() {
        let isa = InstructionSet::get();
        assert!(all().contains(&isa) || isa == InstructionSet::Scalar, "{isa:?} is not a compiled variant");
    }
}

use super::InstructionSet;

/// Uncached detection. `InstructionSet::get` runs this once through a
/// [`DetectOnce`](super::DetectOnce) and publishes the answer.
pub fn detect() -> InstructionSet {
    // Hand-rolled `cpuid` (see `crate::isa::x86`) rather than a detection
    // crate: the same module already has to speak `cpuid` for cache and
    // topology queries, so this costs nothing and drops a dependency.
    // Its AVX-class flags already fold in the `XCR0` check:
    // the OS must save YMM/ZMM state, not just the CPU implement it.
    let features = crate::isa::x86::features();

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
        // every AVX2 CPU. Checking it here lets dispatched code assume it, the
        // same way the V2 level already does.
        best = InstructionSet::X86V3;
    } else if features.sse42 && features.popcnt {
        best = InstructionSet::X86V2;
    } else if features.sse2 {
        best = InstructionSet::X86V1;
    }

    best
}

//! x86-v4 backend: AVX-512, parameterized by which AVX-512 sub-extensions the
//! target CPU actually has.
//!
//! AVX-512 is not one ISA. It is a foundation (F) plus a dozen optional
//! extensions that shipped in different years on different parts. Instead of
//! one lowest-common-denominator backend, the backend is generic over
//! [`Avx512Features`]: one body of register code, with
//! `if const { F::AVX512VBMI }`-style forks selecting the best encoding the
//! instantiated tier allows, each with an explicit else fallback. Everything
//! folds at monomorphization, like the `HAS_NATIVE_FMA` / `HAS_APPROX_RCP`
//! capability gates -- and unlike `cfg` gating, BOTH arms of every fork
//! compile and type-check on every build, so untaken arms cannot rot and can
//! even be differentially tested from builds that never select them.
//!
//! **Exactly one tier is compiled per build.** The `avx512-tier1..3` crate
//! features (which chain, so cargo feature unification resolves to the highest
//! requested) select `DefaultAvx512`, and dispatch carries the single
//! `X86V4Default` instantiation. Hardware below the compiled tier falls back
//! to the x86-v3 (AVX2) backend; with no tier feature enabled the backend is
//! not compiled at all and AVX-512 hardware runs x86-v3.
//!
//! The three rungs below are this crate's, not Intel's -- there is no official
//! "tier" concept. They match, exactly:
//!
//! - the `avx512-tier1..3` crate features,
//! - the `arch::tiers::tierN` intrinsic modules in [`crate::backend::x86`],
//! - [`Avx512Tier`] and `Features::avx512_tier`, which report what the
//!   *hardware* reaches.
//!
//! | Tier | Struct | Set | Silicon | SDE flag |
//! |---|---|---|---|---|
//! | 1 | [`Tier1`] | F + CD + BW + DQ + VL | Skylake-SP+, i.e. everything real | `-skx` |
//! | 2 | [`Tier2`] | + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES, VPCLMULQDQ | Ice Lake, Tiger Lake | `-icx` |
//! | 3 | [`Tier3`] | + BF16 | Sapphire/Granite Rapids, Zen 4/5 | `-spr` |
//!
//! The floor is deliberately BW + DQ + VL, not bare F. The only silicon that
//! ever shipped F without them was Knights Landing/Mill (discontinued
//! 2018-2019); without VL there are no EVEX encodings at 128/256-bit -- and VL
//! is what turns the `_c`/`_m`/`_z` variants into single masked instructions
//! at the register widths thermite already uses -- and without BW there are no
//! 8/16-bit lanes or 32/64-bit opmasks. F+CD-only hardware detects below the
//! ladder (`Features::avx512_tier()` returns `None`) and runs x86-v3.
//!
//! A const being `true` here does not make the intrinsic callable: the
//! `#[target_feature(enable = ...)]` set the dispatch macro emits for this
//! backend must enable the same feature, or the fork it guards is undefined
//! behavior if reached. The two lists are maintained by hand, in
//! `thermite-macros/src/dispatch.rs` and here. The corollary contract for
//! register code: every use of a tier-2+ intrinsic sits behind the `if const`
//! fork naming its feature -- a missed guard is NOT a compile error (the
//! `arch` namespace exposes the full top-tier surface so fallback arms always
//! compile), it is a latent illegal instruction that only the per-tier SDE
//! runs catch.

#![allow(non_camel_case_types)]

use crate::cpu::x86::Avx512Tier;

/// Which AVX-512 sub-extensions a given instantiation of the x86-v4 backend may
/// use.
///
/// Every const is a compile-time answer to "may I emit this instruction here?".
/// Register code reads them through `if const { ... }`, so a disabled feature
/// costs nothing at runtime -- the branch is gone before codegen.
///
/// Implementors are the ZSTs [`Tier1`]..[`Tier3`]. The supertraits mirror what
/// [`NativeIsa`](crate::simd::NativeIsa) demands of a backend marker, since
/// `X86V4` carries this one as a parameter.
pub trait Avx512Features:
    Copy + Clone + core::fmt::Debug + PartialEq + Eq + core::hash::Hash + Send + Sync + 'static
{
    /// Which rung of the ladder this is. Only for diagnostics, ordering, and
    /// the dispatch gate -- register code should ask about a *feature*, never a
    /// tier number, so that adding a tier never silently changes what an
    /// existing fork means.
    const TIER: Avx512Tier;

    // --- The floor: true on every tier, every real AVX-512 part -------------
    //
    // Consts rather than implicit assumptions so the feature list reads
    // completely; register code never needs to fork on these.

    /// AVX-512 Foundation: ZMM registers, opmask registers `k0`-`k7`, EVEX.
    const AVX512F: bool = true;

    /// Conflict Detection (`vpconflict`, `vplzcnt`). Shipped with F on every
    /// part.
    const AVX512CD: bool = true;

    /// Vector Length Extensions. Not new operations: the EVEX encodings applied
    /// to XMM/YMM, i.e. masking/broadcast/regs-16-31 on the 128- and 256-bit
    /// registers this crate already has. The single most important feature in
    /// the whole set for thermite, which is why it is floor, not a rung.
    const AVX512VL: bool = true;

    /// Byte and Word: 8- and 16-bit lane operations, and the 32/64-bit opmask
    /// registers they need.
    const AVX512BW: bool = true;

    /// Doubleword and Quadword: the missing 32/64-bit integer ops (`vpmullq`,
    /// int<->float converts) and the float bitwise/`vfpclass` family.
    const AVX512DQ: bool = true;

    /// F16C half-precision conversion. True on every AVX-512 part.
    const F16C: bool = true;

    /// 128-bit carry-less multiply. True on every part at or above the floor
    /// (only Knights Landing shipped without the AES/PCLMULQDQ block).
    const PCLMULQDQ: bool = true;

    /// BMI2 (`pdep`/`pext`, `shlx`, ...). Scalar GPR instructions, not
    /// AVX-512, but asserted at the floor because every AVX-512 part has it -
    /// and every AVX-512-capable AMD part is Zen 4+, where `pdep`/`pext` are
    /// fast (the microcoded-PDEP trap is Zen 1/2, which never had AVX-512).
    /// The KMask interleave/deinterleave lowerings rely on it.
    const BMI2: bool = true;

    // --- Tier 2: Ice Lake and later -----------------------------------------

    /// Vector Byte Manipulation: full cross-lane byte permutes (`vpermb`,
    /// `vpermt2b`) -- a `pshufb` without the 128-bit lane barrier.
    const AVX512VBMI: bool = false;

    /// VBMI2: funnel shifts (`vpshld`/`vpshrd`) and byte/word compress/expand.
    const AVX512VBMI2: bool = false;

    /// Vector Neural Network Instructions: fused int8/int16 dot-product
    /// accumulate (`vpdpbusd`, `vpdpwssd`).
    const AVX512VNNI: bool = false;

    /// Bit Algorithms: per-byte/word population count (`vpopcntb`/`w`) and
    /// `vpshufbitqmb`.
    const AVX512BITALG: bool = false;

    /// Population count on 32/64-bit lanes (`vpopcntd`/`q`).
    const AVX512VPOPCNTDQ: bool = false;

    /// Integer FMA: 52-bit multiply-accumulate (`vpmadd52luq`/`huq`), for bignum
    /// and exact-integer work.
    const AVX512IFMA: bool = false;

    /// Galois Field New Instructions (`gf2p8affineqb`, `gf2p8mulb`). Not an
    /// AVX-512 feature -- it has a legacy SSE encoding, and Zen 4 shipped it
    /// alongside AVX-512 -- but tier 2 wants the EVEX 512-bit form, and the
    /// affine transform doubles as a general per-bit permute primitive.
    const GFNI: bool = false;

    /// AES on YMM/ZMM. Independent of AVX-512 (Zen 3 has it with AVX2 alone);
    /// tier 2 wants the 512-bit form.
    const VAES: bool = false;

    /// Carry-less multiply on YMM/ZMM. Independent of AVX-512, same as
    /// [`VAES`](Self::VAES); the 512-bit form accelerates wide Morton codes and
    /// GF(2) work.
    const VPCLMULQDQ: bool = false;

    // --- Tier 3: current parts -----------------------------------------------

    /// BF16 dot product (`vdpbf16ps`) and the f32 <-> bf16 converts. Note this
    /// is *not* full BF16 arithmetic -- that is AVX10.2 -- so it accelerates
    /// the packed-bf16 storage path, not a bf16 compute type.
    const AVX512BF16: bool = false;

    // --- Deliberately outside the ladder ----------------------------------

    /// IEEE half-precision *arithmetic* on ZMM (not merely F16C conversion).
    ///
    /// Always false: no tier requires it, on purpose. FP16 is Sapphire Rapids
    /// and later on the Intel side and absent from Zen 4/5, so folding it into
    /// tier 3 would lock every AMD AVX-512 part out of the top rung. It exists
    /// here so a future tier (or a bespoke instantiation) can turn it on
    /// without reshaping the trait.
    const AVX512FP16: bool = false;
}

/// Tier 1: F + CD + BW + DQ + VL. The Skylake-SP set, i.e. what "has AVX-512"
/// means on every part that ever mattered, and this ladder's floor.
///
/// VL makes the `_c`/`_m`/`_z` masked variants single instructions on the
/// 128/256-bit registers, and BW covers the 8/16-bit lane families -- so this
/// rung already accelerates everything thermite does, at every width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier1;

/// Tier 2: + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES,
/// VPCLMULQDQ. Ice Lake (2019) and later on the Intel side.
///
/// Adds lane-crossing byte permutes (VBMI), byte/word compress/expand and
/// funnel shifts (VBMI2), per-byte popcount (BITALG), 52-bit integer FMA
/// (IFMA), and GFNI's affine transform as a general bit permute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier2;

/// Tier 3: + BF16. Sapphire Rapids, Granite Rapids, Zen 4/5.
///
/// Deliberately does **not** require AVX512-FP16 (see
/// [`Avx512Features::AVX512FP16`]) -- that would exclude every AMD part.
///
/// Watch out for Cooper Lake: it has BF16 *without* the tier-2 set, so it
/// reports [`Avx512Tier::Tier1`] and never selects this tier. That is the one
/// part the linear ladder cannot place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier3;

impl Avx512Features for Tier1 {
    const TIER: Avx512Tier = Avx512Tier::Tier1;
}

impl Avx512Features for Tier2 {
    const TIER: Avx512Tier = Avx512Tier::Tier2;

    const AVX512VBMI: bool = true;
    const AVX512VBMI2: bool = true;
    const AVX512VNNI: bool = true;
    const AVX512BITALG: bool = true;
    const AVX512VPOPCNTDQ: bool = true;
    const AVX512IFMA: bool = true;
    const GFNI: bool = true;
    const VAES: bool = true;
    const VPCLMULQDQ: bool = true;
}

impl Avx512Features for Tier3 {
    const TIER: Avx512Tier = Avx512Tier::Tier3;

    const AVX512VBMI: bool = true;
    const AVX512VBMI2: bool = true;
    const AVX512VNNI: bool = true;
    const AVX512BITALG: bool = true;
    const AVX512VPOPCNTDQ: bool = true;
    const AVX512IFMA: bool = true;
    const GFNI: bool = true;
    const VAES: bool = true;
    const VPCLMULQDQ: bool = true;

    const AVX512BF16: bool = true;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct X86V4<F: Avx512Features>(core::marker::PhantomData<F>);

// The compiled backend proper. Exactly one tier per build: the chained
// `avx512-tier*` features resolve here, and everything below (arch namespace,
// polyfills, registers) exists only when some tier is requested -- a default
// build pays zero compile time for AVX-512.

#[cfg(feature = "avx512-tier1")]
cfg_select! {
    feature = "avx512-tier3" => { pub type DefaultAvx512 = Tier3; }
    feature = "avx512-tier2" => { pub type DefaultAvx512 = Tier2; }
    _ => { pub type DefaultAvx512 = Tier1; }
}

/// The single `X86V4` instantiation this build carries; what dispatch and
/// the `BACKENDS` table name.
#[cfg(feature = "avx512-tier1")]
pub type X86V4Default = X86V4<DefaultAvx512>;

/// The rung [`DefaultAvx512`] sits on. The runtime detector compares the
/// hardware's `Features::avx512_tier()` against this: below it, the v4
/// backend must not be selected (its `#[target_feature]` trampolines would
/// execute encodings the CPU lacks) and the hardware falls back to x86-v3.
#[cfg(feature = "avx512-tier1")]
pub const COMPILED_TIER: Avx512Tier = <DefaultAvx512 as Avx512Features>::TIER;

/// The flat namespace register files call into: polyfills + real
/// intrinsics.
///
/// Deliberately the FULL top-tier intrinsic surface regardless of
/// [`DefaultAvx512`]: `if const { F::FEATURE }` fallback arms only work if
/// the guarded intrinsics are in scope on builds that never take them.
/// Codegen stays clean by construction -- LLVM refuses to inline a callee
/// whose `target_feature` set is incompatible with the caller, so an
/// un-enabled intrinsic in a dead arm compiles to a plain call and the
/// `if const` fold dead-code-eliminates it. The `if const` guard, not the
/// import, is the correctness boundary (see the module docs).
#[cfg(feature = "avx512-tier1")]
pub mod arch {
    pub use super::polyfills::*;
    // Both globs are needed: `avx512f::*` carries the F-level intrinsics plus
    // the whole avx2-and-below ladder (prefetch, denormal toggles, ...), while
    // the tier modules re-export only their own extensions -- the
    // `pub(super)` glob inside `tiers` does not propagate the base set.
    pub use crate::backend::x86::avx512f::*;
    pub use crate::backend::x86::avx512f::tiers::tier3::*;
}

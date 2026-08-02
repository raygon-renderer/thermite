//! x86-v4 backend: AVX-512, parameterized by which AVX-512 sub-extensions the
//! target CPU actually has.
//!
//! AVX-512 is not one ISA. It is a foundation (F) plus a dozen optional
//! extensions that shipped in different years on different parts, so a single
//! `X86V4` backend would either target the lowest common denominator (Knights
//! Landing, 2016) or refuse to run on most AVX-512 hardware. Instead the
//! backend is generic over [`Avx512Features`]: one body of register code, with
//! `if const { F::AVX512VBMI }`-style forks selecting the best encoding the
//! instantiated tier allows. Everything folds at monomorphization, like the
//! `HAS_TRUE_FMA` / `HAS_APPROX_RCP` capability gates.
//!
//! The four rungs below are this crate's, not Intel's -- there is no official
//! "tier" concept. They match, exactly:
//!
//! - the `avx512-tier1..4` crate features,
//! - the `arch::tiers::tierN` intrinsic modules in [`crate::backend::x86`],
//! - [`Avx512Tier`] and `Features::avx512_tier`, which report what the
//!   *hardware* reaches.
//!
//! | Tier | Struct | Adds | Silicon | SDE flag |
//! |---|---|---|---|---|
//! | 1 | [`Tier1`] | F + CD | Knights Landing/Mill (dead) | `-knl` |
//! | 2 | [`Tier2`] | + BW + DQ + VL | Skylake-SP, Cascade Lake | `-skx` |
//! | 3 | [`Tier3`] | + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES, VPCLMULQDQ | Ice Lake, Tiger Lake, Zen 4 | `-icx` |
//! | 4 | [`Tier4`] | + BF16 | Sapphire Rapids, Granite Rapids, Zen 4/5 | `-spr` |
//!
//! Tier 2 is the one that matters most. VL is not new operations but the EVEX
//! encodings (masking, zero-masking, embedded broadcast, registers 16-31)
//! applied to XMM/YMM instead of ZMM only, which turns the `_c`/`_m`/`_z`
//! variants into single masked instructions at the register widths thermite
//! already uses, rather than only on new 512-bit ones.
//!
//! A const being `true` here does not make the intrinsic callable: the
//! `#[target_feature(enable = ...)]` set the dispatch macro emits for this
//! backend must enable the same feature, or the fork it guards fails to
//! compile. The two lists are maintained by hand, in
//! `thermite-macros/src/dispatch.rs` and here.

#![allow(non_camel_case_types)]

use crate::cpu::x86::Avx512Tier;

/// Which AVX-512 sub-extensions a given instantiation of the x86-v4 backend may
/// use.
///
/// Every const is a compile-time answer to "may I emit this instruction here?".
/// Register code reads them through `if const { ... }`, so a disabled feature
/// costs nothing at runtime -- the branch is gone before codegen.
///
/// Implementors are the ZSTs [`Tier1`]..[`Tier4`]. The supertraits mirror what
/// [`NativeIsa`](crate::simd::NativeIsa) demands of a backend marker, since
/// `X86V4` carries this one as a parameter.
pub trait Avx512Features:
    Copy + Clone + core::fmt::Debug + PartialEq + Eq + core::hash::Hash + Send + Sync + 'static
{
    /// Which rung of the ladder this is. Only for diagnostics and ordering --
    /// code should ask about a *feature*, never a tier number, so that adding a
    /// tier never silently changes what an existing fork means.
    const TIER: Avx512Tier;

    // --- Tier 1: the foundation -------------------------------------------

    /// AVX-512 Foundation: ZMM registers, opmask registers `k0`-`k7`, EVEX.
    /// Never false -- there is no x86-v4 without it, and it is a const rather
    /// than an implicit assumption so the feature list reads completely.
    const AVX512F: bool = true;

    /// Conflict Detection (`vpconflict`, `vplzcnt`). Shipped with F on every
    /// part, hence tier 1 rather than a rung of its own.
    const AVX512CD: bool = true;

    // --- Tier 2: what "has AVX-512" means to everyone ---------------------

    /// Vector Length Extensions. Not new operations: the EVEX encodings applied
    /// to XMM/YMM, i.e. masking/broadcast/regs-16-31 on the 128- and 256-bit
    /// registers this crate already has.
    const AVX512VL: bool = false;

    /// Byte and Word: 8- and 16-bit lane operations, and the 32/64-bit opmask
    /// registers they need. Required by anything touching `u8xN`/`i16xN`.
    const AVX512BW: bool = false;

    /// Doubleword and Quadword: the missing 32/64-bit integer ops (`vpmullq`,
    /// int<->float converts) and the float bitwise/`vfpclass` family.
    const AVX512DQ: bool = false;

    // --- Tier 3: Ice Lake and later ---------------------------------------

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
    /// alongside AVX-512 -- but tier 3 wants the EVEX 512-bit form, and the
    /// affine transform doubles as a general per-bit permute primitive.
    const GFNI: bool = false;

    /// AES on YMM/ZMM. Independent of AVX-512 (Zen 3 has it with AVX2 alone);
    /// tier 3 wants the 512-bit form.
    const VAES: bool = false;

    /// Carry-less multiply on YMM/ZMM. Independent of AVX-512, same as
    /// [`VAES`](Self::VAES); the 512-bit form accelerates wide Morton codes and
    /// GF(2) work.
    const VPCLMULQDQ: bool = false;

    // --- Tier 4: current parts --------------------------------------------

    /// BF16 dot product (`vdpbf16ps`) and the f32 <-> bf16 converts. Note this
    /// is *not* full BF16 arithmetic -- that is AVX10.2 -- so it accelerates
    /// the packed-bf16 storage path, not a bf16 compute type.
    const AVX512BF16: bool = false;

    // --- Deliberately outside the ladder ----------------------------------

    /// IEEE half-precision *arithmetic* on ZMM (not merely F16C conversion).
    ///
    /// Always false: no tier requires it, on purpose. FP16 is Sapphire Rapids
    /// and later on the Intel side and absent from Zen 4/5, so folding it into
    /// tier 4 would lock every AMD AVX-512 part out of the top rung. It exists
    /// here so a future tier (or a bespoke instantiation) can turn it on
    /// without reshaping the trait.
    const AVX512FP16: bool = false;

    // --- Inherited from x86-v3 --------------------------------------------
    //
    // The v4 backend builds on the v3 (AVX2 + FMA) arch namespace, whose f16c
    // and pclmulqdq intrinsics are gated behind the `avx2-f16c`/`avx2-pclmul`
    // crate features because a few AVX2 parts lack them. At AVX-512 that doubt
    // mostly goes away, so the tiers assert them directly and v4 register code
    // can skip the `cfg`.

    /// F16C half-precision conversion. True on every AVX-512 part, including
    /// Knights Landing.
    const F16C: bool = true;

    /// 128-bit carry-less multiply. True from tier 2 up; **false on tier 1**,
    /// because Knights Landing shipped without the AES/PCLMULQDQ block.
    const PCLMULQDQ: bool = true;
}

/// Tier 1: F + CD only. Exactly the Knights Landing set (which also had the
/// long-dead ER and PF).
///
/// 512-bit or nothing: without VL there are no EVEX encodings at 128/256-bit,
/// so this tier cannot accelerate the `f32x4`/`f32x8` registers thermite
/// already has -- only new 512-bit ones. The hardware is extinct (Knights
/// Landing/Mill were discontinued in 2018-2019) and no other part ever shipped
/// F without BW/DQ/VL, so tier 2 is the realistic floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier1;

/// Tier 2: + BW + DQ + VL. The Skylake-SP set, and what nearly everyone means
/// by "has AVX-512".
///
/// The first tier that pays off for existing code: VL makes the `_c`/`_m`/`_z`
/// masked variants single instructions on the 128/256-bit registers, and BW
/// covers the 8/16-bit lane families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier2;

/// Tier 3: + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES,
/// VPCLMULQDQ. Ice Lake (2019) and later, and Zen 4 on the AMD side.
///
/// Adds lane-crossing byte permutes (VBMI), per-byte popcount (BITALG), 52-bit
/// integer FMA (IFMA), and GFNI's affine transform as a general bit permute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier3;

/// Tier 4: + BF16. Sapphire Rapids, Granite Rapids, Zen 4/5.
///
/// Deliberately does **not** require AVX512-FP16 (see
/// [`Avx512Features::AVX512FP16`]) -- that would exclude every AMD part.
///
/// Watch out for Cooper Lake: it has BF16 *without* the tier-3 set, so it
/// reports [`Avx512Tier::Tier2`] and never selects this tier. That is the one
/// part the linear ladder cannot place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tier4;

impl Avx512Features for Tier1 {
    const TIER: Avx512Tier = Avx512Tier::Tier1;

    // Knights Landing has no AES/PCLMULQDQ block.
    const PCLMULQDQ: bool = false;
}

impl Avx512Features for Tier2 {
    const TIER: Avx512Tier = Avx512Tier::Tier2;

    const AVX512VL: bool = true;
    const AVX512BW: bool = true;
    const AVX512DQ: bool = true;
}

impl Avx512Features for Tier3 {
    const TIER: Avx512Tier = Avx512Tier::Tier3;

    const AVX512VL: bool = true;
    const AVX512BW: bool = true;
    const AVX512DQ: bool = true;

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

impl Avx512Features for Tier4 {
    const TIER: Avx512Tier = Avx512Tier::Tier4;

    const AVX512VL: bool = true;
    const AVX512BW: bool = true;
    const AVX512DQ: bool = true;

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
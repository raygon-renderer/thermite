//! x86 / x86_64 feature detection via `cpuid`: the bits
//! [`InstructionSet::get`](crate::isa::InstructionSet::get) dispatches on, plus
//! the vendor/family/model helpers that the `thermite-cpu` crate builds its
//! machine facts (caches, topology, microcode quirks) on. Leaves used:
//!
//! | Leaf | Field |
//! |---|---|
//! | `0` | max leaf + vendor string |
//! | `1` | family/model, SSE2/SSE4.2/POPCNT/PCLMUL/AVX/FMA/F16C, `OSXSAVE` |
//! | `7`:0 | AVX2, the `avx512*` alphabet, GFNI/VAES/VPCLMULQDQ |
//! | `7`:1 `EAX[5]` | AVX512-BF16 |
//! | `7`:1 `EDX[19]` | AVX10 enumerated (leaf `0x24` is valid) |
//! | `0x24` `EBX[7:0]` | AVX10 converged version ([`features`]) |
//!
//! Plus `xgetbv` for `XCR0`: every AVX-class flag means **usable**, folding in
//! whether the OS saves the YMM/ZMM/opmask state.
//!
//! `cpuid` is serializing (100-250 cycles bare metal, a VM exit under a
//! hypervisor), which is why every caller caches the result behind a
//! [`DetectOnce`](crate::isa::DetectOnce).

/// How much of AVX-512 the CPU implements, in the rungs this crate's
/// `avx512-tier1..3` crate features and `backend::x86::avx512f::tiers` intrinsic
/// modules are cut at. Each tier implies every lower one.
///
/// The ladder is this crate's, not Intel's, as there is no official "tier"
/// concept, so [`Features::avx512_tier`] only reports which rung the hardware
/// reaches.
///
/// The floor is deliberately **F + CD + BW + DQ + VL**, not bare F: the only
/// silicon that ever shipped F without the other four was Knights Landing/Mill
/// (discontinued 2018-2019), and without VL there are no EVEX encodings at
/// 128/256-bit, and without BW no 8/16-bit lanes or 32/64-bit opmasks. A
/// backend for that shape would be a parallel implementation for extinct hardware.
/// F+CD-only parts report `None` and run the AVX2 backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Avx512Tier {
    /// F + CD + BW + DQ + **VL**. The Skylake-SP set, which introduced all of
    /// them together, and what everyone means by "has AVX-512".
    ///
    /// VL is the one that matters most to this crate: it is not new operations
    /// but an orthogonal capability letting the EVEX encodings apply to XMM/YMM:
    /// write-masking, zero-masking, embedded broadcast and registers 16-31 at
    /// 128/256-bit. That is what turns the `_c`/`_m`/`_z` variants into single
    /// masked instructions on the *existing* `f32x4`/`f32x8` register widths,
    /// rather than only on new 512-bit ones.
    Tier1,
    /// Tier 1 + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES,
    /// VPCLMULQDQ (Ice Lake and later).
    Tier2,
    /// Tier 2 + BF16 (Sapphire Rapids, Zen 4+).
    ///
    /// Cooper Lake has BF16 _without_ the tier-2 set, so it reports
    /// [`Avx512Tier::Tier1`], the one part the linear ladder cannot place.
    Tier3,
}

/// AVX10 converged-vector-ISA version, from leaf `0x24`.
///
/// AVX10 retires the per-feature AVX-512 alphabet in favour of a single
/// monotonic version number: version N is a strict superset of version N-1,
/// and there are no optional sub-features to enumerate. AVX10.1 is
/// architecturally defined as the complete Granite Rapids AVX-512 feature set
/// (every `avx512*` flag in [`Features`], including FP16) at 128/256/512-bit
/// vector lengths, so [`Features::avx512_tier`] reports [`Avx512Tier::Tier3`]
/// on any AVX10 part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Avx10Version {
    /// AVX10.1 (Granite Rapids). No new operations: the rebranding rung, fixing
    /// the AVX-512 baseline described above so software can key on one number.
    V10_1,
    /// AVX10.2 (Diamond Rapids). The first rung with new instructions: full
    /// BF16 *arithmetic* (not just `vdpbf16ps`), FP8 conversions, saturating
    /// integer converts, `vminmax*`/`vcomx` compares, and media additions.
    V10_2,
}

#[cfg(target_arch = "x86")]
use core::arch::x86::{__cpuid_count, _xgetbv, CpuidResult};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__cpuid_count, _xgetbv, CpuidResult};

#[inline]
fn bit(value: u32, index: u32) -> bool {
    (value >> index) & 1 != 0
}

#[inline]
pub fn cpuid(leaf: u32, sub: u32) -> CpuidResult {
    // SAFETY: `cpuid` is unprivileged and has no preconditions on any CPU this
    // crate can target (486+). Callers bound `leaf` by the reported maximum.
    __cpuid_count(leaf, sub)
}

/// Highest basic leaf, and highest extended (`0x8000_xxxx`) leaf.
#[inline]
pub fn max_leaves() -> (u32, u32) {
    let basic = cpuid(0, 0).eax;
    let ext = cpuid(0x8000_0000, 0).eax;
    // A CPU with no extended leaves returns something < 0x8000_0000 here.
    (basic, if ext > 0x8000_0000 { ext } else { 0 })
}

/// Whether this CPU uses AMD's extended-leaf layouts (`0x8000001D` caches,
/// `0x8000001E` topology).
///
/// Hygon is a licensed Zen derivative and reports `HygonGenuine` while
/// implementing the same leaves, so it counts. Other x86 vendors exist and are
/// deliberately *not* enumerated here. Zhaoxin (`  Shanghai  ` and the
/// inherited VIA `CentaurHauls`) ships AVX2 parts, and the point of the
/// probe-then-fall-through structure below is that an unrecognised vendor still
/// gets whichever leaves it does implement. This only picks the order to try.
#[inline]
pub fn is_amd_lineage() -> bool {
    let r = cpuid(0, 0);
    // Vendor string arrives split across EBX, EDX, ECX.
    let amd = r.ebx == 0x6874_7541 && r.edx == 0x6974_6e65 && r.ecx == 0x444d_4163; // "AuthenticAMD"
    let hygon = r.ebx == 0x6f677948 && r.edx == 0x6e65476e && r.ecx == 0x656e6975; // "HygonGenuine"
    amd || hygon
}

/// Whether this is a genuine Intel part. Unlike [`is_amd_lineage`], which only
/// picks which leaves to *try*, this gates a model-number table
/// (`thermite-cpu`'s `quirks` table) whose entries are meaningless on a clone.
#[inline]
pub fn is_intel() -> bool {
    let r = cpuid(0, 0);
    r.ebx == 0x756e_6547 && r.edx == 0x4965_6e69 && r.ecx == 0x6c65_746e // "GenuineIntel"
}

/// Family *and* model, with both extended fields folded in per the x86 rules.
///
/// The two fields have different rules and mixing them up is the classic bug:
/// the extended-family field applies only to base family `0xf`, while the
/// extended-model field applies to base families `0x6` **and** `0xf`, which
/// is exactly the pair that matters, since every Intel Core part is family 6
/// and every AMD Zen part is family `0x17`+ (base `0xf`, extended).
#[inline]
pub fn family_model() -> (u32, u32) {
    let eax = cpuid(1, 0).eax;
    let base_family = (eax >> 8) & 0xf;
    let base_model = (eax >> 4) & 0xf;

    let family = if base_family == 0xf {
        base_family + ((eax >> 20) & 0xff)
    } else {
        base_family
    };
    let model = if base_family == 0x6 || base_family == 0xf {
        (((eax >> 16) & 0xf) << 4) | base_model
    } else {
        base_model
    };

    (family, model)
}

/// The feature bits [`InstructionSet`](crate::isa::InstructionSet) dispatches
/// on, with the OS-state check already folded in.
///
/// Every AVX-class flag here means **usable**, not merely "present in
/// silicon": `cpuid` reports what the CPU implements, but `XCR0` reports which
/// register state the kernel has agreed to preserve across a context switch.
/// Touching YMM/ZMM without both is silent corruption, and it is the single
/// easiest thing to get wrong when hand-rolling this.
#[derive(Debug, Clone, Copy, Default)]
pub struct Features {
    pub sse2: bool,
    pub sse42: bool,
    pub popcnt: bool,
    pub pclmulqdq: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,

    // AVX-512. `avx512f` gates every `avx512*` flag below: they are all `false`
    // unless the foundation itself is usable, so one sub-feature can be tested
    // on its own. (`gfni`/`vaes`/`vpclmulqdq` are NOT in this group, see below.)
    pub avx512f: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    /// Vector Length Extensions: the EVEX encodings (masking, zero-masking,
    /// embedded broadcast, registers 16-31) applied to XMM/YMM rather than ZMM
    /// only. Required by every rung of [`Avx512Tier`]: it is part of the floor.
    pub avx512vl: bool,
    pub avx512vbmi: bool,
    pub avx512vbmi2: bool,
    pub avx512vnni: bool,
    pub avx512bitalg: bool,
    pub avx512vpopcntdq: bool,
    pub avx512ifma: bool,
    pub avx512bf16: bool,
    /// IEEE half-precision *arithmetic* on ZMM, not merely F16C conversion.
    /// Sapphire Rapids and later on the Intel side, absent from Zen 4/5, which
    /// is why no [`Avx512Tier`] requires it. Part of the AVX10.1 baseline.
    pub avx512fp16: bool,

    // Enumerated among the AVX-512 bits, but independent features. Zen 3 has
    // VAES and VPCLMULQDQ with AVX2 and no AVX-512 whatsoever (GFNI arrived with
    // Zen 4). Never gated on AVX-512, and tier 3 wants their 512-bit forms, which is
    // why `avx512_tier` only consults them alongside `avx512f`.
    /// Needs no AVX: the legacy SSE encoding of `gf2p8mulb` and friends runs on
    /// any CPU reporting the bit.
    pub gfni: bool,
    pub vaes: bool,
    pub vpclmulqdq: bool,

    /// AVX10 converged version from leaf `0x24` `EBX[7:0]`: `0` = no AVX10,
    /// `1` = AVX10.1, `2` = AVX10.2, higher = a future superset. Raw so an
    /// unknown future version is preserved; [`Features::avx10`] maps it to the
    /// [`Avx10Version`] rungs this crate knows. Like the `avx512*` flags it
    /// means **usable** (zeroed unless the OS saves ZMM/opmask state), and a
    /// non-zero version implies every `avx512*` flag above is set (see
    /// [`features`]).
    pub avx10_version: u8,
}

impl Features {
    /// The highest AVX-512 tier this CPU satisfies, matching the `avx512-tier1..3`
    /// crate features and the `arch::tiers::tierN` intrinsic modules in
    /// `backend/x86.rs` **exactly** -- the tier ladder is defined there, and this
    /// only reports which rung the hardware reaches.
    ///
    /// F+CD-only hardware (Knights Landing) reports `None`, below the ladder's
    /// floor. See [`Avx512Tier`].
    ///
    /// An AVX10 part always reports [`Avx512Tier::Tier3`]: AVX10.1 subsumes the
    /// whole ladder, and [`features`] folds that guarantee into the individual
    /// flags this reads.
    pub fn avx512_tier(&self) -> Option<Avx512Tier> {
        // tier1: F + CD + BW + DQ + VL (Skylake-SP shipped them together, no CPU
        // has BW/DQ without VL, and only Knights Landing had F without the rest).
        if !(self.avx512f && self.avx512cd && self.avx512bw && self.avx512dq && self.avx512vl) {
            return None;
        }
        // tier2: + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES, VPCLMULQDQ
        let tier2 = self.avx512vbmi
            && self.avx512vbmi2
            && self.avx512vnni
            && self.avx512bitalg
            && self.avx512vpopcntdq
            && self.avx512ifma
            && self.gfni
            && self.vaes
            && self.vpclmulqdq;
        if !tier2 {
            return Some(Avx512Tier::Tier1);
        }
        // tier3: + BF16
        if !self.avx512bf16 {
            return Some(Avx512Tier::Tier2);
        }
        Some(Avx512Tier::Tier3)
    }

    /// The AVX10 version this CPU implements, if any.
    pub fn avx10(&self) -> Option<Avx10Version> {
        match self.avx10_version {
            0 => None,
            1 => Some(Avx10Version::V10_1),
            // Versions are strict supersets with no optional parts, so an
            // unknown future version still delivers everything 10.2 promises.
            _ => Some(Avx10Version::V10_2),
        }
    }
}

/// Probe the CPU. Costs a few `cpuid`s, so callers cache the result.
pub fn features() -> Features {
    let mut f = Features::default();

    let (max_basic, _) = max_leaves();
    if max_basic < 1 {
        return f;
    }

    let leaf1 = cpuid(1, 0);

    f.sse2 = bit(leaf1.edx, 26);
    f.sse42 = bit(leaf1.ecx, 20);
    f.popcnt = bit(leaf1.ecx, 23);
    f.pclmulqdq = bit(leaf1.ecx, 1);

    // XMM | YMM_Hi128, and the three AVX-512 state components.
    const XCR0_AVX: u64 = 0b110;
    const XCR0_AVX512: u64 = 0b1110_0000;

    let osxsave = bit(leaf1.ecx, 27);
    // SAFETY: `xgetbv` #UDs unless CR4.OSXSAVE is set, which is exactly what
    // CPUID.1:ECX[27] reports, guarded above.
    let xcr0 = if osxsave { unsafe { _xgetbv(0) } } else { 0 };
    let os_saves_ymm = osxsave && (xcr0 & XCR0_AVX) == XCR0_AVX;
    // AVX-512 and AVX10 both need three more XCR0 components on top of AVX's:
    // the opmask registers, the upper half of ZMM0-15, and ZMM16-31.
    let os_saves_zmm = os_saves_ymm && (xcr0 & XCR0_AVX512) == XCR0_AVX512;

    f.avx = os_saves_ymm && bit(leaf1.ecx, 28);
    // FMA and F16C operate on YMM, so they inherit the same OS requirement.
    f.fma = f.avx && bit(leaf1.ecx, 12);
    f.f16c = f.avx && bit(leaf1.ecx, 29);

    if max_basic >= 7 {
        let leaf7 = cpuid(7, 0);
        // Subleaf 0's EAX reports the max subleaf, and subleaf 1 carries AVX512-BF16
        // and the AVX10 enumeration bit, so check before reading.
        let leaf7_1 = (leaf7.eax >= 1).then(|| cpuid(7, 1));

        f.avx2 = f.avx && bit(leaf7.ebx, 5);

        f.avx512f = f.avx && os_saves_zmm && bit(leaf7.ebx, 16);

        if f.avx512f {
            // Leaf 7 subleaf 0: EBX
            f.avx512dq = bit(leaf7.ebx, 17);
            f.avx512ifma = bit(leaf7.ebx, 21);
            f.avx512cd = bit(leaf7.ebx, 28);
            f.avx512bw = bit(leaf7.ebx, 30);
            f.avx512vl = bit(leaf7.ebx, 31);

            // Leaf 7 subleaf 0: ECX
            f.avx512vbmi = bit(leaf7.ecx, 1);
            f.avx512vbmi2 = bit(leaf7.ecx, 6);
            f.avx512vnni = bit(leaf7.ecx, 11);
            f.avx512bitalg = bit(leaf7.ecx, 12);
            f.avx512vpopcntdq = bit(leaf7.ecx, 14);

            // Leaf 7 subleaf 0: EDX
            f.avx512fp16 = bit(leaf7.edx, 23);

            // BF16 is the odd one out: leaf 7 *subleaf 1*, EAX[5].
            if let Some(l) = leaf7_1 {
                f.avx512bf16 = bit(l.eax, 5);
            }
        }

        // GFNI, VAES and VPCLMULQDQ are enumerated among the AVX-512 bits but
        // are NOT AVX-512 features, and treating them as such is a real and
        // repeated bug (Linux carries a "Fix dependencies for GFNI, VAES, and
        // VPCLMULQDQ" patch for exactly this). Verified on a Ryzen 9 5950X
        // (Zen 3): VAES + VPCLMULQDQ present, GFNI absent, no AVX-512 at all.
        //
        // Their dependencies differ, so they are gated individually:
        //   * VAES / VPCLMULQDQ  -- need AVX for the VEX encodings.
        //   * GFNI               -- needs nothing. The legacy SSE form runs
        //                           without AVX; only the wider forms scale with
        //                           AVX / AVX-512.
        f.gfni = bit(leaf7.ecx, 8);
        f.vaes = f.avx && bit(leaf7.ecx, 9);
        f.vpclmulqdq = f.avx && bit(leaf7.ecx, 10);

        // --- AVX10 ------------------------------------------------------
        // 7:1 EDX[19] only says leaf 0x24 is valid, and the capability itself is
        // that leaf's converged version number. The 256-bit-max option (and
        // with it the vector-length enumeration in 0x24 EBX[18:16]) was
        // dropped from the spec in rev 2.0 (AVX10 always means all three
        // widths), so the length bits are deliberately not consulted: the SDM
        // now marks them reserved-at-1 purely for software written against the
        // original spec (Linux/KVM read only the version too). Usability is
        // therefore gated on the same OS ZMM state as AVX-512.
        if os_saves_zmm
            && f.avx
            && max_basic >= 0x24
            && let Some(l) = leaf7_1
            && bit(l.edx, 19)
        {
            f.avx10_version = (cpuid(0x24, 0).ebx & 0xff) as u8;
        }

        // AVX10.1 is architecturally defined as the complete Granite Rapids
        // AVX-512 feature set at every vector length, so fold that guarantee
        // into the individual flags. On every shipped part this is a no-op,
        // since the legacy bits are still enumerated alongside AVX10, but the
        // spec only promises that for early processors, and dispatch keyed on
        // `avx512f` (or the tier ladder) must keep working when the legacy
        // bits eventually go dark.
        if f.avx10_version >= 1 {
            f.avx512f = true;
            f.avx512cd = true;
            f.avx512bw = true;
            f.avx512dq = true;
            f.avx512vl = true;
            f.avx512vbmi = true;
            f.avx512vbmi2 = true;
            f.avx512vnni = true;
            f.avx512bitalg = true;
            f.avx512vpopcntdq = true;
            f.avx512ifma = true;
            f.avx512bf16 = true;
            f.avx512fp16 = true;
            f.gfni = true;
            f.vaes = true;
            f.vpclmulqdq = true;
        }
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatcher's hand-rolled `cpuid` must agree with the ISA the crate
    /// was actually compiled to run on: if the build enabled a feature
    /// statically, detection has to see it too.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_features_agree_with_build() {
        let f = features();

        if cfg!(target_feature = "sse2") {
            assert!(f.sse2, "built with sse2 but not detected");
        }
        if cfg!(target_feature = "avx2") {
            assert!(f.avx2, "built with avx2 but not detected");
        }
        if cfg!(target_feature = "fma") {
            assert!(f.fma, "built with fma but not detected");
        }

        // Implication chain: the wider level cannot be usable without the narrower.
        assert!(!f.avx2 || f.avx, "avx2 without avx");
        assert!(!f.avx512f || f.avx, "avx512f without avx");
        assert!(!f.fma || f.avx, "fma without avx");
        assert!(!f.sse42 || f.sse2, "sse4.2 without sse2");

        // And it must pick a level consistent with those bits.
        use crate::isa::InstructionSet;
        let isa = InstructionSet::get();
        match isa {
            InstructionSet::X86V3 => assert!(f.avx2 && f.fma && f.popcnt),
            InstructionSet::X86V2 => assert!(f.sse42 && f.popcnt),
            InstructionSet::X86V1 => assert!(f.sse2),
            _ => {}
        }
    }

    /// The oracle for the hand-rolled detection that replaced `core_detect`:
    /// `std`'s `std_detect` is the reference implementation, and it applies the
    /// same `XCR0` rules. Any disagreement means dispatch could pick a backend
    /// the OS or CPU cannot actually run, so this is checked bit for bit.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_features_match_std_detect() {
        let f = features();

        assert_eq!(f.sse2, std::is_x86_feature_detected!("sse2"), "sse2");
        assert_eq!(f.sse42, std::is_x86_feature_detected!("sse4.2"), "sse4.2");
        assert_eq!(f.popcnt, std::is_x86_feature_detected!("popcnt"), "popcnt");
        assert_eq!(f.pclmulqdq, std::is_x86_feature_detected!("pclmulqdq"), "pclmulqdq");
        assert_eq!(f.avx, std::is_x86_feature_detected!("avx"), "avx");
        assert_eq!(f.avx2, std::is_x86_feature_detected!("avx2"), "avx2");
        assert_eq!(f.fma, std::is_x86_feature_detected!("fma"), "fma");
        assert_eq!(f.f16c, std::is_x86_feature_detected!("f16c"), "f16c");
        assert_eq!(f.avx512f, std::is_x86_feature_detected!("avx512f"), "avx512f");

        // The AVX-512 sub-features behind the tier ladder. `std_detect` applies
        // the same XCR0 gate, so these must match on AVX-512 hardware and all be
        // false on this (AVX2) machine.
        assert_eq!(f.avx512cd, std::is_x86_feature_detected!("avx512cd"), "avx512cd");
        assert_eq!(f.avx512bw, std::is_x86_feature_detected!("avx512bw"), "avx512bw");
        assert_eq!(f.avx512dq, std::is_x86_feature_detected!("avx512dq"), "avx512dq");
        assert_eq!(f.avx512vl, std::is_x86_feature_detected!("avx512vl"), "avx512vl");
        assert_eq!(f.avx512vbmi, std::is_x86_feature_detected!("avx512vbmi"), "avx512vbmi");
        assert_eq!(
            f.avx512vbmi2,
            std::is_x86_feature_detected!("avx512vbmi2"),
            "avx512vbmi2"
        );
        assert_eq!(f.avx512vnni, std::is_x86_feature_detected!("avx512vnni"), "avx512vnni");
        assert_eq!(
            f.avx512bitalg,
            std::is_x86_feature_detected!("avx512bitalg"),
            "avx512bitalg"
        );
        assert_eq!(
            f.avx512vpopcntdq,
            std::is_x86_feature_detected!("avx512vpopcntdq"),
            "avx512vpopcntdq"
        );
        assert_eq!(f.avx512ifma, std::is_x86_feature_detected!("avx512ifma"), "avx512ifma");
        assert_eq!(f.avx512bf16, std::is_x86_feature_detected!("avx512bf16"), "avx512bf16");
        assert_eq!(f.avx512fp16, std::is_x86_feature_detected!("avx512fp16"), "avx512fp16");
        assert_eq!(f.gfni, std::is_x86_feature_detected!("gfni"), "gfni");
        assert_eq!(f.vaes, std::is_x86_feature_detected!("vaes"), "vaes");
        assert_eq!(f.vpclmulqdq, std::is_x86_feature_detected!("vpclmulqdq"), "vpclmulqdq");
    }

    /// The tier ladder must match `backend::x86::avx512f::tiers` exactly, and be
    /// monotone: reaching tier N implies every feature of tiers below it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx512_tiers_are_monotone() {
        let f = features();

        // No tier and no `avx512*` sub-feature without the foundation.
        // Deliberately NOT asserted for gfni/vaes/vpclmulqdq: those are separate
        // features that exist on AVX2-only parts (Zen 3+), and asserting
        // otherwise is the bug this test caught in the first place.
        if !f.avx512f {
            assert_eq!(f.avx512_tier(), None, "a tier without AVX512F");
            assert!(!f.avx512cd && !f.avx512bw && !f.avx512dq && !f.avx512vl);
            assert!(!f.avx512vbmi && !f.avx512vbmi2 && !f.avx512vnni && !f.avx512bitalg);
            assert!(!f.avx512vpopcntdq && !f.avx512ifma && !f.avx512bf16 && !f.avx512fp16);
            // AVX10 folds the foundation in, so it cannot outlive it either.
            assert_eq!(f.avx10_version, 0, "AVX10 without AVX512F");
        }

        // Synthesise each rung and check it reports exactly that rung: this pins
        // the ladder itself, on any host, including CI without AVX-512.
        // The Knights Landing shape (F + CD and nothing else) is below the
        // ladder's floor: without BW/DQ/VL there is nothing the backend wants.
        let mut synthetic = Features {
            avx512f: true,
            avx512cd: true,
            ..Default::default()
        };
        assert_eq!(synthetic.avx512_tier(), None, "KNL shape is not a tier");

        // BW + DQ alone is still not tier 1: VL is required with them. (No real
        // CPU is shaped like this, Skylake-SP having brought all three at once,
        // but it pins that VL actually gates the floor.)
        synthetic.avx512bw = true;
        synthetic.avx512dq = true;
        assert_eq!(synthetic.avx512_tier(), None, "promoted without VL");

        synthetic.avx512vl = true;
        assert_eq!(synthetic.avx512_tier(), Some(Avx512Tier::Tier1));

        // Tier 2 needs all nine, so check it does not promote on a partial set.
        synthetic.avx512vbmi = true;
        synthetic.avx512vnni = true;
        assert_eq!(
            synthetic.avx512_tier(),
            Some(Avx512Tier::Tier1),
            "promoted on a partial tier 2"
        );

        synthetic.avx512vbmi2 = true;
        synthetic.avx512bitalg = true;
        synthetic.avx512vpopcntdq = true;
        synthetic.avx512ifma = true;
        synthetic.gfni = true;
        synthetic.vaes = true;
        synthetic.vpclmulqdq = true;
        assert_eq!(synthetic.avx512_tier(), Some(Avx512Tier::Tier2));

        synthetic.avx512bf16 = true;
        assert_eq!(synthetic.avx512_tier(), Some(Avx512Tier::Tier3));

        // Dropping VL from a full-featured part falls below the ladder
        // entirely: without it there are no 128/256-bit EVEX encodings, which
        // is most of what this crate wants from AVX-512. (Also the Cooper Lake
        // pin, inverted: BF16 without the tier-2 set stays tier 1.)
        let mut no_vl = synthetic;
        no_vl.avx512vl = false;
        assert_eq!(no_vl.avx512_tier(), None, "VL is part of the floor");

        let cooper_lake = Features {
            avx512f: true,
            avx512cd: true,
            avx512bw: true,
            avx512dq: true,
            avx512vl: true,
            avx512bf16: true,
            ..Default::default() // no tier-2 set
        };
        assert_eq!(
            cooper_lake.avx512_tier(),
            Some(Avx512Tier::Tier1),
            "BF16 without the tier-2 set must not promote"
        );

        // F alone is below the floor too.
        let f_only = Features {
            avx512f: true,
            ..Default::default()
        };
        assert_eq!(f_only.avx512_tier(), None);

        assert!(Avx512Tier::Tier1 < Avx512Tier::Tier3, "tiers must order");
    }

    /// AVX10 is a version number, not a feature alphabet: `features()` must
    /// fold version >= 1 into the full AVX-512 flag set, and the raw-version
    /// mapping must treat unknown future versions as supersets.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx10_implies_the_full_ladder() {
        let f = features();

        // On real AVX10 hardware the fold must have landed: the whole ladder,
        // plus the pieces no tier requires (FP16).
        if f.avx10().is_some() {
            assert_eq!(f.avx512_tier(), Some(Avx512Tier::Tier3));
            assert!(f.avx512fp16 && f.avx512vl && f.gfni && f.vaes && f.vpclmulqdq);
        }

        // The raw-version -> rung mapping. Versions are strict supersets with
        // no optional parts, so an unknown future version still satisfies
        // everything 10.2 promises and must not report `None`.
        let mut s = Features::default();
        assert_eq!(s.avx10(), None);
        s.avx10_version = 1;
        assert_eq!(s.avx10(), Some(Avx10Version::V10_1));
        s.avx10_version = 2;
        assert_eq!(s.avx10(), Some(Avx10Version::V10_2));
        s.avx10_version = 9;
        assert_eq!(
            s.avx10(),
            Some(Avx10Version::V10_2),
            "future versions are supersets of 10.2"
        );

        assert!(Avx10Version::V10_1 < Avx10Version::V10_2, "versions must order");
    }
}

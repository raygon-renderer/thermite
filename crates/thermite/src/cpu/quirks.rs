//! Microarchitectural quirks: instructions that **exist** but are microcoded.
//!
//! This is the one module here that breaks the rule stated in the [parent
//! module docs](super): every value below comes from a vendor/family/model
//! table, not from the hardware. That is not an oversight. There is no
//! enumeration bit for "this instruction is a microcode sequence", there never
//! has been one, and the gap between the fast and slow implementations of the
//! same opcode reaches **70x** (see [`Quirks::fast_compress_store`](crate::cpu::quirks::Quirks)). Refusing
//! to guess would mean refusing to answer at all.
//!
//! Everything here is therefore:
//!
//! * **A hint, never a correctness input.** Each flag says "the alternative
//!   path is probably faster here", and picking wrong costs speed, never
//!   results. Nothing in this crate may branch on a quirk to decide *what* to
//!   compute.
//! * **Conservative when unsure.** An unrecognised vendor, family or model
//!   reports `false` for everything, which selects the portable path. A CPU we
//!   have never measured is assumed to have the slow implementation.
//! * **Sourced from measurements, not vendor claims.** Every number in these
//!   doc comments is from <https://uops.info>, which measures the shipped
//!   silicon. Where a claim could not be measured, the flag stays `false`.
//!
//! # Naming: instructions, not ISAs
//!
//! The intuition "AVX-512 masking is fast, AVX2 masking is slow" is a real
//! observation with the wrong cause attached. The split is the **encoding**:
//! EVEX masking is native to the load/store pipe on every part that has it,
//! while the VEX `vmaskmov` *store* is a microcode sequence on Zen 1 through
//! Zen 4. The same CPU therefore has a fast masked store and a slow masked
//! store at the same time, and Zen 5, which changed no ISA level, collapses
//! the difference by making the VEX form 2 uops. A flag named after AVX-512
//! would be wrong on both ends of that.
//!
//! # Consuming these
//!
//! Read the flag **once, at the [`dispatch`](crate::dispatch) boundary**, and
//! branch there. A quirk test inside a hot loop costs more than the quirk.
//!
//! ```no_run
//! use thermite::cpu::quirks::Quirks;
//!
//! if Quirks::get().fast_compress_store {
//!     // compress straight to memory
//! } else {
//!     // compress in-register, then store
//! }
//! ```

use core::cell::UnsafeCell;
use core::sync::atomic::AtomicU8;

use super::Cache;

/// Which microcoded-instruction traps this CPU has. Get one from
/// [`Quirks::get`].
///
/// Every field is `true` only when the instruction is known-fast on measured
/// silicon: `false` means either "measured slow" or "never measured", and the
/// caller should take the portable path either way. See the [module
/// docs](self) for why this is a model table rather than a probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Quirks {
    /// `vgather*` issues in a handful of uops rather than a microcode sequence.
    ///
    /// **`false` on every AMD part ever shipped, Zen 5 included.** Measured
    /// `VGATHERDPS ymm`, uops / reciprocal throughput: Zen+ 65/20.0, Zen 2
    /// 60/16.0, Zen 3 39/8.0, Zen 4 42/8.0, Zen 5 32/5.67 -- against Skylake
    /// 5/5.0 and Alder Lake-P 7/3.03. Zen 5 halved the cost and is still
    /// spending 32 uops of front-end and retire bandwidth that the surrounding
    /// code wants.
    ///
    /// Also `false` on Haswell (34/11.0) and Broadwell (13/6.0), and on
    /// anything the Gather Data Sampling microcode touched. See
    /// [`Quirks::detect`] for why that case cannot be answered honestly off
    /// Linux.
    pub fast_gather: bool,

    /// `vscatter*`/`vpscatter*` is worth using over a scalar store loop.
    ///
    /// Measured `VPSCATTERDD zmm`: Skylake-X 35 uops/16.0, Ice Lake 35/8.0,
    /// Emerald Rapids 35/8.0, Zen 4 89/22.0, Zen 5 88/17.0. Scatter is
    /// inherently 16 stores, so 8.0 is close to the floor and Ice Lake onward
    /// earns the flag; Skylake-X at double that does not, and neither Zen does.
    pub fast_scatter: bool,

    /// The **VEX** masked store (`vmaskmovps`/`vpmaskmovd` with a memory
    /// destination) is a real store rather than a microcode sequence.
    ///
    /// The sharpest AMD cliff measured here, and the one most likely to be
    /// mis-attributed to AVX-512. `VMASKMOVPS m256`: Zen+ through Zen 4 all sit
    /// at 42-44 uops / 12.0, then **Zen 5 drops to 2 uops / 0.5**. Intel is
    /// 3-4 uops / 1.0 throughout.
    ///
    /// Deliberately not covering two neighbours that need no flag:
    /// * the masked **load** is 1 uop / 0.5 from Zen 2 on (only Zen+ was bad,
    ///   at 36/10.0) and 2 uops / 0.5 on Intel;
    /// * the **EVEX** masked store (`vmovups m512 {k}`) is 2 uops everywhere it
    ///   exists, at rtp 2.0 on Zen 4 and 1.0 on Zen 5, Skylake-X and Ice Lake.
    ///
    /// One caveat this flag cannot express: on Zen 4 an EVEX masked access
    /// whose *masked-out* lane would have faulted costs roughly 256 cycles on a
    /// load and 355 on a store, because fault suppression is handled as an
    /// assist. Masked tail handling that runs off the end of a mapping into an
    /// unmapped page pays that, on the fast path, silently.
    pub fast_vex_masked_store: bool,

    /// `vpcompress*`/`vpexpand*` with a **memory** destination is worth using
    /// over the register form plus a separate store.
    ///
    /// The largest single penalty in this table. `VPCOMPRESSD m512 {k}`:
    /// Intel Skylake-X through Emerald Rapids 4 uops / 2.0, Zen 5 8 uops / 3.0,
    /// and **Zen 4 144 uops / 72.5**. Seventy-two cycles for one store, on the
    /// crate's stream-compaction path. Zen 4 alone; Zen 5 fixed it.
    pub fast_compress_store: bool,

    /// `vpconflict*` is a hardware operation rather than a microcode sequence.
    ///
    /// The one entry where **Intel is the slow vendor**. `VPCONFLICTD zmm`:
    /// Zen 4 2 uops / 1.33 and Zen 5 2 / 1.0, against Skylake-X 35-37 / 18.5,
    /// Cannon Lake 37 / 18.0 and Ice Lake 37 / 18.5. No Intel part has been
    /// measured with a fast form, so this is `false` for all of them rather
    /// than optimistic about the ones not in the data.
    ///
    /// Gates [`count_conflicts`](crate::vector::IntegerVector::count_conflicts)
    /// and [`group_by_value`](crate::vector::PartialOrdVector::group_by_value). Only
    /// meaningful when `avx512cd` is also present, since on a part without it the
    /// polyfill runs regardless of what this says.
    pub fast_conflict_detect: bool,

    /// `pdep`/`pext` are single-uop rather than microcoded.
    ///
    /// `PEXT r64`: 1 uop / 1.0 on Intel from Haswell on and on AMD from Zen 3
    /// on (Zen 5 reaches 0.33), against **7 uops / 19.0 on Zen 1, Zen+ and
    /// Zen 2** -- all one family, `0x17`. Nineteen cycles versus one.
    ///
    /// Gates the BMI2 path of the Morton interleave, whose alternative is the
    /// carry-less-multiply path behind the `avx2-pclmul` crate feature.
    pub fast_pdep_pext: bool,
}

impl Quirks {
    /// Assume every microcoded trap is present: what an unrecognised CPU, and
    /// every non-x86 target, reports.
    ///
    /// All-`false` is the safe direction. Each flag selects the portable path,
    /// which is never wrong, only slower on hardware that did not need it.
    pub const CONSERVATIVE: Self = Self {
        fast_gather: false,
        fast_scatter: false,
        fast_vex_masked_store: false,
        fast_compress_store: false,
        fast_conflict_detect: false,
        fast_pdep_pext: false,
    };

    /// The cached table, looked up once on first call.
    ///
    /// Unlike [`current_core_type`](super::current_core_type) this is cached
    /// even on a hybrid part: P and E cores of the same CPU can differ here in
    /// principle, but re-deriving the table per call would cost a `cpuid` to
    /// answer a question whose answer is a hint.
    #[inline]
    pub fn get() -> &'static Quirks {
        static CACHE: Cache<Quirks> = Cache {
            state: AtomicU8::new(super::UNINIT),
            value: UnsafeCell::new(Quirks::CONSERVATIVE),
        };

        CACHE.get(Quirks::detect)
    }

    /// Look the table up **now**, bypassing the cache.
    ///
    /// # The one case this cannot answer
    ///
    /// Gather Data Sampling (GDS, "Downfall", CVE-2022-40982) is mitigated by a
    /// microcode update that makes `vgather*` dramatically slower on Skylake
    /// through Rocket Lake. It is visible in the measurements: `VGATHERQPD
    /// zmm` is 4 uops / 5.0 on Ice Lake but 14 uops / 20.0 on Rocket Lake, the
    /// same core generation either side of the mitigation.
    ///
    /// So on that range of parts the honest answer depends on the *loaded
    /// microcode revision*, which no model table encodes. The status lives in
    /// `IA32_ARCH_CAPABILITIES[GDS_CTRL]`/`[GDS_NO]`, an MSR, and is
    /// unreachable from user space. Linux re-exports it through sysfs and this
    /// reads it when `std` is available. Every other OS gets the pessimistic
    /// answer, because reporting a fast gather that the microcode quietly made
    /// 4x slower is the more expensive mistake.
    pub fn detect() -> Quirks {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            detect_x86()
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // Not a stub for want of data: none of these instructions exist off
            // x86, so "take the portable path" is the only correct answer.
            Quirks::CONSERVATIVE
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_x86() -> Quirks {
    use super::x86::{family_model, is_amd_lineage, is_intel};

    let (family, model) = family_model();

    if is_amd_lineage() {
        return amd_quirks(family);
    }
    if is_intel() && family == 6 {
        return intel_quirks(model);
    }

    // Zhaoxin, VIA, a hypervisor masking the vendor string, or a family 5/15
    // part far older than anything this matters for.
    Quirks::CONSERVATIVE
}

/// AMD and Hygon, keyed on family alone, since every quirk here moved on a family
/// boundary, so no model list is needed.
///
/// * `0x17`: Zen 1, Zen+, Zen 2. Also Hygon Dhyana (`0x18`), a Zen 1 clone.
/// * `0x19`: Zen 3, Zen 4.
/// * `0x1A`: Zen 5.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn amd_quirks(family: u32) -> Quirks {
    // No AMD part has ever had a fast gather or scatter, so future families
    // inherit `false` for those two rather than optimism.
    match family {
        // Zen 5 and later. The generation that fixed the VEX masked store
        // (42 uops -> 2) and compress-to-memory (144 uops -> 8).
        f if f >= 0x1A => Quirks {
            fast_gather: false,
            fast_scatter: false,
            fast_vex_masked_store: true,
            fast_compress_store: true,
            fast_conflict_detect: true,
            fast_pdep_pext: true,
        },
        // Zen 3 and Zen 4. `vpconflict` and `pdep`/`pext` are hardware from
        // Zen 3; the masked store and compress-store traps are still here.
        0x19 => Quirks {
            fast_conflict_detect: true,
            fast_pdep_pext: true,
            ..Quirks::CONSERVATIVE
        },
        // Zen 1 / Zen+ / Zen 2 (and the Hygon derivative): every trap present,
        // including the 19-cycle `pext`.
        0x17 | 0x18 => Quirks::CONSERVATIVE,
        // Bulldozer and older, or a family newer than this table knows about
        // that somehow skipped 0x1A.
        _ => Quirks::CONSERVATIVE,
    }
}

/// Intel family 6 (every Core and Atom part), keyed on model.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn intel_quirks(model: u32) -> Quirks {
    // Skylake through Rocket Lake: fast gather in silicon, but the range the
    // GDS microcode slowed down. See `Quirks::detect`.
    const GDS_AFFECTED: &[u32] = &[
        0x4E, 0x5E, // Skylake client
        0x55, // Skylake-SP / Cascade Lake / Cooper Lake
        0x8E, 0x9E, 0xA5, 0xA6, // Kaby / Coffee / Comet Lake
        0x66, // Cannon Lake
        0x7D, 0x7E, // Ice Lake client
        0x6A, 0x6C, // Ice Lake server
        0x8C, 0x8D, // Tiger Lake
        0xA7, // Rocket Lake
    ];

    // Alder Lake and later: never affected by GDS, and the fastest gathers
    // measured anywhere (Alder Lake-P, 7 uops / 3.03).
    const FAST_GATHER: &[u32] = &[
        0x97, 0x9A, // Alder Lake
        0xB7, 0xBA, 0xBF, // Raptor Lake
        0xAA, 0xAC, // Meteor Lake
        0x8F, // Sapphire Rapids
        0xCF, // Emerald Rapids
        0xAD, 0xAE, // Granite Rapids
    ];

    // Ice Lake onward halved scatter's cost (35 uops / 16.0 -> 35 / 8.0).
    // Skylake-X and older do not qualify.
    const FAST_SCATTER: &[u32] = &[0x7D, 0x7E, 0x6A, 0x6C, 0x8C, 0x8D, 0xA7, 0x8F, 0xCF, 0xAD, 0xAE];

    let gds_slowed = GDS_AFFECTED.contains(&model) && gds_mitigated().unwrap_or(true);

    Quirks {
        // On a hybrid part this describes the P core: the E-core gather has
        // not been measured here, and pessimising the whole package on that
        // basis would give up the best gather in the table.
        fast_gather: FAST_GATHER.contains(&model) || (GDS_AFFECTED.contains(&model) && !gds_slowed),
        fast_scatter: FAST_SCATTER.contains(&model),
        // 3-4 uops / 1.0 on every Intel part measured, back to Haswell.
        fast_vex_masked_store: true,
        // 4 uops / 2.0 from Skylake-X to Emerald Rapids. Meaningless without
        // AVX-512, and harmless there: the polyfill runs regardless.
        fast_compress_store: true,
        // 37 uops / 18.5 everywhere it has been measured. No Intel part earns
        // this one.
        fast_conflict_detect: false,
        // 1 uop / 1.0 since Haswell, which predates every model above.
        fast_pdep_pext: true,
    }
}

/// Whether the GDS ("Downfall") microcode mitigation is active, if the platform
/// will say. `None` means it could not be determined, which callers must read
/// as "assume mitigated".
///
/// `Vulnerable` is deliberately *not* treated as mitigated: it means the
/// microcode update was never applied, so gather still runs at its original
/// speed.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn gds_mitigated() -> Option<bool> {
    #[cfg(all(feature = "std", target_os = "linux"))]
    {
        let status = std::fs::read_to_string("/sys/devices/system/cpu/vulnerabilities/gather_data_sampling").ok()?;
        let status = status.trim();
        Some(!(status.starts_with("Not affected") || status.starts_with("Vulnerable")))
    }
    #[cfg(not(all(feature = "std", target_os = "linux")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is a cache: repeated `get()` must not re-derive or change.
    #[test]
    fn quirks_are_stable() {
        let a = *Quirks::get();
        let b = *Quirks::get();
        assert_eq!(a, b);
        assert!(core::ptr::eq(Quirks::get(), Quirks::get()));
        assert_eq!(a, Quirks::detect(), "uncached detect disagrees with the cache");
    }

    /// The conservative answer must actually be conservative: every flag
    /// `false`, so every caller takes the portable path.
    #[test]
    fn conservative_is_all_false() {
        let c = Quirks::CONSERVATIVE;
        assert!(!c.fast_gather && !c.fast_scatter && !c.fast_vex_masked_store);
        assert!(!c.fast_compress_store && !c.fast_conflict_detect && !c.fast_pdep_pext);
        assert_eq!(c, Quirks::default(), "Default must match CONSERVATIVE");
    }

    /// Pin the AMD family ladder, on any host: these are the two generations
    /// where a flag flips, and getting either boundary wrong silently selects
    /// a 19-cycle `pext` or a 72-cycle compress-store.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn amd_ladder_flips_at_the_right_families() {
        // Zen 1 / Zen+ / Zen 2: everything slow, including pdep/pext.
        let zen2 = amd_quirks(0x17);
        assert_eq!(zen2, Quirks::CONSERVATIVE);
        assert_eq!(amd_quirks(0x18), Quirks::CONSERVATIVE, "Hygon Dhyana tracks Zen 1");

        // Zen 3 fixed pdep/pext and brought hardware vpconflict, but did NOT
        // fix the masked store or compress-store.
        let zen4 = amd_quirks(0x19);
        assert!(zen4.fast_pdep_pext && zen4.fast_conflict_detect);
        assert!(!zen4.fast_vex_masked_store, "Zen 4 VEX masked store is 42 uops");
        assert!(!zen4.fast_compress_store, "Zen 4 compress-store is 144 uops");

        // Zen 5 fixed both of those.
        let zen5 = amd_quirks(0x1A);
        assert!(zen5.fast_vex_masked_store && zen5.fast_compress_store);

        // No AMD family, present or future, claims a fast gather or scatter.
        for f in [0x15, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x20] {
            let q = amd_quirks(f);
            assert!(!q.fast_gather, "family {f:#x} claimed a fast gather");
            assert!(!q.fast_scatter, "family {f:#x} claimed a fast scatter");
        }
    }

    /// Intel's table is model-keyed, so the risk is a model falling through to
    /// the wrong answer rather than a boundary being off by one.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn intel_models_land_where_intended() {
        // Alder Lake-P: the fastest gather measured, and not GDS-affected.
        assert!(intel_quirks(0x97).fast_gather, "Alder Lake should have a fast gather");
        // Sapphire Rapids: fast gather and fast scatter.
        let spr = intel_quirks(0x8F);
        assert!(spr.fast_gather && spr.fast_scatter);

        // Skylake-X: GDS-affected, so pessimistic off Linux, and its scatter
        // (16.0) never qualified regardless.
        assert!(!intel_quirks(0x55).fast_scatter, "Skylake-X scatter is 35 uops / 16.0");

        // Haswell (0x3C) is in neither list: gather 34 uops / 11.0.
        let hsw = intel_quirks(0x3C);
        assert!(!hsw.fast_gather && !hsw.fast_scatter);
        // But it has had single-uop pdep/pext and a real masked store all along.
        assert!(hsw.fast_pdep_pext && hsw.fast_vex_masked_store);

        // No Intel part earns fast conflict detection.
        for m in [0x55, 0x66, 0x7D, 0x8F, 0xAD, 0x97] {
            assert!(
                !intel_quirks(m).fast_conflict_detect,
                "model {m:#x} claimed a fast vpconflict; none has been measured"
            );
        }
    }

    /// An unknown vendor must fall through to the conservative answer rather
    /// than inheriting whichever table was checked last.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn unknown_intel_model_is_conservative() {
        // A plausible future model in neither list.
        let unknown = intel_quirks(0xFE);
        assert!(!unknown.fast_gather && !unknown.fast_scatter);
    }
}

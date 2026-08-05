//! x86 / x86_64 machine facts via `cpuid`.
//!
//! The only target that can answer all of this with a plain user-space
//! instruction, so this path needs no OS and works in `no_std`. Leaves used:
//!
//! | Leaf | Field |
//! |---|---|
//! | `0` | max leaf + vendor string |
//! | `1` `EBX[15:8]` | line size (`clflush` granularity, x8) |
//! | `4` / `0x8000001D` | deterministic cache parameters per level |
//! | `0x1F` / `0xB` | extended topology (SMT + core level counts) |
//! | `0x80000008` / `0x8000001E` | AMD's topology, when the above are absent |
//! | `7`:0 `EDX[15]` | hybrid part |
//! | `0x1A` `EAX[31:24]` | this core's type (`0x20` Atom/E, `0x40` Core/P) |
//! | `7`:1 `EDX[19]` | AVX10 enumerated (leaf `0x24` is valid) |
//! | `0x24` `EBX[7:0]` | AVX10 converged version ([`features`](crate::cpu::x86::features)) |
//!
//! Leaves are tried and *checked for an empty answer*, not merely bounded by
//! the reported maximum: a CPU can advertise a max leaf above one it does not
//! implement, in which case it returns zeros (see `read_topology_amd`).
//!
//! `cpuid` is serializing (100-250 cycles bare metal, a VM exit under a
//! hypervisor), which is why the caller caches the result.

use super::{CacheInfo, CacheKind, CoreType, CpuInfo};

/// How much of AVX-512 the CPU implements, in the rungs this crate's
/// `avx512-tier1..4` crate features and `backend::x86::avx512f::tiers` intrinsic
/// modules are cut at. Each tier implies every lower one.
///
/// The ladder is this crate's, not Intel's -- there is no official "tier"
/// concept -- so [`Features::avx512_tier`] only reports which rung the hardware
/// reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Avx512Tier {
    /// F + CD. Exactly the Knights Landing feature set (which also had the
    /// long-dead ER and PF). 512-bit only: no EVEX encodings at 128/256-bit.
    Tier1,
    /// Tier 1 + BW + DQ + **VL**. The Skylake-SP set, which introduced all
    /// three together, and what most people mean by "has AVX-512".
    ///
    /// VL is the one that matters most to this crate: it is not new operations
    /// but an orthogonal capability letting the EVEX encodings apply to XMM/YMM
    /// -- write-masking, zero-masking, embedded broadcast and registers 16-31 at
    /// 128/256-bit. That is what turns the `_c`/`_m`/`_z` variants into single
    /// masked instructions on the *existing* `f32x4`/`f32x8` register widths,
    /// rather than only on new 512-bit ones.
    Tier2,
    /// Tier 2 + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES,
    /// VPCLMULQDQ (Ice Lake and later).
    Tier3,
    /// Tier 3 + BF16 (Cooper Lake, Sapphire Rapids, Zen 4+).
    Tier4,
}

/// AVX10 converged-vector-ISA version, from leaf `0x24`.
///
/// AVX10 retires the per-feature AVX-512 alphabet in favour of a single
/// monotonic version number: version N is a strict superset of version N-1,
/// and there are no optional sub-features to enumerate. AVX10.1 is
/// architecturally defined as the complete Granite Rapids AVX-512 feature set
/// -- every `avx512*` flag in [`Features`], including FP16 -- at 128/256/512-bit
/// vector lengths, so [`Features::avx512_tier`] reports [`Avx512Tier::Tier4`]
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
fn cpuid(leaf: u32, sub: u32) -> CpuidResult {
    // SAFETY: `cpuid` is unprivileged and has no preconditions on any CPU this
    // crate can target (486+). Callers bound `leaf` by the reported maximum.
    __cpuid_count(leaf, sub)
}

/// Highest basic leaf, and highest extended (`0x8000_xxxx`) leaf.
#[inline]
fn max_leaves() -> (u32, u32) {
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
/// deliberately *not* enumerated here -- Zhaoxin (`  Shanghai  ` and the
/// inherited VIA `CentaurHauls`) ships AVX2 parts, and the point of the
/// probe-then-fall-through structure below is that an unrecognised vendor still
/// gets whichever leaves it does implement. This only picks the order to try.
#[inline]
fn is_amd_lineage() -> bool {
    let r = cpuid(0, 0);
    // Vendor string arrives split across EBX, EDX, ECX.
    let amd = r.ebx == 0x6874_7541 && r.edx == 0x6974_6e65 && r.ecx == 0x444d_4163; // "AuthenticAMD"
    let hygon = r.ebx == 0x6f677948 && r.edx == 0x6e65476e && r.ecx == 0x656e6975; // "HygonGenuine"
    amd || hygon
}

/// CPU family, with the extended-family field folded in per the x86 rules.
#[inline]
fn family() -> u32 {
    let eax = cpuid(1, 0).eax;
    let base = (eax >> 8) & 0xf;
    if base == 0xf { base + ((eax >> 20) & 0xff) } else { base }
}

/// Walk the deterministic-cache-parameter leaf. Intel uses `4`; AMD uses the
/// identically-formatted `0x8000001D` (older AMD reported nothing here, which
/// simply leaves the levels `None`).
fn read_caches(info: &mut CpuInfo, leaf: u32) {
    for sub in 0..16 {
        let r = cpuid(leaf, sub);

        let kind = match r.eax & 0x1f {
            0 => break, // no more caches
            1 => CacheKind::Data,
            2 => CacheKind::Instruction,
            _ => CacheKind::Unified,
        };

        let level = ((r.eax >> 5) & 0x7) as u8;
        let fully_associative = (r.eax >> 9) & 1 != 0;
        let shared_by = (((r.eax >> 14) & 0xfff) + 1) as u16;

        let line = u64::from(r.ebx & 0xfff) + 1;
        let partitions = u64::from((r.ebx >> 12) & 0x3ff) + 1;
        let ways = u64::from((r.ebx >> 22) & 0x3ff) + 1;
        let sets = u64::from(r.ecx) + 1;

        let entry = CacheInfo {
            size: (line * partitions * ways * sets) as u32,
            line_size: Some(line as u32),
            // 0 encodes fully associative, matching the sysfs convention.
            associativity: Some(if fully_associative { 0 } else { ways as u16 }),
            shared_by: Some(shared_by),
            kind,
        };

        match (level, kind) {
            (1, CacheKind::Data) => info.l1d = Some(entry),
            (1, CacheKind::Instruction) => info.l1i = Some(entry),
            // A unified L1 (rare, some Atom) counts as both.
            (1, CacheKind::Unified) => {
                info.l1d = Some(entry);
                info.l1i = Some(entry);
            }
            (2, _) => info.l2 = Some(entry),
            (3, _) => info.l3 = Some(entry),
            _ => {}
        }
    }
}

/// Extended topology enumeration. Each subleaf's EBX is the count of logical
/// processors *at and below* that level, so the SMT level gives threads-per-core
/// and the widest level gives logical-per-package.
fn read_topology(leaf: u32) -> (Option<u16>, Option<u16>) {
    const LEVEL_SMT: u32 = 1;

    let mut threads_per_core = None;
    let mut widest = 0u16;

    for sub in 0..8 {
        let r = cpuid(leaf, sub);
        let level_type = (r.ecx >> 8) & 0xff;
        let count = (r.ebx & 0xffff) as u16;

        if level_type == 0 {
            break; // invalid level: enumeration is done
        }
        if count == 0 {
            continue;
        }
        if level_type == LEVEL_SMT {
            threads_per_core = Some(count);
        }
        widest = widest.max(count);
    }

    (threads_per_core, (widest > 0).then_some(widest))
}

/// AMD's own topology leaves.
///
/// Necessary because AMD parts advertise a max basic leaf well above `0xB`
/// while implementing neither `0xB` nor `0x1F` -- both return all zeros, so
/// bounding by the max leaf is not enough to know the standard enumeration
/// exists. Measured on a 16-core Zen: `max_basic = 0xD`, leaf `0xB` all zeros,
/// while `0x80000008`/`0x8000001E` carry the real counts. Linux's topology code
/// documents the same fallback.
fn read_topology_amd(max_ext: u32) -> (Option<u16>, Option<u16>) {
    let mut threads_per_core = None;
    let mut logical = None;

    if max_ext >= 0x8000_0008 {
        // ECX[7:0] "NC": logical processors in this package, minus one.
        logical = u16::try_from((cpuid(0x8000_0008, 0).ecx & 0xff) + 1).ok();
    }

    // EBX[15:8] is threads-per-core minus one -- but only from family 0x17
    // (Zen). Family 0x15 advertises the leaf with a *non-zero* SMT field that
    // does not mean this, which is the exact trap Linux carries a patch for.
    // It also needs TopoExt (0x80000001:ECX[22]).
    if max_ext >= 0x8000_001E && family() >= 0x17 && bit(cpuid(0x8000_0001, 0).ecx, 22) {
        threads_per_core = u16::try_from(((cpuid(0x8000_001E, 0).ebx >> 8) & 0xff) + 1).ok();
    }

    (threads_per_core, logical)
}

pub fn detect() -> CpuInfo {
    let mut info = CpuInfo::UNKNOWN;
    let (max_basic, max_ext) = max_leaves();

    // --- cache geometry -------------------------------------------------
    if max_basic >= 1 {
        // EBX[15:8] is the `clflush` line size in 8-byte units.
        let line = ((cpuid(1, 0).ebx >> 8) & 0xff) * 8;
        if line > 0 {
            info.line_size = Some(line);
            // x86 has no separate writeback granule; a line is the unit of
            // coherence, so false-sharing padding is line-sized.
            info.writeback_granule = Some(line);
        }
    }

    // Try both cache leaves, most-likely-first by vendor, and fall through if
    // the preferred one came back empty. An unrecognised vendor therefore still
    // gets whichever it implements instead of being written off.
    let (first, second) = if is_amd_lineage() {
        (0x8000_001D, 4)
    } else {
        (4, 0x8000_001D)
    };
    for leaf in [first, second] {
        let available = if leaf >= 0x8000_0000 {
            max_ext >= leaf
        } else {
            max_basic >= leaf
        };
        if available {
            read_caches(&mut info, leaf);
        }
        if info.l1d.is_some() {
            break;
        }
    }

    // Prefer the per-level line size if leaf 1 was silent.
    if info.line_size.is_none()
        && let Some(l1d) = info.l1d
    {
        info.line_size = l1d.line_size;
        info.writeback_granule = l1d.line_size;
    }

    // --- topology -------------------------------------------------------
    // 0x1F supersedes 0xB, and either may be *present but empty* (AMD reports a
    // max basic leaf above both while implementing neither), so fall through on
    // an empty result rather than trusting the leaf bound alone.
    let mut topology = (None, None);
    if max_basic >= 0x1F {
        topology = read_topology(0x1F);
    }
    if topology.1.is_none() && max_basic >= 0xB {
        topology = read_topology(0xB);
    }
    if topology.1.is_none() {
        // Not gated on vendor: the leaves are AMD-defined but bounded by
        // `max_ext`, and a non-AMD CPU that does not implement them simply
        // reports nothing rather than garbage.
        topology = read_topology_amd(max_ext);
    }
    let (threads_per_core, logical_per_package) = topology;

    info.topology.threads_per_core = threads_per_core;

    // The OS knows the whole machine (and honours affinity masks / cgroup
    // limits); `cpuid` only ever describes one package.
    #[cfg(feature = "std")]
    {
        info.topology.logical_cores = std::thread::available_parallelism()
            .ok()
            .and_then(|n| u16::try_from(n.get()).ok());
    }
    if info.topology.logical_cores.is_none() {
        info.topology.logical_cores = logical_per_package;
    }

    if let (Some(logical), Some(per_core)) = (info.topology.logical_cores, threads_per_core)
        && per_core > 0
    {
        info.topology.physical_cores = Some(logical / per_core);
    }

    // --- hybrid ---------------------------------------------------------
    if max_basic >= 7 {
        info.hybrid = (cpuid(7, 0).edx >> 15) & 1 != 0;
    }
    // P/E *counts* need every core interrogated in turn (each `cpuid` describes
    // only the core it ran on), which means pinning threads. Deliberately left
    // `None` rather than guessed; `current_core_type()` answers for this core.

    info
}

pub fn current_core_type() -> CoreType {
    let (max_basic, _) = max_leaves();

    // Leaf 0x1A is only architecturally defined on a hybrid part.
    if max_basic < 0x1A || (cpuid(7, 0).edx >> 15) & 1 == 0 {
        return CoreType::Unknown;
    }

    match cpuid(0x1A, 0).eax >> 24 {
        0x20 => CoreType::Efficiency,  // Atom
        0x40 => CoreType::Performance, // Core
        _ => CoreType::Unknown,
    }
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
    // on its own. (`gfni`/`vaes`/`vpclmulqdq` are NOT in this group -- see below.)
    pub avx512f: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    /// Vector Length Extensions: the EVEX encodings (masking, zero-masking,
    /// embedded broadcast, registers 16-31) applied to XMM/YMM rather than ZMM
    /// only. Required from [`Avx512Tier::Tier2`] up.
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
    /// is why no [`Avx512Tier`] requires it; part of the AVX10.1 baseline.
    pub avx512fp16: bool,

    // Enumerated among the AVX-512 bits, but independent features -- Zen 3 has
    // VAES and VPCLMULQDQ with AVX2 and no AVX-512 whatsoever (GFNI arrived with
    // Zen 4). Never gated on AVX-512; tier 3 wants their 512-bit forms, which is
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
    /// means **usable** -- zeroed unless the OS saves ZMM/opmask state -- and a
    /// non-zero version implies every `avx512*` flag above is set (see
    /// [`features`]).
    pub avx10_version: u8,
}

impl Features {
    /// The highest AVX-512 tier this CPU satisfies, matching the `avx512-tier1..4`
    /// crate features and the `arch::tiers::tierN` intrinsic modules in
    /// `backend/x86.rs` **exactly** -- the tier ladder is defined there, and this
    /// only reports which rung the hardware reaches.
    ///
    /// An AVX10 part always reports [`Avx512Tier::Tier4`]: AVX10.1 subsumes the
    /// whole ladder, and [`features`] folds that guarantee into the individual
    /// flags this reads.
    pub fn avx512_tier(&self) -> Option<Avx512Tier> {
        // tier1: F + CD
        if !(self.avx512f && self.avx512cd) {
            return None;
        }
        // tier2: + BW + DQ + VL (Skylake-SP shipped the three together, and no
        // CPU has BW/DQ without VL -- only Knights Landing lacked all three).
        if !(self.avx512bw && self.avx512dq && self.avx512vl) {
            return Some(Avx512Tier::Tier1);
        }
        // tier3: + VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, IFMA, GFNI, VAES, VPCLMULQDQ
        let tier3 = self.avx512vbmi
            && self.avx512vbmi2
            && self.avx512vnni
            && self.avx512bitalg
            && self.avx512vpopcntdq
            && self.avx512ifma
            && self.gfni
            && self.vaes
            && self.vpclmulqdq;
        if !tier3 {
            return Some(Avx512Tier::Tier2);
        }
        // tier4: + BF16
        if !self.avx512bf16 {
            return Some(Avx512Tier::Tier3);
        }
        Some(Avx512Tier::Tier4)
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

/// Probe the CPU. Costs a few `cpuid`s; callers cache the result.
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
    // CPUID.1:ECX[27] reports; guarded above.
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
        // Subleaf 0's EAX reports the max subleaf; subleaf 1 carries AVX512-BF16
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
        // 7:1 EDX[19] only says leaf 0x24 is valid; the capability itself is
        // that leaf's converged version number. The 256-bit-max option (and
        // with it the vector-length enumeration in 0x24 EBX[18:16]) was
        // dropped from the spec in rev 2.0 -- AVX10 always means all three
        // widths -- so the length bits are deliberately not consulted: the SDM
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
        // into the individual flags. On every shipped part this is a no-op --
        // the legacy bits are still enumerated alongside AVX10 -- but the
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

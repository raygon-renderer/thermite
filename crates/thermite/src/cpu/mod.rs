//! Runtime facts about the machine: cache geometry, core topology, and hybrid
//! core type.
//!
//! Everything here is **reported by the hardware or the OS, never estimated**.
//! No value is inferred from a nominal core clock, measured against a wall
//! clock, or guessed from a model-number table. If the machine does not say,
//! the answer is `None`.
//!
//! [`quirks`] is the deliberate exception, and is kept in its own module for
//! exactly that reason: which instructions a CPU implements in *microcode* is
//! not enumerated anywhere, by any vendor, so that table is guessed from
//! family and model. Its values are performance hints and nothing branches on
//! them for correctness.
//!
//! This is deliberately *not* on [`NativeIsa`](crate::simd::NativeIsa). Nothing
//! here varies by backend (`rdtsc` is the same instruction whether the caller
//! is running SSE2 or AVX2 kernels), it varies by **target and host**, so it
//! lives in one place and every backend sees the same answer.
//!
//! # Platform coverage
//!
//! Only x86 can interrogate itself with a plain user-space instruction
//! (`cpuid`), so it is the only target that fills this in without help. On
//! every other ISA the identification registers are privileged (aarch64's
//! `ID_AA64*`/`CCSIDR_EL1` are EL1) and the real source is the OS.
//!
//! | | Cache | Topology | Core type |
//! |---|---|---|---|
//! | x86 / x86_64 (any OS, `no_std`) | `cpuid` 4 / `0x8000001D` | `cpuid` `0x1F`/`0xB`/`0x8000001E` (+ `available_parallelism` under `std`) | `cpuid` `0x1A`, live per-core |
//! | aarch64 macOS **and iOS** | `sysctl hw.perflevelN.*` | `sysctl hw.perflevelN.*` | P/E **counts** only |
//! | aarch64 Linux / **Android** (`std`) | sysfs `cache/index*` | sysfs `topology/` + `cpu_capacity` | capacity split |
//! | aarch64 elsewhere | `CTR_EL0` line size only | -- | -- |
//! | wasm / SPIR-V | -- | -- | -- |
//!
//! Both mobile platforms ship hybrid CPUs, and neither answers "which core am I
//! on" the way x86 does. iOS gives the P/E *counts* but expects you to express
//! intent through a QoS class rather than query placement; Android's sysfs is
//! usually blocked by SELinux for a sandboxed app. See
//! [`aarch64`](self#modules) for what each tier can and cannot reach.
//!
//! Every field is an [`Option`]; `None` means "this target cannot tell us",
//! never "zero". Nothing here is guessed from a model-number table.
//!
//! # Two things that are per-core, not per-machine
//!
//! On a hybrid CPU the answer depends on *which core you are running on*, and
//! threads migrate:
//!
//! * [`current_core_type`] is a **live query** every call (~100 cycles on x86)
//!   and is never cached, because the cached answer would be a lie the moment
//!   the scheduler moved the thread.
//! * [`CpuInfo::get`] caches a snapshot taken on whichever core ran detection
//!   first. On Alder Lake-class parts P and E cores report *different L2 sizes*,
//!   so pin the thread and call [`CpuInfo::detect`] if that distinction matters.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU8, Ordering};

/// x86-specific CPU features, including the AVX-512 tier ladder. Only compiled
/// on x86/x86_64, since nothing in it is meaningful elsewhere.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

/// Apple-specific scheduling control (macOS / iOS): quality-of-service classes,
/// which are how the platform lets you *influence* P-core versus E-core
/// placement given that it will not tell you where a thread is running.
#[cfg(target_vendor = "apple")]
pub mod apple;

#[cfg(target_arch = "aarch64")]
mod aarch64;

/// Which instructions this CPU implements in microcode. The one model-number
/// table in this module. See its own docs for why there is no alternative.
pub mod quirks;

/// What a cache level holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheKind {
    /// Data only (the `L1d` of a split first level).
    Data,
    /// Instructions only (`L1i`).
    Instruction,
    /// Both, in one array (typical for L2/L3).
    Unified,
}

/// Geometry of one cache level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheInfo {
    /// Total size in bytes.
    pub size: u32,
    /// Line size in bytes; `None` if the platform reported a size but no line.
    pub line_size: Option<u32>,
    /// Ways of associativity. `Some(0)` encodes fully associative.
    pub associativity: Option<u16>,
    /// How many logical processors share this cache (1 = private).
    pub shared_by: Option<u16>,
    /// Data / instruction / unified.
    pub kind: CacheKind,
}

/// Which class of core the code is running on, for hybrid (`big.LITTLE`,
/// Intel P/E, Apple performance/efficiency) designs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CoreType {
    /// Not a hybrid part, or the platform cannot tell us.
    #[default]
    Unknown,
    /// The big/performance core (Intel "Core", Apple `perflevel0`).
    Performance,
    /// The little/efficiency core (Intel "Atom", Apple `perflevel1`).
    Efficiency,
}

/// Core counts. `logical`/`physical` are whole-machine where the OS tells us
/// (`std`), and per-package from `cpuid` otherwise, which differ on a
/// multi-socket box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Topology {
    /// Logical processors (hardware threads).
    pub logical_cores: Option<u16>,
    /// Physical cores.
    pub physical_cores: Option<u16>,
    /// Hardware threads per physical core (2 = SMT/hyperthreading on).
    pub threads_per_core: Option<u16>,
    /// Performance ("big") cores, on a hybrid part.
    pub performance_cores: Option<u16>,
    /// Efficiency ("little") cores, on a hybrid part.
    pub efficiency_cores: Option<u16>,
}

/// A snapshot of the machine. Get one from [`CpuInfo::get`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CpuInfo {
    l1d: Option<CacheInfo>,
    l1i: Option<CacheInfo>,
    l2: Option<CacheInfo>,
    l3: Option<CacheInfo>,
    line_size: Option<u32>,
    writeback_granule: Option<u32>,
    topology: Topology,
    hybrid: bool,
}

impl CpuInfo {
    /// An all-`None` snapshot: what a target that can answer nothing returns.
    pub const UNKNOWN: Self = Self {
        l1d: None,
        l1i: None,
        l2: None,
        l3: None,
        line_size: None,
        writeback_granule: None,
        topology: Topology {
            logical_cores: None,
            physical_cores: None,
            threads_per_core: None,
            performance_cores: None,
            efficiency_cores: None,
        },
        hybrid: false,
    };

    /// The cached snapshot, detected once on first call.
    ///
    /// See the [module docs](self) for why this is a snapshot rather than a
    /// live view on hybrid parts.
    #[inline]
    pub fn get() -> &'static CpuInfo {
        static CACHE: Cache<CpuInfo> = Cache {
            state: AtomicU8::new(UNINIT),
            value: UnsafeCell::new(CpuInfo::UNKNOWN),
        };

        CACHE.get(CpuInfo::detect)
    }

    /// Detect **now**, bypassing the cache. Pin the thread first if you are on
    /// a hybrid part and care which core answers.
    pub fn detect() -> CpuInfo {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            x86::detect()
        }
        #[cfg(target_arch = "aarch64")]
        {
            aarch64::detect()
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuInfo::UNKNOWN
        }
    }

    /// Level 1 data cache.
    #[inline]
    pub fn l1d(&self) -> Option<CacheInfo> {
        self.l1d
    }

    /// Level 1 instruction cache.
    #[inline]
    pub fn l1i(&self) -> Option<CacheInfo> {
        self.l1i
    }

    /// Level 2 cache.
    #[inline]
    pub fn l2(&self) -> Option<CacheInfo> {
        self.l2
    }

    /// Level 3 (last-level) cache. `None` on Apple Silicon: the M-series system
    /// level cache is not reported as an L3 and is not a CPU cache.
    #[inline]
    pub fn l3(&self) -> Option<CacheInfo> {
        self.l3
    }

    /// Cache line size in bytes -- 64 on x86 and most aarch64, **128 on Apple
    /// silicon**. The number to align hot structures to and to stride prefetches
    /// by. See [`crate::backend::prefetch`].
    #[inline]
    pub fn cache_line_size(&self) -> Option<u32> {
        self.line_size
    }

    /// Cache writeback granule: the span two cores can contend over, which is
    /// the correct padding to avoid false sharing. From `CTR_EL0.CWG` on
    /// aarch64; equal to the line size on x86.
    #[inline]
    pub fn writeback_granule(&self) -> Option<u32> {
        self.writeback_granule
    }

    /// Core counts.
    #[inline]
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// Whether this CPU mixes performance and efficiency cores.
    #[inline]
    pub fn is_hybrid(&self) -> bool {
        self.hybrid
    }
}

pub(crate) const UNINIT: u8 = 0;
const BUSY: u8 = 1;
const READY: u8 = 2;

/// Write-once cell, in the same shape as the ISA detector
/// (`isa/x86_detector.rs`): a state machine in an atomic guarding a single
/// publish, rather than a lock.
///
/// Generic over the payload so the snapshot and the [`quirks`] table share one
/// implementation, since both are "run a short `cpuid` sequence once, publish the
/// result forever", and a second hand-rolled copy of this is exactly the kind
/// of thing that acquires a subtle ordering bug in only one of its versions.
pub(crate) struct Cache<T: 'static> {
    pub(crate) state: AtomicU8,
    pub(crate) value: UnsafeCell<T>,
}

// SAFETY: `value` is written exactly once, by whichever thread wins the CAS to
// `BUSY`, and is only ever read after an `Acquire` load observes `READY`,
// which synchronizes with that writer's `Release` store.
unsafe impl<T: Send> Sync for Cache<T> {}

impl<T> Cache<T> {
    /// The cached value, running `detect` exactly once across all threads.
    #[inline]
    pub(crate) fn get(&'static self, detect: fn() -> T) -> &'static T {
        // Fast path: already published by whoever won the race.
        if self.state.load(Ordering::Acquire) != READY {
            self.init(detect);
        }

        // SAFETY: the state is `READY`, reached through an `Acquire` load that
        // synchronizes with the writer's `Release` store, so the write has
        // completed and no writer can still be running. Nothing mutates it again.
        unsafe { &*self.value.get() }
    }

    #[inline(never)]
    fn init(&self, detect: fn() -> T) {
        match self
            .state
            .compare_exchange(UNINIT, BUSY, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                let detected = detect();
                // SAFETY: the CAS made this thread the unique writer, and no
                // reader can observe the cell until the store below publishes it.
                unsafe { *self.value.get() = detected };
                self.state.store(READY, Ordering::Release);
            }
            // Another thread is detecting. It is a short, lock-free, non-blocking
            // job (a handful of `cpuid`s), so spin rather than park.
            Err(BUSY) => {
                while self.state.load(Ordering::Acquire) != READY {
                    core::hint::spin_loop();
                }
            }
            // Already `READY`.
            Err(_) => {}
        }
    }
}

/// The core type of the core executing this call, queried **live**.
///
/// Never cached: a thread can be migrated between a P and an E core at any
/// moment, so a stored answer goes stale silently. Costs a `cpuid` (~100
/// cycles, more under a hypervisor).
///
/// Only x86 can answer this. Every aarch64 platform returns
/// [`CoreType::Unknown`], for platform-specific reasons rather than an omission
/// here:
///
/// * **macOS / iOS** report how many performance and efficiency cores *exist*
///   ([`Topology`]) but expose no user-space way to ask which one is running the
///   calling thread. Apple's model is that you declare intent with a QoS class
///   and the scheduler places the work.
/// * **Linux / Android** could answer it in principle, with `sched_getcpu()` plus a
///   per-CPU capacity table, but the capacity table lives in sysfs, which a
///   sandboxed Android app cannot read. The alternative, reading `MIDR_EL1`
///   directly, is a SIGILL-trap emulation that kills the process on kernels
///   lacking it.
///
/// On a hybrid aarch64 part, use [`Topology::performance_cores`] /
/// [`Topology::efficiency_cores`] to learn the machine's shape, and pin the
/// thread if you need to *control* where work runs.
#[inline]
pub fn current_core_type() -> CoreType {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        x86::current_core_type()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        CoreType::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever a target reports has to be self-consistent: no zero sizes, line
    /// sizes a power of two, L1 <= L2 <= L3, counts that multiply out.
    #[test]
    fn snapshot_is_coherent() {
        let info = CpuInfo::get();

        for c in [info.l1d(), info.l1i(), info.l2(), info.l3()].into_iter().flatten() {
            assert!(c.size > 0, "cache reported with zero size: {c:?}");
            if let Some(line) = c.line_size {
                assert!(
                    line.is_power_of_two() && (16..=256).contains(&line),
                    "implausible line size {line}"
                );
            }
        }

        if let (Some(l1d), Some(l2)) = (info.l1d(), info.l2()) {
            assert!(l1d.size <= l2.size, "L1d {} > L2 {}", l1d.size, l2.size);
        }
        if let (Some(l2), Some(l3)) = (info.l2(), info.l3()) {
            assert!(l2.size <= l3.size, "L2 {} > L3 {}", l2.size, l3.size);
        }

        if let Some(line) = info.cache_line_size() {
            assert!(
                line.is_power_of_two() && (16..=256).contains(&line),
                "implausible line size {line}"
            );
        }

        let topo = info.topology();
        if let (Some(l), Some(p)) = (topo.logical_cores, topo.physical_cores) {
            assert!(l >= p, "logical {l} < physical {p}");
        }
        if let (Some(p), Some(e), Some(total)) = (topo.performance_cores, topo.efficiency_cores, topo.logical_cores) {
            assert!(p + e <= total, "P {p} + E {e} exceeds {total} logical");
        }
        if let Some(t) = topo.threads_per_core {
            assert!((1..=8).contains(&t), "implausible threads/core {t}");
        }
    }

    /// The cache is a cache: repeated `get()` must not re-detect or change.
    #[test]
    fn snapshot_is_stable() {
        let a = *CpuInfo::get();
        let b = *CpuInfo::get();
        assert_eq!(a, b);
        assert!(core::ptr::eq(CpuInfo::get(), CpuInfo::get()));
    }

    /// Both x86 tiers must agree, and both must be self-consistent with the
    /// cached snapshot, since this is the same machine either way.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_fills_in_the_basics() {
        let info = CpuInfo::get();

        assert!(info.cache_line_size().is_some(), "x86 always reports a line size");
        assert!(info.l1d().is_some(), "x86 always enumerates L1d");
        assert_eq!(info.cache_line_size(), info.writeback_granule());

        // Re-detecting must agree, except on a hybrid part, where the thread
        // may have been migrated to a core with a different L2 in between.
        if !info.is_hybrid() {
            assert_eq!(*info, CpuInfo::detect(), "uncached detect disagrees with the snapshot");
        }
    }

    /// The dispatcher's hand-rolled `cpuid` must agree with the ISA the crate
    /// was actually compiled to run on: if the build enabled a feature
    /// statically, detection has to see it too.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_features_agree_with_build() {
        let f = crate::cpu::x86::features();

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
        let f = crate::cpu::x86::features();

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
        use crate::cpu::x86::{Avx512Tier, Features};

        let f = crate::cpu::x86::features();

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
        use crate::cpu::x86::{Avx10Version, Avx512Tier, Features};

        let f = crate::cpu::x86::features();

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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_reads_ctr_el0() {
        let info = CpuInfo::get();

        // CTR_EL0 is readable at EL0 on every aarch64, with or without an OS.
        assert!(info.cache_line_size().is_some(), "CTR_EL0 line size unavailable");

        // CWG is genuinely optional: the architecture allows 0 for "not
        // specified" and qemu-user reports exactly that, so `None` is the
        // correct answer there rather than a bogus 2048-byte padding hint.
        if let Some(cwg) = info.writeback_granule() {
            assert!(
                cwg.is_power_of_two() && (16..=2048).contains(&cwg),
                "implausible CWG {cwg}"
            );
        }
    }
}

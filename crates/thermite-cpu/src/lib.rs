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
//! This is deliberately *not* on [`NativeIsa`](thermite::simd::NativeIsa). Nothing
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

#![no_std]

#[cfg(feature = "std")]
extern crate std;

use thermite::isa::DetectOnce;

/// x86 cache geometry, topology and live core type via `cpuid`. Only compiled
/// on x86/x86_64; the feature bits live in `thermite::isa::x86`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

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
        static CACHE: DetectOnce<CpuInfo> = DetectOnce::new(CpuInfo::UNKNOWN);

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
    /// by. See [`thermite::backend::prefetch`].
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

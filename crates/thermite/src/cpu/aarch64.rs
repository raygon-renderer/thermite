//! aarch64 machine facts.
//!
//! The identification registers are EL1 (`ID_AA64*`, `CCSIDR_EL1`, `MIDR_EL1`),
//! so unlike x86 there is no user-space instruction that describes the machine.
//! Three tiers, in the order this module tries them:
//!
//! 1. **`CTR_EL0`** -- genuinely readable at EL0 on every aarch64, OS or not.
//!    Gives the cache *line* size and the writeback granule. This is the whole
//!    story in `no_std`.
//! 2. **macOS `sysctl`** -- Apple silicon reports per-perflevel cache sizes and
//!    P/E core counts, the richest of any platform here.
//! 3. **Linux / Android sysfs** (`std` only) -- `cache/index*` for geometry,
//!    `topology/` for SMT, `cpu_capacity` for the big.LITTLE split.
//!
//! Cache *sizes* are unreachable without 2 or 3: `CCSIDR_EL1` traps to EL1.
//!
//! # Mobile
//!
//! Both mobile platforms are hybrid, and they are *not* equally forthcoming:
//!
//! * **iOS** rides the same `target_vendor = "apple"` sysctl path as macOS, so
//!   it reports caches and the P/E core counts. What it will not tell you is
//!   which core the calling thread is on -- Apple's model is that you express
//!   intent with a QoS class and the scheduler places the work.
//! * **Android** is far more restricted. `untrusted_app` cannot read most of
//!   `/sys`, so the sysfs path frequently yields nothing.
//!
//! Deliberately *not* attempted: reading `MIDR_EL1` with a bare `mrs` to
//! identify the current core. Linux does emulate that access, but by **trapping
//! a SIGILL** -- so it costs a trap per read, and on any kernel without the
//! emulation it does not return a wrong answer, it kills the process. The
//! sysfs mirror (`regs/identification/midr_el1`) is the safe way to the same
//! data, and it is subject to the same SELinux limits as everything else here.

use super::CpuInfo;

/// `CTR_EL0`: cache type register, readable at EL0 by default.
///
/// Line/granule fields are `log2` of a count of **words** (4 bytes), so each
/// decodes as `4 << field`.
#[inline]
fn ctr_el0() -> u64 {
    let value: u64;
    // SAFETY: unprivileged read of an ID register. `pure`+`nomem` are accurate:
    // it touches no memory and never changes, so the compiler may hoist or CSE it.
    unsafe {
        core::arch::asm!("mrs {}, ctr_el0", out(reg) value, options(nostack, nomem, preserves_flags, pure));
    }
    value
}

pub fn detect() -> CpuInfo {
    let mut info = CpuInfo::UNKNOWN;

    // --- tier 1: what the hardware will tell EL0 directly ----------------
    let ctr = ctr_el0();

    // DminLine [19:16]: smallest data cache line, in log2(words).
    info.line_size = Some(4 << ((ctr >> 16) & 0xf));

    // CWG [27:24]: cache writeback granule -- the span two cores contend over,
    // so this and not the line size is the right false-sharing padding. Zero
    // means "not specified", in which case the architecture says to assume the
    // maximum (2048), which is useless as padding, so report nothing instead.
    let cwg = (ctr >> 24) & 0xf;
    info.writeback_granule = (cwg != 0).then(|| 4u32 << cwg);

    // --- tiers 2 and 3: everything that needs the OS ---------------------
    #[cfg(target_vendor = "apple")]
    apple::fill(&mut info);

    #[cfg(all(any(target_os = "linux", target_os = "android"), feature = "std"))]
    linux::fill(&mut info);

    #[cfg(feature = "std")]
    if info.topology.logical_cores.is_none() {
        info.topology.logical_cores = std::thread::available_parallelism()
            .ok()
            .and_then(|n| u16::try_from(n.get()).ok());
    }

    info
}

/// macOS / iOS, including Apple silicon.
///
/// `sysctlbyname` lives in libSystem, which every Apple target links
/// unconditionally, so this needs neither `std` nor a `libc` dependency.
#[cfg(target_vendor = "apple")]
mod apple {
    use crate::cpu::{CacheInfo, CacheKind, CpuInfo};
    use core::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// Read an integer sysctl by name. `name` must be NUL-terminated.
    ///
    /// Keys are a mix of 32- and 64-bit; asking for 8 bytes and letting the
    /// kernel report back a smaller length handles both, since the buffer is
    /// zeroed and Apple targets are little-endian.
    fn sysctl(name: &[u8]) -> Option<u64> {
        debug_assert_eq!(name.last(), Some(&0), "sysctl name must be NUL-terminated");

        let mut value = 0u64;
        let mut len = size_of::<u64>();

        // SAFETY: `name` is NUL-terminated, and the out-buffer/length pair
        // describes `value` exactly. The kernel writes at most `len` bytes.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr().cast::<c_char>(),
                (&raw mut value).cast::<c_void>(),
                &raw mut len,
                core::ptr::null_mut(),
                0,
            )
        };

        // A missing key (e.g. `hw.l3cachesize` on Apple silicon) returns -1.
        (rc == 0 && value != 0).then_some(value)
    }

    fn cache(size: Option<u64>, line: Option<u32>, kind: CacheKind) -> Option<CacheInfo> {
        Some(CacheInfo {
            size: u32::try_from(size?).ok()?,
            line_size: line,
            associativity: None, // not exposed by sysctl
            shared_by: None,
            kind,
        })
    }

    pub fn fill(info: &mut CpuInfo) {
        // Prefer `hw.cachelinesize` over CTR_EL0: on Apple silicon the register
        // reports 64 while the actual line -- and what every Apple tool, and
        // `hw.cachelinesize`, reports -- is 128.
        if let Some(line) = sysctl(b"hw.cachelinesize\0").and_then(|v| u32::try_from(v).ok()) {
            info.line_size = Some(line);
            info.writeback_granule.get_or_insert(line);
        }
        let line = info.line_size;

        // The flat keys describe the performance cores on a hybrid part; the
        // `perflevel1.*` equivalents describe the efficiency cores.
        info.l1d = cache(sysctl(b"hw.l1dcachesize\0"), line, CacheKind::Data);
        info.l1i = cache(sysctl(b"hw.l1icachesize\0"), line, CacheKind::Instruction);
        info.l2 = cache(sysctl(b"hw.l2cachesize\0"), line, CacheKind::Unified);
        // Absent on Apple silicon: the M-series system level cache is not a CPU
        // cache and is deliberately not reported as an L3.
        info.l3 = cache(sysctl(b"hw.l3cachesize\0"), line, CacheKind::Unified);

        let u16_of = |key: &[u8]| sysctl(key).and_then(|v| u16::try_from(v).ok());

        info.topology.logical_cores = u16_of(b"hw.logicalcpu\0");
        info.topology.physical_cores = u16_of(b"hw.physicalcpu\0");

        if let (Some(logical), Some(physical)) = (info.topology.logical_cores, info.topology.physical_cores)
            && physical > 0
        {
            info.topology.threads_per_core = Some(logical / physical);
        }

        // Apple silicon: perflevel0 is the fastest cluster (P), perflevel1 the
        // efficiency cluster (E). An Intel Mac reports a single perflevel.
        if sysctl(b"hw.nperflevels\0").unwrap_or(1) > 1 {
            info.hybrid = true;
            info.topology.performance_cores = u16_of(b"hw.perflevel0.logicalcpu\0");
            info.topology.efficiency_cores = u16_of(b"hw.perflevel1.logicalcpu\0");
        }
    }
}

/// Linux **and Android**, via sysfs. Needs `std` for the filesystem.
///
/// Android is a separate `target_os` in Rust, not a flavour of `linux`, so it
/// has to be named explicitly or it silently gets nothing.
///
/// Every read fails soft, which matters more here than on desktop Linux: an
/// Android app runs in the `untrusted_app` SELinux domain, where most of
/// `/sys` is denied outright. Expect `None` in a sandboxed app and real values
/// from a native binary, a system app, or `adb shell`.
#[cfg(all(any(target_os = "linux", target_os = "android"), feature = "std"))]
mod linux {
    use crate::cpu::{CacheInfo, CacheKind, CpuInfo};

    fn read(path: &str) -> Option<std::string::String> {
        std::fs::read_to_string(path).ok().map(|s| s.trim().into())
    }

    /// sysfs cache sizes are written `32K` / `1M`.
    fn parse_size(text: &str) -> Option<u32> {
        let (digits, scale) = match text.as_bytes().last()? {
            b'K' => (&text[..text.len() - 1], 1024),
            b'M' => (&text[..text.len() - 1], 1024 * 1024),
            b'G' => (&text[..text.len() - 1], 1024 * 1024 * 1024),
            _ => (text, 1),
        };
        digits.parse::<u32>().ok()?.checked_mul(scale)
    }

    /// Count CPUs in a `0-3,8` style mask list.
    fn count_cpu_list(text: &str) -> u16 {
        text.split(',')
            .filter_map(|part| match part.split_once('-') {
                Some((lo, hi)) => Some(hi.parse::<u16>().ok()? - lo.parse::<u16>().ok()? + 1),
                None => part.parse::<u16>().ok().map(|_| 1),
            })
            .sum()
    }

    pub fn fill(info: &mut CpuInfo) {
        for index in 0..10 {
            let dir = std::format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
            let Some(level) = read(&std::format!("{dir}/level")).and_then(|s| s.parse::<u8>().ok()) else {
                break; // no more cache levels described
            };

            let Some(size) = read(&std::format!("{dir}/size")).and_then(|s| parse_size(&s)) else {
                continue;
            };

            let kind = match read(&std::format!("{dir}/type")).as_deref() {
                Some("Data") => CacheKind::Data,
                Some("Instruction") => CacheKind::Instruction,
                _ => CacheKind::Unified,
            };

            let entry = CacheInfo {
                size,
                line_size: read(&std::format!("{dir}/coherency_line_size")).and_then(|s| s.parse().ok()),
                associativity: read(&std::format!("{dir}/ways_of_associativity")).and_then(|s| s.parse().ok()),
                shared_by: read(&std::format!("{dir}/shared_cpu_list")).map(|s| count_cpu_list(&s)),
                kind,
            };

            match (level, kind) {
                (1, CacheKind::Instruction) => info.l1i = Some(entry),
                (1, _) => info.l1d = Some(entry),
                (2, _) => info.l2 = Some(entry),
                (3, _) => info.l3 = Some(entry),
                _ => {}
            }
        }

        if let Some(siblings) = read("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") {
            let threads = count_cpu_list(&siblings);
            if threads > 0 {
                info.topology.threads_per_core = Some(threads);
            }
        }

        let logical = std::thread::available_parallelism()
            .ok()
            .and_then(|n| u16::try_from(n.get()).ok());
        info.topology.logical_cores = logical;

        if let (Some(logical), Some(per_core)) = (logical, info.topology.threads_per_core)
            && per_core > 0
        {
            info.topology.physical_cores = Some(logical / per_core);
        }

        // big.LITTLE: the scheduler's per-CPU capacity (1024 = the biggest core
        // on the machine). Differing values are exactly what "hybrid" means here.
        if let Some(count) = logical {
            let mut capacities = std::vec::Vec::with_capacity(usize::from(count));
            for cpu in 0..count {
                let path = std::format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
                match read(&path).and_then(|s| s.parse::<u32>().ok()) {
                    Some(capacity) => capacities.push(capacity),
                    // Not every kernel/arch exports capacities; all or nothing.
                    None => {
                        capacities.clear();
                        break;
                    }
                }
            }

            if let Some(&max) = capacities.iter().max()
                && capacities.iter().any(|&c| c != max)
            {
                let big = capacities.iter().filter(|&&c| c == max).count() as u16;
                info.hybrid = true;
                info.topology.performance_cores = Some(big);
                info.topology.efficiency_cores = Some(count - big);
            }
        }
    }
}

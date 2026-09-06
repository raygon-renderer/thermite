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
//!
//! The feature bits (SSE..AVX10) live in [`thermite::isa::x86`], which is also
//! where the `cpuid` wrapper and vendor/family helpers used here come from.
//!
//! Leaves are tried and _checked for an empty answer_ instead of only bounded by
//! the reported maximum: a CPU can advertise a max leaf above one it does not
//! implement, in which case it returns zeros (see `read_topology_amd`).
//!
//! `cpuid` is serializing (100-250 cycles bare metal, a VM exit under a
//! hypervisor), which is why the caller caches the result.

use super::{CacheInfo, CacheKind, CoreType, CpuInfo};
use thermite::isa::x86::{cpuid, family_model, is_amd_lineage, max_leaves};

#[inline]
fn bit(value: u32, index: u32) -> bool {
    (value >> index) & 1 != 0
}

/// CPU family, with the extended-family field folded in per the x86 rules.
#[inline]
fn family() -> u32 {
    family_model().0
}

/// Walk the deterministic-cache-parameter leaf. Intel uses `4`, AMD uses the
/// identically-formatted `0x8000001D` (older AMD reported nothing here, which
/// leaves the levels `None`).
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
/// processors _at and below_ that level, so the SMT level gives threads-per-core
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
/// while implementing neither `0xB` nor `0x1F`, both of which return all zeros, so
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

    // EBX[15:8] is threads-per-core minus one, but only from family 0x17
    // (Zen). Family 0x15 advertises the leaf with a _non-zero_ SMT field that
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
            // x86 has no separate writeback granule, and a line is the unit of
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
    // 0x1F supersedes 0xB, and either may be _present but empty_ (AMD reports a
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
        // `max_ext`, and a non-AMD CPU that does not implement them
        // reports nothing rather than garbage.
        topology = read_topology_amd(max_ext);
    }
    let (threads_per_core, logical_per_package) = topology;

    info.topology.threads_per_core = threads_per_core;

    // The OS knows the whole machine (and honours affinity masks / cgroup
    // limits). `cpuid` only ever describes one package.
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
    // P/E _counts_ need every core interrogated in turn (each `cpuid` describes
    // only the core it ran on), which means pinning threads. Deliberately left
    // `None` rather than guessed. `current_core_type()` answers for this core.

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

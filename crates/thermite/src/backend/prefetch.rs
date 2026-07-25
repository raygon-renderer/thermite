//! Software prefetch: one per-architecture implementation shared by every backend.
//!
//! Prefetch is an ISA-level *memory* hint rather than a lane operation -- the
//! same instruction is issued whether the surrounding code is running scalar,
//! SSE2 or AVX2 kernels -- so every [`NativeIsa::prefetch`] impl routes here
//! instead of re-deriving it per tier. The x86 path is re-exported through
//! `backend::x86::sse` and the aarch64 path through the NEON polyfills, so
//! backend code can reach it as `arch::prefetch` like any other primitive.
//!
//! # Parameterization
//!
//! The knobs follow the LLVM / `__builtin_prefetch` convention rather than any
//! one ISA's encoding, because it is the only one every target can express:
//!
//! * `LOCALITY` -- how long the line should be kept: `0` = none (streaming,
//!   evict as soon as possible), `1` = low, `2` = moderate, `3` = high (keep it
//!   in every cache level). Outside `0..=3` is a compile error.
//! * `WRITE` -- `true` when the line is about to be *written*, letting the
//!   hardware fetch it in an exclusive/owned state and skip the later
//!   read-for-ownership. Targets with no write form fall back to the read form;
//!   it is only ever a hint.
//!
//! | `LOCALITY` | x86 (read / write) | aarch64 (read / write) |
//! |---|---|---|
//! | 3 | `prefetcht0` / `prefetchw` | `prfm pldl1keep` / `prfm pstl1keep` |
//! | 2 | `prefetcht1` / `prefetchw`(t1) | `prfm pldl2keep` / `prfm pstl2keep` |
//! | 1 | `prefetcht2` / `prefetchw`(t1) | `prfm pldl3keep` / `prfm pstl3keep` |
//! | 0 | `prefetchnta` | `prfm pldl1strm` / `prfm pstl1strm` |
//!
//! The x86 write forms need `prfchw` (or `prefetchwt1`) to be enabled at compile
//! time; without them LLVM quietly lowers the write hints back to `prefetcht0` /
//! `prefetcht1`, which is exactly the right degradation for a hint. Verified in
//! emitted assembly at every `LOCALITY`, both directions, on x86-64 (with and
//! without `+prfchw`) and aarch64.
//!
//! # Safety
//!
//! A prefetch is architecturally invisible: the address is never dereferenced,
//! never faults and never traps, so [`prefetch`] is a **safe** function that
//! accepts any pointer -- dangling, null, unaligned or wildly out of bounds.
//! That is what makes the branchless `base.wrapping_add(i)` idiom (compute the
//! address of the *next* iteration's data without bounds-checking it first)
//! usable in a hot loop, and it keeps Miri quiet because no provenance is ever
//! used.
//!
//! [`NativeIsa::prefetch`]: crate::simd::NativeIsa::prefetch

/// `true` when [`prefetch`] lowers to a real instruction on the target being
/// compiled for, `false` when it compiles away to nothing (wasm, SPIR-V, and
/// any architecture without a software-prefetch hint). Mirrored by
/// [`NativeIsa::HAS_PREFETCH`](crate::simd::NativeIsa::HAS_PREFETCH).
pub const HAS_PREFETCH: bool = cfg!(any(
    target_arch = "x86_64",
    all(target_arch = "x86", target_feature = "sse"),
    target_arch = "aarch64",
));

/// Hint that the cache line containing `ptr` should be fetched now, ahead of
/// the access that actually needs it. See the [module docs](self) for the
/// meaning of `LOCALITY`/`WRITE` and the per-target lowering.
///
/// Safe for *any* pointer value: nothing is read, so nothing can fault.
#[inline(always)]
pub fn prefetch<const LOCALITY: u8, const WRITE: bool>(ptr: *const u8) {
    const {
        assert!(
            LOCALITY <= 3,
            "prefetch LOCALITY must be 0 (non-temporal) ..= 3 (keep in every cache level)"
        )
    };

    cfg_if::cfg_if! {
        if #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            any(target_arch = "x86_64", target_feature = "sse"),
        ))] {
            use crate::backend::x86::sse::{
                _MM_HINT_ET0, _MM_HINT_ET1, _MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch,
            };

            let ptr = ptr.cast::<i8>();

            // `_mm_prefetch` is a safe intrinsic (it says so on the tin: "safe to use
            // even though it takes a raw pointer"); the `unsafe` block is only the
            // `target_feature(enable = "sse")` obligation, satisfied by the cfg above.
            // LOCALITY/WRITE are const, so this collapses to a single instruction.
            unsafe {
                match const { (WRITE, LOCALITY) } {
                    (false, 3) => _mm_prefetch::<_MM_HINT_T0>(ptr),
                    (false, 2) => _mm_prefetch::<_MM_HINT_T1>(ptr),
                    (false, 1) => _mm_prefetch::<_MM_HINT_T2>(ptr),
                    (true, 3) => _mm_prefetch::<_MM_HINT_ET0>(ptr),
                    // No `prefetchwt2`: both mid tiers share the one write-intent
                    // encoding below T0. LLVM demotes it to a read prefetch on CPUs
                    // without PRFCHW/PREFETCHWT1.
                    (true, 2 | 1) => _mm_prefetch::<_MM_HINT_ET1>(ptr),
                    // LOCALITY == 0, either direction: x86 has no write form of
                    // `prefetchnta`, and streaming beats write-intent here.
                    _ => _mm_prefetch::<_MM_HINT_NTA>(ptr),
                }
            }
        } else if #[cfg(target_arch = "aarch64")] {
            // `core::arch::aarch64::_prefetch` is still nightly-only
            // (`stdarch_aarch64_prefetch`), and the instruction takes its operation
            // as part of the mnemonic, so go straight to `prfm`. `nomem` is honest:
            // a prefetch has no architecturally visible effect on memory, which lets
            // LLVM schedule it freely without letting it delete the hint.
            macro_rules! prfm {
                ($op:literal) => {
                    unsafe {
                        core::arch::asm!(
                            concat!("prfm ", $op, ", [{ptr}]"),
                            ptr = in(reg) ptr,
                            options(nostack, preserves_flags, nomem),
                        )
                    }
                };
            }

            match const { (WRITE, LOCALITY) } {
                (false, 3) => prfm!("pldl1keep"),
                (false, 2) => prfm!("pldl2keep"),
                (false, 1) => prfm!("pldl3keep"),
                (false, _) => prfm!("pldl1strm"),
                (true, 3) => prfm!("pstl1keep"),
                (true, 2) => prfm!("pstl2keep"),
                (true, 1) => prfm!("pstl3keep"),
                (true, _) => prfm!("pstl1strm"),
            }
        } else {
            let _ = ptr; // no software-prefetch hint on this target
        }
    }
}

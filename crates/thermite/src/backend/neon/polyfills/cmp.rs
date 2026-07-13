use super::*;

// NEON has no 64-bit lane min/max instructions (`vminq_s64`/`vmaxq_s64` do not
// exist); compare + bitwise-select is the canonical two-instruction fill.
// aarch64 *does* have native 64-bit compares (`vcgtq_s64`/`vcgtq_u64`), unlike
// 32-bit NEON and SSE2.

#[inline(always)]
pub fn neon_min_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    unsafe { vbslq_s64(vcgtq_s64(a, b), b, a) }
}

#[inline(always)]
pub fn neon_max_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    unsafe { vbslq_s64(vcgtq_s64(a, b), a, b) }
}

#[inline(always)]
pub fn neon_min_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe { vbslq_u64(vcgtq_u64(a, b), b, a) }
}

#[inline(always)]
pub fn neon_max_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe { vbslq_u64(vcgtq_u64(a, b), a, b) }
}

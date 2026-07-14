//! Radix-3 register de-interleave/interleave for 256-bit registers.
//!
//! AVX2 has no cross-lane 32-bit two-source shuffle, so the 128-bit sequences in
//! [`x86_v1::polyfills::interleave`](crate::backend::x86_v1::polyfills) cannot
//! simply be widened - `vshufps` picks its immediate once and applies it to both
//! 128-bit lanes independently.
//!
//! That restriction turns out to be exactly what makes this cheap. Number the
//! six 128-bit lanes of `a`, `b`, `c` as `L0..L5`. With `E` elements per lane, a
//! 3-element group repeats every `lcm(3, E) / E == 3` lanes, so `L0 L1 L2` hold a
//! whole number of complete `xyz` triples and so do `L3 L4 L5`. Three
//! `vperm2f128`s regroup the two halves into
//!
//! ```text
//!   P0 = [L0, L3]     P1 = [L1, L4]     P2 = [L2, L5]
//! ```
//!
//! and now *every* 128-bit lane is an independent, identical, self-contained
//! 3-by-`E` de-interleave - precisely the problem the SSE sequence solves, and
//! `vshufps`/`vshufpd` solve both lanes at once. Total: 3 permutes + the lane
//! sequence (5 for 32-bit, 3 for 64-bit), against 21 for the generic gather.
//!
//! Interleaving is the same idea played backwards: run the lane-local interleave
//! first, then three `vperm2f128`s to re-thread `[L0,L3] [L1,L4] [L2,L5]` back
//! into `[L0,L1] [L2,L3] [L4,L5]`.

use super::*;

/// Regroup `a`, `b`, `c` so each 128-bit lane holds a self-contained radix-3
/// problem: `[L0,L3]`, `[L1,L4]`, `[L2,L5]`. Its own inverse is [`restitch_ps`].
#[inline(always)]
unsafe fn regroup_ps(a: __m256, b: __m256, c: __m256) -> (__m256, __m256, __m256) {
    (
        _mm256_permute2f128_ps::<0x30>(a, b), // [a.lo, b.hi] = [L0, L3]
        _mm256_permute2f128_ps::<0x21>(a, c), // [a.hi, c.lo] = [L1, L4]
        _mm256_permute2f128_ps::<0x30>(b, c), // [b.lo, c.hi] = [L2, L5]
    )
}

/// Undo [`regroup_ps`]: `[L0,L3] [L1,L4] [L2,L5]` -> `[L0,L1] [L2,L3] [L4,L5]`.
#[inline(always)]
unsafe fn restitch_ps(p0: __m256, p1: __m256, p2: __m256) -> (__m256, __m256, __m256) {
    (
        _mm256_permute2f128_ps::<0x20>(p0, p1), // [L0, L1]
        _mm256_permute2f128_ps::<0x30>(p2, p0), // [L2, L3]
        _mm256_permute2f128_ps::<0x31>(p1, p2), // [L4, L5]
    )
}

/// Radix-3 de-interleave of eight-lane 32-bit registers: 24 elements of
/// `xyzxyz...` in `a`, `b`, `c` become `xxxxxxxx`, `yyyyyyyy`, `zzzzzzzz`.
/// Three `vperm2f128` + five `vshufps`.
#[inline(always)]
pub unsafe fn _mm256_deinterleave3_ps(a: __m256, b: __m256, c: __m256) -> (__m256, __m256, __m256) {
    let (p0, p1, p2) = regroup_ps(a, b, c);

    // Per 128-bit lane: p0 = x0 y0 z0 x1, p1 = y1 z1 x2 y2, p2 = z2 x3 y3 z3.
    let t0 = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(2, 3, 1, 2) }>(p1, p2); // x2 y2 x3 y3
    let t1 = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(1, 2, 0, 1) }>(p0, p1); // y0 z0 y1 z1

    let x = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(0, 3, 0, 2) }>(p0, t0);
    let y = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 1, 3) }>(t1, t0);
    let z = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(1, 3, 0, 3) }>(t1, p2);

    (x, y, z)
}

/// Radix-3 interleave of eight-lane 32-bit registers - the exact inverse of
/// [`_mm256_deinterleave3_ps`]. Six `vshufps` + three `vperm2f128`.
#[inline(always)]
pub unsafe fn _mm256_interleave3_ps(x: __m256, y: __m256, z: __m256) -> (__m256, __m256, __m256) {
    let xy = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 0, 2) }>(x, y); // x0 x2 y0 y2
    let yz = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(1, 3, 1, 3) }>(y, z); // y1 y3 z1 z3
    let zx = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 1, 3) }>(z, x); // z0 z2 x1 x3

    let p0 = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 0, 2) }>(xy, zx); // x0 y0 z0 x1
    let p1 = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 1, 3) }>(yz, xy); // y1 z1 x2 y2
    let p2 = _mm256_shuffle_ps::<{ MM_SHUFFLE_R!(1, 3, 1, 3) }>(zx, yz); // z2 x3 y3 z3

    restitch_ps(p0, p1, p2)
}

/// Radix-3 de-interleave of four-lane 64-bit registers. Same regrouping, but a
/// 128-bit lane now holds only two elements, so each output element comes from a
/// different register and three `vshufpd`s finish the job.
///
/// After the regroup, per pair of lanes:
/// `p0 = x0 y0 | x2 y2`, `p1 = z0 x1 | z2 x3`, `p2 = y1 z1 | y3 z3`.
/// The `vshufpd` immediate is one bit per output element (low bit of each nibble
/// selects within the source's 128-bit lane), so the constants below are written
/// in binary, LSB = element 0.
#[inline(always)]
pub unsafe fn _mm256_deinterleave3_pd(a: __m256d, b: __m256d, c: __m256d) -> (__m256d, __m256d, __m256d) {
    let p0 = _mm256_permute2f128_pd::<0x30>(a, b);
    let p1 = _mm256_permute2f128_pd::<0x21>(a, c);
    let p2 = _mm256_permute2f128_pd::<0x30>(b, c);

    let x = _mm256_shuffle_pd::<0b1010>(p0, p1); // p0[0] p1[1] | p0[2] p1[3]
    let y = _mm256_shuffle_pd::<0b0101>(p0, p2); // p0[1] p2[0] | p0[3] p2[2]
    let z = _mm256_shuffle_pd::<0b1010>(p1, p2); // p1[0] p2[1] | p1[2] p2[3]

    (x, y, z)
}

/// Radix-3 interleave of four-lane 64-bit registers - the exact inverse of
/// [`_mm256_deinterleave3_pd`].
#[inline(always)]
pub unsafe fn _mm256_interleave3_pd(x: __m256d, y: __m256d, z: __m256d) -> (__m256d, __m256d, __m256d) {
    let p0 = _mm256_shuffle_pd::<0b0000>(x, y); // x0 y0 | x2 y2
    let p1 = _mm256_shuffle_pd::<0b1010>(z, x); // z0 x1 | z2 x3
    let p2 = _mm256_shuffle_pd::<0b1111>(y, z); // y1 z1 | y3 z3

    (
        _mm256_permute2f128_pd::<0x20>(p0, p1),
        _mm256_permute2f128_pd::<0x30>(p2, p0),
        _mm256_permute2f128_pd::<0x31>(p1, p2),
    )
}

#[inline(always)]
pub unsafe fn _mm256_deinterleave3_epi32(a: __m256i, b: __m256i, c: __m256i) -> (__m256i, __m256i, __m256i) {
    let (x, y, z) = _mm256_deinterleave3_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), _mm256_castsi256_ps(c));

    (_mm256_castps_si256(x), _mm256_castps_si256(y), _mm256_castps_si256(z))
}

#[inline(always)]
pub unsafe fn _mm256_interleave3_epi32(x: __m256i, y: __m256i, z: __m256i) -> (__m256i, __m256i, __m256i) {
    let (a, b, c) = _mm256_interleave3_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y), _mm256_castsi256_ps(z));

    (_mm256_castps_si256(a), _mm256_castps_si256(b), _mm256_castps_si256(c))
}

#[inline(always)]
pub unsafe fn _mm256_deinterleave3_epi64(a: __m256i, b: __m256i, c: __m256i) -> (__m256i, __m256i, __m256i) {
    let (x, y, z) = _mm256_deinterleave3_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), _mm256_castsi256_pd(c));

    (_mm256_castpd_si256(x), _mm256_castpd_si256(y), _mm256_castpd_si256(z))
}

#[inline(always)]
pub unsafe fn _mm256_interleave3_epi64(x: __m256i, y: __m256i, z: __m256i) -> (__m256i, __m256i, __m256i) {
    let (a, b, c) = _mm256_interleave3_pd(_mm256_castsi256_pd(x), _mm256_castsi256_pd(y), _mm256_castsi256_pd(z));

    (_mm256_castpd_si256(a), _mm256_castpd_si256(b), _mm256_castpd_si256(c))
}

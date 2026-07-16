//! Radix-3 register de-interleave/interleave for 128-bit registers.
//!
//! The 3-way siblings of `unpcklps`/`unpckhps`, wired into
//! `Register::deinterleave_radix::<3>` via `impl_native_radix3!`. That method's
//! default is an `O(N^2)` permute+blend gather; three registers of `xyzxyz...`
//! have far more structure than that, and a fixed shuffle sequence exploits it:
//! every index here is a compile-time immediate, so these are pure register
//! shuffles with no constant loads, no blends, and no cross-register masks.
//!
//! These are SSE2-only (`shufps`/`shufpd` and the free domain casts), so they
//! are shared by every x86 backend - v2 and v3 re-export this module, and v3
//! builds its 256-bit versions (`_mm256_deinterleave3_ps` and friends) on top of
//! exactly these lane-local sequences.
//!
//! The `_epi32`/`_epi64` variants forward to the float ones through the free
//! `castsi128_ps`/`castsi128_pd` bitcasts - `shufps` on integer data costs at
//! most a 1-cycle domain-crossing bypass on the microarchitectures that still
//! have one, and there is no 2-source integer shuffle to use instead.

use super::*;

/// Radix-3 de-interleave of four-lane 32-bit registers, in five `shufps`.
///
/// Reads `a`, `b`, `c` as one 12-element span and splits it by residue mod 3:
///
/// ```text
///   a = x0 y0 z0 x1     ->  x = x0 x1 x2 x3
///   b = y1 z1 x2 y2         y = y0 y1 y2 y3
///   c = z2 x3 y3 z3         z = z0 z1 z2 z3
/// ```
///
/// Two shuffles first gather the pairs that straddle a register boundary
/// (`t0` = the x/y elements living in `b` and `c`, `t1` = the y/z elements
/// living in `a` and `b`); each output is then one more shuffle away.
#[inline(always)]
pub unsafe fn _mm_deinterleave3_ps(a: __m128, b: __m128, c: __m128) -> (__m128, __m128, __m128) {
    let t0 = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(2, 3, 1, 2) }>(b, c); // x2 y2 x3 y3
    let t1 = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(1, 2, 0, 1) }>(a, b); // y0 z0 y1 z1

    let x = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(0, 3, 0, 2) }>(a, t0);
    let y = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 1, 3) }>(t1, t0);
    let z = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(1, 3, 0, 3) }>(t1, c);

    (x, y, z)
}

/// Radix-3 interleave of four-lane 32-bit registers, in six `shufps` - the
/// exact inverse of [`_mm_deinterleave3_ps`].
///
/// Three shuffles pack the elements into the register pairs each output draws
/// from (`x0 x2 y0 y2`, `y1 y3 z1 z3`, `z0 z2 x1 x3`), and three more emit the
/// interleaved registers.
#[inline(always)]
pub unsafe fn _mm_interleave3_ps(x: __m128, y: __m128, z: __m128) -> (__m128, __m128, __m128) {
    let xy = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 0, 2) }>(x, y); // x0 x2 y0 y2
    let yz = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(1, 3, 1, 3) }>(y, z); // y1 y3 z1 z3
    let zx = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 1, 3) }>(z, x); // z0 z2 x1 x3

    let a = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 0, 2) }>(xy, zx); // x0 y0 z0 x1
    let b = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(0, 2, 1, 3) }>(yz, xy); // y1 z1 x2 y2
    let c = _mm_shuffle_ps::<{ MM_SHUFFLE_R!(1, 3, 1, 3) }>(zx, yz); // z2 x3 y3 z3

    (a, b, c)
}

/// Radix-3 de-interleave of two-lane 64-bit registers, in three `shufpd`:
/// `a = x0 y0`, `b = z0 x1`, `c = y1 z1` -> `x = x0 x1`, `y = y0 y1`,
/// `z = z0 z1`. Each output element is a different one of the three inputs, so
/// no gather step is needed at all.
#[inline(always)]
pub unsafe fn _mm_deinterleave3_pd(a: __m128d, b: __m128d, c: __m128d) -> (__m128d, __m128d, __m128d) {
    let x = _mm_shuffle_pd::<{ MM_SHUFFLE_R!(0, 1) }>(a, b);
    let y = _mm_shuffle_pd::<{ MM_SHUFFLE_R!(1, 0) }>(a, c);
    let z = _mm_shuffle_pd::<{ MM_SHUFFLE_R!(0, 1) }>(b, c);

    (x, y, z)
}

/// Radix-3 interleave of two-lane 64-bit registers, in three `shufpd` - the
/// exact inverse of [`_mm_deinterleave3_pd`].
#[inline(always)]
pub unsafe fn _mm_interleave3_pd(x: __m128d, y: __m128d, z: __m128d) -> (__m128d, __m128d, __m128d) {
    let a = _mm_shuffle_pd::<{ MM_SHUFFLE_R!(0, 0) }>(x, y); // x0 y0
    let b = _mm_shuffle_pd::<{ MM_SHUFFLE_R!(0, 1) }>(z, x); // z0 x1
    let c = _mm_shuffle_pd::<{ MM_SHUFFLE_R!(1, 1) }>(y, z); // y1 z1

    (a, b, c)
}

#[inline(always)]
pub unsafe fn _mm_deinterleave3_epi32(a: __m128i, b: __m128i, c: __m128i) -> (__m128i, __m128i, __m128i) {
    let (x, y, z) = _mm_deinterleave3_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _mm_castsi128_ps(c));

    (_mm_castps_si128(x), _mm_castps_si128(y), _mm_castps_si128(z))
}

#[inline(always)]
pub unsafe fn _mm_interleave3_epi32(x: __m128i, y: __m128i, z: __m128i) -> (__m128i, __m128i, __m128i) {
    let (a, b, c) = _mm_interleave3_ps(_mm_castsi128_ps(x), _mm_castsi128_ps(y), _mm_castsi128_ps(z));

    (_mm_castps_si128(a), _mm_castps_si128(b), _mm_castps_si128(c))
}

#[inline(always)]
pub unsafe fn _mm_deinterleave3_epi64(a: __m128i, b: __m128i, c: __m128i) -> (__m128i, __m128i, __m128i) {
    let (x, y, z) = _mm_deinterleave3_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b), _mm_castsi128_pd(c));

    (_mm_castpd_si128(x), _mm_castpd_si128(y), _mm_castpd_si128(z))
}

#[inline(always)]
pub unsafe fn _mm_interleave3_epi64(x: __m128i, y: __m128i, z: __m128i) -> (__m128i, __m128i, __m128i) {
    let (a, b, c) = _mm_interleave3_pd(_mm_castsi128_pd(x), _mm_castsi128_pd(y), _mm_castsi128_pd(z));

    (_mm_castpd_si128(a), _mm_castpd_si128(b), _mm_castpd_si128(c))
}

use crate::register::ZeroUpper;

use super::*;

/// POLYFILL: Shift right and sign extend 64-bit integers
#[inline(always)]
pub unsafe fn _mm_srai_epi64x_v1(v: __m128i, cnt: i32) -> __m128i {
    let m = _mm_set1_epi64x(1i64 << (63 - cnt));
    _mm_sub_epi64(_mm_xor_si128(_mm_srl_epi64(v, _mm_cvtsi32_si128(cnt)), m), m)
}

/// POLYFILL: Shift right 64-bit integers (variable)
///
/// <https://stackoverflow.com/a/38608465/2083075>
#[inline(always)]
pub unsafe fn _mm_srlv_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let count_high = _mm_unpackhi_epi64(shifts, shifts); // move higher 64 bits to lower 64 bits

    let shifted_low = _mm_srl_epi64(value, shifts); // uses lower 64 bits of shifts
    let mut shifted_high = _mm_srl_epi64(value, count_high); // shift value by higher 64 bits (now in lower 64 bits)

    shifted_high = _mm_unpackhi_epi64(shifted_high, shifted_high); // move result to higher 64 bits

    _mm_unpacklo_epi64(shifted_low, shifted_high) // combine results (lane0 from low-shift, lane1 from high-shift)
}

#[target_feature(enable = "sse2")] // LLVM can probably auto-vectorize this to some degree
#[inline]
pub unsafe fn _mm_srlv_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let shifts: [u32; 4] = core::mem::transmute(shifts);
    let mut value: [u32; 4] = core::mem::transmute(value);

    for (value, shift) in value.iter_mut().zip(shifts) {
        // `wrapping_shr` rather than `>>`: an out-of-range count yields an
        // unspecified value either way, but `>>` *panics* under overflow
        // checks, which would make a debug build fault on a lane value a
        // release build shifts. `unbounded_shr` would pin the result to zero,
        // but costs a compare and a select per lane to tidy up input nothing
        // promises anything about.
        *value = value.wrapping_shr(shift);
    }

    core::mem::transmute(value)
}

#[inline(always)]
pub unsafe fn _mm_sllv_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let count_high = _mm_unpackhi_epi64(shifts, shifts); // move higher 64 bits to lower 64 bits

    let shifted_low = _mm_sll_epi64(value, shifts); // uses lower 64 bits of shifts
    let mut shifted_high = _mm_sll_epi64(value, count_high); // shift value by higher 64 bits (now in lower 64 bits)

    shifted_high = _mm_unpackhi_epi64(shifted_high, shifted_high); // move result to higher 64 bits

    _mm_unpacklo_epi64(shifted_low, shifted_high) // combine results (lane0 from low-shift, lane1 from high-shift)
}

#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn _mm_sllv_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let shifts: [u32; 4] = core::mem::transmute(shifts);
    let mut value: [u32; 4] = core::mem::transmute(value);

    for (value, shift) in value.iter_mut().zip(shifts) {
        // See `_mm_srlv_epi32x_v1` on `wrapping_` vs `>>`.
        *value = value.wrapping_shl(shift);
    }

    core::mem::transmute(value)
}

/// POLYFILL: Shift right and sign extend 64-bit integers (variable)
#[inline(always)]
pub unsafe fn _mm_srav_epi64x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let m = _mm_srlv_epi64x_v1(_mm_set1_epu64x(1 << 63), shifts);
    _mm_sub_epi64(_mm_xor_si128(_mm_srlv_epi64x_v1(value, shifts), m), m)
}

// This would have been like the 64-bit version, but for 32-bit integers
// it's simpler to just do it scalar-wise instead of trying to be clever
#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn _mm_srav_epi32x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let shifts: [u32; 4] = core::mem::transmute(shifts);
    let mut value: [i32; 4] = core::mem::transmute(value);

    for (value, shift) in value.iter_mut().zip(shifts) {
        // See `_mm_srlv_epi32x_v1` on `wrapping_` vs `>>`.
        *value = value.wrapping_shr(shift);
    }

    core::mem::transmute(value)
}

/// POLYFILL: per-byte population count via the classic SWAR reduction
/// (no `pshufb` nibble lookup available on SSE2).
#[inline(always)]
pub unsafe fn _mm_popcnt_epi8x_v1(v: __m128i) -> __m128i {
    // v = v - ((v >> 1) & 0x55)
    let v = _mm_sub_epi8(v, _mm_and_si128(_mm_srli_epi16(v, 1), _mm_set1_epi8(0x55)));
    // v = (v & 0x33) + ((v >> 2) & 0x33)
    let v = _mm_add_epi8(
        _mm_and_si128(v, _mm_set1_epi8(0x33)),
        _mm_and_si128(_mm_srli_epi16(v, 2), _mm_set1_epi8(0x33)),
    );
    // v = (v + (v >> 4)) & 0x0F
    _mm_and_si128(_mm_add_epi8(v, _mm_srli_epi16(v, 4)), _mm_set1_epi8(0x0F))
}

/// POLYFILL: per-`i16`-lane population count.
#[inline(always)]
pub unsafe fn _mm_popcnt_epi16x_v1(v: __m128i) -> __m128i {
    let bytes = _mm_popcnt_epi8x_v1(v);
    // sum the two bytes within each 16-bit lane (no `pmaddubsw` on SSE2)
    _mm_add_epi16(_mm_and_si128(bytes, _mm_set1_epi16(0x00FF)), _mm_srli_epi16(bytes, 8))
}

/// POLYFILL: byte-swap within each 16-bit lane (no `pshufb` on SSE2).
#[inline(always)]
pub unsafe fn _mm_bswap_epi16x_v1(v: __m128i) -> __m128i {
    _mm_or_si128(_mm_slli_epi16(v, 8), _mm_srli_epi16(v, 8))
}

/// POLYFILL: per-`i32`-lane population count.
#[inline(always)]
pub unsafe fn _mm_popcnt_epi32x_v1(v: __m128i) -> __m128i {
    let bytes = _mm_popcnt_epi8x_v1(v);

    // horizontal byte sums within each 32-bit lane (no `pmaddubsw` on SSE2)
    let sum16 = _mm_add_epi16(_mm_and_si128(bytes, _mm_set1_epi16(0x00FF)), _mm_srli_epi16(bytes, 8));
    _mm_add_epi32(
        _mm_and_si128(sum16, _mm_set1_epi32(0x0000FFFF)),
        _mm_srli_epi32(sum16, 16),
    )
}

/// POLYFILL: per-`i64`-lane population count (`psadbw` sums bytes per 64-bit half).
#[inline(always)]
pub unsafe fn _mm_popcnt_epi64x_v1(v: __m128i) -> __m128i {
    _mm_sad_epu8(_mm_popcnt_epi8x_v1(v), _mm_setzero_si128())
}

#[inline(always)]
pub unsafe fn _mm_bswap_epi32x_v1(x: __m128i) -> __m128i {
    let t = _mm_or_si128(_mm_slli_epi16(x, 8), _mm_srli_epi16(x, 8));
    _mm_or_si128(_mm_slli_epi32(t, 16), _mm_srli_epi32(t, 16))
}

#[inline(always)]
pub unsafe fn _mm_bswap_epi64x_v1(x: __m128i) -> __m128i {
    // swap bytes in each 32-bit half, then swap the halves
    _mm_shuffle_epi32::<{ MM_SHUFFLE!(2, 3, 0, 1) }>(_mm_bswap_epi32x_v1(x))
}

#[inline(always)]
pub unsafe fn _mm_bswap_psx_v1(x: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_bswap_epi32x_v1(_mm_castps_si128(x)))
}

#[inline(always)]
pub unsafe fn _mm_bswap_pdx_v1(x: __m128d) -> __m128d {
    _mm_castsi128_pd(_mm_bswap_epi64x_v1(_mm_castpd_si128(x)))
}

#[inline(always)]
pub unsafe fn _mm_zeroupper_mask_epi32<Z: ZeroUpper>() -> __m128i {
    use crate::element::MaskElement;

    _mm_setr_epi32(
        i32::from_bool(0 < Z::N),
        i32::from_bool(1 < Z::N),
        i32::from_bool(2 < Z::N),
        i32::from_bool(3 < Z::N),
    )
}

// ===========================================================================
// 8-bit (byte) lane polyfills. x86 has no native 8-bit shift or multiply, so
// these emulate them with 16-bit ops + masking. They use only SSE2 intrinsics,
// so v2/v3 inherit them through the polyfill re-export chain. All shift counts
// are assumed to be in `0..8` (the byte element width); the *variable*-count
// forms below handle out-of-range counts explicitly.
// ===========================================================================

/// POLYFILL: logical left shift of each byte lane by a compile-time count.
/// Shift as 16-bit, then mask off the bits that crossed byte boundaries.
#[inline(always)]
pub unsafe fn _mm_slli_epi8x_v1<const IMM8: i32>(v: __m128i) -> __m128i {
    let keep = 0xFFu8.wrapping_shl(IMM8 as u32) as i8; // bits surviving inside a byte
    _mm_and_si128(_mm_slli_epi16(v, IMM8), _mm_set1_epi8(keep))
}

/// POLYFILL: logical right shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm_srli_epi8x_v1<const IMM8: i32>(v: __m128i) -> __m128i {
    let keep = (0xFFu8 >> IMM8) as i8;
    _mm_and_si128(_mm_srli_epi16(v, IMM8), _mm_set1_epi8(keep))
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a compile-time count.
/// Logical shift, then branchless sign extension via `(x ^ m) - m`.
#[inline(always)]
pub unsafe fn _mm_srai_epi8x_v1<const IMM8: i32>(v: __m128i) -> __m128i {
    let logical = _mm_srli_epi8x_v1::<IMM8>(v);
    let m = _mm_set1_epi8((0x80u8 >> IMM8) as i8);
    _mm_sub_epi8(_mm_xor_si128(logical, m), m)
}

/// POLYFILL: logical left shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm_sll_epi8x_v1(v: __m128i, shift: u32) -> __m128i {
    let keep = (0xFFu32.wrapping_shl(shift) as u8) as i8;
    _mm_and_si128(_mm_sll_epi16(v, _mm_cvtsi32_si128(shift as i32)), _mm_set1_epi8(keep))
}

/// POLYFILL: logical right shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm_srl_epi8x_v1(v: __m128i, shift: u32) -> __m128i {
    let keep = ((0xFFu32 >> shift.min(31)) as u8) as i8;
    _mm_and_si128(_mm_srl_epi16(v, _mm_cvtsi32_si128(shift as i32)), _mm_set1_epi8(keep))
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm_sra_epi8x_v1(v: __m128i, shift: u32) -> __m128i {
    let logical = _mm_srl_epi8x_v1(v, shift);
    let m = _mm_set1_epi8(((0x80u32 >> shift.min(31)) as u8) as i8);
    _mm_sub_epi8(_mm_xor_si128(logical, m), m)
}

// ---------------------------------------------------------------------------
// Per-lane variable shifts for 8- and 16-bit lanes.
//
// x86 has no variable shift below 32-bit lanes at any level before
// AVX-512BW+VL (`vpsllvw`), and none at all for bytes. These walk the low bits
// of the per-lane count and conditionally apply a *constant* shift for each,
// staying in registers throughout instead of round-tripping through memory.
//
// Only bits `0..log2(width)` of the count are ever examined, so a count of `s`
// behaves as `s & (width - 1)`. That is not an accident to be tidied up later:
// it matches the scalar oracle, because Rust's `<<`/`>>` mask the shift amount
// (`3u8 << 8 == 3`). Hardware `vpsllvd` instead flushes to zero for
// out-of-range counts, so the two conventions genuinely differ. Masking is the
// one the differential suite compares against, and here it costs nothing.
//
// Composition is exact for all three shift kinds: `(x << a) << b == x << (a+b)`,
// and likewise for logical and arithmetic right shifts, so applying the
// power-of-two steps in sequence yields the full shift.
// ---------------------------------------------------------------------------

/// Conditionally apply `$shift` to `$x` on the lanes where bit `$bit` of
/// `$shifts` is set, using `$cmpeq` to broadcast that bit to a full lane mask.
///
/// Blend is `x ^ ((x ^ shifted) & m)`: three ops, versus four for
/// and/andnot/or, and no `pblendvb` so it stays SSE2-clean.
macro_rules! varshift_step {
    ($x:ident, $shifts:ident, $set1:ident, $cmpeq:ident, $bit:expr, $shifted:expr) => {{
        let sel = $set1($bit);
        let m = $cmpeq(_mm_and_si128($shifts, sel), sel);
        $x = _mm_xor_si128($x, _mm_and_si128(_mm_xor_si128($x, $shifted), m));
    }};
}

/// POLYFILL: per-lane variable logical left shift of 16-bit lanes (`vpsllvw`).
#[inline(always)]
pub unsafe fn _mm_sllv_epi16x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let mut x = value;
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 1, _mm_slli_epi16(x, 1));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 2, _mm_slli_epi16(x, 2));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 4, _mm_slli_epi16(x, 4));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 8, _mm_slli_epi16(x, 8));
    x
}

/// POLYFILL: per-lane variable logical right shift of 16-bit lanes (`vpsrlvw`).
#[inline(always)]
pub unsafe fn _mm_srlv_epi16x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let mut x = value;
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 1, _mm_srli_epi16(x, 1));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 2, _mm_srli_epi16(x, 2));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 4, _mm_srli_epi16(x, 4));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 8, _mm_srli_epi16(x, 8));
    x
}

/// POLYFILL: per-lane variable arithmetic right shift of `i16` lanes (`vpsravw`).
///
/// SSE2 *does* have `psraw` with a constant count, so unlike the 8-bit case
/// this needs no sign-extension fixup: the steps sign-fill on their own.
#[inline(always)]
pub unsafe fn _mm_srav_epi16x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let mut x = value;
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 1, _mm_srai_epi16(x, 1));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 2, _mm_srai_epi16(x, 2));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 4, _mm_srai_epi16(x, 4));
    varshift_step!(x, shifts, _mm_set1_epi16, _mm_cmpeq_epi16, 8, _mm_srai_epi16(x, 8));
    x
}

/// POLYFILL: per-lane variable logical left shift of byte lanes.
///
/// Each step reuses the constant-count byte shift, which already masks off the
/// bits that bled across the byte boundary during the 16-bit shift it is built
/// from, so the truncation is applied per step and cannot accumulate.
#[inline(always)]
pub unsafe fn _mm_sllv_epi8x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let mut x = value;
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 1, _mm_slli_epi8x_v1::<1>(x));
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 2, _mm_slli_epi8x_v1::<2>(x));
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 4, _mm_slli_epi8x_v1::<4>(x));
    x
}

/// POLYFILL: per-lane variable logical right shift of byte lanes.
#[inline(always)]
pub unsafe fn _mm_srlv_epi8x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let mut x = value;
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 1, _mm_srli_epi8x_v1::<1>(x));
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 2, _mm_srli_epi8x_v1::<2>(x));
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 4, _mm_srli_epi8x_v1::<4>(x));
    x
}

/// POLYFILL: per-lane variable arithmetic right shift of `i8` lanes.
#[inline(always)]
pub unsafe fn _mm_srav_epi8x_v1(value: __m128i, shifts: __m128i) -> __m128i {
    let mut x = value;
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 1, _mm_srai_epi8x_v1::<1>(x));
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 2, _mm_srai_epi8x_v1::<2>(x));
    varshift_step!(x, shifts, _mm_set1_epi8, _mm_cmpeq_epi8, 4, _mm_srai_epi8x_v1::<4>(x));
    x
}

/// POLYFILL: low 8 bits of each byte product (`a[i].wrapping_mul(b[i])`).
/// Multiply even/odd bytes as 16-bit lanes, then recombine the low bytes.
#[inline(always)]
pub unsafe fn _mm_mullo_epi8x_v1(a: __m128i, b: __m128i) -> __m128i {
    let lo_mask = _mm_set1_epi16(0x00FF);
    let even = _mm_mullo_epi16(_mm_and_si128(a, lo_mask), _mm_and_si128(b, lo_mask));
    let odd = _mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8));
    _mm_or_si128(_mm_and_si128(even, lo_mask), _mm_slli_epi16(odd, 8))
}

/// POLYFILL: high 8 bits of each signed byte product (`((a as i16 * b as i16) >> 8) as i8`).
#[inline(always)]
pub unsafe fn _mm_mulhi_epi8x_v1(a: __m128i, b: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    // place each byte in the high half of a 16-bit lane, then arithmetic-shift to sign-extend
    let a_lo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, a), 8);
    let b_lo = _mm_srai_epi16(_mm_unpacklo_epi8(zero, b), 8);
    let a_hi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, a), 8);
    let b_hi = _mm_srai_epi16(_mm_unpackhi_epi8(zero, b), 8);
    let p_lo = _mm_srai_epi16(_mm_mullo_epi16(a_lo, b_lo), 8);
    let p_hi = _mm_srai_epi16(_mm_mullo_epi16(a_hi, b_hi), 8);
    _mm_packs_epi16(p_lo, p_hi)
}

/// POLYFILL: high 8 bits of each unsigned byte product (`((a as u16 * b as u16) >> 8) as u8`).
#[inline(always)]
pub unsafe fn _mm_mulhi_epu8x_v1(a: __m128i, b: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let a_lo = _mm_unpacklo_epi8(a, zero); // zero-extend low 8 bytes
    let b_lo = _mm_unpacklo_epi8(b, zero);
    let a_hi = _mm_unpackhi_epi8(a, zero);
    let b_hi = _mm_unpackhi_epi8(b, zero);
    let p_lo = _mm_srli_epi16(_mm_mullo_epi16(a_lo, b_lo), 8);
    let p_hi = _mm_srli_epi16(_mm_mullo_epi16(a_hi, b_hi), 8);
    _mm_packus_epi16(p_lo, p_hi)
}

// ---------------------------------------------------------------------------
// Mask population counts: merge several masks with a saturating narrowing pack
// and extract ONE bitmask, instead of a movemask + popcount per register.
//
// `_mm_packs_epi32`/`_mm_packs_epi16` saturate signed, which maps an all-ones
// lane (-1) to -1 and an all-zeros lane to 0 - both mask lanes survive the
// narrowing exactly. The pack interleaves its two sources, so the lane ORDER of
// the result is scrambled; a population count cannot observe that, which is
// precisely why this is only legal for `count_set` and not for
// `first_set`/`last_set`.
// ---------------------------------------------------------------------------

/// POLYFILL: set lanes across two 32-bit-lane masks.
///
/// Narrows to 8 16-bit lanes, so `movmskb` reports 2 bits per lane.
#[inline(always)]
pub unsafe fn _mm_count_mask2_epi32x_v1(a: __m128i, b: __m128i) -> usize {
    (_mm_movemask_epi8(_mm_packs_epi32(a, b)) as u32).count_ones() as usize / 2
}

/// POLYFILL: set lanes across four 32-bit-lane masks.
///
/// Two levels of packing land on 16 8-bit lanes, so `movmskb` reports exactly
/// one bit per lane and no division is needed.
#[inline(always)]
pub unsafe fn _mm_count_mask4_epi32x_v1(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> usize {
    let lo = _mm_packs_epi32(a, b);
    let hi = _mm_packs_epi32(c, d);
    (_mm_movemask_epi8(_mm_packs_epi16(lo, hi)) as u32).count_ones() as usize
}

/// POLYFILL: set lanes across two 16-bit-lane masks. Exact, one bit per lane.
#[inline(always)]
pub unsafe fn _mm_count_mask2_epi16x_v1(a: __m128i, b: __m128i) -> usize {
    (_mm_movemask_epi8(_mm_packs_epi16(a, b)) as u32).count_ones() as usize
}

/// POLYFILL: total set lanes across `N` 32-bit-lane masks.
///
/// Descending ladder: fours, then a pair, then a single. Four is the ceiling
/// for 32-bit lanes - two narrowing steps reach 8-bit lanes, which is the floor
/// (`movmskb` is already one bit per byte, and there is nothing narrower to
/// pack to). Larger `N` is therefore chunks of four, which is optimal: OR-ing
/// several chunks' bitmasks into one word to save popcounts costs exactly the
/// shift-and-or it saves.
#[inline(always)]
pub unsafe fn _mm_count_mask_epi32x_v1<const N: usize>(values: [__m128i; N]) -> usize {
    let mut total = 0;
    let mut i = 0;

    while i + 3 < N {
        total += _mm_count_mask4_epi32x_v1(values[i], values[i + 1], values[i + 2], values[i + 3]);
        i += 4;
    }

    if i + 1 < N {
        total += _mm_count_mask2_epi32x_v1(values[i], values[i + 1]);
        i += 2;
    }

    if i < N {
        total += (_mm_movemask_ps(_mm_castsi128_ps(values[i])) as u32).count_ones() as usize;
    }

    total
}

/// POLYFILL: total set lanes across `N` 16-bit-lane masks.
#[inline(always)]
pub unsafe fn _mm_count_mask_epi16x_v1<const N: usize>(values: [__m128i; N]) -> usize {
    let mut total = 0;
    let mut i = 0;

    while i + 1 < N {
        total += _mm_count_mask2_epi16x_v1(values[i], values[i + 1]);
        i += 2;
    }

    if i < N {
        // 8 16-bit lanes -> 2 bits per lane out of `movmskb`.
        total += (_mm_movemask_epi8(values[i]) as u32).count_ones() as usize / 2;
    }

    total
}

/// POLYFILL: [`_mm_count_mask_epi32x_v1`] for masks held in float registers.
/// The casts are register renames, so the copy loop costs nothing.
#[inline(always)]
pub unsafe fn _mm_count_mask_ps_v1<const N: usize>(values: [__m128; N]) -> usize {
    let mut ints = [_mm_setzero_si128(); N];

    let mut i = 0;
    while i < N {
        ints[i] = _mm_castps_si128(values[i]);
        i += 1;
    }

    _mm_count_mask_epi32x_v1(ints)
}

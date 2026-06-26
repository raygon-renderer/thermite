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
        *value >>= shift;
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
        *value <<= shift;
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
        *value >>= shift;
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
// are assumed to be in `0..8` (the byte element width), matching the scalar
// contract where shifting by >= the bit width is undefined.
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

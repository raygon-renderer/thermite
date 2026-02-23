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

    _mm_unpacklo_epi64(shifted_high, shifted_low) // combine results
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

    _mm_unpacklo_epi64(shifted_high, shifted_low) // combine results
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
pub unsafe fn sse2_bswap_psx_v1(x: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_bswap_epi32x_v1(_mm_castps_si128(x)))
}

#[inline(always)]
pub unsafe fn sse2_bswap_pdx_v1(x: __m128d) -> __m128d {
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

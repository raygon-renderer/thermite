use crate::register::ZeroUpper;

use super::*;

// ===========================================================================
// CLMUL-based 2D Morton (Z-order) encode.
//
// The carry-less self-square identity `clmul(x, x) == spread1(x)` interleaves a
// value's bits with zeros (bit `i` -> bit `2i`): the diagonal terms land at the
// even positions and every off-diagonal pair `(i, j) + (j, i)` cancels mod 2, so
// the odd positions are zero. That makes a single PCLMULQDQ the whole 2D bit
// spread. Only 2D falls out of CLMUL - composing self-squares doubles the gap,
// so it yields 2^k-way interleaves (2D, 4D, ...), never the factor-of-3 of 3D.
//
// PCLMULQDQ is 64-bit-granular (one 64x64 -> 128 product, selecting one qword
// from each operand via the imm), so a 128-bit register's two lanes are squared
// separately and recombined. Gated on `avx2-pclmul`, which adds `pclmulqdq` to
// the dispatched target-feature set (every AVX2 CPU has it).
// ===========================================================================

/// Spread the low 32 bits of each 64-bit lane by one (bit `i` -> bit `2i`) via
/// carry-less self-multiply. Input lanes are masked to 32 bits so each 64-bit
/// spread cannot overflow its lane.
#[cfg(feature = "avx2-pclmul")]
#[inline(always)]
pub unsafe fn _mm_morton2_spread_epu64x_v3(v: __m128i) -> __m128i {
    let v = _mm_and_si128(v, _mm_set1_epi64x(0xFFFF_FFFF));
    _mm_unpacklo_epi64(
        _mm_clmulepi64_si128(v, v, 0x00), // spread of lane 0 in low 64
        _mm_clmulepi64_si128(v, v, 0x11), // spread of lane 1 in low 64
    )
}

/// Per-64-bit-lane 2D Morton encode (two 32-bit coords -> one 64-bit code per
/// lane): interleave the low 32 bits of `x` (even output bits) with the low 32
/// bits of `y` (odd output bits).
#[cfg(feature = "avx2-pclmul")]
#[inline(always)]
pub unsafe fn _mm_morton2_epu64x_v3(x: __m128i, y: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_morton2_spread_epu64x_v3(x),
        _mm_slli_epi64(_mm_morton2_spread_epu64x_v3(y), 1),
    )
}

/// 256-bit (`u64x4`) form of [`_mm_morton2_epu64x_v3`]: PCLMULQDQ has no 256-bit
/// form (that needs VPCLMULQDQ), so the two 128-bit halves are encoded separately
/// and reassembled.
#[cfg(feature = "avx2-pclmul")]
#[inline(always)]
pub unsafe fn _mm256_morton2_epu64x_v3(x: __m256i, y: __m256i) -> __m256i {
    let lo = _mm_morton2_epu64x_v3(_mm256_castsi256_si128(x), _mm256_castsi256_si128(y));
    let hi = _mm_morton2_epu64x_v3(_mm256_extracti128_si256(x, 1), _mm256_extracti128_si256(y, 1));
    _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
}

// 256-bit PSHUFB-based 2D Morton: the per-128-bit-lane nibble LUT (see the v2
// `_mm_morton2_*` helpers for the method) replicated across both halves. The
// nibble-isolate and shifts are per-element, so widening to 256 is mechanical.

/// Spread the low 8 bits of each 16-bit lane by one (2D Morton), via the nibble LUT.
#[inline(always)]
pub unsafe fn _mm256_morton2_spread_epu16x_v3(v: __m256i) -> __m256i {
    let lut = _mm256_setr_epi8(
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55, //
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
    );
    let c = _mm256_and_si256(v, _mm256_set1_epi16(0x00FF));
    let n = _mm256_and_si256(_mm256_or_si256(c, _mm256_slli_epi16(c, 4)), _mm256_set1_epi16(0x0F0F));
    _mm256_shuffle_epi8(lut, n)
}

/// Spread the low 16 bits of each 32-bit lane by one (2D Morton), via the nibble LUT.
#[inline(always)]
pub unsafe fn _mm256_morton2_spread_epu32x_v3(v: __m256i) -> __m256i {
    let lut = _mm256_setr_epi8(
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55, //
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
    );
    let c = _mm256_and_si256(v, _mm256_set1_epi32(0x0000_FFFF));
    let c = _mm256_and_si256(
        _mm256_or_si256(c, _mm256_slli_epi32(c, 8)),
        _mm256_set1_epi32(0x00FF_00FF),
    );
    let n = _mm256_and_si256(
        _mm256_or_si256(c, _mm256_slli_epi32(c, 4)),
        _mm256_set1_epi32(0x0F0F_0F0F),
    );
    _mm256_shuffle_epi8(lut, n)
}

/// Per-16-bit-lane 2D Morton encode: low 8 bits of `x` (even) interleaved with
/// low 8 bits of `y` (odd).
#[inline(always)]
pub unsafe fn _mm256_morton2_epu16x_v3(x: __m256i, y: __m256i) -> __m256i {
    _mm256_or_si256(
        _mm256_morton2_spread_epu16x_v3(x),
        _mm256_slli_epi16(_mm256_morton2_spread_epu16x_v3(y), 1),
    )
}

/// Per-32-bit-lane 2D Morton encode: low 16 bits of `x` (even) interleaved with
/// low 16 bits of `y` (odd).
#[inline(always)]
pub unsafe fn _mm256_morton2_epu32x_v3(x: __m256i, y: __m256i) -> __m256i {
    _mm256_or_si256(
        _mm256_morton2_spread_epu32x_v3(x),
        _mm256_slli_epi32(_mm256_morton2_spread_epu32x_v3(y), 1),
    )
}

// 256-bit PSHUFB-based 2D Morton DECODE (inverse spread / compress): the v2
// two-nibble-LUT compress (see `_mm_morton2_compress_*`) widened to 256, with the
// byte->2-bit compress LUT replicated across both 128-bit halves.

/// Compress the even bits of each 16-bit lane back into a contiguous low 8 bits -
/// the inverse of [`_mm256_morton2_spread_epu16x_v3`].
#[inline(always)]
pub unsafe fn _mm256_morton2_compress_epu16x_v3(v: __m256i) -> __m256i {
    let lut = _mm256_setr_epi8(
        0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, //
        0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
    );
    let e = _mm256_and_si256(v, _mm256_set1_epi16(0x5555));
    let lo = _mm256_shuffle_epi8(lut, e);
    let hi = _mm256_shuffle_epi8(lut, _mm256_srli_epi16(e, 4));
    let n = _mm256_and_si256(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 2)), _mm256_set1_epi16(0x0F0F));
    _mm256_and_si256(_mm256_or_si256(n, _mm256_srli_epi16(n, 4)), _mm256_set1_epi16(0x00FF))
}

/// Compress the even bits of each 32-bit lane back into a contiguous low 16 bits -
/// the inverse of [`_mm256_morton2_spread_epu32x_v3`].
#[inline(always)]
pub unsafe fn _mm256_morton2_compress_epu32x_v3(v: __m256i) -> __m256i {
    let lut = _mm256_setr_epi8(
        0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, //
        0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3,
    );
    let e = _mm256_and_si256(v, _mm256_set1_epi32(0x5555_5555));
    let lo = _mm256_shuffle_epi8(lut, e);
    let hi = _mm256_shuffle_epi8(lut, _mm256_srli_epi32(e, 4));
    let n = _mm256_and_si256(
        _mm256_or_si256(lo, _mm256_slli_epi32(hi, 2)),
        _mm256_set1_epi32(0x0F0F_0F0F),
    );
    let c = _mm256_and_si256(
        _mm256_or_si256(n, _mm256_srli_epi32(n, 4)),
        _mm256_set1_epi32(0x00FF_00FF),
    );
    _mm256_and_si256(
        _mm256_or_si256(c, _mm256_srli_epi32(c, 8)),
        _mm256_set1_epi32(0x0000_FFFF),
    )
}

// https://arxiv.org/pdf/1611.07612.pdf
#[inline(always)] #[rustfmt::skip]
pub unsafe fn _mm256_popcnt_epi8x_v3(v: __m256i) -> __m256i {
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(popcnt1, popcnt2)
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi64x_v3(v: __m256i) -> __m256i {
    _mm256_sad_epu8(_mm256_popcnt_epi8x_v3(v), _mm256_setzero_si256())
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi16x_v3(v: __m256i) -> __m256i {
    // Per-byte popcount, then sum adjacent byte pairs into 16-bit lanes via maddubs.
    _mm256_maddubs_epi16(_mm256_popcnt_epi8x_v3(v), _mm256_set1_epi8(1))
}

#[inline(always)]
pub unsafe fn _mm256_popcnt_epi32x_v3(v: __m256i) -> __m256i {
    // https://stackoverflow.com/a/51106873/2083075
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(_mm256_popcnt_epi8x_v3(v), _mm256_set1_epi8(1)),
        _mm256_set1_epi16(1),
    )
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi16x_v3(value: __m256i) -> __m256i {
    // Mask: 1 0 | 3 2 | 5 4 | 7 6 | 9 8 | 11 10 | 13 12 | 15 14
    _mm256_shuffle_epi8(
        value,
        _mm256_setr_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // Lane 1
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // Lane 2
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi32x_v3(x: __m256i) -> __m256i {
    // Note: vpshufb works within 128-bit lanes.
    // We repeat the 128-bit mask for both lanes.
    let mask = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // Lane 1
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // Lane 2
    );
    _mm256_shuffle_epi8(x, mask)
}

#[inline(always)]
pub unsafe fn _mm256_bswap_epi64x_v3(x: __m256i) -> __m256i {
    let mask = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, // Lane 1
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, // Lane 2
    );
    _mm256_shuffle_epi8(x, mask)
}

#[inline(always)]
pub unsafe fn _mm256_bswap_psx_v3(x: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_bswap_epi32x_v3(_mm256_castps_si256(x)))
}

#[inline(always)]
pub unsafe fn _mm256_bswap_pdx_v3(x: __m256d) -> __m256d {
    _mm256_castsi256_pd(_mm256_bswap_epi64x_v3(_mm256_castpd_si256(x)))
}

#[inline(always)]
pub unsafe fn _mm256_zeroupper_mask_epi32<Z: ZeroUpper>() -> __m256i {
    use crate::element::MaskElement;

    _mm256_setr_epi32(
        i32::from_bool(0 < Z::N),
        i32::from_bool(1 < Z::N),
        i32::from_bool(2 < Z::N),
        i32::from_bool(3 < Z::N),
        i32::from_bool(4 < Z::N),
        i32::from_bool(5 < Z::N),
        i32::from_bool(6 < Z::N),
        i32::from_bool(7 < Z::N),
    )
}

// ===========================================================================
// 256-bit 8-bit (byte) lane polyfills (AVX2). x86 has no native 8-bit shift or
// multiply at any width, so these emulate them with 16-bit ops + masking. Shift
// counts are assumed in `0..8`. The even/odd byte-multiply and the
// unpack/pack-based high-multiply are purely per-128-bit-lane, so no cross-lane
// fixup is needed.
// ===========================================================================

/// POLYFILL: logical left shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm256_slli_epi8x_v3<const IMM8: i32>(v: __m256i) -> __m256i {
    let keep = 0xFFu8.wrapping_shl(IMM8 as u32) as i8;
    _mm256_and_si256(_mm256_slli_epi16(v, IMM8), _mm256_set1_epi8(keep))
}

/// POLYFILL: logical right shift of each byte lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm256_srli_epi8x_v3<const IMM8: i32>(v: __m256i) -> __m256i {
    let keep = (0xFFu8 >> IMM8) as i8;
    _mm256_and_si256(_mm256_srli_epi16(v, IMM8), _mm256_set1_epi8(keep))
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a compile-time count.
#[inline(always)]
pub unsafe fn _mm256_srai_epi8x_v3<const IMM8: i32>(v: __m256i) -> __m256i {
    let logical = _mm256_srli_epi8x_v3::<IMM8>(v);
    let m = _mm256_set1_epi8((0x80u8 >> IMM8) as i8);
    _mm256_sub_epi8(_mm256_xor_si256(logical, m), m)
}

/// POLYFILL: logical left shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm256_sll_epi8x_v3(v: __m256i, shift: u32) -> __m256i {
    let keep = (0xFFu32.wrapping_shl(shift) as u8) as i8;
    _mm256_and_si256(
        _mm256_sll_epi16(v, _mm_cvtsi32_si128(shift as i32)),
        _mm256_set1_epi8(keep),
    )
}

/// POLYFILL: logical right shift of each byte lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm256_srl_epi8x_v3(v: __m256i, shift: u32) -> __m256i {
    let keep = ((0xFFu32 >> shift.min(31)) as u8) as i8;
    _mm256_and_si256(
        _mm256_srl_epi16(v, _mm_cvtsi32_si128(shift as i32)),
        _mm256_set1_epi8(keep),
    )
}

/// POLYFILL: arithmetic right shift of each `i8` lane by a runtime count.
#[inline(always)]
pub unsafe fn _mm256_sra_epi8x_v3(v: __m256i, shift: u32) -> __m256i {
    let logical = _mm256_srl_epi8x_v3(v, shift);
    let m = _mm256_set1_epi8(((0x80u32 >> shift.min(31)) as u8) as i8);
    _mm256_sub_epi8(_mm256_xor_si256(logical, m), m)
}

// ---------------------------------------------------------------------------
// Per-lane variable shifts for 8- and 16-bit lanes, 256-bit.
//
// AVX2's `vpsllvd`/`vpsllvq` stop at 32-bit lanes; `vpsllvw` needs
// AVX-512BW+VL. Same conditional-constant-shift decomposition as the v1
// 128-bit forms, but blending with `vpblendvb` (one instruction instead of the
// SSE2 xor/and/xor triple). The mask is uniform across each lane, so
// byte-granular blending is exact for 16-bit lanes too.
//
// As at v1, only bits `0..log2(width)` of the count are examined, so counts
// behave as `s & (width - 1)`, matching the scalar oracle.
// ---------------------------------------------------------------------------

macro_rules! varshift256_step {
    ($x:ident, $shifts:ident, $set1:ident, $cmpeq:ident, $bit:expr, $shifted:expr) => {{
        let sel = $set1($bit);
        let m = $cmpeq(_mm256_and_si256($shifts, sel), sel);
        $x = _mm256_blendv_epi8($x, $shifted, m);
    }};
}

/// POLYFILL: per-lane variable logical left shift of 16-bit lanes (`vpsllvw`).
#[inline(always)]
pub unsafe fn _mm256_sllv_epi16x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let mut x = value;
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        1,
        _mm256_slli_epi16(x, 1)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        2,
        _mm256_slli_epi16(x, 2)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        4,
        _mm256_slli_epi16(x, 4)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        8,
        _mm256_slli_epi16(x, 8)
    );
    x
}

/// POLYFILL: per-lane variable logical right shift of 16-bit lanes (`vpsrlvw`).
#[inline(always)]
pub unsafe fn _mm256_srlv_epi16x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let mut x = value;
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        1,
        _mm256_srli_epi16(x, 1)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        2,
        _mm256_srli_epi16(x, 2)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        4,
        _mm256_srli_epi16(x, 4)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        8,
        _mm256_srli_epi16(x, 8)
    );
    x
}

/// POLYFILL: per-lane variable arithmetic right shift of `i16` lanes (`vpsravw`).
#[inline(always)]
pub unsafe fn _mm256_srav_epi16x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let mut x = value;
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        1,
        _mm256_srai_epi16(x, 1)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        2,
        _mm256_srai_epi16(x, 2)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        4,
        _mm256_srai_epi16(x, 4)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi16,
        _mm256_cmpeq_epi16,
        8,
        _mm256_srai_epi16(x, 8)
    );
    x
}

/// POLYFILL: per-lane variable logical left shift of byte lanes.
#[inline(always)]
pub unsafe fn _mm256_sllv_epi8x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let mut x = value;
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        1,
        _mm256_slli_epi8x_v3::<1>(x)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        2,
        _mm256_slli_epi8x_v3::<2>(x)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        4,
        _mm256_slli_epi8x_v3::<4>(x)
    );
    x
}

/// POLYFILL: per-lane variable logical right shift of byte lanes.
#[inline(always)]
pub unsafe fn _mm256_srlv_epi8x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let mut x = value;
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        1,
        _mm256_srli_epi8x_v3::<1>(x)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        2,
        _mm256_srli_epi8x_v3::<2>(x)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        4,
        _mm256_srli_epi8x_v3::<4>(x)
    );
    x
}

/// POLYFILL: per-lane variable arithmetic right shift of `i8` lanes.
#[inline(always)]
pub unsafe fn _mm256_srav_epi8x_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let mut x = value;
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        1,
        _mm256_srai_epi8x_v3::<1>(x)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        2,
        _mm256_srai_epi8x_v3::<2>(x)
    );
    varshift256_step!(
        x,
        shifts,
        _mm256_set1_epi8,
        _mm256_cmpeq_epi8,
        4,
        _mm256_srai_epi8x_v3::<4>(x)
    );
    x
}

// ---------------------------------------------------------------------------
// 16-bit variable shifts, second lowering: widen to 32-bit lanes and use the
// native `vpsllvd`/`vpsrlvd`/`vpsravd`, then narrow back.
//
// Only worth it at 16-bit: the widening is 2x, so one 256-bit vector becomes
// two, and AVX2 unpack *and* pack both operate within 128-bit halves, so the
// permutation cancels and the round trip needs no `vperm2i128`/`vpermq` fixup.
// At 8-bit the widening would be 4x and the extract/pack overhead swamps the
// win, which is why the byte forms keep the decomposition.
//
// Two hazards, both load-bearing:
//
// 1. The count must be reduced mod 16 explicitly. Widening hands `vpsllvd` a
//    32-bit lane, so a count of 16..=31 would shift for real instead of
//    wrapping, and a count >= 32 would flush to zero. Neither matches the
//    scalar oracle. The decomposition got this for free. Here it costs an AND.
// 2. `vpackusdw` SATURATES. A left shift that pushes bits above 16 must be
//    masked to 0xFFFF before packing or the lane clamps to 0xFFFF instead of
//    truncating, which is exactly fearless_simd #287, in its original habitat.
//    The signed form needs `vpackssdw` for the same reason in the other
//    direction: `vpackusdw` would clamp negatives to zero.
// ---------------------------------------------------------------------------

/// POLYFILL: `vpsllvw` via 32-bit widening. See [`_mm256_sllv_epi16x_v3`] for
/// the alternative lowering.
#[inline(always)]
pub unsafe fn _mm256_sllv_epi16_widex_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let z = _mm256_setzero_si256();
    // Masked, not clamped: widening would otherwise let a count of 16..=31
    // shift for real in a 32-bit lane and then saturate through the pack. The
    // AND is one op and keeps an out-of-range result merely unspecified rather
    // than wild. See `BitshiftVector::shrv`.
    let s = _mm256_and_si256(shifts, _mm256_set1_epi16(15));
    let lo = _mm256_sllv_epi32(_mm256_unpacklo_epi16(value, z), _mm256_unpacklo_epi16(s, z));
    let hi = _mm256_sllv_epi32(_mm256_unpackhi_epi16(value, z), _mm256_unpackhi_epi16(s, z));
    let m = _mm256_set1_epi32(0xFFFF);
    _mm256_packus_epi32(_mm256_and_si256(lo, m), _mm256_and_si256(hi, m))
}

/// POLYFILL: `vpsrlvw` via 32-bit widening. No post-shift mask needed: the
/// lanes were zero-extended and a logical right shift keeps them in range.
#[inline(always)]
pub unsafe fn _mm256_srlv_epi16_widex_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let z = _mm256_setzero_si256();
    // Masked, not clamped: widening would otherwise let a count of 16..=31
    // shift for real in a 32-bit lane and then saturate through the pack. The
    // AND is one op and keeps an out-of-range result merely unspecified rather
    // than wild. See `BitshiftVector::shrv`.
    let s = _mm256_and_si256(shifts, _mm256_set1_epi16(15));
    let lo = _mm256_srlv_epi32(_mm256_unpacklo_epi16(value, z), _mm256_unpacklo_epi16(s, z));
    let hi = _mm256_srlv_epi32(_mm256_unpackhi_epi16(value, z), _mm256_unpackhi_epi16(s, z));
    _mm256_packus_epi32(lo, hi)
}

/// POLYFILL: `vpsravw` via 32-bit widening. Sign-extends by duplicating each
/// word into a dword and arithmetic-shifting it back down.
#[inline(always)]
pub unsafe fn _mm256_srav_epi16_widex_v3(value: __m256i, shifts: __m256i) -> __m256i {
    let z = _mm256_setzero_si256();
    // Masked, not clamped: widening would otherwise let a count of 16..=31
    // shift for real in a 32-bit lane and then saturate through the pack. The
    // AND is one op and keeps an out-of-range result merely unspecified rather
    // than wild. See `BitshiftVector::shrv`.
    let s = _mm256_and_si256(shifts, _mm256_set1_epi16(15));
    let lo = _mm256_srav_epi32(
        _mm256_srai_epi32(_mm256_unpacklo_epi16(value, value), 16),
        _mm256_unpacklo_epi16(s, z),
    );
    let hi = _mm256_srav_epi32(
        _mm256_srai_epi32(_mm256_unpackhi_epi16(value, value), 16),
        _mm256_unpackhi_epi16(s, z),
    );
    _mm256_packs_epi32(lo, hi)
}

/// POLYFILL: 128-bit `vpsllvw` via 32-bit widening (AVX2 supplies the 128-bit
/// `vpsllvd` too, so v3's 128-bit registers get the same treatment).
#[inline(always)]
pub unsafe fn _mm_sllv_epi16_widex_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    // See the 256-bit form: masked, not clamped.
    let s = _mm_and_si128(shifts, _mm_set1_epi16(15));
    let lo = _mm_sllv_epi32(_mm_unpacklo_epi16(value, z), _mm_unpacklo_epi16(s, z));
    let hi = _mm_sllv_epi32(_mm_unpackhi_epi16(value, z), _mm_unpackhi_epi16(s, z));
    let m = _mm_set1_epi32(0xFFFF);
    _mm_packus_epi32(_mm_and_si128(lo, m), _mm_and_si128(hi, m))
}

/// POLYFILL: 128-bit `vpsrlvw` via 32-bit widening.
#[inline(always)]
pub unsafe fn _mm_srlv_epi16_widex_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    // See the 256-bit form: masked, not clamped.
    let s = _mm_and_si128(shifts, _mm_set1_epi16(15));
    let lo = _mm_srlv_epi32(_mm_unpacklo_epi16(value, z), _mm_unpacklo_epi16(s, z));
    let hi = _mm_srlv_epi32(_mm_unpackhi_epi16(value, z), _mm_unpackhi_epi16(s, z));
    _mm_packus_epi32(lo, hi)
}

/// POLYFILL: 128-bit `vpsravw` via 32-bit widening.
#[inline(always)]
pub unsafe fn _mm_srav_epi16_widex_v3(value: __m128i, shifts: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    // See the 256-bit form: masked, not clamped.
    let s = _mm_and_si128(shifts, _mm_set1_epi16(15));
    let lo = _mm_srav_epi32(
        _mm_srai_epi32(_mm_unpacklo_epi16(value, value), 16),
        _mm_unpacklo_epi16(s, z),
    );
    let hi = _mm_srav_epi32(
        _mm_srai_epi32(_mm_unpackhi_epi16(value, value), 16),
        _mm_unpackhi_epi16(s, z),
    );
    _mm_packs_epi32(lo, hi)
}

/// POLYFILL: low 8 bits of each byte product (`a[i].wrapping_mul(b[i])`).
#[inline(always)]
pub unsafe fn _mm256_mullo_epi8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let lo_mask = _mm256_set1_epi16(0x00FF);
    let even = _mm256_mullo_epi16(_mm256_and_si256(a, lo_mask), _mm256_and_si256(b, lo_mask));
    let odd = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8), _mm256_srli_epi16(b, 8));
    _mm256_or_si256(_mm256_and_si256(even, lo_mask), _mm256_slli_epi16(odd, 8))
}

/// POLYFILL: high 8 bits of each signed byte product.
#[inline(always)]
pub unsafe fn _mm256_mulhi_epi8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let a_lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(zero, a), 8);
    let b_lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(zero, b), 8);
    let a_hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(zero, a), 8);
    let b_hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(zero, b), 8);
    let p_lo = _mm256_srai_epi16(_mm256_mullo_epi16(a_lo, b_lo), 8);
    let p_hi = _mm256_srai_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
    _mm256_packs_epi16(p_lo, p_hi)
}

/// POLYFILL: high 8 bits of each unsigned byte product.
#[inline(always)]
pub unsafe fn _mm256_mulhi_epu8x_v3(a: __m256i, b: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let a_lo = _mm256_unpacklo_epi8(a, zero);
    let b_lo = _mm256_unpacklo_epi8(b, zero);
    let a_hi = _mm256_unpackhi_epi8(a, zero);
    let b_hi = _mm256_unpackhi_epi8(b, zero);
    let p_lo = _mm256_srli_epi16(_mm256_mullo_epi16(a_lo, b_lo), 8);
    let p_hi = _mm256_srli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
    _mm256_packus_epi16(p_lo, p_hi)
}

#[inline(always)]
pub unsafe fn _mm256_zeroupper_mask_epi64<Z: ZeroUpper>() -> __m256i {
    use crate::element::MaskElement;

    _mm256_setr_epi64x(
        i64::from_bool(0 < Z::N),
        i64::from_bool(1 < Z::N),
        i64::from_bool(2 < Z::N),
        i64::from_bool(3 < Z::N),
    )
}

// ---------------------------------------------------------------------------
// Mask population counts, 256-bit. See the `_v1` versions in
// `x86_v1::polyfills::bits` for why the saturating pack is legal here and why
// it is legal ONLY for a population count.
//
// One wrinkle over the 128-bit case: `_mm256_packs_epi32` packs per 128-bit
// half, so the result is [a.lo, b.lo, a.hi, b.hi] in 64-bit groups rather than
// [a, b]. The usual fix is a `vpermq` restitch (see the `CastRegister`
// impl for `I16x16V3`); a popcount does not care, so this deliberately skips it
// - that permute is exactly the instruction the whole exercise is here to save.
// ---------------------------------------------------------------------------

/// POLYFILL: set lanes across two 32-bit-lane masks.
///
/// Narrows to 16 16-bit lanes, so `vpmovmskb` reports 2 bits per lane.
#[inline(always)]
pub unsafe fn _mm256_count_mask2_epi32x_v3(a: __m256i, b: __m256i) -> usize {
    (_mm256_movemask_epi8(_mm256_packs_epi32(a, b)) as u32).count_ones() as usize / 2
}

/// POLYFILL: set lanes across four 32-bit-lane masks.
///
/// Two levels of packing land 32 lanes on 32 8-bit lanes, so `vpmovmskb`
/// reports exactly one bit per lane and no division is needed.
#[inline(always)]
pub unsafe fn _mm256_count_mask4_epi32x_v3(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> usize {
    let lo = _mm256_packs_epi32(a, b);
    let hi = _mm256_packs_epi32(c, d);
    (_mm256_movemask_epi8(_mm256_packs_epi16(lo, hi)) as u32).count_ones() as usize
}

/// POLYFILL: set lanes across two 16-bit-lane masks. Exact, one bit per lane.
#[inline(always)]
pub unsafe fn _mm256_count_mask2_epi16x_v3(a: __m256i, b: __m256i) -> usize {
    (_mm256_movemask_epi8(_mm256_packs_epi16(a, b)) as u32).count_ones() as usize
}

/// POLYFILL: total set lanes across `N` 32-bit-lane masks.
///
/// Descending ladder: fours, then a pair, then a single. Four is the ceiling
/// for 32-bit lanes - two narrowing steps reach 8-bit lanes, which is the floor
/// (`vpmovmskb` is already one bit per byte, and there is nothing narrower to
/// pack to). Larger `N` is therefore chunks of four, which is optimal: OR-ing
/// several chunks' bitmasks into one word to save popcounts costs exactly the
/// shift-and-or it saves.
#[inline(always)]
pub unsafe fn _mm256_count_mask_epi32x_v3<const N: usize>(values: [__m256i; N]) -> usize {
    let mut total = 0;
    let mut i = 0;

    while i + 3 < N {
        total += _mm256_count_mask4_epi32x_v3(values[i], values[i + 1], values[i + 2], values[i + 3]);
        i += 4;
    }

    if i + 1 < N {
        total += _mm256_count_mask2_epi32x_v3(values[i], values[i + 1]);
        i += 2;
    }

    if i < N {
        total += (_mm256_movemask_ps(_mm256_castsi256_ps(values[i])) as u32).count_ones() as usize;
    }

    total
}

/// POLYFILL: total set lanes across `N` 16-bit-lane masks.
#[inline(always)]
pub unsafe fn _mm256_count_mask_epi16x_v3<const N: usize>(values: [__m256i; N]) -> usize {
    let mut total = 0;
    let mut i = 0;

    while i + 1 < N {
        total += _mm256_count_mask2_epi16x_v3(values[i], values[i + 1]);
        i += 2;
    }

    if i < N {
        // 16 16-bit lanes -> 2 bits per lane out of `vpmovmskb`.
        total += (_mm256_movemask_epi8(values[i]) as u32).count_ones() as usize / 2;
    }

    total
}

/// POLYFILL: [`_mm256_count_mask_epi32x_v3`] for masks held in float registers.
#[inline(always)]
pub unsafe fn _mm256_count_mask_ps_v3<const N: usize>(values: [__m256; N]) -> usize {
    let mut ints = [_mm256_setzero_si256(); N];

    let mut i = 0;
    while i < N {
        ints[i] = _mm256_castps_si256(values[i]);
        i += 1;
    }

    _mm256_count_mask_epi32x_v3(ints)
}

// ---------------------------------------------------------------------------
// VPSHUFB byte-table lookup (`Register::lookup` on the 256-bit byte registers).
// ---------------------------------------------------------------------------

/// Scalar fallback for [`_mm256_lookup_epi8x_v3`]: bounds-checked, same as the
/// `Register::lookup` trait default.
#[inline(always)]
unsafe fn _mm256_lookup_scalar_epi8x_v3(values: *const u8, len: usize, idxs: __m256i) -> __m256i {
    let table = core::slice::from_raw_parts(values, len);

    let mut sel = [0u8; 32];
    _mm256_storeu_si256(sel.as_mut_ptr() as *mut __m256i, idxs);

    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = table[sel[i] as usize];
        i += 1;
    }

    _mm256_loadu_si256(out.as_ptr() as *const __m256i)
}

/// Byte-table lookup: `result[i] = values[idx[i]]`, for tables of exactly
/// 16/32/48/64 entries.
///
/// - 16: `vpshufb` is per-128-bit-lane, so broadcasting the table gives both
///   lanes the same 16 entries, two instructions.
/// - 32: one [`_mm256_permutev_epi8x_v3`] over the loaded table (the
///   swapped-halves + bit-4 select idiom, since `vpermb` is AVX-512).
/// - 48/64: the 32-entry block plus a second block (broadcast 16 or a second
///   full permute), selected by bit 5 of the index.
///
/// `Register::lookup`'s safety contract requires every index to be in range, so
/// bit 7 of each control byte is always clear and `idx & 15` is the within-block
/// offset, and indices are deliberately NOT clamped or masked. (Divergence note, as
/// on NEON's `vqtbl` overrides: an out-of-range index shuffles rather than
/// panicking like the scalar default, which the `unsafe fn` contract permits.)
///
/// `_mm256_slli_epi16::<2>` lifts bit 5 of each index byte into that byte's MSB,
/// the only bit `vpblendvb` inspects.
///
/// Any other length falls back to the scalar bounds-checked loop.
#[inline(always)]
pub unsafe fn _mm256_lookup_epi8x_v3(values: *const u8, len: usize, idxs: __m256i) -> __m256i {
    let p16 = values as *const __m128i;
    let p32 = values as *const __m256i;

    match len {
        16 => _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128(p16)), idxs),
        32 => _mm256_permutev_epi8x_v3(_mm256_loadu_si256(p32), idxs),
        48 => {
            let lo = _mm256_permutev_epi8x_v3(_mm256_loadu_si256(p32), idxs);
            let hi = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128(p16.add(2))), idxs);
            _mm256_blendv_epi8(lo, hi, _mm256_slli_epi16::<2>(idxs))
        }
        64 => {
            let lo = _mm256_permutev_epi8x_v3(_mm256_loadu_si256(p32), idxs);
            let hi = _mm256_permutev_epi8x_v3(_mm256_loadu_si256(p32.add(1)), idxs);
            _mm256_blendv_epi8(lo, hi, _mm256_slli_epi16::<2>(idxs))
        }
        _ => _mm256_lookup_scalar_epi8x_v3(values, len, idxs),
    }
}

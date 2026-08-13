use super::*;

/// POLYFILL: full-range `u32x4 -> f32x4` conversion.
///
/// Same magic-number algorithm as the v2 version, but the SSE4.1
/// `_mm_blend_epi16` word blends are replaced by and/or merges (the merged
/// halves never overlap, so plain bitwise ops suffice).
#[inline(always)]
pub unsafe fn _mm_cvtepu32_psx_v1(x: __m128i) -> __m128 {
    // low 16 bits of each lane with exponent 2^23
    let lo = _mm_or_si128(_mm_and_si128(x, _mm_set1_epi32(0xFFFF)), _mm_set1_epu32x(0x4B000000));
    // high 16 bits of each lane with exponent 2^39
    let hi = _mm_or_si128(_mm_srli_epi32(x, 16), _mm_set1_epu32x(0x53000000));

    // subtract the combined bias, then add the halves
    let hi = _mm_sub_ps(_mm_castsi128_ps(hi), _mm_set1_ps(f32::from_bits(0x53000080)));
    _mm_add_ps(_mm_castsi128_ps(lo), hi)
}

/// POLYFILL: `f32x4 -> u32x4` conversion (truncating, matching `f32 as u32` in range).
///
/// Same as the v2 version with the `_mm_blendv_ps` replaced by a bitwise select.
#[inline(always)]
pub unsafe fn _mm_cvtps_epu32x_v1(x: __m128) -> __m128i {
    let bound = _mm_set1_ps(f32::from_bits(0x4f000000)); // 2^31
    let in_range = _mm_cmplt_ps(x, bound);

    // high range: subtract 2^31, convert, then flip the MSB back on
    let offset = _mm_xor_si128(_mm_cvttps_epi32(_mm_sub_ps(x, bound)), _mm_set1_epu32x(0x80000000));
    let direct = _mm_cvttps_epi32(x);

    _mm_castps_si128(_mm_blendv_psx_v1(
        _mm_castsi128_ps(offset),
        _mm_castsi128_ps(direct),
        in_range,
    ))
}

/// POLYFILL: full-range `f64x2 -> u32x2` conversion (truncating). SSE2-clean
/// copy of the v2 algorithm.
#[inline(always)]
pub unsafe fn _mm_cvtpd_epu32x_v1(xmm0: __m128d) -> __m128i {
    // 1. Threshold: 2^31
    let bound = _mm_set1_pd(f64::from_bits(0x41e0000000000000));

    // 2. Mask: xmm0 < 2^31, as 64-bit lane masks
    let cmp_mask = _mm_cmplt_pd(xmm0, bound);

    // 3. Align the 64-bit masks to the packed 32-bit output lanes
    let mask_32 = _mm_shuffle_epi32(_mm_castpd_si128(cmp_mask), 0b11_11_10_00);

    // 4. High-range path (offset conversion)
    let offset_converted = _mm_xor_si128(
        _mm_cvttpd_epi32(_mm_sub_pd(xmm0, bound)),
        _mm_set1_epi32(0x80000000u32 as i32),
    );

    // 5. Low-range path (direct conversion)
    let direct_converted = _mm_cvttpd_epi32(xmm0);

    // 6. Bitwise blend: (direct & mask) | (offset & ~mask)
    _mm_or_si128(
        _mm_and_si128(mask_32, direct_converted),
        _mm_andnot_si128(mask_32, offset_converted),
    )
}

/// POLYFILL: full-range `f64x2 -> i64x2` conversion (truncating) via two
/// scalar `cvttsd` instructions (SSE2, x86-64 only - same as the v2 version).
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn _mm_cvtpd_epi64x_v1(x: __m128d) -> __m128i {
    let x0 = _mm_cvttsd_si64(x);
    let x1 = _mm_cvttsd_si64(_mm_shuffle_pd(x, x, 0b11));

    _mm_set_epi64x(x1, x0)
}

/// POLYFILL: full-range `f64x2 -> u64x2` conversion (truncating, matching
/// `f64 as u64` for in-range values). Lanes >= 2^63 are offset into the signed
/// range before the scalar converts, then the high bit is OR'd back on.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn _mm_cvtpd_epu64x_v1(x: __m128d) -> __m128i {
    let bound = _mm_set1_pd(9223372036854775808.0); // 2^63
    let big = _mm_cmpge_pd(x, bound);
    let xs = _mm_sub_pd(x, _mm_and_pd(big, bound));

    _mm_or_si128(
        _mm_cvtpd_epi64x_v1(xs),
        _mm_and_si128(_mm_castpd_si128(big), _mm_set1_epi64x(i64::MIN)),
    )
}

/// POLYFILL: `f32x4 -> i32x4` saturating cast (`f32 as i32` semantics: NaN -> 0,
/// out-of-range clamps). `cvttps2dq` already returns `i32::MIN` for NaN and both
/// overflow directions, which is correct for negative overflow; positive overflow
/// flips to `i32::MAX` with one XOR against the compare mask (MIN ^ !0 == MAX),
/// and NaN lanes are zeroed with an ANDNOT.
#[inline(always)]
pub unsafe fn _mm_cvtps_epi32_satx_v1(x: __m128) -> __m128i {
    let t = _mm_cvttps_epi32(x);
    let hi = _mm_castps_si128(_mm_cmpge_ps(x, _mm_set1_ps(2147483648.0))); // 2^31
    let nan = _mm_castps_si128(_mm_cmpunord_ps(x, x));

    _mm_andnot_si128(nan, _mm_xor_si128(t, hi))
}

/// POLYFILL: `f32x4 -> u32x4` saturating cast (`f32 as u32`: NaN and negatives
/// -> 0, >= 2^32 -> `u32::MAX`). `maxps` returns its second operand on unordered
/// inputs, handling NaN and the low clamp in one instruction; lanes >= 2^32 come
/// out of the in-range converter as 0 and are forced to MAX with an OR.
#[inline(always)]
pub unsafe fn _mm_cvtps_epu32_satx_v1(x: __m128) -> __m128i {
    let x0 = _mm_max_ps(x, _mm_setzero_ps());
    let t = _mm_cvtps_epu32x_v1(x0);
    let hi = _mm_castps_si128(_mm_cmpge_ps(x0, _mm_set1_ps(4294967296.0))); // 2^32

    _mm_or_si128(t, hi)
}

/// POLYFILL: `f64x2 -> i64x2` saturating cast (`f64 as i64`). Same shape as
/// [`_mm_cvtps_epi32_satx_v1`]: scalar `cvttsd` gives `i64::MIN` for NaN and
/// overflow, positive overflow XORs to MAX, NaN zeroes.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn _mm_cvtpd_epi64_satx_v1(x: __m128d) -> __m128i {
    let t = _mm_cvtpd_epi64x_v1(x);
    let hi = _mm_castpd_si128(_mm_cmpge_pd(x, _mm_set1_pd(9223372036854775808.0))); // 2^63
    let nan = _mm_castpd_si128(_mm_cmpunord_pd(x, x));

    _mm_andnot_si128(nan, _mm_xor_si128(t, hi))
}

/// POLYFILL: `f64x2 -> u64x2` saturating cast (`f64 as u64`).
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub unsafe fn _mm_cvtpd_epu64_satx_v1(x: __m128d) -> __m128i {
    let x0 = _mm_max_pd(x, _mm_setzero_pd()); // NaN and negatives -> 0
    let t = _mm_cvtpd_epu64x_v1(x0);
    let hi = _mm_castpd_si128(_mm_cmpge_pd(x0, _mm_set1_pd(18446744073709551616.0))); // 2^64

    _mm_or_si128(t, hi)
}

/// POLYFILL: full-range `u64x2 -> f64x2` conversion.
///
/// Same magic-number algorithm as the v2 version; the `_mm_blend_epi16`
/// (select the low dword of each lane from `v`) is replaced by an and/or
/// merge, since `magic_i_lo` has a zero low dword.
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepu64_pdx_v1(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000);  // 2^52        encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000000000000);  // 2^84        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000000100000);  // 2^84 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_or_si128(_mm_and_si128(v, _mm_set1_epi64x(0xFFFFFFFF)), magic_i_lo);
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Blend v_hi with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

/// POLYFILL: full-range `i64x2 -> f64x2` conversion. See [`_mm_cvtepu64_pdx_v1`].
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepi64_pdx_v1(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_or_si128(_mm_and_si128(v, _mm_set1_epi64x(0xFFFFFFFF)), magic_i_lo);
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

/// Exact for every `u32` and branchless, via the standard magic-constant trick:
/// a `u32` fits in a `f64` mantissa with room to spare, so zero-extending it
/// into the low bits of the `f64` encoding of `2^52` yields exactly `2^52 + x`,
/// and one subtraction recovers `x`. No rounding step is involved, so this is
/// bit-exact rather than merely accurate.
#[inline(always)]
pub unsafe fn _mm_cvtepu32_pdx_v1(v: __m128i) -> __m128d {
    let magic_i = _mm_set1_epi64x(0x4330000000000000u64 as i64); // 2^52 as f64 bits
    let zext = _mm_cvtepu32_epi64x_v1(v);
    let biased = _mm_or_si128(zext, magic_i);

    _mm_sub_pd(_mm_castsi128_pd(biased), _mm_castsi128_pd(magic_i))
}

/// POLYFILL: `_mm_cvtepi32_epi64` (SSE4.1 `pmovsxdq`) - sign-extend the low
/// two `i32` lanes to `i64`.
#[inline(always)]
pub unsafe fn _mm_cvtepi32_epi64x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi32(v, _mm_srai_epi32(v, 31))
}

/// POLYFILL: `_mm_cvtepu32_epi64` (SSE4.1 `pmovzxdq`) - zero-extend the low
/// two `u32` lanes to `u64`.
#[inline(always)]
pub unsafe fn _mm_cvtepu32_epi64x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi32(v, _mm_setzero_si128())
}

/// POLYFILL: build a full-width `i32` lane mask from 4 `bool`s.
#[inline(always)]
pub unsafe fn _mm_cvtboolx4_to_epi32_mask_v1(
    value: generic_array::GenericArray<bool, generic_array::typenum::U4>,
) -> __m128i {
    // -(b as i32) is 0 or all-ones; branchless and SSE-level agnostic
    _mm_setr_epi32(
        -(value[0] as i32),
        -(value[1] as i32),
        -(value[2] as i32),
        -(value[3] as i32),
    )
}

/// POLYFILL: build a full-width `i64` lane mask from 2 `bool`s.
#[inline(always)]
pub unsafe fn _mm_cvtboolx2_to_epi64_mask_v1(
    value: generic_array::GenericArray<bool, generic_array::typenum::U2>,
) -> __m128i {
    _mm_setr_epi64x(-(value[0] as i64), -(value[1] as i64))
}

/// POLYFILL: `_mm_extract_epi32` (SSE4.1 `pextrd`) via shuffle + `movd`.
///
/// `IMM8` must be in `0..=3`; passed directly as the shuffle immediate it
/// selects lane `IMM8` into position 0 (the upper index fields are don't-cares).
#[inline(always)]
pub unsafe fn _mm_extract_epi32x_v1<const IMM8: i32>(v: __m128i) -> i32 {
    _mm_cvtsi128_si32(_mm_shuffle_epi32::<IMM8>(v))
}

/// POLYFILL: `_mm_extract_ps` (SSE4.1 `extractps`) returning the raw bits like the intrinsic.
#[inline(always)]
pub unsafe fn _mm_extract_psx_v1<const IMM8: i32>(v: __m128) -> i32 {
    _mm_extract_epi32x_v1::<IMM8>(_mm_castps_si128(v))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64x_limited_v1(mut x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(0x0018000000000000u64 as i64 as f64);
    x = _mm_add_pd(x, m);
    _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(m))
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm_cvtpd_epu64x_limited_v1(x: __m128d) -> __m128i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm_castpd_si128(_mm_xor_pd(_mm_add_pd(x, m), m))
}

/// Only works for inputs in the range: [-2^51, 2^51]
#[inline(always)]
pub unsafe fn _mm_cvtepi64_pdx_limited_v1(x: __m128i) -> __m128d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm_set1_pd(0x0018000000000000u64 as i64 as f64);
    _mm_sub_pd(_mm_castsi128_pd(_mm_add_epi64(x, _mm_castpd_si128(m))), m)
}

/// Only works for inputs in the range: [0, 2^52)
#[inline(always)]
pub unsafe fn _mm_cvtepu64_pdx_limited_v1(x: __m128i) -> __m128d {
    // https://stackoverflow.com/a/41223013/2083075
    let m = _mm_set1_pd(0x0010000000000000u64 as i64 as f64);
    _mm_sub_pd(_mm_castsi128_pd(_mm_or_si128(x, _mm_castpd_si128(m))), m)
}

#[inline(always)]
pub unsafe fn _mm_cvtepi64_epi32x_v1(a: __m128i, b: __m128i) -> __m128i {
    // Domain change might incur some performance penalty, but this is the simplest way to do it.
    _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), 0b10_00_10_00))
}

// ===========================================================================================
// Cross-family integer narrow/widen "instructions" that x86 lacks at this level. SSE2 has no
// `pmovsx`/`pmovzx` (SSE4.1) or `pshufb` (SSSE3), so the widens use unpack chains (sign/zero
// extend) and the narrows use the store-and-rebuild pattern (`_mm_setr_epi8`/`_mm_setr_epi16` of
// the low byte/word of each lane). These are the register-level polyfills behind the 8/16 <->
// 16/32/64 and 8/16 <-> f32/f64 casts in `registers/half8.rs` / `half16.rs`. Multi-output
// (`[__m128i; N]`) forms are treated as instructions that write more than one register.
// ===========================================================================================

// --- i8 sign helper ---

// i8 sign mask: all-ones where the byte is negative (SSE2, no pmovsx).
#[inline(always)]
pub unsafe fn _mm_signbits_epi8x_v1(v: __m128i) -> __m128i {
    _mm_cmpgt_epi8(_mm_setzero_si128(), v)
}

// --- 8 <-> 16 widen/narrow ---

// Sign-extend the low 8 bytes of `v` to 8x i16.
#[inline(always)]
pub unsafe fn _mm_cvtepi8_epi16x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi8(v, _mm_signbits_epi8x_v1(v))
}
// Zero-extend the low 8 bytes of `v` to 8x i16.
#[inline(always)]
pub unsafe fn _mm_cvtepu8_epi16x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi8(v, _mm_setzero_si128())
}
// Pack the low byte of 8 i16 lanes into the low 8 bytes (store-and-rebuild; no pshufb on SSE2).
#[inline(always)]
pub unsafe fn _mm_cvtepi16_epi8x_v1(v: __m128i) -> __m128i {
    let mut a = [0i16; 8];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v);
    _mm_setr_epi8(
        a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8, a[4] as i8, a[5] as i8, a[6] as i8, a[7] as i8, 0, 0, 0, 0, 0,
        0, 0, 0,
    )
}
// Narrow 2x i16x8 (16 i16) -> 16 i8 (low byte of each lane; store-and-rebuild).
#[inline(always)]
pub unsafe fn _mm_cvt2epi16_epi8x_v1(v: [__m128i; 2]) -> __m128i {
    let mut lo = [0i16; 8];
    let mut hi = [0i16; 8];
    _mm_storeu_si128(lo.as_mut_ptr() as *mut _, v[0]);
    _mm_storeu_si128(hi.as_mut_ptr() as *mut _, v[1]);
    _mm_setr_epi8(
        lo[0] as i8,
        lo[1] as i8,
        lo[2] as i8,
        lo[3] as i8,
        lo[4] as i8,
        lo[5] as i8,
        lo[6] as i8,
        lo[7] as i8,
        hi[0] as i8,
        hi[1] as i8,
        hi[2] as i8,
        hi[3] as i8,
        hi[4] as i8,
        hi[5] as i8,
        hi[6] as i8,
        hi[7] as i8,
    )
}

// --- 8 <-> 32 widen/narrow ---

// Sign-extend the low 4 bytes of `v` to 4x i32 (byte -> word -> dword).
#[inline(always)]
pub unsafe fn _mm_cvtepi8_epi32x_v1(v: __m128i) -> __m128i {
    let words = _mm_unpacklo_epi8(v, _mm_signbits_epi8x_v1(v)); // low 8 bytes -> 8x i16 (sign-extended)
    _mm_unpacklo_epi16(words, _mm_srai_epi16(words, 15)) // low 4 words -> 4x i32
}
// Zero-extend the low 4 bytes of `v` to 4x i32.
#[inline(always)]
pub unsafe fn _mm_cvtepu8_epi32x_v1(v: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, z), z)
}
// Narrow 4x i32x4 (16 i32) -> 16 i8 (low byte of each lane; store-and-rebuild).
#[inline(always)]
pub unsafe fn _mm_cvt4epi32_epi8x_v1(v: [__m128i; 4]) -> __m128i {
    let mut a = [0i32; 4];
    let mut b = [0i32; 4];
    let mut c = [0i32; 4];
    let mut d = [0i32; 4];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v[0]);
    _mm_storeu_si128(b.as_mut_ptr() as *mut _, v[1]);
    _mm_storeu_si128(c.as_mut_ptr() as *mut _, v[2]);
    _mm_storeu_si128(d.as_mut_ptr() as *mut _, v[3]);
    _mm_setr_epi8(
        a[0] as i8, a[1] as i8, a[2] as i8, a[3] as i8, b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8, c[0] as i8,
        c[1] as i8, c[2] as i8, c[3] as i8, d[0] as i8, d[1] as i8, d[2] as i8, d[3] as i8,
    )
}
// Widen the low 16 bytes of `v` to 4x i32x4 (sign-extend; writes four regs).
#[inline(always)]
pub unsafe fn _mm_cvtepi8_4epi32x_v1(v: __m128i) -> [__m128i; 4] {
    [
        _mm_cvtepi8_epi32x_v1(v),
        _mm_cvtepi8_epi32x_v1(_mm_srli_si128(v, 4)),
        _mm_cvtepi8_epi32x_v1(_mm_srli_si128(v, 8)),
        _mm_cvtepi8_epi32x_v1(_mm_srli_si128(v, 12)),
    ]
}
// Widen the low 16 bytes of `v` to 4x u32x4 (zero-extend; writes four regs).
#[inline(always)]
pub unsafe fn _mm_cvtepu8_4epi32x_v1(v: __m128i) -> [__m128i; 4] {
    [
        _mm_cvtepu8_epi32x_v1(v),
        _mm_cvtepu8_epi32x_v1(_mm_srli_si128(v, 4)),
        _mm_cvtepu8_epi32x_v1(_mm_srli_si128(v, 8)),
        _mm_cvtepu8_epi32x_v1(_mm_srli_si128(v, 12)),
    ]
}

// --- 8 <-> 64 widen/narrow ---

// Sign-extend two i8 lanes into an I64x2V1 (high-then-low arg order for `_mm_set_epi64x`).
#[inline(always)]
pub unsafe fn _mm_cvt2epi8_epi64x_v1(lo: i8, hi: i8) -> __m128i {
    _mm_set_epi64x(hi as i64, lo as i64)
}
// Zero-extend two u8 lanes into a U64x2V1.
#[inline(always)]
pub unsafe fn _mm_cvt2epu8_epi64x_v1(lo: u8, hi: u8) -> __m128i {
    _mm_set_epi64x(hi as i64, lo as i64)
}
// Narrow 4x i64x2 (8 i64) -> low 8 bytes (byte 0 of each lane; store-and-rebuild).
#[inline(always)]
pub unsafe fn _mm_cvt4epi64_epi8x_v1(v: [__m128i; 4]) -> __m128i {
    let mut a = [0i64; 2];
    let mut b = [0i64; 2];
    let mut c = [0i64; 2];
    let mut d = [0i64; 2];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v[0]);
    _mm_storeu_si128(b.as_mut_ptr() as *mut _, v[1]);
    _mm_storeu_si128(c.as_mut_ptr() as *mut _, v[2]);
    _mm_storeu_si128(d.as_mut_ptr() as *mut _, v[3]);
    _mm_setr_epi8(
        a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8, 0, 0, 0, 0, 0,
        0, 0, 0,
    )
}
// Narrow 8x i64x2 (16 i64) -> 16 bytes (byte 0 of each lane; store-and-rebuild).
#[inline(always)]
pub unsafe fn _mm_cvt8epi64_epi8x_v1(v: [__m128i; 8]) -> __m128i {
    let mut a = [0i64; 2];
    let mut b = [0i64; 2];
    let mut c = [0i64; 2];
    let mut d = [0i64; 2];
    let mut e = [0i64; 2];
    let mut f = [0i64; 2];
    let mut g = [0i64; 2];
    let mut h = [0i64; 2];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v[0]);
    _mm_storeu_si128(b.as_mut_ptr() as *mut _, v[1]);
    _mm_storeu_si128(c.as_mut_ptr() as *mut _, v[2]);
    _mm_storeu_si128(d.as_mut_ptr() as *mut _, v[3]);
    _mm_storeu_si128(e.as_mut_ptr() as *mut _, v[4]);
    _mm_storeu_si128(f.as_mut_ptr() as *mut _, v[5]);
    _mm_storeu_si128(g.as_mut_ptr() as *mut _, v[6]);
    _mm_storeu_si128(h.as_mut_ptr() as *mut _, v[7]);
    _mm_setr_epi8(
        a[0] as i8, a[1] as i8, b[0] as i8, b[1] as i8, c[0] as i8, c[1] as i8, d[0] as i8, d[1] as i8, e[0] as i8,
        e[1] as i8, f[0] as i8, f[1] as i8, g[0] as i8, g[1] as i8, h[0] as i8, h[1] as i8,
    )
}

// --- 16 <-> 32 widen ---

// Sign-extend the low 4 i16 lanes of `v` to 4x i32 (SSE2 unpack, no pmovsx).
#[inline(always)]
pub unsafe fn _mm_cvtepi16_epi32x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi16(v, _mm_srai_epi16(v, 15))
}
// Zero-extend the low 4 u16 lanes of `v` to 4x i32.
#[inline(always)]
pub unsafe fn _mm_cvtepu16_epi32x_v1(v: __m128i) -> __m128i {
    _mm_unpacklo_epi16(v, _mm_setzero_si128())
}

// --- 16 <-> 64 widen/narrow ---

// Sign-extend two i16 lanes into an I64x2V1 (high-then-low arg order for `_mm_set_epi64x`).
#[inline(always)]
pub unsafe fn _mm_cvt2epi16_epi64x_v1(lo: i16, hi: i16) -> __m128i {
    _mm_set_epi64x(hi as i64, lo as i64)
}
// Zero-extend two u16 lanes into a U64x2V1.
#[inline(always)]
pub unsafe fn _mm_cvt2epu16_epi64x_v1(lo: u16, hi: u16) -> __m128i {
    _mm_set_epi64x(hi as i64, lo as i64)
}
// Narrow 4x i64x2 (8 i64) -> 8 words (word 0 of each lane; store-and-rebuild).
#[inline(always)]
pub unsafe fn _mm_cvt4epi64_epi16x_v1(v: [__m128i; 4]) -> __m128i {
    let mut a = [0i64; 2];
    let mut b = [0i64; 2];
    let mut c = [0i64; 2];
    let mut d = [0i64; 2];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v[0]);
    _mm_storeu_si128(b.as_mut_ptr() as *mut _, v[1]);
    _mm_storeu_si128(c.as_mut_ptr() as *mut _, v[2]);
    _mm_storeu_si128(d.as_mut_ptr() as *mut _, v[3]);
    _mm_setr_epi16(
        a[0] as i16,
        a[1] as i16,
        b[0] as i16,
        b[1] as i16,
        c[0] as i16,
        c[1] as i16,
        d[0] as i16,
        d[1] as i16,
    )
}
// Widen the low 8 i16 lanes of `v` to 4x i64x2 (sign-extend; writes four regs).
#[inline(always)]
pub unsafe fn _mm_cvtepi16_4epi64x_v1(v: __m128i) -> [__m128i; 4] {
    let mut a = [0i16; 8];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v);
    [
        _mm_cvt2epi16_epi64x_v1(a[0], a[1]),
        _mm_cvt2epi16_epi64x_v1(a[2], a[3]),
        _mm_cvt2epi16_epi64x_v1(a[4], a[5]),
        _mm_cvt2epi16_epi64x_v1(a[6], a[7]),
    ]
}
// Widen the low 8 u16 lanes of `v` to 4x u64x2 (zero-extend; writes four regs).
#[inline(always)]
pub unsafe fn _mm_cvtepu16_4epi64x_v1(v: __m128i) -> [__m128i; 4] {
    let mut a = [0u16; 8];
    _mm_storeu_si128(a.as_mut_ptr() as *mut _, v);
    [
        _mm_cvt2epu16_epi64x_v1(a[0], a[1]),
        _mm_cvt2epu16_epi64x_v1(a[2], a[3]),
        _mm_cvt2epu16_epi64x_v1(a[4], a[5]),
        _mm_cvt2epu16_epi64x_v1(a[6], a[7]),
    ]
}

// --- f64 <-> i32 fan-out/fan-in (shared by half8.rs and half16.rs) ---

// Truncate an F64x2V1 to its low 2 lanes as [i32; 2] (the low byte/word of each is the narrowed value).
#[inline(always)]
pub unsafe fn _mm_cvttpd_2i32x_v1(v: __m128d) -> [i32; 2] {
    let mut d = [0i32; 4];
    _mm_storeu_si128(d.as_mut_ptr() as *mut _, _mm_cvttpd_epi32(v));
    [d[0], d[1]]
}
// Convert one i32x4 into 2 F64x2V1 (low 2 lanes, high 2 lanes).
#[inline(always)]
pub unsafe fn _mm_cvtepi32_2pdx_v1(ints: __m128i) -> [__m128d; 2] {
    [_mm_cvtepi32_pd(ints), _mm_cvtepi32_pd(_mm_srli_si128(ints, 8))]
}

// Unsigned twin of the above. `cvtdq2pd` is signed-only at every x86 level, so
// this goes through the magic-constant conversion rather than the instruction.
#[inline(always)]
pub unsafe fn _mm_cvtepu32_2pdx_v1(ints: __m128i) -> [__m128d; 2] {
    [_mm_cvtepu32_pdx_v1(ints), _mm_cvtepu32_pdx_v1(_mm_srli_si128(ints, 8))]
}

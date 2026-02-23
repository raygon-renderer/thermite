use super::*;

#[inline(always)]
pub unsafe fn _mm_cvtepu32_psx_v2(x: __m128i) -> __m128 {
    let xmm0 = x;
    let xmm1 = _mm_set1_epu32x(0x4B000000);
    let xmm1 = _mm_blend_epi16(xmm0, xmm1, 170);
    let xmm0 = _mm_srli_epi32(xmm0, 16);
    let xmm2 = _mm_set1_epu32x(0x53000000);
    let xmm0 = _mm_castsi128_ps(_mm_blend_epi16(xmm0, xmm2, 170));
    let xmm2 = _mm_set1_ps(f32::from_bits(0x53000080));
    let xmm0 = _mm_sub_ps(xmm0, xmm2);
    let xmm0 = _mm_add_ps(_mm_castsi128_ps(xmm1), xmm0);

    xmm0
}

#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64x_v2(x: __m128d) -> __m128i {
    let x0 = _mm_cvttsd_si64(x);
    let x1 = _mm_cvttsd_si64(_mm_shuffle_pd(x, x, 0b11));

    _mm_set_epi64x(x1, x0)
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepu64_pdx_v2(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000);  // 2^52        encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000000000000);  // 2^84        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000000100000);  // 2^84 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_blend_epi16(magic_i_lo, v, 0b00110011);      // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Blend v_hi with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm_cvtepi64_pdx_v2(v: __m128i) -> __m128d {
    let magic_i_lo   = _mm_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm_castsi128_pd(magic_i_all);

    let     v_lo     = _mm_blend_epi16(magic_i_lo, v, 0b00110011);      // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm_srli_epi64(v, 32);                           // Extract the 32 most significant bits of v
            v_hi     = _mm_xor_si128(v_hi, magic_i_hi32);               // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline(always)]
pub unsafe fn _mm_cvtps_epu32x_v2(x: __m128) -> __m128i {
    // TODO: This is exactly what LLVM generates for `simd_cast(f32x4 -> u32x4)`, but it's not ideal and
    // produces different results from `f32 as u32` with negative values and values larger than some value
    let xmm0 = x;
    let xmm1 = _mm_set1_ps(f32::from_bits(0x4f000000));
    let xmm2 = _mm_cmplt_ps(xmm0, xmm1);
    let xmm1 = _mm_sub_ps(xmm0, xmm1);
    let xmm1 = _mm_cvtps_epi32(xmm1);
    let xmm3 = _mm_set1_epu32x(0x80000000);
    let xmm1 = _mm_xor_si128(xmm1, xmm3);
    let xmm0 = _mm_cvtps_epi32(xmm0);
    let xmm0 = _mm_blendv_ps(_mm_castsi128_ps(xmm1), _mm_castsi128_ps(xmm0), xmm2);

    _mm_castps_si128(xmm0)
}

#[inline(always)]
pub unsafe fn _mm_cvtboolx4_to_epi32_mask_v2(
    value: generic_array::GenericArray<bool, generic_array::typenum::U4>,
) -> __m128i {
    let value: [u8; 4] = core::mem::transmute(value);

    #[rustfmt::skip]
    let mask = _mm_setr_epi8(
        value[0] as i8,
        value[1] as i8,
        value[2] as i8,
        value[3] as i8,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    );

    // take 1-byte mask, convert it to epi32, then compare it with zero to fill gaps
    _mm_cmpgt_epi32(_mm_cvtepi8_epi32(mask), _mm_setzero_si128())
}

#[inline(always)]
pub unsafe fn _mm_cvtboolx2_to_epi64_mask_v2(
    value: generic_array::GenericArray<bool, generic_array::typenum::U2>,
) -> __m128i {
    let value: [u8; 2] = core::mem::transmute(value);

    #[rustfmt::skip]
    let mask = _mm_setr_epi8(
        value[0] as i8,
        value[1] as i8,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    );

    // take 1-byte mask, convert it to epi64, then compare it with zero to fill gaps
    _mm_cmpgt_epi64(_mm_cvtepi8_epi64(mask), _mm_setzero_si128())
}

#[inline(always)]
pub unsafe fn _mm_cvtepi64_epi32x_v2(a: __m128i, b: __m128i) -> __m128i {
    // a = [ a3 | a2 | a1 | a0 ] (32-bit dwords)
    // 64-bit ints are [a3|a2] and [a1|a0]. We want a2 and a0.

    // b = [ b3 | b2 | b1 | b0 ]
    // 64-bit ints are [b3|b2] and [b1|b0]. We want b2 and b0.

    // 1. Create a vector with [a2|a0] in the low 64 bits.
    // _MM_SHUFFLE(z, y, x, w) creates [ a[z] | a[y] | a[x] | a[w] ]
    // We use _MM_SHUFFLE(3, 2, 2, 0) to create [ a3 | a2 | a2 | a0 ].
    // The low 64 bits are [a2|a0].
    let a_shuffled = _mm_shuffle_epi32::<{ MM_SHUFFLE!(3, 2, 2, 0) }>(a);

    // 2. Create a vector with [b2|b0] in the high 64 bits.
    // We use _MM_SHUFFLE(2, 0, 1, 0) to create [ b2 | b0 | b1 | b0 ].
    // The high 64 bits are [b2|b0].
    let b_shuffled = _mm_shuffle_epi32::<{ MM_SHUFFLE!(2, 0, 1, 0) }>(b);

    // 3. Blend them.
    // The mask 0xF0 = 0b11110000 selects 16-bit words.
    // It takes the high 4 words (64 bits) from b_shuffled.
    // It takes the low 4 words (64 bits) from a_shuffled.
    // Result: [ b_shuffled_high64 | a_shuffled_low64 ]
    // Result: [ b2 | b0 | a2 | a0 ]
    _mm_blend_epi16(a_shuffled, b_shuffled, 0xF0)
}

#[inline(always)]
pub unsafe fn _mm_cvtpd_epu32x_v2(xmm0: __m128d) -> __m128i {
    // 1. Threshold: 2^31
    let bound = _mm_set1_pd(f64::from_bits(0x41e0000000000000));

    // 2. Generate Mask: xmm0 < 2^31
    // Yields [M0, M1] where M_n is a 64-bit mask (0xFFFFFFFFFFFFFFFF or 0)
    let cmp_mask = _mm_cmplt_pd(xmm0, bound);

    // 3. Shuffle Mask: Align 64-bit masks to 32-bit output lanes
    // cmp_mask in 32-bit chunks is conceptually [M0_lo, M0_hi, M1_lo, M1_hi].
    // We shuffle it to [M0_lo, M1_lo, M1_hi, M1_hi] using 0b11_11_10_00.
    let mask_32 = _mm_shuffle_epi32(_mm_castpd_si128(cmp_mask), 0b11_11_10_00);

    // 4. High-Range Path (Offset Conversion)
    let offset_f64 = _mm_sub_pd(xmm0, bound);
    let offset_i32 = _mm_cvttpd_epi32(offset_f64);
    let sign_flip = _mm_set1_epi32(0x80000000u32 as i32);
    let offset_converted = _mm_xor_si128(offset_i32, sign_flip);

    // 5. Low-Range Path (Direct Conversion)
    let direct_converted = _mm_cvttpd_epi32(xmm0);

    // 6. Bitwise Blend: (direct & mask) | (offset & ~mask)
    _mm_or_si128(
        _mm_and_si128(mask_32, direct_converted),
        _mm_andnot_si128(mask_32, offset_converted),
    )
}

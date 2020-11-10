use super::*;

macro_rules! _mm256_blend_epi32x {
    ($ymm0:expr, $ymm1:expr, $imm8:expr) => {
        _mm256_castps_si256(_mm256_blend_ps(
            _mm256_castsi256_ps($ymm0),
            _mm256_castsi256_ps($ymm0),
            $imm8,
        ))
    };
}

#[inline(always)]
unsafe fn split_si256(ymm0: __m256i) -> (__m128i, __m128i) {
    (_mm256_castsi256_si128(ymm0), _mm256_extractf128_si256(ymm0, 1))
}

#[inline(always)]
unsafe fn recombine_si256(xmm0: __m128i, xmm1: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(xmm0), xmm1, 1)
}

#[inline(always)]
pub unsafe fn _mm256_blendv_epi32x(ymm0: __m256i, ymm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(ymm0),
        _mm256_castsi256_ps(ymm0),
        _mm256_castsi256_ps(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm256_blendv_epi64x(ymm0: __m256i, ymm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(ymm0),
        _mm256_castsi256_pd(ymm0),
        _mm256_castsi256_pd(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm256_xor_si256x(ymm0: __m256i, ymm1: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(ymm0), _mm256_castsi256_ps(ymm0)))
}

#[inline(always)]
pub unsafe fn _mm256_and_si256x(ymm0: __m256i, ymm1: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(ymm0), _mm256_castsi256_ps(ymm0)))
}

#[inline(always)]
pub unsafe fn _mm256_or_si256x(ymm0: __m256i, ymm1: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(ymm0), _mm256_castsi256_ps(ymm0)))
}

// AVX1 version
#[inline(always)]
pub unsafe fn _mm256_cvtepu32_ps(ymm0: __m256i) -> __m256 {
    let xmm0 = _mm256_castsi256_si128(ymm0);

    let xmm1 = _mm_srli_epi32(xmm0, 16);
    let ymm1 = _mm256_castsi256_ps(_mm256_castsi128_si256(xmm1));

    let xmm2 = _mm256_extractf128_si256(ymm0, 1);
    let xmm2 = _mm_castsi128_ps(_mm_srli_epi32(xmm2, 16));

    let ymm1 = _mm256_insertf128_ps(ymm1, xmm2, 1);
    let ymm1 = _mm256_cvtepi32_ps(_mm256_castps_si256(ymm1));
    let ymm1 = _mm256_mul_ps(ymm1, _mm256_set1_ps(transmute(0x47800000u32)));
    let ymm0 = _mm256_and_si256x(ymm0, _mm256_set1_epi32(0xFFFF));

    let ymm0 = _mm256_cvtepi32_ps(ymm0);
    let ymm0 = _mm256_add_ps(ymm0, ymm1);

    ymm0
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64_limited(x: __m256d) -> __m256i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm256_set1_pd(transmute::<u64, i64>(0x0018000000000000) as f64);
    _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(x, m)), _mm256_castpd_si256(m))
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu64_limited(x: __m256d) -> __m256i {
    // https://stackoverflow.com/a/41148578/2083075
    let m = _mm256_set1_pd(transmute::<u64, i64>(0x0010000000000000) as f64);
    _mm256_xor_si256x(_mm256_castpd_si256(_mm256_add_pd(x, m)), _mm256_castpd_si256(m))
}

#[inline(always)]
unsafe fn _mm256_srli32_epi64x(ymm0: __m256i) -> __m256i {
    let (low, high) = split_si256(ymm0);

    recombine_si256(_mm_srli_epi64(low, 32), _mm_srli_epi64(high, 32))
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm256_cvtepu64_pd(v: __m256i) -> __m256d {
    let magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);  // 2^52        encoded as floating-point
    let magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);  // 2^84        encoded as floating-point
    let magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);  // 2^84 + 2^52 encoded as floating-point
    let magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    let     v_lo     = _mm256_blend_epi32x!(magic_i_lo, v, 0b01010101);       // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm256_srli32_epi64x(v /*, 32*/);                 // Extract the 32 most significant bits of v
            v_hi     = _mm256_xor_si256x(v_hi, magic_i_hi32);       // Blend v_hi with 0x45300000
    let     v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

// https://stackoverflow.com/a/41223013/2083075
#[inline(always)]
#[rustfmt::skip]
pub unsafe fn _mm256_cvtepi64_pd(v: __m256i) -> __m256d {
    let magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000); // 2^52               encoded as floating-point
    let magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000); // 2^84 + 2^63        encoded as floating-point
    let magic_i_all  = _mm256_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52 encoded as floating-point
    let magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    let     v_lo     = _mm256_blend_epi32x!(magic_i_lo, v, 0b01010101);       // Blend the 32 lowest significant bits of v with magic_int_lo
    let mut v_hi     = _mm256_srli32_epi64x(v /*, 32*/);                 // Extract the 32 most significant bits of v
            v_hi     = _mm256_xor_si256x(v_hi, magic_i_hi32);       // Flip the msb of v_hi and blend with 0x45300000
    let     v_hi_dbl = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); // Compute in double precision:
                       _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo))     // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!
}

#[inline(always)]
pub unsafe fn _mm256_adds_epi32(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_add_epi32(lhs, rhs);

    _mm256_blendv_epi32x(
        res,
        // cheeky hack relying on only the highest significant bit, which is the effective "sign" bit
        _mm256_blendv_epi32x(_mm256_set1_epi32(i32::MIN), _mm256_set1_epi32(i32::MAX), res),
        _mm256_xor_si256x(rhs, _mm256_cmpgt_epi32(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_adds_epi64(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_add_epi64(lhs, rhs);

    _mm256_blendv_epi64x(
        res,
        _mm256_blendv_epi64x(_mm256_set1_epi64x(i64::MIN), _mm256_set1_epi64x(i64::MAX), res),
        _mm256_xor_si256x(rhs, _mm256_cmpgt_epi64(lhs, res)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_subs_epi32(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi32(lhs, rhs);

    _mm256_blendv_epi32x(
        res,
        _mm256_blendv_epi32x(_mm256_set1_epi32(i32::MIN), _mm256_set1_epi32(i32::MAX), res),
        _mm256_xor_si256x(
            _mm256_cmpgt_epi32(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi32(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_subs_epi64(lhs: __m256i, rhs: __m256i) -> __m256i {
    let res = _mm256_sub_epi64(lhs, rhs);

    _mm256_blendv_epi64x(
        res,
        _mm256_blendv_epi64x(_mm256_set1_epi64x(i64::MIN), _mm256_set1_epi64x(i64::MAX), res),
        _mm256_xor_si256x(
            _mm256_cmpgt_epi64(rhs, _mm256_setzero_si256()),
            _mm256_cmpgt_epi64(lhs, res),
        ),
    )
}

#[inline(always)]
pub unsafe fn _mm256_abs_epi64(x: __m256i) -> __m256i {
    let should_negate = _mm256_xor_si256x(_mm256_cmpgt_epi64(x, _mm256_setzero_si256()), _mm256_set1_epi64x(-1));

    _mm256_add_epi64(
        _mm256_xor_si256x(should_negate, x),
        _mm256_and_si256x(should_negate, _mm256_set1_epi64x(1)),
    )
}

#[inline(always)]
pub unsafe fn _mm256_cvtps_epi64(x: __m128) -> __m256i {
    let x0 = _mm_cvttss_si64(x);
    let x1 = _mm_cvttss_si64(_mm_permute_ps(x, 1));
    let x2 = _mm_cvttss_si64(_mm_permute_ps(x, 2));
    let x3 = _mm_cvttss_si64(_mm_permute_ps(x, 3));

    _mm256_setr_epi64x(x0, x1, x2, x3)
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64(x: __m256d) -> __m256i {
    let low = _mm256_castpd256_pd128(x);
    let high = _mm256_extractf128_pd(x, 1);

    let x0 = _mm_cvttsd_si64(low);
    let x1 = _mm_cvttsd_si64(_mm_permute_pd(low, 1));
    let x2 = _mm_cvttsd_si64(high);
    let x3 = _mm_cvttsd_si64(_mm_permute_pd(high, 1));

    _mm256_setr_epi64x(x0, x1, x2, x3)
}

#[inline(always)]
pub unsafe fn _mm256_cvtps_epu32(x: __m256) -> __m256i {
    // TODO: This is exactly what LLVM generates for `simd_cast(f32x4 -> u32x4)`, but it's not ideal and
    // produces different results from `f32 as u32` with negaitve values and values larger than some value
    let xmm0 = x;
    let xmm1 = _mm256_set1_ps(f32::from_bits(0x4f000000));
    let xmm2 = _mm256_cmp_ps(xmm0, xmm1, _CMP_LT_OQ);
    let xmm1 = _mm256_sub_ps(xmm0, xmm1);
    let xmm1 = _mm256_cvtps_epi32(xmm1);
    let xmm3 = _mm256_set1_epi32(0x80000000u32 as i32);
    let xmm1 = _mm256_xor_si256x(xmm1, xmm3);
    let xmm0 = _mm256_cvtps_epi32(xmm0);
    let xmm0 = _mm256_blendv_ps(_mm256_castsi256_ps(xmm1), _mm256_castsi256_ps(xmm0), xmm2);

    _mm256_castps_si256(xmm0)
}

#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu32(ymm0: __m256d) -> __m128i {
    let ymm1 = _mm256_set1_pd(f64::from_bits(0x41e0000000000000));
    let ymm2 = _mm256_cmp_pd(ymm0, ymm1, _CMP_LT_OQ);
    let xmm2 = _mm256_castpd256_pd128(ymm2); // lower half of ymm2
    let xmm3 = _mm256_extractf128_pd(ymm2, 1);
    let xmm2 = _mm_packs_epi32(_mm_castpd_si128(xmm2), _mm_castpd_si128(xmm3));
    let ymm1 = _mm256_sub_pd(ymm0, ymm1);
    let xmm1 = _mm256_cvttpd_epi32(ymm1);
    let xmm3 = _mm_set1_ps(f32::from_bits(0x80000000));
    let xmm1 = _mm_xor_ps(_mm_castsi128_ps(xmm1), xmm3);
    let xmm0 = _mm256_cvttpd_epi32(ymm0);
    let xmm0 = _mm_blendv_ps(xmm1, _mm_castsi128_ps(xmm0), _mm_castsi128_ps(xmm2));

    _mm_castps_si128(xmm0)
}

macro_rules! decl_split_integer {
    (@BINARY $($name:ident),*) => {paste::paste! {$(
        #[inline(always)]
        pub unsafe fn [<_mm256_ $name x>](ymm0: __m256i, ymm1: __m256i) -> __m256i {
            let (low0, high0) = split_si256(ymm0);
            let (low1, high1) = split_si256(ymm1);

            recombine_si256([<_mm_ $name>](low0, low1), [<_mm_ $name>](high0, high1))
        }
    )*}}
}

decl_split_integer!(@BINARY add_epi32, sub_epi32, mullo_epi32, add_epi64, sub_epi64, cmpgt_epi32, cmpeq_epi32, cmpgt_epi64, cmpeq_epi64);

#[inline(always)]
pub unsafe fn _mm256_andnot_si256x(a: __m256i, b: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)))
}

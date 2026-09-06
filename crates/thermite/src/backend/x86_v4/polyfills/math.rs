//! 512-bit float step functions (`next_up`/`next_down`). The algorithm is the
//! generic `FloatRegister` default's, with the blendv chains replaced by
//! k-mask embedded-mask ops: start from the "negative" branch, `mask_add` the
//! positive lanes, then two `mask_mov` fixups (zero and unchanged classes).
//! `is_nan` is `_CMP_UNORD_Q`, so NaN inputs classify correctly and pass
//! through. (The hand-written v1/v3 polyfills this was originally ported from
//! are gone. Those backends inherit the generic default now.)

use crate::backend::x86::avx512f::*;

#[inline(always)]
pub unsafe fn _mm512_nextupps_v4(value: __m512) -> __m512 {
    unsafe {
        let bits = _mm512_castps_si512(value);
        let abs = _mm512_andnot_si512(_mm512_set1_epi32(0x8000_0000u32 as i32), bits);

        let is_nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
        let is_infinity = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(0x7F80_0000));
        let unchanged = is_nan | is_infinity;

        let is_positive = _mm512_cmpeq_epi32_mask(abs, bits);
        let is_zero = _mm512_cmpeq_epi32_mask(abs, _mm512_setzero_si512());

        let one = _mm512_set1_epi32(1);
        // negative: bits - 1. Positive lanes overwritten with bits + 1.
        let next = _mm512_sub_epi32(bits, one);
        let next = _mm512_mask_add_epi32(next, is_positive, bits, one);
        // +/-0.0 -> smallest positive subnormal.
        let next = _mm512_mask_mov_epi32(next, is_zero, one);

        _mm512_castsi512_ps(_mm512_mask_mov_epi32(next, unchanged, bits))
    }
}

#[inline(always)]
pub unsafe fn _mm512_nextdownps_v4(value: __m512) -> __m512 {
    unsafe {
        let bits = _mm512_castps_si512(value);
        let abs = _mm512_andnot_si512(_mm512_set1_epi32(0x8000_0000u32 as i32), bits);

        let is_nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
        let is_neg_infinity = _mm512_cmpeq_epi32_mask(bits, _mm512_set1_epi32(0xFF80_0000u32 as i32));
        let unchanged = is_nan | is_neg_infinity;

        let is_positive = _mm512_cmpeq_epi32_mask(abs, bits);
        let is_zero = _mm512_cmpeq_epi32_mask(abs, _mm512_setzero_si512());

        let one = _mm512_set1_epi32(1);
        // negative: bits + 1 (magnitude grows). Positive lanes get bits - 1.
        let next = _mm512_add_epi32(bits, one);
        let next = _mm512_mask_sub_epi32(next, is_positive, bits, one);
        // +/-0.0 -> smallest negative subnormal.
        let next = _mm512_mask_mov_epi32(next, is_zero, _mm512_set1_epi32(0x8000_0001u32 as i32));

        _mm512_castsi512_ps(_mm512_mask_mov_epi32(next, unchanged, bits))
    }
}

#[inline(always)]
pub unsafe fn _mm512_nextuppd_v4(value: __m512d) -> __m512d {
    unsafe {
        let bits = _mm512_castpd_si512(value);
        let abs = _mm512_andnot_si512(_mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64), bits);

        let is_nan = _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q);
        let is_infinity = _mm512_cmpeq_epi64_mask(bits, _mm512_set1_epi64(0x7FF0_0000_0000_0000));
        let unchanged = is_nan | is_infinity;

        let is_positive = _mm512_cmpeq_epi64_mask(abs, bits);
        let is_zero = _mm512_cmpeq_epi64_mask(abs, _mm512_setzero_si512());

        let one = _mm512_set1_epi64(1);
        let next = _mm512_sub_epi64(bits, one);
        let next = _mm512_mask_add_epi64(next, is_positive, bits, one);
        let next = _mm512_mask_mov_epi64(next, is_zero, one);

        _mm512_castsi512_pd(_mm512_mask_mov_epi64(next, unchanged, bits))
    }
}

#[inline(always)]
pub unsafe fn _mm512_nextdownpd_v4(value: __m512d) -> __m512d {
    unsafe {
        let bits = _mm512_castpd_si512(value);
        let abs = _mm512_andnot_si512(_mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64), bits);

        let is_nan = _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q);
        let is_neg_infinity = _mm512_cmpeq_epi64_mask(bits, _mm512_set1_epi64(0xFFF0_0000_0000_0000u64 as i64));
        let unchanged = is_nan | is_neg_infinity;

        let is_positive = _mm512_cmpeq_epi64_mask(abs, bits);
        let is_zero = _mm512_cmpeq_epi64_mask(abs, _mm512_setzero_si512());

        let one = _mm512_set1_epi64(1);
        let next = _mm512_add_epi64(bits, one);
        let next = _mm512_mask_sub_epi64(next, is_positive, bits, one);
        let next = _mm512_mask_mov_epi64(next, is_zero, _mm512_set1_epi64(0x8000_0000_0000_0001u64 as i64));

        _mm512_castsi512_pd(_mm512_mask_mov_epi64(next, unchanged, bits))
    }
}

// --- 256-bit and 128-bit forms (VL) -----------------------------------------
//
// Same bodies at ymm/xmm width. The v3 blendv polyfills at these widths are
// not reused: the k-mask forms are shorter and carry the `_CMP_UNORD_Q` NaN
// classification the v3 bodies lack.

macro_rules! nextupdown_vl {
    ($($nextup:ident, $nextdown:ident: $v:ty;
       $castps_si:ident, $castsi_ps:ident, $andnot:ident, $set1:ident, $setzero:ident,
       $cmp_mask:ident, $cmpeq_mask:ident, $add:ident, $sub:ident, $mask_add:ident, $mask_sub:ident, $mask_mov:ident;
       $sign:expr, $pos_inf:expr, $neg_inf:expr, $neg_min:expr;)*) => {$(
        #[inline(always)]
        pub unsafe fn $nextup(value: $v) -> $v {
            unsafe {
                let bits = $castps_si(value);
                let abs = $andnot($set1($sign), bits);

                let is_nan = $cmp_mask(value, value, _CMP_UNORD_Q);
                let is_infinity = $cmpeq_mask(bits, $set1($pos_inf));
                let unchanged = is_nan | is_infinity;

                let is_positive = $cmpeq_mask(abs, bits);
                let is_zero = $cmpeq_mask(abs, $setzero());

                let one = $set1(1);
                let next = $sub(bits, one);
                let next = $mask_add(next, is_positive, bits, one);
                let next = $mask_mov(next, is_zero, one);

                $castsi_ps($mask_mov(next, unchanged, bits))
            }
        }

        #[inline(always)]
        pub unsafe fn $nextdown(value: $v) -> $v {
            unsafe {
                let bits = $castps_si(value);
                let abs = $andnot($set1($sign), bits);

                let is_nan = $cmp_mask(value, value, _CMP_UNORD_Q);
                let is_neg_infinity = $cmpeq_mask(bits, $set1($neg_inf));
                let unchanged = is_nan | is_neg_infinity;

                let is_positive = $cmpeq_mask(abs, bits);
                let is_zero = $cmpeq_mask(abs, $setzero());

                let one = $set1(1);
                let next = $add(bits, one);
                let next = $mask_sub(next, is_positive, bits, one);
                let next = $mask_mov(next, is_zero, $set1($neg_min));

                $castsi_ps($mask_mov(next, unchanged, bits))
            }
        }
    )*};
}

nextupdown_vl! {
    _mm256_nextupps_v4, _mm256_nextdownps_v4: __m256;
        _mm256_castps_si256, _mm256_castsi256_ps, _mm256_andnot_si256, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_cmp_ps_mask, _mm256_cmpeq_epi32_mask, _mm256_add_epi32, _mm256_sub_epi32, _mm256_mask_add_epi32, _mm256_mask_sub_epi32, _mm256_mask_mov_epi32;
        0x8000_0000u32 as i32, 0x7F80_0000, 0xFF80_0000u32 as i32, 0x8000_0001u32 as i32;
    _mm256_nextuppd_v4, _mm256_nextdownpd_v4: __m256d;
        _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_andnot_si256, _mm256_set1_epi64x, _mm256_setzero_si256,
        _mm256_cmp_pd_mask, _mm256_cmpeq_epi64_mask, _mm256_add_epi64, _mm256_sub_epi64, _mm256_mask_add_epi64, _mm256_mask_sub_epi64, _mm256_mask_mov_epi64;
        0x8000_0000_0000_0000u64 as i64, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000u64 as i64, 0x8000_0000_0000_0001u64 as i64;
    _mm_nextupps_v4, _mm_nextdownps_v4: __m128;
        _mm_castps_si128, _mm_castsi128_ps, _mm_andnot_si128, _mm_set1_epi32, _mm_setzero_si128,
        _mm_cmp_ps_mask, _mm_cmpeq_epi32_mask, _mm_add_epi32, _mm_sub_epi32, _mm_mask_add_epi32, _mm_mask_sub_epi32, _mm_mask_mov_epi32;
        0x8000_0000u32 as i32, 0x7F80_0000, 0xFF80_0000u32 as i32, 0x8000_0001u32 as i32;
    _mm_nextuppd_v4, _mm_nextdownpd_v4: __m128d;
        _mm_castpd_si128, _mm_castsi128_pd, _mm_andnot_si128, _mm_set1_epi64x, _mm_setzero_si128,
        _mm_cmp_pd_mask, _mm_cmpeq_epi64_mask, _mm_add_epi64, _mm_sub_epi64, _mm_mask_add_epi64, _mm_mask_sub_epi64, _mm_mask_mov_epi64;
        0x8000_0000_0000_0000u64 as i64, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000u64 as i64, 0x8000_0000_0000_0001u64 as i64;
}

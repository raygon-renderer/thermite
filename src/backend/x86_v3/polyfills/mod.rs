#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use crate::MM_SHUFFLE;

use super::arch::*;

pub use crate::backend::x86_v2::polyfills::*;

use generic_array::{GenericArray, typenum};

pub mod bits;
pub mod casts;
pub mod cmp;
pub mod divider;
pub mod math;

pub use bits::*;
pub use casts::*;
pub use cmp::*;
pub use divider::*;
pub use math::*;

#[inline(always)]
pub unsafe fn u32x2_to_i64x2(value: GenericArray<u32, typenum::U2>) -> __m128i {
    _mm_cvtepi32_epi64(_mm_setr_epi32(value[0] as i32, value[1] as i32, 0, 0))
    //_mm_set_epi64x(value[1] as i64, value[0] as i64)
}

#[inline(always)]
pub unsafe fn u32x4_to_i64x4(value: GenericArray<u32, typenum::U4>) -> __m256i {
    //_mm256_set_epi64x(value[3] as i64, value[2] as i64, value[1] as i64, value[0] as i64)
    _mm256_cvtepi32_epi64(_mm_setr_epi32(
        value[0] as i32,
        value[1] as i32,
        value[2] as i32,
        value[3] as i32,
    ))
}

#[inline(always)]
pub unsafe fn _mm256_blendv_epi32x_v3(ymm0: __m256i, ymm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(ymm0),
        _mm256_castsi256_ps(ymm1),
        _mm256_castsi256_ps(mask),
    ))
}

#[inline(always)]
pub unsafe fn _mm256_blendv_epi64x_v3(ymm0: __m256i, ymm1: __m256i, mask: __m256i) -> __m256i {
    _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd(ymm0),
        _mm256_castsi256_pd(ymm1),
        _mm256_castsi256_pd(mask),
    ))
}

/// POLYFILL: Shift right and sign extend 64-bit integers
#[inline(always)]
pub unsafe fn _mm256_srai_epi64x_v3(v: __m256i, cnt: i32) -> __m256i {
    let m = _mm256_set1_epi64x(1i64 << (63 - cnt));
    _mm256_sub_epi64(_mm256_xor_si256(_mm256_srl_epi64(v, _mm_cvtsi32_si128(cnt)), m), m)
}

#[inline(always)]
pub unsafe fn _mm256_set1_epu32x(value: u32) -> __m256i {
    _mm256_set1_epi32(value as i32)
}

#[inline(always)]
pub unsafe fn _mm256_set1_epu64x(value: u64) -> __m256i {
    _mm256_set1_epi64x(value as i64)
}

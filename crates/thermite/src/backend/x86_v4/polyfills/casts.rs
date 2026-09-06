//! Saturating float -> int conversions with exact Rust `as` semantics
//! (NaN -> 0, clamp to the destination range), for the `sat` slots of
//! `impl_float_to_int_casts!`.
//!
//! The plain `cast` slots use the bare `vcvtt*` intrinsics directly (fast,
//! documented in-range precondition). The hardware already does most of the
//! saturation work: `vcvttps2dq`-class instructions return the "integer
//! indefinite" (INT_MIN) for out-of-range and NaN inputs, which IS the
//! correct `as` result for negative overflow, so the signed fixups are only
//! "positive overflow -> MAX" and "NaN -> 0". The unsigned converts return
//! all-ones for out-of-range/NaN in either direction, which is already
//! correct for positive overflow, so a single zero-mask (`x >= 0.0`, false
//! for NaN and negatives) finishes the job in one extra instruction.
//!
//! Imports go straight to `avx512f` (F + VL surface) and its `avx512dq`
//! submodule. The `tiers::tierN` modules do not re-export the base set
//! (see the arch-namespace note in `backend/x86_v4/mod.rs`).

use crate::backend::x86::avx512f::avx512dq::*;
#[allow(unused_imports)]
use crate::backend::x86::avx512f::*;

// Smallest float >= the signed max: 2^31 / 2^63 are exactly representable.
const F32_I32_HI: f32 = 2147483648.0;
const F32_I64_HI: f32 = 9223372036854775808.0;
const F64_I32_HI: f64 = 2147483648.0;
const F64_I64_HI: f64 = 9223372036854775808.0;

/// `f32x16 -> i32x16` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm512_cvtps_epi32_satx_v4(x: __m512) -> __m512i {
    unsafe {
        // Out-of-range/NaN lanes come back INT_MIN: correct for negative
        // overflow already.
        let r = _mm512_cvttps_epi32(x);
        // Positive overflow -> MAX.
        let hi = _mm512_cmp_ps_mask(x, _mm512_set1_ps(F32_I32_HI), _CMP_GE_OQ);
        let r = _mm512_mask_mov_epi32(r, hi, _mm512_set1_epi32(i32::MAX));
        // NaN -> 0 (ordered-compare mask is false exactly on NaN lanes).
        _mm512_maskz_mov_epi32(_mm512_cmp_ps_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f32x16 -> u32x16` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm512_cvtps_epu32_satx_v4(x: __m512) -> __m512i {
    unsafe {
        // vcvttps2udq: out-of-range/NaN -> all-ones, correct for positive
        // overflow. `x >= 0.0` is false for negatives AND NaN, so one
        // zero-masked move fixes both remaining cases.
        let ge_zero = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GE_OQ);
        _mm512_maskz_mov_epi32(ge_zero, _mm512_cvttps_epu32(x))
    }
}

/// `f64x8 -> i64x8` with `as` semantics (vcvttpd2qq: DQ, floor).
#[inline(always)]
pub unsafe fn _mm512_cvtpd_epi64_satx_v4(x: __m512d) -> __m512i {
    unsafe {
        let r = _mm512_cvttpd_epi64(x);
        let hi = _mm512_cmp_pd_mask(x, _mm512_set1_pd(F64_I64_HI), _CMP_GE_OQ);
        let r = _mm512_mask_mov_epi64(r, hi, _mm512_set1_epi64(i64::MAX));
        _mm512_maskz_mov_epi64(_mm512_cmp_pd_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f64x8 -> u64x8` with `as` semantics (vcvttpd2uqq: DQ, floor).
#[inline(always)]
pub unsafe fn _mm512_cvtpd_epu64_satx_v4(x: __m512d) -> __m512i {
    unsafe {
        let ge_zero = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GE_OQ);
        _mm512_maskz_mov_epi64(ge_zero, _mm512_cvttpd_epu64(x))
    }
}

/// `f64x8 -> i32x8` (`__m256i`) with `as` semantics: vcvttpd2dq at 512-bit.
#[inline(always)]
pub unsafe fn _mm512_cvtpd_epi32_satx_v4(x: __m512d) -> __m256i {
    unsafe {
        let r = _mm512_cvttpd_epi32(x);
        let hi = _mm512_cmp_pd_mask(x, _mm512_set1_pd(F64_I32_HI), _CMP_GE_OQ);
        let r = _mm256_mask_mov_epi32(r, hi, _mm256_set1_epi32(i32::MAX));
        _mm256_maskz_mov_epi32(_mm512_cmp_pd_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f32x8` (`__m256`) -> `i64x8` with `as` semantics: vcvttps2qq widens
/// 8 floats to 8 qwords (DQ, floor).
#[inline(always)]
pub unsafe fn _mm512_cvtps_epi64_satx_v4(x: __m256) -> __m512i {
    unsafe {
        let r = _mm512_cvttps_epi64(x);
        let hi = _mm256_cmp_ps_mask(x, _mm256_set1_ps(F32_I64_HI), _CMP_GE_OQ);
        let r = _mm512_mask_mov_epi64(r, hi, _mm512_set1_epi64(i64::MAX));
        _mm512_maskz_mov_epi64(_mm256_cmp_ps_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f32x8` (`__m256`) -> `u64x8` with `as` semantics (vcvttps2uqq: DQ, floor).
#[inline(always)]
pub unsafe fn _mm512_cvtps_epu64_satx_v4(x: __m256) -> __m512i {
    unsafe {
        let ge_zero = _mm256_cmp_ps_mask(x, _mm256_setzero_ps(), _CMP_GE_OQ);
        _mm512_maskz_mov_epi64(ge_zero, _mm512_cvttps_epu64(x))
    }
}

/// `f64x8 -> u32x8` (`__m256i`) with `as` semantics: vcvttpd2udq at 512-bit.
#[inline(always)]
pub unsafe fn _mm512_cvtpd_epu32_satx_v4(x: __m512d) -> __m256i {
    unsafe {
        let ge_zero = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GE_OQ);
        _mm256_maskz_mov_epi32(ge_zero, _mm512_cvttpd_epu32(x))
    }
}

// --- 256-bit and 128-bit forms (VL) -----------------------------------------

/// `f32x8 -> i32x8` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm256_cvtps_epi32_satx_v4(x: __m256) -> __m256i {
    unsafe {
        let r = _mm256_cvttps_epi32(x);
        let hi = _mm256_cmp_ps_mask(x, _mm256_set1_ps(F32_I32_HI), _CMP_GE_OQ);
        let r = _mm256_mask_mov_epi32(r, hi, _mm256_set1_epi32(i32::MAX));
        _mm256_maskz_mov_epi32(_mm256_cmp_ps_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f32x8 -> u32x8` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm256_cvtps_epu32_satx_v4(x: __m256) -> __m256i {
    unsafe {
        let ge_zero = _mm256_cmp_ps_mask(x, _mm256_setzero_ps(), _CMP_GE_OQ);
        _mm256_maskz_mov_epi32(ge_zero, _mm256_cvttps_epu32(x))
    }
}

/// `f64x4 -> i64x4` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi64_satx_v4(x: __m256d) -> __m256i {
    unsafe {
        let r = _mm256_cvttpd_epi64(x);
        let hi = _mm256_cmp_pd_mask(x, _mm256_set1_pd(F64_I64_HI), _CMP_GE_OQ);
        let r = _mm256_mask_mov_epi64(r, hi, _mm256_set1_epi64x(i64::MAX));
        _mm256_maskz_mov_epi64(_mm256_cmp_pd_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f64x4 -> u64x4` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu64_satx_v4(x: __m256d) -> __m256i {
    unsafe {
        let ge_zero = _mm256_cmp_pd_mask(x, _mm256_setzero_pd(), _CMP_GE_OQ);
        _mm256_maskz_mov_epi64(ge_zero, _mm256_cvttpd_epu64(x))
    }
}

/// `f64x4 -> i32x4` (`__m128i`) with `as` semantics.
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epi32_satx_v4(x: __m256d) -> __m128i {
    unsafe {
        let r = _mm256_cvttpd_epi32(x);
        let hi = _mm256_cmp_pd_mask(x, _mm256_set1_pd(F64_I32_HI), _CMP_GE_OQ);
        let r = _mm_mask_mov_epi32(r, hi, _mm_set1_epi32(i32::MAX));
        _mm_maskz_mov_epi32(_mm256_cmp_pd_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f64x4 -> u32x4` (`__m128i`) with `as` semantics (vcvttpd2udq, VL).
#[inline(always)]
pub unsafe fn _mm256_cvtpd_epu32_satx_v4(x: __m256d) -> __m128i {
    unsafe {
        let ge_zero = _mm256_cmp_pd_mask(x, _mm256_setzero_pd(), _CMP_GE_OQ);
        _mm_maskz_mov_epi32(ge_zero, _mm256_cvttpd_epu32(x))
    }
}

/// `f32x4` (`__m128`) -> `i64x4` with `as` semantics (vcvttps2qq, DQ+VL).
#[inline(always)]
pub unsafe fn _mm256_cvtps_epi64_satx_v4(x: __m128) -> __m256i {
    unsafe {
        let r = _mm256_cvttps_epi64(x);
        let hi = _mm_cmp_ps_mask(x, _mm_set1_ps(F32_I64_HI), _CMP_GE_OQ);
        let r = _mm256_mask_mov_epi64(r, hi, _mm256_set1_epi64x(i64::MAX));
        _mm256_maskz_mov_epi64(_mm_cmp_ps_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f32x4` (`__m128`) -> `u64x4` with `as` semantics (vcvttps2uqq, DQ+VL).
#[inline(always)]
pub unsafe fn _mm256_cvtps_epu64_satx_v4(x: __m128) -> __m256i {
    unsafe {
        let ge_zero = _mm_cmp_ps_mask(x, _mm_setzero_ps(), _CMP_GE_OQ);
        _mm256_maskz_mov_epi64(ge_zero, _mm256_cvttps_epu64(x))
    }
}

/// `f32x4 -> i32x4` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm_cvtps_epi32_satx_v4(x: __m128) -> __m128i {
    unsafe {
        let r = _mm_cvttps_epi32(x);
        let hi = _mm_cmp_ps_mask(x, _mm_set1_ps(F32_I32_HI), _CMP_GE_OQ);
        let r = _mm_mask_mov_epi32(r, hi, _mm_set1_epi32(i32::MAX));
        _mm_maskz_mov_epi32(_mm_cmp_ps_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f32x4 -> u32x4` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm_cvtps_epu32_satx_v4(x: __m128) -> __m128i {
    unsafe {
        let ge_zero = _mm_cmp_ps_mask(x, _mm_setzero_ps(), _CMP_GE_OQ);
        _mm_maskz_mov_epi32(ge_zero, _mm_cvttps_epu32(x))
    }
}

/// `f64x2 -> i64x2` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm_cvtpd_epi64_satx_v4(x: __m128d) -> __m128i {
    unsafe {
        let r = _mm_cvttpd_epi64(x);
        let hi = _mm_cmp_pd_mask(x, _mm_set1_pd(F64_I64_HI), _CMP_GE_OQ);
        let r = _mm_mask_mov_epi64(r, hi, _mm_set1_epi64x(i64::MAX));
        _mm_maskz_mov_epi64(_mm_cmp_pd_mask(x, x, _CMP_ORD_Q), r)
    }
}

/// `f64x2 -> u64x2` with `as` semantics.
#[inline(always)]
pub unsafe fn _mm_cvtpd_epu64_satx_v4(x: __m128d) -> __m128i {
    unsafe {
        let ge_zero = _mm_cmp_pd_mask(x, _mm_setzero_pd(), _CMP_GE_OQ);
        _mm_maskz_mov_epi64(ge_zero, _mm_cvttpd_epu64(x))
    }
}

// --- unsigned dword -> float at ymm/xmm -------------------------------------
//
// `core::arch` has no `_mm256_cvtepu32_ps`/`_mm_cvtepu32_ps` (it does have
// the `_pd` pair), so widen to zmm for the one `vcvtudq2ps` and take the low
// lanes back. The zero-extend is free and LLVM narrows the encoding.

/// `u32x8 -> f32x8` (vcvtudq2ps).
#[inline(always)]
pub unsafe fn _mm256_cvtepu32_psx_v4(x: __m256i) -> __m256 {
    unsafe { _mm512_castps512_ps256(_mm512_cvtepu32_ps(_mm512_zextsi256_si512(x))) }
}

/// `u32x4 -> f32x4` (vcvtudq2ps).
#[inline(always)]
pub unsafe fn _mm_cvtepu32_psx_v4(x: __m128i) -> __m128 {
    unsafe { _mm512_castps512_ps128(_mm512_cvtepu32_ps(_mm512_zextsi128_si512(x))) }
}
